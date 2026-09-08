"""Derive and qualify R1 binary comparisons without rewriting study datasets.

Run: python -m ml.policy.binary_policy_dataset --output <new-directory>
Validation: python -m ml.policy.binary_policy_dataset --validate <directory>
Historical candidate costs are retained even when incompatible with R1 execution.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

from .binary_contract import (
    ACTIONS, CONFIG, FEATURES, ROOT, VERSION, compare_candidates, feature_vector,
    load_contract, load_reward, read_json, validate_record,
)
from tools.policy_dataset.reward_engine.engine import digest
from tools.policy_dataset.reward_engine.__main__ import strict_json
from tools.policy_dataset.run_controlled_campaign import validate_output as validate_controlled
from tools.policy_dataset.run_state_expansion_campaign import validate_output as validate_expanded
from tools.policy_dataset.validate_matched_dataset import validate_campaign

TOOL_VERSION = "binary-dataset-derivation-v1"
V2 = ROOT / "data/policy/policy_training_dataset_v2"
MEASURED = V2 / "campaigns/connected_lab_run02"
V3 = ROOT / "data/policy/policy_training_dataset_v3"
DEFAULT_OUTPUT = ROOT / f"data/policy/{VERSION}"


def read_rows(path):
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def action_name(row):
    action = row["label"]["optimal_action"]
    return "TIE" if action is None else ACTIONS[str(action)]


def source_inventory():
    """Validate raw traces, model/feature hashes and replay rebuilds before use."""
    validate_campaign(MEASURED)
    base_path = MEASURED / "policy_training_dataset_v2.jsonl"
    base_rows = read_rows(base_path)
    if base_rows != read_rows(V2 / "policy_training_dataset_v2.jsonl"):
        raise ValueError("Canonical M30 copy diverges from accepted campaign")
    sources = [(base_path, "measured", "connected-lab-pilot-v1", base_rows)]
    for directory in sorted((V2 / "controlled_network_v1").iterdir()):
        if not directory.is_dir():
            continue
        validate_controlled(directory, MEASURED)
        path = directory / "policy_training_dataset_v2.jsonl"
        sources.append((path, "simulated", directory.name, read_rows(path)))
    validate_expanded(V3, MEASURED)
    path = V3 / "policy_training_dataset_v3.jsonl"
    sources.append((path, "simulated", "M33", read_rows(path)))
    return sources, {row["sample_id"]: row for row in base_rows}


def partition_groups(groups, config):
    groups = set(groups)
    if len(groups) < 2:
        raise ValueError("At least two underlying source groups required")
    spec = config["grouping"]
    ordered = sorted(groups, key=lambda key: digest([spec["seed"], key]))
    count = math.ceil(len(groups) * spec["holdout_fraction"])
    if not 0 < count < len(groups):
        raise ValueError("Invalid group holdout size")
    holdout = set(ordered[:count])
    return {group: "holdout" if group in holdout else "train" for group in groups}


def original_reward(row):
    if "condition" in row:
        return row["condition"]["reward"]
    if "reward" in row:
        return row["reward"]["configuration"]
    return row["reward_configuration"]


def derive_one(row, source, path, kind, profile, config):
    reward = load_reward(config)
    historical_reward = original_reward(row)
    for key in ("version", "weights", "scales", "tie_tolerance"):
        if historical_reward[key] != reward[key]:
            raise ValueError(f"Historical reward {key} differs from frozen M30")
    outcomes = row.get("candidate_actions", row.get("action_outcomes"))
    candidates, label = compare_candidates(outcomes, reward)
    normalized = feature_vector(row["state"], config)
    metadata = source["metadata"]
    identity = {"source_dataset_version": metadata["source_dataset_version"],
                "feature_version": metadata["feature_version"],
                "source_session": metadata["session_id"],
                "window_id": source["state"]["application"]["window_id"]}
    if (row["state"]["application"]["window_id"] != identity["window_id"]
            or metadata["feature_version"] != config["feature_version"]
            or metadata["model_version"] != config["gesture_model_version"]):
        raise ValueError("Source window/model/feature contract mismatch")
    if kind == "simulated" and row["source_sha256"] != digest(source):
        raise ValueError("Replay source hash mismatch")
    field_provenance = {}
    for feature in FEATURES:
        group, key = feature.split(".")[1:]
        changed = row["state"][group][key] != source["state"][group][key]
        field_provenance[feature] = "measured" if kind == "measured" else (
            "simulated" if changed else "carried_from_measured_source_not_remeasured")
    scenario = row.get("scenario")
    profile = scenario["name"] if scenario else profile
    record = {
        "dataset_version": VERSION, "sample_id": digest([relative(path), row["sample_id"]]),
        "state": deepcopy(row["state"]), "normalized_state": normalized,
        "source": {"dataset_version": row["dataset_version"], "path": relative(path),
                   "record_id": row["sample_id"], "record_sha256": digest(row),
                   "underlying_record_id": source["sample_id"], "underlying_record_sha256": digest(source),
                   "metadata": deepcopy(metadata), "window_identity": identity,
                   "reward_configuration": deepcopy(historical_reward),
                   "historical_optimal_action": deepcopy(row.get("label", row.get("optimal_action")))},
        "group_key": digest(identity), "partition": "train",  # Assigned globally below.
        "provenance": {"kind": kind, "profile": profile, "state_features": field_provenance,
                       "energy": "estimated proxy" if kind == "measured" else "simulated proxy",
                       "network_condition": deepcopy(row.get("network_condition")),
                       "scenario": deepcopy(scenario),
                       "original_measurement_provenance": deepcopy(row.get("measurement_provenance")),
                       "action_execution_contract": "m30-local-reexecutes-after-state-v1",
                       "prediction_origin": "M30 measured candidates; replay assumes prediction invariance"},
        "original_candidate_outcomes": deepcopy(outcomes), "candidates": candidates, "label": label,
        "feature_contract_sha256": digest(config), "reward_configuration_sha256": digest(reward),
        "production_eligible": False,
    }
    return record


def derive(config):
    sources, by_id = source_inventory()
    rows, excluded = [], []
    for path, kind, profile, records in sources:
        for record in records:
            source = by_id[record.get("source_sample_id", record["sample_id"])]
            try:
                row = derive_one(record, source, path, kind, profile, config)
            except ValueError as exc:
                # Valid source rows can still lack binary candidate or state inputs.
                excluded.append({"path": relative(path), "record_id": record["sample_id"],
                                 "source_record_id": source["sample_id"], "reason": str(exc)})
                continue
            rows.append(row)
    partitions = partition_groups((r["group_key"] for r in rows), config)
    for row in rows:
        row["partition"] = partitions[row["group_key"]]
        validate_record(row, config)
    return rows, excluded, sources, by_id


def qualification(rows, config, sources):
    groups, vectors = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)
        vectors[digest(row["normalized_state"])].append(row)
    partitions = {part: {r["group_key"] for r in rows if r["partition"] == part}
                  for part in ("train", "holdout")}
    overlap = sorted(partitions["train"] & partitions["holdout"])
    features = {}
    for name in FEATURES:
        section, leaf = name.split(".")[1:]
        values = [r["state"][section][leaf] for r in rows]
        features[name] = {"present": len(values), "missing": sum(v is None for v in values),
                          "minimum": min(values), "maximum": max(values), "mean": mean(values),
                          "unique_values": len(set(values)),
                          "provenance": dict(Counter(r["provenance"]["state_features"][name] for r in rows))}
    measured = list(sources.values())
    local_traces = [entry["trace"] for r in measured for entry in r["source_entries"] if entry["trace"]["action"] == 0]
    measured_local_costs = [t["action_us"] / 1000 for t in local_traces]
    source_correct = [entry["trace"]["state_class"] == entry["record"]["outcome"]["true_class"]
                      for r in measured for entry in r["source_entries"]]
    harness = ROOT / "firmware/test/test_phase10_matched_campaign/test_main.cpp"
    code = harness.read_text(encoding="utf-8")
    markers = ["if (!local(normalized, state)) return;", "if (action == 0)",
               "success = local(normalized, result);", "totalUs = sharedUs + actionUs"]
    if not all(marker in code for marker in markers):
        raise ValueError("Historical execution harness changed; re-audit action semantics")
    semantic = {
        "production_local_definition": config["action_execution"]["0"],
        "historical_local_definition": "state = local(normalized); action0 = local(normalized) again",
        "harness_path": relative(harness), "harness_text_sha256": digest(code),
        "code_evidence": [{"line": next(i for i, line in enumerate(code.splitlines(), 1) if marker in line),
                           "text": marker} for marker in markers],
        "measured_local_repeat_count": len(local_traces),
        "extra_local_execution_ms": {"minimum": min(measured_local_costs),
                                     "maximum": max(measured_local_costs), "mean": mean(measured_local_costs)},
        "predecision_local_prediction_correct_in_all_source_entries": all(source_correct),
        "r1_reuse_result_action_measurements_available": 0,
        "conclusion": "Historical LOCAL latency and proxy charge a second inference that R1 forbids; derived labels cannot be asserted to optimize production LOCAL reuse.",
        "algebraic_observation_not_measurement": "With already-correct local results, common predecision work is sunk. CLOUD adds positive communication and compute; removing redundant LOCAL inference requires a new action-cost contract, not silent reward relabeling.",
    }
    labels = Counter(action_name(r) for r in rows)
    gates = {
        "both_binary_labels": labels["LOCAL"] > 0 and labels["CLOUD"] > 0,
        "binary_schema_valid": True, "frozen_feature_contract": True,
        "runtime_sources_defined": all(f["runtime_source"] and f["available_before_decision"] for f in config["features"]),
        "reward_exactly_reproducible_and_frozen": True, "source_provenance_validated": True,
        "group_holdout_exists": all(partitions.values()), "no_source_group_overlap": not overlap,
        "historical_labels_recomputed_without_fabrication": True,
        "real_diversity_limitations_documented": True,
        "candidate_execution_matches_production_action_contract": False,
    }
    return {
        "version": "binary-dataset-qualification-v1", "dataset_version": VERSION,
        "record_count": len(rows), "unique_underlying_windows": len(groups),
        "underlying_states_with_multiple_records": sum(len(v) > 1 for v in groups.values()),
        "repeated_records_beyond_one_per_window": len(rows) - len(groups),
        "source_sessions": sorted({r["source"]["window_identity"]["source_session"] for r in rows}),
        "source_acquisition_runs": sorted({r["source"]["metadata"]["run_id"] for r in rows}),
        "label_counts": dict(labels), "provenance_counts": dict(Counter(r["provenance"]["kind"] for r in rows)),
        "profile_counts": dict(Counter(r["provenance"]["profile"] for r in rows)),
        "by_profile": {p: dict(Counter(action_name(r) for r in rows if r["provenance"]["profile"] == p)) for p in sorted({r["provenance"]["profile"] for r in rows})},
        "by_provenance": {kind: dict(Counter(action_name(r) for r in rows if r["provenance"]["kind"] == kind))
                          for kind in ("measured", "simulated")},
        "by_partition": {part: {"records": sum(r["partition"] == part for r in rows),
                                "groups": len(partitions[part]),
                                "labels": dict(Counter(action_name(r) for r in rows if r["partition"] == part))}
                         for part in partitions},
        "source_group_overlap": overlap,
        "partition_groups": {k: sorted(v) for k, v in partitions.items()},
        "exact_feature_vectors": len(vectors),
        "duplicate_feature_vectors": sum(len(v) > 1 for v in vectors.values()),
        "feature_vectors_with_conflicting_labels": sum(len({r["label"]["optimal_action"] for r in v}) > 1 for v in vectors.values()),
        "feature_statistics": features,
        "missing_unselected_fields": {name: sum(r["state"]["network"][name] is None for r in rows)
                                      for name in ("packet_size_estimate", "estimated_bandwidth")},
        "reward_configuration": load_reward(config), "normalization": config["normalization"],
        "production_action_semantics": semantic, "gates": gates,
        "training_allowed": all(gates.values()), "production_eligible_records": 0,
        "blockers": [semantic["conclusion"]],
        "limitations": ["One physical session/run and 24 source windows; replay records are dependent variants.",
                        "CLOUD labels arise only in simulated conditions; no real network-diversity claim.",
                        "Identical replay vectors are retained with explicit grouping, not independent samples.",
                        "All recorded gesture outcomes are correct; no uncertainty-driven accuracy benefit demonstrated.",
                        "Energy is a dimensionless estimated/simulated proxy, never measured joules.",
                        "MQTT probe source exists in isolated harness; production integration not yet done.",
                        "Historical CLOUD payload bytes describe benchmark JSON, not future R1 versioned request overhead."],
    }


def validate_output(directory):
    directory = Path(directory)
    config = load_contract()
    manifest = read_json(directory / "manifest.json")
    expected, excluded, sources, underlying = derive(config)
    rows = read_rows(directory / f"{VERSION}.jsonl")
    if rows != expected or manifest["records_sha256"] != digest(rows):
        raise ValueError("Derived dataset differs from source reconstruction")
    if (manifest["version"] != "binary-dataset-manifest-v1"
            or manifest["creation_tool_version"] != TOOL_VERSION
            or manifest["creation_tool"] != "ml.policy.binary_policy_dataset"
            or manifest["feature_contract_sha256"] != digest(config)
            or manifest["reward_configuration_sha256"] != digest(load_reward(config))
            or manifest["exclusions"] != excluded or manifest["dataset_version"] != VERSION
            or manifest["action_mapping"] != ACTIONS):
        raise ValueError("Manifest contract/exclusion mismatch")
    inventory = [{"path": relative(p), "records_sha256": digest(rs), "records": len(rs), "kind": kind}
                 for p, kind, _, rs in sources]
    if manifest["source_datasets"] != inventory:
        raise ValueError("Source manifest provenance mismatch")
    report = qualification(rows, config, underlying)
    if (read_json(directory / "qualification.json") != report
            or manifest["training_enabled"] != report["training_allowed"]
            or manifest["production_eligible_records"] != report["production_eligible_records"]
            or manifest["grouping"] != config["grouping"]):
        raise ValueError("Qualification report mismatch")
    return report


def build(output):
    config = load_contract()
    rows, excluded, sources, underlying = derive(config)
    report = qualification(rows, config, underlying)
    output = Path(output)
    if output.exists():
        raise ValueError("Output must be a new directory; historical datasets are immutable")
    output.mkdir(parents=True)
    (output / f"{VERSION}.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    configuration = read_json(MEASURED / "configuration.json")
    manifest = {
        "version": "binary-dataset-manifest-v1", "dataset_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "creation_tool": "ml.policy.binary_policy_dataset",
        "creation_tool_version": TOOL_VERSION,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tool_text_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
        "source_datasets": [{"path": relative(p), "records_sha256": digest(rs), "records": len(rs), "kind": kind}
                            for p, kind, _, rs in sources],
        "source_artifact_hashes": configuration["artifact_hashes"],
        "source_feature_matrix_sha256": configuration["feature_matrix_sha256"],
        "source_labels_sha256": configuration["labels_sha256"],
        "action_mapping": ACTIONS, "historical_candidate_mapping": {"0": 0, "1": 3},
        "transformation": "Revalidate M30/M31/M33; retain all original outcomes; recompute M29 mean rewards for historical candidates 0 and 3 only; ties remain unlabeled. No old argmax mapping or production cost correction.",
        "feature_contract_path": relative(CONFIG), "feature_contract_sha256": digest(config),
        "reward_reference": config["reward_reference"], "reward_configuration_sha256": digest(load_reward(config)),
        "records": len(rows), "records_sha256": digest(rows), "exclusions": excluded,
        "other_sources_not_imported": [
            {"path": "data/policy/policy_training_dataset_v1", "reason": "100 unmatched fixed-action observations lack all candidate outcomes for the same state"},
            {"path": relative(V2 / "policy_training_dataset_v2.jsonl"), "reason": "Exact canonical mirror of accepted M30 campaign; not imported twice"},
            {"path": relative(V2 / "campaigns/connected_lab_run01"), "reason": "Diagnostic first run; prior campaign qualification selected run02"}],
        "provenance_rules": "Source record/content hash, original window/session/run, original outcomes and reward configs retained; simulated projections never become measured data; energy always proxy.",
        "grouping": config["grouping"], "training_enabled": report["training_allowed"],
        "production_eligible_records": report["production_eligible_records"],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "qualification.json", report)
    write_json(output / "schema_description.json", {"dataset_version": VERSION, "required_fields": list(rows[0]),
        "action_mapping": ACTIONS, "validator": "ml.policy.binary_policy_dataset.validate_output",
        "state_order": list(FEATURES), "scope": "Historical binary candidate comparison; production eligibility is separately gated"})
    (output / "README.md").write_text("# Derived binary comparison dataset v1\n\n"
        "Contains historical LOCAL/CLOUD candidate comparisons, not approved R1 training targets.\n"
        "See manifest.json and qualification.json for immutable sources, exclusions and the\n"
        "LOCAL re-execution versus result-reuse blocker. Historical datasets are unchanged.\n"
        "All replays of an original window share a partition. Energy is a proxy only.\n\n"
        "Validate: `python -m ml.policy.binary_policy_dataset --validate " + output.as_posix() + "`\n",
        encoding="utf-8")
    return validate_output(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--validate", type=Path)
    args = parser.parse_args()
    report = validate_output(args.validate) if args.validate else build(args.output)
    print(json.dumps({k: report[k] for k in ("record_count", "unique_underlying_windows", "label_counts", "training_allowed", "blockers")}, indent=2))


if __name__ == "__main__":
    main()
