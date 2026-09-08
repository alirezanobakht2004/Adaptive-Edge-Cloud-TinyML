"""M25 descriptive analysis and readiness audit. Does not create training labels or fit anything."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from ml.policy.formulation import CONFIG_PATH, load_config, validate_config, value_at, missing_state_features
from .schema import ACTION_NAMES, validate_record
from .validator import read_records, validate_dataset, validate_campaign

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/policy/policy_training_dataset_v1"


def leaf_paths(value, prefix=""):
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            yield from leaf_paths(child, path)
        else:
            yield path


def describe(values, *, categorical=False):
    present = [v for v in values if v is not None]
    result = {"count": len(values), "present": len(present), "missing": len(values) - len(present),
              "unique": len(set(present)), "constant_when_present": bool(present) and len(set(present)) == 1}
    if present and not categorical and all(type(v) in (int, float) for v in present):
        result.update(min=min(present), max=max(present), mean=statistics.mean(present),
                      median=statistics.median(present), std_population=statistics.pstdev(present))
    else:
        result["categories"] = dict(Counter(map(str, present)))
    return result


def histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    missing, outside = 0, 0
    for value in values:
        if value is None:
            missing += 1
            continue
        for index in range(len(counts)):
            if edges[index] <= value < edges[index + 1] or (index == len(counts) - 1 and value == edges[-1]):
                counts[index] += 1
                break
        else:
            outside += 1
    return {"edges": edges, "counts": counts, "missing": missing, "outside": outside,
            "interpretation": "Descriptive bins only; not policy thresholds. Final bin includes upper endpoint."}


def analyze_records(records, config):
    validate_config(config)
    seen = set()
    by_window, by_state, by_run = defaultdict(list), defaultdict(set), defaultdict(set)
    for record in records:
        validate_record(record)
        if record["metadata"]["record_kind"] != "observation":
            raise ValueError("Examples cannot enter policy analysis")
        identity = record["metadata"]["run_id"], record["metadata"]["sample_id"]
        if identity in seen:
            raise ValueError("Duplicate observation identity")
        seen.add(identity)
        by_window[(record["metadata"]["session_id"], record["state"]["application"]["window_id"])].append(record)
        by_state[tuple(value_at(record, path) for path in config["state_feature_names"])].add(record["action"])
        by_run[record["metadata"]["run_id"]].add(record["action"])
    actions = Counter(r["action"] for r in records)
    state_paths = list(leaf_paths(records[0]["state"], "state")) if records else config["state_feature_names"]
    state = {path: describe([value_at(r, path) for r in records], categorical=path in config["categorical_features"]) for path in state_paths}
    missing = {path: sum(value_at(r, path) is None for r in records)
               for path in (leaf_paths(records[0]) if records else []) if not path.startswith("provenance.")}
    correct = sum(r["outcome"]["status"] == "success" and r["outcome"]["true_class"] is not None and
                  r["outcome"]["true_class"] == r["outcome"]["predicted_class"] for r in records)
    sessions = sorted({r["metadata"]["session_id"] for r in records})
    availability = {name: sum(value_at(r, name) is not None for r in records) for name in config["state_feature_names"]}
    return {"sample_count": len(records), "action_distribution": {ACTION_NAMES[k]: actions[k] for k in ACTION_NAMES},
            "state_statistics": state, "state_statistics_by_action": {ACTION_NAMES[action]: {
                path: describe([value_at(r, path) for r in records if r["action"] == action], categorical=path in config["categorical_features"]) for path in state_paths}
                for action in ACTION_NAMES},
            "uncertainty_distribution": {key: histogram([r["state"]["uncertainty"][key] for r in records], edges)
                for key, edges in (("confidence", [0, .5, .8, .9, .95, 1]),
                                   ("entropy", [0, .05, .1, .2, .5, math.log(5)]),
                                   ("margin", [0, .2, .5, .8, .95, 1]))},
            "missing_values": missing, "selected_feature_availability": availability,
            "complete_selected_state_rows": sum(not missing_state_features(r, config) for r in records),
            "coverage": {"independent_source_sessions": sessions, "unique_source_windows": len(by_window),
                "devices": sorted({r["metadata"]["device_id"] for r in records}),
                "measurement_scopes": dict(Counter(r["metadata"]["measurement_scope"] for r in records)),
                "runs": len(by_run), "actions_per_run": {key: sorted(value) for key, value in by_run.items()},
                "runs_with_multiple_actions": sum(len(value) > 1 for value in by_run.values()),
                "source_windows_with_all_four_actions": sum({r["action"] for r in rows} == set(ACTION_NAMES) for rows in by_window.values()),
                "windows_with_identical_uncertainty_across_repeats": sum(len({tuple(r["state"]["uncertainty"][key] for key in ("confidence", "entropy", "margin")) for r in rows}) == 1 for rows in by_window.values()),
                "unique_selected_states": len(by_state), "exact_selected_states_with_all_four_actions": sum(value == set(ACTION_NAMES) for value in by_state.values()),
                "label_distribution": dict(Counter(str(r["outcome"]["true_class"]) for r in records)),
                "outcomes": dict(Counter(r["outcome"]["status"] for r in records)), "correct_labeled_outcomes": correct,
                "network_connections": dict(Counter(r["state"]["network"]["connection_state"] for r in records)),
                "constant_selected_features": [name for name in config["state_feature_names"] if state[name]["constant_when_present"]],
                "reward_values_present": sum(r["reward"]["final_reward"] is not None for r in records),
                "energy_proxy_values_present": sum(r["reward"]["energy_component"] is not None for r in records)},
            "training_ready": False, "training_enabled": config["training_enabled"],
            "readiness_gates": {"selected_state_complete": bool(records) and all(count == len(records) for count in availability.values()),
                "all_actions_observed": all(actions[k] > 0 for k in ACTION_NAMES),
                "independent_train_validation_test_sessions_possible": len(sessions) >= 3,
                "reward_configuration_frozen": False, "matched_condition_cost_labels_qualified": False,
                "coverage_protocol_approved": False, "training_authorized_in_m25": False},
            "limitations": ["Action balance is not state-conditioned action overlap. Executed action is not an optimal supervision label.",
                "Repeats of a source window must stay together in any future partition; no row-random train/test split.",
                "This analysis does not infer counterfactual rewards or causal action effects.",
                "v1 reward is a negative-cost placeholder and cannot store the proposed accuracy-inclusive reward unchanged.",
                "Energy, reward parameters, coverage criteria and target-label construction remain unconfigured."]}


def run(directory=DATASET, config_path=CONFIG_PATH, output=None):
    directory = Path(directory)
    validate_dataset(directory)
    canonical = directory / "observations.jsonl"
    snapshot = directory / "policy_training_dataset_v1.jsonl"
    if snapshot.exists() and canonical.read_text(encoding="utf-8") != snapshot.read_text(encoding="utf-8"):
        raise ValueError("Named dataset snapshot differs from canonical observations")
    report = analyze_records(read_records(canonical), load_config(config_path))
    campaigns = []
    for path in sorted((directory / "campaigns").glob("*/configuration.json")):
        configuration = json.loads(path.read_text(encoding="utf-8"))
        if (path.parent / "policy_training_dataset_v1.jsonl").exists():
            validation = validate_campaign(path.parent)
        else:
            validation = {"status": configuration["status"], "observations": 0, "note": "No observation file; startup failure retained"}
        campaigns.append({"path": path.parent.name, **validation})
    report.update(dataset_version="policy_training_dataset_v1", config_version="policy-config-v1",
                  dataset_sha256_lf_utf8=hashlib.sha256(canonical.read_text(encoding="utf-8").encode()).hexdigest(),
                  config_sha256_lf_utf8=hashlib.sha256(Path(config_path).read_text(encoding="utf-8").encode()).hexdigest(),
                  input_policy="Canonical observations only; snapshots and per-run copies are not additional samples. External Kaggle data excluded.",
                  campaign_validation=campaigns)
    path = Path(output or ROOT / "docs/evidence/phase10_policy_dataset_analysis.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DATASET)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.directory, args.config, args.output)
    print(json.dumps({"samples": report["sample_count"], "training_ready": report["training_ready"], "coverage": report["coverage"]}))


if __name__ == "__main__":
    main()
