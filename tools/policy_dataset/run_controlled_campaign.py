"""Simulated controlled-network replay of validated M30 four-action measurements."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import subprocess

from tools.network_conditions.profiles import NAMES, apply_condition, load_profile
from .matched_dataset import VERSION, validate_sample
from .reward_engine.engine import digest, evaluate_group
from .reward_engine.__main__ import strict_json
from .run_matched_campaign import write_json

DEFAULT_SOURCE = Path("data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02")
EXTENSION = "controlled-network-replay-v1"


def simulate_sample(source, profile):
    validate_sample(source)
    config = deepcopy(source["condition"]["reward"])
    config.update(energy_kind="simulated", energy_proxy_version="m31-compute-payload-proxy-v1",
                  energy_method="Simulated scaled edge compute and retained payload proxy; " + config["energy_method"])
    coefficients = source["condition"]["energy_proxy"]
    projections = []
    for entry in source["source_entries"]:
        costs = apply_condition(entry, profile)
        record = deepcopy(entry["record"])
        record["metadata"].update(measurement_scope="replay")
        record["state"] = costs["state"]
        # Only these copied fields are consumed by M29. They are not exported
        # as hardware observations or appended to the measured dataset.
        record["measurements"]["total_latency_ms"] = costs["total_latency_ms"]
        record["provenance"]["state_source"] = "M31 simulated controlled-network replay"
        record["provenance"]["notes"] = "Not a new execution; predictions/bytes retained, latency and proxy simulated"
        payload = entry["trace"]["request_bytes"] + entry["trace"]["response_bytes"]
        energy = {"kind": "simulated", "proxy_version": config["energy_proxy_version"], "method": config["energy_method"],
                  "value": (costs["preprocessing_ms"] + costs["state_inference_ms"] + costs["selected_edge_compute_ms"])
                  / coefficients["compute_reference_ms"] + payload / coefficients["payload_reference_bytes"]}
        projections.append({"repeat": entry["guard"]["repeat"], "record": record, "energy": energy, "costs": costs})
    results = []
    for repeat in range(source["repeats"]):
        batch = [p for p in projections if p["repeat"] == repeat]
        group = {"version": "candidate-actions-v1", "decision_id": f"{source['sample_id']}/{profile['profile']}/{repeat}",
                 "matching_evidence": f"SIMULATED replay, measured source SHA256={digest(source)}",
                 "decision": batch[0]["record"], "candidates": [{"record": p["record"], "energy": p["energy"]} for p in batch]}
        results.append(evaluate_group(group, config))
    candidates = []
    for action in range(4):
        trials = [r["candidates"][action] for r in results]
        candidates.append({"action": action, "metric_kind": "simulated_from_measured_source",
                           "prediction_and_payload_origin": "retained M30 measurements; invariance assumed",
                           "measurements": trials, "mean_reward": mean(t["reward"] for t in trials),
                           "mean_latency_ms": mean(t["latency_ms"] for t in trials),
                           "accuracy": mean(t["prediction_correct"] for t in trials),
                           "mean_communication_bytes": mean(t["communication_bytes"] for t in trials),
                           "mean_simulated_energy_proxy": mean(t["energy"]["value"] for t in trials)})
    best = max(c["mean_reward"] for c in candidates)
    winners = [c["action"] for c in candidates if best - c["mean_reward"] <= config["tie_tolerance"]]
    return {"dataset_version": VERSION, "schema_extension": EXTENSION,
            "sample_id": f"{source['sample_id']}/{profile['profile']}", "source_sample_id": source["sample_id"],
            "source_sha256": digest(source), "source_session": source["metadata"]["session_id"],
            "state": projections[0]["costs"]["state"], "network_condition": deepcopy(profile),
            "candidate_actions": candidates, "reward": {"configuration": config, "aggregation": "mean across source repeats"},
            "optimal_action": winners[0] if len(winners) == 1 else None, "optimal_actions": winners,
            "repeat_optimal_actions": [r["optimal_actions"] for r in results],
            "repeats": source["repeats"], "training_enabled": False}


def validate_controlled(sample, source, profile):
    if digest(sample) != digest(simulate_sample(source, profile)):
        raise ValueError("Controlled v2 schema/simulation/reward differs from source and profile")


def read_rows(path):
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def summarize(rows):
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate controlled records")
    return {"dataset_version": VERSION, "schema_extension": EXTENSION, "simulated": True,
            "samples": len(rows), "real_new_measurements": 0,
            "unique_source_samples": len({r["source_sample_id"] for r in rows}),
            "profiles": sorted({r["network_condition"]["profile"] for r in rows}),
            "action_coverage": {str(a): sum(r["repeats"] for r in rows) for a in range(4)},
            "label_distribution": {str(a): sum(r["optimal_action"] == a for r in rows) for a in range(4)},
            "ambiguous_labels": sum(r["optimal_action"] is None for r in rows),
            "training_allowed": False,
            "diversity_gate": {"all_four_optimal_labels_present": {r["optimal_action"] for r in rows} >= set(range(4)),
                               "independent_measured_conditions": False, "calibration_qualified": False,
                               "passed": False}}


def validate_output(output, source_directory):
    output, source_directory = Path(output), Path(source_directory)
    manifest = strict_json((output / "configuration.json").read_text())
    if manifest["version"] != EXTENSION or manifest["dataset_version"] != VERSION or manifest["training_enabled"] is not False:
        raise ValueError("Invalid controlled campaign version/training metadata")
    sources = read_rows(source_directory / f"{VERSION}.jsonl")
    if digest(sources) != manifest["source_dataset_sha256"]:
        raise ValueError("Source dataset hash mismatch")
    source_by_id = {s["sample_id"]: s for s in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate source sample identities")
    profile = manifest["network_condition"]
    rows = read_rows(output / f"{VERSION}.jsonl")
    if len(rows) != manifest["samples_requested"] or manifest["simulated"] is not True:
        raise ValueError("Campaign count/simulation metadata mismatch")
    if [r["source_sample_id"] for r in rows] != manifest["source_sample_ids"]:
        raise ValueError("Campaign source selection mismatch")
    for row in rows:
        validate_controlled(row, source_by_id[row["source_sample_id"]], profile)
    report = summarize(rows)
    report["valid"] = True
    return report


def run(args):
    profile = load_profile(args.profile)
    from .validate_matched_dataset import validate_campaign
    source_directory = Path(args.source)
    validate_campaign(source_directory)
    sources = read_rows(source_directory / f"{VERSION}.jsonl")
    if type(args.samples) is not int or not 1 <= args.samples <= len(sources):
        raise ValueError(f"samples must be 1..{len(sources)}; no silent duplicated windows")
    rows = [simulate_sample(source, profile) for source in sources[:args.samples]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": EXTENSION, "dataset_version": VERSION, "simulated": True,
                "timestamp": datetime.now(timezone.utc).isoformat(), "samples_requested": args.samples,
                "source_directory": source_directory.as_posix(), "source_dataset_sha256": digest(sources),
                "network_condition": profile, "source_sample_ids": [s["sample_id"] for s in sources[:args.samples]],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "implementation_sha256": {p.name: digest(p.read_text()) for p in
                    (Path(__file__), Path("tools/network_conditions/profiles.py"))}, "training_enabled": False}
    write_json(output / "configuration.json", manifest)
    with (output / f"{VERSION}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = validate_output(output, source_directory)
    write_json(output / "dataset_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=NAMES)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
