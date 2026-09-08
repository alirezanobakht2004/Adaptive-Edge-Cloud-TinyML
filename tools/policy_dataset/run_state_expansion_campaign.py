"""Generate and validate simulated M33 policy_training_dataset_v3 campaigns."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .reward_engine.engine import digest
from .reward_engine.__main__ import strict_json
from .run_controlled_campaign import DEFAULT_SOURCE, read_rows
from .run_matched_campaign import write_json
from .state_expansion import VERSION, SCENARIOS, coverage_report, expand_sample, load_scenario, validate_v3
from .validate_matched_dataset import validate_campaign


def validate_output(directory, source_directory=DEFAULT_SOURCE):
    directory, source_directory = Path(directory), Path(source_directory)
    validate_campaign(source_directory)
    metadata = strict_json((directory / "metadata.json").read_text())
    if metadata["dataset_version"] != VERSION or metadata["simulated"] is not True or metadata["training_enabled"] is not False:
        raise ValueError("Invalid v3 version/provenance/training metadata")
    sources = read_rows(source_directory / "policy_training_dataset_v2.jsonl")
    if digest(sources) != metadata["source_sha256"]:
        raise ValueError("Source hash mismatch")
    source_by_id = {s["sample_id"]: s for s in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate source identity")
    rows = read_rows(directory / f"{VERSION}.jsonl")
    expected_pairs = {(s, name) for s in metadata["selected_source_ids"] for name in metadata["scenarios"]}
    actual_pairs = {(r["source_sample_id"], r["scenario"]["name"]) for r in rows}
    if len(rows) != metadata["sample_count"] or len(rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError("Campaign selection/count/duplicate mismatch")
    for row in rows:
        scenario = metadata["scenarios"][row["scenario"]["name"]]
        validate_v3(row, source_by_id[row["source_sample_id"]], scenario)
    report = coverage_report(rows)
    report["valid"] = True
    return report


def run(args):
    source_directory = Path(args.source)
    validate_campaign(source_directory)
    sources = read_rows(source_directory / "policy_training_dataset_v2.jsonl")
    if type(args.samples) is not int or not 1 <= args.samples <= len(sources):
        raise ValueError(f"samples must be 1..{len(sources)}")
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    scenarios = {name: load_scenario(name) for name in names}
    selected = sources[:args.samples]
    rows = [expand_sample(source, scenario) for scenario in scenarios.values() for source in selected]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"dataset_version": VERSION, "simulated": True, "training_enabled": False,
                "timestamp": datetime.now(timezone.utc).isoformat(), "source_directory": source_directory.as_posix(),
                "source_sha256": digest(sources), "source_schema_version": "policy_training_dataset_v2",
                "selected_source_ids": [s["sample_id"] for s in selected], "scenarios": scenarios,
                "sample_count": len(rows), "independent_source_windows": len(selected),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "implementation_hashes": {p.name: digest(p.read_text()) for p in
                    (Path(__file__), Path(__file__).with_name("state_expansion.py"))}}
    write_json(output / "metadata.json", metadata)
    with (output / f"{VERSION}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    schema = {"dataset_version": VERSION, "required_record_fields": list(rows[0]),
              "new_state_fields": {"state.device.local_compute_pressure": "float [0,1], simulated; available compute multiplier=1/(1-pressure)",
                  "state.device.inference_queue_pressure": "float [0,1], simulated; queued wait=pressure*queue_reference_ms",
                  "state.network.cloud_availability": "boolean, simulated cloud service availability",
                  "state.network.network_quality_score": "float [0,1], derived from simulated RTT and configured reference"},
              "availability_fields": ["state.device.local_inference_available", "state.network.transport_available"],
              "all_actions": {"0": "ALL_LOCAL", "1": "SPLIT1", "2": "SPLIT2", "3": "ALL_CLOUD"},
              "infeasible_action": "Explicit reason; outcomes, measurements, reward and components null; excluded from argmax",
              "missing_state": "Null means unavailable/unmeasured, not zero; C has no fresh uncertainty",
              "state_provenance": "One provenance entry per state leaf; no new measured values",
              "validator": "tools.policy_dataset.run_state_expansion_campaign.validate_output",
              "training_enabled": False}
    write_json(output / "schema_description.json", schema)
    report = validate_output(output, source_directory)
    write_json(output / "state_coverage_report.json", report)
    print(json.dumps({k: report[k] for k in ("samples", "label_distribution", "split_coverage", "minimum_readiness_criteria", "training_allowed")}, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--samples", type=int, required=True, help="Distinct accepted source windows per scenario")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
