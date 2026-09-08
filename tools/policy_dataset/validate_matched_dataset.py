"""Revalidate v2 rewards against raw device/server evidence and source labels."""

import argparse
import hashlib
from pathlib import Path
from statistics import mean

from .matched_dataset import VERSION, validate_file
from .reward_engine.__main__ import strict_json
from .reward_engine.engine import digest
from .run_matched_campaign import write_json


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_campaign(directory):
    directory = Path(directory)
    manifest = strict_json((directory / "configuration.json").read_text(encoding="utf-8"))
    path = directory / f"{VERSION}.jsonl"
    report = validate_file(path)
    samples = read_jsonl(path)
    if manifest["dataset_version"] != VERSION or manifest["status"] not in ("complete", "complete_with_rejections"):
        raise ValueError("Campaign is not complete")
    if len(samples) != manifest["matched_samples"] or len(samples) + len(manifest["rejected_windows"]) != manifest["samples_requested"]:
        raise ValueError("Manifest sample counts mismatch")
    raw = read_jsonl(directory / "raw_entries.jsonl")
    events = read_jsonl(directory / "server_events.jsonl")
    raw_by_id, server_by_id = {}, {}
    for entry in raw:
        identity = entry["trace"]["request_id"]
        if identity in raw_by_id:
            raise ValueError("Duplicate raw trace identity")
        raw_by_id[identity] = entry
    for event in events:
        if event["request_id"] in server_by_id:
            raise ValueError("Duplicate server event identity")
        server_by_id[event["request_id"]] = event
    if len(raw) != manifest["samples_requested"] * manifest["repeats"] * 4:
        raise ValueError("Incomplete raw execution evidence")
    from ml.features.extractor import load_feature_split
    features = load_feature_split("validation")
    if (manifest["feature_matrix_sha256"] != hashlib.sha256(features.features.tobytes()).hexdigest()
            or manifest["labels_sha256"] != hashlib.sha256(features.labels.tobytes()).hexdigest()
            or manifest["source_session"] != features.session):
        raise ValueError("Source validation features/labels changed")
    used, windows = set(), set()
    for sample in samples:
        if sample["condition"] != manifest["condition"] or sample["repeats"] != manifest["repeats"]:
            raise ValueError("Condition/repeat manifest mismatch")
        window = int(sample["state"]["application"]["window_id"])
        if window in windows or window not in manifest["window_indices"]:
            raise ValueError("Duplicate/unrequested source window")
        windows.add(window)
        for entry in sample["source_entries"]:
            trace, record = entry["trace"], entry["record"]
            identity = trace["request_id"]
            if identity in used or raw_by_id.get(identity) != {"trace": trace, "guard": entry["guard"]}:
                raise ValueError("Raw trace mismatch/duplicate across samples")
            used.add(identity)
            if (record["metadata"]["run_id"] != manifest["run_id"]
                    or record["metadata"]["session_id"] != features.session
                    or record["outcome"]["true_class"] != int(features.labels[window])
                    or record["provenance"]["artifact_hashes"] != manifest["artifact_hashes"]):
                raise ValueError("Source label/run/model provenance mismatch")
            if trace["action"]:
                event = server_by_id.get(identity)
                if event is None:
                    raise ValueError("Missing server event")
                response = event["response"]
                if (response["action"] != trace["action"] or response["predicted_class_id"] != trace["prediction"]
                        or response["model_version"] != manifest["model_versions"][str(trace["action"])]
                        or event["publish_rc"] != 0 or event["request_bytes"] != trace["request_bytes"]
                        or event["response_bytes"] != trace["response_bytes"]
                        or abs(response["confidence"] - trace["result_confidence"]) > 1e-6
                        or abs(response["server_latency_ms"] - trace["cloud_ms"]) > 1e-5
                        or any(record["measurements"][k] != event[k] for k in ("request_receive_time", "response_publish_time"))):
                    raise ValueError("Server/device evidence mismatch")
    report["campaign_run_id"] = manifest["run_id"]
    report["dataset_sha256"] = digest(samples)
    report["source_and_server_evidence_validated"] = True
    report["rejected_windows"] = manifest["rejected_windows"]
    report["condition"] = manifest["condition"]
    report["action_metrics"] = {}
    for action in range(4):
        rows = [s["candidate_actions"][action] for s in samples]
        report["action_metrics"][str(action)] = {key: mean(r[key] for r in rows) for key in
            ("accuracy", "mean_reward", "mean_latency_ms", "mean_communication_bytes", "mean_estimated_energy_proxy")}
    report["repeat_label_agreement_samples"] = sum(
        all(winners == s["label"]["optimal_actions"] for winners in s["label"]["repeat_optimal_actions"]) for s in samples)
    report["maximum_observed_drift"] = {key: max(s["maximum_observed_drift"][key] for s in samples)
                                          for key in samples[0]["maximum_observed_drift"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Campaign directory with raw/server evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_campaign(args.directory)
    write_json(args.output, report)
    print(f"Validated {report['matched_samples']} matched samples and {report['candidate_executions']} executions")


if __name__ == "__main__":
    main()
