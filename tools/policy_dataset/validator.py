"""Validate dataset headers, observations and duplicate sample identities."""

import json
from pathlib import Path

from .schema import VERSION, ACTION_NAMES, schema_description
from .serializer import deserialize


def read_records(path):
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = deserialize(line)
                key = (record["metadata"]["run_id"], record["metadata"]["sample_id"])
                if key in seen:
                    raise ValueError(f"Duplicate sample identity: {key}")
                seen.add(key)
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def validate_dataset(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_version") != VERSION:
        raise ValueError("Dataset metadata version mismatch")
    for key, value in (("model_version", "gesture-model-v1.1.0"), ("feature_version", "features-v1"),
                       ("source_dataset_version", "dataset-v1")):
        if metadata.get(key) != value:
            raise ValueError(f"Dataset metadata version mismatch: {key}")
    if metadata.get("actions") != {str(k): v for k, v in ACTION_NAMES.items()}:
        raise ValueError("Dataset action mapping mismatch")
    expected_schema = json.loads(json.dumps(schema_description()))
    if json.loads((directory / "schema.json").read_text(encoding="utf-8")) != expected_schema:
        raise ValueError("Schema description differs from canonical version")
    example = deserialize((directory / "sample_000001.json").read_text(encoding="utf-8"))
    if example["metadata"]["record_kind"] != "example":
        raise ValueError("Example must be labeled example")
    records = read_records(directory / "observations.jsonl")
    for record in records:
        if record["metadata"]["record_kind"] != "observation":
            raise ValueError("Examples cannot enter observations.jsonl")
    return {"observations": len(records), "examples": 1,
            "training_ready": False, "reason": "Collection validation is not policy-training qualification"}


def validate_campaign(directory):
    """Cross-check run configuration, raw device traces and their v1 projection."""
    from .device_trace import validate_trace
    directory = Path(directory)
    configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    for key, expected in (("dataset_version", VERSION), ("source_dataset_version", "dataset-v1"),
                          ("feature_version", "features-v1"), ("model_version", "gesture-model-v1.1.0")):
        if configuration.get(key) != expected:
            raise ValueError(f"Campaign version mismatch: {key}")
    modes = {"local": 0, "split1": 1, "split2": 2, "cloud": 3}
    if configuration.get("mode") not in modes or configuration.get("action") != modes[configuration["mode"]]:
        raise ValueError("Campaign mode/action mismatch")
    records = read_records(directory / f"{VERSION}.jsonl")
    traces = [json.loads(line) for line in (directory / "raw_traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(traces) or len(records) != configuration["samples_collected"]:
        raise ValueError("Campaign record/trace count mismatch")
    if configuration["status"] == "complete" and len(records) != configuration["samples_requested"]:
        raise ValueError("Incomplete campaign marked complete")
    for index, (record, trace) in enumerate(zip(records, traces)):
        validate_trace(trace)
        if (record["metadata"]["run_id"] != configuration["run_id"]
                or record["metadata"]["session_id"] != configuration["input_session"]
                or trace["window_id"] != configuration["window_indices"][index]
                or record["state"]["application"]["window_id"] != str(trace["window_id"])
                or record["action"] != configuration["action"] or trace["action"] != record["action"]
                or record["outcome"]["predicted_class"] != trace["prediction"]
                or record["outcome"]["status"] != trace["status"]
                or record["measurements"]["total_latency_ms"] != trace["total_us"] / 1000
                or record["provenance"]["artifact_hashes"] != configuration["artifact_hashes"]):
            raise ValueError(f"Campaign trace/projection mismatch at sample {index}")
    return {"status": configuration["status"], "observations": len(records)}
