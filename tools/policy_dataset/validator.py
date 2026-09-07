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
