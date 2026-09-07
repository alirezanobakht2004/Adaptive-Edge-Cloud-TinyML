"""Single-writer JSONL collection and import of unchanged production serial logs."""

import argparse
import json
import math
from pathlib import Path
import re

from .schema import VERSION, ACTION_NAMES, schema_description, example_record
from .serializer import serialize
from .validator import read_records, validate_dataset
from .instrumentation import PolicySample

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data/policy" / VERSION


def initialize(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to overwrite a dataset directory")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset_version": VERSION, "actions": ACTION_NAMES, "feature_version": "features-v1",
                "source_dataset_version": "dataset-v1", "model_version": "gesture-model-v1.1.0",
                "energy": "estimated/simulated only", "writer_mode": "single writer; reject duplicate run/sample IDs"}
    for name, content in (("metadata.json", metadata), ("schema.json", schema_description()),
                          ("sample_000001.json", example_record())):
        (directory / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "observations.jsonl").write_text("", encoding="utf-8")
    (directory / "README.md").write_text(
        "# policy_training_dataset_v1\n\n"
        "Versioned collection directory. sample_000001.json is an explicitly unmeasured example; "
        "observations.jsonl starts empty and accepts observations only. Use the validator for its current count. "
        "Collection alone does not establish benchmark results or policy-training readiness.\n\n"
        "Actions: 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD. Action 3 is NOT MQTT split=3.\n\n"
        "Unknown metrics are null. Energy is an estimated/simulated proxy, never battery measured. "
        "Timestamps name their clock domain; host measurements are not ESP32 measurements.\n\n"
        "Validate: `python -m tools.policy_dataset.collector validate`. "
        "Import validated JSONL: `python -m tools.policy_dataset.collector collect --input records.jsonl`. "
        "Use --directory to select another dataset directory. One writer only.\n\n"
        "See docs/phase8_policy_dataset.md for the contract and instrumentation limits.\n", encoding="utf-8")


def collect(directory, records):
    directory = Path(directory)
    validate_dataset(directory)
    existing = read_records(directory / "observations.jsonl")
    seen = {(r["metadata"]["run_id"], r["metadata"]["sample_id"]) for r in existing}
    lines = []
    for record in records:
        line = serialize(record)
        if record["metadata"]["record_kind"] != "observation":
            raise ValueError("Example records cannot enter observations")
        key = record["metadata"]["run_id"], record["metadata"]["sample_id"]
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        lines.append(line)
    # Validate the whole batch first, so malformed input never causes a partial import.
    with (directory / "observations.jsonl").open("a", encoding="utf-8", newline="\n") as output:
        output.writelines(lines)
    return len(lines)


def production_serial_record(line, *, device_id, run_id, session_id, timestamp):
    """Import an existing W=... diagnostic line; receipt time must be supplied by the caller."""
    pairs = dict(re.findall(r"(\w+)=([^\s]+)", line))
    required = {"W", "c", "conf", "unc", "pipe_us", "feat", "norm", "prefix", "mc"}
    if not required <= pairs.keys():
        raise ValueError("Not a complete production inference diagnostic line")
    sample = PolicySample.create(sample_id=f"{device_id}-{pairs['W']}", device_id=device_id,
                                 window_id=pairs["W"], run_id=run_id, session_id=session_id, scope="device")
    record = sample.record
    record["metadata"]["timestamp"] = timestamp
    measurements = record["measurements"]
    measurements.update(preprocessing_latency_ms=(int(pairs["feat"]) + int(pairs["norm"])) / 1000,
                        prefix_latency_ms=int(pairs["prefix"]) / 1000,
                        local_inference_latency_ms=(int(pairs["prefix"]) + int(pairs["mc"])) / 1000,
                        total_latency_ms=int(pairs["pipe_us"]) / 1000, request_bytes=0, response_bytes=0,
                        local_confidence=float(pairs["conf"]), local_entropy=float(pairs["unc"]) * math.log(5))
    record["outcome"].update(status="success", predicted_class=int(pairs["c"]), confidence=float(pairs["conf"]))
    record["provenance"].update(state_source="unavailable: log is post-inference", notes=
        "Existing production serial log; unc converted from normalized entropy to nats. "
        "Margin/heap/arena/frequency/network state are unavailable. No pre-decision state is inferred.")
    record["provenance"]["clock_domains"] = {"timestamp": "host UTC receipt supplied by caller",
                                               "durations": "ESP32 micros()"}
    serialize(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "validate", "collect"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.directory)
    elif args.command == "collect":
        if args.input is None:
            parser.error("collect requires --input")
        print(json.dumps({"imported": collect(args.directory, read_records(args.input))}))
    print(json.dumps(validate_dataset(args.directory), indent=2))


if __name__ == "__main__":
    main()
