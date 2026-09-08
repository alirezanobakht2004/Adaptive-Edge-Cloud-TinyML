"""Lossless numeric conversion into isolated ragged recording arrays; no resampling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from .inspect_kaggle_dataset import ROOT, SOURCE, SENSORS, inspect, read_source, grouped_rows, write_json

EXTERNAL_ROOT = ROOT / "data/external"
OUTPUT = SOURCE.parent / "converted"
CONVERSION_VERSION = "external-recordings-v1"


def isolated_output(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(EXTERNAL_ROOT.resolve()) or resolved == EXTERNAL_ROOT.resolve():
        raise ValueError("External output must stay under data/external; production paths are forbidden")
    return resolved


def convert(source=SOURCE, output=OUTPUT, source_context=None):
    output = isolated_output(output)
    report = inspect(source, source_context)
    if (report["missing_columns"] or any(report["missing_values"].values())
            or any(report["invalid_numeric_values"].values()) or not report["row_count"]
            or report.get("nonpositive_timestamp_deltas")):
        raise ValueError("Invalid source rows/timestamps; conversion never silently drops or sorts samples")
    _, rows = read_source(source)
    groups = grouped_rows(rows)
    values, times, labels, ids, offsets = [], [], [], [], [0]
    for (label, sample_id), group in groups.items():
        labels.append(label)
        ids.append(sample_id)
        values.extend([[float(row[key]) for key in SENSORS] for row in group])
        times.extend([int(row["timestamp_ms"]) for row in group])
        offsets.append(len(values))
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "recordings.npz"
    np.savez_compressed(archive, sensor_values=np.asarray(values, dtype=np.float64),
                        timestamps_ms=np.asarray(times, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64),
                        labels=np.asarray(labels, dtype=str), sample_ids=np.asarray(ids, dtype=str))
    metadata = {"source_dataset_name": report["source_name"], "source_filename": Path(source).name,
                "source_sha256": report["source_sha256"], "source_url": report["source_url"],
                "original_labels": report["classes"], "conversion_version": CONVERSION_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(), "recording_count": len(labels), "row_count": len(values),
                "recording_identity": ["label", "sample_id"], "sensor_columns": list(SENSORS),
                "axis_mapping": {name: name for name in SENSORS}, "axis_orientation": "unknown",
                "axis_mapping_assumptions": "Preserve named sensor axes and signs; no rotation or mounting correction.",
                "units": report["units"], "source_context": report.get("source_context"),
                "sampling_hz_observed": report["inferred_sampling_hz"], "samples_per_recording": report["samples_per_recording"],
                "resampling": "none", "unit_conversion": "none", "dtype": "float64 sensor values; int64 timestamps/offsets",
                "array_layout": "Recording i uses sensor_values[offsets[i]:offsets[i+1]] and matching timestamps; original label/id arrays align.",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "production_dataset_compatible": False}
    write_json(output / "external_dataset_metadata.json", metadata)
    return metadata


def load_converted(directory=OUTPUT):
    directory = Path(directory)
    metadata = json.loads((directory / "external_dataset_metadata.json").read_text(encoding="utf-8"))
    archive = directory / "recordings.npz"
    if metadata["conversion_version"] != CONVERSION_VERSION or hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("Converted artifact version/hash mismatch")
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    offsets = arrays["offsets"]
    if (arrays["sensor_values"].shape != (metadata["row_count"], 6)
            or offsets.shape != (metadata["recording_count"] + 1,) or offsets[0] != 0
            or offsets[-1] != metadata["row_count"] or np.any(np.diff(offsets) <= 0)
            or len(arrays["labels"]) != metadata["recording_count"]
            or len(arrays["sample_ids"]) != metadata["recording_count"]
            or arrays["timestamps_ms"].shape != (metadata["row_count"],)
            or not np.isfinite(arrays["sensor_values"]).all()):
        raise ValueError("Converted array contract mismatch")
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-context", type=Path)
    args = parser.parse_args()
    metadata = convert(args.input, args.output, args.source_context)
    print(json.dumps({"converted_recordings": metadata["recording_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
