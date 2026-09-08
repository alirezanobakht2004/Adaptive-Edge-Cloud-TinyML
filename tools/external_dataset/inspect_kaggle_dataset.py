"""Inspect the supplied CSV without modifying it or inferring gesture semantics."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/kaggle_6axis_motion_v1/gesture_dataset.csv"
SENSORS = ("ax", "ay", "az", "gx", "gy", "gz")
REQUIRED = ("label", "sample_id", "timestamp_ms", *SENSORS)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_source(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Missing or duplicate CSV column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row width")
    return columns, rows


def grouped_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["sample_id"])].append(row)
    return groups


def inspect(path=SOURCE, source_context=None):
    path = Path(path)
    columns, rows = read_source(path)
    missing = {key: sum(not row[key].strip() for row in rows) for key in columns}
    invalid = {}
    for key in ("timestamp_ms", *SENSORS):
        if key not in columns:
            continue
        count = 0
        for row in rows:
            try:
                value = float(row[key])
                count += not np.isfinite(value) or (key == "timestamp_ms" and not value.is_integer())
            except ValueError:
                count += 1
        invalid[key] = int(count)
    report = {"source_name": "kaggle_6axis_motion_v1 (user-supplied name)", "source_filename": path.name,
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_url": None, "source_documentation_available": False, "columns": columns,
              "row_count": len(rows), "sensor_fields": [key for key in SENSORS if key in columns],
              "missing_columns": sorted(set(REQUIRED) - set(columns)), "missing_values": missing,
              "invalid_numeric_values": invalid, "timestamp_available": "timestamp_ms" in columns,
              "duplicate_rows": len(rows) - len({tuple(row[key] for key in columns) for row in rows}),
              "units": {"acceleration": "unknown", "gyroscope": "unknown"}, "axis_orientation": "unknown"}
    if source_context is not None:
        context = json.loads(Path(source_context).read_text(encoding="utf-8"))
        if context["source_sha256"] != report["source_sha256"]:
            raise ValueError("Source context does not match this CSV hash")
        report.update(source_name=context["source_dataset_name"], source_url=context["source_url"],
                      source_documentation_available=True, units=context["units"],
                      units_evidence=context["units_evidence"], source_context=context)
    if report["missing_columns"] or any(missing.values()) or any(invalid.values()):
        report.update(recording_count=None, compatibility_notes=["Resolve missing/invalid fields before conversion."])
        return report
    groups = grouped_rows(rows)
    id_labels = defaultdict(set)
    content_counts = Counter()
    deltas = Counter()
    for (label, sample_id), group in groups.items():
        id_labels[sample_id].add(label)
        times = np.asarray([int(row["timestamp_ms"]) for row in group], dtype=np.int64)
        deltas.update(map(int, np.diff(times)))
        values = np.asarray([[float(row[key]) for key in SENSORS] for row in group], dtype="<f8")
        content_counts[hashlib.sha256(values.tobytes()).hexdigest()] += 1
    report.update(recording_count=len(groups), recording_identity=["label", "sample_id"],
                  sample_ids_reused_across_labels=sum(len(labels) > 1 for labels in id_labels.values()),
                  classes=sorted({label for label, _ in groups}),
                  class_distribution_rows=dict(Counter(row["label"] for row in rows)),
                  class_distribution_recordings=dict(Counter(label for label, _ in groups)),
                  samples_per_recording=dict(Counter(map(len, groups.values()))),
                  within_recording_timestamp_delta_ms=dict(sorted(deltas.items())),
                  nonpositive_timestamp_deltas=sum(count for delta, count in deltas.items() if delta <= 0),
                  inferred_sampling_hz=1000 / next(iter(deltas)) if len(deltas) == 1 and next(iter(deltas)) > 0 else None,
                  duplicate_sensor_recordings=sum(count - 1 for count in content_counts.values()),
                  compatibility_notes=["Rows are time samples; recordings are (label, sample_id) groups in source order.",
                                       "timestamp_ms supplies relative timing, not verified UTC timestamps.",
                                       "features-v1 requires 100 samples, 100 Hz, acceleration in g and gyro in degrees/s.",
                                       "Units, orientation, participant/session provenance and gesture definitions are undocumented in this CSV.",
                                       "Label names do not authorize wave/flick mappings to production gestures."])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/external_dataset_inspection.json")
    parser.add_argument("--source-context", type=Path)
    args = parser.parse_args()
    report = inspect(args.input, args.source_context)
    write_json(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("row_count", "recording_count", "classes", "inferred_sampling_hz")}))


if __name__ == "__main__":
    main()
