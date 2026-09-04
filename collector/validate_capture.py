from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_PATH = REPO_ROOT / "config" / "hardware.yaml"
GESTURES_PATH = REPO_ROOT / "config" / "gestures.yaml"
VERSION_HEADER_PATH = REPO_ROOT / "firmware" / "include" / "version.h"

EXPECTED_COLUMNS = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]
EXPECTED_SAMPLES = 100

VERSION_DEFINE_NAMES = {
    "firmware_version": "FIRMWARE_VERSION",
    "accel_calibration_version": "ACCEL_CALIBRATION_VERSION",
    "orientation_version": "ORIENTATION_VERSION",
    "dataset_version": "DATASET_VERSION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one dataset-v1 gesture capture and its metadata."
    )
    parser.add_argument("csv_path", type=Path, help="Path to one raw gesture CSV file.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON path. By default it is inferred from data/raw/...",
    )
    return parser.parse_args()


def load_expected_versions() -> dict[str, str]:
    text = VERSION_HEADER_PATH.read_text(encoding="utf-8")
    versions: dict[str, str] = {}

    for key, define_name in VERSION_DEFINE_NAMES.items():
        match = re.search(
            rf'^#define\s+{re.escape(define_name)}\s+"([^"]+)"\s*$',
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"Missing {define_name} in firmware/include/version.h.")
        versions[key] = match.group(1)

    return versions


def load_expected_hardware() -> dict:
    with HARDWARE_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    imu = config["imu"]
    return {
        "sample_rate_hz": int(imu["sampling_rate_hz"]),
        "accel_range_g": int(imu["accel_range_g"]),
        "gyro_range_dps": int(imu["gyro_range_dps"]),
    }


def load_gestures() -> set[str]:
    with GESTURES_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return {str(item["name"]).upper() for item in config.get("gestures", [])}


def infer_metadata_path(csv_path: Path) -> Path:
    resolved = csv_path.resolve()
    raw_root = (REPO_ROOT / "data" / "raw").resolve()

    try:
        relative = resolved.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(
            "Cannot infer metadata path because CSV is outside data/raw. "
            "Use --metadata explicitly."
        ) from exc

    return REPO_ROOT / "data" / "metadata" / relative.with_suffix(".json")


def validate_csv(csv_path: Path, sample_rate_hz: int) -> dict:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Invalid columns: {reader.fieldnames}; expected {EXPECTED_COLUMNS}."
            )
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} rows, got {len(rows)}.")

    timestamps: list[int] = []

    for row_index, row in enumerate(rows, start=2):
        try:
            timestamp_ms = int(row["timestamp_ms"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp_ms at CSV line {row_index}.") from exc

        if timestamp_ms < 0:
            raise ValueError(f"Negative timestamp_ms at CSV line {row_index}.")

        for name in EXPECTED_COLUMNS[1:]:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric {name} at CSV line {row_index}.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {name} at CSV line {row_index}.")

        timestamps.append(timestamp_ms)

    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    expected_period_ms = 1000 // sample_rate_hz

    if any(delta <= 0 for delta in deltas):
        raise ValueError("Timestamps must be strictly increasing.")

    if any(delta != expected_period_ms for delta in deltas):
        bad = [delta for delta in deltas if delta != expected_period_ms]
        raise ValueError(
            f"Timing mismatch: expected every delta to be {expected_period_ms} ms; "
            f"found {len(bad)} mismatched interval(s)."
        )

    return {
        "rows": len(rows),
        "first_timestamp_ms": timestamps[0],
        "last_timestamp_ms": timestamps[-1],
        "period_ms": expected_period_ms,
        "duration_ms": timestamps[-1] - timestamps[0],
    }


def validate_metadata(
    csv_path: Path,
    metadata_path: Path,
    expected_versions: dict[str, str],
    hardware: dict,
    gestures: set[str],
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required = {
        "dataset_version",
        "gesture",
        "user",
        "session",
        "sample_rate_hz",
        "window_ms",
        "sample_count",
        "accel_range_g",
        "gyro_range_dps",
        "firmware_version",
        "accel_calibration_version",
        "orientation_version",
        "source_csv",
        "malformed_lines_during_capture",
        "notes",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError("Metadata is missing fields: " + ", ".join(missing))

    expected_values = {
        "dataset_version": expected_versions["dataset_version"],
        "firmware_version": expected_versions["firmware_version"],
        "accel_calibration_version": expected_versions["accel_calibration_version"],
        "orientation_version": expected_versions["orientation_version"],
        "sample_rate_hz": hardware["sample_rate_hz"],
        "window_ms": 1000,
        "sample_count": EXPECTED_SAMPLES,
        "accel_range_g": hardware["accel_range_g"],
        "gyro_range_dps": hardware["gyro_range_dps"],
    }

    for key, expected in expected_values.items():
        if metadata[key] != expected:
            raise ValueError(
                f"Metadata mismatch for {key}: expected {expected!r}, got {metadata[key]!r}."
            )

    gesture = str(metadata["gesture"]).upper()
    if gesture not in gestures:
        raise ValueError(f"Unsupported gesture in metadata: {gesture!r}.")

    expected_stem = gesture.lower()
    if not re.fullmatch(rf"{re.escape(expected_stem)}_\d{{3}}", csv_path.stem):
        raise ValueError(
            f"CSV filename {csv_path.name!r} does not match gesture {gesture!r}."
        )

    expected_source = csv_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    observed_source = str(metadata["source_csv"]).replace("\\", "/")
    if observed_source != expected_source:
        raise ValueError(
            f"Metadata source_csv mismatch: expected {expected_source!r}, "
            f"got {metadata['source_csv']!r}."
        )

    raw_root = (REPO_ROOT / "data" / "raw").resolve()
    try:
        relative = csv_path.resolve().relative_to(raw_root)
    except ValueError as exc:
        raise ValueError("CSV must be stored under data/raw for dataset-v1.") from exc

    if len(relative.parts) != 3:
        raise ValueError(
            "CSV path must follow data/raw/<user>/<session>/<gesture>_NNN.csv."
        )

    user_id, session_id, _ = relative.parts
    if metadata["user"] != user_id:
        raise ValueError(
            f"Metadata user mismatch: path has {user_id!r}, metadata has {metadata['user']!r}."
        )
    if metadata["session"] != session_id:
        raise ValueError(
            f"Metadata session mismatch: path has {session_id!r}, metadata has {metadata['session']!r}."
        )

    if int(metadata["malformed_lines_during_capture"]) != 0:
        raise ValueError(
            "Capture recorded malformed serial line(s); recollect this window for dataset-v1."
        )

    return metadata


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path

    try:
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        metadata_path = args.metadata
        if metadata_path is None:
            metadata_path = infer_metadata_path(csv_path)
        elif not metadata_path.is_absolute():
            metadata_path = (Path.cwd() / metadata_path).resolve()

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        versions = load_expected_versions()
        hardware = load_expected_hardware()
        gestures = load_gestures()

        csv_result = validate_csv(csv_path, hardware["sample_rate_hz"])
        metadata = validate_metadata(
            csv_path=csv_path,
            metadata_path=metadata_path,
            expected_versions=versions,
            hardware=hardware,
            gestures=gestures,
        )

        print("PASS: dataset-v1 capture is structurally valid.")
        print(f"CSV: {csv_path.relative_to(REPO_ROOT)}")
        print(f"Metadata: {metadata_path.relative_to(REPO_ROOT)}")
        print(f"Gesture: {metadata['gesture']}")
        print(f"Samples: {csv_result['rows']}")
        print(f"Period: {csv_result['period_ms']} ms")
        print(f"Duration: {csv_result['duration_ms']} ms")
        return 0

    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
