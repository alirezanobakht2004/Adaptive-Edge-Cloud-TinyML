from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import serial
import yaml
from serial import SerialException

try:
    from .serial_collector import EXPECTED_HEADER, EXPECTED_HEADER_LINE, validate_sample
except ImportError:
    from serial_collector import EXPECTED_HEADER, EXPECTED_HEADER_LINE, validate_sample


REPO_ROOT = Path(__file__).resolve().parents[1]
GESTURES_PATH = REPO_ROOT / "config" / "gestures.yaml"
HARDWARE_PATH = REPO_ROOT / "config" / "hardware.yaml"
VERSION_HEADER_PATH = REPO_ROOT / "firmware" / "include" / "version.h"
RAW_ROOT = REPO_ROOT / "data" / "raw"
METADATA_ROOT = REPO_ROOT / "data" / "metadata"

WINDOW_SAMPLES = 100
WINDOW_MS = 1000

VERSION_DEFINE_NAMES = {
    "firmware_version": "FIRMWARE_VERSION",
    "accel_calibration_version": "ACCEL_CALIBRATION_VERSION",
    "orientation_version": "ORIENTATION_VERSION",
    "dataset_version": "DATASET_VERSION",
}

BOOT_PREFIXES = {
    "firmware_version": "Firmware version:",
    "accel_calibration_version": "Accelerometer calibration:",
    "orientation_version": "Orientation protocol:",
    "dataset_version": "Dataset target:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record labeled 1-second IMU windows for dataset-v1."
    )
    parser.add_argument("--gesture", required=True, help="Gesture class name.")
    parser.add_argument("--user", default="user_01", help="Dataset user id.")
    parser.add_argument("--session", required=True, help="Session id, for example session_01.")
    parser.add_argument("--count", type=int, default=1, help="Number of windows to record.")
    parser.add_argument("--port", default="COM10", help="Serial port name.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before each capture.")
    parser.add_argument("--notes", default="", help="Optional note stored in metadata.")
    return parser.parse_args()


def load_gestures() -> list[str]:
    with GESTURES_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gestures = config.get("gestures", [])
    names = [str(item["name"]).upper() for item in gestures]

    if not names:
        raise ValueError("No gesture classes found in config/gestures.yaml.")

    return names


def load_hardware_config() -> dict:
    with HARDWARE_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    imu = config["imu"]

    return {
        "sample_rate_hz": int(imu["sampling_rate_hz"]),
        "accel_range_g": int(imu["accel_range_g"]),
        "gyro_range_dps": int(imu["gyro_range_dps"]),
        "who_am_i": str(imu["who_am_i"]),
    }


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


def sanitize_id(value: str, field_name: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '_' or '-'."
        )
    return value


def next_capture_index(directory: Path, gesture_slug: str) -> int:
    pattern = re.compile(rf"^{re.escape(gesture_slug)}_(\d{{3}})\.csv$")
    indices = []

    if directory.exists():
        for path in directory.iterdir():
            match = pattern.match(path.name)
            if match:
                indices.append(int(match.group(1)))

    return max(indices, default=0) + 1


def read_boot_and_verify(
    ser: serial.Serial,
    expected_versions: dict[str, str],
    expected_who_am_i: str,
) -> None:
    observed_versions: dict[str, str] = {}
    observed_who_am_i: str | None = None

    print("Waiting for ESP32 boot header...")
    print("Reset the ESP32 now and keep the device completely still during gyro calibration.")
    print()

    while True:
        raw_line = ser.readline()
        if not raw_line:
            continue

        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        print(f"[BOOT] {line}")

        for key, prefix in BOOT_PREFIXES.items():
            if line.startswith(prefix):
                observed_versions[key] = line[len(prefix):].strip()

        if line.startswith("WHO_AM_I:"):
            observed_who_am_i = line.split(":", 1)[1].strip().lower()

        if line != EXPECTED_HEADER_LINE:
            continue

        missing = [key for key in expected_versions if key not in observed_versions]
        if missing:
            raise RuntimeError(
                "Missing required version information in boot log: " + ", ".join(missing)
            )

        mismatches = [
            f"{key}: expected {expected_versions[key]!r}, observed {observed_versions[key]!r}"
            for key in expected_versions
            if observed_versions[key] != expected_versions[key]
        ]
        if mismatches:
            raise RuntimeError("Boot version mismatch: " + "; ".join(mismatches))

        if observed_who_am_i is None:
            raise RuntimeError("WHO_AM_I was not observed before the CSV header.")

        if observed_who_am_i != expected_who_am_i.lower():
            raise RuntimeError(
                f"WHO_AM_I mismatch: expected {expected_who_am_i}, observed {observed_who_am_i}."
            )

        print()
        print("Firmware/version verification passed.")
        print()
        ser.reset_input_buffer()
        return


def capture_window(ser: serial.Serial) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    malformed_lines = 0

    ser.reset_input_buffer()

    while len(rows) < WINDOW_SAMPLES:
        raw_line = ser.readline()
        if not raw_line:
            continue

        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        sample = validate_sample(line)
        if sample is None:
            if rows:
                malformed_lines += 1
            continue

        rows.append(sample)

    return rows, malformed_lines


def validate_timing(rows: list[list[str]], sample_rate_hz: int) -> None:
    if len(rows) != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {len(rows)}.")

    timestamps = [int(row[0]) for row in rows]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    expected_period_ms = 1000 // sample_rate_hz

    if any(delta <= 0 for delta in deltas):
        raise ValueError("Timestamps are not strictly increasing.")

    if any(delta != expected_period_ms for delta in deltas):
        raise ValueError(
            f"Dataset-v1 timing check failed: expected every delta to be "
            f"{expected_period_ms} ms."
        )


def write_capture(
    rows: list[list[str]],
    csv_path: Path,
    metadata_path: Path,
    metadata: dict,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Capture already exists: {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(EXPECTED_HEADER)
        writer.writerows(rows)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    try:
        user_id = sanitize_id(args.user, "user")
        session_id = sanitize_id(args.session, "session")

        if args.count <= 0:
            raise ValueError("count must be greater than zero.")
        if args.countdown < 0:
            raise ValueError("countdown must not be negative.")

        gestures = load_gestures()
        gesture = args.gesture.strip().upper()
        if gesture not in gestures:
            raise ValueError(
                f"Unsupported gesture {gesture!r}. Allowed: {', '.join(gestures)}"
            )

        hardware = load_hardware_config()
        versions = load_expected_versions()

        if hardware["sample_rate_hz"] != 100:
            raise ValueError("dataset-v1 recorder requires sampling_rate_hz=100.")

        gesture_slug = gesture.lower()
        raw_dir = RAW_ROOT / user_id / session_id
        metadata_dir = METADATA_ROOT / user_id / session_id
        capture_index = next_capture_index(raw_dir, gesture_slug)

        with serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=1.0,
        ) as ser:
            read_boot_and_verify(
                ser=ser,
                expected_versions=versions,
                expected_who_am_i=hardware["who_am_i"],
            )

            for item_number in range(1, args.count + 1):
                index = capture_index + item_number - 1
                filename = f"{gesture_slug}_{index:03d}.csv"
                metadata_filename = f"{gesture_slug}_{index:03d}.json"
                csv_path = raw_dir / filename
                metadata_path = metadata_dir / metadata_filename

                print(
                    f"[{item_number}/{args.count}] {gesture} -> "
                    f"{csv_path.relative_to(REPO_ROOT)}"
                )
                input("Place the device in orientation-v1 and press Enter when ready...")

                for remaining in range(args.countdown, 0, -1):
                    print(remaining, flush=True)
                    time.sleep(1.0)

                print("GO", flush=True)
                rows, malformed_lines = capture_window(ser)
                validate_timing(rows, hardware["sample_rate_hz"])

                metadata = {
                    "dataset_version": versions["dataset_version"],
                    "gesture": gesture,
                    "user": user_id,
                    "session": session_id,
                    "sample_rate_hz": hardware["sample_rate_hz"],
                    "window_ms": WINDOW_MS,
                    "sample_count": WINDOW_SAMPLES,
                    "accel_range_g": hardware["accel_range_g"],
                    "gyro_range_dps": hardware["gyro_range_dps"],
                    "firmware_version": versions["firmware_version"],
                    "accel_calibration_version": versions["accel_calibration_version"],
                    "orientation_version": versions["orientation_version"],
                    "source_csv": str(csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "malformed_lines_during_capture": malformed_lines,
                    "notes": args.notes,
                }

                write_capture(
                    rows=rows,
                    csv_path=csv_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )

                print(f"Saved {WINDOW_SAMPLES} samples.")
                print(f"Metadata: {metadata_path.relative_to(REPO_ROOT)}")
                print()

    except (ValueError, RuntimeError, FileExistsError, OSError, SerialException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nRecording cancelled by user.")
        return 130

    print("Recording complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
