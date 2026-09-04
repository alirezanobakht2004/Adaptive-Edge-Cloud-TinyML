from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import serial
from serial import SerialException


EXPECTED_HEADER = [
    "timestamp_ms",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

EXPECTED_HEADER_LINE = ",".join(EXPECTED_HEADER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture calibrated ESP32 IMU samples from a serial port."
    )

    parser.add_argument(
        "--port",
        default="COM10",
        help="Serial port name, for example COM10.",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path.",
    )

    return parser.parse_args()


def validate_sample(line: str) -> list[str] | None:
    parts = [part.strip() for part in line.split(",")]

    if len(parts) != 7:
        return None

    try:
        timestamp_ms = int(parts[0])

        values = [
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
        ]
    except ValueError:
        return None

    if timestamp_ms < 0:
        return None

    if not all(math.isfinite(value) for value in values):
        return None

    return parts


def collect_serial_data(
    port: str,
    baud: int,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_samples = 0
    malformed_lines = 0
    header_found = False

    try:
        with serial.Serial(
            port=port,
            baudrate=baud,
            timeout=1.0,
        ) as ser:
            print(f"Connected to {port} at {baud} baud.")
            print("Waiting for CSV header...")
            print("Reset the ESP32 if the header has already passed.")
            print("Press Ctrl+C to stop.")
            print()

            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                while True:
                    raw_line = ser.readline()

                    if not raw_line:
                        continue

                    line = raw_line.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()

                    if not line:
                        continue

                    if not header_found:
                        print(f"[BOOT] {line}")

                        if line == EXPECTED_HEADER_LINE:
                            header_found = True
                            writer.writerow(EXPECTED_HEADER)
                            file.flush()

                            print()
                            print("CSV header detected.")
                            print("Capture started.")
                            print()

                        continue

                    sample = validate_sample(line)

                    if sample is None:
                        malformed_lines += 1
                        print(
                            f"[SKIP] Malformed line: {line}",
                            file=sys.stderr,
                        )
                        continue

                    writer.writerow(sample)
                    valid_samples += 1

                    if valid_samples % 100 == 0:
                        file.flush()

                        print(
                            f"Captured {valid_samples} samples..."
                        )

    except KeyboardInterrupt:
        print()
        print("Capture stopped by user.")

    except SerialException as exc:
        print(
            f"Serial error: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        print()
        print("Capture summary:")
        print(f"  Valid samples   : {valid_samples}")
        print(f"  Malformed lines : {malformed_lines}")
        print(f"  Output file     : {output_path}")

    if not header_found:
        print(
            "ERROR: CSV header was never detected.",
            file=sys.stderr,
        )
        return 1

    if valid_samples == 0:
        print(
            "ERROR: No valid samples were captured.",
            file=sys.stderr,
        )
        return 1

    return 0


def main() -> int:
    args = parse_args()

    return collect_serial_data(
        port=args.port,
        baud=args.baud,
        output_path=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())