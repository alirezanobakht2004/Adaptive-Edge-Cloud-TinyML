from __future__ import annotations

import csv
import math
from pathlib import Path

EXPECTED_COLUMNS = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]


def validate_window_csv(path: str | Path, expected_samples: int = 100) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Invalid columns: {reader.fieldnames}; expected {EXPECTED_COLUMNS}"
            )
        rows = list(reader)

    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} rows, got {len(rows)}")

    timestamps = []
    for row_idx, row in enumerate(rows):
        vals = []
        for name in EXPECTED_COLUMNS:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric {name} at row {row_idx}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {name} at row {row_idx}")
            vals.append(value)
        timestamps.append(vals[0])

    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Timestamps must be strictly increasing.")

    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    mean_period_ms = sum(deltas) / len(deltas)

    return {
        "rows": len(rows),
        "mean_period_ms": mean_period_ms,
        "approx_sampling_rate_hz": 1000.0 / mean_period_ms,
    }
