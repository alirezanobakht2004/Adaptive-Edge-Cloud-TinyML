from __future__ import annotations

import csv
from pathlib import Path

import pytest

from collector.validate_capture import validate_csv


HEADER = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]


def write_capture(path: Path, timestamps: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for timestamp in timestamps:
            writer.writerow([timestamp, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_validate_csv_accepts_exact_dataset_v1_window(tmp_path: Path) -> None:
    path = tmp_path / "idle_001.csv"
    write_capture(path, [4000 + 10 * index for index in range(100)])

    result = validate_csv(path, sample_rate_hz=100)

    assert result["rows"] == 100
    assert result["period_ms"] == 10
    assert result["duration_ms"] == 990


def test_validate_csv_rejects_wrong_sample_count(tmp_path: Path) -> None:
    path = tmp_path / "idle_001.csv"
    write_capture(path, [4000 + 10 * index for index in range(99)])

    with pytest.raises(ValueError, match="Expected 100 rows"):
        validate_csv(path, sample_rate_hz=100)


def test_validate_csv_rejects_missing_interval(tmp_path: Path) -> None:
    path = tmp_path / "idle_001.csv"
    timestamps = [4000 + 10 * index for index in range(100)]
    timestamps[50:] = [value + 10 for value in timestamps[50:]]
    write_capture(path, timestamps)

    with pytest.raises(ValueError, match="Timing mismatch"):
        validate_csv(path, sample_rate_hz=100)
