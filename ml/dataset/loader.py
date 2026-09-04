from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATASET_VERSION = "dataset-v1"
MANIFEST_RELATIVE_PATH = Path("data/splits/dataset-v1.json")

GESTURES = (
    "IDLE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "ROTATE_CW",
    "SHAKE",
)

CLASS_TO_ID = {gesture: index for index, gesture in enumerate(GESTURES)}
ID_TO_CLASS = {index: gesture for gesture, index in CLASS_TO_ID.items()}

CSV_COLUMNS = (
    "timestamp_ms",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
)

SENSOR_COLUMNS = CSV_COLUMNS[1:]
EXPECTED_SAMPLES_PER_WINDOW = 100


@dataclass(frozen=True)
class DatasetSplit:
    """One fully loaded dataset-v1 split."""

    name: str
    session: str
    windows: np.ndarray
    timestamps_ms: np.ndarray
    labels: np.ndarray
    csv_paths: tuple[Path, ...]
    metadata_paths: tuple[Path, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_manifest_path(manifest_path: str | Path | None) -> Path:
    if manifest_path is None:
        return project_root() / MANIFEST_RELATIVE_PATH

    path = Path(manifest_path)

    if not path.is_absolute():
        path = project_root() / path

    return path


def load_manifest(manifest_path: str | Path | None = None) -> dict:
    """Load and validate the top-level dataset-v1 split manifest."""

    path = _resolve_manifest_path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected dataset version {DATASET_VERSION}, "
            f"got {manifest.get('dataset_version')!r}."
        )

    if manifest.get("split_strategy") != "session-based":
        raise ValueError(
            "dataset-v1 loader requires split_strategy='session-based'."
        )

    splits = manifest.get("splits")

    if not isinstance(splits, dict):
        raise ValueError("Manifest is missing a valid 'splits' mapping.")

    required_splits = ("train", "validation", "test")
    missing = [name for name in required_splits if name not in splits]

    if missing:
        raise ValueError(f"Manifest is missing splits: {missing}")

    return manifest


def _resolve_data_path(relative_path: str, root: Path) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError(f"Manifest paths must be project-relative: {path}")

    if "_rejected" in path.parts:
        raise ValueError(f"Rejected capture cannot be loaded: {path}")

    return root / path


def _load_csv_window(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {csv_path}") from exc

        if header != CSV_COLUMNS:
            raise ValueError(
                f"Unexpected CSV schema in {csv_path}: {header}"
            )

        timestamps: list[int] = []
        sensor_rows: list[list[float]] = []

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(
                    f"Malformed row in {csv_path} "
                    f"at line {line_number}: {row}"
                )

            try:
                timestamps.append(int(row[0]))
                sensor_rows.append(
                    [float(value) for value in row[1:]]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in {csv_path} "
                    f"at line {line_number}."
                ) from exc

    if len(sensor_rows) != EXPECTED_SAMPLES_PER_WINDOW:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES_PER_WINDOW} samples "
            f"in {csv_path}, got {len(sensor_rows)}."
        )

    timestamps_array = np.asarray(timestamps, dtype=np.int64)
    window = np.asarray(sensor_rows, dtype=np.float32)

    expected_shape = (
        EXPECTED_SAMPLES_PER_WINDOW,
        len(SENSOR_COLUMNS),
    )

    if window.shape != expected_shape:
        raise ValueError(
            f"Unexpected window shape in {csv_path}: {window.shape}"
        )

    if not np.isfinite(window).all():
        raise ValueError(f"Non-finite sensor value in {csv_path}.")

    return timestamps_array, window


def _validate_metadata(
    metadata: dict,
    *,
    metadata_path: Path,
    gesture: str,
    session: str,
    csv_relative_path: str,
) -> None:
    expected_fields = {
        "dataset_version": DATASET_VERSION,
        "gesture": gesture,
        "session": session,
        "sample_rate_hz": 100,
        "window_ms": 1000,
        "sample_count": EXPECTED_SAMPLES_PER_WINDOW,
    }

    for field, expected in expected_fields.items():
        actual = metadata.get(field)

        if actual != expected:
            raise ValueError(
                f"Metadata mismatch in {metadata_path}: "
                f"{field}={actual!r}, expected {expected!r}."
            )

    source_csv = metadata.get("source_csv")

    if source_csv != csv_relative_path:
        raise ValueError(
            f"Metadata source_csv mismatch in {metadata_path}: "
            f"{source_csv!r} != {csv_relative_path!r}."
        )


def load_split(
    split_name: str,
    manifest_path: str | Path | None = None,
) -> DatasetSplit:
    """Load one dataset-v1 split from the frozen session-based manifest."""

    manifest = load_manifest(manifest_path)
    splits = manifest["splits"]

    if split_name not in splits:
        raise ValueError(
            f"Unknown split {split_name!r}. "
            f"Expected one of {tuple(splits)}."
        )

    split_info = splits[split_name]
    session = split_info.get("session")
    samples = split_info.get("samples")

    if not isinstance(session, str) or not isinstance(samples, list):
        raise ValueError(
            f"Invalid manifest entry for split {split_name!r}."
        )

    root = project_root()

    windows: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    labels: list[int] = []
    csv_paths: list[Path] = []
    metadata_paths: list[Path] = []

    for sample in samples:
        gesture = sample.get("gesture")

        if gesture not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown gesture in manifest: {gesture!r}"
            )

        csv_relative = sample.get("csv")
        metadata_relative = sample.get("metadata")

        if (
            not isinstance(csv_relative, str)
            or not isinstance(metadata_relative, str)
        ):
            raise ValueError(
                "Sample entry must contain string csv/metadata paths."
            )

        csv_path = _resolve_data_path(csv_relative, root)
        metadata_path = _resolve_data_path(
            metadata_relative,
            root,
        )

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing capture CSV: {csv_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing capture metadata: {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        _validate_metadata(
            metadata,
            metadata_path=metadata_path,
            gesture=gesture,
            session=session,
            csv_relative_path=csv_relative,
        )

        sample_timestamps, window = _load_csv_window(csv_path)

        timestamps.append(sample_timestamps)
        windows.append(window)
        labels.append(CLASS_TO_ID[gesture])
        csv_paths.append(csv_path)
        metadata_paths.append(metadata_path)

    expected_count = split_info.get("count")

    if len(windows) != expected_count:
        raise ValueError(
            f"Split {split_name!r} expected {expected_count} samples, "
            f"loaded {len(windows)}."
        )

    windows_array = np.stack(windows).astype(
        np.float32,
        copy=False,
    )
    timestamps_array = np.stack(timestamps).astype(
        np.int64,
        copy=False,
    )
    labels_array = np.asarray(labels, dtype=np.int64)

    return DatasetSplit(
        name=split_name,
        session=session,
        windows=windows_array,
        timestamps_ms=timestamps_array,
        labels=labels_array,
        csv_paths=tuple(csv_paths),
        metadata_paths=tuple(metadata_paths),
    )


__all__ = [
    "CLASS_TO_ID",
    "CSV_COLUMNS",
    "DATASET_VERSION",
    "DatasetSplit",
    "GESTURES",
    "ID_TO_CLASS",
    "SENSOR_COLUMNS",
    "load_manifest",
    "load_split",
]