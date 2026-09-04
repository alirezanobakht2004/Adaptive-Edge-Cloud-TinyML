from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.dataset.loader import load_split

from .features_v1 import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_VERSION,
    SENSOR_CHANNELS,
    WINDOW_SAMPLES,
    extract_features_v1,
)


@dataclass(frozen=True)
class FeatureSplit:
    """One dataset split represented by features-v1."""

    name: str
    session: str
    feature_version: str
    features: np.ndarray
    labels: np.ndarray
    csv_paths: tuple[Path, ...]
    metadata_paths: tuple[Path, ...]


def extract_features(
    window: np.ndarray,
    version: str = FEATURE_VERSION,
) -> np.ndarray:
    """Extract one feature vector using the requested feature version."""

    if version != FEATURE_VERSION:
        raise ValueError(
            f"Unsupported feature version: {version!r}"
        )

    return extract_features_v1(window)


def extract_feature_matrix(
    windows: np.ndarray,
    version: str = FEATURE_VERSION,
) -> np.ndarray:
    """Convert a batch of raw IMU windows into a feature matrix."""

    x = np.asarray(windows)

    expected_window_shape = (
        WINDOW_SAMPLES,
        SENSOR_CHANNELS,
    )

    if x.ndim != 3 or x.shape[1:] != expected_window_shape:
        raise ValueError(
            "Expected windows with shape "
            f"(N, {WINDOW_SAMPLES}, {SENSOR_CHANNELS}), "
            f"got {x.shape}."
        )

    if x.shape[0] == 0:
        raise ValueError("Feature extraction batch cannot be empty.")

    if not np.isfinite(x).all():
        raise ValueError(
            "Raw window batch contains NaN or infinite values."
        )

    matrix = np.empty(
        (x.shape[0], FEATURE_COUNT),
        dtype=np.float32,
    )

    for index, window in enumerate(x):
        matrix[index] = extract_features(
            window,
            version=version,
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return matrix


def load_feature_split(
    split_name: str,
    manifest_path: str | Path | None = None,
    version: str = FEATURE_VERSION,
) -> FeatureSplit:
    """Load one raw split and convert it to features-v1."""

    raw_split = load_split(
        split_name,
        manifest_path=manifest_path,
    )

    matrix = extract_feature_matrix(
        raw_split.windows,
        version=version,
    )

    return FeatureSplit(
        name=raw_split.name,
        session=raw_split.session,
        feature_version=version,
        features=matrix,
        labels=raw_split.labels.copy(),
        csv_paths=raw_split.csv_paths,
        metadata_paths=raw_split.metadata_paths,
    )


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeatureSplit",
    "extract_feature_matrix",
    "extract_features",
    "load_feature_split",
]