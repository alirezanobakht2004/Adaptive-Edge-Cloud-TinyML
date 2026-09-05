from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ml.dataset.loader import GESTURES


AMBIGUITY_PROBE_VERSION = "ambiguity-probe-v1"
PROBES_PER_CLASS_PAIR = 5
MIDPOINT_ALPHA = 0.5


@dataclass(frozen=True)
class AmbiguousProbeBatch:
    features: np.ndarray
    left_class_id: np.ndarray
    right_class_id: np.ndarray
    left_sample_index: np.ndarray
    right_sample_index: np.ndarray
    normalized_distance: np.ndarray
    alpha: np.ndarray


def _validate_feature_matrix(
    features: np.ndarray,
    *,
    expected_feature_count: int,
) -> np.ndarray:
    matrix = np.asarray(
        features,
        dtype=np.float32,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1]
        != expected_feature_count
    ):
        raise ValueError(
            "Expected feature matrix with shape "
            f"(N, {expected_feature_count}), "
            f"got {matrix.shape}."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Feature matrix cannot be empty."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return matrix


def _validate_labels(
    labels: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    vector = np.asarray(
        labels,
        dtype=np.int64,
    )

    if vector.shape != (sample_count,):
        raise ValueError(
            "Expected labels with shape "
            f"({sample_count},), got {vector.shape}."
        )

    if np.any(vector < 0) or np.any(
        vector >= len(GESTURES)
    ):
        raise ValueError(
            "Labels contain an invalid class id."
        )

    return vector


def normalized_feature_matrix(
    features: np.ndarray,
    *,
    normalization_mean: np.ndarray,
    normalization_variance: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        features,
        dtype=np.float64,
    )

    mean = np.asarray(
        normalization_mean,
        dtype=np.float64,
    ).reshape(-1)

    variance = np.asarray(
        normalization_variance,
        dtype=np.float64,
    ).reshape(-1)

    if mean.shape != (matrix.shape[1],):
        raise ValueError(
            "Normalization mean shape does not match feature count."
        )

    if variance.shape != (matrix.shape[1],):
        raise ValueError(
            "Normalization variance shape does not match feature count."
        )

    if np.any(variance < 0.0):
        raise ValueError(
            "Normalization variance cannot be negative."
        )

    scale = np.sqrt(
        np.maximum(
            variance,
            1e-12,
        )
    )

    normalized = (
        matrix - mean
    ) / scale

    if not np.isfinite(normalized).all():
        raise ValueError(
            "Normalized feature matrix contains non-finite values."
        )

    return normalized.astype(
        np.float32
    )


def class_pairs() -> tuple[
    tuple[int, int],
    ...,
]:
    return tuple(
        combinations(
            range(len(GESTURES)),
            2,
        )
    )


def build_ambiguous_midpoint_probes(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    normalization_mean: np.ndarray,
    normalization_variance: np.ndarray,
    probes_per_class_pair: int = PROBES_PER_CLASS_PAIR,
    alpha: float = MIDPOINT_ALPHA,
) -> AmbiguousProbeBatch:
    """Build controlled cross-class ambiguity probes.

    For every unordered pair of the five gesture classes:
      1. normalize the real VALIDATION feature vectors using the
         candidate model's TRAIN-fitted normalization statistics,
      2. find the closest cross-class sample pairs,
      3. interpolate their original features at alpha=0.5.

    The result is a controlled *feature-space* ambiguity probe set.
    It is NOT a new training dataset and it is NOT claimed to be a
    physically captured gesture dataset.
    """

    if probes_per_class_pair <= 0:
        raise ValueError(
            "probes_per_class_pair must be positive."
        )

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "alpha must be strictly between 0 and 1."
        )

    matrix = _validate_feature_matrix(
        features,
        expected_feature_count=(
            np.asarray(features).shape[1]
            if np.asarray(features).ndim == 2
            else 10
        ),
    )

    label_vector = _validate_labels(
        labels,
        sample_count=matrix.shape[0],
    )

    normalized = normalized_feature_matrix(
        matrix,
        normalization_mean=normalization_mean,
        normalization_variance=normalization_variance,
    )

    probe_features: list[np.ndarray] = []
    left_classes: list[int] = []
    right_classes: list[int] = []
    left_indices: list[int] = []
    right_indices: list[int] = []
    distances: list[float] = []
    alphas: list[float] = []

    for left_class, right_class in class_pairs():
        left = np.flatnonzero(
            label_vector == left_class
        )

        right = np.flatnonzero(
            label_vector == right_class
        )

        if left.size == 0 or right.size == 0:
            raise ValueError(
                "Every gesture class must be present "
                "in the ambiguity source split."
            )

        delta = (
            normalized[left, None, :]
            - normalized[None, right, :]
        )

        distance_matrix = np.linalg.norm(
            delta,
            axis=2,
        )

        flat_order = np.argsort(
            distance_matrix,
            axis=None,
            kind="stable",
        )

        if flat_order.size < probes_per_class_pair:
            raise ValueError(
                "Not enough cross-class sample pairs "
                "for the requested probe count."
            )

        selected = flat_order[
            :probes_per_class_pair
        ]

        for flat_index in selected:
            left_local, right_local = (
                np.unravel_index(
                    int(flat_index),
                    distance_matrix.shape,
                )
            )

            left_index = int(
                left[left_local]
            )

            right_index = int(
                right[right_local]
            )

            midpoint = (
                (1.0 - alpha)
                * matrix[left_index]
                + alpha
                * matrix[right_index]
            ).astype(
                np.float32
            )

            probe_features.append(
                midpoint
            )

            left_classes.append(
                left_class
            )

            right_classes.append(
                right_class
            )

            left_indices.append(
                left_index
            )

            right_indices.append(
                right_index
            )

            distances.append(
                float(
                    distance_matrix[
                        left_local,
                        right_local,
                    ]
                )
            )

            alphas.append(
                float(alpha)
            )

    features_array = np.stack(
        probe_features
    ).astype(
        np.float32,
        copy=False,
    )

    return AmbiguousProbeBatch(
        features=features_array,
        left_class_id=np.asarray(
            left_classes,
            dtype=np.int64,
        ),
        right_class_id=np.asarray(
            right_classes,
            dtype=np.int64,
        ),
        left_sample_index=np.asarray(
            left_indices,
            dtype=np.int64,
        ),
        right_sample_index=np.asarray(
            right_indices,
            dtype=np.int64,
        ),
        normalized_distance=np.asarray(
            distances,
            dtype=np.float32,
        ),
        alpha=np.asarray(
            alphas,
            dtype=np.float32,
        ),
    )


__all__ = [
    "AMBIGUITY_PROBE_VERSION",
    "AmbiguousProbeBatch",
    "MIDPOINT_ALPHA",
    "PROBES_PER_CLASS_PAIR",
    "build_ambiguous_midpoint_probes",
    "class_pairs",
    "normalized_feature_matrix",
]
