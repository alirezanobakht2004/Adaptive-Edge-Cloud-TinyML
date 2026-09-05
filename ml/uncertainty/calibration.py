from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_CALIBRATION_BINS = 10
NLL_EPSILON = 1e-12


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    absolute_gap: float


@dataclass(frozen=True)
class CalibrationMetrics:
    sample_count: int
    bin_count: int
    accuracy: float
    mean_confidence: float
    signed_confidence_gap: float
    absolute_confidence_gap: float
    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    multiclass_brier_score: float
    reliability_bins: tuple[ReliabilityBin, ...]


def _validate_probabilities_and_labels(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    targets = np.asarray(
        labels,
        dtype=np.int64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape (N, C)."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Calibration input cannot be empty."
        )

    if matrix.shape[1] < 2:
        raise ValueError(
            "Calibration requires at least two classes."
        )

    if targets.shape != (
        matrix.shape[0],
    ):
        raise ValueError(
            "Labels must have shape (N,) matching probabilities."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Probabilities contain NaN or infinite values."
        )

    tolerance = 1e-5

    if (
        np.any(matrix < -tolerance)
        or np.any(matrix > 1.0 + tolerance)
    ):
        raise ValueError(
            "Probabilities contain values outside [0, 1]."
        )

    if not np.allclose(
        np.sum(matrix, axis=1),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Each probability row must sum to 1."
        )

    if (
        np.any(targets < 0)
        or np.any(targets >= matrix.shape[1])
    ):
        raise ValueError(
            "Labels contain an invalid class id."
        )

    return matrix, targets


def compute_reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[ReliabilityBin, ...]:
    """Compute equal-width top-label reliability bins on [0, 1]."""

    matrix, targets = (
        _validate_probabilities_and_labels(
            probabilities,
            labels,
        )
    )

    if bin_count <= 0:
        raise ValueError(
            "bin_count must be positive."
        )

    predictions = np.argmax(
        matrix,
        axis=1,
    )

    confidence = np.max(
        matrix,
        axis=1,
    )

    correct = (
        predictions == targets
    )

    edges = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
        dtype=np.float64,
    )

    result: list[ReliabilityBin] = []

    for index in range(bin_count):
        lower = float(
            edges[index]
        )

        upper = float(
            edges[index + 1]
        )

        if index == bin_count - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(
            np.count_nonzero(mask)
        )

        if count == 0:
            mean_confidence = 0.0
            accuracy = 0.0
            gap = 0.0
        else:
            mean_confidence = float(
                np.mean(
                    confidence[mask]
                )
            )

            accuracy = float(
                np.mean(
                    correct[mask]
                )
            )

            gap = abs(
                mean_confidence
                - accuracy
            )

        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )

    return tuple(result)


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationMetrics:
    """Evaluate descriptive calibration metrics.

    This function evaluates the frozen predictive distribution only.
    It does NOT fit temperature scaling or any other calibration transform.

    ECE uses equal-width top-label confidence bins.
    Multiclass Brier score is mean(sum_c((p_c - y_c)^2)).
    NLL uses natural logarithms.
    """

    matrix, targets = (
        _validate_probabilities_and_labels(
            probabilities,
            labels,
        )
    )

    bins = compute_reliability_bins(
        matrix,
        targets,
        bin_count=bin_count,
    )

    predictions = np.argmax(
        matrix,
        axis=1,
    )

    confidence = np.max(
        matrix,
        axis=1,
    )

    correct = (
        predictions == targets
    )

    accuracy = float(
        np.mean(correct)
    )

    mean_confidence = float(
        np.mean(confidence)
    )

    signed_confidence_gap = (
        mean_confidence
        - accuracy
    )

    sample_count = (
        matrix.shape[0]
    )

    expected_calibration_error = float(
        sum(
            (
                reliability_bin.count
                / sample_count
            )
            * reliability_bin.absolute_gap
            for reliability_bin
            in bins
        )
    )

    non_empty_gaps = [
        reliability_bin.absolute_gap
        for reliability_bin in bins
        if reliability_bin.count > 0
    ]

    maximum_calibration_error = float(
        max(non_empty_gaps)
        if non_empty_gaps
        else 0.0
    )

    true_probability = matrix[
        np.arange(sample_count),
        targets,
    ]

    negative_log_likelihood = float(
        -np.mean(
            np.log(
                np.clip(
                    true_probability,
                    NLL_EPSILON,
                    1.0,
                )
            )
        )
    )

    one_hot = np.zeros_like(
        matrix
    )

    one_hot[
        np.arange(sample_count),
        targets,
    ] = 1.0

    multiclass_brier_score = float(
        np.mean(
            np.sum(
                (
                    matrix
                    - one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )

    return CalibrationMetrics(
        sample_count=int(
            sample_count
        ),
        bin_count=int(
            bin_count
        ),
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        signed_confidence_gap=float(
            signed_confidence_gap
        ),
        absolute_confidence_gap=float(
            abs(
                signed_confidence_gap
            )
        ),
        expected_calibration_error=(
            expected_calibration_error
        ),
        maximum_calibration_error=(
            maximum_calibration_error
        ),
        negative_log_likelihood=(
            negative_log_likelihood
        ),
        multiclass_brier_score=(
            multiclass_brier_score
        ),
        reliability_bins=bins,
    )


__all__ = [
    "CalibrationMetrics",
    "DEFAULT_CALIBRATION_BINS",
    "NLL_EPSILON",
    "ReliabilityBin",
    "compute_reliability_bins",
    "evaluate_calibration",
]
