from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.models.base_model import CLASS_COUNT
from ml.uncertainty.mc_dropout import MC_DROPOUT_PASSES


@dataclass(frozen=True)
class MCDropoutUncertaintyMetrics:
    """Per-sample metrics computed from frozen MC-Dropout passes."""

    mean_probabilities: np.ndarray
    predictive_entropy: np.ndarray
    normalized_predictive_entropy: np.ndarray
    class_probability_variance: np.ndarray
    mean_class_variance: np.ndarray
    max_class_variance: np.ndarray
    max_mean_confidence: np.ndarray
    predicted_class: np.ndarray


def _validate_probability_tensor(
    probabilities: np.ndarray,
) -> np.ndarray:
    tensor = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    expected_prefix = (
        MC_DROPOUT_PASSES,
    )

    if (
        tensor.ndim != 3
        or tensor.shape[0:1]
        != expected_prefix
        or tensor.shape[2] != CLASS_COUNT
    ):
        raise ValueError(
            "Expected probability tensor with shape "
            f"({MC_DROPOUT_PASSES}, N, {CLASS_COUNT}); "
            f"got {tensor.shape}."
        )

    if tensor.shape[1] == 0:
        raise ValueError(
            "Probability tensor cannot contain zero samples."
        )

    if not np.isfinite(tensor).all():
        raise ValueError(
            "Probability tensor contains NaN or infinite values."
        )

    tolerance = 1e-5

    if (
        np.any(tensor < -tolerance)
        or np.any(tensor > 1.0 + tolerance)
    ):
        raise ValueError(
            "Probability tensor contains values outside [0, 1]."
        )

    row_sums = np.sum(
        tensor,
        axis=2,
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Each pass/sample probability row must sum to 1."
        )

    return tensor


def compute_mc_dropout_uncertainty_metrics(
    probabilities: np.ndarray,
) -> MCDropoutUncertaintyMetrics:
    """Compute Phase-5 entropy/variance outputs.

    Definitions:

    mean_probabilities:
        Mean class probability across the five stochastic passes.

    predictive_entropy:
        H[p_bar] = -sum_c p_bar(c) * ln(p_bar(c))
        measured in nats.

    normalized_predictive_entropy:
        predictive_entropy / ln(CLASS_COUNT), therefore nominally [0, 1].

    class_probability_variance:
        Population variance of each class probability across the five passes.

    mean_class_variance:
        Mean of the per-class variances for each sample.

    max_class_variance:
        Maximum per-class variance for each sample.

    max_mean_confidence:
        Maximum value of the mean class-probability vector.

    predicted_class:
        Argmax of the mean class-probability vector.

    This function deliberately does NOT choose an offloading threshold and
    does NOT define a final learned-policy feature contract yet.
    """

    tensor = _validate_probability_tensor(
        probabilities
    )

    mean_probabilities = np.mean(
        tensor,
        axis=0,
    )

    entropy_terms = np.zeros_like(
        mean_probabilities,
        dtype=np.float64,
    )

    positive = (
        mean_probabilities > 0.0
    )

    entropy_terms[positive] = (
        mean_probabilities[positive]
        * np.log(
            mean_probabilities[positive]
        )
    )

    predictive_entropy = -np.sum(
        entropy_terms,
        axis=1,
    )

    max_entropy = float(
        np.log(CLASS_COUNT)
    )

    normalized_predictive_entropy = (
        predictive_entropy
        / max_entropy
    )

    class_probability_variance = np.var(
        tensor,
        axis=0,
        ddof=0,
    )

    mean_class_variance = np.mean(
        class_probability_variance,
        axis=1,
    )

    max_class_variance = np.max(
        class_probability_variance,
        axis=1,
    )

    max_mean_confidence = np.max(
        mean_probabilities,
        axis=1,
    )

    predicted_class = np.argmax(
        mean_probabilities,
        axis=1,
    ).astype(
        np.int64
    )

    return MCDropoutUncertaintyMetrics(
        mean_probabilities=(
            mean_probabilities.astype(
                np.float32
            )
        ),
        predictive_entropy=(
            predictive_entropy.astype(
                np.float32
            )
        ),
        normalized_predictive_entropy=(
            normalized_predictive_entropy.astype(
                np.float32
            )
        ),
        class_probability_variance=(
            class_probability_variance.astype(
                np.float32
            )
        ),
        mean_class_variance=(
            mean_class_variance.astype(
                np.float32
            )
        ),
        max_class_variance=(
            max_class_variance.astype(
                np.float32
            )
        ),
        max_mean_confidence=(
            max_mean_confidence.astype(
                np.float32
            )
        ),
        predicted_class=predicted_class,
    )


__all__ = [
    "MCDropoutUncertaintyMetrics",
    "compute_mc_dropout_uncertainty_metrics",
]
