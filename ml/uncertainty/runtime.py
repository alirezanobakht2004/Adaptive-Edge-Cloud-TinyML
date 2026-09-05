from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ml.models.base_model import (
    CLASS_COUNT,
    INPUT_FEATURES,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    mc_dropout_predict,
)
from ml.uncertainty.metrics import (
    compute_mc_dropout_uncertainty_metrics,
)


UNCERTAINTY_SCORE_NAME = (
    "normalized_predictive_entropy"
)


@dataclass(frozen=True)
class UncertaintyInferenceResult:
    """Single-window Phase-5 uncertainty result.

    `uncertainty_score` is the canonical scalar uncertainty value for
    the Phase-5 runtime contract:

        H(mean_probability) / ln(CLASS_COUNT)

    where H is predictive entropy using natural logarithms.

    The score is nominally in [0, 1]:
      0 -> concentrated mean predictive distribution
      1 -> uniform mean predictive distribution

    This result does NOT make a LOCAL/OFFLOAD decision. Thresholds and
    the learned policy belong to later phases.
    """

    predicted_class: int
    confidence: float
    uncertainty_score: float
    predictive_entropy_nats: float
    mean_class_variance: float
    max_class_variance: float
    mean_probabilities: np.ndarray


def _validate_single_feature_vector(
    features: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        features,
        dtype=np.float32,
    )

    if vector.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Per-inference uncertainty expects one "
            f"features-v1 vector with shape ({INPUT_FEATURES},), "
            f"got {vector.shape}."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Per-inference feature vector contains "
            "NaN or infinite values."
        )

    return vector


def infer_with_uncertainty(
    model: tf.keras.Model,
    features: np.ndarray,
) -> UncertaintyInferenceResult:
    """Run one logical inference with exactly five MC-Dropout passes.

    Input:
        one raw features-v1 vector with shape (10,)

    Runtime:
        exactly 5 stochastic model calls via `training=True`

    Output:
        class prediction + confidence + scalar uncertainty score +
        variance diagnostics for this one inference.

    Canonical Phase-5 scalar:
        normalized predictive entropy

        score = -sum_c(mean_p[c] * ln(mean_p[c])) / ln(CLASS_COUNT)

    No offload threshold is applied here.
    """

    vector = _validate_single_feature_vector(
        features
    )

    probabilities = mc_dropout_predict(
        model,
        vector[None, :],
        passes=MC_DROPOUT_PASSES,
    )

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    if metrics.mean_probabilities.shape != (
        1,
        CLASS_COUNT,
    ):
        raise RuntimeError(
            "Unexpected single-inference mean-probability shape."
        )

    uncertainty_score = float(
        metrics.normalized_predictive_entropy[0]
    )

    if (
        not np.isfinite(
            uncertainty_score
        )
        or uncertainty_score < -1e-6
        or uncertainty_score > 1.0 + 1e-6
    ):
        raise RuntimeError(
            "Normalized predictive entropy is outside "
            "the expected [0, 1] range."
        )

    confidence = float(
        metrics.max_mean_confidence[0]
    )

    return UncertaintyInferenceResult(
        predicted_class=int(
            metrics.predicted_class[0]
        ),
        confidence=confidence,
        uncertainty_score=(
            uncertainty_score
        ),
        predictive_entropy_nats=float(
            metrics.predictive_entropy[0]
        ),
        mean_class_variance=float(
            metrics.mean_class_variance[0]
        ),
        max_class_variance=float(
            metrics.max_class_variance[0]
        ),
        mean_probabilities=(
            metrics.mean_probabilities[0]
            .copy()
        ),
    )


__all__ = [
    "UNCERTAINTY_SCORE_NAME",
    "UncertaintyInferenceResult",
    "infer_with_uncertainty",
]
