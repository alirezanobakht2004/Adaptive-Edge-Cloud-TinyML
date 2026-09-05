from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ml.models.base_model import (
    BLOCK_1_UNITS,
    BLOCK_2_UNITS,
    BLOCK_3_UNITS,
    CLASS_COUNT,
    INPUT_FEATURES,
)


UNCERTAINTY_MODEL_VERSION = "gesture-model-v1.1.0"

MC_DROPOUT_PASSES = 5
MC_DROPOUT_RATE = 0.2


@dataclass(frozen=True)
class MCDropoutVariationDiagnostics:
    sample_count: int
    varying_sample_count: int
    top1_changed_sample_count: int
    max_probability_range: float
    tolerance: float


def build_mc_dropout_model(
    normalization_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    """Build the Phase-5 dropout-enabled local classifier candidate.

    The deterministic path remains:

        Input(10)
        -> train-fitted normalization
        -> Dense(64) + ReLU
        -> Dense(48) + ReLU
        -> Dense(32) + ReLU
        -> Edge Head(5)

    A Dropout(0.2) layer is inserted between Block 3 and the Edge Head.

    During ordinary inference (`training=False`) Dropout is disabled.
    During MC-Dropout evaluation (`training=True`) the same trained model
    produces stochastic predictions.

    ESP32 deployment must not assume that a normal TFLite Dropout layer
    remains stochastic; explicit masking will be handled later if required.
    """

    inputs = tf.keras.Input(
        shape=(INPUT_FEATURES,),
        dtype=tf.float32,
        name="features_v1",
    )

    x = normalization_layer(inputs)

    x = tf.keras.layers.Dense(
        BLOCK_1_UNITS,
        activation="relu",
        name="block1",
    )(x)

    x = tf.keras.layers.Dense(
        BLOCK_2_UNITS,
        activation="relu",
        name="block2",
    )(x)

    x = tf.keras.layers.Dense(
        BLOCK_3_UNITS,
        activation="relu",
        name="block3",
    )(x)

    x = tf.keras.layers.Dropout(
        MC_DROPOUT_RATE,
        name="mc_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        CLASS_COUNT,
        activation="softmax",
        name="edge_head",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_mc_dropout_model",
    )


def _validate_feature_matrix(
    features: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        features,
        dtype=np.float32,
    )

    expected_shape_suffix = (
        INPUT_FEATURES,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1:]
        != expected_shape_suffix
    ):
        raise ValueError(
            "Expected MC-Dropout input with shape "
            f"(N, {INPUT_FEATURES}), "
            f"got {matrix.shape}."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "MC-Dropout input cannot be empty."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "MC-Dropout input contains NaN "
            "or infinite values."
        )

    return matrix


def _validate_probability_tensor(
    probabilities: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    tensor = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    expected_shape = (
        MC_DROPOUT_PASSES,
        sample_count,
        CLASS_COUNT,
    )

    if tensor.shape != expected_shape:
        raise ValueError(
            "Unexpected MC-Dropout probability "
            f"shape {tensor.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(tensor).all():
        raise ValueError(
            "MC-Dropout probabilities contain "
            "NaN or infinite values."
        )

    probability_tolerance = 1e-5

    if (
        np.any(
            tensor < -probability_tolerance
        )
        or np.any(
            tensor
            > 1.0 + probability_tolerance
        )
    ):
        raise ValueError(
            "MC-Dropout output contains values "
            "outside the probability range."
        )

    row_sums = np.sum(
        tensor,
        axis=2,
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=probability_tolerance,
    ):
        raise ValueError(
            "MC-Dropout output rows do not sum "
            "to 1 within tolerance."
        )

    return tensor


def mc_dropout_predict(
    model: tf.keras.Model,
    features: np.ndarray,
    *,
    passes: int = MC_DROPOUT_PASSES,
) -> np.ndarray:
    """Run exactly five stochastic forward passes.

    Returns:
        Float32 tensor with shape:

            (5, N, CLASS_COUNT)

    Dropout is explicitly enabled by calling the model with
    `training=True` on every pass.
    """

    if passes != MC_DROPOUT_PASSES:
        raise ValueError(
            "Phase-5 contract requires exactly "
            f"{MC_DROPOUT_PASSES} stochastic passes; "
            f"got {passes}."
        )

    matrix = _validate_feature_matrix(
        features
    )

    pass_outputs: list[np.ndarray] = []

    for _ in range(
        MC_DROPOUT_PASSES
    ):
        output = model(
            matrix,
            training=True,
        )

        pass_outputs.append(
            np.asarray(
                output.numpy(),
                dtype=np.float32,
            )
        )

    probabilities = np.stack(
        pass_outputs,
        axis=0,
    )

    return _validate_probability_tensor(
        probabilities,
        sample_count=matrix.shape[0],
    )


def mc_dropout_variation_diagnostics(
    probabilities: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> MCDropoutVariationDiagnostics:
    """Measure whether stochastic passes actually differ.

    This deliberately does NOT compute the project uncertainty score yet.
    Entropy and variance belong to the next M6 sub-step.
    """

    tensor = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if tensor.ndim != 3:
        raise ValueError(
            "Expected probability tensor with "
            "shape (passes, samples, classes)."
        )

    if tensor.shape[0] != MC_DROPOUT_PASSES:
        raise ValueError(
            "Variation diagnostics require exactly "
            f"{MC_DROPOUT_PASSES} passes."
        )

    if tensor.shape[2] != CLASS_COUNT:
        raise ValueError(
            "Variation diagnostics expected "
            f"{CLASS_COUNT} classes, "
            f"got {tensor.shape[2]}."
        )

    if tensor.shape[1] == 0:
        raise ValueError(
            "Variation diagnostics require at least "
            "one sample."
        )

    if not np.isfinite(tensor).all():
        raise ValueError(
            "Variation diagnostics received "
            "non-finite probabilities."
        )

    if tolerance < 0.0:
        raise ValueError(
            "Variation tolerance cannot be negative."
        )

    probability_range = (
        np.max(
            tensor,
            axis=0,
        )
        - np.min(
            tensor,
            axis=0,
        )
    )

    per_sample_max_range = np.max(
        probability_range,
        axis=1,
    )

    varying_samples = (
        per_sample_max_range
        > tolerance
    )

    top1 = np.argmax(
        tensor,
        axis=2,
    )

    top1_changed = np.any(
        top1
        != top1[0:1, :],
        axis=0,
    )

    return MCDropoutVariationDiagnostics(
        sample_count=int(
            tensor.shape[1]
        ),
        varying_sample_count=int(
            np.count_nonzero(
                varying_samples
            )
        ),
        top1_changed_sample_count=int(
            np.count_nonzero(
                top1_changed
            )
        ),
        max_probability_range=float(
            np.max(
                probability_range
            )
        ),
        tolerance=float(
            tolerance
        ),
    )


__all__ = [
    "MC_DROPOUT_PASSES",
    "MC_DROPOUT_RATE",
    "MCDropoutVariationDiagnostics",
    "UNCERTAINTY_MODEL_VERSION",
    "build_mc_dropout_model",
    "mc_dropout_predict",
    "mc_dropout_variation_diagnostics",
]
