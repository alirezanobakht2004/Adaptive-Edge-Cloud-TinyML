from __future__ import annotations

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
    can produce stochastic predictions.

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


__all__ = [
    "MC_DROPOUT_PASSES",
    "MC_DROPOUT_RATE",
    "UNCERTAINTY_MODEL_VERSION",
    "build_mc_dropout_model",
]
