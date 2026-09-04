from __future__ import annotations

import tensorflow as tf


MODEL_VERSION = "gesture-model-v1.0.0"

INPUT_FEATURES = 10
CLASS_COUNT = 5

BLOCK_1_UNITS = 64
BLOCK_2_UNITS = 48
BLOCK_3_UNITS = 32


def build_base_model(
    normalization_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    """Build the Phase-3 local gesture classifier.

    Architecture:
        Input(10)
        -> train-fitted normalization
        -> Dense(64) + ReLU
        -> Dense(48) + ReLU
        -> Dense(32) + ReLU
        -> Edge Head(5)
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

    outputs = tf.keras.layers.Dense(
        CLASS_COUNT,
        activation="softmax",
        name="edge_head",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_base_model",
    )


__all__ = [
    "BLOCK_1_UNITS",
    "BLOCK_2_UNITS",
    "BLOCK_3_UNITS",
    "CLASS_COUNT",
    "INPUT_FEATURES",
    "MODEL_VERSION",
    "build_base_model",
]