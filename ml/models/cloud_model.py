"""Phase 6 fixed Split-3 cloud-tail model."""

from __future__ import annotations

import tensorflow as tf


CLOUD_TAIL_VERSION = "gesture-cloud-tail-v1.0.0"

SOURCE_EDGE_MODEL_VERSION = "gesture-model-v1.1.0"

INPUT_EMBEDDING_DIM = 32
CLOUD_BLOCK_4_UNITS = 64
CLOUD_BLOCK_5_UNITS = 32
CLASS_COUNT = 5


def build_cloud_tail() -> tf.keras.Model:
    """Build B4 -> B5 -> Cloud Head for a 32-D B3 embedding."""

    inputs = tf.keras.Input(
        shape=(INPUT_EMBEDDING_DIM,),
        dtype=tf.float32,
        name="split3_embedding",
    )

    x = tf.keras.layers.Dense(
        CLOUD_BLOCK_4_UNITS,
        activation="relu",
        name="block4",
    )(inputs)

    x = tf.keras.layers.Dense(
        CLOUD_BLOCK_5_UNITS,
        activation="relu",
        name="block5",
    )(x)

    outputs = tf.keras.layers.Dense(
        CLASS_COUNT,
        activation="softmax",
        name="cloud_head",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_cloud_tail_split3",
    )


__all__ = [
    "CLASS_COUNT",
    "CLOUD_BLOCK_4_UNITS",
    "CLOUD_BLOCK_5_UNITS",
    "CLOUD_TAIL_VERSION",
    "INPUT_EMBEDDING_DIM",
    "SOURCE_EDGE_MODEL_VERSION",
    "build_cloud_tail",
]
