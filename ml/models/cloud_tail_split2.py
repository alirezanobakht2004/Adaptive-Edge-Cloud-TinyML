"""Phase 7 Split2 continuation; independent of the frozen Split3 tail."""

from pathlib import Path

import tensorflow as tf

SPLIT_ID = 2
INPUT_EMBEDDING_DIM = 48
CLASS_COUNT = 5
CLOUD_TAIL_VERSION = "gesture-cloud-tail-split2-v1.0.0"
SOURCE_MODEL_VERSION = "gesture-model-v1.1.0"
MODEL_DIR = (Path(__file__).resolve().parents[2] / "data/processed"
             / "dataset-v1/features-v1/models" / CLOUD_TAIL_VERSION)
MODEL_PATH = MODEL_DIR / f"{CLOUD_TAIL_VERSION}.keras"
MODEL_METADATA = {
    "split_id": SPLIT_ID,
    "input_embedding_dim": INPUT_EMBEDDING_DIM,
    "output_classes": CLASS_COUNT,
}


def build_cloud_tail_split2() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(48,), dtype=tf.float32, name="split2_embedding")
    x = tf.keras.layers.Dense(32, activation="relu", name="split2_block3")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="split2_block4")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="split2_block5")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", name="split2_cloud_head")(x)
    return tf.keras.Model(inputs, outputs, name="gesture_cloud_tail_split2")


def validate_cloud_tail_split2(model: tf.keras.Model) -> None:
    if model.input_shape != (None, 48) or model.output_shape != (None, 5):
        raise ValueError("Split2 requires (None, 48) -> (None, 5)")
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if [(layer.units, layer.activation.__name__) for layer in dense] != [
        (32, "relu"), (64, "relu"), (32, "relu"), (5, "softmax")
    ]:
        raise ValueError("Unexpected Split2 continuation architecture")
