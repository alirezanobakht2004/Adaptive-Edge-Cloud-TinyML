"""Phase 7 Split1 continuation; independent of the frozen Split3 tail."""

from pathlib import Path

import tensorflow as tf

SPLIT_ID = 1
INPUT_EMBEDDING_DIM = 64
CLASS_COUNT = 5
CLOUD_TAIL_VERSION = "gesture-cloud-tail-split1-v1.0.0"
SOURCE_MODEL_VERSION = "gesture-model-v1.1.0"
MODEL_DIR = (Path(__file__).resolve().parents[2] / "data/processed"
             / "dataset-v1/features-v1/models" / CLOUD_TAIL_VERSION)
MODEL_PATH = MODEL_DIR / f"{CLOUD_TAIL_VERSION}.keras"
MODEL_METADATA = {
    "split_id": SPLIT_ID,
    "input_embedding_dim": INPUT_EMBEDDING_DIM,
    "output_classes": CLASS_COUNT,
}


def build_cloud_tail_split1() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64,), dtype=tf.float32, name="split1_embedding")
    x = tf.keras.layers.Dense(48, activation="relu", name="split1_block2")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="split1_block3")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", name="split1_cloud_head")(x)
    return tf.keras.Model(inputs, outputs, name="gesture_cloud_tail_split1")


def validate_cloud_tail_split1(model: tf.keras.Model) -> None:
    if model.input_shape != (None, 64) or model.output_shape != (None, 5):
        raise ValueError("Split1 requires (None, 64) -> (None, 5)")
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if [(layer.units, layer.activation.__name__) for layer in dense] != [
        (48, "relu"), (32, "relu"), (5, "softmax")
    ]:
        raise ValueError("Unexpected Split1 continuation architecture")
