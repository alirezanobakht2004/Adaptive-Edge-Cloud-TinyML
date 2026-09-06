"""Phase 7 split-point contracts and prefix builders."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


SOURCE_MODEL_VERSION = "gesture-model-v1.1.0"
INPUT_FEATURES = 10


@dataclass(frozen=True)
class SplitPoint:
    split_id: int
    edge_layers: tuple[str, ...]
    output_layer: str
    embedding_dim: int


SPLIT_POINTS = {
    1: SplitPoint(
        split_id=1,
        edge_layers=("block1",),
        output_layer="block1",
        embedding_dim=64,
    ),
    2: SplitPoint(
        split_id=2,
        edge_layers=("block1", "block2"),
        output_layer="block2",
        embedding_dim=48,
    ),
    3: SplitPoint(
        split_id=3,
        edge_layers=("block1", "block2", "block3"),
        output_layer="block3",
        embedding_dim=32,
    ),
}


def get_split_point(split_id: int) -> SplitPoint:
    try:
        return SPLIT_POINTS[split_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported split_id={split_id}; expected 1, 2, or 3."
        ) from exc


def validate_source_model(model: tf.keras.Model) -> None:
    expected_units = {
        "block1": 64,
        "block2": 48,
        "block3": 32,
    }

    for layer_name, units in expected_units.items():
        layer = model.get_layer(layer_name)

        if not isinstance(layer, tf.keras.layers.Dense):
            raise TypeError(
                f"{layer_name} must be a Dense layer."
            )

        if layer.units != units:
            raise ValueError(
                f"{layer_name} units mismatch: "
                f"expected {units}, got {layer.units}."
            )


def build_normalized_prefix(
    model: tf.keras.Model,
    split_id: int,
) -> tf.keras.Model:
    """Build a split prefix that consumes externally normalized features-v1."""

    validate_source_model(model)
    spec = get_split_point(split_id)

    inputs = tf.keras.Input(
        shape=(INPUT_FEATURES,),
        dtype=tf.float32,
        name="normalized_features_v1",
    )

    x = inputs

    for layer_name in spec.edge_layers:
        x = model.get_layer(layer_name)(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name=f"gesture_split{split_id}_prefix",
    )


__all__ = [
    "INPUT_FEATURES",
    "SOURCE_MODEL_VERSION",
    "SPLIT_POINTS",
    "SplitPoint",
    "build_normalized_prefix",
    "get_split_point",
    "validate_source_model",
]
