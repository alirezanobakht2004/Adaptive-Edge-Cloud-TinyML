import numpy as np
import tensorflow as tf

from ml.models.base_model import (
    MODEL_VERSION,
    build_base_model,
)


def test_base_model_contract() -> None:
    normalization = (
        tf.keras.layers.Normalization(
            axis=-1,
            name="feature_normalization",
        )
    )

    normalization.adapt(
        np.zeros(
            (4, 10),
            dtype=np.float32,
        )
    )

    model = build_base_model(
        normalization
    )

    assert MODEL_VERSION == (
        "gesture-model-v1.0.0"
    )

    assert model.input_shape == (
        None,
        10,
    )

    assert model.output_shape == (
        None,
        5,
    )

    assert (
        model.get_layer("block1").units
        == 64
    )

    assert (
        model.get_layer("block2").units
        == 48
    )

    assert (
        model.get_layer("block3").units
        == 32
    )

    assert (
        model.get_layer("edge_head").units
        == 5
    )