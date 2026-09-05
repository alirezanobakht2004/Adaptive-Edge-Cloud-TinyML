import numpy as np
import tensorflow as tf

from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
    build_mc_dropout_model,
)


def _normalization() -> tf.keras.layers.Normalization:
    normalization = tf.keras.layers.Normalization(
        axis=-1,
        name="feature_normalization",
    )

    calibration_data = np.asarray(
        [
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8],
            [0.5, -0.5, 0.7, -0.7, 0.9, -0.9, 1.1, -1.1, 1.3, -1.3],
        ],
        dtype=np.float32,
    )

    normalization.adapt(calibration_data)
    return normalization


def test_mc_dropout_contract() -> None:
    tf.keras.utils.set_random_seed(42)

    model = build_mc_dropout_model(
        _normalization()
    )

    assert UNCERTAINTY_MODEL_VERSION == (
        "gesture-model-v1.1.0"
    )

    assert MC_DROPOUT_PASSES == 5
    assert MC_DROPOUT_RATE == 0.2

    assert model.input_shape == (
        None,
        10,
    )

    assert model.output_shape == (
        None,
        5,
    )

    assert model.get_layer("block1").units == 64
    assert model.get_layer("block2").units == 48
    assert model.get_layer("block3").units == 32
    assert model.get_layer("edge_head").units == 5

    dropout = model.get_layer("mc_dropout")

    assert isinstance(
        dropout,
        tf.keras.layers.Dropout,
    )

    assert np.isclose(
        dropout.rate,
        0.2,
    )


def test_mc_dropout_is_disabled_for_normal_inference() -> None:
    tf.keras.utils.set_random_seed(42)

    model = build_mc_dropout_model(
        _normalization()
    )

    sample = np.asarray(
        [[
            0.15,
            -0.25,
            0.35,
            -0.45,
            0.55,
            -0.65,
            0.75,
            -0.85,
            0.95,
            -1.05,
        ]],
        dtype=np.float32,
    )

    first = model(
        sample,
        training=False,
    ).numpy()

    second = model(
        sample,
        training=False,
    ).numpy()

    np.testing.assert_allclose(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_mc_dropout_mask_is_stochastic_when_training_true() -> None:
    tf.keras.utils.set_random_seed(42)

    model = build_mc_dropout_model(
        _normalization()
    )

    dropout = model.get_layer(
        "mc_dropout"
    )

    probe = tf.ones(
        (1, 32),
        dtype=tf.float32,
    )

    stochastic_outputs = [
        dropout(
            probe,
            training=True,
        ).numpy()
        for _ in range(
            MC_DROPOUT_PASSES
        )
    ]

    differences = [
        not np.array_equal(
            stochastic_outputs[0],
            candidate,
        )
        for candidate
        in stochastic_outputs[1:]
    ]

    assert any(differences)
