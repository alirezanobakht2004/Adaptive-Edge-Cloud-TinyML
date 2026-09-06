import numpy as np

from ml.models.cloud_model import (
    CLASS_COUNT,
    CLOUD_BLOCK_4_UNITS,
    CLOUD_BLOCK_5_UNITS,
    CLOUD_TAIL_VERSION,
    INPUT_EMBEDDING_DIM,
    SOURCE_EDGE_MODEL_VERSION,
    build_cloud_tail,
)


def test_cloud_tail_contract() -> None:
    model = build_cloud_tail()

    assert CLOUD_TAIL_VERSION == "gesture-cloud-tail-v1.0.0"
    assert SOURCE_EDGE_MODEL_VERSION == "gesture-model-v1.1.0"

    assert INPUT_EMBEDDING_DIM == 32
    assert CLOUD_BLOCK_4_UNITS == 64
    assert CLOUD_BLOCK_5_UNITS == 32
    assert CLASS_COUNT == 5

    assert model.input_shape == (None, 32)
    assert model.output_shape == (None, 5)

    assert model.get_layer("block4").units == 64
    assert model.get_layer("block5").units == 32
    assert model.get_layer("cloud_head").units == 5

    x = np.zeros((3, 32), dtype=np.float32)

    probabilities = model(
        x,
        training=False,
    ).numpy()

    assert probabilities.shape == (3, 5)
    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(3, dtype=np.float32),
        atol=1e-6,
        rtol=0.0,
    )
