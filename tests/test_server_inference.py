import math

import numpy as np
import pytest

from ml.models.cloud_model import (
    CLOUD_TAIL_VERSION,
    INPUT_EMBEDDING_DIM,
    SOURCE_EDGE_MODEL_VERSION,
)
from server.app.inference import Split3CloudInference


@pytest.fixture(scope="module")
def runtime() -> Split3CloudInference:
    return Split3CloudInference()


def test_split3_cloud_runtime_contract(
    runtime: Split3CloudInference,
) -> None:
    assert runtime.model_version == CLOUD_TAIL_VERSION
    assert (
        runtime.source_edge_model_version
        == SOURCE_EDGE_MODEL_VERSION
    )
    assert runtime.split_point == 3
    assert runtime.embedding_dimension == 32

    assert len(runtime.model_sha256) == 64


def test_split3_cloud_runtime_inference(
    runtime: Split3CloudInference,
) -> None:
    embedding = np.zeros(
        INPUT_EMBEDDING_DIM,
        dtype=np.float32,
    )

    result = runtime.infer(embedding)

    assert result.predicted_class in {
        "IDLE",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "ROTATE_CW",
        "SHAKE",
    }

    assert math.isfinite(result.confidence)
    assert 0.0 <= result.confidence <= 1.0

    assert math.isfinite(result.server_latency_ms)
    assert result.server_latency_ms >= 0.0

    assert result.model_version == CLOUD_TAIL_VERSION


def test_split3_cloud_runtime_rejects_wrong_embedding_size(
    runtime: Split3CloudInference,
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly 32 values",
    ):
        runtime.infer([0.0] * 31)


def test_split3_cloud_runtime_rejects_nonfinite_embedding(
    runtime: Split3CloudInference,
) -> None:
    embedding = [0.0] * INPUT_EMBEDDING_DIM
    embedding[7] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        runtime.infer(embedding)
