"""Artifact, prefix-to-tail parity, routing, and MQTT callback integration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from ml.export.export_split1 import run_tflite as run_prefix
from ml.export.export_split1_cloud_tail import run_tail_tflite
from ml.models.cloud_tail_split1 import (
    CLOUD_TAIL_VERSION, MODEL_DIR, MODEL_PATH, SOURCE_MODEL_VERSION,
)
from server.app.inference import Split1CloudInference, SplitCloudInference
from server.app.mqtt import ServerState, on_message
from server.app.schemas import parse_inference_request


@pytest.fixture(scope="module")
def runtime():
    return SplitCloudInference()


def request(embedding, split=1):
    return dict(request_id="split1-test-001", device_id="esp32-01", timestamp_ms=123,
                split=split, embedding=list(embedding), model_version=SOURCE_MODEL_VERSION)


def test_model_load_shape_and_determinism():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    assert model.name == "gesture_cloud_tail_split1"
    assert model.input_shape == (None, 64)
    assert model.output_shape == (None, 5)
    with np.load(MODEL_DIR / "split1_cloud_parity_vectors.npz") as vectors:
        inputs = vectors["embeddings"]
        first = model(inputs, training=False).numpy()
        np.testing.assert_array_equal(first, model(inputs, training=False).numpy())
        np.testing.assert_allclose(first, vectors["expected_probabilities"], atol=2e-5, rtol=0)
        np.testing.assert_allclose(first.sum(1), 1, atol=1e-6)


def test_frozen_prefix_to_cloud_tail_parity(runtime):
    prefix = (MODEL_DIR.parent / SOURCE_MODEL_VERSION / "tflite/splits"
              / f"{SOURCE_MODEL_VERSION}-split1-prefix-float32-normalized-input.tflite")
    with np.load(MODEL_DIR / "split1_cloud_parity_vectors.npz") as vectors:
        embeddings = run_prefix(prefix, vectors["normalized_inputs"])
        np.testing.assert_allclose(embeddings, vectors["embeddings"], atol=2e-5, rtol=0)
        probabilities = run_tail_tflite(MODEL_DIR / f"{CLOUD_TAIL_VERSION}.tflite", embeddings)
        np.testing.assert_allclose(probabilities, vectors["expected_probabilities"], atol=2e-5, rtol=0)
        for embedding, expected in zip(embeddings, vectors["expected_probabilities"]):
            result = runtime.infer(embedding, split=1)
            assert result.split == 1
            assert result.model_version == CLOUD_TAIL_VERSION
            assert result.confidence == pytest.approx(float(expected.max()), abs=2e-5)
            assert result.predicted_class == runtime.runtimes[1]._id_to_class[int(expected.argmax())]


@pytest.mark.parametrize("split,dimension", [(1, 64), (3, 32)])
def test_mqtt_request_to_real_model_response(runtime, split, dimension):
    published = []

    class Client:
        def publish(self, topic, **kwargs):
            published.append((topic, json.loads(kwargs["payload"])))
            return SimpleNamespace(rc=0)

    payload = json.dumps(request([0.0] * dimension, split)).encode()
    message = SimpleNamespace(topic="gesture/esp32-01/inference/request", payload=payload)
    on_message(Client(), ServerState(runtime), message)
    assert len(published) == 1
    topic, response = published[0]
    assert topic == "gesture/esp32-01/inference/response"
    assert response["request_id"] == "split1-test-001"
    assert response["split"] == split
    assert response["model_version"] == runtime.runtimes[split].model_version
    assert 0 <= response["confidence"] <= 1


@pytest.mark.parametrize("embedding", [[0.0] * 32, [0.0] * 65, [float("nan")] * 64,
                                       [float("inf")] * 64, [True] * 64, ["0"] * 64])
def test_reject_invalid_split1_payload(embedding):
    with pytest.raises(ValueError):
        parse_inference_request(json.dumps(request(embedding)).encode())


@pytest.mark.parametrize("split", [2, 0, True, "1"])
def test_reject_unsupported_routing(runtime, split):
    with pytest.raises(ValueError):
        runtime.infer([0.0] * 64, split=split)


def test_reject_wrong_source_metadata(tmp_path):
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    metadata["source_edge_model_sha256"] = "0" * 64
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="source_edge_model_sha256"):
        Split1CloudInference(metadata_path=path)


def test_reject_corrupted_artifact(tmp_path):
    path = tmp_path / "corrupt.keras"
    path.write_bytes(MODEL_PATH.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        Split1CloudInference(model_path=path)
