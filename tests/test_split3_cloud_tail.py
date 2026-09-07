import json

import numpy as np
import tensorflow as tf

from ml.export.split_validation import MODELS, SOURCE_VERSION, run_float32
from ml.models.cloud_model import CLOUD_TAIL_VERSION
from server.app.inference import Split3CloudInference
from server.app.mqtt import build_inference_response
from server.app.schemas import parse_inference_request


def test_existing_split3_artifacts_and_prefix_tail_parity():
    directory = MODELS / CLOUD_TAIL_VERSION
    model = tf.keras.models.load_model(directory / f"{CLOUD_TAIL_VERSION}.keras", compile=False)
    assert model.input_shape == (None, 32)
    assert model.output_shape == (None, 5)
    prefix = MODELS / SOURCE_VERSION / "tflite" / f"{SOURCE_VERSION}-prefix-b3-float32-normalized-input.tflite"
    runtime = Split3CloudInference()
    with np.load(directory / "split3_cloud_parity_vectors.npz") as vectors:
        embeddings = run_float32(prefix, vectors["normalized_inputs"], 10, 32)
        np.testing.assert_allclose(embeddings, vectors["expected_split3"], atol=1e-5, rtol=0)
        expected = model(embeddings, training=False).numpy()
        actual = run_float32(directory / f"{CLOUD_TAIL_VERSION}.tflite", embeddings, 32, 5)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        np.testing.assert_allclose(expected, vectors["expected_probabilities"], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(expected, model(embeddings, training=False).numpy())
        for row, probabilities in zip(embeddings, expected):
            request = dict(request_id="split3-regression", device_id="esp32-test", timestamp_ms=0,
                           split=3, embedding=row.tolist(), model_version=SOURCE_VERSION)
            response = build_inference_response(parse_inference_request(json.dumps(request).encode()), runtime)
            assert response["model_version"] == CLOUD_TAIL_VERSION
            assert response["split"] == 3
            assert response["predicted_class"] == runtime._id_to_class[int(probabilities.argmax())]
