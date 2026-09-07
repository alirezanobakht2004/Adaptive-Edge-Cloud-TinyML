import json

import numpy as np
import tensorflow as tf

from ml.models.cloud_tail_split2 import MODEL_DIR, MODEL_PATH, CLOUD_TAIL_VERSION, validate_cloud_tail_split2
from ml.export.split_validation import run_float32
from ml.training.train_cloud_tail import sha256_file


def test_split2_model_contract_and_parity():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    validate_cloud_tail_split2(model)
    assert model.name == "gesture_cloud_tail_split2"
    assert model.input_shape == (None, 48)
    assert model.output_shape == (None, 5)
    with np.load(MODEL_DIR / "split2_cloud_parity_vectors.npz") as vectors:
        assert vectors["embeddings"].shape == (5, 48)
        first = model(vectors["embeddings"], training=False).numpy()
        np.testing.assert_array_equal(first, model(vectors["embeddings"], training=False).numpy())
        actual = run_float32(MODEL_DIR / f"{CLOUD_TAIL_VERSION}.tflite", vectors["embeddings"], 48, 5)
        np.testing.assert_allclose(actual, first, atol=1e-5, rtol=0)
        np.testing.assert_allclose(first, vectors["expected_probabilities"], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(actual.argmax(1), first.argmax(1))
        np.testing.assert_allclose(first.sum(1), 1, atol=1e-6)


def test_split2_export_provenance():
    report = json.loads((MODEL_DIR / "split2_export_report.json").read_text())
    assert report["split_id"] == 2
    assert report["embedding_dimension"] == 48
    assert report["test_split_used"] is False
    assert report["frozen_prefix_verified"] is True
    assert report["model_sha256"] == sha256_file(MODEL_PATH)
    assert report["tflite_sha256"] == sha256_file(MODEL_DIR / f"{CLOUD_TAIL_VERSION}.tflite")
