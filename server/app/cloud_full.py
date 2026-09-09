"""Complete cloud network for R1 production and retained benchmarks: 10 features -> B1..B5 -> head.

Uses frozen existing weights. The production R1 route calls this complete network with ten features.
"""
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from ml.models.split_models import build_normalized_prefix
from ml.training.train_cloud_tail import load_source_model, SOURCE_MODEL_RELATIVE_PATH, SOURCE_MODEL_SHA256
from .inference import Split3CloudInference

MODEL_VERSION = "gesture-full-cloud-v1.0.0"


class FullCloudInference:
    def __init__(self):
        source = load_source_model(Path(__file__).resolve().parents[2] / SOURCE_MODEL_RELATIVE_PATH)
        self.tail = Split3CloudInference()
        self.prefix = build_normalized_prefix(source, 3)
        inputs = tf.keras.Input(shape=(10,), name="normalized_features_v1")
        self.model = tf.keras.Model(inputs, self.tail._model(self.prefix(inputs)), name="gesture_full_cloud")
        self.model.trainable = False
        self.model_version = MODEL_VERSION
        self.artifact_hashes = {"source_model": SOURCE_MODEL_SHA256, "cloud_tail": self.tail.model_sha256}

    def infer(self, features):
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (10,) or not np.isfinite(vector).all():
            raise ValueError("ALL_CLOUD requires exactly 10 finite normalized features")
        started = time.perf_counter_ns()
        probabilities = self.model(vector[None, :], training=False).numpy()[0]
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1, atol=1e-5):
            raise RuntimeError("Invalid full-cloud probabilities")
        predicted = int(np.argmax(probabilities))
        return {"predicted_class_id": predicted, "confidence": float(probabilities[predicted]),
                "server_latency_ms": elapsed, "model_version": self.model_version}
