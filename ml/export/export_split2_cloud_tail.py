"""Export Split2 cloud continuation and five float32 parity vectors."""

import json

import numpy as np
import tensorflow as tf

from ml.models.cloud_tail_split2 import (
    MODEL_DIR, MODEL_PATH, CLOUD_TAIL_VERSION, validate_cloud_tail_split2,
)
from ml.export.split_validation import SPLITS, MODELS, SOURCE_VERSION, convert_float32, run_float32
from ml.models.split_models import build_normalized_prefix
from ml.training.train_cloud_tail import load_source_model, sha256_file


def main() -> None:
    output = MODEL_DIR / f"{CLOUD_TAIL_VERSION}.tflite"
    parity_path = MODEL_DIR / "split2_cloud_parity_vectors.npz"
    if output.exists() or parity_path.exists():
        raise RuntimeError("Refusing to overwrite Split2 export evidence")
    report_path = MODEL_DIR / "split2_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256_file(MODEL_PATH) != report["model_sha256"]:
        raise RuntimeError("Split2 Keras hash mismatch")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    validate_cloud_tail_split2(model)
    source = load_source_model(MODELS / SOURCE_VERSION / f"{SOURCE_VERSION}.keras")
    prefix = build_normalized_prefix(source, 2)
    with np.load(SPLITS / "split1_parity_vectors.npz") as vectors:
        normalized = vectors["normalized_inputs"].copy()
        labels = vectors["true_classes"].copy()
        indices = vectors["validation_indices"].copy()
    embeddings = prefix(normalized, training=False).numpy()
    expected = model(embeddings, training=False).numpy()
    output.write_bytes(convert_float32(model))
    actual = run_float32(output, embeddings, 48, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    np.savez_compressed(parity_path, embeddings=embeddings, normalized_inputs=normalized,
                        true_classes=labels, validation_indices=indices,
                        expected_probabilities=expected, tflite_probabilities=actual)
    report.update({"tflite_exported": True, "quantized": False,
                   "tflite_sha256": sha256_file(output), "tflite_bytes": output.stat().st_size,
                   "parity_vector_count": len(embeddings), "parity_tolerance": 1e-5,
                   "keras_tflite_max_abs_diff": float(np.max(np.abs(expected - actual)))})
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
