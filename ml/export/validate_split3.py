"""Add float32 and fixed-vector evidence for the existing, unchanged Split3 models."""

import json

import numpy as np
import tensorflow as tf

from ml.export.split_validation import MODELS, SOURCE_VERSION, SPLITS, convert_float32, run_float32
from ml.models.cloud_model import CLOUD_TAIL_VERSION
from ml.models.split_models import build_normalized_prefix
from ml.training.train_cloud_tail import load_source_model, sha256_file


def main() -> None:
    directory = MODELS / CLOUD_TAIL_VERSION
    model_path = directory / f"{CLOUD_TAIL_VERSION}.keras"
    output = directory / f"{CLOUD_TAIL_VERSION}.tflite"
    parity_path = directory / "split3_cloud_parity_vectors.npz"
    report_path = directory / "split3_export_report.json"
    if any(path.exists() for path in (output, parity_path, report_path)):
        raise RuntimeError("Refusing to overwrite Split3 validation evidence")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
    if sha256_file(model_path) != metadata["model_sha256"]:
        raise RuntimeError("Frozen Split3 Keras hash mismatch")
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 32) or model.output_shape != (None, 5):
        raise ValueError("Split3 must be (None, 32) -> (None, 5)")
    source = load_source_model(MODELS / SOURCE_VERSION / f"{SOURCE_VERSION}.keras")
    source_dir = MODELS / SOURCE_VERSION / "tflite"
    prefix_path = source_dir / f"{SOURCE_VERSION}-prefix-b3-float32-normalized-input.tflite"
    prefix_report = json.loads((source_dir / "edge_uncertainty_deployment_report.json").read_text())
    if sha256_file(prefix_path) != prefix_report["prefix_tflite"]["sha256"]:
        raise RuntimeError("Frozen B3 prefix hash mismatch")
    with np.load(SPLITS / "split1_parity_vectors.npz") as vectors:
        normalized = vectors["normalized_inputs"].copy()
        labels = vectors["true_classes"].copy()
        indices = vectors["validation_indices"].copy()
    expected_prefix = build_normalized_prefix(source, 3)(normalized, training=False).numpy()
    embeddings = run_float32(prefix_path, normalized, 10, 32)
    np.testing.assert_allclose(embeddings, expected_prefix, atol=1e-5, rtol=0)
    expected = model(embeddings, training=False).numpy()
    output.write_bytes(convert_float32(model))
    actual = run_float32(output, embeddings, 32, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    np.savez_compressed(parity_path, normalized_inputs=normalized, embeddings=embeddings,
                        expected_split3=expected_prefix, true_classes=labels, validation_indices=indices,
                        expected_probabilities=expected, tflite_probabilities=actual)
    report = dict(split_id=3, input_embedding_dim=32, output_classes=5,
                  source_model_version=SOURCE_VERSION, model_version=CLOUD_TAIL_VERSION,
                  dataset_version="dataset-v1", feature_version="features-v1", test_split_used=False,
                  model_sha256=sha256_file(model_path), tflite_sha256=sha256_file(output),
                  tflite_bytes=output.stat().st_size, prefix_sha256=sha256_file(prefix_path),
                  prefix_bytes=prefix_path.stat().st_size, parity_vector_count=5, parity_tolerance=1e-5,
                  prefix_max_abs_diff=float(np.max(np.abs(embeddings - expected_prefix))),
                  keras_tflite_max_abs_diff=float(np.max(np.abs(actual - expected))))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
