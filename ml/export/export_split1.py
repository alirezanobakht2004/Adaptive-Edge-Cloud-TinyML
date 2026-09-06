"""Export and validate the Phase 7 Split-1 edge prefix.

This step is desktop-only. It does not generate firmware yet.
The prefix consumes the already-frozen externally normalized features-v1
vector and produces the 64-D Block-1 embedding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import DATASET_VERSION, GESTURES
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.split_models import (
    INPUT_FEATURES,
    SOURCE_MODEL_VERSION,
    build_normalized_prefix,
    get_split_point,
    validate_source_model,
)


SPLIT_ID = 1
SPLIT_SPEC = get_split_point(SPLIT_ID)

PRODUCTION_MODEL_VERSION = "gesture-model-v1.0.0"

MODEL_DIR = Path(
    "data/processed"
) / DATASET_VERSION / FEATURE_VERSION / "models" / SOURCE_MODEL_VERSION

MODEL_PATH = MODEL_DIR / f"{SOURCE_MODEL_VERSION}.keras"

PHASE5_REPORT_PATH = (
    MODEL_DIR
    / "tflite"
    / "edge_uncertainty_deployment_report.json"
)

REFERENCE_NORMALIZATION_PATH = (
    Path("data/processed")
    / DATASET_VERSION
    / FEATURE_VERSION
    / "models"
    / PRODUCTION_MODEL_VERSION
    / "tflite"
    / "deployment_normalization.json"
)

OUTPUT_DIR = MODEL_DIR / "tflite" / "splits"

PREFIX_FILENAME = (
    f"{SOURCE_MODEL_VERSION}-split1-prefix-"
    "float32-normalized-input.tflite"
)

REPORT_FILENAME = "split1_export_report.json"
PARITY_VECTORS_FILENAME = "split1_parity_vectors.npz"

EXPECTED_VALIDATION_SESSION = "session_02"
EXPECTED_VALIDATION_INDICES = (0, 120, 160, 40, 80)

NORMALIZATION_TOLERANCE = 1e-6
PREFIX_PARITY_TOLERANCE = 2e-5


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_outputs_are_new(output_dir: Path) -> None:
    protected = (
        output_dir / PREFIX_FILENAME,
        output_dir / REPORT_FILENAME,
        output_dir / PARITY_VECTORS_FILENAME,
    )

    existing = [path for path in protected if path.exists()]

    if existing:
        raise RuntimeError(
            "Refusing to overwrite existing Split-1 evidence: "
            + ", ".join(str(path) for path in existing)
        )


def load_phase5_source_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-5 deployment report not found: {path}"
        )

    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("model_version") != SOURCE_MODEL_VERSION:
        raise ValueError(
            "Phase-5 report model version mismatch."
        )

    if report.get("test_split_used") is not False:
        raise ValueError(
            "Expected Phase-5 report test_split_used=false."
        )

    source_hash = report.get("source_model_sha256")

    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError(
            "Phase-5 report has no valid source_model_sha256."
        )

    return source_hash


def load_reference_normalization(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen normalization file not found: {path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Normalization dataset version mismatch.")

    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Normalization feature version mismatch.")

    mean = np.asarray(payload["mean"], dtype=np.float32)
    variance = np.asarray(payload["variance"], dtype=np.float32)

    if mean.shape != (INPUT_FEATURES,):
        raise ValueError("Normalization mean shape mismatch.")

    if variance.shape != (INPUT_FEATURES,):
        raise ValueError("Normalization variance shape mismatch.")

    return mean, variance


def candidate_normalization_statistics(
    model: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray]:
    layer = model.get_layer("feature_normalization")

    if not isinstance(layer, tf.keras.layers.Normalization):
        raise TypeError(
            "Expected feature_normalization to be a Keras Normalization layer."
        )

    mean = layer.mean.numpy().reshape(-1).astype(np.float32)
    variance = layer.variance.numpy().reshape(-1).astype(np.float32)

    return mean, variance


def convert_float32_tflite(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []

    converted = converter.convert()

    if not converted:
        raise RuntimeError("TFLite converter returned an empty model.")

    return converted


def run_tflite(
    model_path: Path,
    inputs: np.ndarray,
) -> np.ndarray:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError(
            "Split-1 TFLite must have exactly one input and one output."
        )

    input_detail = input_details[0]
    output_detail = output_details[0]

    if input_detail["dtype"] != np.float32:
        raise ValueError("Split-1 input must be float32.")

    if output_detail["dtype"] != np.float32:
        raise ValueError("Split-1 output must be float32.")

    if tuple(input_detail["shape"]) != (1, INPUT_FEATURES):
        raise ValueError(
            f"Unexpected Split-1 input shape: {input_detail['shape']}"
        )

    if tuple(output_detail["shape"]) != (
        1,
        SPLIT_SPEC.embedding_dim,
    ):
        raise ValueError(
            f"Unexpected Split-1 output shape: {output_detail['shape']}"
        )

    outputs = np.empty(
        (inputs.shape[0], SPLIT_SPEC.embedding_dim),
        dtype=np.float32,
    )

    for index, row in enumerate(inputs):
        interpreter.set_tensor(
            input_detail["index"],
            row.reshape(1, INPUT_FEATURES).astype(
                np.float32,
                copy=False,
            ),
        )

        interpreter.invoke()

        outputs[index] = interpreter.get_tensor(
            output_detail["index"]
        ).reshape(SPLIT_SPEC.embedding_dim)

    return outputs


def selected_validation_indices(
    labels: np.ndarray,
) -> tuple[int, ...]:
    selected = []

    for class_id in range(len(GESTURES)):
        matches = np.flatnonzero(labels == class_id)

        if matches.size == 0:
            raise ValueError(
                f"Validation is missing class {class_id}."
            )

        selected.append(int(matches[0]))

    result = tuple(selected)

    if result != EXPECTED_VALIDATION_INDICES:
        raise RuntimeError(
            "Frozen validation-vector indices changed: "
            f"expected {EXPECTED_VALIDATION_INDICES}, got {result}."
        )

    return result


def main() -> None:
    root = project_root()

    model_path = root / MODEL_PATH
    phase5_report_path = root / PHASE5_REPORT_PATH
    normalization_path = root / REFERENCE_NORMALIZATION_PATH
    output_dir = root / OUTPUT_DIR

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen source model not found: {model_path}"
        )

    ensure_outputs_are_new(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_source_hash = load_phase5_source_hash(
        phase5_report_path
    )
    actual_source_hash = sha256_file(model_path)

    if actual_source_hash != expected_source_hash:
        raise RuntimeError(
            "Frozen gesture-model-v1.1.0 hash mismatch. "
            f"Expected {expected_source_hash}, got {actual_source_hash}."
        )

    validation = load_feature_split("validation")

    if validation.session != EXPECTED_VALIDATION_SESSION:
        raise ValueError(
            f"Expected VALIDATION={EXPECTED_VALIDATION_SESSION}."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    validate_source_model(model)

    reference_mean, reference_variance = (
        load_reference_normalization(normalization_path)
    )

    candidate_mean, candidate_variance = (
        candidate_normalization_statistics(model)
    )

    mean_diff = float(
        np.max(np.abs(candidate_mean - reference_mean))
    )
    variance_diff = float(
        np.max(np.abs(candidate_variance - reference_variance))
    )

    if (
        mean_diff > NORMALIZATION_TOLERANCE
        or variance_diff > NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError(
            "Source-model normalization no longer matches the "
            "frozen firmware preprocessing contract."
        )

    normalized = (
        (
            validation.features.astype(np.float32)
            - reference_mean
        )
        / np.sqrt(reference_variance).astype(np.float32)
    ).astype(np.float32)

    raw_reference_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            SPLIT_SPEC.output_layer
        ).output,
        name="split1_raw_reference",
    )

    raw_reference = raw_reference_model(
        validation.features,
        training=False,
    ).numpy().astype(np.float32)

    prefix_model = build_normalized_prefix(
        model,
        SPLIT_ID,
    )

    keras_external = prefix_model(
        normalized,
        training=False,
    ).numpy().astype(np.float32)

    keras_external_diff = float(
        np.max(np.abs(raw_reference - keras_external))
    )

    if keras_external_diff > PREFIX_PARITY_TOLERANCE:
        raise RuntimeError(
            "Split-1 external-normalization Keras parity failed: "
            f"max_abs_diff={keras_external_diff}"
        )

    prefix_bytes = convert_float32_tflite(
        prefix_model
    )

    prefix_path = output_dir / PREFIX_FILENAME
    prefix_path.write_bytes(prefix_bytes)

    tflite_output = run_tflite(
        prefix_path,
        normalized,
    )

    tflite_diff = float(
        np.max(np.abs(raw_reference - tflite_output))
    )

    if tflite_diff > PREFIX_PARITY_TOLERANCE:
        raise RuntimeError(
            "Split-1 TFLite parity failed: "
            f"max_abs_diff={tflite_diff}"
        )

    selected = selected_validation_indices(
        validation.labels
    )

    parity_vectors_path = (
        output_dir / PARITY_VECTORS_FILENAME
    )

    np.savez_compressed(
        parity_vectors_path,
        validation_indices=np.asarray(
            selected,
            dtype=np.int64,
        ),
        true_classes=validation.labels[
            list(selected)
        ].astype(np.int64),
        raw_features=validation.features[
            list(selected)
        ].astype(np.float32),
        normalized_inputs=normalized[
            list(selected)
        ],
        expected_split1=raw_reference[
            list(selected)
        ],
        tflite_split1=tflite_output[
            list(selected)
        ],
    )

    report = {
        "phase": 7,
        "milestone": "M8",
        "split": SPLIT_ID,
        "source_model_version": SOURCE_MODEL_VERSION,
        "source_model_sha256": actual_source_hash,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "test_split_used": False,
        "validation_session": validation.session,
        "validation_samples": int(
            validation.features.shape[0]
        ),
        "input_contract":
            "externally normalized features-v1 float32",
        "input_shape": [1, INPUT_FEATURES],
        "output_layer": SPLIT_SPEC.output_layer,
        "embedding_dim": SPLIT_SPEC.embedding_dim,
        "output_shape": [
            1,
            SPLIT_SPEC.embedding_dim,
        ],
        "prefix_tflite": {
            "filename": PREFIX_FILENAME,
            "sha256": sha256_file(prefix_path),
            "bytes": len(prefix_bytes),
        },
        "normalization_compatibility": {
            "reference_model_version":
                PRODUCTION_MODEL_VERSION,
            "max_abs_mean_diff": mean_diff,
            "max_abs_variance_diff":
                variance_diff,
            "tolerance":
                NORMALIZATION_TOLERANCE,
            "pass": True,
        },
        "desktop_parity": {
            "keras_external_vs_raw_max_abs_diff":
                keras_external_diff,
            "tflite_vs_raw_max_abs_diff":
                tflite_diff,
            "tolerance":
                PREFIX_PARITY_TOLERANCE,
            "pass": True,
        },
        "parity_vectors": {
            "filename":
                PARITY_VECTORS_FILENAME,
            "validation_indices":
                list(selected),
            "count":
                len(selected),
        },
        "firmware_generated": False,
        "esp32_parity_completed": False,
        "server_tail_completed": False,
        "end_to_end_completed": False,
    }

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("PHASE 7 / M8 — SPLIT 1 EXPORT COMPLETE")
    print("======================================")
    print(
        f"Source model:             {SOURCE_MODEL_VERSION}"
    )
    print(
        f"Source SHA-256:           {actual_source_hash}"
    )
    print("TEST loaded:              NO")
    print(
        f"Validation samples:       {validation.features.shape[0]}"
    )
    print(
        f"Input/output:             "
        f"(1,{INPUT_FEATURES}) -> "
        f"(1,{SPLIT_SPEC.embedding_dim})"
    )
    print(
        f"Keras external max diff:  {keras_external_diff:.9g}"
    )
    print(
        f"TFLite max abs diff:      {tflite_diff:.9g}"
    )
    print(
        f"TFLite bytes:             {len(prefix_bytes)}"
    )
    print(
        f"TFLite SHA-256:           {sha256_file(prefix_path)}"
    )
    print("Desktop parity:           PASS")
    print("Firmware generated:       NO")
    print("ESP32 parity completed:   NO")
    print("Server tail completed:    NO")
    print("End-to-end completed:     NO")
    print()
    print(f"TFLite:  {prefix_path}")
    print(f"Vectors: {parity_vectors_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
