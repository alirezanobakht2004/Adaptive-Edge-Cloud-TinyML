from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
)
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import MODEL_VERSION


DATASET_VERSION = "dataset-v1"

MODEL_DIR = Path(
    "data/processed/"
    "dataset-v1/"
    "features-v1/"
    "models/"
    f"{MODEL_VERSION}"
)

KERAS_FILENAME = f"{MODEL_VERSION}.keras"
TEST_EVALUATION_FILENAME = "test_evaluation.json"

TFLITE_DIRNAME = "tflite"
TFLITE_FILENAME = f"{MODEL_VERSION}-float32.tflite"
PARITY_REPORT_FILENAME = "float32_parity.json"
PARITY_VECTORS_FILENAME = "parity_vectors.json"

EXPECTED_INPUT_FEATURES = 10
EXPECTED_CLASSES = 5

PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-5

PARITY_VECTORS_PER_CLASS = 2


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_final_test_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "Final held-out test record not found: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        record = json.load(file)

    expected = {
        "evaluation_type": "final-held-out-test",
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "test_split": "session_03",
    }

    for field, expected_value in expected.items():
        actual = record.get(field)

        if actual != expected_value:
            raise ValueError(
                f"Final-test record mismatch: "
                f"{field}={actual!r}, "
                f"expected {expected_value!r}."
            )

    model_hash = record.get("model_sha256")

    if not isinstance(model_hash, str):
        raise ValueError(
            "Final-test record has no valid model_sha256."
        )

    return record


def verify_frozen_model(
    model_path: Path,
    final_test_record: dict,
) -> str:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen Keras model not found: {model_path}"
        )

    actual_hash = sha256_file(model_path)
    expected_hash = final_test_record["model_sha256"]

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen model hash does not match the model "
            "used for the final held-out test.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            "Do not export this model."
        )

    return actual_hash


def convert_to_float32_tflite(
    model: tf.keras.Model,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    # No quantization in this step.
    converter.optimizations = []

    tflite_model = converter.convert()

    if not tflite_model:
        raise RuntimeError(
            "TFLite converter returned an empty model."
        )

    return tflite_model


def create_interpreter(
    tflite_path: Path,
) -> tuple[
    tf.lite.Interpreter,
    dict,
    dict,
]:
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise ValueError(
            "Expected exactly one TFLite input tensor."
        )

    if len(output_details) != 1:
        raise ValueError(
            "Expected exactly one TFLite output tensor."
        )

    input_detail = input_details[0]
    output_detail = output_details[0]

    if input_detail["dtype"] != np.float32:
        raise ValueError(
            "Float32 export does not have float32 input."
        )

    if output_detail["dtype"] != np.float32:
        raise ValueError(
            "Float32 export does not have float32 output."
        )

    return (
        interpreter,
        input_detail,
        output_detail,
    )


def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    output_detail: dict,
    features: np.ndarray,
) -> np.ndarray:
    features = np.asarray(
        features,
        dtype=np.float32,
    )

    outputs = np.empty(
        (
            features.shape[0],
            EXPECTED_CLASSES,
        ),
        dtype=np.float32,
    )

    for index, feature_vector in enumerate(features):
        model_input = feature_vector.reshape(
            1,
            EXPECTED_INPUT_FEATURES,
        )

        interpreter.set_tensor(
            input_detail["index"],
            model_input,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_detail["index"]
        )

        outputs[index] = output.reshape(
            EXPECTED_CLASSES
        )

    return outputs


def select_fixed_parity_vectors(
    features: np.ndarray,
    labels: np.ndarray,
) -> list[int]:
    selected: list[int] = []

    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]

        indices = np.flatnonzero(
            labels == class_id
        )

        if indices.size < PARITY_VECTORS_PER_CLASS:
            raise ValueError(
                f"Not enough validation vectors for "
                f"{gesture}."
            )

        selected.extend(
            indices[
                :PARITY_VECTORS_PER_CLASS
            ].tolist()
        )

    return selected


def save_parity_vectors(
    *,
    path: Path,
    selected_indices: list[int],
    validation,
    keras_probabilities: np.ndarray,
    tflite_probabilities: np.ndarray,
) -> None:
    vectors = []

    for index in selected_indices:
        true_id = int(
            validation.labels[index]
        )

        vectors.append(
            {
                "validation_index": index,
                "source_csv":
                    validation.csv_paths[
                        index
                    ].as_posix(),
                "true_id": true_id,
                "true_class":
                    GESTURES[true_id],
                "features": (
                    validation.features[index]
                    .astype(float)
                    .tolist()
                ),
                "keras_probabilities": (
                    keras_probabilities[index]
                    .astype(float)
                    .tolist()
                ),
                "tflite_probabilities": (
                    tflite_probabilities[index]
                    .astype(float)
                    .tolist()
                ),
            }
        )

    payload = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_split": "validation",
        "vectors_per_class":
            PARITY_VECTORS_PER_CLASS,
        "vectors": vectors,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def main() -> None:
    root = project_root()
    model_dir = root / MODEL_DIR

    keras_path = (
        model_dir / KERAS_FILENAME
    )

    final_test_path = (
        model_dir
        / TEST_EVALUATION_FILENAME
    )

    tflite_dir = (
        model_dir / TFLITE_DIRNAME
    )

    tflite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tflite_path = (
        tflite_dir / TFLITE_FILENAME
    )

    report_path = (
        tflite_dir
        / PARITY_REPORT_FILENAME
    )

    vectors_path = (
        tflite_dir
        / PARITY_VECTORS_FILENAME
    )

    final_test_record = load_final_test_record(
        final_test_path
    )

    keras_hash = verify_frozen_model(
        keras_path,
        final_test_record,
    )

    model = tf.keras.models.load_model(
        keras_path
    )

    if model.input_shape != (
        None,
        EXPECTED_INPUT_FEATURES,
    ):
        raise ValueError(
            f"Unexpected Keras input shape: "
            f"{model.input_shape}"
        )

    if model.output_shape != (
        None,
        EXPECTED_CLASSES,
    ):
        raise ValueError(
            f"Unexpected Keras output shape: "
            f"{model.output_shape}"
        )

    tflite_bytes = convert_to_float32_tflite(
        model
    )

    tflite_path.write_bytes(
        tflite_bytes
    )

    tflite_hash = sha256_file(
        tflite_path
    )

    (
        interpreter,
        input_detail,
        output_detail,
    ) = create_interpreter(
        tflite_path
    )

    # Deployment parity uses validation data.
    # No model tuning is performed here.
    validation = load_feature_split(
        "validation"
    )

    keras_probabilities = model.predict(
        validation.features,
        verbose=0,
    ).astype(np.float32)

    tflite_probabilities = run_tflite_inference(
        interpreter,
        input_detail,
        output_detail,
        validation.features,
    )

    if keras_probabilities.shape != (
        200,
        EXPECTED_CLASSES,
    ):
        raise ValueError(
            f"Unexpected Keras prediction shape: "
            f"{keras_probabilities.shape}"
        )

    difference = np.abs(
        keras_probabilities
        - tflite_probabilities
    )

    max_abs_difference = float(
        np.max(difference)
    )

    mean_abs_difference = float(
        np.mean(difference)
    )

    keras_predictions = np.argmax(
        keras_probabilities,
        axis=1,
    )

    tflite_predictions = np.argmax(
        tflite_probabilities,
        axis=1,
    )

    class_matches = int(
        np.sum(
            keras_predictions
            == tflite_predictions
        )
    )

    class_match_rate = float(
        class_matches
        / validation.features.shape[0]
    )

    numeric_parity = bool(
        np.allclose(
            keras_probabilities,
            tflite_probabilities,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
    )

    class_parity = (
        class_matches
        == validation.features.shape[0]
    )

    parity_passed = (
        numeric_parity
        and class_parity
    )

    selected_indices = (
        select_fixed_parity_vectors(
            validation.features,
            validation.labels,
        )
    )

    save_parity_vectors(
        path=vectors_path,
        selected_indices=selected_indices,
        validation=validation,
        keras_probabilities=keras_probabilities,
        tflite_probabilities=tflite_probabilities,
    )

    report = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "export_type": "float32-tflite",
        "source_model_sha256": keras_hash,
        "tflite_sha256": tflite_hash,
        "source_model_bytes":
            keras_path.stat().st_size,
        "tflite_model_bytes":
            tflite_path.stat().st_size,
        "parity_split": "validation",
        "parity_sample_count": int(
            validation.features.shape[0]
        ),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "max_abs_probability_difference":
            max_abs_difference,
        "mean_abs_probability_difference":
            mean_abs_difference,
        "class_matches": class_matches,
        "class_match_rate":
            class_match_rate,
        "numeric_parity_passed":
            numeric_parity,
        "class_parity_passed":
            class_parity,
        "parity_passed":
            parity_passed,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print("FLOAT32 TFLITE EXPORT")
    print("---------------------")
    print(
        f"Model version:   {MODEL_VERSION}"
    )
    print(
        f"Source SHA-256: {keras_hash}"
    )
    print(
        f"TFLite SHA-256: {tflite_hash}"
    )
    print()
    print(
        f"Keras bytes:     "
        f"{keras_path.stat().st_size}"
    )
    print(
        f"TFLite bytes:    "
        f"{tflite_path.stat().st_size}"
    )
    print()
    print(
        "Parity split:    validation"
    )
    print(
        "Parity samples:  "
        f"{validation.features.shape[0]}"
    )
    print(
        "Class matches:   "
        f"{class_matches}/"
        f"{validation.features.shape[0]}"
    )
    print(
        "Class match rate:"
        f" {class_match_rate:.6f}"
    )
    print(
        "Max abs diff:    "
        f"{max_abs_difference:.10f}"
    )
    print(
        "Mean abs diff:   "
        f"{mean_abs_difference:.10f}"
    )
    print(
        f"Numeric parity:  {numeric_parity}"
    )
    print(
        f"Class parity:    {class_parity}"
    )
    print(
        f"PARITY PASSED:   {parity_passed}"
    )
    print()
    print(
        f"TFLite: {tflite_path}"
    )
    print(
        f"Report: {report_path}"
    )
    print(
        f"Vectors: {vectors_path}"
    )

    if not parity_passed:
        raise SystemExit(
            "Float32 Keras/TFLite parity failed."
        )


if __name__ == "__main__":
    main()