from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import GESTURES
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import MODEL_VERSION
from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)
from ml.export.tflite_export import (
    DATASET_VERSION,
    EXPECTED_CLASSES,
    EXPECTED_INPUT_FEATURES,
    KERAS_FILENAME,
    MODEL_DIR,
    TEST_EVALUATION_FILENAME,
    load_final_test_record,
    project_root,
    sha256_file,
    verify_frozen_model,
)


TFLITE_DIRNAME = "tflite"

INT8_FILENAME = (
    f"{MODEL_VERSION}-int8-normalized-input.tflite"
)

INT8_REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

FLOAT_CORE_ATOL = 1e-6
FLOAT_CORE_RTOL = 1e-6


def get_normalization_parameters(
    model: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = model.get_layer(
        "feature_normalization"
    )

    mean = (
        layer.mean.numpy()
        .reshape(-1)
        .astype(np.float32)
    )

    variance = (
        layer.variance.numpy()
        .reshape(-1)
        .astype(np.float32)
    )

    if mean.shape != (EXPECTED_INPUT_FEATURES,):
        raise ValueError(
            f"Unexpected normalization mean shape: "
            f"{mean.shape}"
        )

    if variance.shape != (
        EXPECTED_INPUT_FEATURES,
    ):
        raise ValueError(
            "Unexpected normalization variance "
            f"shape: {variance.shape}"
        )

    if not np.isfinite(mean).all():
        raise ValueError(
            "Normalization mean contains "
            "non-finite values."
        )

    if (
        not np.isfinite(variance).all()
        or np.any(variance <= 0.0)
    ):
        raise ValueError(
            "Normalization variance must "
            "be finite and positive."
        )

    std = np.sqrt(variance).astype(
        np.float32
    )

    return mean, variance, std


def normalize_features(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        features,
        dtype=np.float32,
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    variance = np.asarray(
        variance,
        dtype=np.float32,
    )

    if x.shape[-1] != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_FEATURES} "
            f"features, got {x.shape}."
        )

    if mean.shape != (
        EXPECTED_INPUT_FEATURES,
    ):
        raise ValueError(
            f"Unexpected mean shape: {mean.shape}"
        )

    if variance.shape != (
        EXPECTED_INPUT_FEATURES,
    ):
        raise ValueError(
            f"Unexpected variance shape: "
            f"{variance.shape}"
        )

    if np.any(variance <= 0.0):
        raise ValueError(
            "Variance must be positive."
        )

    normalized = (
        x - mean
    ) / np.sqrt(variance)

    normalized = normalized.astype(
        np.float32
    )

    if not np.isfinite(normalized).all():
        raise ValueError(
            "Normalization produced "
            "non-finite values."
        )

    return normalized


def build_normalized_input_model(
    source_model: tf.keras.Model,
) -> tf.keras.Model:
    """Reuse frozen Dense weights without the Normalization layer."""

    inputs = tf.keras.Input(
        shape=(EXPECTED_INPUT_FEATURES,),
        dtype=tf.float32,
        name="normalized_features_v1",
    )

    x = inputs

    for layer_name in (
        "block1",
        "block2",
        "block3",
        "edge_head",
    ):
        layer = source_model.get_layer(
            layer_name
        )
        x = layer(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name="gesture_deployment_core",
    )


def representative_dataset(
    normalized_train: np.ndarray,
):
    for vector in normalized_train:
        yield [
            vector.reshape(
                1,
                EXPECTED_INPUT_FEATURES,
            ).astype(np.float32)
        ]


def convert_to_full_int8(
    model: tf.keras.Model,
    normalized_train: np.ndarray,
) -> bytes:
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda: representative_dataset(
            normalized_train
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    result = converter.convert()

    if not result:
        raise RuntimeError(
            "INT8 converter returned "
            "an empty model."
        )

    return result


def quantize_tensor(
    values: np.ndarray,
    *,
    scale: float,
    zero_point: int,
    dtype=np.int8,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(
            f"Invalid quantization scale: {scale}"
        )

    dtype = np.dtype(dtype)
    limits = np.iinfo(dtype)

    q = np.round(
        np.asarray(
            values,
            dtype=np.float32,
        )
        / scale
        + zero_point
    )

    q = np.clip(
        q,
        limits.min,
        limits.max,
    )

    return q.astype(dtype)


def dequantize_tensor(
    values: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(
            f"Invalid quantization scale: {scale}"
        )

    return (
        np.asarray(
            values,
            dtype=np.float32,
        )
        - float(zero_point)
    ) * float(scale)


def run_int8_inference(
    model_path: Path,
    normalized_features: np.ndarray,
) -> tuple[np.ndarray, dict, dict]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Expected one input and one output."
        )

    input_detail = inputs[0]
    output_detail = outputs[0]

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            "INT8 model input is not int8."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "INT8 model output is not int8."
        )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    probabilities = np.empty(
        (
            normalized_features.shape[0],
            EXPECTED_CLASSES,
        ),
        dtype=np.float32,
    )

    for index, vector in enumerate(
        normalized_features
    ):
        model_input = quantize_tensor(
            vector.reshape(
                1,
                EXPECTED_INPUT_FEATURES,
            ),
            scale=float(input_scale),
            zero_point=int(input_zero_point),
        )

        interpreter.set_tensor(
            input_detail["index"],
            model_input,
        )

        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            output_detail["index"]
        )

        probabilities[index] = (
            dequantize_tensor(
                raw_output,
                scale=float(output_scale),
                zero_point=int(
                    output_zero_point
                ),
            ).reshape(
                EXPECTED_CLASSES
            )
        )

    return (
        probabilities,
        input_detail,
        output_detail,
    )


def main() -> None:
    root = project_root()
    model_dir = root / MODEL_DIR
    tflite_dir = (
        model_dir / TFLITE_DIRNAME
    )

    tflite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    keras_path = (
        model_dir / KERAS_FILENAME
    )

    final_test_path = (
        model_dir
        / TEST_EVALUATION_FILENAME
    )

    int8_path = (
        tflite_dir / INT8_FILENAME
    )

    report_path = (
        tflite_dir
        / INT8_REPORT_FILENAME
    )

    normalization_path = (
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    final_test_record = (
        load_final_test_record(
            final_test_path
        )
    )

    source_hash = verify_frozen_model(
        keras_path,
        final_test_record,
    )

    source_model = (
        tf.keras.models.load_model(
            keras_path
        )
    )

    mean, variance, std = (
        get_normalization_parameters(
            source_model
        )
    )

    train = load_feature_split(
        "train"
    )

    validation = load_feature_split(
        "validation"
    )

    normalized_train = normalize_features(
        train.features,
        mean=mean,
        variance=variance,
    )

    normalized_validation = normalize_features(
        validation.features,
        mean=mean,
        variance=variance,
    )

    # Verify our explicit preprocessing exactly reproduces
    # the frozen Keras Normalization layer.
    normalization_layer = (
        source_model.get_layer(
            "feature_normalization"
        )
    )

    keras_normalized_validation = (
        normalization_layer(
            validation.features
        )
        .numpy()
        .astype(np.float32)
    )

    normalization_max_abs_diff = float(
        np.max(
            np.abs(
                normalized_validation
                - keras_normalized_validation
            )
        )
    )

    normalization_parity = bool(
        np.allclose(
            normalized_validation,
            keras_normalized_validation,
            rtol=FLOAT_CORE_RTOL,
            atol=FLOAT_CORE_ATOL,
        )
    )

    if not normalization_parity:
        raise RuntimeError(
            "Explicit normalization does not "
            "match the frozen Keras layer."
        )

    core_model = (
        build_normalized_input_model(
            source_model
        )
    )

    source_probabilities = (
        source_model.predict(
            validation.features,
            verbose=0,
        ).astype(np.float32)
    )

    core_probabilities = (
        core_model.predict(
            normalized_validation,
            verbose=0,
        ).astype(np.float32)
    )

    core_max_abs_diff = float(
        np.max(
            np.abs(
                source_probabilities
                - core_probabilities
            )
        )
    )

    core_parity = bool(
        np.allclose(
            source_probabilities,
            core_probabilities,
            rtol=FLOAT_CORE_RTOL,
            atol=FLOAT_CORE_ATOL,
        )
    )

    source_predictions = np.argmax(
        source_probabilities,
        axis=1,
    )

    core_predictions = np.argmax(
        core_probabilities,
        axis=1,
    )

    core_class_matches = int(
        np.sum(
            source_predictions
            == core_predictions
        )
    )

    if (
        not core_parity
        or core_class_matches
        != validation.features.shape[0]
    ):
        raise RuntimeError(
            "Normalized-input float core "
            "does not match frozen Keras model."
        )

    int8_bytes = convert_to_full_int8(
        core_model,
        normalized_train,
    )

    int8_path.write_bytes(
        int8_bytes
    )

    int8_hash = sha256_file(
        int8_path
    )

    (
        int8_probabilities,
        input_detail,
        output_detail,
    ) = run_int8_inference(
        int8_path,
        normalized_validation,
    )

    int8_predictions = np.argmax(
        int8_probabilities,
        axis=1,
    ).astype(np.int64)

    keras_matrix = confusion_matrix(
        validation.labels,
        source_predictions,
        class_count=len(GESTURES),
    )

    int8_matrix = confusion_matrix(
        validation.labels,
        int8_predictions,
        class_count=len(GESTURES),
    )

    keras_metrics = (
        classification_metrics(
            keras_matrix
        )
    )

    int8_metrics = (
        classification_metrics(
            int8_matrix
        )
    )

    class_matches = int(
        np.sum(
            source_predictions
            == int8_predictions
        )
    )

    class_match_rate = float(
        class_matches
        / validation.features.shape[0]
    )

    probability_difference = np.abs(
        source_probabilities
        - int8_probabilities
    )

    max_abs_probability_difference = float(
        np.max(
            probability_difference
        )
    )

    mean_abs_probability_difference = float(
        np.mean(
            probability_difference
        )
    )

    accuracy_delta = float(
        int8_metrics["accuracy"]
        - keras_metrics["accuracy"]
    )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    normalization_record = {
        "dataset_version":
            DATASET_VERSION,
        "feature_version":
            FEATURE_VERSION,
        "model_version":
            MODEL_VERSION,
        "fit_split":
            "train",
        "formula":
            "(x - mean) / sqrt(variance)",
        "mean":
            mean.astype(float).tolist(),
        "variance":
            variance.astype(float).tolist(),
        "std":
            std.astype(float).tolist(),
    }

    with normalization_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            normalization_record,
            file,
            indent=2,
        )

    report = {
        "model_version":
            MODEL_VERSION,
        "dataset_version":
            DATASET_VERSION,
        "feature_version":
            FEATURE_VERSION,
        "export_type":
            "full-integer-int8",
        "deployment_input":
            "train-standardized-features-v1",
        "source_model_sha256":
            source_hash,
        "int8_tflite_sha256":
            int8_hash,
        "int8_model_bytes":
            int8_path.stat().st_size,
        "representative_split":
            "train",
        "evaluation_split":
            "validation",
        "test_split_used":
            False,
        "normalization_parity": {
            "passed":
                normalization_parity,
            "max_abs_difference":
                normalization_max_abs_diff,
        },
        "float_core_parity": {
            "passed":
                core_parity,
            "class_matches":
                core_class_matches,
            "max_abs_probability_difference":
                core_max_abs_diff,
        },
        "input_quantization": {
            "dtype":
                str(input_detail["dtype"]),
            "scale":
                float(input_scale),
            "zero_point":
                int(input_zero_point),
        },
        "output_quantization": {
            "dtype":
                str(output_detail["dtype"]),
            "scale":
                float(output_scale),
            "zero_point":
                int(output_zero_point),
        },
        "keras_validation_metrics":
            keras_metrics,
        "int8_validation_metrics":
            int8_metrics,
        "accuracy_delta":
            accuracy_delta,
        "keras_int8_class_matches":
            class_matches,
        "keras_int8_class_match_rate":
            class_match_rate,
        "max_abs_probability_difference":
            max_abs_probability_difference,
        "mean_abs_probability_difference":
            mean_abs_probability_difference,
        "int8_confusion_matrix":
            int8_matrix.tolist(),
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
    print(
        "INT8 EXPORT WITH EXTERNAL NORMALIZATION"
    )
    print(
        "---------------------------------------"
    )

    print(
        f"Model version:   {MODEL_VERSION}"
    )
    print(
        f"Source SHA-256: {source_hash}"
    )
    print(
        f"INT8 SHA-256:   {int8_hash}"
    )

    print()
    print(
        "Normalization parity:"
        f" {normalization_parity}"
    )
    print(
        "Normalization max Δ: "
        f"{normalization_max_abs_diff:.10f}"
    )

    print()
    print(
        f"Float core parity: {core_parity}"
    )
    print(
        "Float class matches: "
        f"{core_class_matches}/"
        f"{validation.features.shape[0]}"
    )
    print(
        "Float core max Δ: "
        f"{core_max_abs_diff:.10f}"
    )

    print()
    print(
        f"INT8 bytes:      "
        f"{int8_path.stat().st_size}"
    )

    print(
        f"Input scale:     "
        f"{float(input_scale):.12g}"
    )
    print(
        f"Input zero point:"
        f" {int(input_zero_point)}"
    )

    print(
        f"Output scale:    "
        f"{float(output_scale):.12g}"
    )
    print(
        f"Output zero point:"
        f" {int(output_zero_point)}"
    )

    print()
    print(
        "Representative:  "
        f"train ({train.features.shape[0]})"
    )
    print(
        "Evaluation:      "
        f"validation "
        f"({validation.features.shape[0]})"
    )
    print(
        "Test split was not loaded."
    )

    print()
    print(
        "Keras accuracy:  "
        f"{keras_metrics['accuracy']:.6f}"
    )
    print(
        "INT8 accuracy:   "
        f"{int8_metrics['accuracy']:.6f}"
    )
    print(
        "Accuracy delta:  "
        f"{accuracy_delta:+.6f}"
    )

    print()
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
        "Max abs prob Δ:  "
        f"{max_abs_probability_difference:.10f}"
    )
    print(
        "Mean abs prob Δ: "
        f"{mean_abs_probability_difference:.10f}"
    )

    print()
    print(
        "INT8 validation confusion matrix:"
    )
    print(
        int8_matrix
    )

    print()
    print(
        f"INT8 model:    {int8_path}"
    )
    print(
        f"Normalization: {normalization_path}"
    )
    print(
        f"Report:        {report_path}"
    )


if __name__ == "__main__":
    main()