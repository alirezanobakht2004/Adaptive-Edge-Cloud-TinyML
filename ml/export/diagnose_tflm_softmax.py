from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.extractor import load_feature_split
from ml.models.base_model import MODEL_VERSION
from ml.export.quantize import (
    convert_to_full_int8,
    normalize_features,
)
from ml.export.tflite_export import (
    KERAS_FILENAME,
    MODEL_DIR,
    TEST_EVALUATION_FILENAME,
    load_final_test_record,
    project_root,
    sha256_file,
    verify_frozen_model,
)


NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

INT8_REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

DIAGNOSTIC_MODEL_FILENAME = (
    f"{MODEL_VERSION}-int8-logits-diagnostic.tflite"
)

DIAGNOSTIC_REPORT_FILENAME = (
    "tflm_logits_diagnostic_report.json"
)

FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-6


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_logits_model(
    source_model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Build the deployment core without Softmax.

    All frozen Dense weights are reused.
    No training is performed.
    """

    inputs = tf.keras.Input(
        shape=(10,),
        dtype=tf.float32,
        name="normalized_features_v1",
    )

    x = source_model.get_layer(
        "block1"
    )(inputs)

    x = source_model.get_layer(
        "block2"
    )(x)

    x = source_model.get_layer(
        "block3"
    )(x)

    source_head = source_model.get_layer(
        "edge_head"
    )

    logits_layer = tf.keras.layers.Dense(
        units=5,
        activation=None,
        use_bias=source_head.use_bias,
        name="edge_logits_diagnostic",
    )

    logits = logits_layer(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=logits,
        name="gesture_logits_diagnostic",
    )

    logits_layer.set_weights(
        source_head.get_weights()
    )

    return model


def create_int8_interpreter(
    model_path: Path,
) -> tuple[
    tf.lite.Interpreter,
    dict,
    dict,
]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental
            .OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )

    interpreter.allocate_tensors()

    input_detail = (
        interpreter.get_input_details()[0]
    )

    output_detail = (
        interpreter.get_output_details()[0]
    )

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic model input "
            "is not INT8."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic model output "
            "is not INT8."
        )

    return (
        interpreter,
        input_detail,
        output_detail,
    )


def quantize_input(
    values: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(
            "Invalid input quantization scale."
        )

    quantized = np.round(
        values / scale
        + zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127,
    )

    return quantized.astype(
        np.int8
    )


def run_int8_logits(
    model_path: Path,
    normalized_features: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
    dict,
]:
    (
        interpreter,
        input_detail,
        output_detail,
    ) = create_int8_interpreter(
        model_path
    )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    if input_scale <= 0.0:
        raise ValueError(
            "Invalid model input scale."
        )

    if output_scale <= 0.0:
        raise ValueError(
            "Invalid model output scale."
        )

    quantized_inputs = quantize_input(
        normalized_features,
        scale=float(input_scale),
        zero_point=int(input_zero_point),
    )

    raw_outputs = np.empty(
        (
            normalized_features.shape[0],
            5,
        ),
        dtype=np.int8,
    )

    for index, vector in enumerate(
        quantized_inputs
    ):
        interpreter.set_tensor(
            input_detail["index"],
            vector.reshape(1, 10),
        )

        interpreter.invoke()

        raw_outputs[index] = (
            interpreter.get_tensor(
                output_detail["index"]
            )
            .reshape(5)
            .astype(np.int8)
        )

    dequantized_outputs = (
        (
            raw_outputs.astype(np.float32)
            - float(output_zero_point)
        )
        * float(output_scale)
    )

    return (
        raw_outputs,
        dequantized_outputs,
        input_detail,
        output_detail,
    )


def main() -> None:
    root = project_root()

    model_dir = (
        root / MODEL_DIR
    )

    tflite_dir = (
        model_dir / "tflite"
    )

    keras_path = (
        model_dir
        / KERAS_FILENAME
    )

    final_test_path = (
        model_dir
        / TEST_EVALUATION_FILENAME
    )

    normalization_path = (
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    deployment_report_path = (
        tflite_dir
        / INT8_REPORT_FILENAME
    )

    diagnostic_model_path = (
        tflite_dir
        / DIAGNOSTIC_MODEL_FILENAME
    )

    diagnostic_report_path = (
        tflite_dir
        / DIAGNOSTIC_REPORT_FILENAME
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

    normalization_record = load_json(
        normalization_path
    )

    deployment_report = load_json(
        deployment_report_path
    )

    if (
        deployment_report.get(
            "test_split_used"
        )
        is not False
    ):
        raise ValueError(
            "Deployment report unexpectedly "
            "used the test split."
        )

    source_model = (
        tf.keras.models.load_model(
            keras_path
        )
    )

    mean = np.asarray(
        normalization_record["mean"],
        dtype=np.float32,
    )

    variance = np.asarray(
        normalization_record["variance"],
        dtype=np.float32,
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

    normalized_validation = (
        normalize_features(
            validation.features,
            mean=mean,
            variance=variance,
        )
    )

    logits_model = build_logits_model(
        source_model
    )

    #
    # Float diagnostic parity
    #

    source_probabilities = (
        source_model.predict(
            validation.features,
            verbose=0,
        )
        .astype(np.float32)
    )

    float_logits = (
        logits_model.predict(
            normalized_validation,
            verbose=0,
        )
        .astype(np.float32)
    )

    reconstructed_probabilities = (
        tf.nn.softmax(
            float_logits,
            axis=1,
        )
        .numpy()
        .astype(np.float32)
    )

    float_probability_difference = (
        np.abs(
            source_probabilities
            - reconstructed_probabilities
        )
    )

    float_max_probability_difference = (
        float(
            np.max(
                float_probability_difference
            )
        )
    )

    float_softmax_parity = bool(
        np.allclose(
            source_probabilities,
            reconstructed_probabilities,
            atol=FLOAT_ATOL,
            rtol=FLOAT_RTOL,
        )
    )

    source_classes = np.argmax(
        source_probabilities,
        axis=1,
    )

    float_logits_classes = np.argmax(
        float_logits,
        axis=1,
    )

    float_class_matches = int(
        np.sum(
            source_classes
            == float_logits_classes
        )
    )

    if not float_softmax_parity:
        raise RuntimeError(
            "Diagnostic float logits do not "
            "reconstruct the frozen source model."
        )

    if (
        float_class_matches
        != validation.features.shape[0]
    ):
        raise RuntimeError(
            "Diagnostic float logits changed "
            "the frozen source class prediction."
        )

    #
    # Quantize diagnostic logits model
    #

    diagnostic_bytes = (
        convert_to_full_int8(
            logits_model,
            normalized_train,
        )
    )

    diagnostic_model_path.write_bytes(
        diagnostic_bytes
    )

    diagnostic_hash = sha256_file(
        diagnostic_model_path
    )

    (
        raw_int8_logits,
        dequantized_int8_logits,
        input_detail,
        output_detail,
    ) = run_int8_logits(
        diagnostic_model_path,
        normalized_validation,
    )

    int8_logits_classes = np.argmax(
        raw_int8_logits,
        axis=1,
    )

    int8_class_matches = int(
        np.sum(
            int8_logits_classes
            == source_classes
        )
    )

    int8_correct = int(
        np.sum(
            int8_logits_classes
            == validation.labels
        )
    )

    int8_accuracy = float(
        int8_correct
        / validation.labels.shape[0]
    )

    source_correct = int(
        np.sum(
            source_classes
            == validation.labels
        )
    )

    source_accuracy = float(
        source_correct
        / validation.labels.shape[0]
    )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    report = {
        "purpose":
            "diagnose TFLM Dense stack vs Softmax",
        "model_version":
            MODEL_VERSION,
        "source_model_sha256":
            source_hash,
        "diagnostic_model_sha256":
            diagnostic_hash,
        "diagnostic_model_bytes":
            diagnostic_model_path.stat().st_size,
        "source_split":
            "validation",
        "representative_split":
            "train",
        "test_split_used":
            False,
        "float_softmax_parity":
            float_softmax_parity,
        "float_max_probability_difference":
            float_max_probability_difference,
        "float_class_matches":
            float_class_matches,
        "source_validation_accuracy":
            source_accuracy,
        "int8_logits_validation_accuracy":
            int8_accuracy,
        "int8_logits_class_matches_source":
            int8_class_matches,
        "input_quantization": {
            "scale":
                float(input_scale),
            "zero_point":
                int(input_zero_point),
        },
        "logits_output_quantization": {
            "scale":
                float(output_scale),
            "zero_point":
                int(output_zero_point),
        },
        "vector_82": {
            "true_class":
                int(validation.labels[82]),
            "source_class":
                int(source_classes[82]),
            "int8_logits_class":
                int(int8_logits_classes[82]),
            "raw_int8_logits":
                raw_int8_logits[
                    82
                ].astype(int).tolist(),
            "dequantized_logits":
                dequantized_int8_logits[
                    82
                ].astype(float).tolist(),
        },
        "vector_119": {
            "true_class":
                int(validation.labels[119]),
            "source_class":
                int(source_classes[119]),
            "int8_logits_class":
                int(int8_logits_classes[119]),
            "raw_int8_logits":
                raw_int8_logits[
                    119
                ].astype(int).tolist(),
            "dequantized_logits":
                dequantized_int8_logits[
                    119
                ].astype(float).tolist(),
        },
    }

    with diagnostic_report_path.open(
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
        "TFLM SOFTMAX ISOLATION - DESKTOP"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Frozen source SHA: "
        f"{source_hash}"
    )

    print(
        f"Diagnostic SHA:    "
        f"{diagnostic_hash}"
    )

    print(
        f"Diagnostic bytes:  "
        f"{diagnostic_model_path.stat().st_size}"
    )

    print()

    print(
        "Float softmax reconstruction parity: "
        f"{float_softmax_parity}"
    )

    print(
        "Float max probability diff: "
        f"{float_max_probability_difference:.10f}"
    )

    print(
        "Float class matches: "
        f"{float_class_matches}/"
        f"{validation.features.shape[0]}"
    )

    print()

    print(
        "Frozen source validation accuracy: "
        f"{source_accuracy:.6f} "
        f"({source_correct}/"
        f"{validation.labels.shape[0]})"
    )

    print(
        "INT8 logits validation accuracy:    "
        f"{int8_accuracy:.6f} "
        f"({int8_correct}/"
        f"{validation.labels.shape[0]})"
    )

    print(
        "INT8 logits class matches source:   "
        f"{int8_class_matches}/"
        f"{validation.features.shape[0]}"
    )

    print()

    print(
        f"Input scale:      "
        f"{float(input_scale):.12g}"
    )

    print(
        f"Input zero point: "
        f"{int(input_zero_point)}"
    )

    print(
        f"Logits scale:     "
        f"{float(output_scale):.12g}"
    )

    print(
        f"Logits zero point:"
        f" {int(output_zero_point)}"
    )

    print()

    for index in (
        82,
        119,
    ):
        print(
            f"VECTOR {index}"
        )

        print(
            f"  true="
            f"{int(validation.labels[index])}"
        )

        print(
            f"  source_class="
            f"{int(source_classes[index])}"
        )

        print(
            f"  int8_logits_class="
            f"{int(int8_logits_classes[index])}"
        )

        print(
            "  raw_int8_logits="
            f"{raw_int8_logits[index].tolist()}"
        )

    print()

    print(
        "Representative split: train"
    )

    print(
        "Evaluation split:     validation"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        f"Diagnostic model: "
        f"{diagnostic_model_path}"
    )

    print(
        f"Report:           "
        f"{diagnostic_report_path}"
    )


if __name__ == "__main__":
    main()