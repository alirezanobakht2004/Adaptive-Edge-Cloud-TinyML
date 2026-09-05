from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.extractor import load_feature_split
from ml.export.quantize import (
    normalize_features,
    quantize_tensor,
)
from ml.export.tflite_export import (
    MODEL_DIR,
    project_root,
    sha256_file,
)


MODEL_FILENAME = (
    "gesture-model-v1.0.0-"
    "int8-normalized-input.tflite"
)

REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def create_interpreter(
    model_path: Path,
    resolver_type,
) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=resolver_type,
    )

    interpreter.allocate_tensors()

    return interpreter


def run_interpreter(
    interpreter: tf.lite.Interpreter,
    quantized_inputs: np.ndarray,
) -> np.ndarray:
    input_detail = (
        interpreter.get_input_details()[0]
    )

    output_detail = (
        interpreter.get_output_details()[0]
    )

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            "Expected INT8 input tensor."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Expected INT8 output tensor."
        )

    outputs = np.empty(
        (
            quantized_inputs.shape[0],
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

        outputs[index] = (
            interpreter.get_tensor(
                output_detail["index"]
            )
            .reshape(5)
            .astype(np.int8)
        )

    return outputs


def accuracy(
    outputs: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, float]:
    predictions = np.argmax(
        outputs,
        axis=1,
    )

    correct = int(
        np.sum(
            predictions
            == labels
        )
    )

    return (
        correct,
        correct / labels.shape[0],
    )


def compare(
    name_a: str,
    outputs_a: np.ndarray,
    name_b: str,
    outputs_b: np.ndarray,
) -> None:
    predictions_a = np.argmax(
        outputs_a,
        axis=1,
    )

    predictions_b = np.argmax(
        outputs_b,
        axis=1,
    )

    class_matches = (
        predictions_a
        == predictions_b
    )

    class_match_count = int(
        np.sum(class_matches)
    )

    differences = np.abs(
        outputs_a.astype(np.int16)
        - outputs_b.astype(np.int16)
    )

    max_difference = int(
        np.max(differences)
    )

    mean_difference = float(
        np.mean(differences)
    )

    mismatch_indices = np.flatnonzero(
        ~class_matches
    )

    print()
    print(
        f"{name_a} vs {name_b}"
    )

    print(
        "-" * (
            len(name_a)
            + len(name_b)
            + 4
        )
    )

    print(
        "Class matches: "
        f"{class_match_count}/"
        f"{outputs_a.shape[0]}"
    )

    print(
        f"Max INT8 LSB diff: "
        f"{max_difference}"
    )

    print(
        f"Mean INT8 LSB diff: "
        f"{mean_difference:.6f}"
    )

    if mismatch_indices.size == 0:
        print(
            "Class mismatch indices: none"
        )
    else:
        print(
            "Class mismatch indices: "
            + ", ".join(
                str(int(index))
                for index
                in mismatch_indices
            )
        )

        for index in mismatch_indices:
            print(
                f"  vector={int(index)} "
                f"{name_a}="
                f"{int(predictions_a[index])} "
                f"{name_b}="
                f"{int(predictions_b[index])}"
            )


def print_vector(
    index: int,
    labels: np.ndarray,
    auto_outputs: np.ndarray,
    builtin_outputs: np.ndarray,
    reference_outputs: np.ndarray,
) -> None:
    print()
    print(
        f"VECTOR {index}"
    )

    print(
        f"Ground truth: "
        f"{int(labels[index])}"
    )

    print(
        "AUTO/XNNPACK: "
        f"{auto_outputs[index].tolist()} "
        f"class="
        f"{int(np.argmax(auto_outputs[index]))}"
    )

    print(
        "BUILTIN/no delegate: "
        f"{builtin_outputs[index].tolist()} "
        f"class="
        f"{int(np.argmax(builtin_outputs[index]))}"
    )

    print(
        "BUILTIN_REF: "
        f"{reference_outputs[index].tolist()} "
        f"class="
        f"{int(np.argmax(reference_outputs[index]))}"
    )


def main() -> None:
    root = project_root()

    tflite_dir = (
        root
        / MODEL_DIR
        / "tflite"
    )

    model_path = (
        tflite_dir
        / MODEL_FILENAME
    )

    report = load_json(
        tflite_dir
        / REPORT_FILENAME
    )

    normalization = load_json(
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    expected_hash = report[
        "int8_tflite_sha256"
    ]

    actual_hash = sha256_file(
        model_path
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen INT8 model SHA mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    if report.get(
        "test_split_used"
    ) is not False:
        raise ValueError(
            "INT8 report unexpectedly "
            "used the held-out test split."
        )

    validation = load_feature_split(
        "validation"
    )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    variance = np.asarray(
        normalization["variance"],
        dtype=np.float32,
    )

    normalized = normalize_features(
        validation.features,
        mean=mean,
        variance=variance,
    )

    input_scale = float(
        report[
            "input_quantization"
        ]["scale"]
    )

    input_zero_point = int(
        report[
            "input_quantization"
        ]["zero_point"]
    )

    quantized_inputs = quantize_tensor(
        normalized,
        scale=input_scale,
        zero_point=input_zero_point,
        dtype=np.int8,
    )

    resolver = (
        tf.lite.experimental
        .OpResolverType
    )

    print()
    print(
        "Creating AUTO interpreter..."
    )

    auto_interpreter = (
        create_interpreter(
            model_path,
            resolver.AUTO,
        )
    )

    print(
        "Creating BUILTIN/no-delegate "
        "interpreter..."
    )

    builtin_interpreter = (
        create_interpreter(
            model_path,
            resolver
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    )

    print(
        "Creating BUILTIN_REF "
        "interpreter..."
    )

    reference_interpreter = (
        create_interpreter(
            model_path,
            resolver.BUILTIN_REF,
        )
    )

    auto_outputs = run_interpreter(
        auto_interpreter,
        quantized_inputs,
    )

    builtin_outputs = run_interpreter(
        builtin_interpreter,
        quantized_inputs,
    )

    reference_outputs = run_interpreter(
        reference_interpreter,
        quantized_inputs,
    )

    labels = (
        validation.labels
        .astype(np.int64)
    )

    auto_correct, auto_accuracy = (
        accuracy(
            auto_outputs,
            labels,
        )
    )

    builtin_correct, builtin_accuracy = (
        accuracy(
            builtin_outputs,
            labels,
        )
    )

    reference_correct, reference_accuracy = (
        accuracy(
            reference_outputs,
            labels,
        )
    )

    print()
    print(
        "TFLITE RUNTIME DIAGNOSTIC"
    )

    print(
        "-------------------------"
    )

    print(
        f"Model SHA-256: "
        f"{actual_hash}"
    )

    print(
        "Source: validation / "
        f"{validation.session}"
    )

    print(
        f"Vectors: "
        f"{validation.features.shape[0]}"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        "AUTO accuracy: "
        f"{auto_correct}/"
        f"{labels.shape[0]} "
        f"({auto_accuracy:.6f})"
    )

    print(
        "BUILTIN/no-delegate accuracy: "
        f"{builtin_correct}/"
        f"{labels.shape[0]} "
        f"({builtin_accuracy:.6f})"
    )

    print(
        "BUILTIN_REF accuracy: "
        f"{reference_correct}/"
        f"{labels.shape[0]} "
        f"({reference_accuracy:.6f})"
    )

    compare(
        "AUTO",
        auto_outputs,
        "BUILTIN",
        builtin_outputs,
    )

    compare(
        "AUTO",
        auto_outputs,
        "BUILTIN_REF",
        reference_outputs,
    )

    compare(
        "BUILTIN",
        builtin_outputs,
        "BUILTIN_REF",
        reference_outputs,
    )

    # Vector 82 produced the ESP32 Top-1 mismatch.
    print_vector(
        82,
        labels,
        auto_outputs,
        builtin_outputs,
        reference_outputs,
    )

    # Vector 119 produced the largest measured
    # ESP32-vs-desktop output difference.
    print_vector(
        119,
        labels,
        auto_outputs,
        builtin_outputs,
        reference_outputs,
    )


if __name__ == "__main__":
    main()