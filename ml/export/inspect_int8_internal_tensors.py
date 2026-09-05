from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.export.quantize import (
    normalize_features,
    quantize_tensor,
)
from ml.export.tflite_export import (
    MODEL_DIR,
    project_root,
    sha256_file,
)
from ml.features.extractor import load_feature_split


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

OUTPUT_REPORT_FILENAME = (
    "int8_internal_tensor_report.json"
)

PROBE_INDICES = (
    82,
    119,
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_intermediate_tensor(
    interpreter: tf.lite.Interpreter,
    tensor_index: int,
) -> np.ndarray:
    try:
        return interpreter.get_tensor(
            tensor_index
        ).copy()
    except (ValueError, RuntimeError):
        tensor_accessor = interpreter.tensor(
            tensor_index
        )

        return np.asarray(
            tensor_accessor()
        ).copy()


def byte_hash(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values).tobytes()
    ).hexdigest()


def quantization_record(
    detail: dict,
) -> dict:
    scale, zero_point = detail[
        "quantization"
    ]

    return {
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


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

    report_path = (
        tflite_dir
        / REPORT_FILENAME
    )

    normalization_path = (
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    output_report_path = (
        tflite_dir
        / OUTPUT_REPORT_FILENAME
    )

    report = load_json(
        report_path
    )

    normalization = load_json(
        normalization_path
    )

    if report.get(
        "test_split_used"
    ) is not False:
        raise ValueError(
            "INT8 report unexpectedly used "
            "the held-out test split."
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

    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected validation/session_02, "
            f"got {validation.session!r}."
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

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental
            .OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
        experimental_preserve_all_tensors=True,
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
            "Expected INT8 input."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Expected INT8 output."
        )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    quantized_inputs = quantize_tensor(
        normalized,
        scale=float(input_scale),
        zero_point=int(input_zero_point),
        dtype=np.int8,
    )

    tensor_details = {
        int(detail["index"]): detail
        for detail
        in interpreter.get_tensor_details()
    }

    # This is an internal TensorFlow diagnostic API.
    # It is deliberately isolated to this debugging script.
    ops = interpreter._get_ops_details()

    interesting_ops: list[dict] = []

    fully_connected_count = 0
    softmax_count = 0

    for op in ops:
        op_name = str(
            op["op_name"]
        )

        if op_name not in (
            "FULLY_CONNECTED",
            "SOFTMAX",
        ):
            continue

        output_indices = [
            int(value)
            for value in op["outputs"]
            if int(value) >= 0
        ]

        if len(output_indices) != 1:
            raise RuntimeError(
                f"{op_name} has unexpected "
                f"outputs: {output_indices}"
            )

        tensor_index = (
            output_indices[0]
        )

        detail = tensor_details.get(
            tensor_index
        )

        if detail is None:
            raise RuntimeError(
                f"No tensor detail for "
                f"tensor {tensor_index}."
            )

        if op_name == "FULLY_CONNECTED":
            fully_connected_count += 1

            stage_name = (
                f"FC{fully_connected_count}"
            )

        else:
            softmax_count += 1

            stage_name = "SOFTMAX"

        interesting_ops.append(
            {
                "stage":
                    stage_name,
                "op_index":
                    int(op["index"]),
                "op_name":
                    op_name,
                "tensor_index":
                    tensor_index,
                "tensor_name":
                    str(detail["name"]),
                "shape":
                    [
                        int(value)
                        for value
                        in detail["shape"]
                    ],
                "dtype":
                    str(detail["dtype"]),
                "quantization":
                    quantization_record(
                        detail
                    ),
            }
        )

    if fully_connected_count != 4:
        raise RuntimeError(
            "Expected exactly four "
            "FULLY_CONNECTED ops, got "
            f"{fully_connected_count}."
        )

    if softmax_count != 1:
        raise RuntimeError(
            "Expected exactly one SOFTMAX op, "
            f"got {softmax_count}."
        )

    diagnostic_vectors: list[dict] = []

    for vector_index in PROBE_INDICES:
        interpreter.set_tensor(
            input_detail["index"],
            quantized_inputs[
                vector_index
            ].reshape(1, 10),
        )

        interpreter.invoke()

        vector_record = {
            "validation_index":
                vector_index,
            "true_class":
                int(
                    validation.labels[
                        vector_index
                    ]
                ),
            "input_int8":
                quantized_inputs[
                    vector_index
                ].astype(int).tolist(),
            "stages": [],
        }

        for stage in interesting_ops:
            tensor_index = int(
                stage["tensor_index"]
            )

            values = (
                get_intermediate_tensor(
                    interpreter,
                    tensor_index,
                )
                .reshape(-1)
            )

            vector_record[
                "stages"
            ].append(
                {
                    "stage":
                        stage["stage"],
                    "tensor_index":
                        tensor_index,
                    "value_count":
                        int(values.size),
                    "values":
                        values.astype(
                            int
                        ).tolist(),
                    "min":
                        int(
                            np.min(values)
                        ),
                    "max":
                        int(
                            np.max(values)
                        ),
                    "sha256":
                        byte_hash(values),
                }
            )

        diagnostic_vectors.append(
            vector_record
        )

    result = {
        "model_sha256":
            actual_hash,
        "source_split":
            "validation",
        "source_session":
            validation.session,
        "test_split_used":
            False,
        "operators":
            interesting_ops,
        "probe_vectors":
            diagnostic_vectors,
    }

    with output_report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print(
        "FROZEN INT8 INTERNAL TENSOR MAP"
    )
    print(
        "-------------------------------"
    )

    print(
        f"Model SHA-256: {actual_hash}"
    )

    print(
        "Source:        validation / "
        f"{validation.session}"
    )

    print(
        "Test split was not loaded."
    )

    print()

    for stage in interesting_ops:
        print(
            f"{stage['stage']:<7} "
            f"op={stage['op_index']:<3} "
            f"tensor={stage['tensor_index']:<3} "
            f"shape={stage['shape']} "
            f"dtype={stage['dtype']} "
            f"scale="
            f"{stage['quantization']['scale']:.12g} "
            f"zp="
            f"{stage['quantization']['zero_point']}"
        )

        print(
            f"        name="
            f"{stage['tensor_name']}"
        )

    for vector in diagnostic_vectors:
        print()

        print(
            f"VECTOR "
            f"{vector['validation_index']} "
            f"true={vector['true_class']}"
        )

        for stage in vector["stages"]:
            values = stage["values"]

            preview = (
                values
                if len(values) <= 10
                else values[:10]
            )

            print(
                f"  {stage['stage']:<7} "
                f"tensor={stage['tensor_index']:<3} "
                f"n={stage['value_count']:<3} "
                f"min={stage['min']:<4} "
                f"max={stage['max']:<4} "
                f"first={preview}"
            )

            print(
                "          sha256="
                f"{stage['sha256']}"
            )

    print()

    print(
        f"Report: {output_report_path}"
    )


if __name__ == "__main__":
    main()