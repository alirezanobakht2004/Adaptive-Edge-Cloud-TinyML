from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.extractor import load_feature_split
from ml.export.quantize import normalize_features
from ml.export.tflite_export import (
    MODEL_DIR,
    project_root,
    sha256_file,
)


DIAGNOSTIC_MODEL_FILENAME = (
    "gesture-model-v1.0.0-"
    "int8-logits-diagnostic.tflite"
)

DIAGNOSTIC_REPORT_FILENAME = (
    "tflm_logits_diagnostic_report.json"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

TEST_HEADER = Path(
    "firmware/test/"
    "test_tflm_logits_diagnostic/"
    "tflm_logits_diagnostic_vectors.h"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def quantize_tensor(
    values: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            f"Invalid quantization scale: {scale}"
        )

    quantized = np.round(
        np.asarray(
            values,
            dtype=np.float32,
        )
        / scale
        + zero_point
    )

    return np.clip(
        quantized,
        -128,
        127,
    ).astype(np.int8)


def run_desktop_model(
    *,
    model_path: Path,
    quantized_inputs: np.ndarray,
) -> tuple[
    np.ndarray,
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

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1:
        raise ValueError(
            f"Expected one input tensor, got {len(inputs)}."
        )

    if len(outputs) != 1:
        raise ValueError(
            f"Expected one output tensor, got {len(outputs)}."
        )

    input_detail = inputs[0]
    output_detail = outputs[0]

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic model input is not INT8."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic model output is not INT8."
        )

    raw_outputs = np.empty(
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

        raw_outputs[index] = (
            interpreter.get_tensor(
                output_detail["index"]
            )
            .reshape(5)
            .astype(np.int8)
        )

    return (
        raw_outputs,
        input_detail,
        output_detail,
    )


def write_header(
    *,
    path: Path,
    model_bytes: bytes,
    model_hash: str,
    quantized_inputs: np.ndarray,
    raw_outputs: np.ndarray,
    expected_classes: np.ndarray,
    true_classes: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "namespace tflm_logits_diagnostic {",
        "",
        f'constexpr char MODEL_SHA256[] = "{model_hash}";',
        "",
        (
            "constexpr size_t MODEL_LEN = "
            f"{len(model_bytes)};"
        ),
        "constexpr size_t VECTOR_COUNT = 200;",
        "constexpr size_t INPUT_COUNT = 10;",
        "constexpr size_t OUTPUT_COUNT = 5;",
        "",
        (
            "constexpr float INPUT_SCALE = "
            f"{float(np.float32(input_scale)):.9g}f;"
        ),
        (
            "constexpr int32_t INPUT_ZERO_POINT = "
            f"{input_zero_point};"
        ),
        (
            "constexpr float OUTPUT_SCALE = "
            f"{float(np.float32(output_scale)):.9g}f;"
        ),
        (
            "constexpr int32_t OUTPUT_ZERO_POINT = "
            f"{output_zero_point};"
        ),
        "",
        "alignas(16) static const unsigned char MODEL[MODEL_LEN] = {",
    ]

    for start in range(
        0,
        len(model_bytes),
        12,
    ):
        chunk = model_bytes[
            start:start + 12
        ]

        values = ", ".join(
            f"0x{value:02x}"
            for value in chunk
        )

        lines.append(
            f"    {values},"
        )

    lines.extend(
        [
            "};",
            "",
            (
                "static const int8_t INPUTS"
                "[VECTOR_COUNT][INPUT_COUNT] = {"
            ),
        ]
    )

    for row in quantized_inputs:
        values = ", ".join(
            str(int(value))
            for value in row
        )

        lines.append(
            f"    {{{values}}},"
        )

    lines.extend(
        [
            "};",
            "",
            (
                "static const int8_t EXPECTED_LOGITS"
                "[VECTOR_COUNT][OUTPUT_COUNT] = {"
            ),
        ]
    )

    for row in raw_outputs:
        values = ", ".join(
            str(int(value))
            for value in row
        )

        lines.append(
            f"    {{{values}}},"
        )

    lines.extend(
        [
            "};",
            "",
            (
                "static const int EXPECTED_CLASSES"
                "[VECTOR_COUNT] = {"
            ),
            "    "
            + ", ".join(
                str(int(value))
                for value in expected_classes
            ),
            "};",
            "",
            (
                "static const int TRUE_CLASSES"
                "[VECTOR_COUNT] = {"
            ),
            "    "
            + ", ".join(
                str(int(value))
                for value in true_classes
            ),
            "};",
            "",
            "}  // namespace tflm_logits_diagnostic",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
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
        / DIAGNOSTIC_MODEL_FILENAME
    )

    report_path = (
        tflite_dir
        / DIAGNOSTIC_REPORT_FILENAME
    )

    normalization_path = (
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    report = load_json(
        report_path
    )

    normalization = load_json(
        normalization_path
    )

    if (
        report.get("test_split_used")
        is not False
    ):
        raise ValueError(
            "Diagnostic report unexpectedly "
            "used the held-out test split."
        )

    if (
        report.get("source_split")
        != "validation"
    ):
        raise ValueError(
            "Diagnostic model was not "
            "evaluated on validation."
        )

    if (
        report.get("representative_split")
        != "train"
    ):
        raise ValueError(
            "Diagnostic model was not "
            "calibrated using train."
        )

    expected_hash = report[
        "diagnostic_model_sha256"
    ]

    actual_hash = sha256_file(
        model_path
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Diagnostic INT8 model SHA mismatch.\n"
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

    if validation.features.shape != (
        200,
        10,
    ):
        raise ValueError(
            "Expected validation features "
            f"shape (200, 10), got "
            f"{validation.features.shape}."
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

    probe = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental
            .OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )

    probe.allocate_tensors()

    input_detail = (
        probe.get_input_details()[0]
    )

    output_detail = (
        probe.get_output_details()[0]
    )

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic input is not INT8."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Diagnostic output is not INT8."
        )

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    if input_scale <= 0.0:
        raise ValueError(
            "Invalid diagnostic input scale."
        )

    if output_scale <= 0.0:
        raise ValueError(
            "Invalid diagnostic output scale."
        )

    quantized_inputs = quantize_tensor(
        normalized,
        scale=float(input_scale),
        zero_point=int(input_zero_point),
    )

    (
        raw_outputs,
        confirmed_input,
        confirmed_output,
    ) = run_desktop_model(
        model_path=model_path,
        quantized_inputs=quantized_inputs,
    )

    confirmed_input_scale, confirmed_input_zp = (
        confirmed_input["quantization"]
    )

    confirmed_output_scale, confirmed_output_zp = (
        confirmed_output["quantization"]
    )

    if (
        float(confirmed_input_scale)
        != float(input_scale)
        or int(confirmed_input_zp)
        != int(input_zero_point)
    ):
        raise RuntimeError(
            "Input quantization changed "
            "between interpreters."
        )

    if (
        float(confirmed_output_scale)
        != float(output_scale)
        or int(confirmed_output_zp)
        != int(output_zero_point)
    ):
        raise RuntimeError(
            "Output quantization changed "
            "between interpreters."
        )

    expected_classes = np.argmax(
        raw_outputs,
        axis=1,
    ).astype(np.int64)

    true_classes = (
        validation.labels
        .astype(np.int64)
    )

    correct = int(
        np.sum(
            expected_classes
            == true_classes
        )
    )

    if correct != 197:
        raise RuntimeError(
            "Diagnostic validation result changed. "
            f"Expected frozen measured result "
            f"197/200, got {correct}/200."
        )

    header_path = (
        root / TEST_HEADER
    )

    model_bytes = (
        model_path.read_bytes()
    )

    write_header(
        path=header_path,
        model_bytes=model_bytes,
        model_hash=actual_hash,
        quantized_inputs=quantized_inputs,
        raw_outputs=raw_outputs,
        expected_classes=expected_classes,
        true_classes=true_classes,
        input_scale=float(input_scale),
        input_zero_point=int(input_zero_point),
        output_scale=float(output_scale),
        output_zero_point=int(output_zero_point),
    )

    print()
    print(
        "ESP32 TFLM LOGITS DIAGNOSTIC EXPORT"
    )

    print(
        "-----------------------------------"
    )

    print(
        f"Diagnostic SHA-256: "
        f"{actual_hash}"
    )

    print(
        f"Diagnostic bytes:   "
        f"{len(model_bytes)}"
    )

    print(
        "Source:             "
        f"validation / {validation.session}"
    )

    print(
        "Vectors:            "
        f"{validation.features.shape[0]}"
    )

    print(
        "Desktop correct:    "
        f"{correct}/"
        f"{validation.features.shape[0]}"
    )

    print()

    print(
        f"Input scale:        "
        f"{float(input_scale):.12g}"
    )

    print(
        f"Input zero point:   "
        f"{int(input_zero_point)}"
    )

    print(
        f"Logits scale:       "
        f"{float(output_scale):.12g}"
    )

    print(
        f"Logits zero point:  "
        f"{int(output_zero_point)}"
    )

    print()

    print(
        "Vector 82 desktop logits: "
        f"{raw_outputs[82].tolist()}"
    )

    print(
        "Vector 82 desktop class:  "
        f"{int(expected_classes[82])}"
    )

    print(
        "Vector 119 desktop logits:"
        f" {raw_outputs[119].tolist()}"
    )

    print(
        "Vector 119 desktop class: "
        f"{int(expected_classes[119])}"
    )

    print()

    print(
        "Representative data: train"
    )

    print(
        "Evaluation data:     validation"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        f"Generated header: {header_path}"
    )


if __name__ == "__main__":
    main()