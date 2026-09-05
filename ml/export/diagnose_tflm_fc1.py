from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.export.quantize import (
    convert_to_full_int8,
    normalize_features,
    quantize_tensor,
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
from ml.features.extractor import load_feature_split


FULL_MODEL_FILENAME = (
    "gesture-model-v1.0.0-"
    "int8-normalized-input.tflite"
)

DEPLOYMENT_REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

INTERNAL_REPORT_FILENAME = (
    "int8_internal_tensor_report.json"
)

FC1_MODEL_FILENAME = (
    "gesture-model-v1.0.0-"
    "int8-fc1-diagnostic.tflite"
)

FC1_REPORT_FILENAME = (
    "tflm_fc1_diagnostic_report.json"
)

OUTPUT_HEADER = Path(
    "firmware/test/"
    "test_tflm_fc1_diagnostic/"
    "tflm_fc1_diagnostic_vectors.h"
)

PROBE_INDICES = (
    82,
    119,
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


def build_fc1_model(
    source_model: tf.keras.Model,
) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(10,),
        dtype=tf.float32,
        name="normalized_features_v1",
    )

    outputs = source_model.get_layer(
        "block1"
    )(inputs)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_fc1_diagnostic",
    )


def create_interpreter(
    model_path: Path,
    *,
    preserve_all_tensors: bool = False,
) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental
            .OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
        experimental_preserve_all_tensors=(
            preserve_all_tensors
        ),
    )

    interpreter.allocate_tensors()

    return interpreter


def quantization_equal(
    first: tuple[float, int],
    second: tuple[float, int],
) -> bool:
    return (
        np.isclose(
            float(first[0]),
            float(second[0]),
            rtol=0.0,
            atol=1e-12,
        )
        and int(first[1])
        == int(second[1])
    )


def cpp_float(value: float) -> str:
    value = float(
        np.float32(value)
    )

    if not np.isfinite(value):
        raise ValueError(
            "Cannot emit non-finite C++ float."
        )

    text = f"{value:.9g}"

    if (
        "." not in text
        and "e" not in text.lower()
    ):
        text += ".0"

    return f"{text}f"


def write_header(
    *,
    path: Path,
    model_bytes: bytes,
    model_hash: str,
    inputs: np.ndarray,
    outputs: np.ndarray,
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
        "namespace tflm_fc1_diagnostic {",
        "",
        f'constexpr char MODEL_SHA256[] = "{model_hash}";',
        "",
        f"constexpr size_t MODEL_LEN = {len(model_bytes)};",
        f"constexpr size_t PROBE_COUNT = {len(PROBE_INDICES)};",
        "constexpr size_t INPUT_COUNT = 10;",
        "constexpr size_t OUTPUT_COUNT = 64;",
        "",
        (
            "constexpr float INPUT_SCALE = "
            f"{cpp_float(input_scale)};"
        ),
        (
            "constexpr int32_t INPUT_ZERO_POINT = "
            f"{input_zero_point};"
        ),
        (
            "constexpr float OUTPUT_SCALE = "
            f"{cpp_float(output_scale)};"
        ),
        (
            "constexpr int32_t OUTPUT_ZERO_POINT = "
            f"{output_zero_point};"
        ),
        "",
        (
            "alignas(16) static const unsigned char "
            "MODEL[MODEL_LEN] = {"
        ),
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
                "static const int "
                "VALIDATION_INDICES[PROBE_COUNT] = {"
            ),
            "    "
            + ", ".join(
                str(index)
                for index in PROBE_INDICES
            ),
            "};",
            "",
            (
                "static const int8_t "
                "INPUTS[PROBE_COUNT][INPUT_COUNT] = {"
            ),
        ]
    )

    for index in PROBE_INDICES:
        values = ", ".join(
            str(int(value))
            for value in inputs[index]
        )

        lines.append(
            f"    {{{values}}},"
        )

    lines.extend(
        [
            "};",
            "",
            (
                "static const int8_t "
                "EXPECTED_FC1"
                "[PROBE_COUNT][OUTPUT_COUNT] = {"
            ),
        ]
    )

    for index in PROBE_INDICES:
        values = ", ".join(
            str(int(value))
            for value in outputs[index]
        )

        lines.append(
            f"    {{{values}}},"
        )

    lines.extend(
        [
            "};",
            "",
            "}  // namespace tflm_fc1_diagnostic",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
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

    full_model_path = (
        tflite_dir
        / FULL_MODEL_FILENAME
    )

    fc1_model_path = (
        tflite_dir
        / FC1_MODEL_FILENAME
    )

    deployment_report = load_json(
        tflite_dir
        / DEPLOYMENT_REPORT_FILENAME
    )

    normalization = load_json(
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    internal_report = load_json(
        tflite_dir
        / INTERNAL_REPORT_FILENAME
    )

    if (
        deployment_report.get(
            "test_split_used"
        )
        is not False
    ):
        raise ValueError(
            "Deployment report used test split."
        )

    expected_full_hash = (
        deployment_report[
            "int8_tflite_sha256"
        ]
    )

    actual_full_hash = sha256_file(
        full_model_path
    )

    if actual_full_hash != expected_full_hash:
        raise RuntimeError(
            "Frozen deployment model SHA mismatch."
        )

    if (
        internal_report[
            "model_sha256"
        ]
        != actual_full_hash
    ):
        raise RuntimeError(
            "Internal tensor report does not "
            "belong to frozen deployment model."
        )

    fc1_operator = next(
        (
            operator
            for operator
            in internal_report["operators"]
            if operator["stage"] == "FC1"
        ),
        None,
    )

    if fc1_operator is None:
        raise RuntimeError(
            "FC1 tensor metadata not found."
        )

    full_fc1_tensor_index = int(
        fc1_operator["tensor_index"]
    )

    if full_fc1_tensor_index != 9:
        raise RuntimeError(
            "Unexpected FC1 tensor index: "
            f"{full_fc1_tensor_index}"
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

    train = load_feature_split(
        "train"
    )

    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected validation/session_02."
        )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    variance = np.asarray(
        normalization["variance"],
        dtype=np.float32,
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

    #
    # Verify float FC1 diagnostic parity first.
    #

    source_fc1_model = tf.keras.Model(
        inputs=source_model.input,
        outputs=source_model.get_layer(
            "block1"
        ).output,
    )

    source_fc1_float = (
        source_fc1_model.predict(
            validation.features,
            verbose=0,
        )
        .astype(np.float32)
    )

    fc1_model = build_fc1_model(
        source_model
    )

    diagnostic_fc1_float = (
        fc1_model.predict(
            normalized_validation,
            verbose=0,
        )
        .astype(np.float32)
    )

    float_parity = bool(
        np.allclose(
            source_fc1_float,
            diagnostic_fc1_float,
            atol=1e-6,
            rtol=1e-6,
        )
    )

    float_max_diff = float(
        np.max(
            np.abs(
                source_fc1_float
                - diagnostic_fc1_float
            )
        )
    )

    if not float_parity:
        raise RuntimeError(
            "Float FC1 diagnostic model does "
            "not reproduce frozen block1."
        )

    #
    # Quantize FC1-only diagnostic model.
    #

    fc1_bytes = convert_to_full_int8(
        fc1_model,
        normalized_train,
    )

    fc1_model_path.write_bytes(
        fc1_bytes
    )

    fc1_hash = sha256_file(
        fc1_model_path
    )

    #
    # Open full model with preserved tensors on Desktop.
    #

    full_interpreter = create_interpreter(
        full_model_path,
        preserve_all_tensors=True,
    )

    fc1_interpreter = create_interpreter(
        fc1_model_path,
    )

    full_input = (
        full_interpreter
        .get_input_details()[0]
    )

    fc1_input = (
        fc1_interpreter
        .get_input_details()[0]
    )

    fc1_output = (
        fc1_interpreter
        .get_output_details()[0]
    )

    full_tensor_details = {
        int(detail["index"]): detail
        for detail
        in full_interpreter
        .get_tensor_details()
    }

    full_fc1_detail = (
        full_tensor_details[
            full_fc1_tensor_index
        ]
    )

    if full_input["dtype"] != np.int8:
        raise ValueError(
            "Full model input is not INT8."
        )

    if fc1_input["dtype"] != np.int8:
        raise ValueError(
            "FC1 model input is not INT8."
        )

    if fc1_output["dtype"] != np.int8:
        raise ValueError(
            "FC1 model output is not INT8."
        )

    if full_fc1_detail["dtype"] != np.int8:
        raise ValueError(
            "Full model FC1 tensor is not INT8."
        )

    full_input_q = full_input[
        "quantization"
    ]

    fc1_input_q = fc1_input[
        "quantization"
    ]

    full_fc1_q = full_fc1_detail[
        "quantization"
    ]

    fc1_output_q = fc1_output[
        "quantization"
    ]

    if not quantization_equal(
        full_input_q,
        fc1_input_q,
    ):
        raise RuntimeError(
            "FC1 diagnostic input quantization "
            "does not match frozen model input."
        )

    if not quantization_equal(
        full_fc1_q,
        fc1_output_q,
    ):
        raise RuntimeError(
            "FC1 diagnostic output quantization "
            "does not match frozen FC1 tensor."
        )

    input_scale, input_zero_point = (
        full_input_q
    )

    quantized_inputs = quantize_tensor(
        normalized_validation,
        scale=float(input_scale),
        zero_point=int(input_zero_point),
        dtype=np.int8,
    )

    full_fc1_outputs = np.empty(
        (200, 64),
        dtype=np.int8,
    )

    diagnostic_fc1_outputs = np.empty(
        (200, 64),
        dtype=np.int8,
    )

    for index, vector in enumerate(
        quantized_inputs
    ):
        model_input = vector.reshape(
            1,
            10,
        )

        full_interpreter.set_tensor(
            full_input["index"],
            model_input,
        )

        full_interpreter.invoke()

        full_fc1_outputs[index] = (
            full_interpreter.get_tensor(
                full_fc1_tensor_index
            )
            .reshape(64)
            .astype(np.int8)
        )

        fc1_interpreter.set_tensor(
            fc1_input["index"],
            model_input,
        )

        fc1_interpreter.invoke()

        diagnostic_fc1_outputs[index] = (
            fc1_interpreter.get_tensor(
                fc1_output["index"]
            )
            .reshape(64)
            .astype(np.int8)
        )

    differences = np.abs(
        full_fc1_outputs.astype(np.int16)
        - diagnostic_fc1_outputs.astype(np.int16)
    )

    int8_exact = bool(
        np.array_equal(
            full_fc1_outputs,
            diagnostic_fc1_outputs,
        )
    )

    max_lsb_difference = int(
        np.max(differences)
    )

    mismatch_count = int(
        np.count_nonzero(
            differences
        )
    )

    if not int8_exact:
        raise RuntimeError(
            "FC1 diagnostic model does not "
            "exactly reproduce frozen full-model "
            "FC1 on Desktop. "
            f"mismatches={mismatch_count}, "
            f"max_lsb_diff={max_lsb_difference}"
        )

    output_scale, output_zero_point = (
        fc1_output_q
    )

    header_path = (
        root / OUTPUT_HEADER
    )

    write_header(
        path=header_path,
        model_bytes=fc1_bytes,
        model_hash=fc1_hash,
        inputs=quantized_inputs,
        outputs=diagnostic_fc1_outputs,
        input_scale=float(input_scale),
        input_zero_point=int(input_zero_point),
        output_scale=float(output_scale),
        output_zero_point=int(output_zero_point),
    )

    result = {
        "purpose":
            "isolate first fully-connected layer on TFLM",
        "source_model_sha256":
            source_hash,
        "full_int8_model_sha256":
            actual_full_hash,
        "fc1_diagnostic_sha256":
            fc1_hash,
        "fc1_diagnostic_bytes":
            len(fc1_bytes),
        "source_split":
            "validation",
        "source_session":
            validation.session,
        "test_split_used":
            False,
        "float_fc1_parity":
            float_parity,
        "float_fc1_max_abs_difference":
            float_max_diff,
        "desktop_int8_fc1_exact":
            int8_exact,
        "desktop_int8_fc1_mismatch_count":
            mismatch_count,
        "desktop_int8_fc1_max_lsb_difference":
            max_lsb_difference,
        "input_quantization": {
            "scale":
                float(input_scale),
            "zero_point":
                int(input_zero_point),
        },
        "fc1_output_quantization": {
            "scale":
                float(output_scale),
            "zero_point":
                int(output_zero_point),
        },
        "probe_indices":
            list(PROBE_INDICES),
    }

    report_path = (
        tflite_dir
        / FC1_REPORT_FILENAME
    )

    report_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "TFLM FC1 ISOLATION - DESKTOP"
    )

    print(
        "----------------------------"
    )

    print(
        f"Frozen full model SHA: "
        f"{actual_full_hash}"
    )

    print(
        f"FC1 diagnostic SHA:   "
        f"{fc1_hash}"
    )

    print(
        f"FC1 diagnostic bytes: "
        f"{len(fc1_bytes)}"
    )

    print()

    print(
        f"Float FC1 parity:      "
        f"{float_parity}"
    )

    print(
        f"Float FC1 max diff:    "
        f"{float_max_diff:.10f}"
    )

    print()

    print(
        f"Desktop INT8 FC1 exact:"
        f" {int8_exact}"
    )

    print(
        f"FC1 mismatched values: "
        f"{mismatch_count}/12800"
    )

    print(
        f"FC1 max LSB diff:      "
        f"{max_lsb_difference}"
    )

    print()

    print(
        f"Input scale:           "
        f"{float(input_scale):.12g}"
    )

    print(
        f"Input zero point:      "
        f"{int(input_zero_point)}"
    )

    print(
        f"FC1 output scale:      "
        f"{float(output_scale):.12g}"
    )

    print(
        f"FC1 output zero point: "
        f"{int(output_zero_point)}"
    )

    print()

    print(
        "Probe vectors:         82, 119"
    )

    print(
        "Source:                "
        "validation / session_02"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        f"Header: {header_path}"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()