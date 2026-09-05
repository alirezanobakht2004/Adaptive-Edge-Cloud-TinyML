from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.export.quantize import normalize_features
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


NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

FLOAT_MODEL_FILENAME = (
    "gesture-model-v1.0.0-"
    "float32-normalized-input.tflite"
)

FLOAT_REPORT_FILENAME = (
    "float32_normalized_input_report.json"
)

TEST_HEADER = Path(
    "firmware/test/"
    "test_tflm_float32_runtime/"
    "tflm_float32_runtime_vectors.h"
)

FEATURE_COUNT = 10
CLASS_COUNT = 5
EXPECTED_VALIDATION_COUNT = 200


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


def cpp_float(value: float) -> str:
    value = float(
        np.float32(value)
    )

    if not np.isfinite(value):
        raise ValueError(
            "Cannot emit non-finite float."
        )

    text = f"{value:.9g}"

    if (
        "." not in text
        and "e" not in text.lower()
    ):
        text += ".0"

    return f"{text}f"


def build_float_core(
    source_model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Build a float32 deployment core with external
    frozen TRAIN normalization.

    No weights are retrained or modified.
    """

    inputs = tf.keras.Input(
        shape=(FEATURE_COUNT,),
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

    outputs = source_model.get_layer(
        "edge_head"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_float32_deployment_core",
    )


def run_tflite(
    model_path: Path,
    inputs: np.ndarray,
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

    input_detail = (
        interpreter.get_input_details()[0]
    )

    output_detail = (
        interpreter.get_output_details()[0]
    )

    if input_detail["dtype"] != np.float32:
        raise ValueError(
            "Expected FLOAT32 model input."
        )

    if output_detail["dtype"] != np.float32:
        raise ValueError(
            "Expected FLOAT32 model output."
        )

    outputs = np.empty(
        (
            inputs.shape[0],
            CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    for index, vector in enumerate(
        inputs
    ):
        interpreter.set_tensor(
            input_detail["index"],
            vector.reshape(
                1,
                FEATURE_COUNT,
            ),
        )

        interpreter.invoke()

        outputs[index] = (
            interpreter.get_tensor(
                output_detail["index"]
            )
            .reshape(CLASS_COUNT)
            .astype(np.float32)
        )

    return (
        outputs,
        input_detail,
        output_detail,
    )


def write_header(
    *,
    path: Path,
    model_bytes: bytes,
    model_hash: str,
    inputs: np.ndarray,
    outputs: np.ndarray,
    expected_classes: np.ndarray,
    true_classes: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "",
        "namespace tflm_float32_runtime {",
        "",
        (
            f'constexpr char MODEL_SHA256[] = '
            f'"{model_hash}";'
        ),
        "",
        (
            "constexpr size_t MODEL_LEN = "
            f"{len(model_bytes)};"
        ),
        (
            "constexpr size_t VECTOR_COUNT = "
            f"{inputs.shape[0]};"
        ),
        (
            "constexpr size_t INPUT_COUNT = "
            f"{FEATURE_COUNT};"
        ),
        (
            "constexpr size_t OUTPUT_COUNT = "
            f"{CLASS_COUNT};"
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
                "static const float INPUTS"
                "[VECTOR_COUNT][INPUT_COUNT] = {"
            ),
        ]
    )

    for row in inputs:
        values = ", ".join(
            cpp_float(value)
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
                "static const float EXPECTED_OUTPUTS"
                "[VECTOR_COUNT][OUTPUT_COUNT] = {"
            ),
        ]
    )

    for row in outputs:
        values = ", ".join(
            cpp_float(value)
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
                for value
                in expected_classes
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
                for value
                in true_classes
            ),
            "};",
            "",
            "}  // namespace tflm_float32_runtime",
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
        root
        / MODEL_DIR
    )

    tflite_dir = (
        model_dir
        / "tflite"
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

    float_model_path = (
        tflite_dir
        / FLOAT_MODEL_FILENAME
    )

    float_report_path = (
        tflite_dir
        / FLOAT_REPORT_FILENAME
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

    normalization = load_json(
        normalization_path
    )

    if normalization.get(
        "fit_split"
    ) != "train":
        raise ValueError(
            "Normalization was not "
            "fitted on train."
        )

    source_model = (
        tf.keras.models.load_model(
            keras_path
        )
    )

    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected validation/session_02."
        )

    if validation.features.shape != (
        EXPECTED_VALIDATION_COUNT,
        FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected validation feature "
            f"shape: {validation.features.shape}"
        )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    variance = np.asarray(
        normalization["variance"],
        dtype=np.float32,
    )

    normalized_validation = (
        normalize_features(
            validation.features,
            mean=mean,
            variance=variance,
        )
        .astype(np.float32)
    )

    float_core = build_float_core(
        source_model
    )

    #
    # Verify Keras core parity before conversion.
    #

    source_outputs = (
        source_model.predict(
            validation.features,
            verbose=0,
        )
        .astype(np.float32)
    )

    core_outputs = (
        float_core.predict(
            normalized_validation,
            verbose=0,
        )
        .astype(np.float32)
    )

    keras_difference = np.abs(
        source_outputs
        - core_outputs
    )

    keras_max_difference = float(
        np.max(
            keras_difference
        )
    )

    keras_class_matches = int(
        np.sum(
            np.argmax(
                source_outputs,
                axis=1,
            )
            == np.argmax(
                core_outputs,
                axis=1,
            )
        )
    )

    if keras_class_matches != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            "Float deployment core changed "
            "Keras class predictions."
        )

    #
    # Export float32 TFLite.
    #

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(
            float_core
        )
    )

    float_bytes = (
        converter.convert()
    )

    float_model_path.write_bytes(
        float_bytes
    )

    float_hash = sha256_file(
        float_model_path
    )

    #
    # Desktop TFLite validation.
    #

    (
        tflite_outputs,
        input_detail,
        output_detail,
    ) = run_tflite(
        float_model_path,
        normalized_validation,
    )

    tflite_difference = np.abs(
        core_outputs
        - tflite_outputs
    )

    tflite_max_difference = float(
        np.max(
            tflite_difference
        )
    )

    tflite_mean_difference = float(
        np.mean(
            tflite_difference
        )
    )

    source_classes = np.argmax(
        source_outputs,
        axis=1,
    )

    tflite_classes = np.argmax(
        tflite_outputs,
        axis=1,
    )

    class_matches = int(
        np.sum(
            source_classes
            == tflite_classes
        )
    )

    if class_matches != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            "Float32 TFLite changed "
            "frozen source class predictions."
        )

    correct = int(
        np.sum(
            tflite_classes
            == validation.labels
        )
    )

    accuracy = (
        correct
        / EXPECTED_VALIDATION_COUNT
    )

    if correct != 197:
        raise RuntimeError(
            "Float32 validation regression: "
            f"expected measured frozen result "
            f"197/200, got {correct}/200."
        )

    if input_detail["dtype"] != np.float32:
        raise RuntimeError(
            "Generated input tensor "
            "is not float32."
        )

    if output_detail["dtype"] != np.float32:
        raise RuntimeError(
            "Generated output tensor "
            "is not float32."
        )

    header_path = (
        root
        / TEST_HEADER
    )

    write_header(
        path=header_path,
        model_bytes=float_bytes,
        model_hash=float_hash,
        inputs=normalized_validation,
        outputs=tflite_outputs,
        expected_classes=tflite_classes,
        true_classes=(
            validation.labels
            .astype(np.int64)
        ),
    )

    report = {
        "purpose":
            "evaluate float32 TFLM runtime "
            "as controlled INT8 fallback",
        "source_model_sha256":
            source_hash,
        "float32_tflite_sha256":
            float_hash,
        "float32_tflite_bytes":
            len(float_bytes),
        "normalization_fit_split":
            "train",
        "evaluation_split":
            "validation",
        "evaluation_session":
            validation.session,
        "test_split_used":
            False,
        "keras_core_class_matches":
            keras_class_matches,
        "keras_core_max_abs_difference":
            keras_max_difference,
        "desktop_tflite_class_matches":
            class_matches,
        "desktop_tflite_max_abs_difference":
            tflite_max_difference,
        "desktop_tflite_mean_abs_difference":
            tflite_mean_difference,
        "desktop_validation_correct":
            correct,
        "desktop_validation_accuracy":
            accuracy,
    }

    float_report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "FLOAT32 TFLM RUNTIME CANDIDATE"
    )
    print(
        "------------------------------"
    )

    print(
        f"Frozen source SHA: "
        f"{source_hash}"
    )

    print(
        f"Float32 TFLite SHA: "
        f"{float_hash}"
    )

    print(
        f"Float32 TFLite bytes: "
        f"{len(float_bytes)}"
    )

    print()

    print(
        "Keras core class matches: "
        f"{keras_class_matches}/200"
    )

    print(
        "Keras core max abs diff: "
        f"{keras_max_difference:.10f}"
    )

    print()

    print(
        "Desktop TFLite class matches: "
        f"{class_matches}/200"
    )

    print(
        "Desktop TFLite validation: "
        f"{correct}/200 "
        f"({accuracy:.6f})"
    )

    print(
        "Desktop TFLite max abs diff: "
        f"{tflite_max_difference:.10f}"
    )

    print(
        "Desktop TFLite mean abs diff: "
        f"{tflite_mean_difference:.10f}"
    )

    print()

    print(
        "Normalization fit: train"
    )

    print(
        "Evaluation:        "
        "validation / session_02"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        f"Model:  {float_model_path}"
    )

    print(
        f"Header: {header_path}"
    )

    print(
        f"Report: {float_report_path}"
    )


if __name__ == "__main__":
    main()