from __future__ import annotations

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
from ml.export.quantize import (
    normalize_features,
    quantize_tensor,
)
from ml.export.tflite_export import (
    DATASET_VERSION,
    MODEL_DIR,
    project_root,
    sha256_file,
)


MODEL_FILENAME = (
    f"{MODEL_VERSION}-int8-normalized-input.tflite"
)

REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

MODEL_HEADER = Path(
    "firmware/include/gesture_model_data.h"
)

MODEL_SOURCE = Path(
    "firmware/src/inference/gesture_model_data.cpp"
)

TEST_HEADER = Path(
    "firmware/test/"
    "test_local_inference/"
    "local_inference_vectors.h"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def select_vectors(
    labels: np.ndarray,
) -> list[int]:
    selected: list[int] = []

    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]

        indices = np.flatnonzero(
            labels == class_id
        )

        if indices.size == 0:
            raise ValueError(
                f"No validation sample for {gesture}."
            )

        selected.append(
            int(indices[0])
        )

    return selected


def write_model_files(
    *,
    header_path: Path,
    source_path: Path,
    model_bytes: bytes,
    model_hash: str,
) -> None:
    header_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = f"""#pragma once

#include <stddef.h>

namespace gesture_model_data {{

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;

extern const char MODEL_VERSION[];
extern const char MODEL_SHA256[];

}}  // namespace gesture_model_data
"""

    header_path.write_text(
        header,
        encoding="utf-8",
    )

    byte_lines: list[str] = []

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

        byte_lines.append(
            f"    {values},"
        )

    source = (
        '#include "gesture_model_data.h"\n\n'
        "namespace gesture_model_data {\n\n"
        "alignas(16) const unsigned char MODEL[] = {\n"
        + "\n".join(byte_lines)
        + "\n};\n\n"
        "const size_t MODEL_LEN = sizeof(MODEL);\n\n"
        f'const char MODEL_VERSION[] = "{MODEL_VERSION}";\n'
        f'const char MODEL_SHA256[] = "{model_hash}";\n\n'
        "}  // namespace gesture_model_data\n"
    )

    source_path.write_text(
        source,
        encoding="utf-8",
    )


def write_test_vectors(
    *,
    path: Path,
    labels: list[str],
    true_classes: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    expected_classes: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "namespace local_inference_vectors {",
        "",
        f"constexpr size_t VECTOR_COUNT = {len(labels)};",
        "constexpr size_t INPUT_COUNT = 10;",
        "constexpr size_t OUTPUT_COUNT = 5;",
        "",
        "static const char* const LABELS[VECTOR_COUNT] = {",
    ]

    for label in labels:
        lines.append(
            f'    "{label}",'
        )

    lines.extend(
        [
            "};",
            "",
            "static const int TRUE_CLASSES[VECTOR_COUNT] = {",
            "    "
            + ", ".join(
                str(int(value))
                for value in true_classes
            ),
            "};",
            "",
            "static const int8_t INPUTS"
            "[VECTOR_COUNT][INPUT_COUNT] = {",
        ]
    )

    for row in inputs:
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
            "static const int8_t EXPECTED_OUTPUTS"
            "[VECTOR_COUNT][OUTPUT_COUNT] = {",
        ]
    )

    for row in outputs:
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
            "static const int EXPECTED_CLASSES"
            "[VECTOR_COUNT] = {",
            "    "
            + ", ".join(
                str(int(value))
                for value in expected_classes
            ),
            "};",
            "",
            "}  // namespace local_inference_vectors",
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
        root / MODEL_DIR / "tflite"
    )

    model_path = (
        tflite_dir / MODEL_FILENAME
    )

    report = load_json(
        tflite_dir / REPORT_FILENAME
    )

    normalization = load_json(
        tflite_dir / NORMALIZATION_FILENAME
    )

    if report.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Model version mismatch."
        )

    if report.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            "Dataset version mismatch."
        )

    if report.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            "Feature version mismatch."
        )

    if report.get("test_split_used") is not False:
        raise ValueError(
            "INT8 report unexpectedly used test data."
        )

    expected_hash = report[
        "int8_tflite_sha256"
    ]

    actual_hash = sha256_file(
        model_path
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "INT8 model SHA-256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    model_bytes = model_path.read_bytes()

    validation = load_feature_split(
        "validation"
    )

    selected = list(
        range(
            validation.features.shape[0]
        )
    )

    features = (
        validation.features
        .astype(np.float32)
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
        features,
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

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
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
            "Expected int8 model input."
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            "Expected int8 model output."
        )

    raw_outputs = np.empty(
        (len(selected), 5),
        dtype=np.int8,
    )

    for vector_index, model_input in enumerate(
        quantized_inputs
    ):
        interpreter.set_tensor(
            input_detail["index"],
            model_input.reshape(1, 10),
        )

        interpreter.invoke()

        raw_outputs[vector_index] = (
            interpreter.get_tensor(
                output_detail["index"]
            ).reshape(5)
        )

    expected_classes = np.argmax(
        raw_outputs,
        axis=1,
    ).astype(np.int64)

    selected_labels = [
        GESTURES[
            int(validation.labels[index])
        ]
        for index in selected
    ]

    write_model_files(
        header_path=root / MODEL_HEADER,
        source_path=root / MODEL_SOURCE,
        model_bytes=model_bytes,
        model_hash=actual_hash,
    )

    write_test_vectors(
        path=root / TEST_HEADER,
        labels=selected_labels,
        true_classes=validation.labels[
            selected
        ].astype(np.int64),
        inputs=quantized_inputs,
        outputs=raw_outputs,
        expected_classes=expected_classes,
    )

    print(
        f"Model version:   {MODEL_VERSION}"
    )
    print(
        f"INT8 SHA-256:   {actual_hash}"
    )
    print(
        f"Model bytes:     {len(model_bytes)}"
    )
    print(
        "Source:          validation / "
        f"{validation.session}"
    )
    print(
        f"Vectors:         {len(selected)}"
    )

    print(
        "Desktop class matches ground truth: "
        f"{int(np.sum(expected_classes == validation.labels[selected]))}"
        f"/{len(selected)}"
    )

    print("Test split was not loaded.")
    print(
        f"Model header: {root / MODEL_HEADER}"
    )
    print(
        f"Model source: {root / MODEL_SOURCE}"
    )
    print(
        f"Test vectors: {root / TEST_HEADER}"
    )


if __name__ == "__main__":
    main()