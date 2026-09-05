from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
)
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import MODEL_VERSION
from ml.export.tflite_export import (
    DATASET_VERSION,
    MODEL_DIR,
    project_root,
)


FEATURE_COUNT = 10

PARAMS_HEADER = Path(
    "firmware/include/"
    "deployment_preprocessing_params.h"
)

TEST_HEADER = Path(
    "firmware/test/"
    "test_input_preprocessing/"
    "input_preprocessing_vectors.h"
)

NORMALIZATION_FILENAME = (
    "deployment_normalization.json"
)

REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)


def cpp_float(value: float) -> str:
    """Convert a finite float to a C++ float literal."""

    value = float(
        np.float32(value)
    )

    if not np.isfinite(value):
        raise ValueError(
            "Cannot generate a non-finite C++ float literal."
        )

    text = f"{value:.9g}"

    if (
        "." not in text
        and "e" not in text.lower()
    ):
        text += ".0"

    return f"{text}f"


def load_json(path: Path) -> dict:
    """Load one required JSON artifact."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_version_record(
    record: dict,
    *,
    record_name: str,
) -> None:
    """Verify deployment artifact version compatibility."""

    expected = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
    }

    for field, expected_value in expected.items():
        actual_value = record.get(field)

        if actual_value != expected_value:
            raise ValueError(
                f"{record_name} version mismatch: "
                f"{field}={actual_value!r}, "
                f"expected {expected_value!r}."
            )


def select_vectors(
    labels: np.ndarray,
) -> list[int]:
    """Select the first validation sample from each gesture."""

    selected: list[int] = []

    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]

        indices = np.flatnonzero(
            labels == class_id
        )

        if indices.size == 0:
            raise ValueError(
                f"No validation sample found for {gesture}."
            )

        selected.append(
            int(indices[0])
        )

    return selected


def validate_normalization(
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Validate frozen deployment normalization parameters."""

    expected_shape = (
        FEATURE_COUNT,
    )

    if mean.shape != expected_shape:
        raise ValueError(
            "Unexpected normalization mean shape: "
            f"{mean.shape}. "
            f"Expected {expected_shape}."
        )

    if std.shape != expected_shape:
        raise ValueError(
            "Unexpected normalization std shape: "
            f"{std.shape}. "
            f"Expected {expected_shape}."
        )

    if not np.isfinite(mean).all():
        raise ValueError(
            "Normalization mean contains "
            "non-finite values."
        )

    if not np.isfinite(std).all():
        raise ValueError(
            "Normalization std contains "
            "non-finite values."
        )

    if np.any(std <= 0.0):
        raise ValueError(
            "Normalization std values "
            "must all be positive."
        )


def validate_quantization(
    *,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
) -> None:
    """Validate INT8 input/output quantization parameters."""

    if (
        not np.isfinite(input_scale)
        or input_scale <= 0.0
    ):
        raise ValueError(
            f"Invalid input quantization scale: "
            f"{input_scale}"
        )

    if (
        not np.isfinite(output_scale)
        or output_scale <= 0.0
    ):
        raise ValueError(
            f"Invalid output quantization scale: "
            f"{output_scale}"
        )

    if not -128 <= input_zero_point <= 127:
        raise ValueError(
            "Input zero point is outside "
            f"INT8 range: {input_zero_point}"
        )

    if not -128 <= output_zero_point <= 127:
        raise ValueError(
            "Output zero point is outside "
            f"INT8 range: {output_zero_point}"
        )


def write_params_header(
    *,
    path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    int8_hash: str,
) -> None:
    """Generate frozen ESP32 deployment constants."""

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
        "namespace deployment_preprocessing {",
        "",
        f"constexpr size_t FEATURE_COUNT = {FEATURE_COUNT};",
        "",
        (
            'constexpr char MODEL_VERSION[] = '
            f'"{MODEL_VERSION}";'
        ),
        (
            'constexpr char INT8_MODEL_SHA256[] = '
            f'"{int8_hash}";'
        ),
        "",
        (
            "constexpr float "
            "NORMALIZATION_MEAN[FEATURE_COUNT] = {"
        ),
        "    "
        + ", ".join(
            cpp_float(value)
            for value in mean
        ),
        "};",
        "",
        (
            "constexpr float "
            "NORMALIZATION_STD[FEATURE_COUNT] = {"
        ),
        "    "
        + ", ".join(
            cpp_float(value)
            for value in std
        ),
        "};",
        "",
        (
            "constexpr float INPUT_SCALE = "
            f"{cpp_float(input_scale)};"
        ),
        (
            "constexpr int32_t INPUT_ZERO_POINT = "
            f"{input_zero_point};"
        ),
        "",
        (
            "constexpr float OUTPUT_SCALE = "
            f"{cpp_float(output_scale)};"
        ),
        (
            "constexpr int32_t OUTPUT_ZERO_POINT = "
            f"{output_zero_point};"
        ),
        "",
        "}  // namespace deployment_preprocessing",
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_test_header(
    *,
    path: Path,
    validation,
    selected: list[int],
    features: np.ndarray,
    normalized: np.ndarray,
    quantized: np.ndarray,
) -> None:
    """Generate ESP32 preprocessing parity vectors."""

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
        "namespace input_preprocessing_vectors {",
        "",
        (
            "constexpr size_t VECTOR_COUNT = "
            f"{len(selected)};"
        ),
        (
            "constexpr size_t FEATURE_COUNT = "
            f"{FEATURE_COUNT};"
        ),
        "",
        (
            "static const char* const "
            "LABELS[VECTOR_COUNT] = {"
        ),
    ]

    for index in selected:
        gesture = GESTURES[
            int(validation.labels[index])
        ]

        lines.append(
            f'    "{gesture}",'
        )

    lines.extend(
        [
            "};",
            "",
            (
                "static const float FEATURES"
                "[VECTOR_COUNT][FEATURE_COUNT] = {"
            ),
        ]
    )

    for row in features:
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
                "static const float EXPECTED_NORMALIZED"
                "[VECTOR_COUNT][FEATURE_COUNT] = {"
            ),
        ]
    )

    for row in normalized:
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
                "static const int8_t EXPECTED_INT8"
                "[VECTOR_COUNT][FEATURE_COUNT] = {"
            ),
        ]
    )

    for row in quantized:
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
            "}  // namespace input_preprocessing_vectors",
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

    normalization_path = (
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    report_path = (
        tflite_dir
        / REPORT_FILENAME
    )

    normalization = load_json(
        normalization_path
    )

    report = load_json(
        report_path
    )

    validate_version_record(
        normalization,
        record_name="Normalization record",
    )

    validate_version_record(
        report,
        record_name="INT8 report",
    )

    if normalization.get("fit_split") != "train":
        raise ValueError(
            "Normalization was not fitted on train."
        )

    if report.get("representative_split") != "train":
        raise ValueError(
            "INT8 representative dataset "
            "was not the train split."
        )

    if report.get("evaluation_split") != "validation":
        raise ValueError(
            "INT8 deployment model was not "
            "evaluated on validation."
        )

    if report.get("test_split_used") is not False:
        raise ValueError(
            "INT8 report unexpectedly used "
            "the held-out test split."
        )

    if (
        report.get("deployment_input")
        != "train-standardized-features-v1"
    ):
        raise ValueError(
            "Unexpected INT8 deployment input contract: "
            f"{report.get('deployment_input')!r}"
        )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    validate_normalization(
        mean,
        std,
    )

    input_quantization = report.get(
        "input_quantization"
    )

    output_quantization = report.get(
        "output_quantization"
    )

    if not isinstance(
        input_quantization,
        dict,
    ):
        raise ValueError(
            "INT8 report has no valid "
            "input_quantization record."
        )

    if not isinstance(
        output_quantization,
        dict,
    ):
        raise ValueError(
            "INT8 report has no valid "
            "output_quantization record."
        )

    input_scale = float(
        input_quantization["scale"]
    )

    input_zero_point = int(
        input_quantization["zero_point"]
    )

    output_scale = float(
        output_quantization["scale"]
    )

    output_zero_point = int(
        output_quantization["zero_point"]
    )

    validate_quantization(
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
    )

    int8_hash = report.get(
        "int8_tflite_sha256"
    )

    if (
        not isinstance(int8_hash, str)
        or len(int8_hash) != 64
    ):
        raise ValueError(
            "INT8 report has no valid "
            "SHA-256 model hash."
        )

    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected validation/session_02, "
            f"got {validation.session!r}."
        )

    selected = select_vectors(
        validation.labels
    )

    features = (
        validation.features[
            selected
        ]
        .astype(np.float32)
    )

    if features.shape != (
        len(GESTURES),
        FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected selected feature shape: "
            f"{features.shape}"
        )

    normalized = (
        (features - mean)
        / std
    ).astype(np.float32)

    if not np.isfinite(
        normalized
    ).all():
        raise ValueError(
            "Generated normalized vectors "
            "contain non-finite values."
        )

    quantized_float = np.round(
        normalized
        / input_scale
        + input_zero_point
    )

    quantized = np.clip(
        quantized_float,
        -128,
        127,
    ).astype(np.int8)

    params_path = (
        root
        / PARAMS_HEADER
    )

    write_params_header(
        path=params_path,
        mean=mean,
        std=std,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        int8_hash=int8_hash,
    )

    test_path = (
        root
        / TEST_HEADER
    )

    write_test_header(
        path=test_path,
        validation=validation,
        selected=selected,
        features=features,
        normalized=normalized,
        quantized=quantized,
    )

    print(
        f"Model version:   {MODEL_VERSION}"
    )

    print(
        f"Feature version: {FEATURE_VERSION}"
    )

    print(
        f"Dataset version: {DATASET_VERSION}"
    )

    print(
        "Source:          validation / "
        f"{validation.session}"
    )

    print(
        f"Vectors:         {len(selected)}"
    )

    print()

    print(
        f"Input scale:      {input_scale}"
    )

    print(
        f"Input zero point: {input_zero_point}"
    )

    print(
        f"Output scale:     {output_scale}"
    )

    print(
        f"Output zero point:{output_zero_point: d}"
    )

    print()

    print(
        f"INT8 SHA-256: "
        f"{int8_hash}"
    )

    print()

    print(
        "Normalization fit split: train"
    )

    print(
        "INT8 representative:     train"
    )

    print(
        "Parity source:           validation"
    )

    print(
        "Test split was not loaded."
    )

    print()

    print(
        f"Params: {params_path}"
    )

    print(
        f"Test:   {test_path}"
    )


if __name__ == "__main__":
    main()