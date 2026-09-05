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
    value = float(np.float32(value))

    if not np.isfinite(value):
        raise ValueError(
            "Cannot generate non-finite float."
        )

    text = f"{value:.9g}"

    if (
        "." not in text
        and "e" not in text.lower()
    ):
        text += ".0"

    return f"{text}f"


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
                f"No sample found for {gesture}."
            )

        selected.append(int(indices[0]))

    return selected


def main() -> None:
    root = project_root()

    tflite_dir = (
        root / MODEL_DIR / "tflite"
    )

    normalization = load_json(
        tflite_dir
        / NORMALIZATION_FILENAME
    )

    report = load_json(
        tflite_dir
        / REPORT_FILENAME
    )

    for record in (
        normalization,
        report,
    ):
        if (
            record.get("model_version")
            != MODEL_VERSION
        ):
            raise ValueError(
                "Model version mismatch."
            )

        if (
            record.get("dataset_version")
            != DATASET_VERSION
        ):
            raise ValueError(
                "Dataset version mismatch."
            )

        if (
            record.get("feature_version")
            != FEATURE_VERSION
        ):
            raise ValueError(
                "Feature version mismatch."
            )

    if normalization.get("fit_split") != "train":
        raise ValueError(
            "Normalization was not fitted on train."
        )

    if report.get("test_split_used") is not False:
        raise ValueError(
            "INT8 report unexpectedly used test data."
        )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    if mean.shape != (10,) or std.shape != (10,):
        raise ValueError(
            "Expected 10 normalization values."
        )

    input_scale = float(
        report["input_quantization"]["scale"]
    )

    input_zero_point = int(
        report[
            "input_quantization"
        ]["zero_point"]
    )

    int8_hash = report[
        "int8_tflite_sha256"
    ]

    validation = load_feature_split(
        "validation"
    )

    selected = select_vectors(
        validation.labels
    )

    features = (
        validation.features[selected]
        .astype(np.float32)
    )

    normalized = (
        (features - mean) / std
    ).astype(np.float32)

    quantized = np.round(
        normalized / input_scale
        + input_zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127,
    ).astype(np.int8)

    params_path = root / PARAMS_HEADER
    params_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_lines = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "namespace deployment_preprocessing {",
        "",
        "constexpr size_t FEATURE_COUNT = 10;",
        "",
        f'constexpr char MODEL_VERSION[] = "{MODEL_VERSION}";',
        f'constexpr char INT8_MODEL_SHA256[] = "{int8_hash}";',
        "",
        "constexpr float NORMALIZATION_MEAN[FEATURE_COUNT] = {",
        "    "
        + ", ".join(
            cpp_float(value)
            for value in mean
        ),
        "};",
        "",
        "constexpr float NORMALIZATION_STD[FEATURE_COUNT] = {",
        "    "
        + ", ".join(
            cpp_float(value)
            for value in std
        ),
        "};",
        "",
        f"constexpr float INPUT_SCALE = {cpp_float(input_scale)};",
        f"constexpr int32_t INPUT_ZERO_POINT = {input_zero_point};",
        "",
        "}  // namespace deployment_preprocessing",
        "",
    ]

    params_path.write_text(
        "\n".join(params_lines),
        encoding="utf-8",
    )

    test_path = root / TEST_HEADER
    test_path.parent.mkdir(
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
        f"constexpr size_t VECTOR_COUNT = {len(selected)};",
        "constexpr size_t FEATURE_COUNT = 10;",
        "",
        "static const char* const LABELS[VECTOR_COUNT] = {",
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
            "static const float FEATURES"
            "[VECTOR_COUNT][FEATURE_COUNT] = {",
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
            "static const float EXPECTED_NORMALIZED"
            "[VECTOR_COUNT][FEATURE_COUNT] = {",
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
            "static const int8_t EXPECTED_INT8"
            "[VECTOR_COUNT][FEATURE_COUNT] = {",
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

    test_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Model version:   {MODEL_VERSION}"
    )
    print(
        f"Feature version: {FEATURE_VERSION}"
    )
    print(
        "Source:          validation / "
        f"{validation.session}"
    )
    print(
        f"Vectors:         {len(selected)}"
    )
    print(
        f"Input scale:     {input_scale}"
    )
    print(
        f"Input zero point:{input_zero_point: d}"
    )
    print("Test split was not loaded.")
    print(f"Params: {params_path}")
    print(f"Test:   {test_path}")


if __name__ == "__main__":
    main()