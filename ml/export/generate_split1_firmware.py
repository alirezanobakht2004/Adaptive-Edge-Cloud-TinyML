#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_MODEL_VERSION = "gesture-model-v1.1.0"
SPLIT_ID = 1
INPUT_FEATURES = 10
OUTPUT_UNITS = 64

MODEL_ROOT = Path(
    "data/processed/dataset-v1/features-v1/models/"
    "gesture-model-v1.1.0/tflite/splits"
)

REPORT_PATH = MODEL_ROOT / "split1_export_report.json"

MODEL_PATH = MODEL_ROOT / (
    "gesture-model-v1.1.0-split1-prefix-"
    "float32-normalized-input.tflite"
)

PARITY_PATH = MODEL_ROOT / "split1_parity_vectors.npz"

MODEL_HEADER_PATH = Path(
    "firmware/include/split1_model_data.h"
)

MODEL_SOURCE_PATH = Path(
    "firmware/src/inference/split1_model_data.cpp"
)

PARITY_HEADER_PATH = Path(
    "firmware/test/test_phase7_split1_parity/"
    "split1_parity_vectors.h"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bytes_to_cpp(data: bytes) -> str:
    items = [f"0x{value:02x}" for value in data]
    rows = []

    for start in range(0, len(items), 12):
        rows.append(
            "    " + ", ".join(items[start:start + 12]) + ","
        )

    return "\n".join(rows)


def float_literal(value: float) -> str:
    value32 = np.float32(value)

    if not np.isfinite(value32):
        raise ValueError(
            f"Non-finite parity value encountered: {value32}"
        )

    text = f"{float(value32):.9g}"

    # C++ floating literals need either a decimal point or an exponent
    # before the `f` suffix. Values such as 0f / 1f are invalid.
    if "." not in text and "e" not in text.lower():
        text += ".0"

    return text + "f"


def matrix_to_cpp(matrix: np.ndarray) -> str:
    rows = []

    for row in matrix:
        values = ", ".join(
            float_literal(value)
            for value in row
        )
        rows.append(f"    {{{values}}},")

    return "\n".join(rows)


def integer_array_to_cpp(values: np.ndarray) -> str:
    return ", ".join(str(int(value)) for value in values)


def validate_report(
    report: dict,
    model_bytes: bytes,
) -> None:
    expected_scalars = {
        "phase": 7,
        "milestone": "M8",
        "split": SPLIT_ID,
        "source_model_version": SOURCE_MODEL_VERSION,
        "test_split_used": False,
        "embedding_dim": OUTPUT_UNITS,
    }

    for field, expected in expected_scalars.items():
        actual = report.get(field)

        if actual != expected:
            raise ValueError(
                f"Report mismatch: {field}={actual!r}, "
                f"expected {expected!r}."
            )

    if report.get("input_shape") != [1, INPUT_FEATURES]:
        raise ValueError(
            "Unexpected Split-1 input shape in report."
        )

    if report.get("output_shape") != [1, OUTPUT_UNITS]:
        raise ValueError(
            "Unexpected Split-1 output shape in report."
        )

    desktop_parity = report.get("desktop_parity")

    if not isinstance(desktop_parity, dict):
        raise ValueError(
            "Missing desktop_parity section."
        )

    if desktop_parity.get("pass") is not True:
        raise ValueError(
            "Refusing firmware generation because desktop parity "
            "has not passed."
        )

    prefix = report.get("prefix_tflite")

    if not isinstance(prefix, dict):
        raise ValueError(
            "Missing prefix_tflite section."
        )

    if prefix.get("filename") != MODEL_PATH.name:
        raise ValueError(
            "Split-1 TFLite filename mismatch."
        )

    if prefix.get("bytes") != len(model_bytes):
        raise ValueError(
            "Split-1 TFLite byte count mismatch."
        )

    actual_sha = sha256_bytes(model_bytes)

    if prefix.get("sha256") != actual_sha:
        raise ValueError(
            "Split-1 TFLite SHA-256 mismatch."
        )


def validate_parity_vectors(
    payload: np.lib.npyio.NpzFile,
    report: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "validation_indices",
        "true_classes",
        "normalized_inputs",
        "expected_split1",
    }

    missing = required.difference(payload.files)

    if missing:
        raise ValueError(
            "Missing Split-1 parity arrays: "
            + ", ".join(sorted(missing))
        )

    validation_indices = np.asarray(
        payload["validation_indices"],
        dtype=np.int64,
    )

    true_classes = np.asarray(
        payload["true_classes"],
        dtype=np.int64,
    )

    normalized_inputs = np.asarray(
        payload["normalized_inputs"],
        dtype=np.float32,
    )

    expected_outputs = np.asarray(
        payload["expected_split1"],
        dtype=np.float32,
    )

    vector_count = validation_indices.shape[0]

    if vector_count <= 0:
        raise ValueError(
            "Split-1 parity vector set is empty."
        )

    if true_classes.shape != (vector_count,):
        raise ValueError(
            "Unexpected true_classes shape."
        )

    if normalized_inputs.shape != (
        vector_count,
        INPUT_FEATURES,
    ):
        raise ValueError(
            f"Unexpected normalized_inputs shape: "
            f"{normalized_inputs.shape}"
        )

    if expected_outputs.shape != (
        vector_count,
        OUTPUT_UNITS,
    ):
        raise ValueError(
            f"Unexpected expected_split1 shape: "
            f"{expected_outputs.shape}"
        )

    if not np.all(np.isfinite(normalized_inputs)):
        raise ValueError(
            "normalized_inputs contains non-finite values."
        )

    if not np.all(np.isfinite(expected_outputs)):
        raise ValueError(
            "expected_split1 contains non-finite values."
        )

    report_vectors = report.get("parity_vectors")

    if not isinstance(report_vectors, dict):
        raise ValueError(
            "Missing parity_vectors report section."
        )

    if report_vectors.get("count") != vector_count:
        raise ValueError(
            "Parity-vector count does not match report."
        )

    if report_vectors.get(
        "validation_indices"
    ) != validation_indices.astype(int).tolist():
        raise ValueError(
            "Parity validation indices do not match report."
        )

    return (
        validation_indices,
        true_classes,
        normalized_inputs,
        expected_outputs,
    )


def render_model_header(
    model_len: int,
) -> str:
    return f'''#pragma once

#include <stddef.h>

namespace split1_model_data {{

constexpr size_t SPLIT_ID = {SPLIT_ID};
constexpr size_t INPUT_FEATURES = {INPUT_FEATURES};
constexpr size_t OUTPUT_UNITS = {OUTPUT_UNITS};
constexpr size_t EXPECTED_MODEL_LEN = {model_len};

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char SOURCE_MODEL_VERSION[];
extern const char MODEL_SHA256[];

}}  // namespace split1_model_data
'''


def render_model_source(
    model_bytes: bytes,
    model_sha256: str,
) -> str:
    byte_block = bytes_to_cpp(model_bytes)

    return f'''#include "split1_model_data.h"

namespace split1_model_data {{

alignas(16) const unsigned char MODEL[] = {{
{byte_block}
}};

const size_t MODEL_LEN = sizeof(MODEL);

const char SOURCE_MODEL_VERSION[] =
    "{SOURCE_MODEL_VERSION}";

const char MODEL_SHA256[] =
    "{model_sha256}";

static_assert(
    MODEL_LEN == EXPECTED_MODEL_LEN,
    "Split-1 model byte count mismatch."
);

}}  // namespace split1_model_data
'''


def render_parity_header(
    validation_indices: np.ndarray,
    true_classes: np.ndarray,
    normalized_inputs: np.ndarray,
    expected_outputs: np.ndarray,
) -> str:
    vector_count = validation_indices.shape[0]

    input_rows = matrix_to_cpp(
        normalized_inputs
    )
    output_rows = matrix_to_cpp(
        expected_outputs
    )

    validation_values = integer_array_to_cpp(
        validation_indices
    )
    class_values = integer_array_to_cpp(
        true_classes
    )

    return f'''#pragma once

#include <stddef.h>
#include <stdint.h>

namespace split1_parity_vectors {{

constexpr size_t VECTOR_COUNT = {vector_count};
constexpr size_t INPUT_FEATURES = {INPUT_FEATURES};
constexpr size_t OUTPUT_UNITS = {OUTPUT_UNITS};

constexpr int VALIDATION_INDICES[VECTOR_COUNT] = {{
    {validation_values}
}};

constexpr int TRUE_CLASSES[VECTOR_COUNT] = {{
    {class_values}
}};

constexpr float NORMALIZED_INPUTS
    [VECTOR_COUNT][INPUT_FEATURES] = {{
{input_rows}
}};

constexpr float EXPECTED_OUTPUTS
    [VECTOR_COUNT][OUTPUT_UNITS] = {{
{output_rows}
}};

}}  // namespace split1_parity_vectors
'''


def write_new_file(
    path: Path,
    content: str,
) -> None:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing generated file: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    root = project_root()

    report_path = root / REPORT_PATH
    model_path = root / MODEL_PATH
    parity_path = root / PARITY_PATH

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Split-1 report not found: {report_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Split-1 TFLite not found: {model_path}"
        )

    if not parity_path.is_file():
        raise FileNotFoundError(
            f"Split-1 parity vectors not found: {parity_path}"
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    model_bytes = model_path.read_bytes()

    validate_report(
        report,
        model_bytes,
    )

    with np.load(
        parity_path,
        allow_pickle=False,
    ) as payload:
        (
            validation_indices,
            true_classes,
            normalized_inputs,
            expected_outputs,
        ) = validate_parity_vectors(
            payload,
            report,
        )

    model_sha256 = sha256_bytes(
        model_bytes
    )

    model_header_path = (
        root / MODEL_HEADER_PATH
    )
    model_source_path = (
        root / MODEL_SOURCE_PATH
    )
    parity_header_path = (
        root / PARITY_HEADER_PATH
    )

    write_new_file(
        model_header_path,
        render_model_header(
            len(model_bytes),
        ),
    )

    write_new_file(
        model_source_path,
        render_model_source(
            model_bytes,
            model_sha256,
        ),
    )

    write_new_file(
        parity_header_path,
        render_parity_header(
            validation_indices,
            true_classes,
            normalized_inputs,
            expected_outputs,
        ),
    )

    print()
    print(
        "PHASE 7 / M8 — SPLIT 1 FIRMWARE ASSETS GENERATED"
    )
    print(
        "================================================"
    )
    print(
        f"Source model version:      {SOURCE_MODEL_VERSION}"
    )
    print(
        f"Split:                     {SPLIT_ID}"
    )
    print(
        f"Input/output:              "
        f"{INPUT_FEATURES} -> {OUTPUT_UNITS}"
    )
    print(
        f"TFLite bytes:              {len(model_bytes)}"
    )
    print(
        f"TFLite SHA-256:            {model_sha256}"
    )
    print(
        f"Parity vectors:            "
        f"{validation_indices.shape[0]}"
    )
    print(
        "Existing B3 runtime changed: NO"
    )
    print()
    print(
        f"Model header:  {model_header_path}"
    )
    print(
        f"Model source:  {model_source_path}"
    )
    print(
        f"Parity header: {parity_header_path}"
    )


if __name__ == "__main__":
    main()
