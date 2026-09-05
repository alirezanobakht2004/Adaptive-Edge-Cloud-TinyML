
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_VERSION = "dataset-v1"
FEATURE_VERSION = "features-v1"
MODEL_VERSION = "gesture-model-v1.1.0"

INPUT_FEATURES = 10
B3_UNITS = 32
CLASS_COUNT = 5
MC_PASSES = 5
DROPOUT_RATE = 0.2

MODEL_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{MODEL_VERSION}/"
    "tflite"
)

PREFIX_FILENAME = (
    f"{MODEL_VERSION}"
    "-prefix-b3-float32-normalized-input.tflite"
)

EDGE_HEAD_FILENAME = (
    "edge_head_float32_weights.npz"
)

PARITY_FILENAME = (
    "edge_uncertainty_parity_vectors.npz"
)

REPORT_FILENAME = (
    "edge_uncertainty_deployment_report.json"
)

PREFIX_HEADER = Path(
    "firmware/include/prefix_model_data.h"
)

PREFIX_SOURCE = Path(
    "firmware/src/inference/prefix_model_data.cpp"
)

EDGE_HEAD_HEADER = Path(
    "firmware/include/edge_head_data.h"
)

EDGE_HEAD_SOURCE = Path(
    "firmware/src/inference/edge_head_data.cpp"
)

PARITY_HEADER = Path(
    "firmware/test/test_edge_uncertainty_parity/"
    "edge_uncertainty_parity_vectors.h"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def format_float32(
    value: float | np.floating,
) -> str:
    '''Return a C++ float literal that round-trips float32.'''

    cast = np.float32(value)

    if not np.isfinite(cast):
        raise ValueError(
            "Cannot emit non-finite float."
        )

    if cast == np.float32(0.0):
        return "0.0f"

    return (
        f"{format(float(cast), '.9g')}f"
    )


def format_uint8(
    value: int | np.integer,
) -> str:
    cast = int(value)

    if cast < 0 or cast > 255:
        raise ValueError(
            "uint8 literal is outside [0,255]."
        )

    return str(cast)


def format_hex_bytes(
    data: bytes,
    *,
    per_line: int = 12,
) -> str:
    if per_line <= 0:
        raise ValueError(
            "per_line must be positive."
        )

    values = [
        f"0x{byte:02x}"
        for byte in data
    ]

    lines: list[str] = []

    for start in range(
        0,
        len(values),
        per_line,
    ):
        lines.append(
            "    "
            + ", ".join(
                values[
                    start:
                    start + per_line
                ]
            )
            + ","
        )

    return "\n".join(lines)


def format_flat_values(
    values: Iterable,
    *,
    formatter,
    per_line: int,
    indent: str = "    ",
) -> str:
    materialized = [
        formatter(value)
        for value in values
    ]

    lines: list[str] = []

    for start in range(
        0,
        len(materialized),
        per_line,
    ):
        lines.append(
            indent
            + ", ".join(
                materialized[
                    start:
                    start + per_line
                ]
            )
            + ","
        )

    return "\n".join(lines)


def validate_edge_head_arrays(
    kernel: np.ndarray,
    bias: np.ndarray,
    dropout_rate: float,
) -> None:
    if kernel.shape != (
        B3_UNITS,
        CLASS_COUNT,
    ):
        raise ValueError(
            "Unexpected edge-head kernel shape: "
            f"{kernel.shape}."
        )

    if bias.shape != (
        CLASS_COUNT,
    ):
        raise ValueError(
            "Unexpected edge-head bias shape: "
            f"{bias.shape}."
        )

    if (
        not np.isfinite(kernel).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError(
            "Edge-head weights contain non-finite values."
        )

    if not np.isclose(
        dropout_rate,
        DROPOUT_RATE,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "Unexpected dropout rate: "
            f"{dropout_rate}."
        )


def validate_parity_arrays(
    *,
    validation_indices: np.ndarray,
    true_classes: np.ndarray,
    normalized_inputs: np.ndarray,
    expected_b3: np.ndarray,
    keep_masks: np.ndarray,
    expected_pass_probabilities: np.ndarray,
    expected_mean_probabilities: np.ndarray,
    expected_uncertainty_score: np.ndarray,
    expected_mean_class_variance: np.ndarray,
    expected_max_class_variance: np.ndarray,
) -> None:
    vector_count = CLASS_COUNT

    expected_shapes = {
        "validation_indices":
            (vector_count,),
        "true_classes":
            (vector_count,),
        "normalized_inputs":
            (
                vector_count,
                INPUT_FEATURES,
            ),
        "expected_b3":
            (
                vector_count,
                B3_UNITS,
            ),
        "keep_masks":
            (
                vector_count,
                MC_PASSES,
                B3_UNITS,
            ),
        "expected_pass_probabilities":
            (
                vector_count,
                MC_PASSES,
                CLASS_COUNT,
            ),
        "expected_mean_probabilities":
            (
                vector_count,
                CLASS_COUNT,
            ),
        "expected_uncertainty_score":
            (vector_count,),
        "expected_mean_class_variance":
            (vector_count,),
        "expected_max_class_variance":
            (vector_count,),
    }

    actuals = {
        "validation_indices":
            validation_indices,
        "true_classes":
            true_classes,
        "normalized_inputs":
            normalized_inputs,
        "expected_b3":
            expected_b3,
        "keep_masks":
            keep_masks,
        "expected_pass_probabilities":
            expected_pass_probabilities,
        "expected_mean_probabilities":
            expected_mean_probabilities,
        "expected_uncertainty_score":
            expected_uncertainty_score,
        "expected_mean_class_variance":
            expected_mean_class_variance,
        "expected_max_class_variance":
            expected_max_class_variance,
    }

    for name, expected_shape in (
        expected_shapes.items()
    ):
        if actuals[name].shape != (
            expected_shape
        ):
            raise ValueError(
                f"{name} has shape "
                f"{actuals[name].shape}; "
                f"expected {expected_shape}."
            )

    for name in (
        "normalized_inputs",
        "expected_b3",
        "expected_pass_probabilities",
        "expected_mean_probabilities",
        "expected_uncertainty_score",
        "expected_mean_class_variance",
        "expected_max_class_variance",
    ):
        if not np.isfinite(
            actuals[name]
        ).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if not np.all(
        (keep_masks == 0)
        | (keep_masks == 1)
    ):
        raise ValueError(
            "keep_masks must contain only 0/1."
        )

    if (
        np.any(true_classes < 0)
        or np.any(
            true_classes
            >= CLASS_COUNT
        )
    ):
        raise ValueError(
            "true_classes contains invalid ids."
        )

    if not np.allclose(
        np.sum(
            expected_pass_probabilities,
            axis=2,
        ),
        1.0,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Expected pass probabilities do not sum to 1."
        )

    if not np.allclose(
        np.sum(
            expected_mean_probabilities,
            axis=1,
        ),
        1.0,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Expected mean probabilities do not sum to 1."
        )


def inspect_prefix_ops(
    prefix_path: Path,
) -> tuple[str, ...]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=str(
            prefix_path
        ),
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

    if (
        len(inputs) != 1
        or len(outputs) != 1
    ):
        raise ValueError(
            "Prefix TFLite must have exactly one input/output."
        )

    if tuple(
        inputs[0]["shape"]
    ) != (
        1,
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Unexpected prefix input shape."
        )

    if tuple(
        outputs[0]["shape"]
    ) != (
        1,
        B3_UNITS,
    ):
        raise ValueError(
            "Unexpected prefix output shape."
        )

    if (
        inputs[0]["dtype"]
        != np.float32
        or outputs[0]["dtype"]
        != np.float32
    ):
        raise ValueError(
            "Prefix TFLite must be Float32."
        )

    ops = tuple(
        sorted(
            {
                str(
                    detail[
                        "op_name"
                    ]
                )
                for detail
                in interpreter._get_ops_details()
                if detail.get(
                    "op_name"
                )
                != "DELEGATE"
            }
        )
    )

    if ops != (
        "FULLY_CONNECTED",
    ):
        raise RuntimeError(
            "Unexpected prefix TFLite ops. "
            "Firmware resolver is intentionally limited "
            f"to FULLY_CONNECTED, but found: {ops}"
        )

    return ops


def render_prefix_header() -> str:
    return f'''#pragma once

#include <stddef.h>

namespace prefix_model_data {{

constexpr size_t INPUT_FEATURES = {INPUT_FEATURES};
constexpr size_t OUTPUT_UNITS = {B3_UNITS};

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char MODEL_VERSION[];
extern const char MODEL_SHA256[];

}}  // namespace prefix_model_data
'''


def render_prefix_source(
    *,
    model_bytes: bytes,
    model_sha256: str,
) -> str:
    byte_lines = format_hex_bytes(
        model_bytes
    )

    return f'''#include "prefix_model_data.h"

namespace prefix_model_data {{

alignas(16) const unsigned char MODEL[] = {{
{byte_lines}
}};

const size_t MODEL_LEN = sizeof(MODEL);

const char MODEL_VERSION[] =
    "{MODEL_VERSION}-prefix-b3-float32-normalized-input";

const char MODEL_SHA256[] =
    "{model_sha256}";

}}  // namespace prefix_model_data
'''


def render_edge_head_header() -> str:
    return f'''#pragma once

#include <stddef.h>

namespace edge_head_data {{

constexpr size_t INPUT_UNITS = {B3_UNITS};
constexpr size_t CLASS_COUNT = {CLASS_COUNT};
constexpr float DROPOUT_RATE = 0.2f;
constexpr float KEEP_PROBABILITY = 0.8f;
constexpr float INVERTED_DROPOUT_SCALE = 1.25f;

extern const float KERNEL[INPUT_UNITS][CLASS_COUNT];
extern const float BIAS[CLASS_COUNT];

}}  // namespace edge_head_data
'''


def render_edge_head_source(
    *,
    kernel: np.ndarray,
    bias: np.ndarray,
) -> str:
    kernel_rows: list[str] = []

    for row in kernel:
        values = ", ".join(
            format_float32(value)
            for value in row
        )

        kernel_rows.append(
            "    {"
            + values
            + "},"
        )

    bias_values = format_flat_values(
        bias,
        formatter=format_float32,
        per_line=5,
    )

    return f'''#include "edge_head_data.h"

namespace edge_head_data {{

const float KERNEL[INPUT_UNITS][CLASS_COUNT] = {{
{chr(10).join(kernel_rows)}
}};

const float BIAS[CLASS_COUNT] = {{
{bias_values}
}};

}}  // namespace edge_head_data
'''


def render_parity_header(
    *,
    validation_indices: np.ndarray,
    true_classes: np.ndarray,
    normalized_inputs: np.ndarray,
    expected_b3: np.ndarray,
    keep_masks: np.ndarray,
    expected_pass_probabilities: np.ndarray,
    expected_mean_probabilities: np.ndarray,
    expected_uncertainty_score: np.ndarray,
    expected_mean_class_variance: np.ndarray,
    expected_max_class_variance: np.ndarray,
) -> str:
    def render_2d_float(
        name: str,
        array: np.ndarray,
        second_dim: int,
    ) -> str:
        rows: list[str] = []

        for row in array:
            rows.append(
                "    {"
                + ", ".join(
                    format_float32(v)
                    for v in row
                )
                + "},"
            )

        return (
            f"static const float {name}"
            f"[VECTOR_COUNT][{second_dim}] = {{\n"
            + "\n".join(rows)
            + "\n};"
        )

    mask_vectors: list[str] = []

    for vector in keep_masks:
        pass_rows: list[str] = []

        for mask in vector:
            pass_rows.append(
                "        {"
                + ", ".join(
                    format_uint8(v)
                    for v in mask
                )
                + "},"
            )

        mask_vectors.append(
            "    {\n"
            + "\n".join(
                pass_rows
            )
            + "\n    },"
        )

    probability_vectors: list[str] = []

    for vector in expected_pass_probabilities:
        pass_rows: list[str] = []

        for probs in vector:
            pass_rows.append(
                "        {"
                + ", ".join(
                    format_float32(v)
                    for v in probs
                )
                + "},"
            )

        probability_vectors.append(
            "    {\n"
            + "\n".join(
                pass_rows
            )
            + "\n    },"
        )

    validation_values = ", ".join(
        str(
            int(v)
        )
        for v in validation_indices
    )

    class_values = ", ".join(
        str(
            int(v)
        )
        for v in true_classes
    )

    score_values = format_flat_values(
        expected_uncertainty_score,
        formatter=format_float32,
        per_line=5,
    )

    mean_variance_values = (
        format_flat_values(
            expected_mean_class_variance,
            formatter=format_float32,
            per_line=5,
        )
    )

    max_variance_values = (
        format_flat_values(
            expected_max_class_variance,
            formatter=format_float32,
            per_line=5,
        )
    )

    return f'''#pragma once

#include <stddef.h>
#include <stdint.h>

namespace edge_uncertainty_parity {{

constexpr size_t VECTOR_COUNT = {CLASS_COUNT};
constexpr size_t INPUT_FEATURES = {INPUT_FEATURES};
constexpr size_t B3_UNITS = {B3_UNITS};
constexpr size_t PASS_COUNT = {MC_PASSES};
constexpr size_t CLASS_COUNT = {CLASS_COUNT};

static const int VALIDATION_INDICES[VECTOR_COUNT] = {{
    {validation_values}
}};

static const int TRUE_CLASSES[VECTOR_COUNT] = {{
    {class_values}
}};

{render_2d_float(
    "NORMALIZED_INPUTS",
    normalized_inputs,
    INPUT_FEATURES,
)}

{render_2d_float(
    "EXPECTED_B3",
    expected_b3,
    B3_UNITS,
)}

static const uint8_t KEEP_MASKS
[VECTOR_COUNT][PASS_COUNT][B3_UNITS] = {{
{chr(10).join(mask_vectors)}
}};

static const float EXPECTED_PASS_PROBABILITIES
[VECTOR_COUNT][PASS_COUNT][CLASS_COUNT] = {{
{chr(10).join(probability_vectors)}
}};

{render_2d_float(
    "EXPECTED_MEAN_PROBABILITIES",
    expected_mean_probabilities,
    CLASS_COUNT,
)}

static const float EXPECTED_UNCERTAINTY_SCORE[VECTOR_COUNT] = {{
{score_values}
}};

static const float EXPECTED_MEAN_CLASS_VARIANCE[VECTOR_COUNT] = {{
{mean_variance_values}
}};

static const float EXPECTED_MAX_CLASS_VARIANCE[VECTOR_COUNT] = {{
{max_variance_values}
}};

}}  // namespace edge_uncertainty_parity
'''


def ensure_targets_are_new(
    root: Path,
) -> None:
    targets = (
        PREFIX_HEADER,
        PREFIX_SOURCE,
        EDGE_HEAD_HEADER,
        EDGE_HEAD_SOURCE,
        PARITY_HEADER,
    )

    existing = [
        root / path
        for path in targets
        if (
            root
            / path
        ).exists()
    ]

    if existing:
        joined = ", ".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Refusing to overwrite generated firmware assets: "
            f"{joined}"
        )


def write_text(
    root: Path,
    relative_path: Path,
    content: str,
) -> None:
    path = (
        root
        / relative_path
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
    model_dir = root / MODEL_DIR

    prefix_path = (
        model_dir
        / PREFIX_FILENAME
    )

    edge_head_path = (
        model_dir
        / EDGE_HEAD_FILENAME
    )

    parity_path = (
        model_dir
        / PARITY_FILENAME
    )

    report_path = (
        model_dir
        / REPORT_FILENAME
    )

    for path in (
        prefix_path,
        edge_head_path,
        parity_path,
        report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "Required frozen deployment asset not found: "
                f"{path}"
            )

    ensure_targets_are_new(
        root
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if report.get(
        "test_split_used"
    ) is not False:
        raise ValueError(
            "Deployment report does not confirm TEST stayed locked."
        )

    expected_prefix_sha = (
        report[
            "prefix_tflite"
        ][
            "sha256"
        ]
    )

    actual_prefix_sha = (
        sha256_file(
            prefix_path
        )
    )

    if actual_prefix_sha != (
        expected_prefix_sha
    ):
        raise RuntimeError(
            "Frozen prefix SHA-256 does not match deployment report."
        )

    prefix_ops = (
        inspect_prefix_ops(
            prefix_path
        )
    )

    with np.load(
        edge_head_path,
        allow_pickle=False,
    ) as archive:
        kernel = np.asarray(
            archive["kernel"],
            dtype=np.float32,
        )

        bias = np.asarray(
            archive["bias"],
            dtype=np.float32,
        )

        dropout_rate = float(
            np.asarray(
                archive[
                    "dropout_rate"
                ],
                dtype=np.float32,
            )
        )

    validate_edge_head_arrays(
        kernel,
        bias,
        dropout_rate,
    )

    with np.load(
        parity_path,
        allow_pickle=False,
    ) as archive:
        validation_indices = np.asarray(
            archive[
                "validation_indices"
            ],
            dtype=np.int64,
        )

        true_classes = np.asarray(
            archive[
                "true_classes"
            ],
            dtype=np.int64,
        )

        normalized_inputs = np.asarray(
            archive[
                "normalized_inputs"
            ],
            dtype=np.float32,
        )

        expected_b3 = np.asarray(
            archive[
                "expected_b3"
            ],
            dtype=np.float32,
        )

        keep_masks = np.asarray(
            archive[
                "keep_masks"
            ],
            dtype=np.uint8,
        )

        expected_pass_probabilities = np.asarray(
            archive[
                "expected_pass_probabilities"
            ],
            dtype=np.float32,
        )

        expected_mean_probabilities = np.asarray(
            archive[
                "expected_mean_probabilities"
            ],
            dtype=np.float32,
        )

        expected_uncertainty_score = np.asarray(
            archive[
                "expected_uncertainty_score"
            ],
            dtype=np.float32,
        )

        expected_mean_class_variance = np.asarray(
            archive[
                "expected_mean_class_variance"
            ],
            dtype=np.float32,
        )

        expected_max_class_variance = np.asarray(
            archive[
                "expected_max_class_variance"
            ],
            dtype=np.float32,
        )

    validate_parity_arrays(
        validation_indices=validation_indices,
        true_classes=true_classes,
        normalized_inputs=normalized_inputs,
        expected_b3=expected_b3,
        keep_masks=keep_masks,
        expected_pass_probabilities=(
            expected_pass_probabilities
        ),
        expected_mean_probabilities=(
            expected_mean_probabilities
        ),
        expected_uncertainty_score=(
            expected_uncertainty_score
        ),
        expected_mean_class_variance=(
            expected_mean_class_variance
        ),
        expected_max_class_variance=(
            expected_max_class_variance
        ),
    )

    model_bytes = prefix_path.read_bytes()

    write_text(
        root,
        PREFIX_HEADER,
        render_prefix_header(),
    )

    write_text(
        root,
        PREFIX_SOURCE,
        render_prefix_source(
            model_bytes=model_bytes,
            model_sha256=(
                actual_prefix_sha
            ),
        ),
    )

    write_text(
        root,
        EDGE_HEAD_HEADER,
        render_edge_head_header(),
    )

    write_text(
        root,
        EDGE_HEAD_SOURCE,
        render_edge_head_source(
            kernel=kernel,
            bias=bias,
        ),
    )

    write_text(
        root,
        PARITY_HEADER,
        render_parity_header(
            validation_indices=validation_indices,
            true_classes=true_classes,
            normalized_inputs=normalized_inputs,
            expected_b3=expected_b3,
            keep_masks=keep_masks,
            expected_pass_probabilities=(
                expected_pass_probabilities
            ),
            expected_mean_probabilities=(
                expected_mean_probabilities
            ),
            expected_uncertainty_score=(
                expected_uncertainty_score
            ),
            expected_mean_class_variance=(
                expected_mean_class_variance
            ),
            expected_max_class_variance=(
                expected_max_class_variance
            ),
        ),
    )

    print()
    print(
        "PHASE 5 / M6 — FIRMWARE PARITY ASSETS GENERATED"
    )
    print(
        "=============================================="
    )

    print(
        "Model version:          "
        f"{MODEL_VERSION}"
    )

    print(
        "TEST used:              NO"
    )

    print(
        "Prefix SHA-256:         "
        f"{actual_prefix_sha}"
    )

    print(
        "Prefix bytes:           "
        f"{len(model_bytes)}"
    )

    print(
        "Prefix TFLite ops:      "
        f"{', '.join(prefix_ops)}"
    )

    print(
        "Edge-head kernel:       "
        f"{kernel.shape}"
    )

    print(
        "Edge-head bias:         "
        f"{bias.shape}"
    )

    print(
        "Dropout rate:           "
        f"{dropout_rate}"
    )

    print(
        "Parity vectors:         "
        f"{validation_indices.shape[0]}"
    )

    print(
        "Passes/vector:          "
        f"{MC_PASSES}"
    )

    print()

    for relative_path in (
        PREFIX_HEADER,
        PREFIX_SOURCE,
        EDGE_HEAD_HEADER,
        EDGE_HEAD_SOURCE,
        PARITY_HEADER,
    ):
        print(
            f"Generated: {root / relative_path}"
        )

    print()
    print(
        "FIRMWARE_PARITY_ASSETS_READY"
    )


if __name__ == "__main__":
    main()
