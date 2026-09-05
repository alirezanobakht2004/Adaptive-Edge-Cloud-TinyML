from __future__ import annotations

import numpy as np
import pytest

from ml.export.generate_edge_uncertainty_firmware import (
    B3_UNITS,
    CLASS_COUNT,
    DROPOUT_RATE,
    MC_PASSES,
    format_float32,
    format_hex_bytes,
    validate_edge_head_arrays,
    validate_parity_arrays,
)


def test_format_float32_emits_float_suffix() -> None:
    assert (
        format_float32(
            np.float32(
                0.125
            )
        )
        == "0.125f"
    )

    assert (
        format_float32(
            np.float32(
                0.0
            )
        )
        == "0.0f"
    )


def test_format_hex_bytes_is_stable() -> None:
    assert format_hex_bytes(
        bytes(
            [
                0,
                1,
                255,
            ]
        ),
        per_line=2,
    ) == (
        "    0x00, 0x01,\n"
        "    0xff,"
    )


def test_validate_edge_head_arrays_accepts_contract_shapes() -> None:
    validate_edge_head_arrays(
        np.zeros(
            (
                B3_UNITS,
                CLASS_COUNT,
            ),
            dtype=np.float32,
        ),
        np.zeros(
            CLASS_COUNT,
            dtype=np.float32,
        ),
        DROPOUT_RATE,
    )


def test_validate_parity_arrays_rejects_non_binary_masks() -> None:
    vector_count = (
        CLASS_COUNT
    )

    kwargs = {
        "validation_indices":
            np.arange(
                vector_count,
                dtype=np.int64,
            ),
        "true_classes":
            np.arange(
                vector_count,
                dtype=np.int64,
            ),
        "normalized_inputs":
            np.zeros(
                (
                    vector_count,
                    10,
                ),
                dtype=np.float32,
            ),
        "expected_b3":
            np.zeros(
                (
                    vector_count,
                    B3_UNITS,
                ),
                dtype=np.float32,
            ),
        "keep_masks":
            np.ones(
                (
                    vector_count,
                    MC_PASSES,
                    B3_UNITS,
                ),
                dtype=np.uint8,
            ),
        "expected_pass_probabilities":
            np.full(
                (
                    vector_count,
                    MC_PASSES,
                    CLASS_COUNT,
                ),
                1.0 / CLASS_COUNT,
                dtype=np.float32,
            ),
        "expected_mean_probabilities":
            np.full(
                (
                    vector_count,
                    CLASS_COUNT,
                ),
                1.0 / CLASS_COUNT,
                dtype=np.float32,
            ),
        "expected_uncertainty_score":
            np.zeros(
                vector_count,
                dtype=np.float32,
            ),
        "expected_mean_class_variance":
            np.zeros(
                vector_count,
                dtype=np.float32,
            ),
        "expected_max_class_variance":
            np.zeros(
                vector_count,
                dtype=np.float32,
            ),
    }

    kwargs[
        "keep_masks"
    ][0, 0, 0] = 2

    with pytest.raises(
        ValueError,
        match="0/1",
    ):
        validate_parity_arrays(
            **kwargs
        )
