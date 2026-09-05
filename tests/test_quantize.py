import numpy as np
import pytest

from ml.export.quantize import (
    dequantize_tensor,
    quantize_tensor,
)


def test_quantize_tensor_known_values() -> None:
    values = np.asarray(
        [-1.0, 0.0, 1.0],
        dtype=np.float32,
    )

    quantized = quantize_tensor(
        values,
        scale=0.1,
        zero_point=0,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        quantized,
        np.asarray(
            [-10, 0, 10],
            dtype=np.int8,
        ),
    )


def test_quantize_tensor_clips_to_int8() -> None:
    values = np.asarray(
        [-1000.0, 1000.0],
        dtype=np.float32,
    )

    quantized = quantize_tensor(
        values,
        scale=0.1,
        zero_point=0,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        quantized,
        np.asarray(
            [-128, 127],
            dtype=np.int8,
        ),
    )


def test_dequantize_tensor_known_values() -> None:
    values = np.asarray(
        [-10, 0, 10],
        dtype=np.int8,
    )

    result = dequantize_tensor(
        values,
        scale=0.1,
        zero_point=0,
    )

    np.testing.assert_allclose(
        result,
        np.asarray(
            [-1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_invalid_quantization_scale() -> None:
    values = np.zeros(
        3,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        quantize_tensor(
            values,
            scale=0.0,
            zero_point=0,
            dtype=np.int8,
        )

    with pytest.raises(ValueError):
        dequantize_tensor(
            values,
            scale=0.0,
            zero_point=0,
        )

def test_normalize_features_known_values() -> None:
    from ml.export.quantize import (
        normalize_features,
    )

    values = np.asarray(
        [[
            3.0,
            7.0,
            5.0,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            21.0,
        ]],
        dtype=np.float32,
    )

    mean = np.asarray(
        [
            1.0,
            3.0,
            1.0,
            5.0,
            7.0,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
        ],
        dtype=np.float32,
    )

    variance = np.asarray(
        [
            4.0,
            16.0,
            16.0,
            16.0,
            16.0,
            16.0,
            16.0,
            16.0,
            16.0,
            16.0,
        ],
        dtype=np.float32,
    )

    result = normalize_features(
        values,
        mean=mean,
        variance=variance,
    )

    expected = np.ones(
        (1, 10),
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-7,
    )