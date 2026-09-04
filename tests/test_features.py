from __future__ import annotations

import numpy as np
import pytest

from ml.features.features_v1 import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_VERSION,
    STD_DDOF,
    extract_features_v1,
)


def test_feature_contract_constants() -> None:
    assert FEATURE_VERSION == "features-v1"
    assert FEATURE_COUNT == 10
    assert STD_DDOF == 0

    assert FEATURE_NAMES == (
        "std_ax",
        "max_abs_ax",
        "ax_half_mean_delta",
        "std_ay",
        "std_az",
        "rms_acc_mag_deviation",
        "mean_gz",
        "std_gz",
        "rms_gyro_mag",
        "max_gyro_mag",
    )


def test_zero_window_is_exactly_zero() -> None:
    window = np.zeros((100, 6), dtype=np.float32)

    features = extract_features_v1(window)

    assert features.shape == (10,)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(
        features,
        np.zeros(10, dtype=np.float32),
    )


def test_constant_gravity_window_has_expected_features() -> None:
    window = np.zeros((100, 6), dtype=np.float64)
    window[:, 2] = 1.0

    features = extract_features_v1(window)

    expected = np.zeros(10, dtype=np.float32)

    np.testing.assert_allclose(
        features,
        expected,
        rtol=0.0,
        atol=1e-7,
    )


def test_hand_computable_feature_vector() -> None:
    window = np.zeros((100, 6), dtype=np.float64)

    window[:50, 0] = 1.0
    window[50:, 0] = -1.0

    window[:, 1] = 2.0
    window[:, 2] = 0.0

    window[:50, 5] = 3.0
    window[50:, 5] = -3.0

    features = extract_features_v1(window)

    expected = np.asarray(
        [
            1.0,
            1.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            3.0,
            3.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        features,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_population_standard_deviation_is_used() -> None:
    window = np.zeros((100, 6), dtype=np.float64)
    window[:, 0] = np.arange(100, dtype=np.float64)

    features = extract_features_v1(window)

    expected_population_std = np.std(
        window[:, 0],
        ddof=0,
    )
    sample_std = np.std(
        window[:, 0],
        ddof=1,
    )

    assert np.isclose(
        features[0],
        expected_population_std,
    )
    assert not np.isclose(
        features[0],
        sample_std,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (99, 6),
        (101, 6),
        (100, 5),
        (100, 7),
        (600,),
    ],
)
def test_invalid_window_shape_is_rejected(
    shape: tuple[int, ...],
) -> None:
    window = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        extract_features_v1(window)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_input_is_rejected(
    bad_value: float,
) -> None:
    window = np.zeros((100, 6), dtype=np.float32)
    window[17, 2] = bad_value

    with pytest.raises(ValueError):
        extract_features_v1(window)