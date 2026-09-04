from __future__ import annotations

import numpy as np


FEATURE_VERSION = "features-v1"

WINDOW_SAMPLES = 100
SENSOR_CHANNELS = 6
FEATURE_COUNT = 10
STD_DDOF = 0

FEATURE_NAMES = (
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


def extract_features_v1(window: np.ndarray) -> np.ndarray:
    """Extract the canonical features-v1 vector from one IMU window.

    Input shape:
        (100, 6)

    Column order:
        ax, ay, az, gx, gy, gz

    Units:
        acceleration: g
        gyroscope: degrees/second

    Numerical contract:
        - Standard deviations use population convention (ddof=0).
        - First half is samples [0:50].
        - Second half is samples [50:100].
        - Output dtype is float32.
    """

    x = np.asarray(window, dtype=np.float64)

    expected_shape = (WINDOW_SAMPLES, SENSOR_CHANNELS)

    if x.shape != expected_shape:
        raise ValueError(
            f"Expected window shape {expected_shape}, got {x.shape}."
        )

    if not np.isfinite(x).all():
        raise ValueError("Window contains NaN or infinite values.")

    ax, ay, az, gx, gy, gz = x.T

    half = WINDOW_SAMPLES // 2

    acc_mag = np.sqrt(
        ax * ax
        + ay * ay
        + az * az
    )

    gyro_mag = np.sqrt(
        gx * gx
        + gy * gy
        + gz * gz
    )

    acc_mag_mean = np.mean(acc_mag)

    features = np.asarray(
        [
            np.std(ax, ddof=STD_DDOF),
            np.max(np.abs(ax)),
            np.mean(ax[:half]) - np.mean(ax[half:]),
            np.std(ay, ddof=STD_DDOF),
            np.std(az, ddof=STD_DDOF),
            np.sqrt(
                np.mean(
                    (acc_mag - acc_mag_mean) ** 2
                )
            ),
            np.mean(gz),
            np.std(gz, ddof=STD_DDOF),
            np.sqrt(np.mean(gyro_mag ** 2)),
            np.max(gyro_mag),
        ],
        dtype=np.float32,
    )

    if features.shape != (FEATURE_COUNT,):
        raise RuntimeError(
            f"Unexpected feature shape: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature extractor produced non-finite values."
        )

    return features


__all__ = [
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "SENSOR_CHANNELS",
    "STD_DDOF",
    "WINDOW_SAMPLES",
    "extract_features_v1",
]