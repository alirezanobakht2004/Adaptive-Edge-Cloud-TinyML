from __future__ import annotations

import numpy as np

FEATURE_VERSION = "features-v1"

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
    """Extract the canonical 10-feature vector from one IMU window.

    Parameters
    ----------
    window:
        Array with shape (N, 6), columns:
        [ax, ay, az, gx, gy, gz].

        Acceleration should use one consistent unit throughout the dataset
        (recommended: g). Gyroscope should use one consistent unit
        (recommended: degrees/second).

    Returns
    -------
    np.ndarray
        float32 vector of shape (10,).
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"Expected window shape (N, 6), got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError("Window must contain at least 2 samples.")
    if not np.isfinite(x).all():
        raise ValueError("Window contains NaN or infinite values.")

    ax, ay, az, gx, gy, gz = x.T
    half = x.shape[0] // 2

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    features = np.array(
        [
            np.std(ax),
            np.max(np.abs(ax)),
            np.mean(ax[:half]) - np.mean(ax[half:]),
            np.std(ay),
            np.std(az),
            np.sqrt(np.mean((acc_mag - np.mean(acc_mag)) ** 2)),
            np.mean(gz),
            np.std(gz),
            np.sqrt(np.mean(gyro_mag ** 2)),
            np.max(gyro_mag),
        ],
        dtype=np.float32,
    )

    if not np.isfinite(features).all():
        raise ValueError("Feature extractor produced non-finite values.")
    return features
