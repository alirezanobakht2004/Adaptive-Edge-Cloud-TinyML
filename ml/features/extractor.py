from __future__ import annotations

import numpy as np

from .features_v1 import FEATURE_NAMES, FEATURE_VERSION, extract_features_v1


def extract_features(window: np.ndarray, version: str = FEATURE_VERSION) -> np.ndarray:
    if version != FEATURE_VERSION:
        raise ValueError(f"Unsupported feature version: {version}")
    return extract_features_v1(window)


__all__ = ["extract_features", "FEATURE_NAMES", "FEATURE_VERSION"]
