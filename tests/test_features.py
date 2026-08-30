import numpy as np

from ml.features.features_v1 import FEATURE_NAMES, extract_features_v1


def test_feature_vector_shape_and_finiteness():
    rng = np.random.default_rng(123)
    window = rng.normal(size=(100, 6))
    features = extract_features_v1(window)
    assert features.shape == (10,)
    assert len(FEATURE_NAMES) == 10
    assert np.isfinite(features).all()


def test_zero_window_is_stable():
    window = np.zeros((100, 6), dtype=np.float32)
    features = extract_features_v1(window)
    assert np.isfinite(features).all()
    assert np.allclose(features, 0.0)


def test_invalid_shape_rejected():
    bad = np.zeros((100, 5))
    try:
        extract_features_v1(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid input shape was not rejected.")
