import numpy as np
import pytest
from ml.features.features_v1 import extract_features_v1
from tools.external_dataset.run_feature_compatibility import resample_diagnostic


def test_native_windows_are_rejected_and_adapter_is_explicit():
    times = np.arange(50) * 20
    window = np.tile(np.arange(50)[:, None], (1, 6)).astype(float)
    with pytest.raises(ValueError, match="Expected window shape"):
        extract_features_v1(window)
    adapted = resample_diagnostic(window, times)
    np.testing.assert_array_equal(adapted[::2], window)
    np.testing.assert_array_equal(adapted[-1], window[-1])
    assert adapted[1, 0] == .5
    assert extract_features_v1(adapted).shape == (10,)


def test_adapter_never_silently_repairs_irregular_timing():
    times = np.arange(50) * 20
    times[10] += 1
    with pytest.raises(ValueError, match="20 ms"):
        resample_diagnostic(np.zeros((50, 6)), times)
