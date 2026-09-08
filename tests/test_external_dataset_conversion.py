import hashlib
import numpy as np
import pytest
from tools.external_dataset import convert_kaggle_dataset as conversion
from tests.test_external_dataset_inspection import make_csv


def test_conversion_preserves_axes_times_labels_and_input(tmp_path, monkeypatch):
    monkeypatch.setattr(conversion, "EXTERNAL_ROOT", tmp_path / "external")
    path = tmp_path / "source.csv"
    make_csv(path, [["a", 0, t, -1, 2, 3, -4, 5, 6] for t in (12, 32, 52)])
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "external/converted"
    conversion.convert(path, output)
    data, metadata = conversion.load_converted(output)
    np.testing.assert_array_equal(data["sensor_values"], [[-1, 2, 3, -4, 5, 6]] * 3)
    np.testing.assert_array_equal(data["timestamps_ms"], [12, 32, 52])
    assert metadata["axis_orientation"] == "unknown"
    assert metadata["resampling"] == "none"
    assert original == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        conversion.convert(path, output)
    conversion.convert(path, output, rebuild=True)
    rebuilt, _ = conversion.load_converted(output)
    np.testing.assert_array_equal(rebuilt["sensor_values"], data["sensor_values"])


def test_production_output_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="data/external"):
        conversion.convert(tmp_path / "absent.csv", conversion.ROOT / "data/raw/forbidden")
