import csv
from tools.external_dataset.inspect_kaggle_dataset import REQUIRED, inspect


def make_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(REQUIRED)
        writer.writerows(rows)


def test_recording_ids_are_scoped_by_label(tmp_path):
    path = tmp_path / "test.csv"
    make_csv(path, [[label, 0, t, 0, 0, 1, 0, 0, 0] for label in ("a", "b") for t in (0, 20)])
    report = inspect(path)
    assert report["row_count"] == 4
    assert report["recording_count"] == 2
    assert report["sample_ids_reused_across_labels"] == 1
    assert report["inferred_sampling_hz"] == 50
    assert report["duplicate_sensor_recordings"] == 1


def test_missing_and_nonfinite_values_are_reported(tmp_path):
    path = tmp_path / "test.csv"
    make_csv(path, [["a", 0, 0, "", 0, 1, "nan", 0, 0]])
    report = inspect(path)
    assert report["missing_values"]["ax"] == 1
    assert report["invalid_numeric_values"]["gx"] == 1
    assert report["recording_count"] is None
