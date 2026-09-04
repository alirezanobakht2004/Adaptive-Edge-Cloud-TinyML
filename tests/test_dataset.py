from __future__ import annotations

from collections import Counter

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
    load_manifest,
    load_split,
)


EXPECTED_COUNTS = {
    "train": 600,
    "validation": 200,
    "test": 200,
}

EXPECTED_PER_CLASS = {
    "train": 120,
    "validation": 40,
    "test": 40,
}


def test_manifest_has_frozen_session_based_split() -> None:
    manifest = load_manifest()

    assert manifest["dataset_version"] == "dataset-v1"
    assert manifest["split_strategy"] == "session-based"
    assert manifest["total_samples"] == 1000

    assert manifest["splits"]["train"]["session"] == "session_01"
    assert (
        manifest["splits"]["validation"]["session"]
        == "session_02"
    )
    assert manifest["splits"]["test"]["session"] == "session_03"


def test_class_mapping_is_deterministic() -> None:
    assert GESTURES == (
        "IDLE",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "ROTATE_CW",
        "SHAKE",
    )

    assert CLASS_TO_ID == {
        "IDLE": 0,
        "SWIPE_LEFT": 1,
        "SWIPE_RIGHT": 2,
        "ROTATE_CW": 3,
        "SHAKE": 4,
    }


def test_manifest_contains_no_rejected_capture() -> None:
    manifest = load_manifest()

    for split in manifest["splits"].values():
        for sample in split["samples"]:
            assert "_rejected" not in sample["csv"]
            assert "_rejected" not in sample["metadata"]


def test_all_splits_load_with_expected_shapes_and_counts() -> None:
    for split_name, expected_count in EXPECTED_COUNTS.items():
        split = load_split(split_name)

        assert split.windows.shape == (
            expected_count,
            100,
            6,
        )
        assert split.timestamps_ms.shape == (
            expected_count,
            100,
        )
        assert split.labels.shape == (expected_count,)
        assert split.windows.dtype.name == "float32"

        counts = Counter(split.labels.tolist())
        expected_per_class = EXPECTED_PER_CLASS[split_name]

        assert counts == {
            CLASS_TO_ID[gesture]: expected_per_class
            for gesture in GESTURES
        }