from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
    load_split,
)
from ml.features.extractor import (
    extract_feature_matrix,
    extract_features,
    load_feature_split,
)


EXPECTED_COUNTS = {
    "train": 600,
    "validation": 200,
}

EXPECTED_PER_CLASS = {
    "train": 120,
    "validation": 40,
}


def test_feature_matrix_matches_direct_extraction() -> None:
    raw_split = load_split("validation")

    matrix = extract_feature_matrix(
        raw_split.windows[:3]
    )

    assert matrix.shape == (3, 10)
    assert matrix.dtype == np.float32

    np.testing.assert_allclose(
        matrix[0],
        extract_features(raw_split.windows[0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_and_validation_feature_splits() -> None:
    for split_name, expected_count in EXPECTED_COUNTS.items():
        split = load_feature_split(split_name)

        assert split.features.shape == (
            expected_count,
            10,
        )
        assert split.labels.shape == (
            expected_count,
        )
        assert split.features.dtype == np.float32
        assert np.isfinite(split.features).all()

        counts = Counter(split.labels.tolist())

        assert counts == {
            CLASS_TO_ID[gesture]:
            EXPECTED_PER_CLASS[split_name]
            for gesture in GESTURES
        }


@pytest.mark.parametrize(
    "shape",
    [
        (100, 6),
        (1, 99, 6),
        (1, 100, 5),
        (1, 100, 7),
        (2, 600),
    ],
)
def test_invalid_feature_batch_shape_is_rejected(
    shape: tuple[int, ...],
) -> None:
    windows = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        extract_feature_matrix(windows)


def test_empty_feature_batch_is_rejected() -> None:
    windows = np.empty(
        (0, 100, 6),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        extract_feature_matrix(windows)


def test_unknown_feature_version_is_rejected() -> None:
    window = np.zeros(
        (100, 6),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        extract_features(
            window,
            version="features-unknown",
        )