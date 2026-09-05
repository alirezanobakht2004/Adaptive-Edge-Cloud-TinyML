from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.dataset.loader import GESTURES
from ml.features.extractor import FeatureSplit
from ml.uncertainty import evaluate_ambiguous_probes
from ml.uncertainty.ambiguous_probe import (
    MIDPOINT_ALPHA,
    build_ambiguous_midpoint_probes,
    class_pairs,
    normalized_feature_matrix,
)
from ml.uncertainty.evaluate_ambiguous_probes import (
    endpoint_mean,
    fraction_true,
)


def _validation_fixture() -> FeatureSplit:
    feature_rows = []
    labels = []

    for class_id in range(
        len(GESTURES)
    ):
        for local_index in range(3):
            row = np.full(
                10,
                float(class_id),
                dtype=np.float32,
            )

            row[0] += (
                local_index * 0.01
            )

            feature_rows.append(
                row
            )

            labels.append(
                class_id
            )

    return FeatureSplit(
        name="validation",
        session="session_02",
        feature_version="features-v1",
        features=np.stack(
            feature_rows
        ),
        labels=np.asarray(
            labels,
            dtype=np.int64,
        ),
        csv_paths=tuple(
            Path(
                f"validation_{index}.csv"
            )
            for index
            in range(len(labels))
        ),
        metadata_paths=tuple(
            Path(
                f"validation_{index}.json"
            )
            for index
            in range(len(labels))
        ),
    )


def test_class_pairs_cover_all_unordered_pairs() -> None:
    pairs = class_pairs()

    expected_count = (
        len(GESTURES)
        * (len(GESTURES) - 1)
        // 2
    )

    assert len(pairs) == expected_count

    assert len(
        set(pairs)
    ) == expected_count

    for left, right in pairs:
        assert left < right


def test_normalized_feature_matrix_uses_supplied_statistics() -> None:
    features = np.asarray(
        [
            [1.0, 4.0],
            [3.0, 8.0],
        ],
        dtype=np.float32,
    )

    normalized = (
        normalized_feature_matrix(
            features,
            normalization_mean=np.asarray(
                [1.0, 2.0],
                dtype=np.float32,
            ),
            normalization_variance=np.asarray(
                [4.0, 4.0],
                dtype=np.float32,
            ),
        )
    )

    np.testing.assert_allclose(
        normalized,
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 3.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_ambiguous_probe_builder_creates_midpoints_for_every_pair() -> None:
    validation = (
        _validation_fixture()
    )

    probes = (
        build_ambiguous_midpoint_probes(
            validation.features,
            validation.labels,
            normalization_mean=np.zeros(
                10,
                dtype=np.float32,
            ),
            normalization_variance=np.ones(
                10,
                dtype=np.float32,
            ),
            probes_per_class_pair=2,
            alpha=MIDPOINT_ALPHA,
        )
    )

    expected_count = (
        len(class_pairs())
        * 2
    )

    assert probes.features.shape == (
        expected_count,
        10,
    )

    assert probes.left_class_id.shape == (
        expected_count,
    )

    assert np.all(
        probes.left_class_id
        < probes.right_class_id
    )

    for index in range(
        expected_count
    ):
        left = validation.features[
            probes.left_sample_index[index]
        ]

        right = validation.features[
            probes.right_sample_index[index]
        ]

        expected = (
            0.5 * left
            + 0.5 * right
        )

        np.testing.assert_allclose(
            probes.features[index],
            expected,
            rtol=0.0,
            atol=1e-7,
        )


def test_ambiguous_probe_builder_rejects_endpoint_alpha() -> None:
    validation = (
        _validation_fixture()
    )

    with pytest.raises(
        ValueError,
        match="strictly between",
    ):
        build_ambiguous_midpoint_probes(
            validation.features,
            validation.labels,
            normalization_mean=np.zeros(
                10,
                dtype=np.float32,
            ),
            normalization_variance=np.ones(
                10,
                dtype=np.float32,
            ),
            probes_per_class_pair=1,
            alpha=0.0,
        )


def test_ambiguity_evaluator_loads_validation_only(
    monkeypatch,
) -> None:
    validation = (
        _validation_fixture()
    )

    requested: list[str] = []

    def fake_loader(
        split_name: str,
    ) -> FeatureSplit:
        requested.append(
            split_name
        )

        if split_name == "test":
            raise AssertionError(
                "TEST split must remain locked."
            )

        if split_name != "validation":
            raise AssertionError(
                "Ambiguity probe may load only VALIDATION."
            )

        return validation

    monkeypatch.setattr(
        evaluate_ambiguous_probes,
        "load_feature_split",
        fake_loader,
    )

    loaded = (
        evaluate_ambiguous_probes
        .load_validation_only()
    )

    assert loaded.session == "session_02"

    assert requested == [
        "validation",
    ]


def test_endpoint_mean_and_fraction_true() -> None:
    clear_values = np.asarray(
        [0.1, 0.2, 0.8, 0.6],
        dtype=np.float32,
    )

    validation = (
        _validation_fixture()
    )

    probes = (
        build_ambiguous_midpoint_probes(
            validation.features,
            validation.labels,
            normalization_mean=np.zeros(
                10,
                dtype=np.float32,
            ),
            normalization_variance=np.ones(
                10,
                dtype=np.float32,
            ),
            probes_per_class_pair=1,
            alpha=0.5,
        )
    )

    # Replace indices for a small direct endpoint-mean check.
    object.__setattr__(
        probes,
        "left_sample_index",
        np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
    )

    object.__setattr__(
        probes,
        "right_sample_index",
        np.asarray(
            [2, 3],
            dtype=np.int64,
        ),
    )

    means = endpoint_mean(
        clear_values,
        probes,
    )

    np.testing.assert_allclose(
        means,
        np.asarray(
            [0.45, 0.4],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )

    assert fraction_true(
        np.asarray(
            [True, False, True, True]
        )
    ) == 0.75
