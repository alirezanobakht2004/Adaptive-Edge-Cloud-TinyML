from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.uncertainty.calibration import (
    compute_reliability_bins,
    evaluate_calibration,
)
from ml.uncertainty.evaluate_calibration import (
    load_frozen_mean_probabilities,
)


def test_perfect_one_hot_predictions_are_perfectly_calibrated() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    metrics = evaluate_calibration(
        probabilities,
        labels,
        bin_count=10,
    )

    assert metrics.accuracy == 1.0
    assert metrics.mean_confidence == 1.0
    assert metrics.expected_calibration_error == 0.0
    assert metrics.maximum_calibration_error == 0.0
    assert metrics.negative_log_likelihood == 0.0
    assert metrics.multiclass_brier_score == 0.0


def test_ece_matches_manual_two_bin_example() -> None:
    probabilities = np.asarray(
        [
            [0.60, 0.40],
            [0.70, 0.30],
            [0.80, 0.20],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [0, 1, 0, 0],
        dtype=np.int64,
    )

    metrics = evaluate_calibration(
        probabilities,
        labels,
        bin_count=2,
    )

    # All samples fall into the [0.5, 1.0] bin.
    # Mean confidence = 0.75
    # Accuracy = 0.75
    assert np.isclose(
        metrics.expected_calibration_error,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        metrics.mean_confidence,
        0.75,
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        metrics.accuracy,
        0.75,
        rtol=0.0,
        atol=1e-7,
    )


def test_overconfident_wrong_predictions_have_positive_gap() -> None:
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [1, 1],
        dtype=np.int64,
    )

    metrics = evaluate_calibration(
        probabilities,
        labels,
        bin_count=10,
    )

    assert metrics.accuracy == 0.0

    assert metrics.mean_confidence > 0.9

    assert (
        metrics.signed_confidence_gap
        > 0.9
    )

    assert (
        metrics.expected_calibration_error
        > 0.9
    )


def test_reliability_bins_cover_confidence_one() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [0],
        dtype=np.int64,
    )

    bins = compute_reliability_bins(
        probabilities,
        labels,
        bin_count=10,
    )

    assert sum(
        reliability_bin.count
        for reliability_bin in bins
    ) == 1

    assert bins[-1].count == 1


def test_calibration_rejects_invalid_probability_rows() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.8],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [0],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        evaluate_calibration(
            probabilities,
            labels,
        )


def test_load_frozen_mean_probabilities_round_trip(
    tmp_path: Path,
) -> None:
    # The production loader intentionally enforces the project's
    # fixed five-class gesture contract, so this fixture must also
    # use five class probabilities per sample.
    probabilities = np.asarray(
        [
            [0.80, 0.10, 0.05, 0.03, 0.02],
            [0.05, 0.85, 0.04, 0.03, 0.03],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [0, 1],
        dtype=np.int64,
    )

    path = (
        tmp_path
        / "metrics.npz"
    )

    np.savez_compressed(
        path,
        mean_probabilities=probabilities,
        labels=labels,
    )

    loaded_probabilities, loaded_labels = (
        load_frozen_mean_probabilities(
            path
        )
    )

    np.testing.assert_array_equal(
        loaded_probabilities,
        probabilities,
    )

    np.testing.assert_array_equal(
        loaded_labels,
        labels,
    )
