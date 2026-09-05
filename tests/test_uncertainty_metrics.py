from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.models.base_model import CLASS_COUNT
from ml.uncertainty.compute_uncertainty_metrics import (
    load_frozen_passes,
    summary_stats,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
)
from ml.uncertainty.metrics import (
    compute_mc_dropout_uncertainty_metrics,
)


def _one_hot_pass_tensor(
    *,
    sample_count: int = 2,
    class_index: int = 0,
) -> np.ndarray:
    probabilities = np.zeros(
        (
            MC_DROPOUT_PASSES,
            sample_count,
            CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    probabilities[
        :,
        :,
        class_index,
    ] = 1.0

    return probabilities


def test_consensus_one_hot_has_zero_entropy_and_variance() -> None:
    probabilities = _one_hot_pass_tensor()

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    np.testing.assert_allclose(
        metrics.predictive_entropy,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        metrics.normalized_predictive_entropy,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        metrics.class_probability_variance,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        metrics.max_mean_confidence,
        1.0,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_array_equal(
        metrics.predicted_class,
        np.zeros(
            2,
            dtype=np.int64,
        ),
    )


def test_uniform_mean_distribution_has_normalized_entropy_one() -> None:
    probabilities = np.full(
        (
            MC_DROPOUT_PASSES,
            1,
            CLASS_COUNT,
        ),
        1.0 / CLASS_COUNT,
        dtype=np.float32,
    )

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    np.testing.assert_allclose(
        metrics.predictive_entropy,
        np.log(CLASS_COUNT),
        rtol=0.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        metrics.normalized_predictive_entropy,
        1.0,
        rtol=0.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        metrics.max_mean_confidence,
        1.0 / CLASS_COUNT,
        rtol=0.0,
        atol=1e-6,
    )


def test_probability_variance_matches_population_variance() -> None:
    probabilities = np.zeros(
        (
            MC_DROPOUT_PASSES,
            1,
            CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    class_zero = np.asarray(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        dtype=np.float32,
    )

    probabilities[:, 0, 0] = class_zero
    probabilities[:, 0, 1] = (
        1.0 - class_zero
    )

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    expected_variance = np.var(
        class_zero.astype(
            np.float64
        ),
        ddof=0,
    )

    assert np.isclose(
        metrics.class_probability_variance[
            0,
            0,
        ],
        expected_variance,
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        metrics.class_probability_variance[
            0,
            1,
        ],
        expected_variance,
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        metrics.mean_class_variance[0],
        (
            2.0
            * expected_variance
            / CLASS_COUNT
        ),
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        metrics.max_class_variance[0],
        expected_variance,
        rtol=0.0,
        atol=1e-7,
    )


def test_uncertainty_metrics_reject_wrong_pass_count() -> None:
    probabilities = np.full(
        (
            MC_DROPOUT_PASSES - 1,
            2,
            CLASS_COUNT,
        ),
        1.0 / CLASS_COUNT,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Expected probability tensor",
    ):
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )


def test_load_frozen_passes_round_trip(
    tmp_path: Path,
) -> None:
    probabilities = _one_hot_pass_tensor(
        sample_count=3
    )

    labels = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    path = (
        tmp_path
        / "passes.npz"
    )

    np.savez_compressed(
        path,
        probabilities=probabilities,
        labels=labels,
    )

    loaded_probabilities, loaded_labels = (
        load_frozen_passes(
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


def test_summary_stats_are_deterministic() -> None:
    values = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    stats = summary_stats(
        values
    )

    assert stats["min"] == 0.0
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0
    assert stats["max"] == 4.0

    assert np.isclose(
        stats["p95"],
        3.8,
        rtol=0.0,
        atol=1e-12,
    )
