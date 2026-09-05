from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from ml.features.extractor import FeatureSplit
from ml.uncertainty import evaluate_mc_dropout
from ml.uncertainty.mc_dropout import (
    CLASS_COUNT,
    INPUT_FEATURES,
    MC_DROPOUT_PASSES,
    mc_dropout_predict,
    mc_dropout_variation_diagnostics,
)


class FakeStochasticModel:
    def __init__(self) -> None:
        self.calls = 0
        self.training_flags: list[bool] = []

    def __call__(
        self,
        features: np.ndarray,
        *,
        training: bool,
    ) -> tf.Tensor:
        self.training_flags.append(
            training
        )

        sample_count = features.shape[0]

        probabilities = np.zeros(
            (
                sample_count,
                CLASS_COUNT,
            ),
            dtype=np.float32,
        )

        delta = (
            0.01
            * self.calls
        )

        probabilities[:, 0] = (
            0.70 - delta
        )

        probabilities[:, 1] = (
            0.30 + delta
        )

        self.calls += 1

        return tf.convert_to_tensor(
            probabilities
        )


def _feature_matrix(
    sample_count: int = 3,
) -> np.ndarray:
    return np.zeros(
        (
            sample_count,
            INPUT_FEATURES,
        ),
        dtype=np.float32,
    )


def _fake_validation_split() -> FeatureSplit:
    return FeatureSplit(
        name="validation",
        session="session_02",
        feature_version="features-v1",
        features=_feature_matrix(
            sample_count=2
        ),
        labels=np.zeros(
            2,
            dtype=np.int64,
        ),
        csv_paths=(
            Path("validation_0.csv"),
            Path("validation_1.csv"),
        ),
        metadata_paths=(
            Path("validation_0.json"),
            Path("validation_1.json"),
        ),
    )


def test_mc_dropout_predict_runs_exactly_five_training_passes() -> None:
    model = FakeStochasticModel()

    probabilities = mc_dropout_predict(
        model,
        _feature_matrix(),
    )

    assert probabilities.shape == (
        MC_DROPOUT_PASSES,
        3,
        CLASS_COUNT,
    )

    assert model.calls == (
        MC_DROPOUT_PASSES
    )

    assert model.training_flags == (
        [True] * MC_DROPOUT_PASSES
    )

    np.testing.assert_allclose(
        probabilities.sum(
            axis=2
        ),
        1.0,
        rtol=0.0,
        atol=1e-6,
    )


def test_mc_dropout_predict_rejects_non_contract_pass_count() -> None:
    model = FakeStochasticModel()

    with pytest.raises(
        ValueError,
        match="exactly 5",
    ):
        mc_dropout_predict(
            model,
            _feature_matrix(),
            passes=4,
        )

    assert model.calls == 0


def test_variation_diagnostics_detect_probability_and_top1_changes() -> None:
    probabilities = np.zeros(
        (
            MC_DROPOUT_PASSES,
            2,
            CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    # Sample 0 stays identical across all passes.
    probabilities[:, 0, 0] = 0.8
    probabilities[:, 0, 1] = 0.2

    # Sample 1 varies, including a top-1 class change.
    probabilities[:, 1, 0] = np.asarray(
        [0.7, 0.6, 0.4, 0.3, 0.55],
        dtype=np.float32,
    )

    probabilities[:, 1, 1] = (
        1.0
        - probabilities[:, 1, 0]
    )

    diagnostics = (
        mc_dropout_variation_diagnostics(
            probabilities,
            tolerance=1e-7,
        )
    )

    assert diagnostics.sample_count == 2

    assert (
        diagnostics.varying_sample_count
        == 1
    )

    assert (
        diagnostics.top1_changed_sample_count
        == 1
    )

    assert (
        diagnostics.max_probability_range
        > 0.0
    )


def test_mc_dropout_evaluator_loads_validation_only(
    monkeypatch,
) -> None:
    requested: list[str] = []

    validation = (
        _fake_validation_split()
    )

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
                "Only VALIDATION may be loaded."
            )

        return validation

    monkeypatch.setattr(
        evaluate_mc_dropout,
        "load_feature_split",
        fake_loader,
    )

    loaded = (
        evaluate_mc_dropout
        .load_validation_split()
    )

    assert loaded.session == "session_02"

    assert requested == [
        "validation",
    ]
