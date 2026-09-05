from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from ml.models.base_model import (
    CLASS_COUNT,
    INPUT_FEATURES,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
)
from ml.uncertainty.runtime import (
    UNCERTAINTY_SCORE_NAME,
    infer_with_uncertainty,
)


class FakeFivePassModel:
    def __init__(
        self,
        outputs: list[np.ndarray],
    ) -> None:
        self.outputs = outputs
        self.calls = 0
        self.training_flags: list[bool] = []

    def __call__(
        self,
        features: np.ndarray,
        *,
        training: bool,
    ) -> tf.Tensor:
        assert features.shape == (
            1,
            INPUT_FEATURES,
        )

        self.training_flags.append(
            training
        )

        output = self.outputs[
            self.calls
        ]

        self.calls += 1

        return tf.convert_to_tensor(
            output[None, :],
            dtype=tf.float32,
        )


def _feature_vector() -> np.ndarray:
    return np.zeros(
        INPUT_FEATURES,
        dtype=np.float32,
    )


def test_runtime_contract_uses_normalized_predictive_entropy() -> None:
    assert (
        UNCERTAINTY_SCORE_NAME
        == "normalized_predictive_entropy"
    )


def test_single_inference_runs_exactly_five_stochastic_passes() -> None:
    outputs = [
        np.asarray(
            [0.80, 0.10, 0.05, 0.03, 0.02],
            dtype=np.float32,
        )
        for _ in range(
            MC_DROPOUT_PASSES
        )
    ]

    model = FakeFivePassModel(
        outputs
    )

    result = infer_with_uncertainty(
        model,
        _feature_vector(),
    )

    assert model.calls == (
        MC_DROPOUT_PASSES
    )

    assert model.training_flags == (
        [True] * MC_DROPOUT_PASSES
    )

    assert result.predicted_class == 0

    assert np.isclose(
        result.confidence,
        0.80,
        rtol=0.0,
        atol=1e-6,
    )


def test_identical_one_hot_passes_have_zero_uncertainty() -> None:
    output = np.zeros(
        CLASS_COUNT,
        dtype=np.float32,
    )

    output[2] = 1.0

    model = FakeFivePassModel(
        [
            output.copy()
            for _ in range(
                MC_DROPOUT_PASSES
            )
        ]
    )

    result = infer_with_uncertainty(
        model,
        _feature_vector(),
    )

    assert result.predicted_class == 2

    assert np.isclose(
        result.uncertainty_score,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )

    assert np.isclose(
        result.mean_class_variance,
        0.0,
        rtol=0.0,
        atol=1e-7,
    )


def test_uniform_mean_has_maximum_normalized_entropy() -> None:
    output = np.full(
        CLASS_COUNT,
        1.0 / CLASS_COUNT,
        dtype=np.float32,
    )

    model = FakeFivePassModel(
        [
            output.copy()
            for _ in range(
                MC_DROPOUT_PASSES
            )
        ]
    )

    result = infer_with_uncertainty(
        model,
        _feature_vector(),
    )

    assert np.isclose(
        result.uncertainty_score,
        1.0,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.isclose(
        result.confidence,
        1.0 / CLASS_COUNT,
        rtol=0.0,
        atol=1e-6,
    )


def test_runtime_rejects_batch_input() -> None:
    output = np.asarray(
        [0.8, 0.1, 0.05, 0.03, 0.02],
        dtype=np.float32,
    )

    model = FakeFivePassModel(
        [
            output.copy()
            for _ in range(
                MC_DROPOUT_PASSES
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="one features-v1 vector",
    ):
        infer_with_uncertainty(
            model,
            np.zeros(
                (
                    2,
                    INPUT_FEATURES,
                ),
                dtype=np.float32,
            ),
        )

    assert model.calls == 0
