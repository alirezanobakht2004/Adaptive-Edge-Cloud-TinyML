from __future__ import annotations

import numpy as np
import pytest

from ml.export.uncertainty_deployment import (
    apply_inverted_dropout_mask,
    dense_softmax,
    explicit_dropout_edge_head,
)


def test_inverted_dropout_uses_keep_probability_scaling() -> None:
    embedding = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    mask = np.asarray(
        [1, 0, 1, 0],
        dtype=np.uint8,
    )

    output = apply_inverted_dropout_mask(
        embedding,
        mask,
        dropout_rate=0.2,
    )

    np.testing.assert_allclose(
        output,
        np.asarray(
            [1.25, 0.0, 3.75, 0.0],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_dropout_mask_requires_same_shape() -> None:
    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        apply_inverted_dropout_mask(
            np.ones(
                4,
                dtype=np.float32,
            ),
            np.ones(
                3,
                dtype=np.uint8,
            ),
            dropout_rate=0.2,
        )


def test_dense_softmax_is_normalized_and_stable() -> None:
    embedding = np.asarray(
        [1000.0, -1000.0],
        dtype=np.float32,
    )

    kernel = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )

    bias = np.asarray(
        [0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    probabilities = dense_softmax(
        embedding,
        kernel,
        bias,
    )

    assert np.isfinite(
        probabilities
    ).all()

    np.testing.assert_allclose(
        np.sum(
            probabilities
        ),
        1.0,
        rtol=0.0,
        atol=1e-6,
    )


def test_dense_softmax_supports_batch_input() -> None:
    embedding = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    kernel = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    bias = np.zeros(
        2,
        dtype=np.float32,
    )

    probabilities = dense_softmax(
        embedding,
        kernel,
        bias,
    )

    assert probabilities.shape == (
        2,
        2,
    )

    np.testing.assert_allclose(
        np.sum(
            probabilities,
            axis=1,
        ),
        np.ones(
            2,
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )


def test_explicit_dropout_edge_head_matches_composed_reference() -> None:
    embedding = np.asarray(
        [1.0, 2.0],
        dtype=np.float32,
    )

    mask = np.asarray(
        [1, 0],
        dtype=np.uint8,
    )

    kernel = np.asarray(
        [
            [1.0, -1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    bias = np.asarray(
        [0.1, -0.1],
        dtype=np.float32,
    )

    direct = explicit_dropout_edge_head(
        embedding,
        mask,
        kernel,
        bias,
        dropout_rate=0.2,
    )

    masked = apply_inverted_dropout_mask(
        embedding,
        mask,
        dropout_rate=0.2,
    )

    composed = dense_softmax(
        masked,
        kernel,
        bias,
    )

    np.testing.assert_allclose(
        direct,
        composed,
        rtol=0.0,
        atol=1e-7,
    )
