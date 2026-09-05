from __future__ import annotations

import numpy as np


def apply_inverted_dropout_mask(
    embedding: np.ndarray,
    keep_mask: np.ndarray,
    *,
    dropout_rate: float,
) -> np.ndarray:
    """Apply training-time inverted Dropout to an embedding.

    Keras Dropout semantics during training are:

        y = x * keep_mask / (1 - dropout_rate)

    where keep_mask contains 0/1 values.

    This helper deliberately receives the mask explicitly. Random-mask
    generation belongs to the runtime/platform layer so parity tests can
    inject frozen masks.
    """

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    )

    mask = np.asarray(
        keep_mask,
    )

    if vector.shape != mask.shape:
        raise ValueError(
            "Embedding and keep-mask shapes must match."
        )

    if vector.ndim not in (1, 2):
        raise ValueError(
            "Embedding must be a 1-D vector or 2-D batch."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    if not (
        0.0 <= dropout_rate < 1.0
    ):
        raise ValueError(
            "dropout_rate must be in [0, 1)."
        )

    if not np.all(
        (mask == 0)
        | (mask == 1)
        | (mask == False)
        | (mask == True)
    ):
        raise ValueError(
            "keep_mask must contain only 0/1 values."
        )

    scale = np.float32(
        1.0 / (1.0 - dropout_rate)
    )

    return (
        vector
        * mask.astype(
            np.float32
        )
        * scale
    ).astype(
        np.float32
    )


def dense_softmax(
    embedding: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Float32 Dense + stable Softmax reference implementation."""

    x = np.asarray(
        embedding,
        dtype=np.float32,
    )

    weights = np.asarray(
        kernel,
        dtype=np.float32,
    )

    offsets = np.asarray(
        bias,
        dtype=np.float32,
    )

    if x.ndim not in (1, 2):
        raise ValueError(
            "Embedding must be 1-D or 2-D."
        )

    input_units = x.shape[-1]

    if (
        weights.ndim != 2
        or weights.shape[0]
        != input_units
    ):
        raise ValueError(
            "Dense kernel shape does not match embedding."
        )

    if offsets.shape != (
        weights.shape[1],
    ):
        raise ValueError(
            "Dense bias shape does not match kernel output."
        )

    if (
        not np.isfinite(x).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(offsets).all()
    ):
        raise ValueError(
            "Dense-softmax input contains non-finite values."
        )

    logits = (
        x @ weights
        + offsets
    ).astype(
        np.float32
    )

    max_logits = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp_logits = np.exp(
        (
            logits
            - max_logits
        ).astype(
            np.float32
        )
    ).astype(
        np.float32
    )

    denominator = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    probabilities = (
        exp_logits
        / denominator
    ).astype(
        np.float32
    )

    return probabilities


def explicit_dropout_edge_head(
    embedding: np.ndarray,
    keep_mask: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    dropout_rate: float,
) -> np.ndarray:
    masked = apply_inverted_dropout_mask(
        embedding,
        keep_mask,
        dropout_rate=dropout_rate,
    )

    return dense_softmax(
        masked,
        kernel,
        bias,
    )


__all__ = [
    "apply_inverted_dropout_mask",
    "dense_softmax",
    "explicit_dropout_edge_head",
]
