from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    matrix = np.zeros(
        (class_count, class_count),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        y_true,
        y_pred,
        strict=True,
    ):
        matrix[true_label, predicted_label] += 1

    return matrix


def classification_metrics(
    matrix: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)

    true_positive = np.diag(matrix)

    predicted_count = np.sum(
        matrix,
        axis=0,
    )

    true_count = np.sum(
        matrix,
        axis=1,
    )

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )

    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive),
        where=true_count != 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    total = np.sum(matrix)

    accuracy = (
        float(np.sum(true_positive) / total)
        if total != 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


__all__ = [
    "classification_metrics",
    "confusion_matrix",
]