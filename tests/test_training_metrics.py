import numpy as np

from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)


def test_confusion_matrix() -> None:
    y_true = np.asarray(
        [0, 0, 1, 1, 2],
        dtype=np.int64,
    )

    y_pred = np.asarray(
        [0, 1, 1, 1, 2],
        dtype=np.int64,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        class_count=3,
    )

    expected = np.asarray(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        matrix,
        expected,
    )


def test_perfect_classification_metrics() -> None:
    matrix = np.eye(
        5,
        dtype=np.int64,
    )

    metrics = classification_metrics(
        matrix
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0