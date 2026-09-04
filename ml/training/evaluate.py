from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
    ID_TO_CLASS,
)
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import MODEL_VERSION
from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)


DATASET_VERSION = "dataset-v1"

MODEL_DIR = Path(
    "data/processed/"
    "dataset-v1/"
    "features-v1/"
    "models/"
    f"{MODEL_VERSION}"
)

MODEL_FILENAME = f"{MODEL_VERSION}.keras"
TRAINING_METADATA_FILENAME = "metadata.json"

TEST_EVALUATION_FILENAME = "test_evaluation.json"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
TEST_CONFUSION_FILENAME = "test_confusion_matrix.png"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_training_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    expected = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
    }

    for field, expected_value in expected.items():
        actual = metadata.get(field)

        if actual != expected_value:
            raise ValueError(
                f"Training metadata mismatch: "
                f"{field}={actual!r}, "
                f"expected {expected_value!r}."
            )

    if metadata.get("test_split_used") is not False:
        raise ValueError(
            "Training metadata does not confirm "
            "that the test split was held out."
        )

    return metadata


def save_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(matrix)

    axis.set_xticks(
        np.arange(len(GESTURES)),
        labels=GESTURES,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(len(GESTURES)),
        labels=GESTURES,
    )

    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(
        "Final held-out test confusion matrix"
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=150,
    )

    plt.close(figure)


def save_predictions(
    *,
    path: Path,
    csv_paths: tuple[Path, ...],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            (
                "sample_index",
                "source_csv",
                "true_id",
                "true_class",
                "predicted_id",
                "predicted_class",
                "confidence",
                "correct",
            )
        )

        for index, (
            source_csv,
            true_id,
            predicted_id,
            probability_vector,
        ) in enumerate(
            zip(
                csv_paths,
                labels,
                predictions,
                probabilities,
                strict=True,
            )
        ):
            confidence = float(
                probability_vector[predicted_id]
            )

            writer.writerow(
                (
                    index,
                    source_csv.as_posix(),
                    int(true_id),
                    ID_TO_CLASS[int(true_id)],
                    int(predicted_id),
                    ID_TO_CLASS[int(predicted_id)],
                    confidence,
                    bool(true_id == predicted_id),
                )
            )


def run_final_test() -> None:
    root = project_root()
    model_dir = root / MODEL_DIR

    model_path = (
        model_dir / MODEL_FILENAME
    )

    training_metadata_path = (
        model_dir
        / TRAINING_METADATA_FILENAME
    )

    evaluation_path = (
        model_dir
        / TEST_EVALUATION_FILENAME
    )

    predictions_path = (
        model_dir
        / TEST_PREDICTIONS_FILENAME
    )

    confusion_path = (
        model_dir
        / TEST_CONFUSION_FILENAME
    )

    if evaluation_path.exists():
        raise RuntimeError(
            "Final test evaluation already exists. "
            "Refusing to evaluate the held-out test set again."
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen model artifact not found: {model_path}"
        )

    training_metadata = load_training_metadata(
        training_metadata_path
    )

    model_hash = sha256_file(
        model_path
    )

    model = tf.keras.models.load_model(
        model_path
    )

    # This is the first intentional ML evaluation
    # using session_03.
    test = load_feature_split("test")

    if test.session != "session_03":
        raise ValueError(
            f"Expected session_03 test split, "
            f"got {test.session!r}."
        )

    probabilities = model.predict(
        test.features,
        verbose=0,
    )

    if probabilities.shape != (
        test.features.shape[0],
        len(GESTURES),
    ):
        raise ValueError(
            f"Unexpected prediction shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    matrix = confusion_matrix(
        test.labels,
        predictions,
        class_count=len(GESTURES),
    )

    metrics = classification_metrics(
        matrix
    )

    per_sample_loss = (
        tf.keras.losses
        .sparse_categorical_crossentropy(
            test.labels,
            probabilities,
        )
        .numpy()
    )

    test_loss = float(
        np.mean(per_sample_loss)
    )

    save_predictions(
        path=predictions_path,
        csv_paths=test.csv_paths,
        labels=test.labels,
        predictions=predictions,
        probabilities=probabilities,
    )

    save_confusion_matrix(
        matrix,
        confusion_path,
    )

    evaluation = {
        "evaluation_type": "final-held-out-test",
        "model_version": MODEL_VERSION,
        "model_sha256": model_hash,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "test_split": "session_03",
        "test_split_used": True,
        "sample_count": int(
            test.features.shape[0]
        ),
        "class_to_id": CLASS_TO_ID,
        "training_validation_metrics":
            training_metadata.get(
                "validation_metrics"
            ),
        "test_metrics": {
            "loss": test_loss,
            **metrics,
        },
        "test_confusion_matrix":
            matrix.tolist(),
        "post_test_tuning_allowed": False,
    }

    with evaluation_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
        )

    print()
    print("FINAL HELD-OUT TEST")
    print("-------------------")
    print(
        f"Model version:   {MODEL_VERSION}"
    )
    print(
        f"Model SHA-256:  {model_hash}"
    )
    print(
        f"Dataset:         {DATASET_VERSION}"
    )
    print(
        f"Feature version: {FEATURE_VERSION}"
    )
    print(
        f"Test session:    {test.session}"
    )
    print(
        f"Test shape:      {test.features.shape}"
    )
    print()

    print(
        f"{'loss':<16} {test_loss:.6f}"
    )

    for name, value in metrics.items():
        print(
            f"{name:<16} {value:.6f}"
        )

    print()
    print("Test confusion matrix:")
    print(matrix)

    print()
    print(
        f"Evaluation:  {evaluation_path}"
    )
    print(
        f"Predictions: {predictions_path}"
    )
    print(
        f"Confusion:   {confusion_path}"
    )

    print()
    print(
        "Final held-out test completed. "
        "Do not tune the model using these results."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-time final held-out "
            "dataset-v1 test evaluation."
        )
    )

    parser.add_argument(
        "--final-test",
        action="store_true",
        help=(
            "Explicitly confirm that session_03 "
            "may now be used for final evaluation."
        ),
    )

    args = parser.parse_args()

    if not args.final_test:
        raise SystemExit(
            "Refusing to load the held-out test split. "
            "Run with --final-test only after the model "
            "is frozen."
        )

    run_final_test()


if __name__ == "__main__":
    main()