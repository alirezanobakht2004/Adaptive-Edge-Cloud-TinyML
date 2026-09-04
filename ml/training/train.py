from __future__ import annotations

import csv
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
)
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import (
    MODEL_VERSION,
    build_base_model,
)
from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)


DATASET_VERSION = "dataset-v1"

SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20

OUTPUT_DIR = Path(
    "data/processed/"
    "dataset-v1/"
    "features-v1/"
    "models/"
    f"{MODEL_VERSION}"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_reproducibility() -> None:
    tf.keras.utils.set_random_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def create_normalization(
    train_features: np.ndarray,
) -> tf.keras.layers.Normalization:
    normalization = tf.keras.layers.Normalization(
        axis=-1,
        name="feature_normalization",
    )

    normalization.adapt(train_features)

    return normalization


def save_history(
    history: tf.keras.callbacks.History,
    path: Path,
) -> None:
    keys = tuple(history.history.keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ("epoch", *keys)
        )

        epochs = len(
            history.history[keys[0]]
        )

        for epoch in range(epochs):
            writer.writerow(
                (
                    epoch + 1,
                    *[
                        history.history[key][epoch]
                        for key in keys
                    ],
                )
            )


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
        "Validation confusion matrix"
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


def main() -> None:
    set_reproducibility()

    output_dir = (
        project_root() / OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Deliberately load only train and validation.
    train = load_feature_split("train")
    validation = load_feature_split(
        "validation"
    )

    normalization = create_normalization(
        train.features
    )

    model = build_base_model(
        normalization
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train.features,
        train.labels,
        validation_data=(
            validation.features,
            validation.labels,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model.predict(
        validation.features,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    matrix = confusion_matrix(
        validation.labels,
        predictions,
        class_count=len(GESTURES),
    )

    metrics = classification_metrics(
        matrix
    )

    model_path = (
        output_dir
        / f"{MODEL_VERSION}.keras"
    )

    model.save(model_path)

    history_path = (
        output_dir
        / "training_history.csv"
    )

    save_history(
        history,
        history_path,
    )

    confusion_path = (
        output_dir
        / "validation_confusion_matrix.png"
    )

    save_confusion_matrix(
        matrix,
        confusion_path,
    )

    normalization_mean = (
        normalization.mean.numpy()
        .reshape(-1)
        .astype(float)
        .tolist()
    )

    normalization_variance = (
        normalization.variance.numpy()
        .reshape(-1)
        .astype(float)
        .tolist()
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "train_split": "session_01",
        "validation_split": "session_02",
        "test_split_used": False,
        "class_to_id": CLASS_TO_ID,
        "seed": SEED,
        "training": {
            "max_epochs": MAX_EPOCHS,
            "epochs_ran": len(
                history.history["loss"]
            ),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,
        },
        "normalization": {
            "fit_split": "train",
            "mean": normalization_mean,
            "variance": normalization_variance,
        },
        "validation_metrics": metrics,
        "validation_confusion_matrix":
            matrix.tolist(),
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        f"Model version: {MODEL_VERSION}"
    )
    print(
        f"Train:      {train.features.shape}"
    )
    print(
        "Validation: "
        f"{validation.features.shape}"
    )
    print("Test split was not loaded.")
    print()

    for name, value in metrics.items():
        print(
            f"{name:<16} {value:.6f}"
        )

    print()
    print("Validation confusion matrix:")
    print(matrix)

    print()
    print(f"Model:    {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"History:  {history_path}")
    print(
        f"Confusion: {confusion_path}"
    )


if __name__ == "__main__":
    main()