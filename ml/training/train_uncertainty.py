from __future__ import annotations

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
    DATASET_VERSION,
    GESTURES,
)
from ml.features.extractor import (
    FeatureSplit,
    load_feature_split,
)
from ml.features.features_v1 import FEATURE_VERSION
from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
    build_mc_dropout_model,
)


SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20

OUTPUT_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}"
)

MODEL_FILENAME = (
    f"{UNCERTAINTY_MODEL_VERSION}.keras"
)

METADATA_FILENAME = "metadata.json"
HISTORY_FILENAME = "training_history.csv"
CONFUSION_FILENAME = (
    "validation_confusion_matrix.png"
)


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


def load_training_splits() -> tuple[
    FeatureSplit,
    FeatureSplit,
]:
    """Load only TRAIN and VALIDATION.

    The held-out TEST split is deliberately not available to
    Phase-5 candidate training or model-selection code.
    """

    train = load_feature_split("train")
    validation = load_feature_split(
        "validation"
    )

    if train.session != "session_01":
        raise ValueError(
            "Expected TRAIN=session_01, "
            f"got {train.session!r}."
        )

    if validation.session != "session_02":
        raise ValueError(
            "Expected VALIDATION=session_02, "
            f"got {validation.session!r}."
        )

    return train, validation


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
        "MC-Dropout candidate "
        "deterministic validation"
    )

    for row in range(matrix.shape[0]):
        for column in range(
            matrix.shape[1]
        ):
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


def build_metadata(
    *,
    train: FeatureSplit,
    validation: FeatureSplit,
    history: tf.keras.callbacks.History,
    normalization: tf.keras.layers.Normalization,
    metrics: dict[str, float],
    matrix: np.ndarray,
    model_sha256: str,
) -> dict:
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

    return {
        "model_version":
            UNCERTAINTY_MODEL_VERSION,
        "model_purpose":
            "phase5-mc-dropout-candidate",
        "model_sha256": model_sha256,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "train_split": train.session,
        "validation_split":
            validation.session,
        "test_split_used": False,
        "class_to_id": CLASS_TO_ID,
        "seed": SEED,
        "mc_dropout": {
            "passes": MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
            "training_dropout_active": True,
            "deterministic_validation_dropout_active":
                False,
        },
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
            "variance":
                normalization_variance,
        },
        "deterministic_validation_metrics":
            metrics,
        "deterministic_validation_confusion_matrix":
            matrix.tolist(),
        "mc_uncertainty_evaluation_completed":
            False,
        "esp32_uncertainty_deployment_completed":
            False,
    }


def ensure_output_is_new(
    output_dir: Path,
) -> None:
    protected = (
        output_dir / MODEL_FILENAME,
        output_dir / METADATA_FILENAME,
    )

    existing = [
        path
        for path in protected
        if path.exists()
    ]

    if existing:
        joined = ", ".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Refusing to overwrite an existing "
            "Phase-5 uncertainty candidate: "
            f"{joined}"
        )


def main() -> None:
    set_reproducibility()

    root = project_root()
    output_dir = root / OUTPUT_DIR

    ensure_output_is_new(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, validation = (
        load_training_splits()
    )

    normalization = create_normalization(
        train.features
    )

    model = build_mc_dropout_model(
        normalization
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),
        metrics=[
            tf.keras.metrics
            .SparseCategoricalAccuracy(
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

    # IMPORTANT:
    # This is deterministic validation only.
    # Dropout is disabled here. The five stochastic
    # MC passes belong to the next M6 sub-step.
    probabilities = model(
        validation.features,
        training=False,
    ).numpy()

    expected_shape = (
        validation.features.shape[0],
        len(GESTURES),
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Unexpected deterministic "
            "validation output shape: "
            f"{probabilities.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Candidate produced non-finite "
            "validation probabilities."
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

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
        / MODEL_FILENAME
    )

    model.save(model_path)

    model_sha256 = sha256_file(
        model_path
    )

    history_path = (
        output_dir
        / HISTORY_FILENAME
    )

    save_history(
        history,
        history_path,
    )

    confusion_path = (
        output_dir
        / CONFUSION_FILENAME
    )

    save_confusion_matrix(
        matrix,
        confusion_path,
    )

    metadata = build_metadata(
        train=train,
        validation=validation,
        history=history,
        normalization=normalization,
        metrics=metrics,
        matrix=matrix,
        model_sha256=model_sha256,
    )

    metadata_path = (
        output_dir
        / METADATA_FILENAME
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
        "PHASE 5 / M6 — DROPOUT CANDIDATE TRAINED"
    )
    print(
        "========================================"
    )

    print(
        "Model version:     "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Dataset version:   "
        f"{DATASET_VERSION}"
    )

    print(
        "Feature version:   "
        f"{FEATURE_VERSION}"
    )

    print(
        "Dropout rate:      "
        f"{MC_DROPOUT_RATE}"
    )

    print(
        "Configured passes: "
        f"{MC_DROPOUT_PASSES}"
    )

    print(
        "Train:             "
        f"{train.features.shape} "
        f"({train.session})"
    )

    print(
        "Validation:        "
        f"{validation.features.shape} "
        f"({validation.session})"
    )

    print(
        "Test split loaded: NO"
    )

    print(
        "Validation mode:   "
        "deterministic / dropout disabled"
    )

    print()

    for name, value in metrics.items():
        print(
            f"{name:<16} {value:.6f}"
        )

    print()
    print(
        "Validation confusion matrix:"
    )
    print(matrix)

    print()
    print(
        "Epochs ran:        "
        f"{len(history.history['loss'])}"
    )

    print(
        "Model SHA-256:     "
        f"{model_sha256}"
    )

    print()
    print(f"Model:      {model_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"History:    {history_path}")
    print(f"Confusion:  {confusion_path}")

    print()
    print(
        "MC stochastic evaluation has NOT "
        "been run yet."
    )


if __name__ == "__main__":
    main()
