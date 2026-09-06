"""Train the Phase-6 fixed Split-3 cloud tail.

Important:
- source B1/B2/B3 model is frozen gesture-model-v1.1.0
- only TRAIN and VALIDATION are loaded
- held-out TEST data is deliberately not loaded
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

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
from ml.models.cloud_model import (
    CLOUD_TAIL_VERSION,
    INPUT_EMBEDDING_DIM,
    SOURCE_EDGE_MODEL_VERSION,
    build_cloud_tail,
)
from ml.training.metrics import (
    classification_metrics,
    confusion_matrix,
)


SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20

SOURCE_MODEL_SHA256 = (
    "4891b4b6d453d96852ddcaea35d847ff"
    "8eea4ef1349ae999d7594925dabc5d6c"
)

SOURCE_MODEL_RELATIVE_PATH = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{SOURCE_EDGE_MODEL_VERSION}/"
    f"{SOURCE_EDGE_MODEL_VERSION}.keras"
)

OUTPUT_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{CLOUD_TAIL_VERSION}"
)

MODEL_FILENAME = f"{CLOUD_TAIL_VERSION}.keras"
METADATA_FILENAME = "metadata.json"
HISTORY_FILENAME = "training_history.csv"


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


def load_training_splits() -> tuple[
    FeatureSplit,
    FeatureSplit,
]:
    train = load_feature_split("train")
    validation = load_feature_split("validation")

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


def load_source_model(
    model_path: Path,
) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Source edge model not found: {model_path}"
        )

    actual_sha256 = sha256_file(model_path)

    if actual_sha256 != SOURCE_MODEL_SHA256:
        raise RuntimeError(
            "Source edge model SHA-256 mismatch: "
            f"{actual_sha256}"
        )

    model = tf.keras.models.load_model(model_path)

    required_layers = (
        "feature_normalization",
        "block1",
        "block2",
        "block3",
        "mc_dropout",
        "edge_head",
    )

    for layer_name in required_layers:
        model.get_layer(layer_name)

    b3 = model.get_layer("block3")

    if int(b3.output.shape[-1]) != INPUT_EMBEDDING_DIM:
        raise RuntimeError(
            "Unexpected B3 embedding dimension: "
            f"{b3.output.shape[-1]}"
        )

    return model


def build_b3_extractor(
    source_model: tf.keras.Model,
) -> tf.keras.Model:
    extractor = tf.keras.Model(
        inputs=source_model.input,
        outputs=source_model.get_layer(
            "block3"
        ).output,
        name="gesture_model_v1_1_0_b3_extractor",
    )

    extractor.trainable = False

    return extractor


def extract_embeddings(
    extractor: tf.keras.Model,
    features: np.ndarray,
) -> np.ndarray:
    embeddings = extractor(
        features,
        training=False,
    ).numpy().astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        features.shape[0],
        INPUT_EMBEDDING_DIM,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "B3 embeddings contain NaN or infinity."
        )

    return embeddings


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
        writer.writerow(("epoch", *keys))

        epoch_count = len(
            history.history[keys[0]]
        )

        for index in range(epoch_count):
            writer.writerow(
                (
                    index + 1,
                    *[
                        history.history[key][index]
                        for key in keys
                    ],
                )
            )


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
        raise RuntimeError(
            "Refusing to overwrite existing "
            "Phase-6 cloud-tail artifact: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )


def main() -> None:
    set_reproducibility()

    root = project_root()

    source_model_path = (
        root / SOURCE_MODEL_RELATIVE_PATH
    )

    output_dir = root / OUTPUT_DIR

    ensure_output_is_new(output_dir)

    train, validation = load_training_splits()

    source_model = load_source_model(
        source_model_path
    )

    extractor = build_b3_extractor(
        source_model
    )

    train_embeddings = extract_embeddings(
        extractor,
        train.features,
    )

    validation_embeddings = extract_embeddings(
        extractor,
        validation.features,
    )

    model = build_cloud_tail()

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
        train_embeddings,
        train.labels,
        validation_data=(
            validation_embeddings,
            validation.labels,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model(
        validation_embeddings,
        training=False,
    ).numpy()

    if probabilities.shape != (
        validation.labels.shape[0],
        len(GESTURES),
    ):
        raise RuntimeError(
            "Unexpected cloud-tail "
            "validation output shape."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Cloud-tail probabilities contain "
            "NaN or infinity."
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / MODEL_FILENAME

    model.save(model_path)

    model_sha256 = sha256_file(
        model_path
    )

    history_path = (
        output_dir / HISTORY_FILENAME
    )

    save_history(
        history,
        history_path,
    )

    metadata = {
        "model_version": CLOUD_TAIL_VERSION,
        "model_purpose":
            "phase6-fixed-split3-cloud-tail",
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_edge_model_version":
            SOURCE_EDGE_MODEL_VERSION,
        "source_edge_model_sha256":
            SOURCE_MODEL_SHA256,
        "source_layer": "block3",
        "split_point": 3,
        "embedding_dimension":
            INPUT_EMBEDDING_DIM,
        "train_split": train.session,
        "validation_split":
            validation.session,
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
        "architecture": {
            "block4_units": 64,
            "block5_units": 32,
            "cloud_head_units":
                len(GESTURES),
        },
        "validation_metrics": metrics,
        "validation_confusion_matrix":
            matrix.tolist(),
        "model_sha256": model_sha256,
    }

    metadata_path = (
        output_dir / METADATA_FILENAME
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
        "PHASE 6 / M7 — FIXED SPLIT-3 "
        "CLOUD TAIL TRAINED"
    )
    print(
        "=========================================="
    )

    print(f"Model version:       {CLOUD_TAIL_VERSION}")
    print(
        f"Source edge model:   "
        f"{SOURCE_EDGE_MODEL_VERSION}"
    )
    print(
        f"Source SHA-256:       "
        f"{SOURCE_MODEL_SHA256}"
    )

    print(
        f"Train embeddings:    "
        f"{train_embeddings.shape}"
    )
    print(
        f"Validation embeddings: "
        f"{validation_embeddings.shape}"
    )

    print("Split point:          3")
    print(
        f"Embedding dimension:  "
        f"{INPUT_EMBEDDING_DIM}"
    )
    print("Test split loaded:    NO")

    print()

    for name, value in metrics.items():
        print(f"{name:<20} {value:.6f}")

    print()
    print("Validation confusion matrix:")
    print(matrix)

    print()
    print(
        f"Epochs ran:          "
        f"{len(history.history['loss'])}"
    )
    print(
        f"Cloud-tail SHA-256:  "
        f"{model_sha256}"
    )

    print()
    print(f"Model:     {model_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"History:   {history_path}")


if __name__ == "__main__":
    main()
