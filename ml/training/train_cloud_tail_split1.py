"""Train only Split1 B2/B3/head on frozen B1 embeddings, without TEST data."""

import json

import numpy as np
import tensorflow as tf

from ml.dataset.loader import CLASS_TO_ID, DATASET_VERSION
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.cloud_tail_split1 import (
    CLOUD_TAIL_VERSION, MODEL_DIR, MODEL_PATH, MODEL_METADATA, SOURCE_MODEL_VERSION,
    build_cloud_tail_split1, validate_cloud_tail_split1,
)
from ml.models.split_models import validate_source_model
from ml.training.train_cloud_tail import (
    SOURCE_MODEL_RELATIVE_PATH, SOURCE_MODEL_SHA256, load_source_model,
    load_training_splits, project_root, save_history, set_reproducibility, sha256_file,
)


def main() -> None:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise RuntimeError(f"Refusing to overwrite Split1 artifacts: {MODEL_DIR}")
    set_reproducibility()
    source_path = project_root() / SOURCE_MODEL_RELATIVE_PATH
    source = load_source_model(source_path)
    validate_source_model(source)
    source.trainable = False
    frozen_weights = [w.copy() for w in source.get_weights()]
    prefix = tf.keras.Model(source.input, source.get_layer("block1").output)
    prefix.trainable = False
    train, validation = load_training_splits()
    train_embeddings = prefix(train.features, training=False).numpy()
    validation_embeddings = prefix(validation.features, training=False).numpy()
    for embeddings in (train_embeddings, validation_embeddings):
        if embeddings.shape[1:] != (64,) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid Split1 embeddings")
    model = build_cloud_tail_split1()
    for target, original in (("split1_block2", "block2"), ("split1_block3", "block3"),
                             ("split1_cloud_head", "edge_head")):
        model.get_layer(target).set_weights(source.get_layer(original).get_weights())
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_embeddings, train.labels,
        validation_data=(validation_embeddings, validation.labels),
        epochs=200, batch_size=32, shuffle=True, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True)],
    )
    for before, after in zip(frozen_weights, source.get_weights()):
        np.testing.assert_array_equal(before, after)
    if sha256_file(source_path) != SOURCE_MODEL_SHA256:
        raise RuntimeError("Frozen source artifact changed")
    validate_cloud_tail_split1(model)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_history(history, MODEL_DIR / "training_history.csv")
    metadata = {
        **MODEL_METADATA, "model_version": CLOUD_TAIL_VERSION,
        "model_purpose": "phase7-split1-cloud-tail",
        "source_model_version": SOURCE_MODEL_VERSION,
        "source_edge_model_version": SOURCE_MODEL_VERSION,
        "source_edge_model_sha256": SOURCE_MODEL_SHA256,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "model_sha256": sha256_file(MODEL_PATH),
        "split_point": 1, "embedding_dimension": 64,
        "input_dimension": 64, "output_dimension": 5,
        "dataset_version": DATASET_VERSION, "feature_version": FEATURE_VERSION,
        "class_to_id": CLASS_TO_ID, "test_split_used": False,
        "train_session": train.session, "validation_session": validation.session,
        "train_samples": len(train.labels), "validation_samples": len(validation.labels),
        "seed": 42, "epochs_ran": len(history.history["loss"]),
        "frozen_prefix_verified": True,
        "initial_weights": "source block2, block3, edge_head; copied, then trained",
        "tflite_exported": False,
    }
    for filename in ("metadata.json", "split1_export_report.json"):
        (MODEL_DIR / filename).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"SPLIT1_TRAINING_PASS: {MODEL_PATH}")


if __name__ == "__main__":
    main()
