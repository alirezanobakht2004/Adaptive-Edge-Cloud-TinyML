"""One bounded R1 server-only experiment; existing architecture, TRAIN-only fitting."""

import hashlib
import platform
import sys

import numpy as np
import tensorflow as tf

from ml.features.extractor import load_feature_split
from server.app.cloud_full import FullCloudInference
from ml.training.train_cloud_tail import load_source_model, SOURCE_MODEL_RELATIVE_PATH
from .binary_contract import ROOT, read_json
from .binary_policy_dataset import write_json

VERSION = "gesture-full-cloud-v1.1.0"
OUTPUT = ROOT / f"data/processed/dataset-v1/features-v1/models/{VERSION}"


def train():
    if OUTPUT.exists():
        raise ValueError("One bounded experiment only; existing candidate output must not be overwritten")
    oracle = read_json(ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json")
    if oracle["summary"]["validation"]["counts"]["C"] != 0:
        raise ValueError("Case B not established; do not start model search")
    config = {"version": "r1-cloud-candidate-experiment-v1", "model_version": VERSION,
              "seed": 42, "learning_rate": 1e-4, "batch_size": 32, "max_epochs": 200,
              "patience": 20, "selection": "minimum validation cross entropy; restore best weights",
              "architecture": [10, 64, 48, 32, 64, 32, 5],
              "change": "Unfreeze private copy of existing full-cloud B1..B5/head; no architecture change",
              "fit_split": "train/session_01", "selection_split": "validation/session_02",
              "test_used": False, "feature_version": "features-v1", "dataset_version": "dataset-v1",
              "acceptance": "validation accuracy >= previous cloud and at least one LOCAL-wrong/CLOUD-correct validation window"}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "training_config.json", config)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    previous = FullCloudInference()
    # A fresh flat graph avoids cloning frozen-variable flags from nested models.
    inputs = tf.keras.Input(shape=(10,), name="normalized_features_v1")
    value = inputs
    copied_layers = []
    for parent in (previous.prefix, previous.tail._model):
        for layer in parent.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                specification = layer.get_config()
                specification["trainable"] = True
                copied = tf.keras.layers.Dense.from_config(specification)
                value = copied(value)
                copied.set_weights(layer.get_weights())
                copied_layers.append(copied)
    model = tf.keras.Model(inputs, value, name="gesture_full_cloud_r1_candidate")
    if len(copied_layers) != 6 or len(model.trainable_weights) != 12:
        raise ValueError("Expected six trainable Dense layers; refusing a frozen no-op fit")
    source = load_source_model(ROOT / SOURCE_MODEL_RELATIVE_PATH)
    norm = next(l for l in source.layers if isinstance(l, tf.keras.layers.Normalization))
    train_data, validation = load_feature_split("train"), load_feature_split("validation")
    x, xv = norm(train_data.features).numpy(), norm(validation.features).numpy()
    initial = model(xv, training=False).numpy()
    if not np.allclose(initial, previous.model(xv, training=False).numpy(), atol=1e-6, rtol=0):
        raise ValueError("Private trainable copy changed the initial model outputs")
    y, yv = train_data.labels, validation.labels
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    random = np.random.default_rng(config["seed"])
    best, stale, best_weights, history = float("inf"), 0, None, []
    for epoch in range(config["max_epochs"]):
        model.reset_metrics()
        indices = random.permutation(len(y))
        for start in range(0, len(y), config["batch_size"]):
            batch = indices[start:start + config["batch_size"]]
            metrics = model.train_on_batch(x[batch], y[batch], return_dict=True)
        probabilities = model(xv, training=False).numpy()
        loss = float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(yv, probabilities).numpy()))
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == yv))
        history.append({"epoch": epoch + 1, "loss": float(metrics["loss"]), "accuracy": float(metrics["accuracy"]),
                        "val_loss": loss, "val_accuracy": accuracy})
        if loss < best:
            best, stale, best_weights = loss, 0, model.get_weights()
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch+1} validation_loss={loss:.6f} validation_accuracy={accuracy:.6f}", flush=True)
        if stale >= config["patience"]:
            break
    model.set_weights(best_weights)
    probabilities = model(xv, training=False).numpy()
    prediction = np.argmax(probabilities, axis=1)
    local = np.array([r["local"]["prediction"] for r in oracle["rows"] if r["split"] == "validation"])
    counts = {"A": int(np.sum((local == yv) & (prediction == yv))), "B": int(np.sum((local == yv) & (prediction != yv))),
              "C": int(np.sum((local != yv) & (prediction == yv))), "D": int(np.sum((local != yv) & (prediction != yv)))}
    accuracy = float(np.mean(prediction == yv))
    old_accuracy = oracle["summary"]["validation"]["cloud_accuracy"]
    path = OUTPUT / f"{VERSION}.keras"
    model.save(path)
    report = {"model_version": VERSION, "previous_model_version": previous.model_version,
              "validation_accuracy": accuracy, "previous_validation_accuracy": old_accuracy,
              "local_cloud_counts": counts, "accepted": accuracy >= old_accuracy and counts["C"] > 0,
              "epochs": len(history), "best_validation_loss": best, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "test_partition_used": False, "validation_predictions": prediction.tolist(),
              "limitations": "Single development experiment; validation-selected model, not final test accuracy or deployed evidence"}
    write_json(OUTPUT / "evaluation_report.json", report)
    write_json(OUTPUT / "training_history.json", history)
    write_json(OUTPUT / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "tensorflow": tf.__version__, "numpy": np.__version__})
    write_json(ROOT / "docs/evidence/phase9_cloud_candidate_experiment.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    train()
