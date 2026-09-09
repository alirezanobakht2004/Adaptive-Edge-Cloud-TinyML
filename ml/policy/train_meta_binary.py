"""One fixed compact R1 binary policy; never tune against the group holdout."""

import hashlib
import platform
import sys

import numpy as np
import tensorflow as tf

from .binary_contract import ROOT, read_json
from .binary_policy_dataset import read_rows, write_json
from .r1_dataset import CONFIG, OUTPUT as DATASET, VERSION as DATASET_VERSION, validate

VERSION = "meta-policy-v1.0.0"
OUTPUT = ROOT / f"data/policy/models/{VERSION}"


def evaluate(rows, actions):
    if not rows:
        return {"samples": 0, "metrics": None}
    labels = np.array([r["label"]["optimal_action"] for r in rows])
    actions = np.asarray(actions, dtype=int)
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(labels, actions):
        matrix[truth, prediction] += 1
    per_class = {}
    for i, name in enumerate(("LOCAL", "CLOUD")):
        precision = matrix[i, i] / matrix[:, i].sum() if matrix[:, i].sum() else 0.
        recall = matrix[i, i] / matrix[i].sum() if matrix[i].sum() else None
        f1 = 2 * precision * (recall or 0) / (precision + (recall or 0)) if precision + (recall or 0) else 0.
        per_class[name] = {"precision": float(precision), "recall": None if recall is None else float(recall),
                           "f1": float(f1), "support": int(matrix[i].sum())}
    values = np.array([[v["value"] for v in r["label"]["rewards"]] for r in rows])
    achieved = values[np.arange(len(rows)), actions]
    oracle = values.max(axis=1)
    return {"samples": len(rows), "accuracy": float(np.mean(labels == actions)), "confusion_matrix": matrix.tolist(),
            "per_class": per_class, "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])),
            "zero_support_convention": "undefined recall is null; zero division F1=0 included in macro F1",
            "local_selection_rate": float(np.mean(actions == 0)), "cloud_selection_rate": float(np.mean(actions == 1)),
            "mean_reward": float(achieved.mean()), "binary_oracle_reward": float(oracle.mean()),
            "mean_regret": float((oracle - achieved).mean())}


def train():
    gate = validate()
    if not gate["training_allowed"]:
        raise ValueError("Policy training gate failed")
    if OUTPUT.exists():
        raise ValueError("Single model run only; output exists")
    all_rows = read_rows(DATASET / f"{DATASET_VERSION}.jsonl")
    fit = [r for r in all_rows if r["partition"] == "train"]
    holdout = [r for r in all_rows if r["partition"] == "holdout"]
    x = np.asarray([r["policy_input"] for r in fit], dtype=np.float32)
    y = np.asarray([r["label"]["optimal_action"] for r in fit], dtype=np.int32)
    class_weights = len(y) / (2 * np.bincount(y, minlength=2))
    config = {"version": VERSION, "seed": 42, "architecture": [6, 8, 4, 2], "epochs": 200,
              "batch_size": 32, "learning_rate": .001, "optimizer": "Adam",
              "class_weights": class_weights.tolist(), "class_weight_source": "inverse training class frequency; no relabeling or oversampled independent windows",
              "holdout_use": "evaluation only after fixed 200 epochs; no tuning/early stopping",
              "dataset_manifest_sha256": hashlib.sha256((DATASET / "manifest.json").read_bytes()).hexdigest()}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "training_config.json", config)
    write_json(OUTPUT / "input_contract.json", read_json(CONFIG))
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([tf.keras.Input(shape=(6,)), tf.keras.layers.Dense(8, activation="relu"),
                                 tf.keras.layers.Dense(4, activation="relu"), tf.keras.layers.Dense(2, activation="softmax")], name="meta_binary_r1")
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    random = np.random.default_rng(config["seed"])
    history = []
    for epoch in range(config["epochs"]):
        model.reset_metrics()
        shuffled = random.permutation(len(y))
        for start in range(0, len(y), config["batch_size"]):
            batch = shuffled[start:start + config["batch_size"]]
            metrics = model.train_on_batch(x[batch], y[batch], sample_weight=class_weights[y[batch]], return_dict=True)
        history.append({"epoch": epoch + 1, **{k: float(v) for k, v in metrics.items()}})
        if (epoch + 1) % 25 == 0:
            print(history[-1], flush=True)
    path = OUTPUT / f"{VERSION}.keras"
    model.save(path)
    actions = np.argmax(model(np.asarray([r["policy_input"] for r in holdout], dtype=np.float32), training=False).numpy(), axis=1)
    report = {"model_version": VERSION, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "partition": "group-held-out development policy data; not final gesture TEST", "holdout": evaluate(holdout, actions),
              "by_provenance": {}, "by_profile": {}, "limitations": gate["limitations"]}
    for key, field in (("by_provenance", "kind"), ("by_profile", "profile")):
        for value in sorted({r["provenance"][field] for r in holdout}):
            indices = [i for i, r in enumerate(holdout) if r["provenance"][field] == value]
            report[key][value] = evaluate([holdout[i] for i in indices], actions[indices])
    write_json(OUTPUT / "evaluation_report.json", report)
    write_json(OUTPUT / "training_history.json", history)
    write_json(OUTPUT / "environment.json", {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__, "numpy": np.__version__})
    write_json(ROOT / "docs/evidence/phase9_learned_policy_evaluation.json", report)
    print(report["holdout"], flush=True)


if __name__ == "__main__":
    train()
