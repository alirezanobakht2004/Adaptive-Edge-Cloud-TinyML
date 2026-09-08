"""Development-only LOCAL MC5 versus full-cloud audit; never loads TEST."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import GESTURES
from ml.export.uncertainty_deployment import explicit_dropout_edge_head
from ml.features.extractor import load_feature_split
from ml.training.train_cloud_tail import load_source_model, SOURCE_MODEL_RELATIVE_PATH, SOURCE_MODEL_SHA256
from server.app.cloud_full import FullCloudInference
from .binary_contract import ROOT
from .binary_policy_dataset import write_json
from .r1_reuse import salvage_m30

SEED_BASE = 42000  # Same seeded diagnostic sequence as M30, not a new PRNG.
OUTPUT = ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json"


def firmware_masks(seed):
    """Exact xorshift32/threshold algorithm in the unchanged ESP32 uncertainty.cpp."""
    state = seed or 0xA341316C
    masks = np.empty((5, 32), dtype=np.uint8)
    for index in range(160):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        state = state or 0xA341316C
        masks.flat[index] = int(state < 3435973837)
    return masks


def local_probabilities(embedding, kernel, bias, seed):
    passes = [explicit_dropout_edge_head(embedding, mask, kernel, bias, dropout_rate=.2)
              for mask in firmware_masks(seed)]
    return np.mean(passes, axis=0, dtype=np.float32)


def uncertainty(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    ordered = np.sort(p)
    return {"confidence": float(ordered[-1]), "entropy": float(-np.sum(p * np.log(np.maximum(p, 1e-30)))),
            "margin": float(ordered[-1] - ordered[-2])}


def category(local_correct, cloud_correct):
    return { (True, True): "A", (True, False): "B", (False, True): "C", (False, False): "D" }[(local_correct, cloud_correct)]


def audit(output=OUTPUT):
    source = load_source_model(ROOT / SOURCE_MODEL_RELATIVE_PATH)
    cloud = FullCloudInference()
    raw_prefix = tf.keras.Model(source.input, source.get_layer("block3").output)
    kernel, bias = source.get_layer("edge_head").get_weights()
    normalization = next(layer for layer in source.layers if isinstance(layer, tf.keras.layers.Normalization))
    rows, split_hashes = [], {}
    for split in ("train", "validation"):
        data = load_feature_split(split)
        embeddings = raw_prefix(data.features, training=False).numpy()
        normalized = normalization(data.features, training=False).numpy()
        cloud_outputs = cloud.model(normalized, training=False).numpy()
        split_hashes[split] = {"session": data.session, "count": len(data.labels),
            "features_sha256": hashlib.sha256(data.features.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(data.labels.tobytes()).hexdigest()}
        for i, (embedding, cp, label) in enumerate(zip(embeddings, cloud_outputs, data.labels)):
            lp = local_probabilities(embedding, kernel, bias, SEED_BASE + i)
            local, remote = int(np.argmax(lp)), int(np.argmax(cp))
            correct_local, correct_cloud = local == int(label), remote == int(label)
            rows.append({"source_id": f"dataset-v1/{data.session}/{i}", "session": data.session,
                         "split": split, "window_index": i, "csv_path": data.csv_paths[i].relative_to(ROOT).as_posix(),
                         "true_class": int(label), "seed": SEED_BASE + i,
                         "normalized_features": normalized[i].tolist(),
                         "local": {"prediction": local, "correct": correct_local, **uncertainty(lp), "probabilities": lp.tolist()},
                         "cloud": {"prediction": remote, "correct": correct_cloud, **uncertainty(cp), "probabilities": cp.tolist()},
                         "category": category(correct_local, correct_cloud), "disagreement": local != remote})
    summary = {}
    for split in ("train", "validation"):
        subset = [r for r in rows if r["split"] == split]
        counts = {k: sum(r["category"] == k for r in subset) for k in "ABCD"}
        summary[split] = {"samples": len(subset), "counts": counts,
                          "local_accuracy": sum(r["local"]["correct"] for r in subset) / len(subset),
                          "cloud_accuracy": sum(r["cloud"]["correct"] for r in subset) / len(subset),
                          "by_class": {name: {"counts": dict(Counter(r["category"] for r in subset if r["true_class"] == j)),
                                              "disagreements": sum(r["disagreement"] for r in subset if r["true_class"] == j)}
                                       for j, name in enumerate(GESTURES)}}
    salvage = salvage_m30()
    comparisons = []
    for saved in salvage:
        record = next(r for r in rows if r["session"] == "session_02" and r["window_index"] == int(saved["source_window"]))
        comparisons.append({"window": saved["source_window"],
            "action_match": record["local"]["prediction"] == saved["local"]["prediction"],
            "max_uncertainty_difference": max(abs(record["local"][key] - saved["state"]["uncertainty"][key]) for key in ("confidence", "entropy", "margin"))})
    if not all(c["action_match"] for c in comparisons) or max(c["max_uncertainty_difference"] for c in comparisons) > 2e-5:
        raise ValueError("Desktop seeded LOCAL reference differs from M30 hardware evidence")
    report = {"version": "local-cloud-oracle-audit-v1", "dataset_version": "dataset-v1", "feature_version": "features-v1",
              "local_model_version": "gesture-model-v1.1.0", "cloud_model_version": cloud.model_version,
              "local_sha256": SOURCE_MODEL_SHA256, "cloud_artifact_hashes": cloud.artifact_hashes,
              "test_partition_used": False, "training_performed": False,
              "prediction_provenance": "Desktop measured model inference; firmware-matched seeded MC5 masks, not new ESP32 measurements",
              "runtime_reference_check": comparisons, "split_hashes": split_hashes, "summary": summary,
              "category_definitions": {"A": "LOCAL correct / CLOUD correct", "B": "LOCAL correct / CLOUD wrong", "C": "LOCAL wrong / CLOUD correct", "D": "both wrong"},
              "rows": rows, "limitations": ["Development data, not final TEST results.", "One deterministic MC seed per window; production stochastic masks can differ.",
                 "Cloud validation was used in historical early stopping; not an independent accuracy test.", "Server capacity alone does not establish an accuracy advantage."]}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import json
    print(json.dumps(audit(args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
