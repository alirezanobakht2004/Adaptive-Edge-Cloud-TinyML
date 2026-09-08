"""Frozen-model inference only, including explicit five-pass MC dropout diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.dataset.loader import CLASS_TO_ID, GESTURES
from ml.features.extractor import load_feature_split
from .inspect_kaggle_dataset import ROOT, write_json
from .convert_kaggle_dataset import OUTPUT, isolated_output, load_converted
from .run_feature_compatibility import MODEL_DIR


def uncertainty(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Expected finite five-class probability distributions")
    p = p / p.sum(axis=1, keepdims=True)  # Remove floating-point sum error, not recalibration.
    ordered = np.sort(p, axis=1)
    return {"confidence": p.max(axis=1), "entropy": np.clip(-np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1), 0, np.log(5)),
            "margin": ordered[:, -1] - ordered[:, -2]}


def describe(probabilities):
    metrics = uncertainty(probabilities)
    return {name: {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                   "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
                   "histogram_edges": np.linspace(0, np.log(5) if name == "entropy" else 1, 11).tolist(),
                   "histogram_counts": np.histogram(values, bins=np.linspace(0, np.log(5) if name == "entropy" else 1, 11))[0].tolist()}
            for name, values in metrics.items()}


def mapped_evaluation(probabilities, labels, mapping):
    metrics = uncertainty(probabilities)
    if len(labels) != len(probabilities):
        raise ValueError("Labels/probabilities length mismatch")
    if set(map(str, labels)) != set(mapping):
        raise ValueError("Every external label must have an explicit mapping entry")
    targets = {}
    for label, rule in mapping.items():
        if not rule.get("basis") or (rule.get("target") is not None and rule["target"] not in CLASS_TO_ID):
            raise ValueError("Unsupported or undocumented label mapping")
        if rule["target"] is not None:
            targets[label] = rule["target"]
    matrix = np.zeros((5, 5), dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    errors, confident_errors = 0, 0
    for index, label in enumerate(labels):
        if str(label) in targets:
            matrix[CLASS_TO_ID[targets[str(label)]], predictions[index]] += 1
            wrong = CLASS_TO_ID[targets[str(label)]] != predictions[index]
            errors += int(wrong)
            confident_errors += int(wrong and metrics["confidence"][index] >= .9)
    count = int(matrix.sum())
    return {"mapped_classes": targets, "unmapped_classes": sorted(set(mapping) - set(targets)),
            "accuracy_evaluated_samples": count, "excluded_from_accuracy": len(labels) - count,
            "mapped_subset_accuracy": float(np.trace(matrix) / count) if count else None,
            "mapped_errors": errors, "mapped_errors_with_confidence_at_least_0_9": confident_errors,
            "overall_external_accuracy": None, "confusion_matrix": matrix.tolist(),
            "confusion_matrix_order": list(GESTURES), "confusion_matrix_axes": "rows=true, columns=predicted; unmapped rows excluded"}


def infer(model, features, seed=42, passes=5):
    import tensorflow as tf
    if passes < 2:
        raise ValueError("MC analysis requires at least two passes")
    # All learned layers execute in inference mode. Only this dropout mask is stochastic.
    prefix = tf.keras.Model(model.input, model.get_layer("block3").output)
    embedding = prefix(features, training=False)
    dropout = tf.keras.layers.Dropout(model.get_layer("mc_dropout").rate, seed=seed)
    head = model.get_layer("edge_head")
    deterministic = model(features, training=False).numpy()
    samples = np.asarray([head(dropout(embedding, training=True), training=False).numpy() for _ in range(passes)])
    return {"deterministic": deterministic, "mc5": samples.mean(axis=0)}


def run(directory=OUTPUT, mapping_path=None, compatibility_path=None, output=None):
    import tensorflow as tf
    from ml.training.train_cloud_tail import load_source_model
    directory = isolated_output(directory)
    _, metadata = load_converted(directory)
    mapping_path = Path(mapping_path or directory.parent / "external_label_mapping.json")
    mapping_doc = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping_doc["mapping_version"] != "external-label-mapping-v1" or mapping_doc["source_sha256"] != metadata["source_sha256"]:
        raise ValueError("Label mapping version/source mismatch")
    compatibility = json.loads(Path(compatibility_path or ROOT / "docs/evidence/external_feature_compatibility_report.json").read_text(encoding="utf-8"))
    archive = directory / "diagnostic_features_v1.npz"
    if (compatibility.get("feature_version") != "features-v1" or compatibility["source_sha256"] != metadata["source_sha256"] or compatibility.get("feature_archive_sha256") != hashlib.sha256(archive.read_bytes()).hexdigest()):
        raise ValueError("Feature evaluation provenance mismatch; rerun compatibility first")
    with np.load(archive, allow_pickle=False) as data:
        features, labels, ids, normalized = data["features"], data["labels"], data["sample_ids"], data["normalized_features"]
    tf.keras.utils.set_random_seed(42)
    model_path = MODEL_DIR / "gesture-model-v1.1.0.keras"
    model = load_source_model(model_path)
    model.trainable = False
    frozen_weights = [weight.copy() for weight in model.get_weights()]
    actual_normalized = model.get_layer("feature_normalization")(features).numpy()
    if not np.allclose(actual_normalized, normalized, atol=1e-4, rtol=1e-5):
        raise ValueError("Feature normalization differs from frozen Keras normalization")
    external = infer(model, features)
    internal = load_feature_split("validation")
    ours = infer(model, internal.features)
    if any(not np.array_equal(before, after) for before, after in zip(frozen_weights, model.get_weights())):
        raise RuntimeError("Inference changed frozen model weights")
    probabilities_path = directory / "evaluation_probabilities.npz"
    np.savez_compressed(probabilities_path, external_deterministic=external["deterministic"], external_mc5=external["mc5"],
                        internal_deterministic=ours["deterministic"], internal_mc5=ours["mc5"],
                        external_labels=labels, external_sample_ids=ids, internal_labels=internal.labels)
    report = {"model_version": "gesture-model-v1.1.0", "feature_version": "features-v1", "source_sha256": metadata["source_sha256"],
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "model_weights_unchanged": True,
              "training_performed": False, "normalization_refitted": False, "number_of_evaluated_samples": len(features),
              "diagnostic_adapter": compatibility["diagnostic_adapter"], "native_input_compatible": compatibility["native_compatible"],
              "mapping": mapping_doc, "seed": 42, "mc_passes": 5, "dropout_rate": float(model.get_layer("mc_dropout").rate),
              "dependencies": {"tensorflow": tf.__version__, "numpy": np.__version__},
              "normalization_max_abs_difference": float(np.max(np.abs(actual_normalized - normalized))),
              "probabilities_archive_sha256": hashlib.sha256(probabilities_path.read_bytes()).hexdigest(),
              "feature_archive_sha256": compatibility["feature_archive_sha256"],
              "internal_reference": {"dataset_version": "dataset-v1", "split": "validation", "session": internal.session,
                                     "samples": len(internal.features), "test_split_used": False},
              "modes": {},
              "limitations": ["External inference uses diagnostic interpolation and a held endpoint, not native 100 Hz measurements.",
                              "Only broad idle semantics are mapped. Overall external gesture accuracy is undefined.",
                              "Axes are preserved; mounting, participants, calibration and capture firmware revision are unknown.",
                              "Class composition differs between external and internal cohorts; pooled uncertainty is descriptive, not calibrated OOD detection.",
                              "Five-pass host MC dropout is an analysis protocol, not an ESP32 parity or latency result."]}
    for mode in external:
        report["modes"][mode] = {**mapped_evaluation(external[mode], labels, mapping_doc["mapping"]),
            "external_uncertainty": describe(external[mode]), "internal_uncertainty": describe(ours[mode]),
            "internal_accuracy": float(np.mean(np.argmax(ours[mode], axis=1) == internal.labels)),
            "internal_by_class": {gesture: describe(ours[mode][internal.labels == class_id]) for class_id, gesture in enumerate(GESTURES)},
            "external_by_original_label": {str(label): {"samples": int(np.sum(labels == label)),
                 "uncertainty": describe(external[mode][labels == label]),
                 "prediction_counts": {gesture: int(np.sum(np.argmax(external[mode][labels == label], axis=1) == class_id))
                                       for class_id, gesture in enumerate(GESTURES)}} for label in np.unique(labels)}}
    write_json(output or ROOT / "docs/evidence/external_model_evaluation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, default=OUTPUT)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.converted, args.mapping, args.compatibility_report, args.output)
    print(json.dumps({"inference_samples": report["number_of_evaluated_samples"],
                      "mapped_subset_accuracy": report["modes"]["deterministic"]["mapped_subset_accuracy"]}))


if __name__ == "__main__":
    main()
