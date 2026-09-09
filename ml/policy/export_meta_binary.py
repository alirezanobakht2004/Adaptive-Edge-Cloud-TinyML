"""Float32 binary policy artifact and fixed runtime-state parity vectors."""

import hashlib
import numpy as np
import tensorflow as tf
from .binary_contract import ROOT, read_json
from .binary_policy_dataset import read_rows, write_json
from .r1_dataset import CONFIG, OUTPUT as DATASET, VERSION as DATASET_VERSION, raw_vector
from .train_meta_binary import OUTPUT, VERSION


def export():
    model = tf.keras.models.load_model(OUTPUT / f"{VERSION}.keras", compile=False)
    config = read_json(CONFIG)
    rows = read_rows(DATASET / f"{DATASET_VERSION}.jsonl")
    raw = np.asarray([raw_vector(r["state"]) for r in rows], dtype=np.float32)
    offsets = np.array([f["normalization"]["offset"] for f in config["features"]])
    divisors = np.array([f["normalization"]["divisor"] for f in config["features"]])
    inputs = ((raw.astype(np.float64) - offsets) / divisors).astype(np.float32)
    expected_all = model(inputs, training=False).numpy()
    actions = expected_all.argmax(axis=1)
    chosen = [int(i) for action in (0, 1) for i in np.where(actions == action)[0][:5]]
    if set(actions[chosen]) != {0, 1}:
        raise ValueError("Model does not produce both actions; do not force outputs")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    content = converter.convert()
    path = OUTPUT / f"{VERSION}.tflite"
    path.write_bytes(content)
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    actual = []
    for vector in inputs[chosen]:
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], vector[None])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0])
    actual = np.asarray(actual)
    expected = expected_all[chosen]
    difference = float(np.max(np.abs(actual - expected)))
    parity = int(np.sum(actual.argmax(axis=1) == expected.argmax(axis=1)))
    if difference > 1e-5 or parity != len(chosen):
        raise ValueError("Float32 policy parity failed")
    oracle = read_json(ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json")
    by_window = {(r["session"], str(r["window_index"])): r for r in oracle["rows"]}
    gestures = [by_window[(rows[i]["source"]["session"], rows[i]["source"]["window"])] for i in chosen]
    np.savez(OUTPUT / "parity_vectors.npz", raw_states=raw[chosen], inputs=inputs[chosen], expected=expected, actions=expected.argmax(axis=1),
             features=np.asarray([r["normalized_features"] for r in gestures], dtype=np.float32),
             local_predictions=np.asarray([r["local"]["prediction"] for r in gestures]), seeds=np.asarray([r["seed"] for r in gestures]))
    report = {"version": VERSION, "format": "float32", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
              "vector_count": len(chosen), "max_abs_diff": difference, "action_parity_count": parity,
              "tolerance": 1e-5, "tolerance_source": "existing Phase7 float32 parity convention",
              "sample_ids": [rows[i]["sample_id"] for i in chosen], "input_preprocessing": "float32 raw state then double-precision frozen affine transform, cast to float32",
              "note": "Chosen by learned output to exercise both actions; not a classification accuracy evaluation"}
    write_json(OUTPUT / "export_report.json", report)
    write_json(ROOT / "docs/evidence/phase9_policy_tflite_parity.json", report)
    print(report)


if __name__ == "__main__":
    export()
