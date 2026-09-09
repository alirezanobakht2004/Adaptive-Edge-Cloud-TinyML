import hashlib
import numpy as np
import pytest
import tensorflow as tf
from ml.policy.binary_contract import read_json, ROOT
from ml.policy.binary_policy_dataset import read_rows
from ml.policy.r1_dataset import OUTPUT as DATASET, VERSION as DATASET_VERSION
from ml.policy.train_meta_binary import OUTPUT, VERSION, evaluate
from ml.policy.binary_baseline import calibrate, select


def test_learned_artifact_and_heldout_metrics():
    model_path = OUTPUT / f"{VERSION}.keras"
    report = read_json(OUTPUT / "evaluation_report.json")
    assert report["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model.input_shape == (None, 6) and model.output_shape == (None, 2)
    assert [layer.units for layer in model.layers] == [8, 4, 2]
    rows = [r for r in read_rows(DATASET / f"{DATASET_VERSION}.jsonl") if r["partition"] == "holdout"]
    actions = np.argmax(model(np.asarray([r["policy_input"] for r in rows], dtype=np.float32), training=False).numpy(), axis=1)
    assert evaluate(rows, actions) == report["holdout"]
    assert report["holdout"]["per_class"]["CLOUD"]["recall"] is None


def test_thresholds_are_train_only_and_connectivity_checked():
    rows = read_rows(DATASET / f"{DATASET_VERSION}.jsonl")
    config = calibrate([r for r in rows if r["partition"] == "train"])
    assert config == read_json(OUTPUT / "rule_based_config.json")
    assert select(rows[0]["state"], config, cloud_available=False) == 0
    with pytest.raises(ValueError): calibrate(rows)


def test_float32_policy_parity_both_actions_and_normalization():
    report = read_json(OUTPUT / "export_report.json")
    content = (OUTPUT / f"{VERSION}.tflite").read_bytes()
    assert hashlib.sha256(content).hexdigest() == report["sha256"]
    assert len(content) == report["bytes"]
    vectors = np.load(OUTPUT / "parity_vectors.npz")
    assert set(vectors["actions"]) == {0, 1}
    contract = read_json(OUTPUT / "input_contract.json")
    offsets = [f["normalization"]["offset"] for f in contract["features"]]
    divisors = [f["normalization"]["divisor"] for f in contract["features"]]
    np.testing.assert_array_equal(((vectors["raw_states"].astype(np.float64) - offsets) / divisors).astype(np.float32), vectors["inputs"])
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    for x, expected, action in zip(vectors["inputs"], vectors["expected"], vectors["actions"]):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x[None])
        interpreter.invoke()
        actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
        np.testing.assert_allclose(actual, expected, atol=report["tolerance"], rtol=0)
        assert np.argmax(actual) == action
