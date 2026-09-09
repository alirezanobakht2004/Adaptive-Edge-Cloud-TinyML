import hashlib
from ml.policy.binary_contract import ROOT, read_json


def test_bounded_cloud_candidate_retained_as_rejected_experiment():
    directory = ROOT / "data/processed/dataset-v1/features-v1/models/gesture-full-cloud-v1.1.0"
    config = read_json(directory / "training_config.json")
    report = read_json(directory / "evaluation_report.json")
    history = read_json(directory / "training_history.json")
    assert config["architecture"] == [10, 64, 48, 32, 64, 32, 5]
    assert config["fit_split"] == "train/session_01"
    assert config["test_used"] is False and report["test_partition_used"] is False
    assert 1 <= len(history) <= 200
    assert report["model_sha256"] == hashlib.sha256((directory / "gesture-full-cloud-v1.1.0.keras").read_bytes()).hexdigest()
    expected = report["validation_accuracy"] >= report["previous_validation_accuracy"] and report["local_cloud_counts"]["C"] > 0
    assert report["accepted"] == expected
    assert max(h["loss"] for h in history) != min(h["loss"] for h in history)
