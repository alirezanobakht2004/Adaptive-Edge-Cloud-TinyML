import json
import pytest
from tools.policy_dataset.dataset_report import generate_report
from tools.policy_dataset.instrumentation import PolicySample
from tools.policy_dataset.serializer import serialize
from tools.policy_dataset.validator import validate_campaign
from tools.policy_dataset.device_trace import trace_to_record
from tests.test_policy_device_trace import trace_fixture


def test_report_collects_errors_and_duplicates(tmp_path):
    record = PolicySample.create(sample_id="s", device_id="d", window_id="w", run_id="r", session_id="s", scope="device").record
    path = tmp_path / "input.jsonl"
    invalid = json.loads(serialize(record))
    invalid["action"] = 4
    path.write_text(serialize(record) * 2 + json.dumps(invalid) + "\n{}\n", encoding="utf-8")
    report = generate_report([path], tmp_path / "report.json")
    assert not report["valid"]
    assert report["duplicate_samples"] == 1
    assert len(report["errors"]) == 3
    assert report["valid_observations"] == 1


def test_empty_dataset_does_not_claim_training_ready(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text("", encoding="utf-8")
    report = generate_report([path], tmp_path / "report.json")
    assert report["valid"]
    assert not report["training_ready"]
    assert report["valid_observations"] == 0


def test_campaign_checks_manifest_and_raw_projection(tmp_path):
    trace = trace_fixture()
    record = trace_to_record(trace, run_id="r", device_id="d", timestamp="2026-09-08T00:00:00+00:00",
                             session_id="s", true_class=0, hashes={})
    config = {"dataset_version": "policy_training_dataset_v1", "source_dataset_version": "dataset-v1",
              "feature_version": "features-v1", "model_version": "gesture-model-v1.1.0", "mode": "local",
              "action": 0, "samples_collected": 1, "samples_requested": 1, "status": "complete",
              "run_id": "r", "input_session": "s", "window_indices": [0], "artifact_hashes": {}}
    (tmp_path / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "raw_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    (tmp_path / "policy_training_dataset_v1.jsonl").write_text(serialize(record), encoding="utf-8")
    assert validate_campaign(tmp_path)["observations"] == 1
    trace["total_us"] += 1
    (tmp_path / "raw_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="projection mismatch"):
        validate_campaign(tmp_path)
    config["feature_version"] = "features-v2"
    (tmp_path / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        validate_campaign(tmp_path)
