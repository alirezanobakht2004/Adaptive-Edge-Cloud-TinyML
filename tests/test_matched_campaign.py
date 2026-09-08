"""Synthetic matched-condition fixtures; never exported as measured campaign data."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_policy_device_trace import trace_fixture
from tools.policy_dataset.device_trace import trace_to_record
from tools.policy_dataset.matched_dataset import build_sample, validate_sample, validate_file, validate_condition


def fixture():
    condition = json.loads(Path("tools/policy_dataset/conditions/connected_lab_v1.json").read_text())
    entries = []
    for repeat in range(4):
        for position in range(4):
            action = (repeat + position) % 4
            trace = trace_fixture()
            trace.update(action=action, request_id=f"synthetic-{repeat}-{action}", probe_us=1000,
                         action_us=500 + action * 1000, total_us=2000 + 500 + action * 1000,
                         request_bytes=100 if action else 0, response_bytes=100 if action else 0,
                         receive_us=9000 if action else None, rssi_dbm=-45)
            record = trace_to_record(trace, run_id="synthetic-test", device_id="fixture",
                timestamp="2026-09-08T00:00:00+00:00", session_id="synthetic-test", true_class=0, hashes={})
            if action:
                record["measurements"].update(request_receive_time="2026-09-08T00:00:00+00:00",
                                              response_publish_time="2026-09-08T00:00:00+00:00")
            guard = dict(window_id=0, repeat=repeat, position=position, action=action, shared_us=2000,
                         snapshot_us=10000, age_us=5000, snapshot_rssi=-45, guard_heap=10000,
                         guard_psram=50, guard_rssi=-46, guard_cpu=240, guard_connected=True, guard_rtt_us=1000)
            entries.append(dict(trace=trace, guard=guard, record=record))
    return entries, condition


def test_deterministic_v2_reward_and_validation(tmp_path):
    entries, condition = fixture()
    original = deepcopy(entries)
    sample = build_sample(entries, condition, "synthetic")
    assert sample == build_sample(entries, condition, "synthetic")
    validate_sample(sample)
    assert sample["label"]["optimal_action"] == 0
    assert len(sample["candidate_actions"]) == 4
    assert entries == original
    assert all(e["record"]["reward"]["final_reward"] is None for e in entries)
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    report = validate_file(path)
    assert report["candidate_executions"] == 16
    assert report["action_coverage"] == {str(a): 4 for a in range(4)}
    assert report["training_allowed"] is False
    path.write_text((json.dumps(sample) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"): validate_file(path)


@pytest.mark.parametrize("problem", ["missing", "state", "drift", "disconnected", "order", "duplicate",
                                      "measurement", "cloud", "snapshot", "label", "energy", "mapping"])
def test_reject_invalid_matching(problem):
    entries, condition = fixture()
    if problem == "missing": entries.pop()
    if problem == "state": entries[0]["record"]["state"]["device"]["free_heap"] += 1
    if problem == "drift": entries[0]["guard"]["guard_rtt_us"] = 100000
    if problem == "disconnected": entries[0]["guard"]["guard_connected"] = False
    if problem == "order": entries[0]["guard"]["position"] = 1
    if problem == "duplicate": entries[0]["trace"]["request_id"] = entries[1]["trace"]["request_id"]
    if problem == "measurement": entries[0]["record"]["measurements"]["total_latency_ms"] = None
    if problem == "cloud": entries[1]["trace"]["cloud_ms"] = None
    if problem == "snapshot": entries[0]["guard"]["snapshot_us"] += 1
    if problem == "label": entries[0]["record"]["outcome"]["true_class"] = 1
    if problem == "energy": condition["reward"]["energy_kind"] = "measured"
    if problem == "mapping": condition["reward"]["actions"]["3"] = "SPLIT3"
    with pytest.raises(ValueError): build_sample(entries, condition, "synthetic")


def test_reject_tampered_derived_labels_and_reward():
    entries, condition = fixture()
    sample = build_sample(entries, condition, "synthetic")
    sample["label"]["optimal_action"] = 3
    with pytest.raises(ValueError): validate_sample(sample)
    sample = build_sample(entries, condition, "synthetic")
    sample["candidate_actions"][0]["measurements"][0]["reward"] += .01
    with pytest.raises(ValueError): validate_sample(sample)
    condition["reward"]["energy_method"] = "unspecified method"
    with pytest.raises(ValueError): validate_condition(condition)


def test_tied_mean_rewards_have_no_arbitrary_label():
    entries, condition = fixture()
    condition["reward"]["weights"] = dict(accuracy=1, latency=0, communication=0, energy_proxy=0)
    result = build_sample(entries, condition, "synthetic")
    assert result["label"]["optimal_action"] is None
    assert result["label"]["optimal_actions"] == [0, 1, 2, 3]


def test_campaign_checks_raw_server_and_source_labels(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    import numpy as np
    from ml.features import extractor
    from tools.policy_dataset.validate_matched_dataset import validate_campaign
    from tools.policy_dataset.run_matched_campaign import write_json
    entries, condition = fixture()
    features = SimpleNamespace(features=np.zeros((1, 10), dtype=np.float32),
                               labels=np.array([0]), session="synthetic-test")
    monkeypatch.setattr(extractor, "load_feature_split", lambda split: features)
    versions = {"1": "gesture-cloud-tail-split1-v1.0.0", "2": "gesture-cloud-tail-split2-v1.0.0",
                "3": "gesture-full-cloud-v1.0.0"}
    manifest = dict(dataset_version="policy_training_dataset_v2", status="complete", matched_samples=1,
        samples_requested=1, repeats=4, rejected_windows=[], source_session=features.session,
        feature_matrix_sha256=hashlib.sha256(features.features.tobytes()).hexdigest(),
        labels_sha256=hashlib.sha256(features.labels.tobytes()).hexdigest(), condition=condition,
        window_indices=[0], run_id="synthetic-test", artifact_hashes={}, model_versions=versions)
    write_json(tmp_path / "configuration.json", manifest)
    def jsonl(name, values):
        (tmp_path / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    jsonl("policy_training_dataset_v2.jsonl", [build_sample(entries, condition, "synthetic")])
    raw = [{"trace": e["trace"], "guard": e["guard"]} for e in entries]
    jsonl("raw_entries.jsonl", raw)
    events = []
    for entry in entries:
        trace = entry["trace"]
        if trace["action"]:
            events.append(dict(request_id=trace["request_id"], publish_rc=0,
                request_bytes=trace["request_bytes"], response_bytes=trace["response_bytes"],
                request_receive_time=entry["record"]["measurements"]["request_receive_time"],
                response_publish_time=entry["record"]["measurements"]["response_publish_time"],
                response=dict(action=trace["action"], model_version=versions[str(trace["action"])],
                              predicted_class_id=trace["prediction"], confidence=trace["result_confidence"],
                              server_latency_ms=trace["cloud_ms"])))
    jsonl("server_events.jsonl", events)
    assert validate_campaign(tmp_path)["source_and_server_evidence_validated"]
    events[0]["response"]["model_version"] = "wrong"
    jsonl("server_events.jsonl", events)
    with pytest.raises(ValueError, match="Server/device"): validate_campaign(tmp_path)
