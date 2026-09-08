"""Synthetic calibration fixtures only; none are hardware measurements."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.policy_dataset.schema import example_record, validate_record
from tools.policy_dataset.reward_engine import collect_candidates, evaluate_group, validate_config
from tools.policy_dataset.reward_engine.__main__ import calibrate_file, strict_json


def fixture():
    config = {"version": "reward-calibration-v1", "dataset_version": "policy_training_dataset_v1",
              "actions": {"0": "ALL_LOCAL", "1": "SPLIT1", "2": "SPLIT2", "3": "ALL_CLOUD"},
              "weights": {"accuracy": 1, "latency": .2, "communication": .1, "energy_proxy": .1},
              "scales": {"latency_ms": 10, "communication_bytes": 100, "energy_proxy": 2},
              "energy_kind": "simulated", "energy_proxy_version": "synthetic-test-v1",
              "energy_method": "Synthetic unit-test score; no measurements", "tie_tolerance": 0}
    record = example_record()
    record["metadata"].update(record_kind="observation", measurement_scope="host")
    record["state"]["uncertainty"].update(confidence=.8, entropy=.4, margin=.6)
    record["state"]["device"].update(free_heap=1000, tensor_arena_usage=100,
                                      cpu_frequency=240, inference_latency_estimate=1)
    record["state"]["network"].update(mqtt_rtt_ms=10, connection_state="connected")
    record["state"]["application"]["predicted_class"] = 0
    record["outcome"].update(status="success", true_class=0, predicted_class=0, confidence=.8)
    record["measurements"].update(total_latency_ms=10, request_bytes=50, response_bytes=50)
    energy = {"value": 2, "kind": config["energy_kind"], "proxy_version": config["energy_proxy_version"],
              "method": config["energy_method"]}
    group = {"version": "candidate-actions-v1", "decision_id": "synthetic-1",
             "matching_evidence": "unit-test fixture only", "decision": record, "candidates": []}
    for action in range(4):
        candidate = deepcopy(record)
        candidate["action"] = action
        candidate["measurements"]["total_latency_ms"] += action * 10
        group["candidates"].append({"record": candidate, "energy": deepcopy(energy)})
    return group, config


def test_deterministic_reward_and_schema_preserved():
    group, config = fixture()
    before = deepcopy(group)
    result = evaluate_group(group, config)
    assert result == evaluate_group(group, config)
    assert result["candidates"][0]["reward"] == pytest.approx(.6)
    assert result["optimal_action"] == 0
    assert result["training_enabled"] is False
    assert group == before
    for candidate in group["candidates"]:
        validate_record(candidate["record"])
        assert candidate["record"]["reward"]["final_reward"] is None


@pytest.mark.parametrize("problem", ["mapping", "missing", "duplicate", "state", "energy", "unknown",
                                      "label", "weight", "scale", "nan", "availability"])
def test_reject_unqualified_inputs(problem):
    group, config = fixture()
    first = group["candidates"][0]
    if problem == "mapping": config["actions"]["3"] = "SPLIT3"
    if problem == "missing": group["candidates"].pop()
    if problem == "duplicate": group["candidates"][1]["record"]["action"] = 0
    if problem == "state": first["record"]["state"]["network"]["mqtt_rtt_ms"] = 99
    if problem == "energy": first["energy"]["kind"] = "measured"
    if problem == "unknown": first["energy"]["value"] = None
    if problem == "label": first["record"]["outcome"]["true_class"] = 1
    if problem == "weight": config["weights"]["accuracy"] = -1
    if problem == "scale": config["scales"]["latency_ms"] = 0
    if problem == "nan": config["tie_tolerance"] = float("nan")
    if problem == "availability": group["decision"]["state"]["uncertainty"]["confidence"] = None
    with pytest.raises(ValueError): evaluate_group(group, config)


def test_ties_and_failure_costs():
    group, config = fixture()
    for candidate in group["candidates"]:
        candidate["record"]["measurements"]["total_latency_ms"] = 10
    result = evaluate_group(group, config)
    assert result["optimal_action"] is None
    assert result["optimal_actions"] == [0, 1, 2, 3]
    group["candidates"][0]["record"]["outcome"].update(status="timeout", predicted_class=None, confidence=None)
    result = evaluate_group(group, config)
    assert result["candidates"][0]["reward"] == pytest.approx(-.4)
    assert result["optimal_actions"] == [1, 2, 3]


def test_explicit_executors_and_isolation():
    group, config = fixture()
    decision = group["decision"]
    original = deepcopy(decision)
    seen = []
    def executor(record):
        seen.append(record["action"])
    kwargs = dict(decision_id="synthetic", matching_evidence="test", config=config,
                  energy_estimator=lambda record: deepcopy(group["candidates"][0]["energy"]))
    collected, result = collect_candidates(decision, executors={a: executor for a in range(4)}, **kwargs)
    assert seen == [0, 1, 2, 3]
    assert decision == original
    assert len(collected["candidates"]) == 4
    with pytest.raises(ValueError): collect_candidates(decision, executors={0: executor}, **kwargs)
    def mutate(record): record["state"]["device"]["free_heap"] = 1
    with pytest.raises(ValueError):
        collect_candidates(decision, executors={a: mutate for a in range(4)}, **kwargs)


def test_template_unconfigured_and_cli_batch_validation(tmp_path):
    template = Path("tools/policy_dataset/reward_engine/config_template.json")
    with pytest.raises(ValueError): validate_config(json.loads(template.read_text()))
    group, config = fixture()
    source, configuration, output = [tmp_path / name for name in ("groups.jsonl", "config.json", "rewards.jsonl")]
    source.write_text(json.dumps(group) + "\n")
    configuration.write_text(json.dumps(config))
    assert calibrate_file(source, configuration, output) == 1
    assert json.loads(output.read_text())["optimal_action"] == 0
    with pytest.raises(FileExistsError): calibrate_file(source, configuration, output)
    source.write_text((json.dumps(group) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate decision_id"):
        calibrate_file(source, configuration, tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()
    with pytest.raises(ValueError): strict_json('{"a":1,"a":2}')
