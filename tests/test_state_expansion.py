from copy import deepcopy

import pytest

from tests.test_matched_campaign import fixture
from tools.policy_dataset.matched_dataset import build_sample
from tools.policy_dataset.state_expansion import (
    SCENARIOS, coverage_report, expand_sample, load_scenario, validate_scenario, validate_v3,
)


def source_fixture():
    entries, condition = fixture()
    return build_sample(entries, condition, "synthetic")


def test_deterministic_state_schema_and_source_preservation():
    source = source_fixture()
    before = deepcopy(source)
    for name in SCENARIOS:
        scenario = load_scenario(name)
        sample = expand_sample(source, scenario)
        validate_v3(sample, source, scenario)
        assert sample["measurement_provenance"]["simulated"] is True
        assert [a["action"] for a in sample["action_outcomes"]] == [0, 1, 2, 3]
        assert sample["reward_configuration"]["weights"] == source["condition"]["reward"]["weights"]
    assert source == before


def test_unavailable_device_is_inference_service_not_transport():
    source = source_fixture()
    scenario = load_scenario(SCENARIOS[2])
    sample = expand_sample(source, scenario)
    assert sample["optimal_action"] == 3
    assert sample["state"]["uncertainty"]["confidence"] is None
    assert sample["state"]["device"]["inference_latency_estimate"] is None
    for action in sample["action_outcomes"][:3]:
        assert action["status"] == "infeasible"
        assert action["measurements"] is action["outcomes"] is action["mean_reward"] is None
    scenario["transport_available"] = False
    sample = expand_sample(source, scenario)
    assert sample["optimal_action"] is None
    assert sample["label_status"] == "no_feasible_action"
    assert sample["state"]["network"]["network_quality_score"] == 0


def test_hypothesis_does_not_select_an_action():
    source = source_fixture()
    scenario = load_scenario(SCENARIOS[0])
    first = expand_sample(source, scenario)
    scenario["expected_actions_hypothesis"] = [1, 2]
    second = expand_sample(source, scenario)
    assert first["optimal_actions"] == second["optimal_actions"]
    assert first["action_outcomes"] == second["action_outcomes"]


def test_cloud_unavailability_excludes_all_remote_actions():
    scenario = load_scenario(SCENARIOS[0])
    scenario["cloud_availability"] = False
    sample = expand_sample(source_fixture(), scenario)
    assert sample["optimal_action"] == 0
    assert all(not a["feasible"] for a in sample["action_outcomes"][1:])


@pytest.mark.parametrize("field,value", [("simulated", False), ("local_compute_pressure", 2),
    ("inference_queue_pressure", float("nan")), ("cloud_availability", 1)])
def test_invalid_scenario_rejected(field, value):
    scenario = load_scenario(SCENARIOS[0])
    scenario[field] = value
    with pytest.raises(ValueError): validate_scenario(scenario)


def test_tampering_rejected_and_coverage_does_not_count_infeasible_splits():
    source = source_fixture()
    scenario = load_scenario(SCENARIOS[2])
    sample = expand_sample(source, scenario)
    report = coverage_report([sample])
    assert report["split_coverage"]["1"]["feasible_states"] == 0
    assert report["split_coverage"]["1"]["non_dominated_states"] == 0
    assert not report["minimum_gate_passed"]
    sample["action_outcomes"][0]["mean_reward"] = 1
    with pytest.raises(ValueError): validate_v3(sample, source, scenario)


def test_queue_is_charged_equally_to_every_local_inference_action():
    source = source_fixture()
    scenario = load_scenario(SCENARIOS[1])
    queued = expand_sample(source, scenario)
    wait = scenario["inference_queue_pressure"] * scenario["queue_reference_ms"]
    scenario["inference_queue_pressure"] = 0
    unqueued = expand_sample(source, scenario)
    for action in range(4):
        a = queued["action_outcomes"][action]["measurements"][0]
        b = unqueued["action_outcomes"][action]["measurements"][0]
        assert a["latency_ms"] - b["latency_ms"] == pytest.approx(wait if action < 3 else 0)
        assert a["energy"]["value"] == b["energy"]["value"]


def test_cloud_only_omits_unavailable_state_compute_cost():
    from tools.policy_dataset.run_controlled_campaign import simulate_sample
    source = source_fixture()
    scenario = load_scenario(SCENARIOS[2])
    baseline = simulate_sample(source, scenario["network_profile"])
    sample = expand_sample(source, scenario)
    state_ms = source["source_entries"][0]["trace"]["state_us"] / 1000
    actual = sample["action_outcomes"][3]["measurements"][0]
    original = baseline["candidate_actions"][3]["measurements"][0]
    assert actual["latency_ms"] == pytest.approx(original["latency_ms"] - state_ms)
    assert actual["energy"]["value"] == pytest.approx(original["energy"]["value"] - state_ms / 10)


def test_persisted_v3_campaign_reconstructs_with_coverage():
    import json
    from pathlib import Path
    from tools.policy_dataset.run_state_expansion_campaign import validate_output
    root = Path("data/policy/policy_training_dataset_v3")
    report = validate_output(root)
    assert report == json.loads((root / "state_coverage_report.json").read_text())
    assert report["samples"] == 72
    assert report["state_statistics"]["state.uncertainty.confidence"]["missing"] == 24
    assert not report["training_allowed"]
