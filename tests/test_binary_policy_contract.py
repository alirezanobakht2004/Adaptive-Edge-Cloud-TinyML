"""R1 contracts must not reinterpret historical action numbers or rewards."""

from copy import deepcopy
import json

import pytest

from ml.policy.binary_contract import (
    ACTIONS, ROOT, compare_candidates, feature_vector, load_contract, load_reward,
    read_json, score_measurement,
)
from tools.policy_dataset.reward_engine.engine import digest


@pytest.fixture
def source():
    path = ROOT / "data/policy/policy_training_dataset_v2/policy_training_dataset_v2.jsonl"
    return json.loads(path.read_text().splitlines()[0])


def test_binary_action_mapping_preserves_historical_config():
    config = load_contract()
    assert config["actions"] == ACTIONS == {"0": "LOCAL", "1": "CLOUD"}
    assert read_json(ROOT / "ml/policy/policy_config_v1.json")["actions"]["1"] == "SPLIT1"
    assert config["learned_model_version"] is None


@pytest.mark.parametrize("change", ["action", "order", "runtime", "divisor"])
def test_reject_contract_drift(tmp_path, change):
    config = load_contract()
    if change == "action":
        config["actions"]["1"] = "SPLIT1"
    elif change == "order":
        config["state_feature_names"].reverse()
    elif change == "runtime":
        config["features"][0]["available_before_decision"] = False
    else:
        config["features"][0]["normalization"]["divisor"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_contract(path)


def test_frozen_reward_and_runtime_features(source):
    config = load_contract()
    original = read_json(ROOT / "tools/policy_dataset/conditions/connected_lab_v1.json")["reward"]
    assert load_reward(config) == original
    assert digest(original) == config["reward_reference"]["configuration_sha256"]
    assert len(feature_vector(source["state"], config)) == 6
    assert feature_vector(source["state"], config)[3] == source["state"]["device"]["free_heap"] / 1024


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, -1])
def test_no_missing_or_fake_features(source, value):
    source["state"]["network"]["mqtt_rtt_ms"] = value
    with pytest.raises(ValueError):
        feature_vector(source["state"], load_contract())


def test_reward_determinism_and_m29_component_parity(source):
    reward = load_reward(load_contract())
    for candidate in source["candidate_actions"]:
        for trial in candidate["measurements"]:
            scored = score_measurement(trial, reward)
            assert scored == score_measurement(deepcopy(trial), reward)
            assert scored["reward"] == pytest.approx(trial["reward"], abs=1e-14)
    candidates, label = compare_candidates(source["candidate_actions"], reward)
    assert [c["historical_action"] for c in candidates] == [0, 3]
    source["label"]["optimal_action"] = 2  # The old four-action argmax is not an input.
    assert compare_candidates(source["candidate_actions"], reward)[1] == label


def test_unknown_energy_and_missing_candidate_rejected(source):
    reward = load_reward(load_contract())
    with pytest.raises(ValueError, match="four"):
        compare_candidates(source["candidate_actions"][:2], reward)
    source["candidate_actions"][0]["measurements"][0]["energy"]["kind"] = "measured"
    with pytest.raises(ValueError, match="Energy"):
        compare_candidates(source["candidate_actions"], reward)


def test_tie_has_no_manufactured_label(source):
    outcomes = source["candidate_actions"]
    outcomes[3]["measurements"] = deepcopy(outcomes[0]["measurements"])
    for trial in outcomes[3]["measurements"]:
        trial["action"] = 3
    _, label = compare_candidates(outcomes, load_reward(load_contract()))
    assert label["optimal_action"] is None
    assert label["optimal_actions"] == [0, 1]


def test_strict_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"action":0,"action":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)
