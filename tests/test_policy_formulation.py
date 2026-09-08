from copy import deepcopy
import pytest
from ml.policy.formulation import load_config, validate_config, missing_state_features
from tools.policy_dataset.schema import ACTION_NAMES
from tools.policy_dataset.validator import read_records
from tools.policy_dataset.analyze_policy_dataset import DATASET, analyze_records, describe, histogram


def test_config_matches_schema_and_current_feature_availability():
    config = load_config()
    assert config["actions"] == {str(k): value for k, value in ACTION_NAMES.items()}
    records = read_records(DATASET / "observations.jsonl")
    assert records
    assert all(not missing_state_features(record, config) for record in records)
    missing = deepcopy(records[0])
    missing["state"]["network"]["mqtt_rtt_ms"] = None
    assert missing_state_features(missing, config) == ["state.network.mqtt_rtt_ms"]


@pytest.mark.parametrize("change", ["split3_alias", "post_action_leakage", "wrong_version", "reward_weight", "training", "encoding"])
def test_reject_incompatible_configuration(change):
    config = load_config()
    if change == "split3_alias":
        config["actions"]["3"] = "SPLIT3"
    elif change == "post_action_leakage":
        config["state_feature_names"][0] = "outcome.confidence"
    elif change == "wrong_version":
        config["dataset_version"] = "policy_training_dataset_v2"
    elif change == "reward_weight":
        config["reward"]["weights"]["accuracy"] = .5
    elif change == "encoding":
        config["encoding_specification"]["encoded_dimension"] = 10
    else:
        config["training_enabled"] = True
    with pytest.raises(ValueError):
        validate_config(config)


def test_analysis_distinguishes_windows_actions_and_states():
    records = read_records(DATASET / "observations.jsonl")
    report = analyze_records(records, load_config())
    assert report["sample_count"] == 100
    assert report["action_distribution"] == dict.fromkeys(ACTION_NAMES.values(), 25)
    assert report["coverage"]["unique_source_windows"] == 25
    assert report["coverage"]["source_windows_with_all_four_actions"] == 25
    assert report["coverage"]["runs_with_multiple_actions"] == 0
    assert not report["training_ready"]
    assert report["missing_values"]["state.network.estimated_bandwidth"] == 100
    assert report["coverage"]["energy_proxy_values_present"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_records(records + records[:1], load_config())


def test_missing_values_are_not_zero_and_histogram_keeps_endpoint():
    stats = describe([None, 2, 4])
    assert stats["mean"] == 3
    assert stats["missing"] == 1
    assert describe([None])["present"] == 0
    histogram_result = histogram([0, .5, 1, None], [0, .5, 1])
    assert histogram_result["counts"] == [1, 2]
    assert histogram_result["missing"] == 1
