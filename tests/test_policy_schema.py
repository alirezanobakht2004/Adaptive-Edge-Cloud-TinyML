import pytest

from tools.policy_dataset.schema import example_record, validate_record, ACTION_NAMES


def test_action_contract_and_example():
    assert ACTION_NAMES == {0: "ALL_LOCAL", 1: "SPLIT1", 2: "SPLIT2", 3: "ALL_CLOUD"}
    validate_record(example_record())


@pytest.mark.parametrize("action", [True, 4, -1, "1", 1.0])
def test_invalid_action(action):
    record = example_record()
    record["action"] = action
    with pytest.raises(ValueError):
        validate_record(record)


@pytest.mark.parametrize("section,key,value", [
    ("metadata", "feature_version", "features-v2"),
    ("metadata", "timestamp", "2026-01-01"),
    ("reward", "energy_kind", "measured"),
    ("measurements", "total_latency_ms", float("nan")),
    ("measurements", "request_bytes", -1),
])
def test_invalid_contract(section, key, value):
    record = example_record()
    record[section][key] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_missing_and_unknown_fields():
    record = example_record()
    del record["state"]["uncertainty"]["margin"]
    with pytest.raises(ValueError):
        validate_record(record)
    record = example_record()
    record["reward"]["measured_energy_j"] = 3
    with pytest.raises(ValueError):
        validate_record(record)
