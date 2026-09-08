"""Reconstruct binary data from original traces; guard leakage and action semantics."""

from copy import deepcopy
import json

import pytest

from ml.policy.binary_contract import ROOT, load_contract, validate_record
from ml.policy.binary_policy_dataset import (
    DEFAULT_OUTPUT, derive_one, partition_groups, read_rows, validate_output,
)
from tools.policy_dataset.reward_engine.engine import digest


@pytest.fixture(scope="module")
def rows():
    return read_rows(DEFAULT_OUTPUT / "policy_training_dataset_binary_v1.jsonl")


def test_reconstruct_dataset_and_gate():
    report = validate_output(DEFAULT_OUTPUT)
    assert report["record_count"] == 168
    assert report["unique_underlying_windows"] == 24
    assert report["label_counts"] == {"LOCAL": 96, "CLOUD": 72}
    assert report["provenance_counts"] == {"measured": 24, "simulated": 144}
    assert report["by_provenance"]["measured"] == {"LOCAL": 24}
    assert report["production_action_semantics"]["measured_local_repeat_count"] == 96
    assert report["production_action_semantics"]["extra_local_execution_ms"]["minimum"] > 0
    assert report["gates"]["both_binary_labels"] is True
    assert report["gates"]["candidate_execution_matches_production_action_contract"] is False
    assert report["training_allowed"] is False
    assert report["production_eligible_records"] == 0


def test_group_split_stable_independent_of_order(rows):
    config = load_contract()
    keys = [r["group_key"] for r in rows]
    expected = partition_groups(keys, config)
    assert expected == partition_groups(reversed(keys), config)
    assert expected == partition_groups(keys + keys, config)
    assert all(r["partition"] == expected[r["group_key"]] for r in rows)
    assert sum(v == "holdout" for v in expected.values()) == 6
    assert {r["group_key"] for r in rows if r["partition"] == "holdout"}.isdisjoint(
        r["group_key"] for r in rows if r["partition"] == "train")


def test_same_window_replay_profiles_stay_together(rows):
    for window in {r["source"]["window_identity"]["window_id"] for r in rows}:
        related = [r for r in rows if r["source"]["window_identity"]["window_id"] == window]
        assert len(related) == 7
        assert len({r["group_key"] for r in related}) == 1
        assert len({r["partition"] for r in related}) == 1


@pytest.mark.parametrize("field", ["label", "state", "reward", "action", "version", "group", "eligibility"])
def test_reject_tampered_record(rows, field):
    row = deepcopy(rows[0])
    if field == "label":
        row["label"]["optimal_action"] = 1
    elif field == "state":
        row["state"]["uncertainty"]["entropy"] = None
    elif field == "reward":
        row["candidates"][0]["mean_reward"] += 1
    elif field == "action":
        row["original_candidate_outcomes"][3]["action"] = 1
    elif field == "version":
        row["dataset_version"] = "policy_training_dataset_v1"
    elif field == "group":
        row["group_key"] = "fake-independent-replay"
    else:
        row["production_eligible"] = True
    with pytest.raises(ValueError):
        validate_record(row, load_contract())


def test_exclusion_and_duplicate_audit(rows):
    manifest = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text())
    assert len(manifest["exclusions"]) == 24
    assert all("action 0" in r["reason"] for r in manifest["exclusions"])
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert len({digest(r["normalized_state"]) for r in rows}) == 120
    assert all(r["source"]["reward_configuration"]["actions"]["3"] == "ALL_CLOUD" for r in rows)


def test_derivation_does_not_use_old_optimal_label():
    path = ROOT / "data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02/policy_training_dataset_v2.jsonl"
    source = read_rows(path)[0]
    changed = deepcopy(source)
    changed["label"]["optimal_action"] = 3
    first = derive_one(source, source, path, "measured", "lab", load_contract())
    second = derive_one(changed, changed, path, "measured", "lab", load_contract())
    assert first["label"] == second["label"]
    assert first["source"]["historical_optimal_action"] != second["source"]["historical_optimal_action"]
