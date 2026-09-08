from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.policy_dataset.analyze_split_sensitivity import (
    best_weight_margin, break_even, dominating_actions, normalized_actions, rewards, run, winners,
)
from tests.test_matched_campaign import fixture
from tools.policy_dataset.matched_dataset import build_sample


def action(latency, communication, energy=0, accuracy=1):
    return dict(accuracy=accuracy, latency=latency, communication=communication, energy_proxy=energy)


def test_dominance_and_weight_feasibility():
    dominated = [action(0, 0), action(1, 1), action(2, 2), action(3, 3)]
    assert dominating_actions(dominated, 1) == [0]
    assert not best_weight_margin(dominated, 1)["can_be_unique_optimum"]
    # A balanced split can win when it really offers an intermediate tradeoff.
    frontier = [action(0, 1), action(.4, .4), action(1, 0), action(2, 2)]
    result = best_weight_margin(frontier, 1)
    assert result["can_be_unique_optimum"]
    assert result["maximum_minimum_margin"] == pytest.approx(.1)
    assert winners(rewards(frontier, result["illustrative_weights_not_recommendation"]), 1e-8) == [1]
    # Nondominated does not guarantee a supported optimum of a linear reward.
    frontier[1] = action(.6, .6)
    assert dominating_actions(frontier, 1) == []
    assert not best_weight_margin(frontier, 1)["can_be_unique_optimum"]


def test_all_equal_actions_tie():
    equal = [action(1, 1)] * 4
    weights = dict(accuracy=1, latency=.1, communication=.1, energy_proxy=.1)
    assert winners(rewards(equal, weights), 0) == [0, 1, 2, 3]
    assert not best_weight_margin(equal, 2)["can_be_unique_optimum"]


def test_break_even_reproduces_reward_equality_and_zero_weight():
    actions = [action(1, 0), action(2, 1), action(3, 1), action(4, 1)]
    config = dict(weights=dict(accuracy=1, latency=.1, communication=.1, energy_proxy=.1),
                  scales=dict(latency_ms=100, communication_bytes=1024, energy_proxy=1))
    threshold = break_even(actions, 1, 0, config)
    changed = deepcopy(actions)
    changed[1]["latency"] = threshold["target_latency_ms_to_tie"] / 100
    scores = rewards(changed, config["weights"])
    assert scores[0] == pytest.approx(scores[1])
    assert not threshold["accuracy_only_change_feasible"]
    config["weights"]["latency"] = 0
    assert break_even(actions, 1, 0, config)["target_latency_ms_to_tie"] is None


def test_analysis_preserves_source_and_validates_mapping():
    entries, condition = fixture()
    sample = build_sample(entries, condition, "synthetic")
    before = deepcopy(sample)
    values = normalized_actions(sample)
    assert rewards(values, condition["reward"]["weights"]) == pytest.approx([a["mean_reward"] for a in sample["candidate_actions"]])
    assert sample == before
    sample["candidate_actions"].pop()
    with pytest.raises(ValueError): normalized_actions(sample)
    with pytest.raises(ValueError): rewards(values, dict(accuracy=1, latency=-1, communication=0, energy_proxy=0))


def test_report_reproduces_evidence_and_keeps_simulations_separate(tmp_path):
    generated = run(tmp_path / "analysis.json")
    expected = json.loads(Path("docs/evidence/phase10_3_split_sensitivity.json").read_text())
    assert generated == expected
    assert [c["simulated"] for c in generated["cohorts"]] == [False, True, True, True, True]
    assert generated["independent_source_windows"] == 24
    assert generated["training_allowed"] is False
    for cohort in generated["cohorts"]:
        for row in cohort["actions"]:
            assert sum(row["mean_latency_contributions"].values()) == pytest.approx(row["mean_latency_ms"])
            parts = row["weighted_reward_components"]
            assert parts["accuracy"] - sum(parts[k] for k in ("latency", "communication", "energy_proxy")) == pytest.approx(row["mean_reward"])
