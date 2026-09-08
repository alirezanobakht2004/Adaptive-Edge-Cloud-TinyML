from copy import deepcopy

import pytest

from tests.test_matched_campaign import fixture
from tools.policy_dataset.matched_dataset import build_sample
from tools.network_conditions.profiles import NAMES, load_profile
from tools.policy_dataset.run_controlled_campaign import simulate_sample, validate_controlled, summarize


def test_reproducible_controlled_rewards_preserve_source():
    entries, condition = fixture()
    source = build_sample(entries, condition, "synthetic")
    original = deepcopy(source)
    rows = []
    for name in NAMES:
        profile = load_profile(name)
        row = simulate_sample(source, profile)
        validate_controlled(row, source, profile)
        assert row["network_condition"]["simulated"] is True
        assert [a["action"] for a in row["candidate_actions"]] == [0, 1, 2, 3]
        assert row["reward"]["configuration"]["weights"] == condition["reward"]["weights"]
        if name == "baseline":
            for a in range(4):
                assert row["candidate_actions"][a]["mean_reward"] == pytest.approx(source["candidate_actions"][a]["mean_reward"])
        rows.append(row)
    assert source == original
    report = summarize(rows)
    assert report["unique_source_samples"] == 1
    assert report["real_new_measurements"] == 0
    assert not report["diversity_gate"]["passed"]


@pytest.mark.parametrize("alter", ["flag", "label", "reward", "state", "coverage"])
def test_reject_fabricated_or_tampered_controlled_records(alter):
    entries, condition = fixture()
    source = build_sample(entries, condition, "synthetic")
    profile = load_profile("baseline")
    row = simulate_sample(source, profile)
    if alter == "flag": row["network_condition"]["simulated"] = False
    if alter == "label": row["optimal_action"] = False
    if alter == "reward": row["candidate_actions"][0]["mean_reward"] += 1
    if alter == "state": row["state"]["network"]["mqtt_rtt_ms"] += 1
    if alter == "coverage": row["candidate_actions"].pop()
    with pytest.raises(ValueError): validate_controlled(row, source, profile)


def test_persisted_profiles_and_diversity_report():
    from pathlib import Path
    import json
    from tools.policy_dataset.run_controlled_campaign import validate_output, DEFAULT_SOURCE, read_rows
    root = Path("data/policy/policy_training_dataset_v2/controlled_network_v1")
    rows = []
    for name in NAMES:
        report = validate_output(root / name, DEFAULT_SOURCE)
        assert report == json.loads((root / name / "dataset_report.json").read_text())
        rows.extend(read_rows(root / name / "policy_training_dataset_v2.jsonl"))
    expected = summarize(rows)
    expected["by_profile"] = {name: summarize([r for r in rows if r["network_condition"]["profile"] == name]) for name in NAMES}
    assert expected == json.loads((root / "diversity_report.json").read_text())
    assert expected["training_allowed"] is False


def test_runner_validates_count_and_never_overwrites(tmp_path):
    from argparse import Namespace
    from tools.policy_dataset.run_controlled_campaign import run, DEFAULT_SOURCE
    args = Namespace(profile="baseline", samples=0, output=tmp_path / "invalid", source=DEFAULT_SOURCE)
    with pytest.raises(ValueError): run(args)
    assert not args.output.exists()
    args.samples = 1
    args.output = tmp_path
    with pytest.raises(FileExistsError): run(args)
