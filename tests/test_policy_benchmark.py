import pytest

from ml.policy.benchmark import strategies, RuleBasedAdaptive, run_strategy, summarize, placeholder_reward
from tools.policy_dataset.schema import example_record
from tools.policy_dataset.instrumentation import PolicySample


def test_baselines_and_missing_all_cloud_executor():
    record = example_record()
    assert [s.select(record["state"]).selected_action for s in strategies()] == [0, 3, 1, 2, 0]
    with pytest.raises(NotImplementedError, match="ALL_CLOUD"):
        run_strategy(strategies()[1], record, {0: lambda r: None})


def test_rule_is_deterministic_and_handles_network():
    state = example_record()["state"]
    state["uncertainty"]["confidence"] = .5
    state["network"].update(connection_state="connected", mqtt_rtt_ms=10)
    rule = RuleBasedAdaptive()
    assert rule.select(state).selected_action == 1
    state["network"]["mqtt_rtt_ms"] = 100
    assert rule.select(state).selected_action == 2
    state["network"]["connection_state"] = "disconnected"
    assert rule.select(state).selected_action == 0


def test_empty_framework_does_not_invent_results():
    result = summarize([example_record()], scope="device")
    assert all(row["accuracy"] is None and row["mean_latency_ms"] is None for row in result.values())


def test_measured_summary_and_placeholder_reward():
    sample = PolicySample.create(sample_id="s", device_id="host", window_id="w", run_id="r", session_id="s")
    sample.local(lambda x: x, lambda x: [1., 0., 0., 0., 0.], None)
    sample.record["outcome"]["true_class"] = 0
    result = summarize([sample.record], scope="host")[0]
    assert result["labeled_count"] == 1 and result["accuracy"] == 1
    assert result["mean_estimated_energy_proxy"] is None
    reward = placeholder_reward(sample.record, latency_scale_ms=100, communication_scale_bytes=1000,
                                estimated_energy_score=.2, energy_proxy_version="test-assumption-v1")
    assert reward["reward"]["energy_kind"] == "estimated"
    assert reward["reward"]["final_reward"] < 0
    assert sample.record["reward"]["final_reward"] is None
