from copy import deepcopy
import pytest
from ml.policy.r1_reuse import cached_local, cloud_candidate, label_candidates, load_reward, reward


def test_reuse_has_no_second_inference():
    local = cached_local(1, 1)
    assert local["second_inference_count"] == 0
    assert local["communication_bytes"] == local["latency_ms"] == local["energy_proxy"] == 0
    assert reward(local)["value"] == 1


def test_no_cloud_bonus_and_correctness_can_justify_cloud():
    cloud = cloud_candidate(1, 1, 20, 400, "simulated")
    assert label_candidates(cached_local(1, 1), cloud)["optimal_action"] == 0
    assert label_candidates(cached_local(2, 1), cloud)["optimal_action"] == 1
    assert label_candidates(cached_local(2, 1), cloud_candidate(1, 1, 2000, 400, "simulated"))["optimal_action"] == 0


def test_duplicate_local_compute_is_rejected():
    local = cached_local(0, 0)
    local["second_inference_count"] = 1
    with pytest.raises(ValueError):
        reward(local)


def test_incremental_reward_is_deterministic_and_weights_unchanged():
    config = load_reward()
    assert config["weights"] == {"accuracy": 1, "latency": .1, "communication": .1, "energy_proxy": .1}
    cloud = cloud_candidate(1, 1, 23, 512, "measured")
    assert reward(cloud) == reward(deepcopy(cloud))
    assert reward(cloud)["value"] == pytest.approx(.877)
