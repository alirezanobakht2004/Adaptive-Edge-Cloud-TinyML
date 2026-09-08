"""R1 incremental action accounting. Never execute LOCAL a second time."""

from copy import deepcopy
from statistics import mean

from .binary_contract import ROOT, read_json
from tools.policy_dataset.reward_engine.engine import digest, number
from tools.policy_dataset.validate_matched_dataset import validate_campaign, read_jsonl

REWARD_PATH = ROOT / "ml/policy/reward_binary_experiment_v2_r1_reuse.json"
SOURCE = ROOT / "data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02"


def load_reward():
    config = read_json(REWARD_PATH)
    parent = read_json(ROOT / "ml/policy/reward_binary_experiment_v1.json")["configuration"]
    if (config["version"] != "binary-reward-experiment-v2-r1-reuse"
            or config["actions"] != {"0": "LOCAL", "1": "CLOUD"}
            or config["parent_configuration_sha256"] != digest(parent)
            or any(config[k] != parent[k] for k in ("weights", "scales", "tie_tolerance"))
            or config["local"]["second_inference_count"] != 0):
        raise ValueError("R1 reuse reward contract mismatch")
    return config


def cached_local(prediction, true_class):
    if type(prediction) is not int or prediction not in range(5) or true_class not in range(5):
        raise ValueError("Known gesture prediction/label required")
    return {"action": 0, "name": "LOCAL", "prediction": prediction,
            "correct": prediction == true_class, "second_inference_count": 0,
            "latency_ms": 0, "communication_bytes": 0, "energy_proxy": 0,
            "cost_provenance": "incremental-accounting-zero-not-physical-timing"}


def cloud_candidate(prediction, true_class, latency_ms, communication_bytes, provenance):
    number(latency_ms, "cloud incremental latency")
    number(communication_bytes, "payload bytes")
    if type(prediction) is not int or prediction not in range(5) or true_class not in range(5):
        raise ValueError("Known cloud prediction/label required")
    if provenance not in ("measured", "simulated"):
        raise ValueError("Explicit cloud cost provenance required")
    return {"action": 1, "name": "CLOUD", "prediction": prediction,
            "correct": prediction == true_class, "latency_ms": latency_ms,
            "communication_bytes": communication_bytes,
            "energy_proxy": communication_bytes / 1024,
            "cost_provenance": provenance, "energy_kind": "estimated" if provenance == "measured" else "simulated",
            "energy_proxy_version": "r1-incremental-payload-proxy-v1"}


def reward(candidate, config=None):
    config = load_reward() if config is None else config
    if candidate["action"] == 0 and (candidate["second_inference_count"] != 0
            or any(candidate[k] != 0 for k in ("latency_ms", "communication_bytes", "energy_proxy"))):
        raise ValueError("LOCAL must reuse the result without duplicate inference cost")
    if type(candidate["correct"]) is not bool:
        raise ValueError("Correctness must be observed, not a confidence estimate")
    weights, scales = config["weights"], config["scales"]
    components = {"accuracy": weights["accuracy"] * int(candidate["correct"]),
                  "latency": weights["latency"] * number(candidate["latency_ms"], "latency") / scales["latency_ms"],
                  "communication": weights["communication"] * number(candidate["communication_bytes"], "bytes") / scales["communication_bytes"],
                  "energy_proxy": weights["energy_proxy"] * number(candidate["energy_proxy"], "proxy") / scales["energy_proxy"]}
    return {"value": components["accuracy"] - sum(components[k] for k in ("latency", "communication", "energy_proxy")),
            "components": components}


def label_candidates(local, cloud, config=None):
    config = load_reward() if config is None else config
    if [local["action"], cloud["action"]] != [0, 1]:
        raise ValueError("Binary candidate order must be LOCAL, CLOUD")
    scores = [reward(c, config) for c in (local, cloud)]
    maximum = max(s["value"] for s in scores)
    ties = [i for i, s in enumerate(scores) if maximum - s["value"] <= config["tie_tolerance"]]
    return {"rewards": scores, "optimal_actions": ties, "optimal_action": ties[0] if len(ties) == 1 else None}


def salvage_m30():
    """Read raw validated evidence, use state_class, discard redundant action-0 timing."""
    validate_campaign(SOURCE)
    samples = read_jsonl(SOURCE / "policy_training_dataset_v2.jsonl")
    rows = []
    for sample in samples:
        entries = [e for e in sample["source_entries"] if e["trace"]["action"] == 3]
        trace = entries[0]["trace"]
        true_class = entries[0]["record"]["outcome"]["true_class"]
        local = cached_local(trace["state_class"], true_class)
        clouds = [cloud_candidate(e["trace"]["prediction"], true_class, e["trace"]["action_us"] / 1000,
                                 e["trace"]["request_bytes"] + e["trace"]["response_bytes"], "measured") for e in entries]
        # Prediction must be consistent before aggregating measured repeats.
        if len({c["prediction"] for c in clouds}) != 1:
            raise ValueError("Variable cloud predictions require repeat-level targets")
        cloud = cloud_candidate(clouds[0]["prediction"], true_class, mean(c["latency_ms"] for c in clouds),
                                mean(c["communication_bytes"] for c in clouds), "measured")
        rows.append({"source_id": sample["sample_id"], "source_sha256": digest(sample),
                     "source_window": sample["state"]["application"]["window_id"],
                     "metadata": deepcopy(sample["metadata"]), "state": deepcopy(sample["state"]),
                     "true_class": true_class, "rssi_dbm": trace["rssi_dbm"],
                     "raw_cloud_traces": [deepcopy(e["trace"]) for e in entries],
                     "shared_state_us": trace["state_us"], "local": local, "cloud": cloud,
                     "label": label_candidates(local, cloud), "provenance": "measured outcomes and cloud timing; LOCAL accounting zero"})
    return rows
