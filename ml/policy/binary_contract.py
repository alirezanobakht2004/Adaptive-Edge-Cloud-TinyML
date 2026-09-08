"""Architecture R1 contracts; historical action enums/configurations are immutable."""

from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean

from tools.policy_dataset.reward_engine.engine import digest, number, validate_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ml/policy/policy_config_v2.json"
VERSION = "policy_training_dataset_binary_v1"
ACTIONS = {"0": "LOCAL", "1": "CLOUD"}
FEATURES = (
    "state.uncertainty.confidence", "state.uncertainty.entropy",
    "state.uncertainty.margin", "state.device.free_heap",
    "state.device.inference_latency_estimate", "state.network.mqtt_rtt_ms",
)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject, object_pairs_hook=unique)


def load_contract(path=CONFIG):
    config = read_json(path)
    if (config["version"] != "policy-config-v2" or config["architecture_revision"] != "R1"
            or config["actions"] != ACTIONS or config["dataset_version"] != VERSION
            or tuple(config["state_feature_names"]) != FEATURES):
        raise ValueError("R1 binary version/action/feature ordering mismatch")
    if ([f["name"] for f in config["features"]] != list(FEATURES)
            or config["normalization"]["version"] != "binary-state-normalization-v1"
            or config["normalization"]["dimension"] != len(FEATURES)):
        raise ValueError("Normalization contract mismatch")
    for feature in config["features"]:
        if (feature["available_before_decision"] is not True or not feature["runtime_source"]
                or not (ROOT / feature["runtime_evidence"]).is_file()):
            raise ValueError("Missing pre-decision runtime source")
        number(feature["normalization"]["divisor"], "normalization divisor", positive=True)
        number(feature["normalization"]["offset"], "normalization offset")
    load_reward(config)
    return config


def load_reward(config):
    reference = config["reward_reference"]
    frozen = read_json(ROOT / reference["path"])
    reward = frozen["configuration"]
    validate_config(reward)  # This copy deliberately retains historical action names.
    source = read_json(ROOT / frozen["source_path"])["reward"]
    if (frozen["version"] != reference["version"] or reward != source
            or digest(reward) != reference["configuration_sha256"]
            or digest(reward) != frozen["source_reward_sha256"]):
        raise ValueError("Frozen reward differs from exact M30 calibration")
    return reward


def feature_vector(state, config):
    values = []
    for feature in config["features"]:
        value = {"state": state}
        try:
            for part in feature["name"].split("."):
                value = value[part]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing feature {feature['name']}") from exc
        number(value, feature["name"])
        if feature["name"].endswith(("confidence", "margin")) and value > 1:
            raise ValueError("Invalid probability/margin")
        if feature["name"].endswith("entropy") and value > math.log(5) + 1e-6:
            raise ValueError("Invalid five-class entropy")
        norm = feature["normalization"]
        normalized = (value - norm["offset"]) / norm["divisor"]
        if not math.isfinite(normalized):
            raise ValueError("Normalization overflow")
        values.append(normalized)
    return values


def score_measurement(trial, reward):
    """Recompute the M29 formula from outcomes, never from the old optimal label."""
    if type(trial["prediction_correct"]) is not bool or trial["status"] != "success":
        raise ValueError("Requires a known successful candidate outcome")
    if type(trial["prediction"]) is not int or trial["prediction"] not in range(5):
        raise ValueError("Invalid candidate prediction")
    energy = trial["energy"]
    if energy["kind"] not in ("estimated", "simulated") or not energy["proxy_version"] or not energy["method"]:
        raise ValueError("Energy must have explicit estimated/simulated provenance")
    scales, weights = reward["scales"], reward["weights"]
    normalized = {
        "accuracy": int(trial["prediction_correct"]),
        "latency": number(trial["latency_ms"], "latency") / scales["latency_ms"],
        "communication": number(trial["communication_bytes"], "bytes") / scales["communication_bytes"],
        "energy_proxy": number(energy["value"], "energy proxy") / scales["energy_proxy"],
    }
    components = {key: weights[key] * value for key, value in normalized.items()}
    score = components["accuracy"] - sum(components[k] for k in ("latency", "communication", "energy_proxy"))
    if not math.isfinite(score):
        raise ValueError("Reward overflow")
    return {"reward": score, "components": components}


def compare_candidates(outcomes, reward):
    if not isinstance(outcomes, list) or len(outcomes) != 4:
        raise ValueError("Preserve all four original candidate outcomes")
    if any(type(o.get("action")) is not int for o in outcomes) or {o["action"] for o in outcomes} != {0, 1, 2, 3}:
        raise ValueError("Historical candidate mapping mismatch")
    by_action = {o["action"]: o for o in outcomes}
    candidates = []
    for binary_action, original_action in ((0, 0), (1, 3)):
        trials = by_action[original_action].get("measurements")
        if not trials or by_action[original_action].get("feasible") is False:
            raise ValueError(f"Missing feasible historical action {original_action} measurements")
        if any(t["action"] != original_action for t in trials):
            raise ValueError("Trial action mismatch")
        scored = [score_measurement(t, reward) for t in trials]
        candidates.append({"action": binary_action, "name": ACTIONS[str(binary_action)],
                           "historical_action": original_action, "repeats": len(trials),
                           "mean_reward": mean(t["reward"] for t in scored),
                           "reward_components": {k: mean(t["components"][k] for t in scored)
                                                 for k in reward["weights"]},
                           "trial_rewards": [t["reward"] for t in scored]})
    if candidates[0]["repeats"] != candidates[1]["repeats"]:
        raise ValueError("Unmatched candidate repeat counts")
    best = max(c["mean_reward"] for c in candidates)
    ties = [c["action"] for c in candidates if best - c["mean_reward"] <= reward["tie_tolerance"]]
    return candidates, {"optimal_action": ties[0] if len(ties) == 1 else None,
                        "optimal_actions": ties, "method": "recomputed-binary-mean-reward-argmax",
                        "scope": "historical candidate execution costs; not qualified R1 production targets"}


def validate_record(row, config):
    required = {"dataset_version", "sample_id", "state", "normalized_state", "source", "group_key",
                "partition", "provenance", "original_candidate_outcomes", "candidates", "label",
                "feature_contract_sha256", "reward_configuration_sha256", "production_eligible"}
    if set(row) != required or row["dataset_version"] != VERSION:
        raise ValueError("Binary record schema mismatch")
    if row["feature_contract_sha256"] != digest(config):
        raise ValueError("Feature contract hash mismatch")
    reward = load_reward(config)
    if row["reward_configuration_sha256"] != digest(reward):
        raise ValueError("Reward reference mismatch")
    if row["normalized_state"] != feature_vector(row["state"], config):
        raise ValueError("Feature vector mismatch")
    candidates, label = compare_candidates(row["original_candidate_outcomes"], reward)
    if row["candidates"] != candidates or row["label"] != label:
        raise ValueError("Recomputed binary rewards/label mismatch")
    if row["partition"] not in ("train", "holdout") or row["group_key"] != digest(row["source"]["window_identity"]):
        raise ValueError("Grouping/partition mismatch")
    if (row["provenance"]["kind"] not in ("measured", "simulated")
            or row["production_eligible"] is not False):
        raise ValueError("Historical execution costs are not R1 production-qualified")
    return deepcopy(row)
