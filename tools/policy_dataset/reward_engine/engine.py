"""Validate matched candidate observations and derive separately versioned targets."""

from copy import deepcopy
import hashlib
import json
import math

from ml.policy.formulation import load_config, missing_state_features
from tools.policy_dataset.schema import ACTION_NAMES, VERSION, validate_record


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def number(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} requires an explicit finite {'positive' if positive else 'nonnegative'} number")
    return value


def validate_config(config):
    expected = {"version", "dataset_version", "actions", "weights", "scales",
                "energy_kind", "energy_proxy_version", "energy_method", "tie_tolerance"}
    if set(config) != expected or config["version"] != "reward-calibration-v1" or config["dataset_version"] != VERSION:
        raise ValueError("Invalid calibration schema/version")
    if config["actions"] != {str(k): v for k, v in ACTION_NAMES.items()}:
        raise ValueError("Invalid action mapping; action 3 is ALL_CLOUD")
    for section, keys in (("weights", {"accuracy", "latency", "communication", "energy_proxy"}),
                          ("scales", {"latency_ms", "communication_bytes", "energy_proxy"})):
        if set(config[section]) != keys:
            raise ValueError(f"Invalid {section}")
        for key, value in config[section].items():
            number(value, key, positive=section == "scales")
    if not any(config["weights"].values()):
        raise ValueError("At least one reward weight must be positive")
    number(config["tie_tolerance"], "tie_tolerance")
    if config["energy_kind"] not in ("estimated", "simulated"):
        raise ValueError("Energy is estimated/simulated only")
    for key in ("energy_proxy_version", "energy_method"):
        if not isinstance(config[key], str) or not config[key].strip() or config[key] == "unconfigured":
            raise ValueError(f"Configure {key} explicitly")


def context(record):
    """Identity and complete pre-decision snapshot, excluding action outcomes."""
    return {"state": record["state"], "metadata": {k: record["metadata"][k] for k in
            ("dataset_version", "model_version", "feature_version", "source_dataset_version",
             "device_id", "session_id", "measurement_scope")}}


def evaluate_group(group, config):
    """Input: reference v1 decision plus four {record, energy} candidate wrappers.

    Evidence is an explicit collection-protocol reference, not proof of physical
    equivalence. Equality checks catch mismatched logged states; the experimenter
    must control/repeat physical conditions and audit that reference.
    """
    validate_config(config)
    if set(group) != {"version", "decision_id", "matching_evidence", "decision", "candidates"} or group["version"] != "candidate-actions-v1":
        raise ValueError("Invalid candidate group schema")
    for key in ("decision_id", "matching_evidence"):
        if not isinstance(group[key], str) or not group[key].strip():
            raise ValueError(f"Missing {key}")
    decision = group["decision"]
    validate_record(decision)
    if missing_state_features(decision, load_config()):
        raise ValueError("Decision lacks required M25 state features")
    candidates = group["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("Exactly four candidate actions are required")
    results, seen, labels = [], set(), set()
    for candidate in candidates:
        if set(candidate) != {"record", "energy"}:
            raise ValueError("Candidate requires record and energy")
        record, energy = candidate["record"], candidate["energy"]
        validate_record(record)
        action = record["action"]
        if action in seen:
            raise ValueError("Duplicate action")
        seen.add(action)
        if record["metadata"]["record_kind"] != "observation" or context(record) != context(decision):
            raise ValueError("Candidates require observations from the same decision context")
        outcome = record["outcome"]
        if outcome["status"] == "not_executed" or outcome["true_class"] is None:
            raise ValueError("Executed candidates with ground truth are required")
        labels.add(outcome["true_class"])
        measurements = record["measurements"]
        latency = number(measurements["total_latency_ms"], "total_latency_ms")
        sent = number(measurements["request_bytes"], "request_bytes")
        received = number(measurements["response_bytes"], "response_bytes")
        if set(energy) != {"value", "kind", "proxy_version", "method"}:
            raise ValueError("Energy needs value, kind, proxy_version and method")
        if (energy["kind"] != config["energy_kind"] or energy["proxy_version"] != config["energy_proxy_version"]
                or energy["method"] != config["energy_method"]):
            raise ValueError("Energy assumptions must match calibration")
        proxy = number(energy["value"], "estimated energy proxy")
        correct = int(outcome["status"] == "success" and outcome["predicted_class"] == outcome["true_class"])
        scales, weights = config["scales"], config["weights"]
        normalized = {"accuracy": correct, "latency": latency / scales["latency_ms"],
                      "communication": (sent + received) / scales["communication_bytes"],
                      "energy_proxy": proxy / scales["energy_proxy"]}
        reward = weights["accuracy"] * correct - sum(weights[k] * normalized[k] for k in
                                                     ("latency", "communication", "energy_proxy"))
        if not all(math.isfinite(v) for v in (*normalized.values(), reward)):
            raise ValueError("Reward normalization overflow")
        results.append({"action": action, "prediction_correct": bool(correct),
                        "prediction": outcome["predicted_class"], "status": outcome["status"],
                        "latency_ms": latency, "communication_bytes": sent + received,
                        "energy": deepcopy(energy), "normalized": normalized, "reward": reward,
                        "record_sha256": digest(record)})
    if len(labels) != 1:
        raise ValueError("Candidate ground truth labels differ")
    if decision["outcome"]["true_class"] not in (None, next(iter(labels))):
        raise ValueError("Decision ground truth differs")
    best = max(row["reward"] for row in results)
    ties = sorted(row["action"] for row in results if best - row["reward"] <= config["tie_tolerance"])
    return {"version": "candidate-rewards-v1", "dataset_version": VERSION,
            "decision_id": group["decision_id"], "context_sha256": digest(context(decision)),
            "input_sha256": digest(group), "calibration_sha256": digest(config),
            "calibration": deepcopy(config), "matching_evidence": group["matching_evidence"],
            "candidates": sorted(results, key=lambda row: row["action"]),
            "optimal_action": ties[0] if len(ties) == 1 else None, "optimal_actions": ties,
            "label_scope": "observed-candidate-argmax-not-expected-optimum", "training_enabled": False}


def collect_candidates(decision, *, decision_id, matching_evidence, executors, energy_estimator, config):
    """Adapters mutate independent v1 copies; estimator returns an explicit energy wrapper.

    Executors must perform the named action (including a true full-cloud action 3).
    No default transport or energy coefficients are supplied. Hardware state is
    not reset by this function; matched-condition collection is the adapter's job.
    """
    validate_config(config)
    validate_record(decision)
    if set(executors) != set(ACTION_NAMES) or any(type(k) is not int for k in executors):
        raise ValueError("Provide four explicit action executors")
    if missing_state_features(decision, load_config()):
        raise ValueError("Decision lacks required M25 state features")
    group = {"version": "candidate-actions-v1", "decision_id": decision_id,
             "matching_evidence": matching_evidence, "decision": deepcopy(decision), "candidates": []}
    for action in ACTION_NAMES:
        record = deepcopy(decision)
        record["action"] = action
        executors[action](record)
        validate_record(record)
        if record["action"] != action or context(record) != context(decision):
            raise ValueError("Executor changed action or pre-decision context")
        energy = energy_estimator(deepcopy(record))
        group["candidates"].append({"record": record, "energy": energy})
    return group, evaluate_group(group, config)
