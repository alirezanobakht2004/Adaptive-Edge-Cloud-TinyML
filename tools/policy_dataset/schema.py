"""Canonical policy_training_dataset_v1 contract (standard library only)."""

from copy import deepcopy
from datetime import datetime
from enum import IntEnum
import math

VERSION = "policy_training_dataset_v1"
MODEL_VERSION = "gesture-model-v1.1.0"
FEATURE_VERSION = "features-v1"


class Action(IntEnum):
    ALL_LOCAL = 0
    SPLIT1 = 1
    SPLIT2 = 2
    ALL_CLOUD = 3


# Policy actions are deliberately NOT the Phase7 MQTT split enumeration.
ACTION_NAMES = {int(action): action.name for action in Action}
METRIC_NAMES = (
    "preprocessing_latency_ms", "local_inference_latency_ms", "prefix_latency_ms",
    "mqtt_send_timestamp", "server_response_timestamp", "round_trip_latency_ms",
    "embedding_size_bytes", "request_bytes", "response_bytes", "cloud_inference_latency_ms",
    "request_receive_time", "response_publish_time", "total_latency_ms",
    "local_confidence", "local_entropy", "local_margin",
)
TEMPLATE = {
    "metadata": {
        "dataset_version": VERSION, "sample_id": "example-000001",
        "timestamp": "2026-09-08T00:00:00+00:00", "device_id": "example-device",
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "source_dataset_version": "dataset-v1", "record_kind": "example",
        "measurement_scope": "unmeasured", "run_id": "example-run", "session_id": "example-session",
    },
    "state": {
        "uncertainty": {"confidence": None, "entropy": None, "margin": None},
        "device": {"free_heap": None, "tensor_arena_usage": None, "cpu_frequency": None,
                   "inference_latency_estimate": None},
        "network": {"mqtt_rtt_ms": None, "packet_size_estimate": None,
                    "connection_state": "unknown", "estimated_bandwidth": None},
        "application": {"predicted_class": None, "window_id": "example-window"},
    },
    "action": 0,
    "reward": {"latency_component": None, "communication_cost_component": None,
               "energy_component": None, "energy_kind": "estimated",
               "energy_proxy_version": "unconfigured", "final_reward": None,
               "calculation": "placeholder-unconfigured"},
    "measurements": dict.fromkeys(METRIC_NAMES),
    "outcome": {"status": "not_executed", "predicted_class": None, "confidence": None,
                "true_class": None, "error": None},
    "provenance": {"state_source": "unavailable", "artifact_hashes": {},
                   "clock_domains": {}, "notes": "Example only; no measured data."},
}


def example_record() -> dict:
    return deepcopy(TEMPLATE)


def _required_shape(value, template, path="record"):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = template.keys() - value.keys()
    extra = value.keys() - template.keys()
    if missing or extra:
        raise ValueError(f"{path}: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key, child in template.items():
        if isinstance(child, dict) and child:
            _required_shape(value[key], child, f"{path}.{key}")


def _number(value, name, maximum=None, integer=False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid range for {name}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _timestamp(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {name}") from exc
    if result.tzinfo is None:
        raise ValueError(f"{name} needs a timezone")


def validate_record(record: dict) -> None:
    """Reject inconsistent versions, missing fields, invalid numbers and energy claims."""
    _required_shape(record, TEMPLATE)
    meta = record["metadata"]
    for field, expected in (("dataset_version", VERSION), ("model_version", MODEL_VERSION),
                            ("feature_version", FEATURE_VERSION), ("source_dataset_version", "dataset-v1")):
        if meta[field] != expected:
            raise ValueError(f"Version mismatch: {field}")
    for key, value in meta.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.{key} must be a nonempty string")
    _timestamp(meta["timestamp"], "timestamp")
    if meta["record_kind"] not in ("example", "observation"):
        raise ValueError("Invalid record_kind")
    if meta["measurement_scope"] not in ("unmeasured", "host", "device", "replay"):
        raise ValueError("Invalid measurement_scope")
    if type(record["action"]) is not int or record["action"] not in ACTION_NAMES:
        raise ValueError("action must be an integer in {0,1,2,3}")
    state = record["state"]
    for key, maximum in (("confidence", 1), ("entropy", math.log(5) + 1e-9), ("margin", 1)):
        _number(state["uncertainty"][key], key, maximum)
    for key, value in state["device"].items():
        _number(value, key, integer=key in ("free_heap", "tensor_arena_usage"))
    for key in ("mqtt_rtt_ms", "packet_size_estimate", "estimated_bandwidth"):
        _number(state["network"][key], key, integer=key == "packet_size_estimate")
    if state["network"]["connection_state"] not in ("connected", "disconnected", "unknown"):
        raise ValueError("Invalid connection_state")
    if not isinstance(state["application"]["window_id"], str) or not state["application"]["window_id"]:
        raise ValueError("window_id must be nonempty")
    _number(state["application"]["predicted_class"], "predicted_class", 4, integer=True)
    for key, value in record["measurements"].items():
        if key.endswith("timestamp") or key in ("request_receive_time", "response_publish_time"):
            if value is not None:
                _timestamp(value, key)
        else:
            maximum = 1 if key in ("local_confidence", "local_margin") else math.log(5) + 1e-9 if key == "local_entropy" else None
            _number(value, key, maximum, integer=key.endswith("bytes"))
    reward = record["reward"]
    if not isinstance(reward["energy_proxy_version"], str) or not reward["energy_proxy_version"].strip():
        raise ValueError("energy_proxy_version must be nonempty")
    if reward["energy_kind"] not in ("estimated", "simulated"):
        raise ValueError("Energy must be estimated or simulated; measured energy is unsupported")
    for key in ("latency_component", "communication_cost_component", "energy_component"):
        _number(reward[key], key)
    if reward["final_reward"] is not None:
        if any(reward[k] is None for k in ("latency_component", "communication_cost_component", "energy_component")):
            raise ValueError("A placeholder reward requires all cost components")
        expected = -sum(reward[k] for k in ("latency_component", "communication_cost_component", "energy_component"))
        if isinstance(reward["final_reward"], bool) or not isinstance(reward["final_reward"], (int, float)) or not math.isclose(reward["final_reward"], expected):
            raise ValueError("Invalid placeholder reward calculation")
        if reward["calculation"] != "placeholder-negative-sum-v1" or reward["energy_proxy_version"] == "unconfigured":
            raise ValueError("Reward assumptions must be versioned")
    elif reward["calculation"] != "placeholder-unconfigured":
        raise ValueError("Missing configured reward")
    outcome = record["outcome"]
    if outcome["status"] not in ("not_executed", "success", "timeout", "failed"):
        raise ValueError("Invalid outcome status")
    for key in ("predicted_class", "true_class"):
        _number(outcome[key], key, 4, integer=True)
    _number(outcome["confidence"], "outcome confidence", 1)
    if outcome["status"] == "success" and (outcome["predicted_class"] is None or outcome["confidence"] is None):
        raise ValueError("Successful outcome requires a classification")
    if outcome["error"] is not None and not isinstance(outcome["error"], str):
        raise ValueError("outcome error must be text or null")
    if meta["record_kind"] == "observation" and meta["measurement_scope"] == "unmeasured":
        raise ValueError("Observation needs a measurement scope")
    for key in ("artifact_hashes", "clock_domains"):
        if not isinstance(record["provenance"][key], dict):
            raise ValueError(f"provenance.{key} must be an object")
    for key in ("state_source", "notes"):
        if not isinstance(record["provenance"][key], str):
            raise ValueError(f"provenance.{key} must be text")
    for value in record["provenance"]["artifact_hashes"].values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Artifact hashes must be lowercase SHA-256 hex")


def schema_description() -> dict:
    return {"version": VERSION, "actions": ACTION_NAMES, "required_structure": example_record(),
            "null_semantics": "Unknown or unmeasured; never silently replaced by zero",
            "units": {"entropy": "nats", "cpu_frequency": "MHz", "estimated_bandwidth": "bytes/second",
                      "free_heap": "bytes", "tensor_arena_usage": "bytes", "energy_component": "proxy score"},
            "validator": "tools.policy_dataset.schema.validate_record",
            "warning": "Action 3 is ALL_CLOUD, not MQTT split=3. Energy is estimated/simulated only."}
