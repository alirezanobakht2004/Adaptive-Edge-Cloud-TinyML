"""Apply explicit, deterministic simulation assumptions to retained measurements."""

from copy import deepcopy
import json
from pathlib import Path

from tools.policy_dataset.reward_engine.engine import number

NAMES = ("baseline", "high_latency", "edge_loaded", "cloud_favorable")


def validate_profile(profile):
    if (set(profile) != {"version", "type", "profile", "simulated", "application", "injected_parameters"}
            or profile["version"] != "network-profile-v1" or profile["type"] != "controlled_network"
            or profile["profile"] not in NAMES or profile["simulated"] is not True
            or profile["application"] != "offline-cost-overlay"):
        raise ValueError("Invalid simulated network profile")
    parameters = profile["injected_parameters"]
    if set(parameters) != {"edge_compute_multiplier", "network_residual_multiplier", "cloud_compute_multiplier", "added_rtt_ms"}:
        raise ValueError("Invalid injected parameters")
    for key, value in parameters.items():
        number(value, key, positive=key != "added_rtt_ms")


def load_profile(name):
    profiles = json.loads(Path(__file__).with_name("profiles_v1.json").read_text())
    if name not in profiles:
        raise ValueError(f"Unknown profile: {name}")
    result = profiles[name]
    validate_profile(result)
    return result


def apply_condition(entry, profile):
    """Return simulated timings/state separately; never edit raw traces.

    Residual action latency includes transport, serialization and scheduling,
    not pure network latency. Multipliers are scenario assumptions, not fits.
    """
    validate_profile(profile)
    trace = entry["trace"]
    p = profile["injected_parameters"]
    edge, network, cloud, delay = (p[k] for k in ("edge_compute_multiplier", "network_residual_multiplier",
                                                "cloud_compute_multiplier", "added_rtt_ms"))
    prep, state_ms = trace["prep_us"] / 1000, trace["state_us"] / 1000
    action_ms = trace["action_us"] / 1000
    shared_ms = (trace["total_us"] - trace["action_us"]) / 1000
    shared_residual = shared_ms - prep - state_ms
    if shared_residual < -1e-9:
        raise ValueError("Negative shared residual")
    prefix_ms = trace["prefix_us"] / 1000
    remote = trace["action"] != 0
    cloud_ms = trace["cloud_ms"] if remote else 0
    if remote and cloud_ms is None:
        raise ValueError("Missing measured cloud time")
    residual = action_ms - prefix_ms - cloud_ms if remote else 0
    if residual < -1e-6:
        raise ValueError("Measured cloud/prefix time exceeds action time")
    selected_compute = edge * (prefix_ms if remote else action_ms)
    selected_ms = selected_compute + (cloud * cloud_ms + network * max(0, residual) + delay if remote else 0)
    total_ms = edge * (prep + state_ms) + network * max(0, shared_residual) + delay + selected_ms
    state = deepcopy(entry["record"]["state"])
    state["device"]["inference_latency_estimate"] *= edge
    state["network"]["mqtt_rtt_ms"] = state["network"]["mqtt_rtt_ms"] * network + delay
    return {"state": state, "total_latency_ms": total_ms, "selected_latency_ms": selected_ms,
            "selected_edge_compute_ms": selected_compute, "preprocessing_ms": edge * prep,
            "state_inference_ms": edge * state_ms, "cloud_inference_ms": cloud * cloud_ms if remote else None,
            "prefix_latency_ms": prefix_ms * edge if remote else 0,
            "network_condition": deepcopy(profile), "metric_kind": "simulated_from_measured_source"}
