"""Fixed and rule-based benchmark strategies. No learned policy or implicit transport."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from tools.policy_dataset.schema import Action, validate_record, _number


@dataclass(frozen=True)
class Selection:
    selected_action: int


class PolicyStrategy(Protocol):
    name: str

    def select(self, state: dict) -> Selection: ...


@dataclass(frozen=True)
class FixedStrategy:
    name: str
    action: Action

    def select(self, state: dict) -> Selection:
        return Selection(int(self.action))


@dataclass(frozen=True)
class RuleBasedAdaptive:
    """Configurable benchmark heuristic; missing state falls back to local."""

    confidence_threshold: float = 0.8
    fast_rtt_ms: float = 50
    maximum_rtt_ms: float = 200
    name: str = "RULE_BASED_ADAPTIVE"

    def __post_init__(self):
        _number(self.confidence_threshold, "confidence_threshold", 1)
        _number(self.fast_rtt_ms, "fast_rtt_ms")
        _number(self.maximum_rtt_ms, "maximum_rtt_ms")
        if any(x is None for x in (self.confidence_threshold, self.fast_rtt_ms, self.maximum_rtt_ms)) or self.fast_rtt_ms > self.maximum_rtt_ms:
            raise ValueError("Invalid rule thresholds")

    def select(self, state: dict) -> Selection:
        network = state["network"]
        confidence = state["uncertainty"]["confidence"]
        rtt = network["mqtt_rtt_ms"]
        if (network["connection_state"] != "connected" or confidence is None or rtt is None
                or confidence >= self.confidence_threshold or rtt > self.maximum_rtt_ms):
            return Selection(int(Action.ALL_LOCAL))
        return Selection(int(Action.SPLIT1 if rtt <= self.fast_rtt_ms else Action.SPLIT2))


def strategies():
    return [FixedStrategy("ALL_LOCAL", Action.ALL_LOCAL),
            FixedStrategy("ALL_CLOUD", Action.ALL_CLOUD),
            FixedStrategy("FIXED_SPLIT1", Action.SPLIT1),
            FixedStrategy("FIXED_SPLIT2", Action.SPLIT2), RuleBasedAdaptive()]


def run_strategy(strategy: PolicyStrategy, record: dict, executors: dict) -> dict:
    """Executors are explicit adapters; missing ALL_CLOUD support fails rather than using Split3."""
    validate_record(record)
    selected = strategy.select(deepcopy(record["state"])).selected_action
    if type(selected) is not int or selected not in range(4):
        raise ValueError("Strategy returned an invalid action")
    if selected not in executors:
        raise NotImplementedError(f"No executor for {Action(selected).name}; no implicit split alias")
    result = deepcopy(record)
    result["action"] = selected
    executors[selected](result)
    validate_record(result)
    if result["action"] != selected:
        raise ValueError("Executor changed the selected action")
    if result["state"] != record["state"]:
        raise ValueError("Executor changed pre-decision state")
    return result


def placeholder_reward(record, *, latency_scale_ms, communication_scale_bytes,
                       estimated_energy_score, energy_proxy_version):
    """Dimensionless negative sum for schema exercises, not a validated reward model."""
    for name, value in (("latency scale", latency_scale_ms), ("communication scale", communication_scale_bytes)):
        _number(value, name)
        if value is None or value == 0:
            raise ValueError("Reward scales must be positive")
    _number(estimated_energy_score, "estimated energy score")
    if estimated_energy_score is None or not energy_proxy_version or energy_proxy_version == "unconfigured":
        raise ValueError("Estimated energy assumptions are required")
    measures = record["measurements"]
    if any(measures[k] is None for k in ("total_latency_ms", "request_bytes", "response_bytes")):
        raise ValueError("Cannot compute reward from missing measurements")
    latency = measures["total_latency_ms"] / latency_scale_ms
    communication = (measures["request_bytes"] + measures["response_bytes"]) / communication_scale_bytes
    reward = {"latency_component": latency, "communication_cost_component": communication,
              "energy_component": estimated_energy_score, "energy_kind": "estimated",
              "energy_proxy_version": energy_proxy_version, "final_reward": -(latency + communication + estimated_energy_score),
              "calculation": "placeholder-negative-sum-v1"}
    result = deepcopy(record)
    result["reward"] = reward
    validate_record(result)
    return result


def summarize(records, *, scope):
    """Keep scopes/actions separate and return null when no measurement supports a metric."""
    if scope not in ("host", "device", "replay"):
        raise ValueError("Select one measurement scope")
    groups = {int(action): [] for action in Action}
    for record in records:
        validate_record(record)
        if (record["metadata"]["record_kind"] == "observation" and record["metadata"]["measurement_scope"] == scope
                and record["outcome"]["status"] != "not_executed"):
            groups[record["action"]].append(record)
    output = {}
    for action, rows in groups.items():
        labeled = [r for r in rows if r["outcome"]["true_class"] is not None]
        correct = sum(r["outcome"]["status"] == "success" and
                      r["outcome"]["predicted_class"] == r["outcome"]["true_class"] for r in labeled)
        latency = [r["measurements"]["total_latency_ms"] for r in rows if r["measurements"]["total_latency_ms"] is not None]
        byte_counts = [r["measurements"]["request_bytes"] + r["measurements"]["response_bytes"] for r in rows
                       if all(r["measurements"][k] is not None for k in ("request_bytes", "response_bytes"))]
        energy = [r["reward"]["energy_component"] for r in rows if r["reward"]["energy_component"] is not None]
        assumptions = {(r["reward"]["energy_kind"], r["reward"]["energy_proxy_version"]) for r in rows
                       if r["reward"]["energy_component"] is not None}
        if len(assumptions) > 1:
            raise ValueError("Do not aggregate incompatible energy-proxy assumptions")
        output[action] = {"attempts": len(rows), "successes": sum(r["outcome"]["status"] == "success" for r in rows),
                          "labeled_count": len(labeled), "accuracy": correct / len(labeled) if labeled else None,
                          "latency_count": len(latency), "mean_latency_ms": sum(latency) / len(latency) if latency else None,
                          "bytes_count": len(byte_counts), "mean_communication_bytes": sum(byte_counts) / len(byte_counts) if byte_counts else None,
                          "energy_count": len(energy), "mean_estimated_energy_proxy": sum(energy) / len(energy) if energy else None}
    return output
