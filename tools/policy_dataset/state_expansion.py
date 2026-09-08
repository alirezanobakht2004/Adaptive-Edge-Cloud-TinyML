"""Explicit simulated v3 state and feasibility extension; no policy model."""

from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

from tools.network_conditions.profiles import validate_profile
from .reward_engine.engine import digest, evaluate_group, number
from .run_controlled_campaign import simulate_sample

VERSION = "policy_training_dataset_v3"
SCENARIOS = ("A_normal_poor_network", "B_pressure_good_network", "C_inference_unavailable")


def load_scenario(name):
    values = json.loads(Path(__file__).with_name("state_scenarios_v1.json").read_text())
    if name not in values:
        raise ValueError("Unknown state scenario")
    validate_scenario(values[name])
    return values[name]


def validate_scenario(scenario):
    keys = {"version", "name", "simulated", "local_compute_pressure", "inference_queue_pressure",
            "local_inference_available", "cloud_availability", "transport_available", "queue_reference_ms",
            "quality_rtt_reference_ms", "network_profile", "expected_actions_hypothesis"}
    if (set(scenario) != keys or scenario["version"] != "adaptive-state-scenario-v1"
            or scenario["name"] not in SCENARIOS or scenario["simulated"] is not True):
        raise ValueError("Invalid v3 scenario")
    for key in ("local_compute_pressure", "inference_queue_pressure"):
        number(scenario[key], key)
        if scenario[key] > 1:
            raise ValueError("Pressure must lie in [0,1]")
    for key in ("local_inference_available", "cloud_availability", "transport_available"):
        if type(scenario[key]) is not bool:
            raise ValueError("Availability must be boolean")
    for key in ("queue_reference_ms", "quality_rtt_reference_ms"):
        number(scenario[key], key, positive=True)
    expected = scenario["expected_actions_hypothesis"]
    if not isinstance(expected, list) or any(type(a) is not int or a not in range(4) for a in expected):
        raise ValueError("Invalid diagnostic action hypothesis")
    validate_profile(scenario["network_profile"])
    if scenario["local_inference_available"]:
        pressure = scenario["local_compute_pressure"]
        multiplier = scenario["network_profile"]["injected_parameters"]["edge_compute_multiplier"]
        if pressure >= 1 or abs(multiplier - 1 / (1 - pressure)) > 1e-8:
            raise ValueError("Available inference pressure must match configured compute multiplier")


def feasibility(action, scenario):
    reasons = []
    if action < 3 and not scenario["local_inference_available"]:
        reasons.append("local_inference_service_unavailable")
    if action > 0 and not scenario["cloud_availability"]:
        reasons.append("cloud_unavailable")
    if action > 0 and not scenario["transport_available"]:
        reasons.append("feature_transport_unavailable")
    return reasons


def leaves(value, prefix="state"):
    result = {}
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict):
            result.update(leaves(child, path))
        else:
            result[path] = child
    return result


def expand_sample(source, scenario):
    validate_scenario(scenario)
    projected = simulate_sample(source, scenario["network_profile"])
    config = projected["reward"]["configuration"]
    config.update(energy_proxy_version="m33-availability-compute-payload-proxy-v1",
                  energy_method="Simulated scaled available edge compute / configured compute reference plus payload / configured byte reference; "
                  "omit unavailable state inference; queue waiting, radio waiting and cloud energy excluded; not measured energy")
    queue_ms = scenario["inference_queue_pressure"] * scenario["queue_reference_ms"]
    results = []
    for repeat in range(source["repeats"]):
        candidates = []
        for entry in source["source_entries"]:
            if entry["guard"]["repeat"] != repeat:
                continue
            action = entry["trace"]["action"]
            trial = projected["candidate_actions"][action]["measurements"][repeat]
            record = deepcopy(entry["record"])
            record["metadata"]["measurement_scope"] = "replay"
            record["state"] = deepcopy(projected["state"])
            record["measurements"]["total_latency_ms"] = trial["latency_ms"] + (queue_ms if action < 3 else 0)
            record["provenance"]["notes"] = "M33 simulated counterfactual; queue wait only, no measured pressure"
            energy = deepcopy(trial["energy"])
            energy.update(proxy_version=config["energy_proxy_version"], method=config["energy_method"])
            if not scenario["local_inference_available"]:
                # A cloud-only opportunity cannot incur a fresh unavailable
                # uncertainty inference. Retain only transport/preprocessing.
                skipped_ms = entry["trace"]["state_us"] / 1000 * scenario["network_profile"]["injected_parameters"]["edge_compute_multiplier"]
                record["measurements"]["total_latency_ms"] -= skipped_ms
                energy["value"] -= skipped_ms / source["condition"]["energy_proxy"]["compute_reference_ms"]
            candidates.append({"record": record, "energy": energy})
        group = {"version": "candidate-actions-v1", "decision_id": f"{source['sample_id']}/{scenario['name']}/{repeat}",
                 "matching_evidence": "SIMULATED M33 source replay; unavailable outcomes excluded in v3",
                 "decision": candidates[0]["record"], "candidates": candidates}
        results.append(evaluate_group(group, config))
    actions = []
    for action in range(4):
        reasons = feasibility(action, scenario)
        trials = [] if reasons else [r["candidates"][action] for r in results]
        components = {key: mean(t["normalized"][key] for t in trials) for key in config["weights"]} if trials else None
        actions.append({"action": action, "feasible": not reasons, "status": "infeasible" if reasons else "simulated_success",
            "infeasibility_reasons": reasons, "measurement_kind": "simulated_not_executed",
            "outcomes": [{"prediction": t["prediction"], "correct": t["prediction_correct"]} for t in trials] or None,
            "measurements": trials or None, "mean_reward": mean(t["reward"] for t in trials) if trials else None,
            "reward_components": {k: config["weights"][k] * v for k, v in components.items()} if components else None,
            "mean_normalized": components, "queue_wait_ms": (queue_ms if action < 3 else 0) if trials else None})
    feasible = [a for a in actions if a["feasible"]]
    best = max((a["mean_reward"] for a in feasible), default=None)
    winners = [a["action"] for a in feasible if best - a["mean_reward"] <= config["tie_tolerance"]]
    state = deepcopy(projected["state"])
    state["device"].update(local_compute_pressure=scenario["local_compute_pressure"],
        inference_queue_pressure=scenario["inference_queue_pressure"], local_inference_available=scenario["local_inference_available"])
    state["network"].update(cloud_availability=scenario["cloud_availability"], transport_available=scenario["transport_available"],
        network_quality_score=1 / (1 + state["network"]["mqtt_rtt_ms"] / scenario["quality_rtt_reference_ms"]))
    if not scenario["transport_available"]:
        state["network"].update(connection_state="disconnected", mqtt_rtt_ms=None, network_quality_score=0)
    if scenario["local_inference_available"]:
        state["device"]["inference_latency_estimate"] += queue_ms
    else:
        state["uncertainty"] = dict.fromkeys(state["uncertainty"])
        state["device"]["inference_latency_estimate"] = None
        state["application"]["predicted_class"] = None
    source_values = leaves(source["state"])
    provenance = {}
    for path, value in leaves(state).items():
        if value is None:
            kind = "unavailable_or_unmeasured"
        elif path in source_values and value == source_values[path]:
            kind = "carried_from_measured_source_not_remeasured"
        else:
            kind = "simulated_or_derived"
        provenance[path] = {"kind": kind}
    # Explicit simulated inputs remain simulated even if numerically unchanged.
    for path in ("state.device.local_compute_pressure", "state.device.inference_queue_pressure",
                 "state.device.local_inference_available", "state.network.cloud_availability",
                 "state.network.transport_available", "state.network.network_quality_score"):
        provenance[path] = {"kind": "simulated_or_derived"}
    return {"dataset_version": VERSION, "sample_id": f"{source['sample_id']}/{scenario['name']}",
            "source_sample_id": source["sample_id"], "source_sha256": digest(source),
            "metadata": {"model_version": source["metadata"]["model_version"], "feature_version": source["metadata"]["feature_version"],
                         "source_session": source["metadata"]["session_id"]},
            "scenario": deepcopy(scenario), "state": state, "action_outcomes": actions,
            "reward_configuration": deepcopy(config), "optimal_action": winners[0] if len(winners) == 1 else None,
            "optimal_actions": winners, "label_status": "no_feasible_action" if not winners else "unique" if len(winners) == 1 else "tie",
            "measurement_provenance": {"simulated": True, "new_hardware_measurements": 0, "state_fields": provenance,
                 "predictions": "Retained source predictions for feasible actions; invariance assumed",
                 "availability": "Simulated inference service availability, not whole-device power state",
                 "queue": "Queue pressure times configured wait reference; waiting excluded from energy proxy",
                 "missing_state": "No fresh uncertainty/local prediction asserted when inference unavailable",
                 "energy": "Simulated dimensionless compute/payload proxy; never measured energy"},
            "training_enabled": False}


def validate_v3(sample, source, scenario):
    if digest(sample) != digest(expand_sample(source, scenario)):
        raise ValueError("V3 schema/state/feasibility/reward/provenance does not reconstruct")


def dominance(actions, target):
    if not actions[target]["feasible"]:
        return None
    t = actions[target]["mean_normalized"]
    return [a["action"] for a in actions if a["feasible"] and a["action"] != target
            and a["mean_normalized"]["accuracy"] >= t["accuracy"]
            and all(a["mean_normalized"][k] <= t[k] for k in ("latency", "communication", "energy_proxy"))
            and a["mean_normalized"] != t]


def coverage_report(rows):
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty/duplicate v3 records")
    split_coverage = {}
    for action in (1, 2):
        feasible = [r for r in rows if r["action_outcomes"][action]["feasible"]]
        split_coverage[str(action)] = {"feasible_states": len(feasible),
            "non_dominated_states": sum(not dominance(r["action_outcomes"], action) for r in feasible),
            "optimal_labels": sum(r["optimal_action"] == action for r in rows)}
    paths = leaves(rows[0]["state"])
    statistics = {}
    for path in paths:
        values = [leaves(r["state"])[path] for r in rows]
        known = [v for v in values if v is not None]
        counts = {str(v): values.count(v) for v in sorted(set(known), key=str)}
        statistics[path] = {"missing": len(values) - len(known), "value_counts": counts,
                            "provenance_kinds": sorted({r["measurement_provenance"]["state_fields"][path]["kind"] for r in rows})}
    labels = {r["optimal_action"] for r in rows if r["optimal_action"] is not None}
    criteria = {"more_than_one_action_label": len(labels) > 1,
                "each_split_has_non_dominated_feasible_state": all(s["non_dominated_states"] > 0 for s in split_coverage.values()),
                "state_coverage_report_exists": True}
    return {"dataset_version": VERSION, "samples": len(rows), "simulated": True, "new_hardware_measurements": 0,
            "independent_source_windows": len({r["source_sample_id"] for r in rows}),
            "label_distribution": {str(a): sum(r["optimal_action"] == a for r in rows) for a in range(4)},
            "unlabeled": sum(r["optimal_action"] is None for r in rows), "split_coverage": split_coverage,
            "action_coverage": {str(a): {"outcome_records": len(rows),
                "feasible_records": sum(r["action_outcomes"][a]["feasible"] for r in rows),
                "infeasible_records": sum(not r["action_outcomes"][a]["feasible"] for r in rows)} for a in range(4)},
            "scenarios": {name: {"samples": sum(r["scenario"]["name"] == name for r in rows),
                "label_distribution": {str(a): sum(r["scenario"]["name"] == name and r["optimal_action"] == a for r in rows) for a in range(4)},
                "hypothesis_matches": sum(r["scenario"]["name"] == name and r["optimal_action"] in r["scenario"]["expected_actions_hypothesis"] for r in rows)} for name in SCENARIOS},
            "state_statistics": statistics, "minimum_readiness_criteria": criteria,
            "minimum_gate_passed": all(criteria.values()), "training_allowed": False,
            "limitations": ["Simulated conditions with uncalibrated pressure/queue response", "Only one source session/device",
                            "No fresh uncertainty when inference service unavailable", "Independent physical diversity and reward calibration remain unqualified"]}
