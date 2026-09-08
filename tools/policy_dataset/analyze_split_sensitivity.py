"""Read-only reward decomposition and split competitiveness analysis; no training."""

import argparse
from itertools import combinations, product
from pathlib import Path
from statistics import mean

import numpy as np

from tools.network_conditions.profiles import NAMES, apply_condition, load_profile
from .reward_engine.engine import digest, number
from .run_controlled_campaign import DEFAULT_SOURCE, read_rows, validate_output
from .run_matched_campaign import write_json
from .validate_matched_dataset import validate_campaign

COMPONENTS = ("accuracy", "latency", "communication", "energy_proxy")


def normalized_actions(sample):
    actions = sample["candidate_actions"]
    if [a["action"] for a in actions] != list(range(4)):
        raise ValueError("Exactly four ordered policy actions required")
    result = []
    for action in actions:
        trials = action["measurements"]
        if not trials:
            raise ValueError("Missing candidate measurements")
        for trial in trials:
            for key in COMPONENTS:
                number(trial["normalized"][key], key)
        result.append({key: mean(t["normalized"][key] for t in trials) for key in COMPONENTS})
    return result


def rewards(actions, weights):
    if set(weights) != set(COMPONENTS):
        raise ValueError("Invalid reward weight fields")
    for key, value in weights.items():
        number(value, key)
    if not any(weights.values()):
        raise ValueError("All-zero reward is uninformative")
    return [weights["accuracy"] * a["accuracy"] - sum(weights[k] * a[k] for k in COMPONENTS[1:]) for a in actions]


def winners(scores, tolerance):
    number(tolerance, "tie tolerance")
    best = max(scores)
    return [i for i, score in enumerate(scores) if best - score <= tolerance]


def dominating_actions(actions, target):
    """Pareto dominance excludes unique optimality, but zero cost weights can tie."""
    target_quality = np.array([actions[target]["accuracy"], *[-actions[target][k] for k in COMPONENTS[1:]]])
    result = []
    for action, row in enumerate(actions):
        if action == target:
            continue
        quality = np.array([row["accuracy"], *[-row[k] for k in COMPONENTS[1:]]])
        if np.all(quality >= target_quality) and np.any(quality > target_quality):
            result.append(action)
    return result


def best_weight_margin(actions, target):
    """Exact small LP by vertex enumeration; nonnegative weights sum to one.

    Maximize the target's minimum reward advantage over all three alternatives.
    Eliminate the last weight; solve vertices in (w0,w1,w2,margin). This does
    not select production weights. Positive margin means some weight vector
    can uniquely favor the target on this fixed observation.
    """
    quality = np.array([[a["accuracy"], *[-a[k] for k in COMPONENTS[1:]]] for a in actions], dtype=float)
    if quality.shape != (4, 4) or not np.isfinite(quality).all():
        raise ValueError("Invalid action quality matrix")
    rows, bounds = [], []
    for other in range(4):
        if other == target:
            continue
        delta = quality[target] - quality[other]
        rows.append([*list(-(delta[:3] - delta[3])), 1.0])
        bounds.append(delta[3])
    rows.extend([[-1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [1., 1., 1., 0]])
    bounds.extend([0., 0., 0., 1.])
    matrix, bound = np.array(rows), np.array(bounds)
    best = None
    for active in combinations(range(7), 4):
        try:
            solution = np.linalg.solve(matrix[list(active)], bound[list(active)])
        except np.linalg.LinAlgError:
            continue
        if np.all(matrix @ solution <= bound + 1e-9) and (best is None or solution[3] > best[3]):
            best = solution
    if best is None:
        raise ValueError("No feasible weight vertex")
    weights = np.maximum(0, np.r_[best[:3], 1 - sum(best[:3])])
    weights /= weights.sum()
    return {"maximum_minimum_margin": float(best[3]),
            "can_be_unique_optimum": bool(best[3] > 1e-8),
            "illustrative_weights_not_recommendation": dict(zip(COMPONENTS, map(float, weights)))}


def break_even(actions, target, competitor, config):
    """Change only target accuracy or latency; other observed costs held fixed."""
    weights, scales = config["weights"], config["scales"]
    scores = rewards(actions, weights)
    gap = scores[competitor] - scores[target]
    accuracy_increment = gap / weights["accuracy"] if weights["accuracy"] else None
    allowed_latency = ((actions[target]["latency"] - gap / weights["latency"]) * scales["latency_ms"]
                       if weights["latency"] else None)
    return {"competitor": competitor, "current_reward_deficit": gap,
            "target_accuracy_increment_to_tie": accuracy_increment,
            "accuracy_only_change_feasible": accuracy_increment is not None and 0 <= actions[target]["accuracy"] + accuracy_increment <= 1,
            "target_latency_ms_to_tie": allowed_latency,
            "latency_only_change_feasible": allowed_latency is not None and allowed_latency >= 0}


def latency_parts(entry, profile):
    costs = apply_condition(entry, profile)
    edge, cloud = costs["selected_edge_compute_ms"], costs["cloud_inference_ms"] or 0
    return {"shared_preprocessing_ms": costs["preprocessing_ms"],
            "shared_state_compute_ms": costs["state_inference_ms"],
            "shared_probe_scheduling_residual_ms": costs["total_latency_ms"] - costs["selected_latency_ms"] - costs["preprocessing_ms"] - costs["state_inference_ms"],
            "selected_edge_compute_ms": edge, "cloud_compute_ms": cloud,
            "selected_transport_scheduling_residual_ms": costs["selected_latency_ms"] - edge - cloud}


def analyze_cohort(samples, sources, name, simulated):
    if not samples:
        raise ValueError("Empty cohort")
    config = samples[0]["reward"]["configuration"] if simulated else samples[0]["condition"]["reward"]
    weights, scales = config["weights"], config["scales"]
    normalized = [normalized_actions(s) for s in samples]
    averaged = [{k: mean(a[action][k] for a in normalized) for k in COMPONENTS} for action in range(4)]
    by_action = []
    for action in range(4):
        row = averaged[action]
        parts = []
        for sample in samples:
            source = sources[sample["source_sample_id"]] if simulated else sample
            profile = sample["network_condition"] if simulated else load_profile("baseline")
            parts.extend(latency_parts(e, profile) for e in source["source_entries"] if e["trace"]["action"] == action)
        weighted = {k: weights[k] * row[k] for k in COMPONENTS}
        by_action.append({"action": action, "correctness_fraction": row["accuracy"],
            "mean_latency_ms": row["latency"] * scales["latency_ms"],
            "mean_payload_bytes": row["communication"] * scales["communication_bytes"],
            "mean_energy_proxy": row["energy_proxy"] * scales["energy_proxy"],
            "weighted_reward_components": weighted, "mean_reward": rewards(averaged, weights)[action],
            "mean_latency_contributions": {k: mean(p[k] for p in parts) for k in parts[0]}})
    sensitivity = []
    for latency, communication, energy in product((0., .01, .1, 1., 10.), repeat=3):
        variant = dict(accuracy=1., latency=latency, communication=communication, energy_proxy=energy)
        selected = [winners(rewards(a, variant), config["tie_tolerance"]) for a in normalized]
        sensitivity.append({"weights": variant, "unique_winners": {str(i): sum(w == [i] for w in selected) for i in range(4)},
                            "ties": sum(len(w) > 1 for w in selected)})
    splits = {}
    for target in (1, 2):
        per_state = []
        for sample, actions in zip(samples, normalized):
            per_state.append({"sample_id": sample["sample_id"], "dominated_by": dominating_actions(actions, target),
                              **best_weight_margin(actions, target)})
        splits[str(target)] = {"mean_cost_weight_feasibility": best_weight_margin(averaged, target),
            "mean_cost_break_even": [break_even(averaged, target, j, config) for j in range(4) if j != target],
            "pareto_dominated_states": sum(bool(p["dominated_by"]) for p in per_state),
            "states_with_some_unique_optimal_weights": sum(p["can_be_unique_optimum"] for p in per_state),
            "per_state_weight_feasibility": per_state}
    return {"name": name, "simulated": simulated, "source_sha256": digest(samples), "samples": len(samples),
            "reward_configuration": config, "actions": by_action, "splits": splits, "weight_grid": sensitivity}


def run(output):
    validate_campaign(DEFAULT_SOURCE)
    real = read_rows(DEFAULT_SOURCE / "policy_training_dataset_v2.jsonl")
    sources = {s["sample_id"]: s for s in real}
    cohorts = [analyze_cohort(real, sources, "real_M30", False)]
    root = Path("data/policy/policy_training_dataset_v2/controlled_network_v1")
    for name in NAMES:
        validate_output(root / name, DEFAULT_SOURCE)
        cohorts.append(analyze_cohort(read_rows(root / name / "policy_training_dataset_v2.jsonl"), sources, name, True))
    report = {"version": "split-sensitivity-analysis-v1", "cohorts": cohorts,
              "independent_source_windows": len(real), "training_allowed": False,
              "notes": ["M30 measured durations with estimated energy; M31 costs are simulated",
                        "Profiles/repeats reuse the same windows; cohorts are not independent observations",
                        "Break-even changes hold all other costs fixed; feasibility is mathematical, not physical",
                        "Weight grid and simplex LP are diagnostics, not calibrated or recommended weights"]}
    write_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/phase10_3_split_sensitivity.json"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
