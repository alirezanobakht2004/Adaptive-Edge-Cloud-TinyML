"""Versioned measured candidate dataset. No policy training or v1 mutations."""

from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

from .device_trace import trace_to_record, validate_trace
from .reward_engine.engine import digest, evaluate_group, number, validate_config
from .reward_engine.__main__ import strict_json
from .schema import ACTION_NAMES, validate_record

VERSION = "policy_training_dataset_v2"
LIMITS = {"maximum_rtt_ms", "rtt_drift_ms", "heap_drift_bytes", "psram_drift_bytes",
          "rssi_drift_db", "maximum_snapshot_age_ms"}
GUARD_KEYS = {"window_id", "repeat", "position", "action", "shared_us", "snapshot_us", "age_us",
              "snapshot_rssi", "guard_heap", "guard_psram", "guard_rssi", "guard_cpu",
              "guard_connected", "guard_rtt_us"}


def validate_condition(condition):
    if set(condition) != {"version", "name", "description", "limits", "energy_proxy", "reward"}:
        raise ValueError("Invalid condition fields")
    if condition["version"] != "matched-condition-v1":
        raise ValueError("Invalid condition version")
    for key in ("name", "description"):
        if not isinstance(condition[key], str) or not condition[key].strip():
            raise ValueError(f"Missing condition {key}")
    if set(condition["limits"]) != LIMITS:
        raise ValueError("Invalid condition limits")
    for key, value in condition["limits"].items():
        number(value, key, positive=True)
    if set(condition["energy_proxy"]) != {"compute_reference_ms", "payload_reference_bytes"}:
        raise ValueError("Invalid proxy coefficients")
    for key, value in condition["energy_proxy"].items():
        number(value, key, positive=True)
    validate_config(condition["reward"])
    if condition["reward"]["energy_kind"] != "estimated":
        raise ValueError("This collector implements an estimated compute/payload proxy")
    if condition["reward"]["energy_method"] != energy_method(condition["energy_proxy"]):
        raise ValueError("Energy method must describe the configured proxy exactly")


def energy_method(coefficients):
    return ("Dimensionless estimated proxy: (preprocessing_ms + state_inference_ms + "
            "selected_edge_compute_ms) / " + str(coefficients["compute_reference_ms"]) +
            " + selected_payload_bytes / " + str(coefficients["payload_reference_bytes"]) +
            "; excludes radio waiting, cloud energy and probe traffic; not joules or battery measurement")


def validate_entry(entry, condition):
    if set(entry) != {"trace", "guard", "record"}:
        raise ValueError("Invalid measured entry")
    trace, guard, record = entry["trace"], entry["guard"], entry["record"]
    validate_trace(trace)
    validate_record(record)
    if set(guard) != GUARD_KEYS:
        raise ValueError("Invalid live guard fields")
    for key in GUARD_KEYS - {"guard_connected", "snapshot_rssi", "guard_rssi"}:
        number(guard[key], key)
        if type(guard[key]) is not int:
            raise ValueError("Guard counters must be integers")
    for key in ("snapshot_rssi", "guard_rssi"):
        if type(guard[key]) is not int or not -127 <= guard[key] <= 0:
            raise ValueError("Invalid guard RSSI")
    if type(guard["guard_connected"]) is not bool:
        raise ValueError("Invalid guard connection flag")
    if guard["action"] != trace["action"] or guard["window_id"] != trace["window_id"]:
        raise ValueError("Guard/trace identity mismatch")
    if guard["position"] not in range(4) or guard["repeat"] not in range(16):
        raise ValueError("Invalid repeat/position")
    if guard["snapshot_rssi"] != trace["rssi_dbm"] or trace["total_us"] != guard["shared_us"] + trace["action_us"]:
        raise ValueError("Shared snapshot accounting mismatch")
    if not trace["connected"] or not guard["guard_connected"] or trace["probe_us"] is None:
        raise ValueError("Connected condition requires successful snapshot and guard probes")
    limits = condition["limits"]
    checks = {
        "maximum_rtt_ms": max(trace["probe_us"], guard["guard_rtt_us"]) / 1000,
        "rtt_drift_ms": abs(trace["probe_us"] - guard["guard_rtt_us"]) / 1000,
        "heap_drift_bytes": abs(trace["free_heap"] - guard["guard_heap"]),
        "psram_drift_bytes": abs(trace["psram_free"] - guard["guard_psram"]),
        "rssi_drift_db": abs(trace["rssi_dbm"] - guard["guard_rssi"]),
        "maximum_snapshot_age_ms": guard["age_us"] / 1000,
    }
    if guard["guard_cpu"] != trace["cpu_mhz"]:
        raise ValueError("CPU frequency changed")
    for key, value in checks.items():
        if value > limits[key]:
            raise ValueError(f"Condition violated: {key}={value} > {limits[key]}")
    # Rebuild the v1 projection from raw measurements; copying a prior state on
    # the host cannot hide a different snapshot in the raw firmware trace.
    meta = record["metadata"]
    projected = trace_to_record(trace, run_id=meta["run_id"], device_id=meta["device_id"],
                               timestamp=meta["timestamp"], session_id=meta["session_id"],
                               true_class=record["outcome"]["true_class"], hashes=record["provenance"]["artifact_hashes"])
    for key in ("state", "action", "outcome"):
        if record[key] != projected[key]:
            raise ValueError(f"Raw trace projection mismatch: {key}")
    for key, value in projected["measurements"].items():
        if key not in ("request_receive_time", "response_publish_time") and record["measurements"][key] != value:
            raise ValueError(f"Raw measurement mismatch: {key}")
    if trace["status"] != "success":
        raise ValueError("Pilot requires complete successful execution measurements; retain failures as rejected evidence")
    if trace["action"] and (trace["cloud_ms"] is None or not trace["receive_us"]
                            or record["measurements"]["request_receive_time"] is None
                            or record["measurements"]["response_publish_time"] is None):
        raise ValueError("Remote execution lacks cloud/response evidence")
    return checks


def estimated_energy(trace, condition):
    coefficients, reward = condition["energy_proxy"], condition["reward"]
    selected_compute = trace["action_us"] if trace["action"] == 0 else trace["prefix_us"]
    value = ((trace["prep_us"] + trace["state_us"] + selected_compute) / 1000 / coefficients["compute_reference_ms"]
             + (trace["request_bytes"] + trace["response_bytes"]) / coefficients["payload_reference_bytes"])
    return {"value": value, "kind": "estimated", "proxy_version": reward["energy_proxy_version"],
            "method": reward["energy_method"]}


def build_sample(entries, condition, sample_id):
    validate_condition(condition)
    if not isinstance(sample_id, str) or not sample_id.strip() or not entries:
        raise ValueError("Sample requires identity and measured entries")
    repeats = len(entries) // 4
    if repeats < 4 or repeats > 16 or repeats % 4 or len(entries) != repeats * 4:
        raise ValueError("Use 4, 8, 12 or 16 repeats for balanced action positions")
    states, snapshots, requests, labels = set(), set(), set(), set()
    guards = []
    for entry in entries:
        guards.append(validate_entry(entry, condition))
        states.add(digest(entry["record"]["state"]))
        snapshots.add((entry["guard"]["snapshot_us"], entry["guard"]["shared_us"]))
        labels.add(entry["record"]["outcome"]["true_class"])
        request = entry["trace"]["request_id"]
        if request in requests:
            raise ValueError("Duplicate execution identity")
        requests.add(request)
    if len(states) != 1 or len(snapshots) != 1 or len(labels) != 1:
        raise ValueError("One frozen decision snapshot/ground truth required across all repeats")
    reference = entries[0]["record"]
    trials = {action: [] for action in ACTION_NAMES}
    evaluations = []
    for repeat in range(repeats):
        batch = [e for e in entries if e["guard"]["repeat"] == repeat]
        if len(batch) != 4 or {e["guard"]["position"] for e in batch} != set(range(4)):
            raise ValueError("Incomplete repeat or duplicate action position")
        candidates = [{"record": e["record"], "energy": estimated_energy(e["trace"], condition)} for e in batch]
        group = {"version": "candidate-actions-v1", "decision_id": f"{sample_id}/repeat-{repeat}",
                 "matching_evidence": f"{sample_id}: retained raw traces and live guards; condition {digest(condition)}",
                 "decision": reference, "candidates": candidates}
        evaluated = evaluate_group(group, condition["reward"])
        evaluations.append(evaluated)
        for result in evaluated["candidates"]:
            trials[result["action"]].append(result)
    actions = []
    for action, results in trials.items():
        positions = [e["guard"]["position"] for e in entries if e["trace"]["action"] == action]
        if any(positions.count(position) != repeats // 4 for position in range(4)):
            raise ValueError("Action order is not counterbalanced")
        actions.append({"action": action, "name": ACTION_NAMES[action], "measurements": results,
                        "mean_reward": mean(r["reward"] for r in results),
                        "accuracy": mean(r["prediction_correct"] for r in results),
                        "mean_latency_ms": mean(r["latency_ms"] for r in results),
                        "mean_communication_bytes": mean(r["communication_bytes"] for r in results),
                        "mean_estimated_energy_proxy": mean(r["energy"]["value"] for r in results)})
    best = max(a["mean_reward"] for a in actions)
    ties = [a["action"] for a in actions if best - a["mean_reward"] <= condition["reward"]["tie_tolerance"]]
    return {"dataset_version": VERSION, "sample_id": sample_id, "metadata": deepcopy(reference["metadata"]),
            "state": deepcopy(reference["state"]), "condition": deepcopy(condition),
            "candidate_actions": actions, "label": {"optimal_action": ties[0] if len(ties) == 1 else None,
            "optimal_actions": ties, "method": "argmax-mean-observed-M29-reward",
            "repeat_optimal_actions": [e["optimal_actions"] for e in evaluations]},
            "repeats": repeats, "source_entries": deepcopy(entries), "source_sha256": digest(entries),
            "maximum_observed_drift": {k: max(g[k] for g in guards) for k in LIMITS},
            "training_enabled": False}


def validate_sample(sample):
    expected = build_sample(sample["source_entries"], sample["condition"], sample["sample_id"])
    if sample != expected:
        raise ValueError("Dataset v2 schema/derived values differ from reproducible measurements")


def validate_file(path):
    samples, identities, windows = [], set(), set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        sample = strict_json(line)
        validate_sample(sample)
        if sample["sample_id"] in identities:
            raise ValueError("Duplicate sample_id")
        identities.add(sample["sample_id"])
        windows.add((sample["metadata"]["session_id"], sample["state"]["application"]["window_id"]))
        samples.append(sample)
    if not samples:
        raise ValueError("No measured samples")
    if len({digest(s["condition"]) for s in samples}) != 1:
        raise ValueError("Do not mix calibrations/conditions in this pilot file")
    return {"dataset_version": VERSION, "valid": True, "matched_samples": len(samples),
            "unique_source_windows": len(windows),
            "candidate_executions": sum(s["repeats"] * 4 for s in samples),
            "action_coverage": {str(a): sum(s["repeats"] for s in samples) for a in ACTION_NAMES},
            "optimal_action_distribution": {str(a): sum(s["label"]["optimal_action"] == a for s in samples) for a in ACTION_NAMES},
            "ambiguous_labels": sum(s["label"]["optimal_action"] is None for s in samples),
            "energy_kind": "estimated", "training_allowed": False,
            "limitations": ["One controlled sequential snapshot, not simultaneous physical executions",
                            "Condition-dependent pilot labels; independent sessions and broader conditions still required",
                            "Energy proxy is dimensionless and unvalidated against battery measurements"]}
