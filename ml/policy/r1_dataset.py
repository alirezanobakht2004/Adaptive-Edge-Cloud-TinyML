"""Corrected R1 dataset: cached LOCAL, actual model predictions, explicit network simulation."""

import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from .binary_contract import ROOT, FEATURES, feature_vector, read_json
from .binary_policy_dataset import read_rows, write_json
from .r1_reuse import salvage_m30, cached_local, cloud_candidate, label_candidates, load_reward
from tools.policy_dataset.reward_engine.engine import digest
from server.app.r1_schema import make_request, encode, CLOUD_VERSION, POLICY_VERSION, SCHEMA_VERSION

VERSION = "policy_training_dataset_binary_r1_v2"
OUTPUT = ROOT / f"data/policy/{VERSION}"
CONFIG = ROOT / "ml/policy/policy_config_v3.json"
SEED = 20260908


def group_key(session, window):
    return digest({"source_dataset_version": "dataset-v1", "feature_version": "features-v1",
                   "source_session": session, "window_id": str(window)})


def raw_vector(state):
    return [state[name.split(".")[1]][name.split(".")[2]] for name in FEATURES]


def assign_groups(rows):
    groups = {r["group_key"] for r in rows}
    positives = {r["group_key"] for r in rows if r["label"]["optimal_action"] == 1}
    holdout = set()
    for stratum in (positives, groups - positives):
        ordered = sorted(stratum, key=lambda key: digest([SEED, key]))
        # A singleton positive cannot appear in both partitions. Keep it in fit;
        # report that held-out CLOUD-oracle recall is not identifiable.
        count = min(len(ordered) - 1, math.ceil(len(ordered) / 4)) if ordered else 0
        holdout.update(ordered[:count])
    for row in rows:
        row["partition"] = "holdout" if row["group_key"] in holdout else "train"


def derive_rows():
    oracle = read_json(ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json")
    if oracle["test_partition_used"] or oracle["cloud_model_version"] != CLOUD_VERSION:
        raise ValueError("Wrong development oracle/model")
    measured = salvage_m30()
    anchor = {"heap": median(r["state"]["device"]["free_heap"] for r in measured),
              "local_ms": median(r["state"]["device"]["inference_latency_estimate"] for r in measured),
              "rtt_ms": median(r["state"]["network"]["mqtt_rtt_ms"] for r in measured),
              "server_ms": median(t["cloud_ms"] for r in measured for t in r["raw_cloud_traces"])}
    reward = load_reward()
    severe_rtt = 1.5 * reward["weights"]["accuracy"] * reward["scales"]["latency_ms"] / reward["weights"]["latency"]
    profiles = {"baseline": anchor["rtt_ms"], "high_latency": anchor["rtt_ms"] + 150,
                "severe_latency": severe_rtt}
    rows = []
    for row in measured:
        rows.append({"dataset_version": VERSION, "sample_id": "m30-" + digest(row["source_id"]),
                     "group_key": group_key(row["metadata"]["session_id"], row["source_window"]),
                     "source": {"dataset": "policy_training_dataset_v2", "record_id": row["source_id"],
                                "session": row["metadata"]["session_id"], "window": row["source_window"],
                                "acquisition_id": row["metadata"]["run_id"], "sha256": row["source_sha256"]},
                     "state": row["state"], "true_class": row["true_class"], "candidates": [row["local"], row["cloud"]],
                     "label": row["label"], "provenance": {"kind": "measured", "profile": "M30-salvaged",
                       "local": "measured cached result; incremental accounting zero", "cloud": "measured legacy request bytes/time; estimated energy proxy"}})
    for original in oracle["rows"]:
        local, cloud = original["local"], original["cloud"]
        for profile, rtt in profiles.items():
            state = {"uncertainty": {k: local[k] for k in ("confidence", "entropy", "margin")},
                     "device": {"free_heap": anchor["heap"], "inference_latency_estimate": anchor["local_ms"]},
                     "network": {"mqtt_rtt_ms": rtt, "connection_state": "connected"}}
            identity = f"controlled-{original['session']}-{original['window_index']}-{profile}"
            request = make_request(identity, original["normalized_features"], raw_vector(state))
            response = {"schema_version": SCHEMA_VERSION, "request_id": identity, "mode": "CLOUD", "success": True,
                        "predicted_class_id": cloud["prediction"], "confidence": cloud["confidence"],
                        "server_latency_ms": anchor["server_ms"], "model_version": CLOUD_VERSION, "policy_version": POLICY_VERSION}
            byte_count = len(encode(request)) + len(encode(response))
            candidates = [cached_local(local["prediction"], original["true_class"]),
                          cloud_candidate(cloud["prediction"], original["true_class"], rtt + anchor["server_ms"], byte_count, "simulated")]
            rows.append({"dataset_version": VERSION, "sample_id": identity,
                         "group_key": group_key(original["session"], original["window_index"]),
                         "source": {"dataset": "dataset-v1", "record_id": original["source_id"],
                                    "session": original["session"], "window": str(original["window_index"]),
                                    "acquisition_id": original["session"], "sha256": digest(original)},
                         "state": state, "true_class": original["true_class"], "normalized_features": original["normalized_features"],
                         "candidates": candidates, "label": label_candidates(*candidates),
                         "provenance": {"kind": "simulated", "profile": profile, "simulated": True,
                             "prediction": "actual frozen-model desktop output; not simulated correctness",
                             "device": "M30 pooled measured resource anchor assigned to development window; not fresh device measurement",
                             "network": "simulated RTT; only label-varying controlled parameter is visible mqtt_rtt_ms",
                             "server_compute": "pooled M30 measured timing anchor; not remeasured on development window",
                             "bytes": "exact serialized controlled template length; not physical traffic measurement",
                             "injected_parameters": {"mqtt_rtt_ms": rtt}, "request": request, "response_template": response}})
    for row in rows:
        row["versions"] = {"dataset": VERSION, "local_model": "gesture-model-v1.1.0", "cloud_model": CLOUD_VERSION,
                           "reward": reward["version"], "feature_contract": "binary-policy-contract-v2-r1-reuse"}
    assign_groups(rows)
    return rows, anchor, profiles


def qualify(rows):
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    for row in rows:
        if row["dataset_version"] != VERSION or row["candidates"][0]["second_inference_count"] != 0:
            raise ValueError("Invalid R1 reuse schema")
        if row["label"] != label_candidates(*row["candidates"]):
            raise ValueError("Incorrect binary reward label")
    train = {r["group_key"] for r in rows if r["partition"] == "train"}
    holdout = {r["group_key"] for r in rows if r["partition"] == "holdout"}
    labels = Counter(r["label"]["optimal_action"] for r in rows)
    report = {"dataset_version": VERSION, "records": len(rows), "unique_windows": len(train | holdout),
              "sessions": sorted({r["source"]["session"] for r in rows}), "train_groups": len(train),
              "holdout_groups": len(holdout), "overlap_count": len(train & holdout),
              "label_counts": {str(k): labels[k] for k in (0, 1)},
              "by_provenance": {p: {"records": sum(r["provenance"]["kind"] == p for r in rows),
                  "labels": {str(k): sum(r["provenance"]["kind"] == p and r["label"]["optimal_action"] == k for r in rows) for k in (0,1)}} for p in ("measured", "simulated")},
              "by_partition": {p: {"records": sum(r["partition"] == p for r in rows),
                  "labels": {str(k): sum(r["partition"] == p and r["label"]["optimal_action"] == k for r in rows) for k in (0,1)}} for p in ("train", "holdout")},
              "cloud_benefit_source_groups": len({r["group_key"] for r in rows if r["label"]["optimal_action"] == 1}),
              "energy": "estimated/simulated incremental proxy only",
              "limitations": ["One CLOUD-benefit source window; replay does not create independent evidence.",
                "The singleton CLOUD group is reserved for fitting; no held-out CLOUD-optimal support exists.",
                "No final TEST used. Gesture classifiers previously used development data for fitting/selection.",
                "Simulated network/resources are not physical measurements; no natural CLOUD-optimal hardware claim.",
                "M31 edge/cloud compute multipliers and M33 hidden queue/availability fields are excluded."]}
    contract = read_json(ROOT / "ml/policy/policy_config_v2.json")
    gates = {
        "reuse_local_no_duplicate_charge": all(r["candidates"][0]["second_inference_count"] == 0 for r in rows),
        "both_labels": all(labels[k] > 0 for k in (0, 1)),
        "schema_valid": True, "reward_frozen_reproducible": load_reward()["version"] == "binary-reward-experiment-v2-r1-reuse",
        "input_order_frozen": tuple(contract["state_feature_names"]) == FEATURES,
        "runtime_sources_defined": all(f["runtime_source"] and f["available_before_decision"] for f in contract["features"]),
        "selected_inputs_available": all(all(np.isfinite(raw_vector(r["state"]))) for r in rows),
        "no_hidden_controlled_state": all(r["provenance"]["kind"] == "measured" or set(r["provenance"]["injected_parameters"]) == {"mqtt_rtt_ms"} for r in rows),
        "provenance_explicit": all(r["provenance"]["kind"] in ("measured", "simulated") for r in rows),
        "group_holdout_no_overlap": not train & holdout and bool(train and holdout),
        "versions_explicit": all(r["versions"]["cloud_model"] == CLOUD_VERSION for r in rows),
        "rewards_reconstructed": True,
    }
    report["gates"] = gates
    report["training_allowed"] = all(gates.values())
    return report


def build():
    if OUTPUT.exists() or CONFIG.exists():
        raise ValueError("Immutable R1 dataset/config output exists")
    rows, anchors, profiles = derive_rows()
    report = qualify(rows)
    old = read_json(ROOT / "ml/policy/policy_config_v2.json")
    config = deepcopy(old)
    config.update(version="policy-config-v3", policy_version="binary-policy-contract-v2-r1-reuse",
                  dataset_version=VERSION, training_enabled=report["training_allowed"], status="qualified with explicit singleton-positive limitation")
    x = np.asarray([raw_vector(r["state"]) for r in rows if r["partition"] == "train"], dtype=np.float64)
    means, deviations = x.mean(axis=0), x.std(axis=0)
    for i, feature in enumerate(config["features"]):
        feature["normalization"] = {"offset": float(means[i]), "divisor": float(deviations[i]) if deviations[i] > 1e-12 else 1.0}
    config["normalization"] = {"version": "binary-state-normalization-v2", "algorithm": "(raw - train mean) / train population std; constant fields divide by one",
                               "fit_partition": "train groups only", "dimension": 6, "missing": "reject", "clip": False}
    config["reward_reference"] = {"path": "ml/policy/reward_binary_experiment_v2_r1_reuse.json", "version": load_reward()["version"], "sha256": digest(load_reward())}
    config["grouping"] = {"version": "r1-window-groups-v2", "seed": SEED, "algorithm": "Group-stratified 25% holdout by presence of CLOUD oracle; singleton positive group retained for training; no record splitting"}
    write_json(CONFIG, config)
    for row in rows:
        row["policy_input"] = feature_vector(row["state"], config)
    OUTPUT.mkdir(parents=True)
    import json
    (OUTPUT / f"{VERSION}.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    manifest = {"version": "r1-binary-manifest-v2", "dataset_version": VERSION,
                "transformation_version": "r1-reuse-raw-and-oracle-v1", "reward": load_reward(), "policy_contract": config,
                "records_sha256": digest(rows), "oracle_sha256": digest(read_json(ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json")),
                "source_versions": ["policy_training_dataset_v2", "dataset-v1", "features-v1"],
                "source_commit": "1572430", "anchors": anchors, "profiles": profiles,
                "profile_rationale": "baseline and +150ms follow M31; severe RTT is 1.5 times accuracy/latency reward break-even, not a cloud bonus",
                "excluded": ["Historical redundant action-0 executions", "M31 hidden edge/cloud multipliers", "M33 hidden pressure, queue and availability outcomes", "final TEST partition"],
                "provenance": "Raw source IDs and hashes per row; actual desktop predictions; simulated network/device assignments; no fake measured energy",
                "gate": report}
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(OUTPUT / "qualification.json", report)
    write_json(ROOT / "docs/evidence/phase9_r1_reuse_dataset_qualification.json", report)
    print(report)


def validate():
    rows, _, _ = derive_rows()
    config = read_json(CONFIG)
    for r in rows:
        r["policy_input"] = feature_vector(r["state"], config)
    stored = read_rows(OUTPUT / f"{VERSION}.jsonl")
    manifest = read_json(OUTPUT / "manifest.json")
    if rows != stored or digest(rows) != manifest["records_sha256"] or config != manifest["policy_contract"]:
        raise ValueError("R1 dataset reconstruction mismatch")
    report = qualify(rows)
    if report != read_json(OUTPUT / "qualification.json"):
        raise ValueError("Qualification mismatch")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(validate()) if args.validate else build()
