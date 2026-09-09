"""Training-only threshold calibration; same fixed group holdout as learned policy."""

import numpy as np
import tensorflow as tf
from .binary_contract import ROOT
from .binary_policy_dataset import read_rows, write_json
from .r1_dataset import OUTPUT as DATASET, VERSION as DATASET_VERSION
from .train_meta_binary import OUTPUT, VERSION, evaluate


def select(state, config, cloud_available=True):
    if not cloud_available or state["network"]["connection_state"] != "connected":
        return 0
    return int(state["uncertainty"]["entropy"] > config["entropy_threshold_nats"]
               and state["network"]["mqtt_rtt_ms"] <= config["maximum_rtt_ms"])


def calibrate(rows):
    if not rows or any(r["partition"] != "train" for r in rows):
        raise ValueError("Threshold calibration requires training-only records")
    entropy = np.asarray([r["state"]["uncertainty"]["entropy"] for r in rows])
    rtt = np.asarray([r["state"]["network"]["mqtt_rtt_ms"] for r in rows])
    rewards = np.asarray([[v["value"] for v in r["label"]["rewards"]] for r in rows])
    thresholds = np.r_[np.nextafter(entropy.min(), -np.inf), np.unique(entropy)]
    best_key, best = None, None
    for h in thresholds:
        for network in np.unique(rtt):
            actions = ((entropy > h) & (rtt <= network)).astype(int)
            achieved = float(rewards[np.arange(len(rows)), actions].mean())
            key = (achieved, -int(actions.sum()), float(h), -float(network))
            if best_key is None or key > best_key:
                best_key = key
                best = {"version": "rule-based-binary-v1", "entropy_threshold_nats": float(h), "maximum_rtt_ms": float(network),
                        "source": "training-group empirical thresholds maximizing mean R1 reward; ties prefer fewer cloud calls, higher entropy threshold, lower RTT",
                        "calibration_records": len(rows), "training_mean_reward": achieved,
                        "cloud_availability_source": "connected MQTT and successful pre-decision application probe"}
    return best


def run():
    rows = read_rows(DATASET / f"{DATASET_VERSION}.jsonl")
    config = calibrate([r for r in rows if r["partition"] == "train"])
    holdout = [r for r in rows if r["partition"] == "holdout"]
    model = tf.keras.models.load_model(OUTPUT / f"{VERSION}.keras", compile=False)
    choices = {"ALL_LOCAL": np.zeros(len(holdout), dtype=int), "ALL_CLOUD": np.ones(len(holdout), dtype=int),
               "RULE_BASED_LOCAL_CLOUD": np.asarray([select(r["state"], config) for r in holdout]),
               "LEARNED_LOCAL_CLOUD": np.argmax(model(np.asarray([r["policy_input"] for r in holdout], dtype=np.float32), training=False).numpy(), axis=1)}
    report = {"version": "phase9-policy-comparison-v1", "rule_config": config,
              "holdout_results": {name: evaluate(holdout, actions) for name, actions in choices.items()},
              "by_provenance": {}, "limitation": "Zero CLOUD-oracle support in holdout. No claim of cloud-benefit generalization or final real-world accuracy."}
    for kind in ("measured", "simulated"):
        indices = [i for i, r in enumerate(holdout) if r["provenance"]["kind"] == kind]
        report["by_provenance"][kind] = {name: evaluate([holdout[i] for i in indices], actions[indices]) for name, actions in choices.items()}
    write_json(OUTPUT / "rule_based_config.json", config)
    write_json(ROOT / "docs/evidence/phase9_policy_comparison.json", report)
    print(config)
    print({k: {m: v[m] for m in ("accuracy", "mean_reward", "mean_regret")} for k, v in report["holdout_results"].items()})


if __name__ == "__main__":
    run()
