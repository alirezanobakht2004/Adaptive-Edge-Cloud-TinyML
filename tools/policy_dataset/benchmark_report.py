"""Measured fixed-baseline summary, with unexecuted rule baseline explicitly missing."""
import argparse
import json
import hashlib
from pathlib import Path
from ml.policy.benchmark import summarize
from .validator import read_records


def generate_benchmark(input_path, output_path, *, scope="device"):
    summary = summarize(read_records(input_path), scope=scope)
    rows = {name: {**summary[action], "status": "measured" if summary[action]["attempts"] else "missing"}
            for name, action in (("ALL_LOCAL", 0), ("ALL_CLOUD", 3), ("SPLIT1", 1), ("SPLIT2", 2))}
    rows["RULE_BASED_ADAPTIVE"] = {**{key: (0 if key.endswith("count") or key in ("attempts", "successes") else None)
                                    for key in summary[0]}, "status": "missing",
                                  "reason": "No rule-based execution campaign; fixed-action traces are not substitute measurements."}
    report = {"scope": scope, "baselines": rows, "policy_trained": False,
              "input_sha256_lf_utf8": hashlib.sha256(Path(input_path).read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
              "metric_definitions": {
                  "accuracy": "Correct successful predictions / labeled attempts, only this replay subset.",
                  "mean_latency_ms": "total_latency_ms as recorded; Phase9 device campaigns include preparation + pre-decision B3/MC + MQTT probe + selected action, excluding serial transfer and feature extraction.",
                  "mean_communication_bytes": "Selected action JSON request + response payload bytes; excludes probe, MQTT framing and link overhead.",
                  "mean_estimated_energy_proxy": "Null until explicit estimated/simulated proxy assumptions are configured; never measured energy."},
              "limitations": ["Fixed modes execute sequentially; uncontrolled host/network conditions and cold starts are included.",
                              "Existing validation features replayed on device; not new live sensor acquisition.",
                              "Local stochastic edge head and separately trained cloud tails differ; results reflect model and placement.",
                              "No rule-based campaign, network sweep, or training qualification claimed."]}
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("device", "host", "replay"), default="device")
    args = parser.parse_args()
    generate_benchmark(args.input, args.output, scope=args.scope)


if __name__ == "__main__":
    main()
