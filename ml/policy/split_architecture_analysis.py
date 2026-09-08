"""Read-only split artifact and hypothetical bottleneck study; no model creation."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import zipfile

from tools.policy_dataset.reward_engine.engine import digest, number
from tools.policy_dataset.run_controlled_campaign import DEFAULT_SOURCE, read_rows
from tools.policy_dataset.run_matched_campaign import write_json
from tools.policy_dataset.validate_matched_dataset import validate_campaign

MODEL_ROOT = Path("data/processed/dataset-v1/features-v1/models")
VERSIONS = ("gesture-model-v1.1.0", "gesture-cloud-tail-split1-v1.0.0",
            "gesture-cloud-tail-split2-v1.0.0", "gesture-cloud-tail-v1.0.0")


def dense_inventory(path):
    """Inspect saved Keras graph metadata without loading/training a model."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        model = json.loads(archive.read("config.json"))
    layers = []
    for layer in model["config"]["layers"]:
        if layer["class_name"] == "Dense":
            inputs = layer["build_config"]["input_shape"][-1]
            outputs = layer["config"]["units"]
            if type(inputs) is not int or type(outputs) is not int or min(inputs, outputs) <= 0:
                raise ValueError("Invalid Dense artifact shape")
            layers.append({"name": layer["config"]["name"], "input_dim": inputs,
                           "output_dim": outputs, "dense_macs": inputs * outputs})
    if not layers:
        raise ValueError("No Dense layers in artifact")
    return {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "dense_layers": layers}


def candidate_placement(dimension, artifacts):
    if type(dimension) is not int or dimension not in (16, 24, 32, 48):
        raise ValueError("Candidate dimension must be 16, 24, 32 or 48")
    blocks = artifacts[0]["dense_layers"][:3]
    if dimension in (16, 24):
        encoder_macs = 64 * dimension
        return {"kind": "hypothetical_B1_encoder_decoder", "requires_new_parameters": True,
                "edge": f"B1(10->64), hypothetical linear encoder(64->{dimension})",
                "cloud": f"hypothetical decoder({dimension}->64), existing Split1 tail",
                "edge_dense_macs": blocks[0]["dense_macs"] + encoder_macs,
                "cloud_dense_macs": encoder_macs + sum(l["dense_macs"] for l in artifacts[1]["dense_layers"]),
                "additional_dense_macs": 2 * encoder_macs,
                "quality": "unknown; no rank reduction can be assumed lossless; no encoder/decoder built"}
    stop = 3 if dimension == 32 else 2
    tail = artifacts[3] if dimension == 32 else artifacts[2]
    return {"kind": "existing_B3_boundary" if dimension == 32 else "existing_B2_control",
            "requires_new_parameters": False, "edge": f"B1 through B{stop}",
            "cloud": "existing Split3 tail" if dimension == 32 else "existing Split2 tail",
            "edge_dense_macs": sum(b["dense_macs"] for b in blocks[:stop]),
            "cloud_dense_macs": sum(l["dense_macs"] for l in tail["dense_layers"]),
            "additional_dense_macs": 0,
            "quality": "existing artifacts; no new candidate benchmark or accuracy measurement"}


def payload_model(bytes64, bytes48):
    number(bytes64, "64-d payload", positive=True)
    number(bytes48, "48-d payload", positive=True)
    slope = (bytes64 - bytes48) / 16
    intercept = bytes64 - 64 * slope
    if slope <= 0 or intercept < 0:
        raise ValueError("Source means do not support a positive affine payload estimate")
    return {"bytes_per_dimension": slope, "intercept_bytes": intercept,
            "kind": "two-point descriptive fit, not measured candidate JSON size",
            "assumptions": "source value formatting/sparsity and metadata overhead extrapolate; response differences absorbed"}


def payload_reward_change(delta_bytes, config, proxy_byte_reference):
    number(proxy_byte_reference, "proxy byte reference", positive=True)
    weights, scales = config["weights"], config["scales"]
    coefficient = (weights["communication"] / scales["communication_bytes"]
                   + weights["energy_proxy"] / scales["energy_proxy"] / proxy_byte_reference)
    return -delta_bytes * coefficient


def run(output, model_root=MODEL_ROOT):
    model_root = Path(model_root)
    artifacts = [dense_inventory(model_root / version / f"{version}.keras") for version in VERSIONS]
    expected = [[(10, 64), (64, 48), (48, 32), (32, 5)],
                [(64, 48), (48, 32), (32, 5)], [(48, 32), (32, 64), (64, 32), (32, 5)],
                [(32, 64), (64, 32), (32, 5)]]
    for artifact, shapes in zip(artifacts, expected):
        if [(l["input_dim"], l["output_dim"]) for l in artifact["dense_layers"]] != shapes:
            raise ValueError("Architecture changed; revise candidate assumptions explicitly")
    prefix_root = model_root / VERSIONS[0] / "tflite/splits"
    prefixes = []
    for split, dimension in ((1, 64), (2, 48)):
        export = json.loads((prefix_root / f"split{split}_export_report.json").read_text())
        specification = export["prefix_tflite"]
        filename = specification["filename"] if isinstance(specification, dict) else specification
        expected_sha = specification["sha256"] if isinstance(specification, dict) else export["prefix_sha256"]
        path = prefix_root / filename
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if (export["output_shape"][-1] != dimension or sha != expected_sha
                or export["source_model_sha256"] != artifacts[0]["sha256"]):
            raise ValueError("Prefix artifact/report mismatch")
        prefixes.append({"split": split, "embedding_dim": dimension, "bytes": path.stat().st_size, "sha256": sha})
    split3 = json.loads((model_root / VERSIONS[3] / "split3_export_report.json").read_text())
    b3 = model_root / VERSIONS[0] / "tflite/gesture-model-v1.1.0-prefix-b3-float32-normalized-input.tflite"
    b3_sha = hashlib.sha256(b3.read_bytes()).hexdigest()
    if split3["prefix_sha256"] != b3_sha or split3["model_sha256"] != artifacts[3]["sha256"]:
        raise ValueError("Split3 artifact/report mismatch")
    prefixes.append({"split": 3, "embedding_dim": 32, "bytes": b3.stat().st_size, "sha256": b3_sha})
    validate_campaign(DEFAULT_SOURCE)
    samples = read_rows(DEFAULT_SOURCE / "policy_training_dataset_v2.jsonl")
    config = samples[0]["condition"]["reward"]
    byte_reference = samples[0]["condition"]["energy_proxy"]["payload_reference_bytes"]
    baselines = {str(a): {k: mean(s["candidate_actions"][a][k] for s in samples) for k in
                 ("mean_latency_ms", "mean_communication_bytes", "mean_reward")} for a in range(4)}
    fit = payload_model(baselines["1"]["mean_communication_bytes"], baselines["2"]["mean_communication_bytes"])
    candidates = []
    for dimension in (16, 24, 32, 48):
        estimated_bytes = fit["intercept_bytes"] + fit["bytes_per_dimension"] * dimension
        delta = payload_reward_change(estimated_bytes - baselines["1"]["mean_communication_bytes"], config, byte_reference)
        conditional_reward = baselines["1"]["mean_reward"] + delta
        candidates.append({"embedding_dim": dimension, "float32_embedding_bytes": dimension * 4,
            "raw_bytes_ratio_to_cloud_features": dimension / 10,
            "placement": candidate_placement(dimension, artifacts),
            "estimated_json_request_response_bytes": estimated_bytes,
            "estimated_direct_communication_penalty": config["weights"]["communication"] * estimated_bytes / config["scales"]["communication_bytes"],
            "conditional_payload_reward_delta_vs_split1": delta,
            "conditional_reward_at_unchanged_split1_accuracy_latency_compute": conditional_reward,
            "conditional_reward_gap_to_local": baselines["0"]["mean_reward"] - conditional_reward,
            "conditional_reward_gap_to_cloud": baselines["3"]["mean_reward"] - conditional_reward,
            "candidate_measured_latency_ms": None, "candidate_measured_accuracy": None,
            "actual_candidate_reward": None})
    report = {"version": "split-architecture-study-v1", "artifacts": artifacts, "prefixes": prefixes,
              "measured_source_sha256": digest(samples), "source_windows": len(samples),
              "current_measured_baselines": baselines, "reward_configuration": config, "payload_estimator": fit,
              "candidates": candidates, "full_cloud_dense_macs": sum(l["dense_macs"] for l in artifacts[0]["dense_layers"][:3])
              + sum(l["dense_macs"] for l in artifacts[3]["dense_layers"]),
              "training_allowed": False, "new_artifacts_created": False,
              "limitations": ["Dense MAC counts omit biases, activations, normalization, uncertainty passes, allocations and transport",
                  "MACs are not latency or energy measurements; no MAC-to-ms conversion performed",
                  "JSON sizes are extrapolated from two existing means; no candidate tensors were invented",
                  "Payload-only reward overlay uses Split1 costs for all dimensions to isolate bytes, not candidate placement costs",
                  "16/24 require separately trained or otherwise justified new bottleneck parameters; no truncation or losslessness claim",
                  "32/48 artifact availability does not supply matched candidate measurements or alter policy action mapping"]}
    write_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/phase10_5_candidate_comparison.json"))
    args = parser.parse_args()
    run(args.output, args.model_root)


if __name__ == "__main__":
    main()
