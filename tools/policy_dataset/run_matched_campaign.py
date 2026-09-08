"""Collect a fresh frozen-state ESP32 campaign; never regroup v1 historical runs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from .matched_dataset import VERSION, build_sample, validate_condition, validate_file
from .reward_engine.__main__ import strict_json
from .reward_engine.engine import digest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def correlate_event(trace, server):
    if not trace["action"] or trace["status"] != "success":
        return None
    deadline = time.monotonic() + 2
    while trace["request_id"] not in server.events and time.monotonic() < deadline:
        time.sleep(.01)
    event = server.events.get(trace["request_id"])
    if event is None:
        raise ValueError("Remote result has no server evidence")
    response = event["response"]
    expected_version = server.cloud.model_version if trace["action"] == 3 else server.splits.runtimes[trace["action"]].model_version
    if (event["publish_rc"] != 0 or response["action"] != trace["action"]
            or response["model_version"] != expected_version
            or response["predicted_class_id"] != trace["prediction"]
            or abs(response["confidence"] - trace["result_confidence"]) > 1e-6
            or abs(response["server_latency_ms"] - trace["cloud_ms"]) > 1e-5
            or event["request_bytes"] != trace["request_bytes"]
            or event["response_bytes"] != trace["response_bytes"]):
        raise ValueError("Device/server action, prediction, version or byte mismatch")
    return event


def run(args):
    # Defer expensive TensorFlow/serial imports until configuration is valid.
    condition = strict_json(Path(args.condition).read_text(encoding="utf-8"))
    validate_condition(condition)
    if args.repeats not in (4, 8, 12, 16):
        raise ValueError("repeats must be 4, 8, 12 or 16")
    from ml.features.extractor import load_feature_split
    from .run_campaign import balanced_indices, read_until
    from .campaign_server import CampaignServer, utc_now
    from .device_trace import parse_trace, trace_to_record
    import serial
    import importlib.metadata

    features = load_feature_split("validation")
    indices = balanced_indices(features.labels, args.samples)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": "matched-campaign-v1", "dataset_version": VERSION, "condition": condition,
                "timestamp": utc_now(), "run_id": uuid4().hex, "samples_requested": args.samples,
                "repeats": args.repeats, "window_indices": indices, "source_session": features.session,
                "feature_matrix_sha256": hashlib.sha256(features.features.tobytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(features.labels.tobytes()).hexdigest(),
                "port": args.port, "broker_host": args.broker_host, "broker_port": args.broker_port,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                "matched_samples": 0, "rejected_windows": [], "status": "running",
                "state_protocol": "One physical snapshot per window; four cyclic orders; live drift guards",
                "training_enabled": False}
    source_paths = [Path(__file__), Path(__file__).with_name("matched_dataset.py"),
                    Path(__file__).with_name("device_trace.py"), Path(__file__).with_name("campaign_server.py"),
                    Path(__file__).parent / "reward_engine/engine.py",
                    Path("firmware/test/test_phase10_matched_campaign/test_main.cpp")]
    manifest["source_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    firmware = Path("firmware/.pio/build/esp32-s3-n8r2/firmware.bin")
    manifest["local_firmware_binary_sha256"] = hashlib.sha256(firmware.read_bytes()).hexdigest() if firmware.exists() else None
    manifest["dependencies"] = {k: importlib.metadata.version(k) for k in ("tensorflow", "numpy", "paho-mqtt", "pyserial")}
    save = lambda: write_json(output / "configuration.json", manifest)
    save()
    try:
        with CampaignServer(args.broker_host, args.broker_port, output / "server_events.jsonl") as server:
            hashes = {**server.cloud.artifact_hashes,
                      **{f"split{k}_tail": v.model_sha256 for k, v in server.splits.runtimes.items() if k < 3}}
            manifest["artifact_hashes"] = hashes
            manifest["model_versions"] = {"0": "gesture-model-v1.1.0/local-stochastic-head",
                **{str(k): v.model_version for k, v in server.splits.runtimes.items() if k < 3}, "3": server.cloud.model_version}
            save()
            device = serial.Serial(port=None, baudrate=115200, timeout=1)
            device.dtr = False
            device.rts = False
            device.port = args.port
            with device, (output / "raw_entries.jsonl").open("w", encoding="utf-8") as raw, (output / f"{VERSION}.jsonl").open("w", encoding="utf-8") as dataset:
                device.write(b"PING\n")
                read_until(device, lambda line: line == "MATCHED_CAMPAIGN_READY", 45)
                for ordinal, index in enumerate(indices):
                    values = " ".join(format(float(v), ".9g") for v in features.features[index])
                    device.write(f"MATCH {index} {ordinal % 4} {args.repeats} {values}\n".encode("ascii"))
                    entries = []
                    for _ in range(args.repeats * 4):
                        guard = strict_json(read_until(device, lambda line: line.startswith("MATCH_META "), 45)[11:])
                        trace = parse_trace(read_until(device, lambda line: line.startswith("POLICY_TRACE "), 5))
                        # Retain bytes even if subsequent correlation or validation fails.
                        raw.write(json.dumps({"trace": trace, "guard": guard}, allow_nan=False) + "\n")
                        raw.flush()
                        if trace["window_id"] != index:
                            raise ValueError("Unexpected device window")
                        record = trace_to_record(trace, run_id=manifest["run_id"], device_id="esp32-s3-n8r2",
                            timestamp=utc_now(), session_id=features.session, true_class=int(features.labels[index]), hashes=hashes)
                        record["metadata"]["sample_id"] += f"-repeat{guard['repeat']}-action{trace['action']}"
                        record["provenance"]["state_source"] = "ESP32 single frozen M30 pre-decision snapshot"
                        record["provenance"]["notes"] = "Shared snapshot overhead plus selected action time; guard/serial/between-action time excluded; live guards retained in v2."
                        event = correlate_event(trace, server)
                        if event:
                            for key in ("request_receive_time", "response_publish_time"):
                                record["measurements"][key] = event[key]
                        entries.append({"trace": trace, "guard": guard, "record": record})
                    read_until(device, lambda line: line == "MATCH_DONE", 5)
                    try:
                        sample = build_sample(entries, condition, f"{manifest['run_id']}/{features.session}/{index}")
                    except ValueError as exc:
                        manifest["rejected_windows"].append({"window_id": index, "reason": str(exc)})
                        with (output / "rejected_entries.jsonl").open("a", encoding="utf-8") as rejected:
                            rejected.write(json.dumps(entries, allow_nan=False) + "\n")
                        print(f"REJECT window={index}: {exc}", flush=True)
                    else:
                        dataset.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
                        dataset.flush()
                        manifest["matched_samples"] += 1
                        print(f"MATCH window={index} label={sample['label']['optimal_action']} repeats={args.repeats}", flush=True)
                    save()
            manifest["server_errors"] = server.errors
            if server.errors:
                raise ValueError("Server errors; campaign cannot be accepted")
        report = validate_file(output / f"{VERSION}.jsonl")
        report["rejected_windows"] = manifest["rejected_windows"]
        write_json(output / "dataset_report.json", report)
        manifest["status"] = "complete" if not manifest["rejected_windows"] else "complete_with_rejections"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True, help="Versioned condition/calibration JSON")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--broker-host", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=1883)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
