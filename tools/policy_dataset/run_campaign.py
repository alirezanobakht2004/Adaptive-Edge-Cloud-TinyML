"""Replay labeled validation features through physical ESP32 benchmark executors.

python -m tools.policy_dataset.run_campaign --samples 25 --mode cloud --output <new-directory>
Flash test_phase9_campaign first. No simulated execution or learned selection.
"""
import argparse
import hashlib
import json
import subprocess
import importlib.metadata
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import serial
from ml.features.extractor import load_feature_split
from .campaign_server import CampaignServer, utc_now
from .device_trace import parse_trace, trace_to_record
from .schema import VERSION, MODEL_VERSION, FEATURE_VERSION
from .serializer import serialize

MODES = {"local": 0, "split1": 1, "split2": 2, "cloud": 3}


def balanced_indices(labels, count):
    groups = [np.flatnonzero(labels == i).tolist() for i in range(5)]
    ordered = [group[j] for j in range(max(map(len, groups))) for group in groups if j < len(group)]
    if not 1 <= count <= len(ordered):
        raise ValueError(f"samples must be 1..{len(ordered)}; no silent repeated windows")
    return ordered[:count]


def read_until(device, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = device.readline().decode("utf-8", errors="replace").strip()
        if predicate(line):
            return line
    raise TimeoutError("ESP32 campaign response timed out; partial output retained")


def run(args):
    features = load_feature_split("validation")
    indices = balanced_indices(features.labels, args.samples)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"dataset_version": VERSION, "source_dataset_version": "dataset-v1", "feature_version": FEATURE_VERSION,
              "model_version": MODEL_VERSION, "timestamp": utc_now(), "run_id": uuid4().hex,
              "mode": args.mode, "action": MODES[args.mode], "samples_requested": args.samples,
              "port": args.port, "broker_host": args.broker_host, "broker_port": args.broker_port,
              "input_session": features.session, "window_indices": indices,
              "input_kind": "existing labeled validation feature replay on physical ESP32",
              "feature_matrix_sha256": hashlib.sha256(features.features.tobytes()).hexdigest(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "seed_rule": "42000 + validation window index", "status": "running", "samples_collected": 0}
    config_path = output / "configuration.json"
    source_files = [Path(__file__), Path(__file__).with_name("campaign_server.py"),
                    Path(__file__).with_name("device_trace.py"), Path("server/app/cloud_full.py"),
                    Path("firmware/test/test_phase9_campaign/test_main.cpp")]
    config["execution_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    firmware_binary = Path("firmware/.pio/build/esp32-s3-n8r2/firmware.bin")
    config["firmware_binary_sha256"] = hashlib.sha256(firmware_binary.read_bytes()).hexdigest() if firmware_binary.exists() else None
    config["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    config["dependencies"] = {name: importlib.metadata.version(name) for name in ("tensorflow", "numpy", "paho-mqtt", "pyserial")}
    def save_config():
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_config()
    try:
        with CampaignServer(args.broker_host, args.broker_port, output / "server_events.jsonl") as server:
            config["model_versions"] = {str(k): v.model_version for k, v in server.splits.runtimes.items() if k < 3}
            config["model_versions"]["3"] = server.cloud.model_version
            hashes = {**server.cloud.artifact_hashes,
                      **{f"split{k}_tail": v.model_sha256 for k, v in server.splits.runtimes.items() if k < 3}}
            config["artifact_hashes"] = hashes
            save_config()
            device = serial.Serial(port=None, baudrate=115200, timeout=1)
            device.dtr = False
            device.rts = False
            device.port = args.port
            with device:
                device.write(b"PING\n")
                read_until(device, lambda line: line == "POLICY_CAMPAIGN_READY", 45)
                with (output / f"{VERSION}.jsonl").open("w", encoding="utf-8") as records, (output / "raw_traces.jsonl").open("w", encoding="utf-8") as raw:
                    for index in indices:
                        values = " ".join(format(float(v), ".9g") for v in features.features[index])
                        command = f"RUN {index} {MODES[args.mode]} {values}\n"
                        device.write(command.encode("ascii"))
                        trace = parse_trace(read_until(device, lambda line: line.startswith("POLICY_TRACE "), 40))
                        if trace["window_id"] != index or trace["action"] != MODES[args.mode]:
                            raise ValueError("Device trace does not match commanded window/action")
                        record = trace_to_record(trace, run_id=config["run_id"], device_id="esp32-s3-n8r2",
                            timestamp=utc_now(), session_id=features.session, true_class=int(features.labels[index]), hashes=hashes)
                        if trace["action"] and trace["status"] == "success":
                            event = server.events.get(trace["request_id"])
                            if event is None:
                                raise ValueError("Successful device trace lacks server event")
                            if (event["response"]["action"] != trace["action"] or event["publish_rc"] != 0
                                    or event["response"]["predicted_class_id"] != trace["prediction"]
                                    or event["response_bytes"] != trace["response_bytes"]
                                    or event["request_bytes"] != trace["request_bytes"]):
                                raise ValueError("Device/server result mismatch")
                            for key in ("request_receive_time", "response_publish_time"):
                                record["measurements"][key] = event[key]
                        raw.write(json.dumps(trace, allow_nan=False) + "\n"); raw.flush()
                        records.write(serialize(record)); records.flush()
                        config["samples_collected"] += 1
                        save_config()
                        print(f"{args.mode} window={index} status={trace['status']} class={trace['prediction']}", flush=True)
            config["server_errors"] = server.errors
            config["status"] = "complete"
    except Exception as exc:
        config.update(status="failed", error=str(exc))
        raise
    finally:
        config["finished_at"] = utc_now()
        save_config()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
