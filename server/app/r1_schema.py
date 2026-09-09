"""Versioned production CLOUD schema, separate from fixed-split contracts."""

import json
import math
import re

SCHEMA_VERSION = "inference-r1-v1"
CLOUD_VERSION = "gesture-full-cloud-v1.0.0"
POLICY_VERSION = "meta-policy-v1.0.0"
FIRMWARE_VERSION = "0.2.0-r1"


def encode(payload):
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_cloud_request(payload):
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else dict(payload)
    required = {"schema_version", "request_id", "device_id", "timestamp_ms", "mode", "features", "feature_version",
                "feature_encoding", "model_version", "policy_version", "firmware_version", "confidence", "entropy", "margin", "policy_state"}
    if set(data) != required:
        raise ValueError("R1 required fields mismatch")
    for key, expected in (("schema_version", SCHEMA_VERSION), ("mode", "CLOUD"), ("feature_version", "features-v1"),
                          ("feature_encoding", "normalized-float32"), ("model_version", CLOUD_VERSION), ("policy_version", POLICY_VERSION)):
        if data[key] != expected:
            raise ValueError(f"Invalid R1 {key}")
    for key in ("request_id", "device_id", "firmware_version"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", data[key]):
            raise ValueError(f"Invalid R1 {key}")
    if type(data["timestamp_ms"]) is not int or data["timestamp_ms"] < 0:
        raise ValueError("Invalid timestamp")
    for key, count in (("features", 10), ("policy_state", 6)):
        if (not isinstance(data[key], list) or len(data[key]) != count
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in data[key])):
            raise ValueError(f"{key} requires exactly {count} finite numeric values")
    for key in ("confidence", "entropy", "margin"):
        value = data[key]
        upper = math.log(5) + 1e-6 if key == "entropy" else 1
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"Invalid {key}")
    return data


def make_request(identity, features, state, timestamp_ms=0, device_id="esp32-r1"):
    request = {"schema_version": SCHEMA_VERSION, "request_id": identity, "device_id": device_id,
               "timestamp_ms": timestamp_ms, "mode": "CLOUD", "features": list(features),
               "feature_version": "features-v1", "feature_encoding": "normalized-float32", "model_version": CLOUD_VERSION,
               "policy_version": POLICY_VERSION, "firmware_version": FIRMWARE_VERSION,
               "confidence": state[0], "entropy": state[1], "margin": state[2], "policy_state": list(state)}
    return parse_cloud_request(request)
