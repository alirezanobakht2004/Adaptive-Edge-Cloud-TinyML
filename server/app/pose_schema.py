"""Versioned sensor-driven device-pose contract for the Phase-11 3D dashboard twin.

This auxiliary stream is visualization telemetry only. It is not a policy input and it
never carries raw 100x6 windows or features-v1 values.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

POSE_SCHEMA_VERSION = "pose-v1"
POSE_TOPIC = "gesture/+/pose"
ESTIMATOR_VERSION = "attitude-complementary-v1"
ORIENTATION_VERSION = "orientation-v1"
POSE_SOURCE = "mpu6050-6axis"
YAW_REFERENCE = "boot-relative"

_POSE_ID_RE = re.compile(r"^r1-pose-[0-9a-fA-F]{8}-[0-9]+$")
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def _finite_number(data: dict[str, Any], key: str, *, minimum: float, maximum: float) -> float:
    value = data[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{key} must be finite numeric")
    normalized = float(value)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{key} is outside its allowed range")
    return normalized


def parse_pose_event(payload: bytes | str | dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(payload) if isinstance(payload, (bytes, str)) else dict(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("pose telemetry is not valid JSON") from exc

    required = {
        "schema_version",
        "pose_id",
        "device_id",
        "timestamp_ms",
        "sequence",
        "roll_deg_est",
        "pitch_deg_est",
        "yaw_rel_deg_est",
        "estimator_version",
        "orientation_version",
        "firmware_version",
        "source",
        "yaw_reference",
    }
    if set(data) != required:
        raise ValueError("pose telemetry required fields mismatch")

    if data["schema_version"] != POSE_SCHEMA_VERSION:
        raise ValueError("unsupported pose schema_version")
    if not isinstance(data["pose_id"], str) or not _POSE_ID_RE.fullmatch(data["pose_id"]):
        raise ValueError("invalid pose_id")
    if not isinstance(data["device_id"], str) or not _DEVICE_ID_RE.fullmatch(data["device_id"]):
        raise ValueError("invalid device_id")
    if type(data["timestamp_ms"]) is not int or data["timestamp_ms"] < 0:
        raise ValueError("timestamp_ms must be a non-negative integer")
    if type(data["sequence"]) is not int or data["sequence"] < 1:
        raise ValueError("sequence must be a positive integer")

    data["roll_deg_est"] = _finite_number(data, "roll_deg_est", minimum=-180.0, maximum=180.0)
    data["pitch_deg_est"] = _finite_number(data, "pitch_deg_est", minimum=-180.0, maximum=180.0)
    data["yaw_rel_deg_est"] = _finite_number(data, "yaw_rel_deg_est", minimum=-180.0, maximum=180.0)

    if data["estimator_version"] != ESTIMATOR_VERSION:
        raise ValueError("unsupported estimator_version")
    if data["orientation_version"] != ORIENTATION_VERSION:
        raise ValueError("unsupported orientation_version")
    if not isinstance(data["firmware_version"], str) or not 1 <= len(data["firmware_version"]) <= 96:
        raise ValueError("invalid firmware_version")
    if data["source"] != POSE_SOURCE:
        raise ValueError("unsupported pose source")
    if data["yaw_reference"] != YAW_REFERENCE:
        raise ValueError("unsupported yaw_reference")

    return data
