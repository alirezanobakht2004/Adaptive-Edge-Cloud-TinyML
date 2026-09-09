"""Versioned R1 decision telemetry contract for Phase 11 persistence/dashboard work."""

from __future__ import annotations

import json
import math
import re
from typing import Any

DECISION_SCHEMA_VERSION = "decision-r1-v1"
TELEMETRY_TOPIC = "gesture/+/telemetry"
GESTURE_CLASSES = (
    "IDLE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "ROTATE_CW",
    "SHAKE",
)
_ALLOWED_REQUESTED_ACTIONS = {"LOCAL", "CLOUD", "NOT_EVALUATED"}
_ALLOWED_EFFECTIVE_ACTIONS = {"LOCAL", "CLOUD"}
_ALLOWED_FAILOVER_REASONS = {
    "NONE",
    "WIFI_UNAVAILABLE",
    "MQTT_UNAVAILABLE",
    "CLOUD_RESPONSE_TIMEOUT",
    "CLOUD_PUBLISH_FAILED",
    "LOCAL_QUEUE_SAFETY",
}


def _finite_number(data: dict[str, Any], key: str, *, minimum: float | None = None,
                   maximum: float | None = None, nullable: bool = False) -> float | None:
    value = data[key]
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{key} must be finite numeric" + (" or null" if nullable else ""))
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{key} is below its minimum")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{key} is above its maximum")
    return normalized


def parse_decision_event(payload: bytes | str | dict[str, Any]) -> dict[str, Any]:
    """Parse one final LOCAL/CLOUD decision event.

    The contract intentionally carries final decision/result telemetry only. It does
    not carry raw 100x6 IMU windows, split embeddings, or features-v1 values.
    """

    try:
        data = json.loads(payload) if isinstance(payload, (bytes, str)) else dict(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("decision telemetry is not valid JSON") from exc

    required = {
        "schema_version",
        "request_id",
        "device_id",
        "timestamp_ms",
        "window_id",
        "requested_action",
        "effective_action",
        "failover",
        "failover_reason",
        "failure_stage",
        "wifi_connected",
        "mqtt_connected",
        "predicted_class_id",
        "confidence",
        "uncertainty",
        "rtt_ms",
        "rtt_source",
        "rtt_age_ms",
        "free_heap_bytes",
        "local_inference_ms",
        "request_elapsed_ms",
        "server_compute_ms",
        "bytes_tx",
        "bytes_rx",
        "model_version",
        "policy_version",
        "firmware_version",
        "success",
        "controlled",
    }
    if set(data) != required:
        raise ValueError("R1 decision telemetry required fields mismatch")

    if data["schema_version"] != DECISION_SCHEMA_VERSION:
        raise ValueError("Invalid decision telemetry schema_version")

    for key in ("request_id", "device_id", "model_version", "policy_version", "firmware_version"):
        value = data[key]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value):
            raise ValueError(f"Invalid decision telemetry {key}")

    if data["requested_action"] not in _ALLOWED_REQUESTED_ACTIONS:
        raise ValueError("Invalid requested_action")
    if data["effective_action"] not in _ALLOWED_EFFECTIVE_ACTIONS:
        raise ValueError("Invalid effective_action")
    if data["failover_reason"] not in _ALLOWED_FAILOVER_REASONS:
        raise ValueError("Invalid failover_reason")
    if not isinstance(data["failure_stage"], str) or not re.fullmatch(r"[A-Z0-9_]{1,48}", data["failure_stage"]):
        raise ValueError("Invalid failure_stage")
    if not isinstance(data["rtt_source"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", data["rtt_source"]):
        raise ValueError("Invalid rtt_source")

    for key in ("timestamp_ms", "window_id", "rtt_age_ms", "free_heap_bytes", "bytes_tx", "bytes_rx"):
        value = data[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")

    predicted = data["predicted_class_id"]
    if type(predicted) is not int or not 0 <= predicted < len(GESTURE_CLASSES):
        raise ValueError("predicted_class_id must be in the canonical class range")

    for key in ("failover", "wifi_connected", "mqtt_connected", "success", "controlled"):
        if type(data[key]) is not bool:
            raise ValueError(f"{key} must be boolean")

    data["confidence"] = _finite_number(data, "confidence", minimum=0.0, maximum=1.0)
    data["uncertainty"] = _finite_number(data, "uncertainty", minimum=0.0, maximum=1.0)
    data["local_inference_ms"] = _finite_number(data, "local_inference_ms", minimum=0.0)
    data["rtt_ms"] = _finite_number(data, "rtt_ms", minimum=0.0, nullable=True)
    data["request_elapsed_ms"] = _finite_number(data, "request_elapsed_ms", minimum=0.0, nullable=True)
    data["server_compute_ms"] = _finite_number(data, "server_compute_ms", minimum=0.0, nullable=True)

    if data["failover"] and data["effective_action"] != "LOCAL":
        raise ValueError("failover decisions must finish LOCAL")
    if not data["failover"] and data["failover_reason"] != "NONE":
        raise ValueError("non-failover decision cannot carry a failover reason")
    if data["effective_action"] == "CLOUD" and not data["success"]:
        raise ValueError("effective CLOUD telemetry must represent a successful cloud result")

    return data


def class_name(class_id: int) -> str:
    return GESTURE_CLASSES[class_id]
