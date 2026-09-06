"""Minimal JSON schema validation for Phase 6 / M7 MQTT transport."""

from __future__ import annotations

import json
from typing import Any


def parse_inference_request(payload: bytes) -> dict[str, Any]:
    """Parse and validate the minimal Phase-6 MQTT request envelope."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload is not valid UTF-8") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("payload is not valid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("request payload must be a JSON object")

    required_fields = (
        "request_id",
        "device_id",
        "timestamp_ms",
    )

    for field in required_fields:
        if field not in data:
            raise ValueError(f"missing required field: {field}")

    if not isinstance(data["request_id"], str) or not data["request_id"].strip():
        raise ValueError("request_id must be a non-empty string")

    if not isinstance(data["device_id"], str) or not data["device_id"].strip():
        raise ValueError("device_id must be a non-empty string")

    timestamp_ms = data["timestamp_ms"]
    if (
        isinstance(timestamp_ms, bool)
        or not isinstance(timestamp_ms, int)
        or timestamp_ms < 0
    ):
        raise ValueError("timestamp_ms must be a non-negative integer")

    return data
