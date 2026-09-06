"""JSON request validation for Phase 6 / M7 MQTT inference."""

from __future__ import annotations

import json
import math
from typing import Any

from ml.models.cloud_model import (
    INPUT_EMBEDDING_DIM,
    SOURCE_EDGE_MODEL_VERSION,
)


EXPECTED_SPLIT_POINT = 3


def parse_inference_request(
    payload: bytes,
) -> dict[str, Any]:
    """Parse and validate the fixed Split-3 MQTT inference request."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "payload is not valid UTF-8"
        ) from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "payload is not valid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "request payload must be a JSON object"
        )

    required_fields = (
        "request_id",
        "device_id",
        "timestamp_ms",
        "split",
        "embedding",
        "model_version",
    )

    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"missing required field: {field}"
            )

    if (
        not isinstance(data["request_id"], str)
        or not data["request_id"].strip()
    ):
        raise ValueError(
            "request_id must be a non-empty string"
        )

    if (
        not isinstance(data["device_id"], str)
        or not data["device_id"].strip()
    ):
        raise ValueError(
            "device_id must be a non-empty string"
        )

    timestamp_ms = data["timestamp_ms"]

    if (
        isinstance(timestamp_ms, bool)
        or not isinstance(timestamp_ms, int)
        or timestamp_ms < 0
    ):
        raise ValueError(
            "timestamp_ms must be a non-negative integer"
        )

    split = data["split"]

    if (
        isinstance(split, bool)
        or not isinstance(split, int)
    ):
        raise ValueError(
            "split must be an integer"
        )

    if split != EXPECTED_SPLIT_POINT:
        raise ValueError(
            "Phase-6 M7 server supports fixed Split 3 only"
        )

    model_version = data["model_version"]

    if not isinstance(model_version, str):
        raise ValueError(
            "model_version must be a string"
        )

    if model_version != SOURCE_EDGE_MODEL_VERSION:
        raise ValueError(
            "incompatible edge model version: "
            f"expected {SOURCE_EDGE_MODEL_VERSION}, "
            f"got {model_version}"
        )

    embedding = data["embedding"]

    if not isinstance(embedding, list):
        raise ValueError(
            "embedding must be a JSON array"
        )

    if len(embedding) != INPUT_EMBEDDING_DIM:
        raise ValueError(
            "embedding must contain exactly "
            f"{INPUT_EMBEDDING_DIM} values"
        )

    normalized_embedding: list[float] = []

    for index, value in enumerate(embedding):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"embedding[{index}] must be numeric"
            )

        value_float = float(value)

        if not math.isfinite(value_float):
            raise ValueError(
                f"embedding[{index}] must be finite"
            )

        normalized_embedding.append(
            value_float
        )

    data["embedding"] = normalized_embedding

    return data
