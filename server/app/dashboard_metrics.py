"""Read-side helpers for the Phase 11 live dashboard.

Only persisted, measured fields are summarized. Missing latency/RSSI/energy fields stay
missing rather than being inferred or relabeled.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import ceil
from typing import Any, Iterable

from .models import DevicePoseEvent, InferenceEvent

DASHBOARD_API_VERSION = "dashboard-api-r1-v2"
DASHBOARD_UI_VERSION = "dashboard-ui-r1-v2"


def event_to_dict(event: InferenceEvent, *, include_raw: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": event.id,
        "received_at": event.received_at.isoformat() if isinstance(event.received_at, datetime) else str(event.received_at),
        "telemetry_schema_version": event.telemetry_schema_version,
        "device_id": event.device_id,
        "request_id": event.request_id,
        "timestamp_ms": event.timestamp_ms,
        "window_id": event.window_id,
        "predicted_class": event.predicted_class,
        "predicted_class_id": event.predicted_class_id,
        "true_label": event.true_label,
        "confidence": event.confidence,
        "uncertainty": event.uncertainty,
        "requested_action": event.requested_action,
        "execution_mode": event.execution_mode,
        "split_point": event.split_point,
        "failover": event.failover,
        "failure_reason": event.failure_reason,
        "failure_stage": event.failure_stage,
        "wifi_connected": event.wifi_connected,
        "mqtt_connected": event.mqtt_connected,
        "rssi": event.rssi,
        "rtt_ms": event.rtt_ms,
        "rtt_source": event.rtt_source,
        "rtt_age_ms": event.rtt_age_ms,
        "free_heap_bytes": event.free_heap_bytes,
        "free_heap_ratio": event.free_heap_ratio,
        "energy_budget": event.energy_budget,
        "local_inference_ms": event.local_inference_ms,
        "request_elapsed_ms": event.request_elapsed_ms,
        "edge_compute_ms": event.edge_compute_ms,
        "network_ms": event.network_ms,
        "server_compute_ms": event.server_compute_ms,
        "total_latency_ms": event.total_latency_ms,
        "bytes_tx": event.bytes_tx,
        "bytes_rx": event.bytes_rx,
        "model_version": event.model_version,
        "policy_version": event.policy_version,
        "firmware_version": event.firmware_version,
        "success": event.success,
        "controlled": event.controlled,
    }
    if include_raw:
        result["raw_event"] = event.raw_event
    return result




def pose_to_dict(pose: DevicePoseEvent, *, include_raw: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": pose.id,
        "received_at": pose.received_at.isoformat() if isinstance(pose.received_at, datetime) else str(pose.received_at),
        "pose_schema_version": pose.pose_schema_version,
        "pose_id": pose.pose_id,
        "device_id": pose.device_id,
        "timestamp_ms": pose.timestamp_ms,
        "sequence": pose.sequence,
        "roll_deg_est": pose.roll_deg_est,
        "pitch_deg_est": pose.pitch_deg_est,
        "yaw_rel_deg_est": pose.yaw_rel_deg_est,
        "estimator_version": pose.estimator_version,
        "orientation_version": pose.orientation_version,
        "firmware_version": pose.firmware_version,
        "source": pose.source,
        "yaw_reference": pose.yaw_reference,
    }
    if include_raw:
        result["raw_event"] = pose.raw_event
    return result

def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def _numeric(events: Iterable[InferenceEvent], field: str) -> list[float]:
    values: list[float] = []
    for event in events:
        value = getattr(event, field)
        if value is not None:
            values.append(float(value))
    return values


def build_summary(events: list[InferenceEvent]) -> dict[str, Any]:
    """Build a bounded-window summary without inventing unavailable measurements."""

    actions = Counter(event.execution_mode for event in events)
    gestures = Counter(event.predicted_class for event in events)
    failover_reasons = Counter(event.failure_reason for event in events if event.failover)

    local_latency = _numeric(events, "local_inference_ms")
    request_elapsed = _numeric(events, "request_elapsed_ms")
    server_compute = _numeric(events, "server_compute_ms")
    rtt = _numeric(events, "rtt_ms")
    confidence = _numeric(events, "confidence")
    uncertainty = _numeric(events, "uncertainty")
    free_heap = _numeric(events, "free_heap_bytes")

    count = len(events)
    latest = events[0] if events else None

    return {
        "api_version": DASHBOARD_API_VERSION,
        "window_event_count": count,
        "latest_event_id": latest.id if latest else None,
        "latest_received_at": (
            latest.received_at.isoformat() if latest and isinstance(latest.received_at, datetime) else None
        ),
        "action_counts": {"LOCAL": actions.get("LOCAL", 0), "CLOUD": actions.get("CLOUD", 0)},
        "gesture_counts": dict(sorted(gestures.items())),
        "failover_count": sum(1 for event in events if event.failover),
        "failover_reasons": dict(sorted(failover_reasons.items())),
        "success_count": sum(1 for event in events if event.success),
        "controlled_count": sum(1 for event in events if event.controlled),
        "confidence_mean": _mean(confidence),
        "uncertainty_mean": _mean(uncertainty),
        "local_inference_ms": {"mean": _mean(local_latency), "p95": _percentile(local_latency, 95)},
        "request_elapsed_ms": {"mean": _mean(request_elapsed), "p95": _percentile(request_elapsed, 95)},
        "server_compute_ms": {"mean": _mean(server_compute), "p95": _percentile(server_compute, 95)},
        "rtt_ms": {"mean": _mean(rtt), "p95": _percentile(rtt, 95)},
        "free_heap_bytes": {"mean": _mean(free_heap), "latest": float(latest.free_heap_bytes) if latest else None},
        "bytes_tx_total": sum(int(event.bytes_tx) for event in events),
        "bytes_rx_total": sum(int(event.bytes_rx) for event in events),
        "unmeasured_fields": [
            "energy_budget",
            "rssi",
            "free_heap_ratio",
            "edge_compute_ms",
            "network_ms",
            "total_latency_ms",
        ],
    }
