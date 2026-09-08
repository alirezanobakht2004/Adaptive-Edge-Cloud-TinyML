"""Convert measured device traces to v1 records, retaining raw device fields in a sidecar."""

import json

from .instrumentation import PolicySample
from .schema import validate_record

TRACE_VERSION = "esp32-decision-trace-v1"
REQUIRED = {
    "trace_version", "request_id", "window_id", "action", "confidence", "entropy", "margin", "state_class",
    "free_heap", "psram_total", "psram_free", "arena_used", "cpu_mhz", "state_us", "prep_us",
    "probe_us", "connected", "rssi_dbm", "prefix_us", "action_us", "total_us", "request_bytes",
    "response_bytes", "roundtrip_us", "send_us", "receive_us", "cloud_ms", "prediction",
    "result_confidence", "status", "seed", "probe_request_bytes", "probe_response_bytes",
}


def validate_trace(trace):
    if not isinstance(trace, dict) or set(trace) != REQUIRED:
        raise ValueError("Device trace fields mismatch")
    if trace["trace_version"] != TRACE_VERSION:
        raise ValueError("Device trace version mismatch")
    if type(trace["action"]) is not int or trace["action"] not in range(4):
        raise ValueError("Invalid device action")
    if trace["status"] not in ("success", "timeout", "failed"):
        raise ValueError("Invalid trace outcome")
    if type(trace["connected"]) is not bool:
        raise ValueError("connected must be boolean")
    from .schema import _number
    if not isinstance(trace["request_id"], str) or not trace["request_id"]:
        raise ValueError("Missing request identity")
    for key in REQUIRED - {"trace_version", "request_id", "connected", "status"}:
        value = trace[key]
        if key == "rssi_dbm":
            if type(value) is not int or not -127 <= value <= 0:
                raise ValueError("Invalid RSSI")
        else:
            _number(value, key)
    if any(trace[k] is None for k in REQUIRED - {"probe_us", "cloud_ms", "prediction", "result_confidence", "send_us", "receive_us", "roundtrip_us"}):
        raise ValueError("Missing required measured trace value")
    if trace["psram_free"] > trace["psram_total"]:
        raise ValueError("Invalid PSRAM usage")
    for key in ("window_id", "state_class", "prediction", "seed", "free_heap", "psram_total", "psram_free",
                "arena_used", "cpu_mhz", "request_bytes", "response_bytes", "probe_request_bytes", "probe_response_bytes"):
        _number(trace[key], key, integer=True)
    for key in ("confidence", "margin", "result_confidence"):
        _number(trace[key], key, maximum=1)
    for key in ("state_class", "prediction"):
        _number(trace[key], key, maximum=4)
    if trace["status"] == "success" and (trace["prediction"] is None or trace["result_confidence"] is None):
        raise ValueError("Successful trace needs prediction and confidence")
    if trace["total_us"] < trace["action_us"] + trace["state_us"] + trace["prep_us"]:
        raise ValueError("Inconsistent measured durations")


def trace_to_record(trace, *, run_id, device_id, timestamp, session_id, true_class, hashes):
    validate_trace(trace)
    sample = PolicySample.create(sample_id=f"{device_id}-{trace['window_id']}", device_id=device_id,
                                 window_id=str(trace["window_id"]), run_id=run_id, session_id=session_id, scope="device")
    record = sample.record
    record["metadata"]["timestamp"] = timestamp
    record["action"] = trace["action"]
    record["state"]["uncertainty"].update(confidence=trace["confidence"], entropy=trace["entropy"], margin=trace["margin"])
    record["state"]["device"].update(free_heap=trace["free_heap"], tensor_arena_usage=trace["arena_used"],
                                      cpu_frequency=trace["cpu_mhz"], inference_latency_estimate=trace["state_us"] / 1000)
    record["state"]["network"].update(mqtt_rtt_ms=trace["probe_us"] / 1000 if trace["probe_us"] is not None else None,
                                       connection_state="connected" if trace["connected"] else "disconnected")
    record["state"]["application"]["predicted_class"] = trace["state_class"]
    record["measurements"].update(preprocessing_latency_ms=trace["prep_us"] / 1000,
        prefix_latency_ms=trace["prefix_us"] / 1000, total_latency_ms=trace["total_us"] / 1000,
        local_inference_latency_ms=trace["action_us"] / 1000 if trace["action"] == 0 else None,
        round_trip_latency_ms=trace["roundtrip_us"] / 1000 if trace["roundtrip_us"] is not None else None,
        request_bytes=trace["request_bytes"], response_bytes=trace["response_bytes"],
        embedding_size_bytes={0: 0, 1: 256, 2: 192, 3: 0}[trace["action"]], cloud_inference_latency_ms=trace["cloud_ms"])
    record["outcome"].update(status=trace["status"], predicted_class=trace["prediction"],
                             confidence=trace["result_confidence"], true_class=true_class,
                             error=None if trace["status"] == "success" else trace["status"])
    record["provenance"].update(artifact_hashes=hashes, state_source="ESP32 pre-decision B3/MC state and MQTT probe",
        clock_domains={"timestamp": "host UTC receipt", "durations": "ESP32 micros()",
                       "send_us/receive_us": "raw trace only; ESP32 boot-relative, not UTC"},
        notes="Labeled dataset-v1 feature replay on physical ESP32. Full trace sidecar contains PSRAM, seed, RSSI, "
              "action time and probe bytes. Total cost includes preparation, state inference and probe overhead. "
              "Bandwidth/packet estimate unavailable; reward/energy unconfigured.")
    validate_record(record)
    return record


def parse_trace(line):
    if not line.startswith("POLICY_TRACE "):
        raise ValueError("Not a device trace line")
    from .serializer import _object, _constant
    trace = json.loads(line[len("POLICY_TRACE "):], object_pairs_hook=_object, parse_constant=_constant)
    validate_trace(trace)
    return trace
