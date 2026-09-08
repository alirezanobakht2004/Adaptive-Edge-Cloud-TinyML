"""Synthetic unit fixtures only; these are never collected as observations."""
import pytest
from tools.policy_dataset.device_trace import REQUIRED, TRACE_VERSION, validate_trace, trace_to_record


def trace_fixture():
    trace = dict.fromkeys(REQUIRED, 0)
    trace.update(trace_version=TRACE_VERSION, request_id="unit-test", connected=True, status="success", confidence=.8,
                 result_confidence=.8, psram_total=100, psram_free=50, total_us=1000, action_us=500,
                 state_us=100, prep_us=10, cpu_mhz=240, free_heap=10000)
    return trace


def test_device_record_preserves_measurements_and_unknown_energy():
    trace = trace_fixture()
    record = trace_to_record(trace, run_id="unit-test", device_id="fixture", timestamp="2026-09-08T00:00:00+00:00",
                             session_id="unit-test", true_class=0, hashes={})
    assert record["measurements"]["total_latency_ms"] == 1
    assert record["metadata"]["measurement_scope"] == "device"
    assert record["reward"]["energy_component"] is None
    assert record["measurements"]["mqtt_send_timestamp"] is None


@pytest.mark.parametrize("key,value", [("action", True), ("prediction", 5), ("confidence", 1.1),
    ("psram_free", 101), ("window_id", 1.5), ("total_us", 1), ("entropy", float("nan"))])
def test_invalid_trace_rejected(key, value):
    trace = trace_fixture()
    trace[key] = value
    with pytest.raises(ValueError):
        validate_trace(trace)
