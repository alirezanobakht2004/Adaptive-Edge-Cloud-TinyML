import pytest

from server.app.telemetry_schema import DECISION_SCHEMA_VERSION, parse_decision_event


def event(**changes):
    value = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "request_id": "r1-1000-1",
        "device_id": "esp32-r1",
        "timestamp_ms": 1000,
        "window_id": 1,
        "requested_action": "LOCAL",
        "effective_action": "LOCAL",
        "failover": False,
        "failover_reason": "NONE",
        "failure_stage": "NONE",
        "wifi_connected": True,
        "mqtt_connected": True,
        "predicted_class_id": 0,
        "confidence": 0.95,
        "uncertainty": 0.1,
        "rtt_ms": 12.5,
        "rtt_source": "last_successful_probe",
        "rtt_age_ms": 200,
        "free_heap_bytes": 160000,
        "local_inference_ms": 2.1,
        "request_elapsed_ms": None,
        "server_compute_ms": None,
        "bytes_tx": 0,
        "bytes_rx": 0,
        "model_version": "gesture-model-v1.1.0",
        "policy_version": "meta-policy-v1.0.0",
        "firmware_version": "0.3.0-r1",
        "success": True,
        "controlled": False,
    }
    value.update(changes)
    return value


def test_accepts_local_decision_without_raw_or_feature_payloads():
    parsed = parse_decision_event(event())
    assert parsed["effective_action"] == "LOCAL"
    assert "features" not in parsed and "embedding" not in parsed and "raw" not in parsed


def test_accepts_cloud_result_and_failover_local_result():
    cloud = parse_decision_event(event(
        requested_action="CLOUD", effective_action="CLOUD", predicted_class_id=2,
        request_elapsed_ms=25.0, server_compute_ms=3.2, bytes_tx=600, bytes_rx=260,
    ))
    assert cloud["effective_action"] == "CLOUD"

    fallback = parse_decision_event(event(
        requested_action="CLOUD", effective_action="LOCAL", failover=True,
        failover_reason="CLOUD_RESPONSE_TIMEOUT", failure_stage="WAIT_RESPONSE",
        request_elapsed_ms=3000.0, bytes_tx=600,
    ))
    assert fallback["failover"] is True


@pytest.mark.parametrize("change", ["extra", "class", "uncertainty", "failover_action", "reason"])
def test_rejects_invalid_contract(change):
    value = event()
    if change == "extra":
        value["features"] = [0.0] * 10
    elif change == "class":
        value["predicted_class_id"] = 5
    elif change == "uncertainty":
        value["uncertainty"] = 1.1
    elif change == "failover_action":
        value.update(failover=True, failover_reason="WIFI_UNAVAILABLE", effective_action="CLOUD")
    else:
        value["failover_reason"] = "WIFI_UNAVAILABLE"
    with pytest.raises(ValueError):
        parse_decision_event(value)
