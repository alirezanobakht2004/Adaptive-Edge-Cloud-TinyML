import pytest
from server.app.database import Database
from server.app.telemetry_schema import DECISION_SCHEMA_VERSION


def event(request_id="r1-db-1", action="LOCAL"):
    is_cloud = action == "CLOUD"
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "request_id": request_id,
        "device_id": "esp32-r1",
        "timestamp_ms": 1234,
        "window_id": 7,
        "requested_action": action,
        "effective_action": action,
        "failover": False,
        "failover_reason": "NONE",
        "failure_stage": "NONE",
        "wifi_connected": True,
        "mqtt_connected": True,
        "predicted_class_id": 2 if is_cloud else 1,
        "confidence": 0.91,
        "uncertainty": 0.12,
        "rtt_ms": 14.0,
        "rtt_source": "last_successful_probe",
        "rtt_age_ms": 50,
        "free_heap_bytes": 160000,
        "local_inference_ms": 2.0,
        "request_elapsed_ms": 20.0 if is_cloud else None,
        "server_compute_ms": 3.0 if is_cloud else None,
        "bytes_tx": 600 if is_cloud else 0,
        "bytes_rx": 260 if is_cloud else 0,
        "model_version": "gesture-full-cloud-v1.0.0" if is_cloud else "gesture-model-v1.1.0",
        "policy_version": "meta-policy-v1.0.0",
        "firmware_version": "0.3.0-r1",
        "success": True,
        "controlled": False,
    }


def test_persists_and_deduplicates_inference_events():
    db = Database("sqlite+pysqlite:///:memory:")
    db.create_schema()
    first = db.save_decision_event(event())
    duplicate = db.save_decision_event(event())
    assert first.id == duplicate.id
    assert db.count_inference_events() == 1
    row = db.recent_inference_events(1)[0]
    assert row.predicted_class == "SWIPE_LEFT"
    assert row.execution_mode == "LOCAL"
    assert row.split_point is None
    assert row.energy_budget is None
    db.close()


def test_keeps_measured_and_unmeasured_fields_distinct():
    db = Database("sqlite+pysqlite:///:memory:")
    db.create_schema()
    row = db.save_decision_event(event("r1-db-cloud", "CLOUD"))
    assert row.request_elapsed_ms == 20.0
    assert row.network_ms is None and row.edge_compute_ms is None
    assert row.server_compute_ms == 3.0
    assert row.total_latency_ms is None
    assert row.rssi is None and row.free_heap_ratio is None and row.energy_budget is None
    db.close()


def test_rejects_conflicting_duplicate_request_id():
    db = Database("sqlite+pysqlite:///:memory:")
    db.create_schema()
    db.save_decision_event(event())
    changed = event()
    changed["confidence"] = 0.5
    with pytest.raises(ValueError, match="conflicting duplicate"):
        db.save_decision_event(changed)
    db.close()
