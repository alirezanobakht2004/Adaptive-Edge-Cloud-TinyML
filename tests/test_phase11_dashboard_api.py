from fastapi.testclient import TestClient

from server.app.database import Database
from server.app.main import create_app
from server.app.telemetry_schema import DECISION_SCHEMA_VERSION


def event(request_id: str, *, action: str = "LOCAL", failover: bool = False, controlled: bool = False):
    is_cloud = action == "CLOUD"
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "request_id": request_id,
        "device_id": "esp32-r1",
        "timestamp_ms": 1000,
        "window_id": 1,
        "requested_action": "CLOUD" if failover else action,
        "effective_action": action,
        "failover": failover,
        "failover_reason": "CLOUD_RESPONSE_TIMEOUT" if failover else "NONE",
        "failure_stage": "WAIT_RESPONSE" if failover else "NONE",
        "wifi_connected": True,
        "mqtt_connected": True,
        "predicted_class_id": 2 if is_cloud else 0,
        "confidence": 0.9,
        "uncertainty": 0.2,
        "rtt_ms": 12.0,
        "rtt_source": "last_successful_probe",
        "rtt_age_ms": 100,
        "free_heap_bytes": 190000,
        "local_inference_ms": 1.2,
        "request_elapsed_ms": 30.0 if is_cloud else (3000.0 if failover else None),
        "server_compute_ms": 3.0 if is_cloud else None,
        "bytes_tx": 500 if is_cloud or failover else 0,
        "bytes_rx": 220 if is_cloud else 0,
        "model_version": "gesture-full-cloud-v1.0.0" if is_cloud else "gesture-model-v1.1.0",
        "policy_version": "meta-policy-v1.0.0",
        "firmware_version": "0.3.1-r1",
        "success": True,
        "controlled": controlled,
    }


def seeded_db(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'dashboard.db'}"
    db = Database(url)
    db.create_schema()
    db.save_decision_event(event("r1-a"))
    db.save_decision_event(event("r1-b", action="CLOUD"))
    db.save_decision_event(event("r1-c", failover=True))
    db.save_decision_event(event("r1-controlled", controlled=True))
    db.close()
    return url


def test_dashboard_endpoints_keep_provenance_and_partial_timings_distinct(tmp_path):
    url = seeded_db(tmp_path)
    with TestClient(create_app(url)) as client:
        health = client.get("/api/dashboard/health").json()
        assert health["checkpoint"] == "M11.2" and health["database_configured"] is True

        events = client.get("/api/dashboard/events", params={"device_id": "esp32-r1"}).json()["events"]
        assert len(events) == 3
        assert all(event["controlled"] is False for event in events)
        assert all(event["split_point"] is None for event in events)
        assert all(event["energy_budget"] is None for event in events)
        assert all(event["network_ms"] is None and event["total_latency_ms"] is None for event in events)

        summary = client.get("/api/dashboard/summary", params={"device_id": "esp32-r1"}).json()
        assert summary["action_counts"] == {"LOCAL": 2, "CLOUD": 1}
        assert summary["failover_count"] == 1
        assert summary["request_elapsed_ms"]["mean"] is not None
        assert "total_latency_ms" in summary["unmeasured_fields"]


def test_dashboard_filters_and_websocket_ready_message(tmp_path):
    url = seeded_db(tmp_path)
    with TestClient(create_app(url)) as client:
        cloud = client.get("/api/dashboard/events", params={"action": "CLOUD"}).json()["events"]
        assert len(cloud) == 1 and cloud[0]["execution_mode"] == "CLOUD"

        devices = client.get("/api/dashboard/devices").json()["devices"]
        assert devices[0]["device_id"] == "esp32-r1" and devices[0]["event_count"] == 3

        with client.websocket_connect("/ws/dashboard/events?device_id=esp32-r1&after_id=999") as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            assert ready["device_id"] == "esp32-r1"
