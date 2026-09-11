import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server.app.database import Database
from server.app.main import create_app
from server.app.pose_schema import (
    ESTIMATOR_VERSION,
    ORIENTATION_VERSION,
    POSE_SCHEMA_VERSION,
    POSE_SOURCE,
    YAW_REFERENCE,
    parse_pose_event,
)


def pose(pose_id="r1-pose-a1b2c3d4-10", **changes):
    value = {
        "schema_version": POSE_SCHEMA_VERSION,
        "pose_id": pose_id,
        "device_id": "esp32-r1",
        "timestamp_ms": 1234,
        "sequence": 10,
        "roll_deg_est": 12.5,
        "pitch_deg_est": -7.25,
        "yaw_rel_deg_est": 31.0,
        "estimator_version": ESTIMATOR_VERSION,
        "orientation_version": ORIENTATION_VERSION,
        "firmware_version": "0.3.2-r1",
        "source": POSE_SOURCE,
        "yaw_reference": YAW_REFERENCE,
    }
    value.update(changes)
    return value


def test_pose_contract_is_strict_and_keeps_raw_windows_out():
    parsed = parse_pose_event(pose())
    assert parsed["yaw_reference"] == "boot-relative"
    assert parsed["source"] == "mpu6050-6axis"
    for forbidden in ("raw", "samples", "features", "embedding"):
        assert forbidden not in parsed

    invalid = pose()
    invalid["features"] = [0.0] * 10
    with pytest.raises(ValueError, match="required fields mismatch"):
        parse_pose_event(invalid)


@pytest.mark.parametrize(
    "changes",
    [
        {"pose_id": "bad"},
        {"sequence": 0},
        {"roll_deg_est": 181.0},
        {"pitch_deg_est": float("nan")},
        {"yaw_rel_deg_est": -181.0},
        {"yaw_reference": "absolute"},
        {"orientation_version": "other"},
    ],
)
def test_pose_contract_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        parse_pose_event(pose(**changes))


def test_pose_database_persists_and_deduplicates():
    db = Database("sqlite+pysqlite:///:memory:")
    db.create_schema()
    first = db.save_pose_event(pose())
    duplicate = db.save_pose_event(pose())
    assert first.id == duplicate.id
    assert db.count_pose_events(device_id="esp32-r1") == 1
    latest = db.latest_pose_event(device_id="esp32-r1")
    assert latest is not None
    assert latest.roll_deg_est == 12.5
    assert latest.yaw_reference == "boot-relative"

    changed = pose(roll_deg_est=13.5)
    with pytest.raises(ValueError, match="conflicting duplicate pose"):
        db.save_pose_event(changed)
    db.close()


def test_pose_mqtt_routing_persists_without_inference():
    from server.app.r1_mqtt import R1Service
    class Store:
        def __init__(self):
            self.saved = []

        def save_pose_event(self, value):
            self.saved.append(value)

    obj = object.__new__(R1Service)
    obj.events, obj.errors, obj.event_path = [], [], None
    obj.database = Store()
    message = SimpleNamespace(topic="gesture/esp32-r1/pose", payload=json.dumps(pose()).encode())
    client = SimpleNamespace()

    obj._message(client, None, message)
    assert not obj.errors
    assert obj.database.saved[0]["pose_id"] == "r1-pose-a1b2c3d4-10"
    assert obj.events[-1]["kind"] == "pose"


def test_pose_api_and_websocket_stream(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'pose-dashboard.db'}"
    db = Database(url)
    db.create_schema()
    row = db.save_pose_event(pose())
    db.close()

    with TestClient(create_app(url)) as client:
        health = client.get("/api/dashboard/health").json()
        assert health["pose_event_count"] == 1

        latest = client.get("/api/dashboard/pose/latest", params={"device_id": "esp32-r1"}).json()
        assert latest["id"] == row.id
        assert latest["yaw_rel_deg_est"] == 31.0

        history = client.get("/api/dashboard/pose", params={"device_id": "esp32-r1"}).json()
        assert history["count"] == 1

        with client.websocket_connect(f"/ws/dashboard/pose?device_id=esp32-r1&after_id={row.id - 1}") as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            streamed = websocket.receive_json()
            assert streamed["type"] == "pose"
            assert streamed["pose"]["pose_id"] == "r1-pose-a1b2c3d4-10"
