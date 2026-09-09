"""Production validation is independent of retained fixed-split routing."""
from types import SimpleNamespace
from server.app.r1_mqtt import R1Service
from server.app.r1_schema import make_request, encode


class Client:
    def __init__(self):
        self.sent = []

    def publish(self, topic, payload, **kwargs):
        self.sent.append((topic, payload))
        return SimpleNamespace(rc=0)


def service():
    obj = object.__new__(R1Service)
    obj.events, obj.errors, obj.event_path = [], [], None
    return obj


def test_rejects_topic_device_mismatch_before_inference():
    obj, client = service(), Client()
    data = make_request('window-1', [0.] * 10, [.8, .3, .6, 100000., 2., 10.])
    msg = SimpleNamespace(topic='gesture/other/inference/request', payload=encode(data))
    obj._message(client, None, msg)
    assert obj.errors == ['Topic/payload device mismatch']
    assert not client.sent


def test_nonobject_payload_is_rejected():
    obj, client = service(), Client()
    obj._message(client, None, SimpleNamespace(topic='gesture/esp32-r1/inference/request', payload=b'[]'))
    assert obj.errors and not client.sent


def test_probe_is_correlated_without_model_inference():
    obj, client = service(), Client()
    data = {'schema_version': 'r1-probe-v1', 'request_id': 'probe-1', 'device_id': 'esp32-r1'}
    obj._message(client, None, SimpleNamespace(topic='gesture/esp32-r1/policy/probe/request', payload=encode(data)))
    assert not obj.errors
    assert client.sent[0][0] == 'gesture/esp32-r1/policy/probe/response'
    assert obj.events[0]['response']['request_id'] == 'probe-1'


def test_retained_split_delegation(monkeypatch):
    import server.app.r1_mqtt as module
    obj, client = service(), Client()
    obj.legacy = object()
    seen = []
    monkeypatch.setattr(module, 'legacy_message', lambda c, s, m: seen.append((s, m)))
    message = SimpleNamespace(topic='gesture/split-test/inference/request', payload=encode({'split': 1, 'embedding': [0.] * 64}))
    obj._message(client, None, message)
    assert seen == [(obj.legacy, message)]
    assert not obj.errors and not obj.events


def test_phase11_decision_telemetry_routes_to_database_without_cloud_inference():
    from server.app.telemetry_schema import DECISION_SCHEMA_VERSION

    class Database:
        def __init__(self):
            self.saved = []

        def save_decision_event(self, value):
            self.saved.append(value)

    obj, client = service(), Client()
    obj.database = Database()
    data = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "request_id": "r1-telemetry-1",
        "device_id": "esp32-r1",
        "timestamp_ms": 100,
        "window_id": 1,
        "requested_action": "LOCAL",
        "effective_action": "LOCAL",
        "failover": False,
        "failover_reason": "NONE",
        "failure_stage": "NONE",
        "wifi_connected": True,
        "mqtt_connected": True,
        "predicted_class_id": 0,
        "confidence": 0.9,
        "uncertainty": 0.1,
        "rtt_ms": 10.0,
        "rtt_source": "last_successful_probe",
        "rtt_age_ms": 20,
        "free_heap_bytes": 160000,
        "local_inference_ms": 2.0,
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
    msg = SimpleNamespace(topic="gesture/esp32-r1/telemetry", payload=encode(data))
    obj._message(client, None, msg)
    assert not obj.errors and not client.sent
    assert obj.database.saved[0]["request_id"] == "r1-telemetry-1"
    assert obj.events[-1]["kind"] == "decision" and obj.events[-1]["persisted"] is True
