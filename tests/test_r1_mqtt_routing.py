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
