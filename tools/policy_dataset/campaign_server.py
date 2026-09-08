"""Isolated MQTT benchmark responder; production split topics are unchanged."""
import json
import threading
from datetime import datetime, timezone
import numpy as np
import paho.mqtt.client as mqtt
from ml.dataset.loader import CLASS_TO_ID
from server.app.inference import SplitCloudInference
from server.app.cloud_full import FullCloudInference
from .schema import MODEL_VERSION

REQUEST_TOPIC = "phase9/esp32-policy/request"
RESPONSE_TOPIC = "phase9/esp32-policy/response"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CampaignServer:
    def __init__(self, host, port, event_path):
        self.splits = SplitCloudInference()
        self.cloud = FullCloudInference()
        self.event_path = event_path
        self.host, self.port = host, port
        self.ready = threading.Event()
        self.events = {}
        self.errors = []
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phase9-campaign-host")
        self.client.on_connect = self._connect
        self.client.on_subscribe = lambda *args: self.ready.set()
        self.client.on_message = self._message

    def _connect(self, client, userdata, flags, reason, properties):
        if reason == 0:
            client.subscribe(REQUEST_TOPIC, qos=0)

    def respond(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("request_id"), str):
            raise ValueError("Missing request identity")
        if request.get("probe") is True:
            if set(request) != {"request_id", "probe"}:
                raise ValueError("Invalid probe")
            return {"request_id": request["request_id"], "probe": True}
        action = request.get("action")
        if type(action) is not int or action not in (1, 2, 3):
            raise ValueError("Unsupported cloud action")
        if set(request) != {"request_id", "action", "values", "model_version"} or request["model_version"] != MODEL_VERSION:
            raise ValueError("Request contract/version mismatch")
        values = request["values"]
        if (not isinstance(values, list) or len(values) != {1: 64, 2: 48, 3: 10}[action]
                or any(type(v) not in (int, float) for v in values) or not np.isfinite(values).all()):
            raise ValueError("Action tensor dimension or values invalid")
        if action == 3:
            result = self.cloud.infer(values)
        else:
            inference = self.splits.infer(values, split=action)
            result = {"predicted_class_id": CLASS_TO_ID[inference.predicted_class], "confidence": inference.confidence,
                      "server_latency_ms": inference.server_latency_ms, "model_version": inference.model_version}
        return {"request_id": request["request_id"], "action": action, **result}

    def _message(self, client, userdata, message):
        receive_time = utc_now()
        try:
            request = json.loads(message.payload)
            response = self.respond(request)
            payload = json.dumps(response, separators=(",", ":"), allow_nan=False)
            publish_time = utc_now()
            info = client.publish(RESPONSE_TOPIC, payload, qos=0, retain=False)
            event = {"request_id": request["request_id"], "request_receive_time": receive_time,
                     "response_publish_time": publish_time, "publish_rc": int(info.rc),
                     "request_bytes": len(message.payload), "response_bytes": len(payload.encode()), "response": response}
            self.events[request["request_id"]] = event
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        except Exception as exc:
            self.errors.append(str(exc))

    def __enter__(self):
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        if not self.ready.wait(10):
            self.__exit__(None, None, None)
            raise RuntimeError("Campaign responder subscription timed out")
        return self

    def __exit__(self, *args):
        self.client.disconnect()
        self.client.loop_stop()
