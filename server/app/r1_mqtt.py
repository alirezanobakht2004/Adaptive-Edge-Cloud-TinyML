"""R1 production service with explicit CLOUD/probe contracts and retained split routing."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

import paho.mqtt.client as mqtt
from sqlalchemy.exc import SQLAlchemyError

from .cloud_full import FullCloudInference
from .inference import SplitCloudInference
from .mqtt import REQUEST_TOPIC, ServerState, on_message as legacy_message, device_id_from_topic, response_topic
from .r1_schema import parse_cloud_request, encode, SCHEMA_VERSION, POLICY_VERSION
from .database import DATABASE_URL_ENV, Database, database_url_from_env
from .telemetry_schema import TELEMETRY_TOPIC, parse_decision_event

PROBE_TOPIC = "gesture/+/policy/probe/request"


def build_response(request, runtime):
    request = parse_cloud_request(request)
    result = runtime.infer(request["features"])
    if result["model_version"] != request["model_version"]:
        raise ValueError("Requested full-cloud version is not loaded")
    return {"schema_version": SCHEMA_VERSION, "request_id": request["request_id"], "mode": "CLOUD",
            "success": True, **result, "policy_version": request["policy_version"]}


class R1Service:
    def __init__(self, host="127.0.0.1", port=1883, event_path=None, database_url=None):
        self.cloud = FullCloudInference()
        self.legacy = ServerState(SplitCloudInference())
        self.host, self.port = host, port
        self.event_path = Path(event_path) if event_path else None
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.events, self.errors = [], []
        resolved_database_url = database_url if database_url is not None else database_url_from_env()
        self.database = Database(resolved_database_url) if resolved_database_url else None
        if self.database is not None:
            self.database.create_schema()
        self.ready = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phase9-r1-server")
        self.client.on_connect = self._connect
        self.client.on_subscribe = lambda *args: self.ready.set()
        self.client.on_message = self._message

    def _connect(self, client, userdata, flags, reason, properties):
        if reason == 0:
            client.subscribe([(REQUEST_TOPIC, 0), (PROBE_TOPIC, 0), (TELEMETRY_TOPIC, 0)])

    def _message(self, client, userdata, message):
        received = datetime.now(timezone.utc).isoformat()
        try:
            data = json.loads(message.payload)
            if not isinstance(data, dict):
                raise ValueError("MQTT request must be an object")
            if message.topic.endswith("/telemetry"):
                parts = message.topic.split("/")
                event = parse_decision_event(data)
                if len(parts) != 3 or event["device_id"] != parts[1]:
                    raise ValueError("Telemetry topic/payload device mismatch")
                database = getattr(self, "database", None)
                if database is not None:
                    database.save_decision_event(event)
                record = {"kind": "decision", "receive_time": received, "event": event,
                          "request_bytes": len(message.payload), "persisted": database is not None}
                self.events.append(record)
                if len(self.events) > 1000:
                    del self.events[:-1000]
                if self.event_path:
                    with self.event_path.open("a", encoding="utf-8") as stream:
                        stream.write(encode(record).decode() + "\n")
                print(f"R1_DECISION_EVENT request_id={event['request_id']} action={event['effective_action']} persisted={database is not None}", flush=True)
                return
            if message.topic.endswith("/policy/probe/request"):
                parts = message.topic.split("/")
                if (len(parts) != 5 or set(data) != {"schema_version", "request_id", "device_id"}
                        or data["schema_version"] != "r1-probe-v1" or data["device_id"] != parts[1]
                        or not isinstance(data["request_id"], str) or not 1 <= len(data["request_id"]) <= 96):
                    raise ValueError("Invalid R1 probe")
                response = {"schema_version": "r1-probe-v1", "request_id": data["request_id"], "probe": True}
                topic = f"gesture/{parts[1]}/policy/probe/response"
                kind = "probe"
            elif "schema_version" not in data:
                legacy_message(client, self.legacy, message)
                return
            else:
                request = parse_cloud_request(data)
                if request["device_id"] != device_id_from_topic(message.topic):
                    raise ValueError("Topic/payload device mismatch")
                response = build_response(request, self.cloud)
                topic, kind = response_topic(request["device_id"]), "CLOUD"
            payload = encode(response)
            publish_time = datetime.now(timezone.utc).isoformat()
            result = client.publish(topic, payload, qos=0, retain=False)
            event = {"kind": kind, "request_receive_time": received, "response_publish_time": publish_time,
                     "publish_rc": int(result.rc), "request": data, "response": response,
                     "request_bytes": len(message.payload), "response_bytes": len(payload), "response_topic": topic}
            self.events.append(event)
            # Keep diagnostics bounded during the long-running production service.
            if len(self.events) > 1000:
                del self.events[:-1000]
            if self.event_path:
                with self.event_path.open("a", encoding="utf-8") as stream:
                    stream.write(encode(event).decode() + "\n")
            if kind == "CLOUD":
                print(f"R1_CLOUD_OK request_id={data['request_id']} features={len(data['features'])} policy={POLICY_VERSION}", flush=True)
        except (ValueError, TypeError, KeyError, RuntimeError, SQLAlchemyError) as exc:
            self.errors.append(str(exc))
            if len(self.errors) > 1000:
                del self.errors[:-1000]
            print(f"R1_REQUEST_REJECTED {exc}", flush=True)

    def __enter__(self):
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        if not self.ready.wait(10):
            self.__exit__(None, None, None)
            raise RuntimeError("R1 MQTT subscription unavailable")
        return self

    def __exit__(self, *args):
        self.client.disconnect()
        self.client.loop_stop()
        database = getattr(self, "database", None)
        if database is not None:
            database.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--events", type=Path, default=Path("data/policy/runtime/r1_server_events.jsonl"))
    parser.add_argument("--database-url", default=os.getenv(DATABASE_URL_ENV),
                        help=f"SQLAlchemy URL; defaults to ${DATABASE_URL_ENV}. Omit to preserve no-database Phase-10 behavior.")
    args = parser.parse_args()
    with R1Service(args.host, args.port, args.events, args.database_url):
        print("R1_SERVER_READY", flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
