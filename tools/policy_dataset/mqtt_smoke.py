"""Opt-in host/broker smoke test using frozen Split1/2 models; no ESP32 flashing."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

from .collector import initialize, collect
from .instrumentation import PolicySample, RoundTrip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    import numpy as np
    import paho.mqtt.client as mqtt
    from server.app.inference import SplitCloudInference
    from server.app.mqtt import build_inference_response
    from server.app.schemas import parse_inference_request
    from ml.export.split_validation import MODELS
    from ml.models.split_models import build_normalized_prefix
    from ml.training.train_cloud_tail import load_source_model, SOURCE_MODEL_SHA256
    runtime = SplitCloudInference()
    source = load_source_model(MODELS / "gesture-model-v1.1.0/gesture-model-v1.1.0.keras")
    run = datetime.now(timezone.utc).strftime("mqtt-host-smoke-%Y%m%dT%H%M%S")
    # Dedicated namespace avoids production subscribers and accidental duplicate replies.
    request_topic, response_topic = f"policy-smoke/{run}/request", f"policy-smoke/{run}/response"
    server = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=run + "-server")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=run + "-client")
    ready_server, ready_client, received = threading.Event(), threading.Event(), threading.Event()
    pending = {}
    errors = []

    def handle_request(connection, userdata, message):
        try:
            request = parse_inference_request(message.payload)
            sample = pending["sample"]
            sample.cloud(lambda value: build_inference_response(value, runtime),
                         lambda result: connection.publish(response_topic, json.dumps(result, separators=(",", ":"))),
                         request)
        except Exception as exc:
            errors.append(exc)
            received.set()

    def handle_response(connection, userdata, message):
        try:
            pending["trip"].finish(json.loads(message.payload), message.payload)
        except Exception as exc:
            errors.append(exc)
        received.set()

    for connection, topic, event, handler in ((server, request_topic, ready_server, handle_request),
                                              (client, response_topic, ready_client, handle_response)):
        connection.on_connect = lambda c, u, f, reason, props, topic=topic: c.subscribe(topic)
        connection.on_subscribe = lambda *unused, event=event: event.set()
        connection.on_message = handler
    records = []
    try:
        for connection in (server, client):
            connection.connect(args.host, args.port)
            connection.loop_start()
        if not ready_server.wait(10) or not ready_client.wait(10):
            raise RuntimeError("Broker subscriptions failed")
        for action in (1, 2):
            prefix = build_normalized_prefix(source, action)
            directory = MODELS / f"gesture-cloud-tail-split{action}-v1.0.0"
            with np.load(directory / f"split{action}_cloud_parity_vectors.npz") as vectors:
                for index, normalized in enumerate(vectors["normalized_inputs"]):
                    sample = PolicySample.create(sample_id=f"split{action}-{index}", device_id="host-smoke",
                                                 window_id=str(index), run_id=run, session_id="session_02")
                    sample.record["provenance"]["notes"] = (
                        "Actual host prefix, broker round trip and frozen cloud model. "
                        "Dedicated smoke topics; not an ESP32 or controlled-network benchmark.")
                    sample.record["provenance"]["artifact_hashes"] = {
                        "source_model": SOURCE_MODEL_SHA256,
                        "cloud_tail": runtime.runtimes[action].model_sha256,
                    }
                    received.clear()
                    with sample.measure("total_latency_ms"):
                        embedding = sample.prefix(lambda value: prefix(value[None], training=False).numpy()[0].tolist(), normalized, action)
                        request_id = f"{run}-{action}-{index}"
                        payload = json.dumps(dict(request_id=request_id, device_id="host-smoke", timestamp_ms=0,
                                                  split=action, embedding=embedding, model_version="gesture-model-v1.1.0"),
                                             separators=(",", ":")).encode()
                        pending.update(sample=sample, trip=RoundTrip(sample, request_id, payload))
                        info = client.publish(request_topic, payload)
                        if info.rc != mqtt.MQTT_ERR_SUCCESS:
                            raise RuntimeError("Request publish failed")
                        if not received.wait(10):
                            pending["trip"].timeout()
                            raise RuntimeError("MQTT smoke response timeout")
                        if errors:
                            raise errors[0]
                    records.append(sample.record)
        initialize(args.output)
        print(f"Recorded {collect(args.output, records)} actual host MQTT observations in {args.output}")
    finally:
        for connection in (server, client):
            connection.disconnect()
            connection.loop_stop()


if __name__ == "__main__":
    main()
