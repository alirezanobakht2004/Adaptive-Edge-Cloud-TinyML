"""MQTT fixed Split-3 inference service for Phase 6 / M7."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import paho.mqtt.client as mqtt

from .inference import Split3CloudInference
from .schemas import parse_inference_request


BROKER_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))

REQUEST_TOPIC = "gesture/+/inference/request"


@dataclass(frozen=True)
class ServerState:
    """Long-lived state shared by MQTT callbacks."""

    runtime: Split3CloudInference


def device_id_from_topic(topic: str) -> str:
    parts = topic.split("/")

    if (
        len(parts) != 4
        or parts[0] != "gesture"
        or parts[2] != "inference"
        or parts[3] != "request"
        or not parts[1]
    ):
        raise ValueError(
            f"invalid inference request topic: {topic}"
        )

    return parts[1]


def response_topic(device_id: str) -> str:
    return f"gesture/{device_id}/inference/response"


def build_inference_response(
    request: dict[str, Any],
    runtime: Split3CloudInference,
) -> dict[str, Any]:
    """Run the fixed Split-3 cloud tail and build response JSON."""

    result = runtime.infer(request["embedding"])

    return {
        "request_id": request["request_id"],
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "server_latency_ms": result.server_latency_ms,
        "model_version": result.model_version,
    }


def on_connect(
    client: mqtt.Client,
    userdata: object,
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    if reason_code.is_failure:
        print(
            "MQTT_CONNECT_FAIL "
            f"reason={reason_code}"
        )
        return

    result, _ = client.subscribe(
        REQUEST_TOPIC,
        qos=0,
    )

    if result != mqtt.MQTT_ERR_SUCCESS:
        print(
            "MQTT_SUBSCRIBE_FAIL "
            f"rc={result}"
        )
        return

    print(
        "MQTT_CONNECTED "
        f"broker={BROKER_HOST}:{BROKER_PORT} "
        f"subscribe={REQUEST_TOPIC}"
    )


def on_message(
    client: mqtt.Client,
    userdata: object,
    message: mqtt.MQTTMessage,
) -> None:
    if not isinstance(userdata, ServerState):
        print(
            "MQTT_SERVER_STATE_ERROR "
            "reason=missing runtime"
        )
        return

    try:
        topic_device_id = device_id_from_topic(
            message.topic
        )

        request = parse_inference_request(
            message.payload
        )

        if request["device_id"] != topic_device_id:
            raise ValueError(
                "device_id in payload does not match "
                "device_id in MQTT topic"
            )

    except ValueError as exc:
        print(
            "MQTT_REQUEST_REJECTED "
            f"topic={message.topic} "
            f"reason={exc}"
        )
        return

    try:
        response = build_inference_response(
            request,
            userdata.runtime,
        )

    except (ValueError, RuntimeError) as exc:
        print(
            "MQTT_INFERENCE_FAIL "
            f"request_id={request['request_id']} "
            f"reason={exc}"
        )
        return

    payload = json.dumps(
        response,
        separators=(",", ":"),
    )

    topic = response_topic(
        request["device_id"]
    )

    publish_result = client.publish(
        topic,
        payload=payload,
        qos=0,
        retain=False,
    )

    if publish_result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(
            "MQTT_RESPONSE_PUBLISH_FAIL "
            f"request_id={request['request_id']} "
            f"rc={publish_result.rc}"
        )
        return

    print(
        "MQTT_INFERENCE_OK "
        f"request_id={request['request_id']} "
        f"device_id={request['device_id']} "
        f"split={request['split']} "
        f"class={response['predicted_class']} "
        f"confidence={response['confidence']:.6f} "
        f"server_ms={response['server_latency_ms']:.6f} "
        f"response_topic={topic}"
    )


def main() -> None:
    runtime = Split3CloudInference()

    state = ServerState(
        runtime=runtime,
    )

    client = mqtt.Client(
        callback_api_version=(
            mqtt.CallbackAPIVersion.VERSION2
        ),
        client_id="phase6-m7-server",
    )

    client.user_data_set(state)

    client.on_connect = on_connect
    client.on_message = on_message

    print(
        "PHASE6_MQTT_SERVER_START "
        f"broker={BROKER_HOST}:{BROKER_PORT}"
    )

    print(
        "CLOUD_TAIL_READY "
        f"model={runtime.model_version} "
        f"split={runtime.split_point} "
        f"embedding_dim={runtime.embedding_dimension} "
        f"sha256={runtime.model_sha256}"
    )

    client.connect(
        BROKER_HOST,
        BROKER_PORT,
        keepalive=60,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()
