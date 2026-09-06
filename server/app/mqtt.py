"""Minimal MQTT request/response service for Phase 6 / M7."""

from __future__ import annotations

import json
import os

import paho.mqtt.client as mqtt

from .schemas import parse_inference_request


BROKER_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))

REQUEST_TOPIC = "gesture/+/inference/request"


def device_id_from_topic(topic: str) -> str:
    parts = topic.split("/")

    if (
        len(parts) != 4
        or parts[0] != "gesture"
        or parts[2] != "inference"
        or parts[3] != "request"
        or not parts[1]
    ):
        raise ValueError(f"invalid inference request topic: {topic}")

    return parts[1]


def response_topic(device_id: str) -> str:
    return f"gesture/{device_id}/inference/response"


def on_connect(
    client: mqtt.Client,
    userdata: object,
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    if reason_code.is_failure:
        print(f"MQTT_CONNECT_FAIL reason={reason_code}")
        return

    client.subscribe(REQUEST_TOPIC, qos=0)

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
    try:
        topic_device_id = device_id_from_topic(message.topic)
        request = parse_inference_request(message.payload)

        if request["device_id"] != topic_device_id:
            raise ValueError(
                "device_id in payload does not match device_id in MQTT topic"
            )

    except ValueError as exc:
        print(
            "MQTT_REQUEST_REJECTED "
            f"topic={message.topic} "
            f"reason={exc}"
        )
        return

    response = {
        "request_id": request["request_id"],
        "device_id": request["device_id"],
        "status": "ok",
    }

    payload = json.dumps(
        response,
        separators=(",", ":"),
    )

    topic = response_topic(request["device_id"])

    result = client.publish(
        topic,
        payload=payload,
        qos=0,
        retain=False,
    )

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(
            "MQTT_RESPONSE_PUBLISH_FAIL "
            f"request_id={request['request_id']} "
            f"rc={result.rc}"
        )
        return

    print(
        "MQTT_REQUEST_OK "
        f"request_id={request['request_id']} "
        f"device_id={request['device_id']} "
        f"response_topic={topic}"
    )


def main() -> None:
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="phase6-m7-server",
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print(
        "PHASE6_MQTT_SERVER_START "
        f"broker={BROKER_HOST}:{BROKER_PORT}"
    )

    client.connect(
        BROKER_HOST,
        BROKER_PORT,
        keepalive=60,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()
