import json
from types import SimpleNamespace

import pytest

from server.app.mqtt import (
    build_inference_response,
    device_id_from_topic,
    response_topic,
)
from server.app.schemas import (
    parse_inference_request,
)


def make_request(
    **overrides: object,
) -> dict[str, object]:
    request: dict[str, object] = {
        "request_id": "m7-test-001",
        "device_id": "esp32-01",
        "timestamp_ms": 123456,
        "split": 3,
        "embedding": [0.0] * 32,
        "model_version":
            "gesture-model-v1.1.0",
    }

    request.update(overrides)

    return request


def encode_request(
    **overrides: object,
) -> bytes:
    return json.dumps(
        make_request(**overrides)
    ).encode("utf-8")


def test_fixed_split3_request_contract() -> None:
    request = parse_inference_request(
        encode_request()
    )

    assert request["request_id"] == "m7-test-001"
    assert request["device_id"] == "esp32-01"
    assert request["split"] == 3
    assert request["model_version"] == (
        "gesture-model-v1.1.0"
    )

    assert len(request["embedding"]) == 32
    assert all(
        isinstance(value, float)
        for value in request["embedding"]
    )


def test_request_rejects_non_split3() -> None:
    with pytest.raises(
        ValueError,
        match="fixed Split 3 only",
    ):
        parse_inference_request(
            encode_request(split=2)
        )


def test_request_rejects_wrong_embedding_size() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 32 values",
    ):
        parse_inference_request(
            encode_request(
                embedding=[0.0] * 31
            )
        )


def test_request_rejects_incompatible_edge_model() -> None:
    with pytest.raises(
        ValueError,
        match="incompatible edge model version",
    ):
        parse_inference_request(
            encode_request(
                model_version="gesture-model-v1.0.0"
            )
        )


def test_topic_contract() -> None:
    assert (
        device_id_from_topic(
            "gesture/esp32-01/inference/request"
        )
        == "esp32-01"
    )

    assert (
        response_topic("esp32-01")
        == "gesture/esp32-01/inference/response"
    )


def test_build_inference_response() -> None:
    class FakeRuntime:
        def infer(
            self,
            embedding: object,
        ) -> object:
            assert len(embedding) == 32

            return SimpleNamespace(
                predicted_class="IDLE",
                confidence=0.875,
                server_latency_ms=1.25,
                model_version=(
                    "gesture-cloud-tail-v1.0.0"
                ),
            )

    request = make_request()

    response = build_inference_response(
        request,
        FakeRuntime(),  # type: ignore[arg-type]
    )

    assert response == {
        "request_id": "m7-test-001",
        "predicted_class": "IDLE",
        "confidence": 0.875,
        "server_latency_ms": 1.25,
        "model_version":
            "gesture-cloud-tail-v1.0.0",
    }
