"""Exercise every split through the unchanged MQTT envelope and real runtimes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from server.app.inference import SplitCloudInference, Split2CloudInference
from server.app.mqtt import ServerState, on_message
from server.app.schemas import parse_inference_request


@pytest.fixture(scope="module")
def runtime():
    return SplitCloudInference()


@pytest.mark.parametrize("split,dim", [(1, 64), (2, 48), (3, 32)])
def test_all_split_mqtt_routes(runtime, split, dim):
    request = dict(request_id=f"routing-{split}", device_id="routing-test", timestamp_ms=1,
                   model_version="gesture-model-v1.1.0", split=split, embedding=[0.0] * dim)
    payload = json.dumps(request).encode()
    assert parse_inference_request(payload)["split"] == split
    published = []

    class Client:
        def publish(self, topic, **kwargs):
            published.append((topic, json.loads(kwargs["payload"])))
            return SimpleNamespace(rc=0)

    on_message(Client(), ServerState(runtime), SimpleNamespace(
        topic="gesture/routing-test/inference/request", payload=payload))
    assert len(published) == 1
    topic, response = published[0]
    expected = runtime.runtimes[split].infer(request["embedding"])
    assert topic == "gesture/routing-test/inference/response"
    assert response["request_id"] == request["request_id"]
    assert response["split"] == split
    assert response["predicted_class"] == expected.predicted_class
    assert response["confidence"] == pytest.approx(expected.confidence)
    assert response["model_version"] == expected.model_version


@pytest.mark.parametrize("split,dim", [(1, 48), (2, 32), (3, 64)])
def test_cross_split_dimensions_rejected(runtime, split, dim):
    with pytest.raises(ValueError, match="exactly"):
        runtime.infer([0.0] * dim, split)


def test_split2_rejects_nonfinite(runtime):
    with pytest.raises(ValueError, match="finite"):
        runtime.infer([np.nan] * 48, 2)


@pytest.mark.parametrize("field,value", [("split_id", 1), ("input_embedding_dim", 64),
                                         ("source_edge_model_sha256", "0" * 64)])
def test_split2_provenance_rejected(tmp_path, field, value):
    from ml.models.cloud_tail_split2 import MODEL_DIR
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    metadata[field] = value
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match=field):
        Split2CloudInference(metadata_path=path)
