import numpy as np
import pytest
from server.app.cloud_full import FullCloudInference
from tools.policy_dataset.campaign_server import CampaignServer
from tools.policy_dataset.run_campaign import balanced_indices


@pytest.fixture(scope="module")
def cloud():
    return FullCloudInference()


def test_full_cloud_executes_all_blocks(cloud):
    features = np.linspace(-1, 1, 10, dtype=np.float32)
    expected = cloud.tail._model(cloud.prefix(features[None, :]), training=False).numpy()[0]
    result = cloud.infer(features)
    assert result["predicted_class_id"] == int(expected.argmax())
    assert result["confidence"] == pytest.approx(float(expected.max()), abs=1e-7)
    assert result["model_version"] == "gesture-full-cloud-v1.0.0"
    assert result["server_latency_ms"] >= 0


@pytest.mark.parametrize("values", [[0.] * 32, [0.] * 64, [float("nan")] * 10])
def test_full_cloud_rejects_embeddings(cloud, values):
    with pytest.raises(ValueError):
        cloud.infer(values)


def test_balanced_unique_validation_windows():
    labels = np.repeat(np.arange(5), 3)
    assert balanced_indices(labels, 5) == [0, 3, 6, 9, 12]
    with pytest.raises(ValueError):
        balanced_indices(labels, 16)


def test_benchmark_action_three_is_full_cloud(cloud):
    server = CampaignServer.__new__(CampaignServer)
    server.cloud = cloud
    result = server.respond({"request_id": "test", "action": 3, "values": [0.] * 10,
                             "model_version": "gesture-model-v1.1.0"})
    assert result["action"] == 3
    assert result["model_version"] == "gesture-full-cloud-v1.0.0"
    with pytest.raises(ValueError):
        server.respond({"request_id": "test", "action": 3, "values": [0.] * 32,
                        "model_version": "gesture-model-v1.1.0"})
