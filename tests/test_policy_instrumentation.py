import pytest

from tools.policy_dataset.instrumentation import PolicySample, RoundTrip, probability_metrics
from tools.policy_dataset.schema import validate_record


def sample():
    return PolicySample.create(sample_id="s", device_id="host", window_id="w", run_id="r", session_id="s")


def test_metrics_and_local_do_not_leak_outcome_to_state():
    metrics, label = probability_metrics([1., 0., 0., 0., 0.])
    assert metrics == {"confidence": 1, "entropy": 0, "margin": 1}
    assert label == 0
    record = sample()
    record.local(lambda x: x, lambda x: [1., 0., 0., 0., 0.], None)
    assert record.record["state"]["uncertainty"]["confidence"] is None
    assert record.record["measurements"]["total_latency_ms"] >= 0
    validate_record(record.record)


def test_roundtrip_correlates_and_times_out():
    record = sample()
    record.record["action"] = 1
    times = iter([1_000_000, 6_000_000])
    request = RoundTrip(record, "r1", b"request", clock=lambda: next(times))
    with pytest.raises(ValueError, match="request_id"):
        request.finish({"request_id": "other"}, b"response")
    request.timeout()
    assert record.record["measurements"]["round_trip_latency_ms"] == 5
    assert record.record["outcome"]["status"] == "timeout"
    assert record.record["measurements"]["response_bytes"] is None
    validate_record(record.record)


def test_split3_is_not_all_cloud():
    record = sample()
    record.record["action"] = 3
    with pytest.raises(ValueError, match="ALL_CLOUD"):
        RoundTrip(record, "id", b"req").finish({"request_id": "id", "split": 3}, b"response")


def test_stage_errors_are_preserved():
    record = sample()
    with pytest.raises(RuntimeError):
        with record.measure("prefix_latency_ms"):
            raise RuntimeError("inference failed")
    assert record.record["outcome"]["status"] == "failed"
    assert record.record["measurements"]["prefix_latency_ms"] >= 0


def test_roundtrip_success_and_cloud_adapter():
    record = sample()
    record.prefix(lambda x: [0.] * 48, None, 2)
    response = {"request_id": "r", "split": 2, "predicted_class": "IDLE", "confidence": .7,
                "server_latency_ms": 2., "model_version": "gesture-cloud-tail-split2-v1.0.0"}
    trip = RoundTrip(record, "r", b"request")
    sent = []
    assert record.cloud(lambda x: response, sent.append, None) == response
    assert sent == [response]
    trip.finish(response, b"response")
    assert record.record["measurements"]["embedding_size_bytes"] == 192
    validate_record(record.record)
