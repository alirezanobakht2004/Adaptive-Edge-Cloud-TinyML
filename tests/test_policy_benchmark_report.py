from tools.policy_dataset.benchmark_report import generate_benchmark
from tools.policy_dataset.instrumentation import PolicySample
from tools.policy_dataset.serializer import serialize


def test_report_uses_only_observed_metrics(tmp_path):
    record = PolicySample.create(sample_id="s", device_id="d", window_id="w", run_id="r", session_id="s", scope="device").record
    record["outcome"].update(status="success", predicted_class=1, true_class=1, confidence=.8)
    record["measurements"].update(total_latency_ms=2, request_bytes=0, response_bytes=0)
    path = tmp_path / "records.jsonl"
    path.write_text(serialize(record), encoding="utf-8")
    report = generate_benchmark(path, tmp_path / "report.json")
    assert report["baselines"]["ALL_LOCAL"]["accuracy"] == 1
    assert report["baselines"]["ALL_LOCAL"]["mean_latency_ms"] == 2
    assert report["baselines"]["ALL_CLOUD"]["accuracy"] is None
    assert report["baselines"]["RULE_BASED_ADAPTIVE"]["status"] == "missing"
    assert report["baselines"]["ALL_LOCAL"]["mean_estimated_energy_proxy"] is None
