import json
from pathlib import Path
import zipfile

import pytest

from ml.policy.split_architecture_analysis import (
    MODEL_ROOT, VERSIONS, candidate_placement, dense_inventory, payload_model,
    payload_reward_change, run,
)


def artifacts():
    return [dense_inventory(MODEL_ROOT / v / f"{v}.keras") for v in VERSIONS]


def test_native_boundaries_and_bottleneck_compute():
    source = artifacts()
    sixteen = candidate_placement(16, source)
    assert sixteen["requires_new_parameters"]
    assert sixteen["edge_dense_macs"] == 640 + 64 * 16
    assert sixteen["cloud_dense_macs"] == 64 * 16 + 4768
    assert candidate_placement(24, source)["additional_dense_macs"] == 3072
    native = candidate_placement(32, source)
    assert native["edge_dense_macs"] == 5248
    assert native["cloud_dense_macs"] == 4256
    assert not native["requires_new_parameters"]
    assert candidate_placement(48, source)["edge_dense_macs"] == 3712
    for invalid in (True, 10, 64, 16.0):
        with pytest.raises(ValueError): candidate_placement(invalid, source)


def test_payload_fit_and_double_penalty_accounting():
    fit = payload_model(768, 672)
    assert fit["bytes_per_dimension"] == 6
    assert fit["intercept_bytes"] == 384
    config = dict(weights=dict(communication=.1, energy_proxy=.1),
                  scales=dict(communication_bytes=1024, energy_proxy=1))
    assert payload_reward_change(-128, config, 1024) == pytest.approx(.025)
    assert payload_reward_change(0, config, 1024) == 0
    with pytest.raises(ValueError): payload_model(500, 600)


def test_missing_dense_layers_rejected(tmp_path):
    path = tmp_path / "empty.keras"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps({"config": {"layers": []}}))
    with pytest.raises(ValueError): dense_inventory(path)


def test_report_reproduces_and_does_not_invent_candidate_measurements(tmp_path):
    report = run(tmp_path / "report.json")
    assert report == json.loads(Path("docs/evidence/phase10_5_candidate_comparison.json").read_text())
    assert report["full_cloud_dense_macs"] == 9504
    assert report["training_allowed"] is False
    assert report["new_artifacts_created"] is False
    for candidate in report["candidates"]:
        assert candidate["float32_embedding_bytes"] == 4 * candidate["embedding_dim"]
        assert candidate["raw_bytes_ratio_to_cloud_features"] > 1
        assert candidate["candidate_measured_latency_ms"] is None
        assert candidate["candidate_measured_accuracy"] is None
        assert candidate["actual_candidate_reward"] is None
