from copy import deepcopy
import pytest
from ml.policy.binary_contract import read_json, ROOT, FEATURES
from ml.policy.r1_dataset import validate, OUTPUT, CONFIG, qualify
from ml.policy.binary_policy_dataset import read_rows
from server.app.r1_schema import parse_cloud_request


def test_r1_dataset_reconstruction_and_qualified_gate():
    report = validate()
    assert report["training_allowed"] is True
    assert report["overlap_count"] == 0
    assert report["unique_windows"] == 800
    assert report["label_counts"] == {"0": 2422, "1": 2}
    assert report["by_provenance"]["measured"]["labels"]["1"] == 0
    assert report["by_partition"]["holdout"]["labels"]["1"] == 0
    assert tuple(read_json(CONFIG)["state_feature_names"]) == FEATURES


def test_reuse_dataset_preserves_raw_window_group_and_no_second_inference():
    rows = read_rows(OUTPUT / "policy_training_dataset_binary_r1_v2.jsonl")
    assignments = {}
    for row in rows:
        identity = row["source"]["session"], row["source"]["window"]
        if identity in assignments:
            assert assignments[identity] == row["partition"]
        assignments[identity] = row["partition"]
        assert row["candidates"][0]["second_inference_count"] == 0
        if row["provenance"]["kind"] == "simulated":
            request = parse_cloud_request(row["provenance"]["request"])
            assert len(request["features"]) == 10
    changed = deepcopy(rows)
    changed[0]["candidates"][0]["second_inference_count"] = 1
    with pytest.raises(ValueError):
        qualify(changed)
