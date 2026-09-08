import numpy as np
from ml.policy.local_cloud_oracle import category, firmware_masks, uncertainty
from ml.policy.binary_contract import ROOT, read_json


def test_firmware_mask_sequence_is_reproducible():
    assert np.array_equal(firmware_masks(42000), firmware_masks(42000))
    assert not np.array_equal(firmware_masks(42000), firmware_masks(42001))
    assert firmware_masks(42000).shape == (5, 32)
    assert set(np.unique(firmware_masks(42000))) == {0, 1}


def test_oracle_categories_and_uncertainty():
    assert [category(*p) for p in ((True, True), (True, False), (False, True), (False, False))] == list("ABCD")
    result = uncertainty([.1, .2, .4, .2, .1])
    assert result["confidence"] == .4
    assert result["margin"] == .2


def test_oracle_report_excludes_final_test_and_matches_raw_state():
    report = read_json(ROOT / "docs/evidence/phase9_local_cloud_oracle_audit.json")
    assert report["test_partition_used"] is False
    assert {r["split"] for r in report["rows"]} == {"train", "validation"}
    assert len(report["rows"]) == 800
    assert all(r["action_match"] for r in report["runtime_reference_check"])
    assert max(r["max_uncertainty_difference"] for r in report["runtime_reference_check"]) < 2e-5
    for split, summary in report["summary"].items():
        assert sum(summary["counts"].values()) == summary["samples"]
        assert summary["counts"]["C"] == sum(r["category"] == "C" for r in report["rows"] if r["split"] == split)
