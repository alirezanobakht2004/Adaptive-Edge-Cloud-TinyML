import json

import pytest

from tools.policy_dataset.collector import initialize, collect, production_serial_record
from tools.policy_dataset.instrumentation import PolicySample
from tools.policy_dataset.serializer import serialize, deserialize
from tools.policy_dataset.schema import example_record
from tools.policy_dataset.validator import validate_dataset


def test_skeleton_and_roundtrip(tmp_path):
    initialize(tmp_path)
    assert validate_dataset(tmp_path)["observations"] == 0
    example = example_record()
    assert deserialize(serialize(example)) == example
    with pytest.raises(ValueError):
        collect(tmp_path, [example])
    sample = PolicySample.create(sample_id="s1", device_id="host", window_id="w", run_id="r", session_id="s")
    sample.local(lambda x: x, lambda x: [1., 0., 0., 0., 0.], None)
    assert collect(tmp_path, [sample.record]) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        collect(tmp_path, [sample.record])
    assert validate_dataset(tmp_path)["observations"] == 1


def test_import_validates_whole_batch(tmp_path):
    initialize(tmp_path)
    sample = PolicySample.create(sample_id="s1", device_id="host", window_id="w", run_id="r", session_id="s")
    with pytest.raises(ValueError):
        collect(tmp_path, [sample.record, {}])
    assert (tmp_path / "observations.jsonl").read_text() == ""


def test_serial_import_preserves_missing_state():
    record = production_serial_record(
        "W=1 c=2 conf=0.9 unc=0.1 pipe_us=6000 feat=1000 norm=100 prefix=2000 mc=2900",
        device_id="esp32", run_id="r", session_id="s", timestamp="2026-09-08T12:00:00Z")
    assert record["measurements"]["preprocessing_latency_ms"] == 1.1
    assert record["measurements"]["local_inference_latency_ms"] == 4.9
    assert record["state"]["device"]["free_heap"] is None
    assert record["state"]["uncertainty"]["confidence"] is None
    assert record["outcome"]["predicted_class"] == 2


def test_json_and_manifest_rejection(tmp_path):
    with pytest.raises(ValueError):
        deserialize('{"x":1,"x":2}')
    initialize(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["actions"]["3"] = "SPLIT3"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="action"):
        validate_dataset(tmp_path)
