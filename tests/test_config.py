from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    with (ROOT / "config" / name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_gesture_config():
    cfg = _load("gestures.yaml")
    gestures = cfg["gestures"]
    assert [g["id"] for g in gestures] == [0, 1, 2, 3, 4]
    assert [g["name"] for g in gestures] == [
        "IDLE", "SWIPE_LEFT", "SWIPE_RIGHT", "ROTATE_CW", "SHAKE"
    ]


def test_hardware_config():
    cfg = _load("hardware.yaml")
    assert cfg["board"]["module"] == "ESP32-S3-WROOM-1-N8R2"
    assert cfg["imu"]["sampling_rate_hz"] == 100
    assert cfg["imu"]["accel_range_g"] == 4
    assert cfg["imu"]["gyro_range_dps"] == 500
    assert cfg["pins"] == {"sda": 8, "scl": 9}


def test_model_config():
    cfg = _load("model.yaml")
    assert cfg["input_features"] == 10
    assert cfg["classes"] == 5
    assert cfg["mc_dropout"]["passes"] == 5


def test_cost_weights_sum_to_one():
    cfg = _load("experiment.yaml")
    weights = cfg["cost_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
