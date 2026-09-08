"""Read-only M25 configuration checks and feature availability; no policy implementation."""
import json
from pathlib import Path
from tools.policy_dataset.schema import ACTION_NAMES, VERSION, MODEL_VERSION, FEATURE_VERSION, TEMPLATE, validate_record

CONFIG_PATH = Path(__file__).with_name("policy_config_v1.json")


def value_at(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Unknown schema path: {path}")
        value = value[key]
    return value


def validate_config(config):
    expected = {"version": "policy-config-v1", "status": "formulation-only", "dataset_version": VERSION,
                "feature_version": FEATURE_VERSION, "model_version": MODEL_VERSION, "training_enabled": False}
    for key, value in expected.items():
        if config.get(key) != value or (key == "training_enabled" and type(config[key]) is not bool):
            raise ValueError(f"Invalid M25 configuration: {key}")
    if config.get("actions") != {str(k): v for k, v in ACTION_NAMES.items()}:
        raise ValueError("Action mapping must preserve ALL_CLOUD as action 3")
    names = config.get("state_feature_names")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("State feature names must be unique")
    for name in names:
        if not name.startswith("state.") or name == "state.application.window_id" or isinstance(value_at(TEMPLATE, name), dict):
            raise ValueError(f"Not an eligible pre-decision state feature: {name}")
    if set(config.get("feature_units", {})) != set(names):
        raise ValueError("Every selected state feature requires units")
    categories = config["categorical_features"]
    if categories != {"state.network.connection_state": ["connected", "disconnected", "unknown"],
                       "state.application.predicted_class": [0, 1, 2, 3, 4]} or not set(categories) <= set(names):
        raise ValueError("Categorical encoding must match dataset-v1 state")
    dimension = len(names) - len(categories) + sum(len(values) for values in categories.values())
    if config["encoding_specification"]["encoded_dimension"] != dimension or config["encoding_specification"]["fitted_parameters"] is not None:
        raise ValueError("Encoding dimension/fit status mismatch")
    reward = config["reward"]
    if (reward["weights"] != dict.fromkeys(("accuracy", "latency", "communication", "energy_proxy"))
            or reward["scales"] != dict.fromkeys(("latency_ms", "communication_bytes", "estimated_energy_proxy"))
            or reward["energy_kind"] != "estimated_or_simulated_only" or reward["energy_proxy_version"] != "unconfigured"):
        raise ValueError("M25 reward parameters must remain explicit unconfigured placeholders")


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def missing_state_features(record, config):
    validate_record(record)
    validate_config(config)
    return [name for name in config["state_feature_names"] if value_at(record, name) is None]
