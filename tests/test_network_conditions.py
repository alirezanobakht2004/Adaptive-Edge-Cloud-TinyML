from copy import deepcopy

import pytest

from tests.test_matched_campaign import fixture
from tools.network_conditions.profiles import NAMES, apply_condition, load_profile, validate_profile


def test_profiles_are_explicit_simulations_and_preserve_sources():
    entries, _ = fixture()
    original = deepcopy(entries)
    for name in NAMES:
        profile = load_profile(name)
        assert profile["simulated"] is True
        assert profile["injected_parameters"]
        for entry in entries:
            result = apply_condition(entry, profile)
            assert result == apply_condition(entry, profile)
            if name == "baseline":
                assert result["total_latency_ms"] == pytest.approx(entry["trace"]["total_us"] / 1000)
    assert entries == original


def test_high_latency_charges_probe_and_remote_round_trip():
    entries, _ = fixture()
    for entry in entries[:4]:
        base = apply_condition(entry, load_profile("baseline"))
        high = apply_condition(entry, load_profile("high_latency"))
        assert high["total_latency_ms"] - base["total_latency_ms"] == pytest.approx(300 if entry["trace"]["action"] else 150)


@pytest.mark.parametrize("field,value", [("simulated", False), ("profile", "SPLIT3"), ("version", "unknown")])
def test_invalid_profile_rejected(field, value):
    profile = load_profile("baseline")
    profile[field] = value
    with pytest.raises(ValueError): validate_profile(profile)


def test_invalid_multiplier_rejected():
    profile = load_profile("baseline")
    profile["injected_parameters"]["edge_compute_multiplier"] = float("nan")
    with pytest.raises(ValueError): validate_profile(profile)
