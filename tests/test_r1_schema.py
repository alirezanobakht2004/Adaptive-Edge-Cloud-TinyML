import pytest
from server.app.r1_schema import make_request, parse_cloud_request, encode


def request():
    return make_request("test-window", [0.] * 10, [.8, .4, .5, 160000., 2., 20.])


def test_production_features_contract():
    value = parse_cloud_request(encode(request()))
    assert value["mode"] == "CLOUD" and len(value["features"]) == 10
    assert "embedding" not in value and "split" not in value
    assert value["policy_version"] == "meta-policy-v1.0.0"


@pytest.mark.parametrize("change", ["count", "nonfinite", "boolean", "mode", "version", "missing", "device"])
def test_reject_malformed_production_request(change):
    value = request()
    if change == "count": value["features"].append(0.)
    elif change == "nonfinite": value["features"][0] = float("nan")
    elif change == "boolean": value["features"][0] = True
    elif change == "mode": value["mode"] = "SPLIT1"
    elif change == "version": value["policy_version"] = "unknown"
    elif change == "missing": del value["firmware_version"]
    else: value["device_id"] = "a/other-device"
    with pytest.raises(ValueError): parse_cloud_request(value)
