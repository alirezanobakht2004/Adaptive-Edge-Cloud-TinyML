import hashlib
import json
from pathlib import Path


def test_frozen_deployment_artifacts_and_feature_order():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'data/policy/models/meta-policy-v1.0.0/deployment_manifest.json').read_text())
    for path, expected in manifest['artifacts_sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    contract = json.loads((root / 'data/policy/models/meta-policy-v1.0.0/input_contract.json').read_text())
    assert [feature['name'].rsplit('.', 1)[-1] for feature in contract['features']] == manifest['input_order']
    assert contract['actions'] == {'0': 'LOCAL', '1': 'CLOUD'}
    assert manifest['policy_version'] == 'meta-policy-v1.0.0'
