import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def executable(tmp_path_factory):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('Native C++ compiler unavailable; ESP32 suite covers the same state transitions')
    target = tmp_path_factory.mktemp('failover') / 'failover_test.exe'
    subprocess.run([compiler, '-std=c++11', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'firmware/src'), str(ROOT / 'tests/cpp/failover_test.cpp'),
                    '-o', str(target)], check=True)
    return target


@pytest.mark.parametrize('scenario', range(9))
def test_actual_cpp_failover_transitions(executable, scenario):
    subprocess.run([str(executable), str(scenario)], check=True)


def test_versioned_production_timeout_contract():
    config = json.loads((ROOT / 'config/r1_failover_v1.json').read_text(encoding='utf-8'))
    header = (ROOT / 'firmware/include/failover_config.h').read_text(encoding='utf-8')
    assert config['version'] in header
    for key, value in config.items():
        if type(value) is int:
            assert re.search(rf'\b{key.upper()}\s*=\s*{value}\s*;', header)
    assert config['cloud_response_timeout_ms'] == 3000
