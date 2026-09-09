"""Generate immutable model bytes, normalization and controlled parity vectors."""
import numpy as np
from .binary_contract import ROOT, read_json
from .train_meta_binary import OUTPUT, VERSION


def generate():
    content = (OUTPUT / f"{VERSION}.tflite").read_bytes()
    config = read_json(OUTPUT / "input_contract.json")
    header = ROOT / "firmware/include/meta_policy_data.h"
    header.write_text('#pragma once\n#include <cstddef>\nnamespace meta_policy_data {\nextern const unsigned char MODEL[];\nextern const unsigned int MODEL_LEN;\nextern const double OFFSET[6];\nextern const double DIVISOR[6];\nconstexpr const char* VERSION = "meta-policy-v1.0.0";\n}\n')
    lines = ['#include "meta_policy_data.h"', 'namespace meta_policy_data {', 'alignas(16) const unsigned char MODEL[] = {']
    for start in range(0, len(content), 12):
        lines.append('    ' + ', '.join(f'0x{value:02x}' for value in content[start:start+12]) + ',')
    lines += ['};', f'const unsigned int MODEL_LEN = {len(content)};',
              'const double OFFSET[6] = {' + ', '.join(format(f['normalization']['offset'], '.17g') for f in config['features']) + '};',
              'const double DIVISOR[6] = {' + ', '.join(format(f['normalization']['divisor'], '.17g') for f in config['features']) + '};', '}']
    (ROOT / 'firmware/src/policy/meta_policy_data.cpp').write_text('\n'.join(lines) + '\n')
    vectors = np.load(OUTPUT / 'parity_vectors.npz')
    lines = ['#pragma once', 'namespace meta_vectors {', f'constexpr unsigned COUNT = {len(vectors["actions"])};']
    for key, name in [('raw_states','RAW'), ('inputs','NORMALIZED'), ('expected','EXPECTED'), ('features','FEATURES')]:
        values = vectors[key]
        lines.append(f'constexpr float {name}[COUNT][{values.shape[1]}] = {{')
        for row in values:
            lines.append('    {' + ', '.join(f'{float(v):.9e}f' for v in row) + '},')
        lines.append('};')
    for key, name in [('actions','ACTIONS'), ('local_predictions','LOCAL_PREDICTIONS'), ('seeds','SEEDS')]:
        lines.append(f'constexpr unsigned {name}[COUNT] = {{' + ', '.join(str(int(v)) for v in vectors[key]) + '};')
    lines.append('}')
    directory = ROOT / 'firmware/test/test_phase9_meta_policy_parity'
    directory.mkdir(exist_ok=True)
    (directory / 'meta_vectors.h').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    generate()
