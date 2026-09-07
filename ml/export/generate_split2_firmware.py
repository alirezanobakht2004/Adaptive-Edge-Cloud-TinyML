"""Export the frozen B1+B2 prefix, verify desktop parity, and generate C++ assets."""

import json

import numpy as np
import tensorflow as tf

from ml.export.generate_split1_firmware import bytes_to_cpp, matrix_to_cpp, write_new_file
from ml.export.split_validation import ROOT, MODELS, SOURCE_VERSION, SPLITS, convert_float32, run_float32
from ml.models.split_models import build_normalized_prefix
from ml.training.train_cloud_tail import load_source_model, sha256_file


def main() -> None:
    output = SPLITS / f"{SOURCE_VERSION}-split2-prefix-float32-normalized-input.tflite"
    parity_path = SPLITS / "split2_parity_vectors.npz"
    report_path = SPLITS / "split2_export_report.json"
    header = ROOT / "firmware/include/split2_model_data.h"
    source_path = ROOT / "firmware/src/inference/split2_model_data.cpp"
    vectors_path = ROOT / "firmware/test/test_phase7_split2_parity/split2_parity_vectors.h"
    if any(path.exists() for path in (output, parity_path, report_path, header, source_path, vectors_path)):
        raise RuntimeError("Refusing to overwrite Split2 prefix evidence")
    model_path = MODELS / SOURCE_VERSION / f"{SOURCE_VERSION}.keras"
    model = load_source_model(model_path)
    prefix = build_normalized_prefix(model, 2)
    with np.load(SPLITS / "split1_parity_vectors.npz") as vectors:
        normalized = vectors["normalized_inputs"].copy()
        raw = vectors["raw_features"].copy()
        labels = vectors["true_classes"].copy()
        indices = vectors["validation_indices"].copy()
    expected = prefix(normalized, training=False).numpy()
    raw_prefix = tf.keras.Model(model.input, model.get_layer("block2").output)
    np.testing.assert_allclose(expected, raw_prefix(raw, training=False).numpy(), atol=1e-5, rtol=0)
    output.write_bytes(convert_float32(prefix))
    actual = run_float32(output, normalized, 10, 48)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    np.savez_compressed(parity_path, normalized_inputs=normalized, raw_features=raw,
                        true_classes=labels, validation_indices=indices,
                        expected_split2=expected, tflite_split2=actual)
    digest = sha256_file(output)
    write_new_file(header, f'''#pragma once
#include <stddef.h>
namespace split2_model_data {{
constexpr size_t SPLIT_ID = 2;
constexpr size_t INPUT_FEATURES = 10;
constexpr size_t OUTPUT_UNITS = 48;
constexpr size_t EXPECTED_MODEL_LEN = {output.stat().st_size};
extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char MODEL_SHA256[];
extern const char SOURCE_MODEL_VERSION[];
}}
''')
    write_new_file(source_path, f'''#include "split2_model_data.h"
namespace split2_model_data {{
alignas(16) const unsigned char MODEL[] = {{
{bytes_to_cpp(output.read_bytes())}
}};
const size_t MODEL_LEN = sizeof(MODEL);
const char MODEL_SHA256[] = "{digest}";
const char SOURCE_MODEL_VERSION[] = "{SOURCE_VERSION}";
static_assert(MODEL_LEN == EXPECTED_MODEL_LEN, "Split2 byte count mismatch");
}}
''')
    write_new_file(vectors_path, f'''#pragma once
#include <stddef.h>
namespace split2_parity_vectors {{
constexpr size_t VECTOR_COUNT = {len(normalized)};
constexpr size_t INPUT_FEATURES = 10;
constexpr size_t OUTPUT_UNITS = 48;
constexpr float NORMALIZED_INPUTS[VECTOR_COUNT][INPUT_FEATURES] = {{
{matrix_to_cpp(normalized)}
}};
constexpr float EXPECTED_OUTPUTS[VECTOR_COUNT][OUTPUT_UNITS] = {{
{matrix_to_cpp(expected)}
}};
}}
''')
    report = dict(phase=7, milestone="M8", split_id=2, source_model_version=SOURCE_VERSION,
                  source_model_sha256=sha256_file(model_path), dataset_version="dataset-v1",
                  feature_version="features-v1", input_shape=[1, 10], output_shape=[1, 48],
                  prefix_tflite=output.name, prefix_sha256=digest, prefix_bytes=output.stat().st_size,
                  parity_count=len(normalized), tolerance=1e-5, test_split_used=False,
                  max_abs_diff=float(np.max(np.abs(actual - expected))))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
