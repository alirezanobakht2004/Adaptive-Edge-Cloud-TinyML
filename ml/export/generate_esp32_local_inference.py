from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import CLASS_TO_ID, GESTURES
from ml.export.quantize import normalize_features
from ml.export.tflite_export import DATASET_VERSION, MODEL_DIR, project_root, sha256_file
from ml.features.extractor import load_feature_split
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import MODEL_VERSION


MODEL_FILENAME = f"{MODEL_VERSION}-float32-normalized-input.tflite"
REPORT_FILENAME = "float32_normalized_input_report.json"
NORMALIZATION_FILENAME = "deployment_normalization.json"

MODEL_HEADER = Path("firmware/include/gesture_model_data.h")
MODEL_SOURCE = Path("firmware/src/inference/gesture_model_data.cpp")
TEST_HEADER = Path(
    "firmware/test/test_local_inference/local_inference_vectors.h"
)

EXPECTED_MODEL_SHA256 = (
    "9b561335a7904f04fdf18ab359aa8d88"
    "c8c90f02929288ea084d121eeea5fd56"
)
EXPECTED_MODEL_BYTES = 25344
EXPECTED_VALIDATION_SESSION = "session_02"
EXPECTED_VALIDATION_INDICES = [0, 120, 160, 40, 80]

FEATURE_COUNT = 10
CLASS_COUNT = 5


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cpp_float(value: float) -> str:
    value = float(np.float32(value))
    if not np.isfinite(value):
        raise ValueError("Cannot emit non-finite float.")
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def select_vectors(labels: np.ndarray) -> list[int]:
    selected: list[int] = []
    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]
        indices = np.flatnonzero(labels == class_id)
        if indices.size == 0:
            raise ValueError(f"No validation sample for {gesture}.")
        selected.append(int(indices[0]))
    return selected


def write_model_files(
    header_path: Path,
    source_path: Path,
    model_bytes: bytes,
    model_hash: str,
) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)

    header_path.write_text(
        """#pragma once

#include <stddef.h>

namespace gesture_model_data {

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;

extern const char MODEL_VERSION[];
extern const char MODEL_SHA256[];

}  // namespace gesture_model_data
""",
        encoding="utf-8",
    )

    byte_lines = []
    for start in range(0, len(model_bytes), 12):
        chunk = model_bytes[start:start + 12]
        values = ", ".join(f"0x{value:02x}" for value in chunk)
        byte_lines.append(f"    {values},")

    source = (
        '#include "gesture_model_data.h"\n\n'
        "namespace gesture_model_data {\n\n"
        "alignas(16) const unsigned char MODEL[] = {\n"
        + "\n".join(byte_lines)
        + "\n};\n\n"
        "const size_t MODEL_LEN = sizeof(MODEL);\n\n"
        f'const char MODEL_VERSION[] = "{MODEL_VERSION}";\n'
        f'const char MODEL_SHA256[] = "{model_hash}";\n\n'
        "}  // namespace gesture_model_data\n"
    )
    source_path.write_text(source, encoding="utf-8")


def write_test_vectors(
    path: Path,
    model_hash: str,
    model_len: int,
    labels: list[str],
    validation_indices: list[int],
    true_classes: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    expected_classes: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "",
        "namespace local_inference_vectors {",
        "",
        f'constexpr char MODEL_SHA256[] = "{model_hash}";',
        f"constexpr size_t MODEL_LEN = {model_len};",
        f"constexpr size_t VECTOR_COUNT = {len(labels)};",
        f"constexpr size_t INPUT_COUNT = {FEATURE_COUNT};",
        f"constexpr size_t OUTPUT_COUNT = {CLASS_COUNT};",
        "",
        "static const size_t VALIDATION_INDICES[VECTOR_COUNT] = {",
        "    " + ", ".join(str(index) for index in validation_indices),
        "};",
        "",
        "static const char* const LABELS[VECTOR_COUNT] = {",
    ]
    lines.extend(f'    "{label}",' for label in labels)
    lines.extend([
        "};",
        "",
        "static const int TRUE_CLASSES[VECTOR_COUNT] = {",
        "    " + ", ".join(str(int(value)) for value in true_classes),
        "};",
        "",
        "static const float INPUTS[VECTOR_COUNT][INPUT_COUNT] = {",
    ])
    for row in inputs:
        values = ", ".join(cpp_float(value) for value in row)
        lines.append(f"    {{{values}}},")

    lines.extend([
        "};",
        "",
        "static const float EXPECTED_OUTPUTS[VECTOR_COUNT][OUTPUT_COUNT] = {",
    ])
    for row in outputs:
        values = ", ".join(cpp_float(value) for value in row)
        lines.append(f"    {{{values}}},")

    lines.extend([
        "};",
        "",
        "static const int EXPECTED_CLASSES[VECTOR_COUNT] = {",
        "    " + ", ".join(str(int(value)) for value in expected_classes),
        "};",
        "",
        "}  // namespace local_inference_vectors",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_contract(report: dict, normalization: dict) -> None:
    expected_metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "fit_split": "train",
    }
    for key, expected in expected_metadata.items():
        actual = normalization.get(key)
        if actual != expected:
            raise ValueError(
                f"Normalization {key} mismatch: expected {expected}, got {actual}."
            )

    if report.get("normalization_fit_split") != "train":
        raise ValueError("Float32 report normalization split mismatch.")
    if report.get("evaluation_split") != "validation":
        raise ValueError("Float32 report evaluation split mismatch.")
    if report.get("evaluation_session") != EXPECTED_VALIDATION_SESSION:
        raise ValueError("Float32 report evaluation session mismatch.")
    if report.get("test_split_used") is not False:
        raise ValueError("Float32 report unexpectedly used test data.")
    if report.get("float32_tflite_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Float32 report SHA is not the frozen production candidate.")
    if report.get("float32_tflite_bytes") != EXPECTED_MODEL_BYTES:
        raise RuntimeError("Float32 report model size mismatch.")


def main() -> None:
    root = project_root()
    tflite_dir = root / MODEL_DIR / "tflite"

    model_path = tflite_dir / MODEL_FILENAME
    report = load_json(tflite_dir / REPORT_FILENAME)
    normalization = load_json(tflite_dir / NORMALIZATION_FILENAME)
    validate_contract(report, normalization)

    actual_hash = sha256_file(model_path)
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Float32 model SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    model_bytes = model_path.read_bytes()
    if len(model_bytes) != EXPECTED_MODEL_BYTES:
        raise RuntimeError(
            "Float32 model size mismatch.\n"
            f"Expected: {EXPECTED_MODEL_BYTES}\n"
            f"Actual:   {len(model_bytes)}"
        )

    validation = load_feature_split("validation")
    if validation.session != EXPECTED_VALIDATION_SESSION:
        raise ValueError("Expected validation/session_02.")

    selected = select_vectors(validation.labels)
    if selected != EXPECTED_VALIDATION_INDICES:
        raise RuntimeError(
            "Frozen validation-vector indices changed.\n"
            f"Expected: {EXPECTED_VALIDATION_INDICES}\n"
            f"Actual:   {selected}"
        )

    features = validation.features[selected].astype(np.float32)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    variance = np.asarray(normalization["variance"], dtype=np.float32)
    normalized = normalize_features(
        features,
        mean=mean,
        variance=variance,
    ).astype(np.float32)

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] != np.float32:
        raise ValueError("Expected float32 model input.")
    if output_detail["dtype"] != np.float32:
        raise ValueError("Expected float32 model output.")
    if tuple(input_detail["shape"]) != (1, FEATURE_COUNT):
        raise ValueError(f"Unexpected input shape: {input_detail['shape']}")
    if tuple(output_detail["shape"]) != (1, CLASS_COUNT):
        raise ValueError(f"Unexpected output shape: {output_detail['shape']}")

    outputs = np.empty((len(selected), CLASS_COUNT), dtype=np.float32)
    for vector_index, model_input in enumerate(normalized):
        interpreter.set_tensor(
            input_detail["index"],
            model_input.reshape(1, FEATURE_COUNT),
        )
        interpreter.invoke()
        outputs[vector_index] = (
            interpreter.get_tensor(output_detail["index"])
            .reshape(CLASS_COUNT)
            .astype(np.float32)
        )

    expected_classes = np.argmax(outputs, axis=1).astype(np.int64)
    true_classes = validation.labels[selected].astype(np.int64)
    selected_labels = [
        GESTURES[int(validation.labels[index])]
        for index in selected
    ]

    write_model_files(
        root / MODEL_HEADER,
        root / MODEL_SOURCE,
        model_bytes,
        actual_hash,
    )
    write_test_vectors(
        root / TEST_HEADER,
        actual_hash,
        len(model_bytes),
        selected_labels,
        selected,
        true_classes,
        normalized,
        outputs,
        expected_classes,
    )

    print()
    print("ESP32 FLOAT32 PRODUCTION ASSETS")
    print("-------------------------------")
    print(f"Model version:   {MODEL_VERSION}")
    print(f"Dataset version: {DATASET_VERSION}")
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Float32 SHA-256: {actual_hash}")
    print(f"Model bytes:     {len(model_bytes)}")
    print(f"Source:          validation / {validation.session}")
    print(f"Vectors:         {len(selected)}")
    print()

    for position, index in enumerate(selected):
        print(
            f"{selected_labels[position]:<12} "
            f"validation_index={index:<3} "
            f"desktop_class={int(expected_classes[position])} "
            f"true_class={int(true_classes[position])}"
        )

    print()
    print("Test split was not loaded.")
    print(f"Model header: {root / MODEL_HEADER}")
    print(f"Model source: {root / MODEL_SOURCE}")
    print(f"Test vectors: {root / TEST_HEADER}")


if __name__ == "__main__":
    main()
