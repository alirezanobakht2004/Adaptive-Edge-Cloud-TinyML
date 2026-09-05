from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import (
    DATASET_VERSION,
    GESTURES,
)
from ml.export.uncertainty_deployment import (
    dense_softmax,
    explicit_dropout_edge_head,
)
from ml.features.extractor import (
    load_feature_split,
)
from ml.features.features_v1 import (
    FEATURE_VERSION,
)
from ml.models.base_model import (
    BLOCK_3_UNITS,
    CLASS_COUNT,
    INPUT_FEATURES,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
)
from ml.uncertainty.metrics import (
    compute_mc_dropout_uncertainty_metrics,
)


SEED = 42
MASK_SEED = 20260905

PRODUCTION_MODEL_VERSION = (
    "gesture-model-v1.0.0"
)

MODEL_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}"
)

MODEL_PATH = (
    MODEL_DIR
    / f"{UNCERTAINTY_MODEL_VERSION}.keras"
)

OUTPUT_DIR = (
    MODEL_DIR
    / "tflite"
)

PREFIX_FILENAME = (
    f"{UNCERTAINTY_MODEL_VERSION}"
    "-prefix-b3-float32-normalized-input.tflite"
)

EDGE_HEAD_FILENAME = (
    "edge_head_float32_weights.npz"
)

PARITY_VECTORS_FILENAME = (
    "edge_uncertainty_parity_vectors.npz"
)

REPORT_FILENAME = (
    "edge_uncertainty_deployment_report.json"
)

REFERENCE_NORMALIZATION_PATH = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{PRODUCTION_MODEL_VERSION}/"
    "tflite/"
    "deployment_normalization.json"
)

EXPECTED_VALIDATION_SESSION = (
    "session_02"
)

EXPECTED_VALIDATION_INDICES = (
    0,
    120,
    160,
    40,
    80,
)

NORMALIZATION_TOLERANCE = 1e-6
PREFIX_PARITY_TOLERANCE = 2e-5
HEAD_PARITY_TOLERANCE = 2e-6


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def set_reproducibility() -> None:
    tf.keras.utils.set_random_seed(
        SEED
    )

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_outputs_are_new(
    output_dir: Path,
) -> None:
    protected = (
        output_dir / PREFIX_FILENAME,
        output_dir / EDGE_HEAD_FILENAME,
        output_dir / PARITY_VECTORS_FILENAME,
        output_dir / REPORT_FILENAME,
    )

    existing = [
        path
        for path in protected
        if path.exists()
    ]

    if existing:
        joined = ", ".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Refusing to overwrite existing edge-uncertainty "
            f"deployment evidence: {joined}"
        )


def load_reference_normalization(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            "Phase-4 deployment normalization was not found: "
            f"{path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get(
        "dataset_version"
    ) != DATASET_VERSION:
        raise ValueError(
            "Reference normalization dataset version mismatch."
        )

    if payload.get(
        "feature_version"
    ) != FEATURE_VERSION:
        raise ValueError(
            "Reference normalization feature version mismatch."
        )

    mean = np.asarray(
        payload["mean"],
        dtype=np.float32,
    )

    variance = np.asarray(
        payload["variance"],
        dtype=np.float32,
    )

    if mean.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Reference normalization mean shape mismatch."
        )

    if variance.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Reference normalization variance shape mismatch."
        )

    return mean, variance


def candidate_normalization_statistics(
    model: tf.keras.Model,
) -> tuple[
    tf.keras.layers.Normalization,
    np.ndarray,
    np.ndarray,
]:
    layer = model.get_layer(
        "feature_normalization"
    )

    if not isinstance(
        layer,
        tf.keras.layers.Normalization,
    ):
        raise TypeError(
            "Expected 'feature_normalization' to be "
            "a Keras Normalization layer."
        )

    mean = (
        layer.mean.numpy()
        .reshape(-1)
        .astype(
            np.float32
        )
    )

    variance = (
        layer.variance.numpy()
        .reshape(-1)
        .astype(
            np.float32
        )
    )

    if mean.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Candidate normalization mean shape mismatch."
        )

    if variance.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Candidate normalization variance shape mismatch."
        )

    return layer, mean, variance


def validate_normalization_compatibility(
    candidate_mean: np.ndarray,
    candidate_variance: np.ndarray,
    reference_mean: np.ndarray,
    reference_variance: np.ndarray,
) -> tuple[float, float]:
    mean_diff = float(
        np.max(
            np.abs(
                candidate_mean
                - reference_mean
            )
        )
    )

    variance_diff = float(
        np.max(
            np.abs(
                candidate_variance
                - reference_variance
            )
        )
    )

    if (
        mean_diff
        > NORMALIZATION_TOLERANCE
        or variance_diff
        > NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError(
            "gesture-model-v1.1.0 normalization is not "
            "compatible with the frozen Phase-4 firmware "
            "preprocessing contract. "
            f"mean_diff={mean_diff}, "
            f"variance_diff={variance_diff}"
        )

    return mean_diff, variance_diff


def validate_model_contract(
    model: tf.keras.Model,
) -> None:
    expected = {
        "block1": 64,
        "block2": 48,
        "block3": 32,
        "edge_head": CLASS_COUNT,
    }

    for layer_name, units in expected.items():
        layer = model.get_layer(
            layer_name
        )

        if not isinstance(
            layer,
            tf.keras.layers.Dense,
        ):
            raise TypeError(
                f"{layer_name} must be a Dense layer."
            )

        if layer.units != units:
            raise ValueError(
                f"{layer_name} units mismatch: "
                f"expected {units}, got {layer.units}."
            )

    dropout = model.get_layer(
        "mc_dropout"
    )

    if not isinstance(
        dropout,
        tf.keras.layers.Dropout,
    ):
        raise TypeError(
            "Expected a Dropout layer named 'mc_dropout'."
        )

    if not np.isclose(
        dropout.rate,
        MC_DROPOUT_RATE,
    ):
        raise ValueError(
            "MC-Dropout rate mismatch."
        )


def build_normalized_input_prefix(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """Build B1→B2→B3 using externally normalized input.

    This preserves the already-validated Phase-4 firmware preprocessing:
    raw features-v1 are normalized in C++, then the normalized 10-vector
    enters this prefix model.
    """

    inputs = tf.keras.Input(
        shape=(INPUT_FEATURES,),
        dtype=tf.float32,
        name="normalized_features_v1",
    )

    x = model.get_layer(
        "block1"
    )(inputs)

    x = model.get_layer(
        "block2"
    )(x)

    outputs = model.get_layer(
        "block3"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_uncertainty_prefix_b3",
    )


def convert_float32_tflite(
    model: tf.keras.Model,
) -> bytes:
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(
            model
        )
    )

    converter.optimizations = []

    converted = converter.convert()

    if not converted:
        raise RuntimeError(
            "TFLite converter returned an empty prefix model."
        )

    return converted


def run_prefix_tflite(
    model_path: Path,
    normalized_features: np.ndarray,
) -> np.ndarray:
    interpreter = tf.lite.Interpreter(
        model_path=str(
            model_path
        ),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental
            .OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )

    interpreter.allocate_tensors()

    inputs = (
        interpreter.get_input_details()
    )

    outputs = (
        interpreter.get_output_details()
    )

    if (
        len(inputs) != 1
        or len(outputs) != 1
    ):
        raise ValueError(
            "Prefix TFLite must have exactly one input/output."
        )

    input_detail = inputs[0]
    output_detail = outputs[0]

    if input_detail[
        "dtype"
    ] != np.float32:
        raise ValueError(
            "Prefix input must be float32."
        )

    if output_detail[
        "dtype"
    ] != np.float32:
        raise ValueError(
            "Prefix output must be float32."
        )

    if tuple(
        input_detail["shape"]
    ) != (
        1,
        INPUT_FEATURES,
    ):
        raise ValueError(
            f"Unexpected prefix input shape: "
            f"{input_detail['shape']}"
        )

    if tuple(
        output_detail["shape"]
    ) != (
        1,
        BLOCK_3_UNITS,
    ):
        raise ValueError(
            f"Unexpected prefix output shape: "
            f"{output_detail['shape']}"
        )

    result = np.empty(
        (
            normalized_features.shape[0],
            BLOCK_3_UNITS,
        ),
        dtype=np.float32,
    )

    for index, row in enumerate(
        normalized_features
    ):
        interpreter.set_tensor(
            input_detail["index"],
            row.reshape(
                1,
                INPUT_FEATURES,
            ).astype(
                np.float32,
                copy=False,
            ),
        )

        interpreter.invoke()

        result[index] = (
            interpreter
            .get_tensor(
                output_detail["index"]
            )
            .reshape(
                BLOCK_3_UNITS
            )
            .astype(
                np.float32
            )
        )

    return result


def selected_validation_indices(
    labels: np.ndarray,
) -> tuple[int, ...]:
    selected: list[int] = []

    for class_id in range(
        len(GESTURES)
    ):
        matches = np.flatnonzero(
            labels == class_id
        )

        if matches.size == 0:
            raise ValueError(
                f"Validation is missing class {class_id}."
            )

        selected.append(
            int(matches[0])
        )

    result = tuple(
        selected
    )

    if result != (
        EXPECTED_VALIDATION_INDICES
    ):
        raise RuntimeError(
            "Frozen validation-vector indices changed. "
            f"Expected {EXPECTED_VALIDATION_INDICES}, "
            f"got {result}."
        )

    return result


def main() -> None:
    set_reproducibility()

    root = project_root()

    model_path = (
        root
        / MODEL_PATH
    )

    output_dir = (
        root
        / OUTPUT_DIR
    )

    reference_normalization_path = (
        root
        / REFERENCE_NORMALIZATION_PATH
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Uncertainty candidate model not found: "
            f"{model_path}"
        )

    ensure_outputs_are_new(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation = (
        load_feature_split(
            "validation"
        )
    )

    if validation.session != (
        EXPECTED_VALIDATION_SESSION
    ):
        raise ValueError(
            "Expected VALIDATION=session_02."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    validate_model_contract(
        model
    )

    normalization_layer, candidate_mean, candidate_variance = (
        candidate_normalization_statistics(
            model
        )
    )

    reference_mean, reference_variance = (
        load_reference_normalization(
            reference_normalization_path
        )
    )

    mean_diff, variance_diff = (
        validate_normalization_compatibility(
            candidate_mean,
            candidate_variance,
            reference_mean,
            reference_variance,
        )
    )

    normalized_by_candidate = (
        normalization_layer(
            validation.features,
            training=False,
        )
        .numpy()
        .astype(
            np.float32
        )
    )

    normalized_by_firmware_contract = (
        (
            validation.features.astype(
                np.float32
            )
            - reference_mean
        )
        / np.sqrt(
            reference_variance
        ).astype(
            np.float32
        )
    ).astype(
        np.float32
    )

    normalization_output_diff = float(
        np.max(
            np.abs(
                normalized_by_candidate
                - normalized_by_firmware_contract
            )
        )
    )

    if (
        normalization_output_diff
        > NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError(
            "Candidate Normalization output does not match "
            "the frozen Phase-4 external normalization. "
            f"max_abs_diff={normalization_output_diff}"
        )

    keras_raw_prefix = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "block3"
        ).output,
        name="keras_raw_prefix_reference",
    )

    keras_b3 = (
        keras_raw_prefix(
            validation.features,
            training=False,
        )
        .numpy()
        .astype(
            np.float32
        )
    )

    prefix_model = (
        build_normalized_input_prefix(
            model
        )
    )

    prefix_bytes = (
        convert_float32_tflite(
            prefix_model
        )
    )

    prefix_path = (
        output_dir
        / PREFIX_FILENAME
    )

    prefix_path.write_bytes(
        prefix_bytes
    )

    tflite_b3 = run_prefix_tflite(
        prefix_path,
        normalized_by_firmware_contract,
    )

    prefix_diff = float(
        np.max(
            np.abs(
                keras_b3
                - tflite_b3
            )
        )
    )

    if (
        prefix_diff
        > PREFIX_PARITY_TOLERANCE
    ):
        raise RuntimeError(
            "B3 prefix TFLite parity failed. "
            f"max_abs_diff={prefix_diff}"
        )

    edge_head = model.get_layer(
        "edge_head"
    )

    kernel, bias = (
        edge_head.get_weights()
    )

    kernel = np.asarray(
        kernel,
        dtype=np.float32,
    )

    bias = np.asarray(
        bias,
        dtype=np.float32,
    )

    if kernel.shape != (
        BLOCK_3_UNITS,
        CLASS_COUNT,
    ):
        raise ValueError(
            "Edge-head kernel shape mismatch."
        )

    if bias.shape != (
        CLASS_COUNT,
    ):
        raise ValueError(
            "Edge-head bias shape mismatch."
        )

    deterministic_keras = (
        model(
            validation.features,
            training=False,
        )
        .numpy()
        .astype(
            np.float32
        )
    )

    deterministic_reconstructed = (
        dense_softmax(
            keras_b3,
            kernel,
            bias,
        )
    )

    deterministic_head_diff = float(
        np.max(
            np.abs(
                deterministic_keras
                - deterministic_reconstructed
            )
        )
    )

    deterministic_top1_agreement = float(
        np.mean(
            np.argmax(
                deterministic_keras,
                axis=1,
            )
            == np.argmax(
                deterministic_reconstructed,
                axis=1,
            )
        )
    )

    if (
        deterministic_head_diff
        > HEAD_PARITY_TOLERANCE
        or deterministic_top1_agreement
        != 1.0
    ):
        raise RuntimeError(
            "Extracted edge-head weights failed deterministic "
            "reconstruction parity."
        )

    selected = (
        selected_validation_indices(
            validation.labels
        )
    )

    selected_b3 = (
        keras_b3[
            list(selected)
        ]
    )

    rng = np.random.default_rng(
        MASK_SEED
    )

    keep_masks = (
        rng.random(
            (
                len(selected),
                MC_DROPOUT_PASSES,
                BLOCK_3_UNITS,
            )
        )
        >= MC_DROPOUT_RATE
    )

    explicit_probabilities = np.empty(
        (
            len(selected),
            MC_DROPOUT_PASSES,
            CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    keras_fixed_mask_probabilities = np.empty_like(
        explicit_probabilities
    )

    for vector_index in range(
        len(selected)
    ):
        for pass_index in range(
            MC_DROPOUT_PASSES
        ):
            mask = (
                keep_masks[
                    vector_index,
                    pass_index,
                ]
            )

            explicit_probabilities[
                vector_index,
                pass_index,
            ] = explicit_dropout_edge_head(
                selected_b3[
                    vector_index
                ],
                mask,
                kernel,
                bias,
                dropout_rate=(
                    MC_DROPOUT_RATE
                ),
            )

            masked = (
                selected_b3[
                    vector_index
                ]
                * mask.astype(
                    np.float32
                )
                * np.float32(
                    1.0
                    / (
                        1.0
                        - MC_DROPOUT_RATE
                    )
                )
            ).astype(
                np.float32
            )

            keras_fixed_mask_probabilities[
                vector_index,
                pass_index,
            ] = (
                edge_head(
                    masked[
                        None,
                        :
                    ],
                    training=False,
                )
                .numpy()
                .reshape(
                    CLASS_COUNT
                )
                .astype(
                    np.float32
                )
            )

    fixed_mask_head_diff = float(
        np.max(
            np.abs(
                explicit_probabilities
                - keras_fixed_mask_probabilities
            )
        )
    )

    if (
        fixed_mask_head_diff
        > HEAD_PARITY_TOLERANCE
    ):
        raise RuntimeError(
            "Explicit masked edge-head parity failed. "
            f"max_abs_diff={fixed_mask_head_diff}"
        )

    probability_tensor = np.transpose(
        explicit_probabilities,
        (
            1,
            0,
            2,
        ),
    )

    uncertainty = (
        compute_mc_dropout_uncertainty_metrics(
            probability_tensor
        )
    )

    per_vector_probability_range = np.max(
        np.max(
            explicit_probabilities,
            axis=1,
        )
        - np.min(
            explicit_probabilities,
            axis=1,
        ),
        axis=1,
    )

    varying_vector_count = int(
        np.count_nonzero(
            per_vector_probability_range
            > 1e-7
        )
    )

    if varying_vector_count == 0:
        raise RuntimeError(
            "Frozen explicit masks produced no stochastic "
            "probability variation."
        )

    edge_head_path = (
        output_dir
        / EDGE_HEAD_FILENAME
    )

    np.savez_compressed(
        edge_head_path,
        kernel=kernel,
        bias=bias,
        dropout_rate=np.asarray(
            MC_DROPOUT_RATE,
            dtype=np.float32,
        ),
        input_units=np.asarray(
            BLOCK_3_UNITS,
            dtype=np.int64,
        ),
        output_units=np.asarray(
            CLASS_COUNT,
            dtype=np.int64,
        ),
    )

    parity_vectors_path = (
        output_dir
        / PARITY_VECTORS_FILENAME
    )

    np.savez_compressed(
        parity_vectors_path,
        validation_indices=np.asarray(
            selected,
            dtype=np.int64,
        ),
        true_classes=validation.labels[
            list(selected)
        ].astype(
            np.int64
        ),
        raw_features=validation.features[
            list(selected)
        ].astype(
            np.float32
        ),
        normalized_inputs=(
            normalized_by_firmware_contract[
                list(selected)
            ]
        ),
        expected_b3=selected_b3,
        keep_masks=keep_masks.astype(
            np.uint8
        ),
        expected_pass_probabilities=(
            explicit_probabilities
        ),
        expected_mean_probabilities=(
            uncertainty.mean_probabilities
        ),
        expected_uncertainty_score=(
            uncertainty
            .normalized_predictive_entropy
        ),
        expected_mean_class_variance=(
            uncertainty
            .mean_class_variance
        ),
        expected_max_class_variance=(
            uncertainty
            .max_class_variance
        ),
    )

    report = {
        "model_version":
            UNCERTAINTY_MODEL_VERSION,
        "dataset_version":
            DATASET_VERSION,
        "feature_version":
            FEATURE_VERSION,
        "test_split_used":
            False,
        "source_model":
            str(MODEL_PATH),
        "source_model_sha256":
            sha256_file(
                model_path
            ),
        "deployment_strategy": {
            "firmware_preprocessing":
                (
                    "reuse frozen Phase-4 external "
                    "features-v1 normalization"
                ),
            "prefix":
                "Float32 TFLite B1->B2->B3",
            "dropout":
                (
                    "explicit Bernoulli mask after B3 "
                    "with inverted-dropout scaling"
                ),
            "edge_head":
                (
                    "manual Float32 Dense(32->5) + Softmax"
                ),
            "passes":
                MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
            "canonical_uncertainty_score":
                "normalized_predictive_entropy",
        },
        "normalization_compatibility": {
            "reference_model_version":
                PRODUCTION_MODEL_VERSION,
            "max_abs_mean_diff":
                mean_diff,
            "max_abs_variance_diff":
                variance_diff,
            "max_abs_output_diff":
                normalization_output_diff,
            "tolerance":
                NORMALIZATION_TOLERANCE,
            "pass":
                True,
        },
        "prefix_tflite": {
            "filename":
                PREFIX_FILENAME,
            "sha256":
                sha256_file(
                    prefix_path
                ),
            "bytes":
                len(
                    prefix_bytes
                ),
            "input_shape":
                [
                    1,
                    INPUT_FEATURES,
                ],
            "output_shape":
                [
                    1,
                    BLOCK_3_UNITS,
                ],
            "validation_samples":
                int(
                    validation.features.shape[0]
                ),
            "max_abs_b3_diff":
                prefix_diff,
            "tolerance":
                PREFIX_PARITY_TOLERANCE,
            "pass":
                True,
        },
        "edge_head": {
            "filename":
                EDGE_HEAD_FILENAME,
            "kernel_shape":
                list(
                    kernel.shape
                ),
            "bias_shape":
                list(
                    bias.shape
                ),
            "deterministic_reconstruction_max_abs_diff":
                deterministic_head_diff,
            "deterministic_top1_agreement":
                deterministic_top1_agreement,
            "fixed_mask_manual_vs_keras_max_abs_diff":
                fixed_mask_head_diff,
            "tolerance":
                HEAD_PARITY_TOLERANCE,
            "pass":
                True,
        },
        "parity_vectors": {
            "filename":
                PARITY_VECTORS_FILENAME,
            "validation_indices":
                list(
                    selected
                ),
            "mask_seed":
                MASK_SEED,
            "shape_keep_masks":
                list(
                    keep_masks.shape
                ),
            "shape_expected_pass_probabilities":
                list(
                    explicit_probabilities.shape
                ),
            "varying_vector_count":
                varying_vector_count,
            "max_probability_range":
                float(
                    np.max(
                        per_vector_probability_range
                    )
                ),
            "note":
                (
                    "Frozen masks are for deterministic parity "
                    "testing only; ESP32 production masks will "
                    "come from the device RNG."
                ),
        },
        "firmware_generated":
            False,
        "esp32_parity_completed":
            False,
        "esp32_stochastic_variation_completed":
            False,
    }

    report_path = (
        output_dir
        / REPORT_FILENAME
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "PHASE 5 / M6 — EDGE UNCERTAINTY EXPORT COMPLETE"
    )
    print(
        "============================================="
    )

    print(
        "Source model:             "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Evaluation source:        "
        "VALIDATION (session_02)"
    )

    print(
        "TEST loaded:              NO"
    )

    print()

    print(
        "Deployment path:"
    )

    print(
        "  features-v1"
        " -> frozen external normalization"
        " -> B1/B2/B3 TFLite"
    )

    print(
        "  -> explicit Dropout(0.2)"
        " -> manual Dense(32->5)"
        " -> Softmax"
    )

    print()

    print(
        "Normalization compatibility:"
    )

    print(
        "  max mean diff:           "
        f"{mean_diff:.9g}"
    )

    print(
        "  max variance diff:       "
        f"{variance_diff:.9g}"
    )

    print(
        "  max normalized diff:     "
        f"{normalization_output_diff:.9g}"
    )

    print(
        "  status:                  PASS"
    )

    print()

    print(
        "B3 prefix TFLite:"
    )

    print(
        "  input/output:            "
        f"(1,{INPUT_FEATURES}) -> "
        f"(1,{BLOCK_3_UNITS})"
    )

    print(
        "  bytes:                   "
        f"{len(prefix_bytes)}"
    )

    print(
        "  SHA-256:                 "
        f"{sha256_file(prefix_path)}"
    )

    print(
        "  validation max abs diff: "
        f"{prefix_diff:.9g}"
    )

    print(
        "  status:                  PASS"
    )

    print()

    print(
        "Edge-head extraction:"
    )

    print(
        "  kernel/bias:             "
        f"{kernel.shape} / {bias.shape}"
    )

    print(
        "  deterministic max diff:  "
        f"{deterministic_head_diff:.9g}"
    )

    print(
        "  deterministic top1:      "
        f"{deterministic_top1_agreement:.6f}"
    )

    print(
        "  fixed-mask max diff:     "
        f"{fixed_mask_head_diff:.9g}"
    )

    print(
        "  status:                  PASS"
    )

    print()

    print(
        "Frozen parity vectors:"
    )

    print(
        "  vectors:                 "
        f"{len(selected)}"
    )

    print(
        "  passes/vector:           "
        f"{MC_DROPOUT_PASSES}"
    )

    print(
        "  masks shape:             "
        f"{keep_masks.shape}"
    )

    print(
        "  vectors with variation:  "
        f"{varying_vector_count}/{len(selected)}"
    )

    print(
        "  max probability range:   "
        f"{float(np.max(per_vector_probability_range)):.9f}"
    )

    print()

    for position, validation_index in enumerate(
        selected
    ):
        class_id = int(
            validation.labels[
                validation_index
            ]
        )

        print(
            f"  {GESTURES[class_id]:<12} "
            f"validation_index={validation_index:<3} "
            f"uncertainty="
            f"{float(uncertainty.normalized_predictive_entropy[position]):.9f}"
        )

    print()

    print(
        "Firmware generated:       NO"
    )

    print(
        "ESP32 parity completed:   NO"
    )

    print(
        "ESP32 stochastic check:   NO"
    )

    print()

    print(
        f"Prefix TFLite:  {prefix_path}"
    )

    print(
        f"Edge head:      {edge_head_path}"
    )

    print(
        f"Parity vectors: {parity_vectors_path}"
    )

    print(
        f"Report:         {report_path}"
    )

    print()
    print(
        "EDGE_UNCERTAINTY_DEPLOYMENT_ASSETS_FROZEN"
    )


if __name__ == "__main__":
    main()
