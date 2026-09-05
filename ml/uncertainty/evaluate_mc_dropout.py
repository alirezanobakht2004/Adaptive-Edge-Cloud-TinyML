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
from ml.features.extractor import (
    FeatureSplit,
    load_feature_split,
)
from ml.features.features_v1 import (
    FEATURE_VERSION,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
    mc_dropout_predict,
    mc_dropout_variation_diagnostics,
)


SEED = 42
VARIATION_TOLERANCE = 1e-7

MODEL_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}"
)

MODEL_FILENAME = (
    f"{UNCERTAINTY_MODEL_VERSION}.keras"
)

PASSES_FILENAME = (
    "mc_dropout_validation_passes.npz"
)

REPORT_FILENAME = (
    "mc_dropout_validation_report.json"
)


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
    tf.keras.utils.set_random_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_validation_split() -> FeatureSplit:
    """Load VALIDATION only.

    The held-out TEST split remains locked.
    """

    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected VALIDATION=session_02, "
            f"got {validation.session!r}."
        )

    return validation


def ensure_outputs_are_new(
    model_dir: Path,
) -> None:
    protected = (
        model_dir / PASSES_FILENAME,
        model_dir / REPORT_FILENAME,
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
            "Refusing to overwrite existing "
            "MC-Dropout evaluation evidence: "
            f"{joined}"
        )


def main() -> None:
    set_reproducibility()

    root = project_root()
    model_dir = root / MODEL_DIR

    model_path = (
        model_dir
        / MODEL_FILENAME
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "MC-Dropout candidate model not found: "
            f"{model_path}"
        )

    ensure_outputs_are_new(
        model_dir
    )

    validation = (
        load_validation_split()
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    dropout_layer = model.get_layer(
        "mc_dropout"
    )

    if not isinstance(
        dropout_layer,
        tf.keras.layers.Dropout,
    ):
        raise TypeError(
            "Candidate model is missing the "
            "expected Dropout layer."
        )

    if not np.isclose(
        dropout_layer.rate,
        MC_DROPOUT_RATE,
    ):
        raise ValueError(
            "Candidate Dropout rate does not "
            "match the Phase-5 contract."
        )

    probabilities = mc_dropout_predict(
        model,
        validation.features,
        passes=MC_DROPOUT_PASSES,
    )

    diagnostics = (
        mc_dropout_variation_diagnostics(
            probabilities,
            tolerance=VARIATION_TOLERANCE,
        )
    )

    if (
        diagnostics.varying_sample_count
        == 0
    ):
        raise RuntimeError(
            "No stochastic output variation was "
            "detected across the five passes. "
            "Do not continue to entropy/variance "
            "until this is resolved."
        )

    pass_predictions = np.argmax(
        probabilities,
        axis=2,
    )

    pass_accuracies = np.mean(
        pass_predictions
        == validation.labels[None, :],
        axis=1,
    )

    passes_path = (
        model_dir
        / PASSES_FILENAME
    )

    np.savez_compressed(
        passes_path,
        probabilities=probabilities,
        labels=validation.labels.astype(
            np.int64,
            copy=False,
        ),
    )

    report = {
        "model_version":
            UNCERTAINTY_MODEL_VERSION,
        "model_sha256":
            sha256_file(
                model_path
            ),
        "dataset_version":
            DATASET_VERSION,
        "feature_version":
            FEATURE_VERSION,
        "evaluation_split":
            validation.name,
        "evaluation_session":
            validation.session,
        "test_split_used":
            False,
        "seed":
            SEED,
        "mc_dropout": {
            "passes":
                MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
            "training_flag":
                True,
        },
        "probability_tensor_shape":
            list(
                probabilities.shape
            ),
        "variation_diagnostic": {
            "tolerance":
                diagnostics.tolerance,
            "sample_count":
                diagnostics.sample_count,
            "varying_sample_count":
                diagnostics
                .varying_sample_count,
            "varying_sample_fraction":
                (
                    diagnostics
                    .varying_sample_count
                    / diagnostics.sample_count
                ),
            "top1_changed_sample_count":
                diagnostics
                .top1_changed_sample_count,
            "top1_changed_sample_fraction":
                (
                    diagnostics
                    .top1_changed_sample_count
                    / diagnostics.sample_count
                ),
            "max_probability_range":
                diagnostics
                .max_probability_range,
            "stochastic_variation_detected":
                True,
        },
        "per_pass_validation_accuracy":
            [
                float(value)
                for value
                in pass_accuracies
            ],
        "entropy_computed":
            False,
        "variance_uncertainty_computed":
            False,
        "uncertainty_score_defined":
            False,
    }

    report_path = (
        model_dir
        / REPORT_FILENAME
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print(
        "PHASE 5 / M6 — 5 STOCHASTIC PASSES COMPLETE"
    )
    print(
        "==========================================="
    )

    print(
        "Model version:           "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Dataset version:         "
        f"{DATASET_VERSION}"
    )

    print(
        "Feature version:         "
        f"{FEATURE_VERSION}"
    )

    print(
        "Evaluation split:        "
        f"{validation.name} "
        f"({validation.session})"
    )

    print(
        "Test split loaded:       NO"
    )

    print(
        "Dropout rate:            "
        f"{MC_DROPOUT_RATE}"
    )

    print(
        "Stochastic passes:       "
        f"{MC_DROPOUT_PASSES}"
    )

    print(
        "Model call training=True:YES"
    )

    print(
        "Probability tensor:      "
        f"{probabilities.shape}"
    )

    print()

    print(
        "Samples with probability "
        "variation: "
        f"{diagnostics.varying_sample_count}"
        f"/{diagnostics.sample_count}"
    )

    print(
        "Samples with top-1 change: "
        f"{diagnostics.top1_changed_sample_count}"
        f"/{diagnostics.sample_count}"
    )

    print(
        "Max probability range:   "
        f"{diagnostics.max_probability_range:.9f}"
    )

    print(
        "Variation tolerance:     "
        f"{diagnostics.tolerance:.1e}"
    )

    print()

    for index, accuracy in enumerate(
        pass_accuracies,
        start=1,
    ):
        print(
            f"Pass {index} validation accuracy: "
            f"{accuracy:.6f}"
        )

    print()

    print(
        "Stochastic variation:    CONFIRMED"
    )

    print(
        "Entropy:                 NOT COMPUTED"
    )

    print(
        "Variance uncertainty:    NOT COMPUTED"
    )

    print(
        "Uncertainty score:       NOT DEFINED YET"
    )

    print()

    print(
        f"Pass tensor: {passes_path}"
    )

    print(
        f"Report:      {report_path}"
    )

    print()
    print(
        "STOCHASTIC_VARIATION_CONFIRMED"
    )


if __name__ == "__main__":
    main()
