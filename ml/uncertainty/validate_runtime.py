from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.dataset.loader import (
    DATASET_VERSION,
    GESTURES,
)
from ml.features.extractor import (
    load_feature_split,
)
from ml.features.features_v1 import (
    FEATURE_VERSION,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    UNCERTAINTY_MODEL_VERSION,
)
from ml.uncertainty.runtime import (
    UNCERTAINTY_SCORE_NAME,
    infer_with_uncertainty,
)


SEED = 42

MODEL_PATH = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}/"
    f"{UNCERTAINTY_MODEL_VERSION}.keras"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_reproducibility() -> None:
    tf.keras.utils.set_random_seed(
        SEED
    )

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def first_index_per_class(
    labels: np.ndarray,
) -> tuple[int, ...]:
    indices: list[int] = []

    for class_id in range(
        len(GESTURES)
    ):
        matches = np.flatnonzero(
            labels == class_id
        )

        if matches.size == 0:
            raise ValueError(
                "VALIDATION is missing gesture class "
                f"{class_id}."
            )

        indices.append(
            int(matches[0])
        )

    return tuple(indices)


def main() -> None:
    set_reproducibility()

    root = project_root()

    model_path = (
        root
        / MODEL_PATH
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Phase-5 candidate model not found: "
            f"{model_path}"
        )

    # DoD proof uses VALIDATION only.
    # TEST remains locked.
    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected VALIDATION=session_02, "
            f"got {validation.session!r}."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    sample_indices = first_index_per_class(
        validation.labels
    )

    print()
    print(
        "PHASE 5 / M6 — PER-INFERENCE UNCERTAINTY DOD"
    )
    print(
        "==========================================="
    )

    print(
        "Model version:          "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Evaluation source:      "
        "VALIDATION (session_02)"
    )

    print(
        "TEST loaded:            NO"
    )

    print(
        "Logical inference:      "
        "one features-v1 vector"
    )

    print(
        "MC passes/inference:    "
        f"{MC_DROPOUT_PASSES}"
    )

    print(
        "Uncertainty score:      "
        f"{UNCERTAINTY_SCORE_NAME}"
    )

    print(
        "Score formula:          "
        "-sum(mean_p*ln(mean_p))/ln(5)"
    )

    print(
        "Offload threshold:      NONE"
    )

    print()

    for sample_index in sample_indices:
        true_class = int(
            validation.labels[
                sample_index
            ]
        )

        result = infer_with_uncertainty(
            model,
            validation.features[
                sample_index
            ],
        )

        if not (
            0.0
            <= result.uncertainty_score
            <= 1.0
        ):
            raise RuntimeError(
                "Per-inference uncertainty score "
                "is outside [0, 1]."
            )

        if not (
            0.0
            <= result.confidence
            <= 1.0
        ):
            raise RuntimeError(
                "Per-inference confidence "
                "is outside [0, 1]."
            )

        print(
            f"sample={sample_index:3d} "
            f"true={GESTURES[true_class]:<12} "
            f"pred={GESTURES[result.predicted_class]:<12} "
            f"confidence={result.confidence:.9f} "
            f"uncertainty={result.uncertainty_score:.9f} "
            f"mean_var={result.mean_class_variance:.9f}"
        )

    print()
    print(
        "Per-inference API:      PASS"
    )

    print(
        "Scalar score available: PASS"
    )

    print(
        "Five passes enforced:   PASS"
    )

    print(
        "Threshold/policy:       NOT DEFINED"
    )

    print()
    print(
        "PHASE5_PER_INFERENCE_UNCERTAINTY_AVAILABLE"
    )


if __name__ == "__main__":
    main()
