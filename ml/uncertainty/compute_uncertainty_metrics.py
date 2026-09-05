from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.dataset.loader import (
    DATASET_VERSION,
    GESTURES,
)
from ml.features.features_v1 import FEATURE_VERSION
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
)
from ml.uncertainty.metrics import (
    MCDropoutUncertaintyMetrics,
    compute_mc_dropout_uncertainty_metrics,
)


MODEL_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}"
)

PASSES_FILENAME = (
    "mc_dropout_validation_passes.npz"
)

SOURCE_REPORT_FILENAME = (
    "mc_dropout_validation_report.json"
)

METRICS_NPZ_FILENAME = (
    "mc_dropout_uncertainty_metrics.npz"
)

METRICS_CSV_FILENAME = (
    "mc_dropout_uncertainty_metrics.csv"
)

REPORT_FILENAME = (
    "mc_dropout_uncertainty_report.json"
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


def load_frozen_passes(
    passes_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not passes_path.is_file():
        raise FileNotFoundError(
            "Frozen MC-Dropout pass tensor was not found: "
            f"{passes_path}"
        )

    with np.load(
        passes_path,
        allow_pickle=False,
    ) as archive:
        if "probabilities" not in archive:
            raise KeyError(
                "Frozen pass artifact is missing "
                "'probabilities'."
            )

        if "labels" not in archive:
            raise KeyError(
                "Frozen pass artifact is missing 'labels'."
            )

        probabilities = np.asarray(
            archive["probabilities"],
            dtype=np.float32,
        )

        labels = np.asarray(
            archive["labels"],
            dtype=np.int64,
        )

    if probabilities.ndim != 3:
        raise ValueError(
            "Frozen probabilities must be a 3-D tensor."
        )

    if (
        probabilities.shape[0]
        != MC_DROPOUT_PASSES
    ):
        raise ValueError(
            "Frozen artifact does not contain exactly "
            f"{MC_DROPOUT_PASSES} MC passes."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Frozen labels must be a 1-D vector."
        )

    if (
        probabilities.shape[1]
        != labels.shape[0]
    ):
        raise ValueError(
            "Frozen probability sample count and label "
            "count do not match."
        )

    return probabilities, labels


def ensure_outputs_are_new(
    model_dir: Path,
) -> None:
    protected = (
        model_dir / METRICS_NPZ_FILENAME,
        model_dir / METRICS_CSV_FILENAME,
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
            "entropy/variance evidence: "
            f"{joined}"
        )


def summary_stats(
    values: np.ndarray,
) -> dict[str, float]:
    vector = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        vector.ndim != 1
        or vector.size == 0
    ):
        raise ValueError(
            "Summary input must be a non-empty 1-D vector."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Summary input contains non-finite values."
        )

    return {
        "min": float(
            np.min(vector)
        ),
        "mean": float(
            np.mean(vector)
        ),
        "median": float(
            np.median(vector)
        ),
        "p95": float(
            np.percentile(
                vector,
                95,
            )
        ),
        "max": float(
            np.max(vector)
        ),
    }


def save_metrics_npz(
    path: Path,
    *,
    metrics: MCDropoutUncertaintyMetrics,
    labels: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        mean_probabilities=(
            metrics.mean_probabilities
        ),
        predictive_entropy=(
            metrics.predictive_entropy
        ),
        normalized_predictive_entropy=(
            metrics.normalized_predictive_entropy
        ),
        class_probability_variance=(
            metrics.class_probability_variance
        ),
        mean_class_variance=(
            metrics.mean_class_variance
        ),
        max_class_variance=(
            metrics.max_class_variance
        ),
        max_mean_confidence=(
            metrics.max_mean_confidence
        ),
        predicted_class=(
            metrics.predicted_class
        ),
        labels=labels,
    )


def save_metrics_csv(
    path: Path,
    *,
    metrics: MCDropoutUncertaintyMetrics,
    labels: np.ndarray,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            (
                "sample_index",
                "true_class_id",
                "true_class_name",
                "predicted_class_id",
                "predicted_class_name",
                "correct",
                "max_mean_confidence",
                "predictive_entropy_nats",
                "normalized_predictive_entropy",
                "mean_class_variance",
                "max_class_variance",
            )
        )

        for index in range(
            labels.shape[0]
        ):
            true_class = int(
                labels[index]
            )

            predicted_class = int(
                metrics.predicted_class[index]
            )

            writer.writerow(
                (
                    index,
                    true_class,
                    GESTURES[true_class],
                    predicted_class,
                    GESTURES[predicted_class],
                    int(
                        predicted_class
                        == true_class
                    ),
                    float(
                        metrics.max_mean_confidence[index]
                    ),
                    float(
                        metrics.predictive_entropy[index]
                    ),
                    float(
                        metrics.normalized_predictive_entropy[index]
                    ),
                    float(
                        metrics.mean_class_variance[index]
                    ),
                    float(
                        metrics.max_class_variance[index]
                    ),
                )
            )


def main() -> None:
    root = project_root()
    model_dir = root / MODEL_DIR

    passes_path = (
        model_dir
        / PASSES_FILENAME
    )

    source_report_path = (
        model_dir
        / SOURCE_REPORT_FILENAME
    )

    ensure_outputs_are_new(
        model_dir
    )

    probabilities, labels = (
        load_frozen_passes(
            passes_path
        )
    )

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    if (
        metrics.mean_probabilities.shape
        != (
            labels.shape[0],
            len(GESTURES),
        )
    ):
        raise ValueError(
            "Unexpected mean-probability output shape."
        )

    if np.any(
        metrics.normalized_predictive_entropy
        < -1e-6
    ) or np.any(
        metrics.normalized_predictive_entropy
        > 1.0 + 1e-6
    ):
        raise ValueError(
            "Normalized predictive entropy is outside "
            "its expected [0, 1] range."
        )

    if np.any(
        metrics.class_probability_variance
        < -1e-12
    ):
        raise ValueError(
            "Probability variance cannot be negative."
        )

    metrics_npz_path = (
        model_dir
        / METRICS_NPZ_FILENAME
    )

    metrics_csv_path = (
        model_dir
        / METRICS_CSV_FILENAME
    )

    report_path = (
        model_dir
        / REPORT_FILENAME
    )

    save_metrics_npz(
        metrics_npz_path,
        metrics=metrics,
        labels=labels,
    )

    save_metrics_csv(
        metrics_csv_path,
        metrics=metrics,
        labels=labels,
    )

    mean_prediction_accuracy = float(
        np.mean(
            metrics.predicted_class
            == labels
        )
    )

    report = {
        "model_version":
            UNCERTAINTY_MODEL_VERSION,
        "dataset_version":
            DATASET_VERSION,
        "feature_version":
            FEATURE_VERSION,
        "evaluation_split":
            "validation",
        "evaluation_session":
            "session_02",
        "test_split_used":
            False,
        "source_passes_filename":
            PASSES_FILENAME,
        "source_passes_sha256":
            sha256_file(
                passes_path
            ),
        "source_stochastic_report_sha256":
            (
                sha256_file(
                    source_report_path
                )
                if source_report_path.is_file()
                else None
            ),
        "mc_dropout": {
            "passes":
                MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
        },
        "sample_count":
            int(
                labels.shape[0]
            ),
        "class_count":
            int(
                len(GESTURES)
            ),
        "definitions": {
            "mean_probability":
                "mean over 5 stochastic class probabilities",
            "predictive_entropy":
                "-sum(mean_p * ln(mean_p)), in nats",
            "normalized_predictive_entropy":
                "predictive_entropy / ln(class_count)",
            "class_probability_variance":
                "population variance across the 5 passes, per class",
            "mean_class_variance":
                "mean per-class probability variance for each sample",
            "max_class_variance":
                "maximum per-class probability variance for each sample",
            "max_mean_confidence":
                "maximum mean class probability",
        },
        "mean_prediction_accuracy":
            mean_prediction_accuracy,
        "predictive_entropy_nats":
            summary_stats(
                metrics.predictive_entropy
            ),
        "normalized_predictive_entropy":
            summary_stats(
                metrics.normalized_predictive_entropy
            ),
        "mean_class_variance":
            summary_stats(
                metrics.mean_class_variance
            ),
        "max_class_variance":
            summary_stats(
                metrics.max_class_variance
            ),
        "max_mean_confidence":
            summary_stats(
                metrics.max_mean_confidence
            ),
        "threshold_selected":
            False,
        "calibration_completed":
            False,
        "ambiguous_gesture_test_completed":
            False,
        "final_policy_feature_contract_frozen":
            False,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    entropy_stats = (
        report[
            "normalized_predictive_entropy"
        ]
    )

    mean_variance_stats = (
        report[
            "mean_class_variance"
        ]
    )

    max_variance_stats = (
        report[
            "max_class_variance"
        ]
    )

    confidence_stats = (
        report[
            "max_mean_confidence"
        ]
    )

    print()
    print(
        "PHASE 5 / M6 — ENTROPY / VARIANCE COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        "Source:                 "
        "frozen 5-pass validation artifact"
    )

    print(
        "Model version:          "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Evaluation split:       "
        "validation (session_02)"
    )

    print(
        "Test split loaded:      NO"
    )

    print(
        "Samples:                "
        f"{labels.shape[0]}"
    )

    print(
        "MC passes:              "
        f"{MC_DROPOUT_PASSES}"
    )

    print(
        "Dropout rate:           "
        f"{MC_DROPOUT_RATE}"
    )

    print(
        "Mean-prob prediction "
        "accuracy: "
        f"{mean_prediction_accuracy:.6f}"
    )

    print()

    print(
        "Normalized predictive entropy:"
    )

    print(
        "  min/mean/median/p95/max = "
        f"{entropy_stats['min']:.9f} / "
        f"{entropy_stats['mean']:.9f} / "
        f"{entropy_stats['median']:.9f} / "
        f"{entropy_stats['p95']:.9f} / "
        f"{entropy_stats['max']:.9f}"
    )

    print()

    print(
        "Mean class-probability variance:"
    )

    print(
        "  min/mean/median/p95/max = "
        f"{mean_variance_stats['min']:.9f} / "
        f"{mean_variance_stats['mean']:.9f} / "
        f"{mean_variance_stats['median']:.9f} / "
        f"{mean_variance_stats['p95']:.9f} / "
        f"{mean_variance_stats['max']:.9f}"
    )

    print()

    print(
        "Max class-probability variance:"
    )

    print(
        "  min/mean/median/p95/max = "
        f"{max_variance_stats['min']:.9f} / "
        f"{max_variance_stats['mean']:.9f} / "
        f"{max_variance_stats['median']:.9f} / "
        f"{max_variance_stats['p95']:.9f} / "
        f"{max_variance_stats['max']:.9f}"
    )

    print()

    print(
        "Maximum mean confidence:"
    )

    print(
        "  min/mean/median/p95/max = "
        f"{confidence_stats['min']:.9f} / "
        f"{confidence_stats['mean']:.9f} / "
        f"{confidence_stats['median']:.9f} / "
        f"{confidence_stats['p95']:.9f} / "
        f"{confidence_stats['max']:.9f}"
    )

    print()

    print(
        "Threshold selected:     NO"
    )

    print(
        "Calibration completed:  NO"
    )

    print(
        "Ambiguous-gesture test: NO"
    )

    print()

    print(
        f"Metrics NPZ: {metrics_npz_path}"
    )

    print(
        f"Metrics CSV: {metrics_csv_path}"
    )

    print(
        f"Report:      {report_path}"
    )

    print()
    print(
        "ENTROPY_VARIANCE_METRICS_FROZEN"
    )


if __name__ == "__main__":
    main()
