from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ml.dataset.loader import (
    DATASET_VERSION,
    GESTURES,
)
from ml.features.features_v1 import FEATURE_VERSION
from ml.uncertainty.calibration import (
    DEFAULT_CALIBRATION_BINS,
    CalibrationMetrics,
    evaluate_calibration,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
)


MODEL_DIR = Path(
    "data/processed/"
    f"{DATASET_VERSION}/"
    f"{FEATURE_VERSION}/"
    "models/"
    f"{UNCERTAINTY_MODEL_VERSION}"
)

SOURCE_METRICS_FILENAME = (
    "mc_dropout_uncertainty_metrics.npz"
)

CALIBRATION_CSV_FILENAME = (
    "calibration_reliability_bins.csv"
)

CALIBRATION_PLOT_FILENAME = (
    "calibration_reliability_diagram.png"
)

CALIBRATION_REPORT_FILENAME = (
    "calibration_report.json"
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


def load_frozen_mean_probabilities(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only already-frozen Phase-5 validation outputs."""

    if not path.is_file():
        raise FileNotFoundError(
            "Frozen uncertainty metrics were not found: "
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        required = (
            "mean_probabilities",
            "labels",
        )

        missing = [
            key
            for key in required
            if key not in archive
        ]

        if missing:
            raise KeyError(
                "Frozen uncertainty artifact is missing: "
                + ", ".join(missing)
            )

        probabilities = np.asarray(
            archive[
                "mean_probabilities"
            ],
            dtype=np.float32,
        )

        labels = np.asarray(
            archive["labels"],
            dtype=np.int64,
        )

    if probabilities.shape != (
        labels.shape[0],
        len(GESTURES),
    ):
        raise ValueError(
            "Frozen mean probabilities and labels "
            "have incompatible shapes."
        )

    return probabilities, labels


def ensure_outputs_are_new(
    model_dir: Path,
) -> None:
    protected = (
        model_dir
        / CALIBRATION_CSV_FILENAME,
        model_dir
        / CALIBRATION_PLOT_FILENAME,
        model_dir
        / CALIBRATION_REPORT_FILENAME,
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
            "Refusing to overwrite existing calibration evidence: "
            f"{joined}"
        )


def save_reliability_csv(
    path: Path,
    *,
    metrics: CalibrationMetrics,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            (
                "bin_index",
                "lower_bound",
                "upper_bound",
                "count",
                "mean_confidence",
                "accuracy",
                "absolute_gap",
            )
        )

        for index, reliability_bin in enumerate(
            metrics.reliability_bins
        ):
            writer.writerow(
                (
                    index,
                    reliability_bin.lower,
                    reliability_bin.upper,
                    reliability_bin.count,
                    reliability_bin.mean_confidence,
                    reliability_bin.accuracy,
                    reliability_bin.absolute_gap,
                )
            )


def save_reliability_plot(
    path: Path,
    *,
    metrics: CalibrationMetrics,
) -> None:
    non_empty = [
        reliability_bin
        for reliability_bin
        in metrics.reliability_bins
        if reliability_bin.count > 0
    ]

    figure, axis = plt.subplots(
        figsize=(6, 6)
    )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="perfect calibration",
    )

    if non_empty:
        axis.plot(
            [
                reliability_bin.mean_confidence
                for reliability_bin
                in non_empty
            ],
            [
                reliability_bin.accuracy
                for reliability_bin
                in non_empty
            ],
            marker="o",
            label="validation reliability",
        )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "Mean confidence"
    )

    axis.set_ylabel(
        "Empirical accuracy"
    )

    axis.set_title(
        "MC-Dropout mean-probability reliability"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=150,
    )

    plt.close(figure)


def metrics_to_dict(
    metrics: CalibrationMetrics,
) -> dict:
    return {
        "sample_count":
            metrics.sample_count,
        "bin_count":
            metrics.bin_count,
        "accuracy":
            metrics.accuracy,
        "mean_confidence":
            metrics.mean_confidence,
        "signed_confidence_gap":
            metrics.signed_confidence_gap,
        "absolute_confidence_gap":
            metrics.absolute_confidence_gap,
        "expected_calibration_error":
            metrics.expected_calibration_error,
        "maximum_calibration_error":
            metrics.maximum_calibration_error,
        "negative_log_likelihood":
            metrics.negative_log_likelihood,
        "multiclass_brier_score":
            metrics.multiclass_brier_score,
        "reliability_bins": [
            {
                "lower":
                    reliability_bin.lower,
                "upper":
                    reliability_bin.upper,
                "count":
                    reliability_bin.count,
                "mean_confidence":
                    reliability_bin.mean_confidence,
                "accuracy":
                    reliability_bin.accuracy,
                "absolute_gap":
                    reliability_bin.absolute_gap,
            }
            for reliability_bin
            in metrics.reliability_bins
        ],
    }


def main() -> None:
    root = project_root()
    model_dir = root / MODEL_DIR

    source_path = (
        model_dir
        / SOURCE_METRICS_FILENAME
    )

    ensure_outputs_are_new(
        model_dir
    )

    probabilities, labels = (
        load_frozen_mean_probabilities(
            source_path
        )
    )

    metrics = evaluate_calibration(
        probabilities,
        labels,
        bin_count=DEFAULT_CALIBRATION_BINS,
    )

    csv_path = (
        model_dir
        / CALIBRATION_CSV_FILENAME
    )

    plot_path = (
        model_dir
        / CALIBRATION_PLOT_FILENAME
    )

    report_path = (
        model_dir
        / CALIBRATION_REPORT_FILENAME
    )

    save_reliability_csv(
        csv_path,
        metrics=metrics,
    )

    save_reliability_plot(
        plot_path,
        metrics=metrics,
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
        "source_artifact":
            SOURCE_METRICS_FILENAME,
        "source_artifact_sha256":
            sha256_file(
                source_path
            ),
        "source_distribution":
            (
                "mean class probabilities from the already-frozen "
                "five-pass MC-Dropout validation artifact"
            ),
        "mc_dropout": {
            "passes":
                MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
        },
        "calibration_protocol": {
            "purpose":
                "descriptive evaluation only",
            "probability_transform_fitted":
                False,
            "temperature_scaling_fitted":
                False,
            "binning":
                "equal-width top-label confidence",
            "bin_count":
                DEFAULT_CALIBRATION_BINS,
            "ece_definition":
                (
                    "sum_b (n_b/N) * "
                    "|accuracy_b - mean_confidence_b|"
                ),
            "mce_definition":
                (
                    "maximum absolute reliability gap "
                    "over non-empty bins"
                ),
            "nll_log_base":
                "natural",
            "brier_definition":
                (
                    "mean over samples of "
                    "sum_c((p_c - y_c)^2)"
                ),
        },
        "metrics":
            metrics_to_dict(
                metrics
            ),
        "threshold_selected":
            False,
        "offload_policy_tuned":
            False,
        "test_evaluation_performed":
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

    print()
    print(
        "PHASE 5 / M6 — CALIBRATION EVALUATION COMPLETE"
    )
    print(
        "============================================="
    )

    print(
        "Source:                    "
        "frozen 5-pass mean probabilities"
    )

    print(
        "Model version:             "
        f"{UNCERTAINTY_MODEL_VERSION}"
    )

    print(
        "Evaluation split:          "
        "validation (session_02)"
    )

    print(
        "Test split loaded:         NO"
    )

    print(
        "Samples:                   "
        f"{metrics.sample_count}"
    )

    print(
        "Reliability bins:          "
        f"{metrics.bin_count}"
    )

    print(
        "Probability transform fit: NO"
    )

    print(
        "Temperature scaling fit:   NO"
    )

    print()

    print(
        "Accuracy:                  "
        f"{metrics.accuracy:.9f}"
    )

    print(
        "Mean confidence:           "
        f"{metrics.mean_confidence:.9f}"
    )

    print(
        "Signed confidence gap:     "
        f"{metrics.signed_confidence_gap:.9f}"
    )

    print(
        "Absolute confidence gap:   "
        f"{metrics.absolute_confidence_gap:.9f}"
    )

    print()

    print(
        "ECE (10 equal-width bins): "
        f"{metrics.expected_calibration_error:.9f}"
    )

    print(
        "MCE:                       "
        f"{metrics.maximum_calibration_error:.9f}"
    )

    print(
        "NLL (natural log):         "
        f"{metrics.negative_log_likelihood:.9f}"
    )

    print(
        "Multiclass Brier score:    "
        f"{metrics.multiclass_brier_score:.9f}"
    )

    print()

    print(
        "Non-empty reliability bins:"
    )

    for index, reliability_bin in enumerate(
        metrics.reliability_bins
    ):
        if reliability_bin.count == 0:
            continue

        print(
            f"  bin {index:02d} "
            f"[{reliability_bin.lower:.1f}, "
            f"{reliability_bin.upper:.1f}"
            f"{']' if index == metrics.bin_count - 1 else ')'} "
            f"n={reliability_bin.count:3d} "
            f"conf={reliability_bin.mean_confidence:.6f} "
            f"acc={reliability_bin.accuracy:.6f} "
            f"gap={reliability_bin.absolute_gap:.6f}"
        )

    print()

    print(
        "Threshold selected:        NO"
    )

    print(
        "Offload policy tuned:      NO"
    )

    print(
        "TEST evaluated:            NO"
    )

    print()

    print(
        f"Reliability CSV:  {csv_path}"
    )

    print(
        f"Reliability plot: {plot_path}"
    )

    print(
        f"Report:           {report_path}"
    )

    print()
    print(
        "CALIBRATION_EVIDENCE_FROZEN"
    )


if __name__ == "__main__":
    main()
