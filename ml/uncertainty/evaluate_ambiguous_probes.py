from __future__ import annotations

import csv
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
from ml.features.features_v1 import FEATURE_VERSION
from ml.models.base_model import INPUT_FEATURES
from ml.uncertainty.ambiguous_probe import (
    AMBIGUITY_PROBE_VERSION,
    MIDPOINT_ALPHA,
    PROBES_PER_CLASS_PAIR,
    AmbiguousProbeBatch,
    build_ambiguous_midpoint_probes,
)
from ml.uncertainty.compute_uncertainty_metrics import (
    summary_stats,
)
from ml.uncertainty.mc_dropout import (
    MC_DROPOUT_PASSES,
    MC_DROPOUT_RATE,
    UNCERTAINTY_MODEL_VERSION,
    mc_dropout_predict,
)
from ml.uncertainty.metrics import (
    MCDropoutUncertaintyMetrics,
    compute_mc_dropout_uncertainty_metrics,
)


SEED = 42

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

CLEAR_METRICS_FILENAME = (
    "mc_dropout_uncertainty_metrics.npz"
)

PROBE_INPUTS_FILENAME = (
    "ambiguity_probe_v1_inputs.npz"
)

PROBE_RESULTS_FILENAME = (
    "ambiguity_probe_v1_results.csv"
)

PROBE_REPORT_FILENAME = (
    "ambiguity_probe_v1_report.json"
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


def load_validation_only() -> FeatureSplit:
    validation = load_feature_split(
        "validation"
    )

    if validation.session != "session_02":
        raise ValueError(
            "Expected VALIDATION=session_02, "
            f"got {validation.session!r}."
        )

    return validation


def load_clear_validation_metrics(
    path: Path,
    *,
    expected_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            "Frozen clear-validation uncertainty metrics "
            f"were not found: {path}"
        )

    required = (
        "normalized_predictive_entropy",
        "mean_class_variance",
        "max_class_variance",
        "max_mean_confidence",
        "labels",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        missing = [
            name
            for name in required
            if name not in archive
        ]

        if missing:
            raise KeyError(
                "Clear-validation metrics artifact is missing: "
                + ", ".join(missing)
            )

        result = {
            name: np.asarray(
                archive[name]
            )
            for name in required
        }

    labels = np.asarray(
        result["labels"],
        dtype=np.int64,
    )

    if not np.array_equal(
        labels,
        expected_labels,
    ):
        raise ValueError(
            "Frozen clear-validation labels do not "
            "match the currently loaded VALIDATION split."
        )

    for name in required[:-1]:
        values = np.asarray(
            result[name],
            dtype=np.float32,
        )

        if values.shape != (
            expected_labels.shape[0],
        ):
            raise ValueError(
                f"Unexpected clear metric shape for {name}: "
                f"{values.shape}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Clear metric {name} contains non-finite values."
            )

        result[name] = values

    result["labels"] = labels

    return result


def normalization_statistics(
    model: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray]:
    layer = model.get_layer(
        "feature_normalization"
    )

    if not isinstance(
        layer,
        tf.keras.layers.Normalization,
    ):
        raise TypeError(
            "Expected a Keras Normalization layer named "
            "'feature_normalization'."
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
            "Unexpected normalization mean shape."
        )

    if variance.shape != (
        INPUT_FEATURES,
    ):
        raise ValueError(
            "Unexpected normalization variance shape."
        )

    return mean, variance


def ensure_outputs_are_new(
    model_dir: Path,
) -> None:
    protected = (
        model_dir / PROBE_INPUTS_FILENAME,
        model_dir / PROBE_RESULTS_FILENAME,
        model_dir / PROBE_REPORT_FILENAME,
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
            "ambiguity-probe evidence: "
            f"{joined}"
        )


def save_probe_inputs(
    path: Path,
    *,
    probes: AmbiguousProbeBatch,
) -> None:
    np.savez_compressed(
        path,
        features=probes.features,
        left_class_id=probes.left_class_id,
        right_class_id=probes.right_class_id,
        left_sample_index=probes.left_sample_index,
        right_sample_index=probes.right_sample_index,
        normalized_distance=probes.normalized_distance,
        alpha=probes.alpha,
    )


def endpoint_mean(
    clear_values: np.ndarray,
    probes: AmbiguousProbeBatch,
) -> np.ndarray:
    values = np.asarray(
        clear_values,
        dtype=np.float32,
    )

    return (
        values[
            probes.left_sample_index
        ]
        + values[
            probes.right_sample_index
        ]
    ) / 2.0


def fraction_true(
    mask: np.ndarray,
) -> float:
    values = np.asarray(
        mask,
        dtype=bool,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Comparison mask must be a non-empty 1-D vector."
        )

    return float(
        np.mean(values)
    )


def pair_key(
    left_class: int,
    right_class: int,
) -> str:
    return (
        f"{GESTURES[left_class]}"
        "__"
        f"{GESTURES[right_class]}"
    )


def per_pair_summary(
    probes: AmbiguousProbeBatch,
    metrics: MCDropoutUncertaintyMetrics,
) -> dict[str, dict]:
    summaries: dict[str, dict] = {}

    unique_pairs = sorted(
        {
            (
                int(left),
                int(right),
            )
            for left, right in zip(
                probes.left_class_id,
                probes.right_class_id,
            )
        }
    )

    for left_class, right_class in unique_pairs:
        mask = (
            (probes.left_class_id == left_class)
            & (
                probes.right_class_id
                == right_class
            )
        )

        summaries[
            pair_key(
                left_class,
                right_class,
            )
        ] = {
            "count": int(
                np.count_nonzero(mask)
            ),
            "normalized_predictive_entropy":
                summary_stats(
                    metrics
                    .normalized_predictive_entropy[
                        mask
                    ]
                ),
            "mean_class_variance":
                summary_stats(
                    metrics
                    .mean_class_variance[
                        mask
                    ]
                ),
            "max_mean_confidence":
                summary_stats(
                    metrics
                    .max_mean_confidence[
                        mask
                    ]
                ),
        }

    return summaries


def save_probe_results_csv(
    path: Path,
    *,
    probes: AmbiguousProbeBatch,
    metrics: MCDropoutUncertaintyMetrics,
    clear_metrics: dict[str, np.ndarray],
) -> None:
    endpoint_entropy = endpoint_mean(
        clear_metrics[
            "normalized_predictive_entropy"
        ],
        probes,
    )

    endpoint_variance = endpoint_mean(
        clear_metrics[
            "mean_class_variance"
        ],
        probes,
    )

    endpoint_confidence = endpoint_mean(
        clear_metrics[
            "max_mean_confidence"
        ],
        probes,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            (
                "probe_index",
                "left_class_id",
                "left_class_name",
                "right_class_id",
                "right_class_name",
                "left_validation_index",
                "right_validation_index",
                "normalized_source_distance",
                "alpha",
                "predicted_class_id",
                "predicted_class_name",
                "normalized_predictive_entropy",
                "mean_class_variance",
                "max_class_variance",
                "max_mean_confidence",
                "endpoint_mean_entropy",
                "endpoint_mean_variance",
                "endpoint_mean_confidence",
            )
        )

        for index in range(
            probes.features.shape[0]
        ):
            left_class = int(
                probes.left_class_id[index]
            )

            right_class = int(
                probes.right_class_id[index]
            )

            predicted = int(
                metrics.predicted_class[index]
            )

            writer.writerow(
                (
                    index,
                    left_class,
                    GESTURES[left_class],
                    right_class,
                    GESTURES[right_class],
                    int(
                        probes.left_sample_index[
                            index
                        ]
                    ),
                    int(
                        probes.right_sample_index[
                            index
                        ]
                    ),
                    float(
                        probes.normalized_distance[
                            index
                        ]
                    ),
                    float(
                        probes.alpha[index]
                    ),
                    predicted,
                    GESTURES[predicted],
                    float(
                        metrics
                        .normalized_predictive_entropy[
                            index
                        ]
                    ),
                    float(
                        metrics.mean_class_variance[
                            index
                        ]
                    ),
                    float(
                        metrics.max_class_variance[
                            index
                        ]
                    ),
                    float(
                        metrics.max_mean_confidence[
                            index
                        ]
                    ),
                    float(
                        endpoint_entropy[index]
                    ),
                    float(
                        endpoint_variance[index]
                    ),
                    float(
                        endpoint_confidence[index]
                    ),
                )
            )


def main() -> None:
    set_reproducibility()

    root = project_root()
    model_dir = root / MODEL_DIR

    model_path = (
        model_dir
        / MODEL_FILENAME
    )

    clear_metrics_path = (
        model_dir
        / CLEAR_METRICS_FILENAME
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
        load_validation_only()
    )

    clear_metrics = (
        load_clear_validation_metrics(
            clear_metrics_path,
            expected_labels=validation.labels,
        )
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    normalization_mean, normalization_variance = (
        normalization_statistics(
            model
        )
    )

    probes = build_ambiguous_midpoint_probes(
        validation.features,
        validation.labels,
        normalization_mean=normalization_mean,
        normalization_variance=normalization_variance,
        probes_per_class_pair=(
            PROBES_PER_CLASS_PAIR
        ),
        alpha=MIDPOINT_ALPHA,
    )

    expected_probe_count = (
        len(GESTURES)
        * (len(GESTURES) - 1)
        // 2
        * PROBES_PER_CLASS_PAIR
    )

    if (
        probes.features.shape
        != (
            expected_probe_count,
            INPUT_FEATURES,
        )
    ):
        raise ValueError(
            "Unexpected ambiguity-probe shape: "
            f"{probes.features.shape}."
        )

    probabilities = mc_dropout_predict(
        model,
        probes.features,
        passes=MC_DROPOUT_PASSES,
    )

    metrics = (
        compute_mc_dropout_uncertainty_metrics(
            probabilities
        )
    )

    endpoint_entropy = endpoint_mean(
        clear_metrics[
            "normalized_predictive_entropy"
        ],
        probes,
    )

    endpoint_variance = endpoint_mean(
        clear_metrics[
            "mean_class_variance"
        ],
        probes,
    )

    endpoint_confidence = endpoint_mean(
        clear_metrics[
            "max_mean_confidence"
        ],
        probes,
    )

    entropy_higher_fraction = fraction_true(
        metrics
        .normalized_predictive_entropy
        > endpoint_entropy
    )

    variance_higher_fraction = fraction_true(
        metrics.mean_class_variance
        > endpoint_variance
    )

    confidence_lower_fraction = fraction_true(
        metrics.max_mean_confidence
        < endpoint_confidence
    )

    probe_inputs_path = (
        model_dir
        / PROBE_INPUTS_FILENAME
    )

    probe_results_path = (
        model_dir
        / PROBE_RESULTS_FILENAME
    )

    probe_report_path = (
        model_dir
        / PROBE_REPORT_FILENAME
    )

    save_probe_inputs(
        probe_inputs_path,
        probes=probes,
    )

    save_probe_results_csv(
        probe_results_path,
        probes=probes,
        metrics=metrics,
        clear_metrics=clear_metrics,
    )

    clear_entropy_stats = summary_stats(
        clear_metrics[
            "normalized_predictive_entropy"
        ]
    )

    probe_entropy_stats = summary_stats(
        metrics
        .normalized_predictive_entropy
    )

    clear_variance_stats = summary_stats(
        clear_metrics[
            "mean_class_variance"
        ]
    )

    probe_variance_stats = summary_stats(
        metrics.mean_class_variance
    )

    clear_confidence_stats = summary_stats(
        clear_metrics[
            "max_mean_confidence"
        ]
    )

    probe_confidence_stats = summary_stats(
        metrics.max_mean_confidence
    )

    report = {
        "ambiguity_probe_version":
            AMBIGUITY_PROBE_VERSION,
        "probe_type":
            "controlled-feature-space-midpoint",
        "important_scope_note":
            (
                "Constructed from real VALIDATION features; "
                "not a newly captured physical gesture dataset "
                "and not training data."
            ),
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
        "source_split":
            "validation",
        "source_session":
            validation.session,
        "test_split_used":
            False,
        "seed":
            SEED,
        "class_count":
            len(GESTURES),
        "class_pair_count":
            (
                len(GESTURES)
                * (len(GESTURES) - 1)
                // 2
            ),
        "probes_per_class_pair":
            PROBES_PER_CLASS_PAIR,
        "probe_count":
            expected_probe_count,
        "interpolation_alpha":
            MIDPOINT_ALPHA,
        "source_pair_selection":
            (
                "closest cross-class pairs in model-normalized "
                "features-v1 space"
            ),
        "mc_dropout": {
            "passes":
                MC_DROPOUT_PASSES,
            "dropout_rate":
                MC_DROPOUT_RATE,
            "training_flag":
                True,
        },
        "clear_validation_reference": {
            "normalized_predictive_entropy":
                clear_entropy_stats,
            "mean_class_variance":
                clear_variance_stats,
            "max_mean_confidence":
                clear_confidence_stats,
        },
        "ambiguity_probe": {
            "normalized_predictive_entropy":
                probe_entropy_stats,
            "mean_class_variance":
                probe_variance_stats,
            "max_mean_confidence":
                probe_confidence_stats,
        },
        "paired_comparisons": {
            "fraction_probe_entropy_gt_endpoint_mean":
                entropy_higher_fraction,
            "fraction_probe_variance_gt_endpoint_mean":
                variance_higher_fraction,
            "fraction_probe_confidence_lt_endpoint_mean":
                confidence_lower_fraction,
        },
        "per_class_pair":
            per_pair_summary(
                probes,
                metrics,
            ),
        "threshold_selected":
            False,
        "calibration_completed":
            False,
        "physical_live_ambiguity_validated":
            False,
    }

    with probe_report_path.open(
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
        "PHASE 5 / M6 — CONTROLLED AMBIGUITY PROBE COMPLETE"
    )
    print(
        "=================================================="
    )

    print(
        "Probe version:            "
        f"{AMBIGUITY_PROBE_VERSION}"
    )

    print(
        "Probe type:               "
        "feature-space midpoint"
    )

    print(
        "Source:                   "
        "real VALIDATION features"
    )

    print(
        "Physical capture claim:   NO"
    )

    print(
        "Training-data claim:      NO"
    )

    print(
        "Test split loaded:        NO"
    )

    print(
        "Class pairs:              "
        f"{len(GESTURES) * (len(GESTURES) - 1) // 2}"
    )

    print(
        "Probes per pair:          "
        f"{PROBES_PER_CLASS_PAIR}"
    )

    print(
        "Total probes:             "
        f"{expected_probe_count}"
    )

    print(
        "Interpolation alpha:      "
        f"{MIDPOINT_ALPHA:.2f}"
    )

    print(
        "MC passes:                "
        f"{MC_DROPOUT_PASSES}"
    )

    print()

    print(
        "Normalized predictive entropy"
    )

    print(
        "  clear validation mean/median/p95 = "
        f"{clear_entropy_stats['mean']:.9f} / "
        f"{clear_entropy_stats['median']:.9f} / "
        f"{clear_entropy_stats['p95']:.9f}"
    )

    print(
        "  ambiguity probe mean/median/p95  = "
        f"{probe_entropy_stats['mean']:.9f} / "
        f"{probe_entropy_stats['median']:.9f} / "
        f"{probe_entropy_stats['p95']:.9f}"
    )

    print()

    print(
        "Mean class-probability variance"
    )

    print(
        "  clear validation mean/median/p95 = "
        f"{clear_variance_stats['mean']:.9f} / "
        f"{clear_variance_stats['median']:.9f} / "
        f"{clear_variance_stats['p95']:.9f}"
    )

    print(
        "  ambiguity probe mean/median/p95  = "
        f"{probe_variance_stats['mean']:.9f} / "
        f"{probe_variance_stats['median']:.9f} / "
        f"{probe_variance_stats['p95']:.9f}"
    )

    print()

    print(
        "Maximum mean confidence"
    )

    print(
        "  clear validation mean/median/p05* = "
        f"{clear_confidence_stats['mean']:.9f} / "
        f"{clear_confidence_stats['median']:.9f} / "
        f"{clear_confidence_stats['min']:.9f}"
    )

    print(
        "  ambiguity probe mean/median/min   = "
        f"{probe_confidence_stats['mean']:.9f} / "
        f"{probe_confidence_stats['median']:.9f} / "
        f"{probe_confidence_stats['min']:.9f}"
    )

    print(
        "  *clear third value is min, printed as "
        "the low-confidence reference."
    )

    print()

    print(
        "Paired probe vs source endpoints"
    )

    print(
        "  entropy higher:          "
        f"{entropy_higher_fraction:.6f}"
    )

    print(
        "  variance higher:         "
        f"{variance_higher_fraction:.6f}"
    )

    print(
        "  confidence lower:        "
        f"{confidence_lower_fraction:.6f}"
    )

    print()

    print(
        "Per class pair "
        "(mean entropy / mean confidence):"
    )

    pair_summaries = report[
        "per_class_pair"
    ]

    for key in sorted(
        pair_summaries
    ):
        pair = pair_summaries[key]

        entropy_mean = (
            pair[
                "normalized_predictive_entropy"
            ][
                "mean"
            ]
        )

        confidence_mean = (
            pair[
                "max_mean_confidence"
            ][
                "mean"
            ]
        )

        print(
            f"  {key:<28} "
            f"H={entropy_mean:.6f} "
            f"conf={confidence_mean:.6f}"
        )

    print()

    print(
        "Threshold selected:       NO"
    )

    print(
        "Calibration completed:    NO"
    )

    print(
        "Physical live ambiguity:  NOT VALIDATED YET"
    )

    print()

    print(
        f"Probe inputs:  {probe_inputs_path}"
    )

    print(
        f"Probe results: {probe_results_path}"
    )

    print(
        f"Probe report:  {probe_report_path}"
    )

    print()
    print(
        "AMBIGUITY_PROBE_EVIDENCE_FROZEN"
    )


if __name__ == "__main__":
    main()
