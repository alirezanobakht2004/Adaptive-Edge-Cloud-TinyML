from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ml.dataset.loader import CLASS_TO_ID, GESTURES
from ml.features.extractor import (
    FeatureSplit,
    load_feature_split,
)
from ml.features.features_v1 import (
    FEATURE_NAMES,
    FEATURE_VERSION,
)


DEFAULT_OUTPUT = Path(
    "data/processed/dataset-v1/features-v1/eda"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path

    return project_root() / path


def write_summary_csv(
    splits: tuple[FeatureSplit, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = (
        "split",
        "session",
        "gesture",
        "feature_version",
        "feature_index",
        "feature_name",
        "count",
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for split in splits:
            for gesture in GESTURES:
                class_id = CLASS_TO_ID[gesture]
                class_mask = split.labels == class_id

                for feature_index, feature_name in enumerate(
                    FEATURE_NAMES
                ):
                    values = split.features[
                        class_mask,
                        feature_index,
                    ].astype(np.float64)

                    q25, median, q75 = np.percentile(
                        values,
                        [25.0, 50.0, 75.0],
                    )

                    writer.writerow(
                        (
                            split.name,
                            split.session,
                            gesture,
                            split.feature_version,
                            feature_index + 1,
                            feature_name,
                            values.size,
                            float(np.mean(values)),
                            float(np.std(values, ddof=0)),
                            float(np.min(values)),
                            float(q25),
                            float(median),
                            float(q75),
                            float(np.max(values)),
                        )
                    )


def write_distribution_plots(
    splits: tuple[FeatureSplit, ...],
    output_dir: Path,
) -> None:
    for split in splits:
        split_dir = output_dir / split.name
        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for feature_index, feature_name in enumerate(
            FEATURE_NAMES
        ):
            all_values = split.features[
                :,
                feature_index,
            ].astype(np.float64)

            bin_edges = np.histogram_bin_edges(
                all_values,
                bins=20,
            )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            for gesture in GESTURES:
                class_id = CLASS_TO_ID[gesture]
                values = split.features[
                    split.labels == class_id,
                    feature_index,
                ]

                axis.hist(
                    values,
                    bins=bin_edges,
                    alpha=0.45,
                    label=gesture,
                )

            axis.set_title(
                f"{split.name}: {feature_name}"
            )
            axis.set_xlabel(feature_name)
            axis.set_ylabel("Count")
            axis.legend()
            axis.grid(
                axis="y",
                alpha=0.25,
            )

            figure.tight_layout()

            filename = (
                f"{feature_index + 1:02d}_"
                f"{feature_name}.png"
            )

            figure.savefig(
                split_dir / filename,
                dpi=150,
            )

            plt.close(figure)


def print_split_summary(split: FeatureSplit) -> None:
    print(
        f"{split.name}: "
        f"session={split.session}, "
        f"X={split.features.shape}, "
        f"y={split.labels.shape}"
    )

    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]
        count = int(
            np.sum(split.labels == class_id)
        )

        print(
            f"  {gesture:<12} {count}"
        )


def run_eda(output_dir: Path) -> None:
    # Test split is deliberately excluded from Phase-3 EDA.
    train = load_feature_split("train")
    validation = load_feature_split("validation")

    splits = (
        train,
        validation,
    )

    for split in splits:
        print_split_summary(split)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir / "feature_summary.csv"
    )

    write_summary_csv(
        splits,
        summary_path,
    )

    write_distribution_plots(
        splits,
        output_dir,
    )

    print()
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Summary: {summary_path}")
    print(f"Plots: {output_dir}")
    print("Test split was not loaded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase-3 EDA for dataset-v1 "
            "using train and validation only."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    run_eda(
        resolve_output_dir(args.output_dir)
    )


if __name__ == "__main__":
    main()