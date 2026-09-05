from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.dataset.loader import (
    CLASS_TO_ID,
    GESTURES,
    load_split,
)
from ml.features.features_v1 import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features_v1,
)


DATASET_VERSION = "dataset-v1"

VECTOR_COUNT = len(GESTURES)

HEADER_RELATIVE_PATH = Path(
    "firmware/test/"
    "test_feature_parity/"
    "feature_parity_vectors.h"
)

MANIFEST_RELATIVE_PATH = Path(
    "data/processed/"
    "dataset-v1/"
    "features-v1/"
    "feature_parity/"
    "feature_parity_manifest.json"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cpp_float(value: float) -> str:
    value = float(
        np.float32(value)
    )

    if not np.isfinite(value):
        raise ValueError(
            "Cannot write non-finite C++ literal."
        )

    text = f"{value:.9g}"

    if (
        "." not in text
        and "e" not in text.lower()
    ):
        text += ".0"

    return f"{text}f"


def select_vectors(
    labels: np.ndarray,
) -> list[int]:
    selected: list[int] = []

    for gesture in GESTURES:
        class_id = CLASS_TO_ID[gesture]

        indices = np.flatnonzero(
            labels == class_id
        )

        if indices.size == 0:
            raise ValueError(
                f"No validation sample for {gesture}."
            )

        selected.append(
            int(indices[0])
        )

    return selected


def write_header(
    *,
    path: Path,
    windows: np.ndarray,
    expected_features: np.ndarray,
    labels: list[str],
) -> None:
    lines: list[str] = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "",
        "namespace feature_parity {",
        "",
        f"constexpr size_t VECTOR_COUNT = {len(labels)};",
        "constexpr size_t WINDOW_SAMPLES = 100;",
        "constexpr size_t SENSOR_CHANNELS = 6;",
        "constexpr size_t FEATURE_COUNT = 10;",
        "",
        "static const char* const LABELS[VECTOR_COUNT] = {",
    ]

    for label in labels:
        lines.append(
            f'    "{label}",'
        )

    lines.extend(
        [
            "};",
            "",
            "static const float WINDOWS"
            "[VECTOR_COUNT]"
            "[WINDOW_SAMPLES]"
            "[SENSOR_CHANNELS] = {",
        ]
    )

    for window in windows:
        lines.append("    {")

        for row in window:
            values = ", ".join(
                cpp_float(value)
                for value in row
            )

            lines.append(
                f"        {{{values}}},"
            )

        lines.append("    },")

    lines.extend(
        [
            "};",
            "",
            "static const float EXPECTED_FEATURES"
            "[VECTOR_COUNT]"
            "[FEATURE_COUNT] = {",
        ]
    )

    for features in expected_features:
        values = ", ".join(
            cpp_float(value)
            for value in features
        )

        lines.append(
            f"    {{{values}}},"
        )

    lines.extend(
        [
            "};",
            "",
            "}  // namespace feature_parity",
            "",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    root = project_root()

    validation = load_split(
        "validation"
    )

    selected_indices = select_vectors(
        validation.labels
    )

    selected_windows = (
        validation.windows[
            selected_indices
        ].astype(np.float32)
    )

    expected_features = np.stack(
        [
            extract_features_v1(window)
            for window in selected_windows
        ]
    ).astype(np.float32)

    selected_labels = [
        GESTURES[
            int(validation.labels[index])
        ]
        for index in selected_indices
    ]

    header_path = (
        root / HEADER_RELATIVE_PATH
    )

    manifest_path = (
        root / MANIFEST_RELATIVE_PATH
    )

    write_header(
        path=header_path,
        windows=selected_windows,
        expected_features=expected_features,
        labels=selected_labels,
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
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
        "vector_count":
            len(selected_indices),
        "feature_names":
            list(FEATURE_NAMES),
        "vectors": [],
    }

    for vector_index, dataset_index in enumerate(
        selected_indices
    ):
        manifest["vectors"].append(
            {
                "vector_index":
                    vector_index,
                "validation_index":
                    dataset_index,
                "gesture":
                    selected_labels[
                        vector_index
                    ],
                "source_csv":
                    validation.csv_paths[
                        dataset_index
                    ].as_posix(),
                "expected_features":
                    expected_features[
                        vector_index
                    ].astype(float).tolist(),
            }
        )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Feature version: {FEATURE_VERSION}"
    )

    print(
        "Source: validation / "
        f"{validation.session}"
    )

    print(
        f"Vectors: {len(selected_indices)}"
    )

    for vector_index, dataset_index in enumerate(
        selected_indices
    ):
        print(
            f"  {selected_labels[vector_index]:<12} "
            f"validation_index={dataset_index}"
        )

    print("Test split was not loaded.")
    print(f"Header:   {header_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()