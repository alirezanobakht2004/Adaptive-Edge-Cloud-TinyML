from __future__ import annotations

import json
from pathlib import Path

from ml.export.tflite_export import (
    MODEL_DIR,
    project_root,
)


INTERNAL_REPORT_FILENAME = (
    "int8_internal_tensor_report.json"
)

DEPLOYMENT_REPORT_FILENAME = (
    "int8_normalized_input_report.json"
)

OUTPUT_HEADER = Path(
    "firmware/test/"
    "test_internal_tensor_diagnostic/"
    "internal_tensor_vectors.h"
)

EXPECTED_STAGES = (
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "SOFTMAX",
)

PROBE_INDICES = (
    82,
    119,
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def cpp_int8_array(
    values: list[int],
) -> str:
    return ", ".join(
        str(int(value))
        for value in values
    )


def main() -> None:
    root = project_root()

    tflite_dir = (
        root
        / MODEL_DIR
        / "tflite"
    )

    internal_report = load_json(
        tflite_dir
        / INTERNAL_REPORT_FILENAME
    )

    deployment_report = load_json(
        tflite_dir
        / DEPLOYMENT_REPORT_FILENAME
    )

    if (
        internal_report.get("test_split_used")
        is not False
    ):
        raise ValueError(
            "Internal tensor report unexpectedly "
            "used the held-out test split."
        )

    if (
        internal_report.get("source_split")
        != "validation"
    ):
        raise ValueError(
            "Expected validation source split."
        )

    if (
        internal_report.get("source_session")
        != "session_02"
    ):
        raise ValueError(
            "Expected validation/session_02."
        )

    model_hash = internal_report[
        "model_sha256"
    ]

    expected_model_hash = (
        deployment_report[
            "int8_tflite_sha256"
        ]
    )

    if model_hash != expected_model_hash:
        raise RuntimeError(
            "Internal tensor report does not "
            "belong to the frozen deployment model.\n"
            f"Internal report: {model_hash}\n"
            f"Deployment:      {expected_model_hash}"
        )

    operators = internal_report[
        "operators"
    ]

    operator_names = tuple(
        operator["stage"]
        for operator in operators
    )

    if operator_names != EXPECTED_STAGES:
        raise RuntimeError(
            "Unexpected operator stage order: "
            f"{operator_names}"
        )

    probe_map = {
        int(probe["validation_index"]): probe
        for probe
        in internal_report["probe_vectors"]
    }

    for index in PROBE_INDICES:
        if index not in probe_map:
            raise RuntimeError(
                f"Probe vector {index} "
                "is missing from the report."
            )

    output_path = (
        root / OUTPUT_HEADER
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "#pragma once",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "namespace internal_tensor_vectors {",
        "",
        f'constexpr char MODEL_SHA256[] = "{model_hash}";',
        "",
        f"constexpr size_t PROBE_COUNT = {len(PROBE_INDICES)};",
        f"constexpr size_t STAGE_COUNT = {len(EXPECTED_STAGES)};",
        "constexpr size_t INPUT_COUNT = 10;",
        "",
        "struct StageExpectation {",
        "    const char* name;",
        "    int tensor_index;",
        "    size_t value_count;",
        "    const int8_t* expected;",
        "};",
        "",
        "struct ProbeExpectation {",
        "    int validation_index;",
        "    int true_class;",
        "    const int8_t* input;",
        "    const StageExpectation* stages;",
        "};",
        "",
    ]

    for probe_index in PROBE_INDICES:
        probe = probe_map[
            probe_index
        ]

        lines.extend(
            [
                (
                    f"static const int8_t "
                    f"VECTOR_{probe_index}_INPUT"
                    "[INPUT_COUNT] = {"
                ),
                "    "
                + cpp_int8_array(
                    probe["input_int8"]
                ),
                "};",
                "",
            ]
        )

        stages = probe[
            "stages"
        ]

        stage_names = tuple(
            stage["stage"]
            for stage in stages
        )

        if stage_names != EXPECTED_STAGES:
            raise RuntimeError(
                f"Unexpected stage order for "
                f"vector {probe_index}: "
                f"{stage_names}"
            )

        for stage in stages:
            stage_name = (
                stage["stage"]
            )

            values = stage[
                "values"
            ]

            expected_count = int(
                stage["value_count"]
            )

            if len(values) != expected_count:
                raise RuntimeError(
                    f"{stage_name} vector "
                    f"{probe_index} value count "
                    "does not match report."
                )

            lines.extend(
                [
                    (
                        f"static const int8_t "
                        f"VECTOR_{probe_index}_"
                        f"{stage_name}"
                        f"[{expected_count}] = {{"
                    ),
                    "    "
                    + cpp_int8_array(
                        values
                    ),
                    "};",
                    "",
                ]
            )

        lines.append(
            (
                f"static const StageExpectation "
                f"VECTOR_{probe_index}_STAGES"
                "[STAGE_COUNT] = {"
            )
        )

        for stage in stages:
            stage_name = (
                stage["stage"]
            )

            tensor_index = int(
                stage["tensor_index"]
            )

            value_count = int(
                stage["value_count"]
            )

            lines.append(
                "    {"
                f'"{stage_name}", '
                f"{tensor_index}, "
                f"{value_count}, "
                f"VECTOR_{probe_index}_{stage_name}"
                "},"
            )

        lines.extend(
            [
                "};",
                "",
            ]
        )

    lines.extend(
        [
            (
                "static const ProbeExpectation "
                "PROBES[PROBE_COUNT] = {"
            ),
        ]
    )

    for probe_index in PROBE_INDICES:
        probe = probe_map[
            probe_index
        ]

        lines.append(
            "    {"
            f"{probe_index}, "
            f"{int(probe['true_class'])}, "
            f"VECTOR_{probe_index}_INPUT, "
            f"VECTOR_{probe_index}_STAGES"
            "},"
        )

    lines.extend(
        [
            "};",
            "",
            "}  // namespace internal_tensor_vectors",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        "ESP32 INTERNAL TENSOR DIAGNOSTIC EXPORT"
    )

    print(
        "---------------------------------------"
    )

    print(
        f"Model SHA-256: "
        f"{model_hash}"
    )

    print(
        "Source:        "
        "validation / session_02"
    )

    print(
        "Probe vectors: "
        + ", ".join(
            str(index)
            for index in PROBE_INDICES
        )
    )

    print()

    for operator in operators:
        print(
            f"{operator['stage']:<7} "
            f"tensor={operator['tensor_index']} "
            f"shape={operator['shape']}"
        )

    print()

    print(
        "Test split was not loaded."
    )

    print(
        f"Header: {output_path}"
    )


if __name__ == "__main__":
    main()