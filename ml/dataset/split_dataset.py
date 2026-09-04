from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


DATASET_VERSION = "dataset-v1"
USER_ID = "user_01"

SPLIT_SESSIONS = {
    "train": "session_01",
    "validation": "session_02",
    "test": "session_03",
}

EXPECTED_PER_CLASS = {
    "train": 120,
    "validation": 40,
    "test": 40,
}

GESTURES = (
    "IDLE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "ROTATE_CW",
    "SHAKE",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def gesture_from_filename(path: Path) -> str:
    match = re.match(
        r"^(idle|swipe_left|swipe_right|rotate_cw|shake)_\d+$",
        path.stem,
    )
    if match is None:
        raise ValueError(f"Invalid dataset-v1 filename: {path.name}")

    return match.group(1).upper()


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_split_manifest() -> dict:
    root = project_root()

    raw_root = root / "data" / "raw" / USER_ID
    metadata_root = root / "data" / "metadata" / USER_ID

    manifest = {
        "dataset_version": DATASET_VERSION,
        "user": USER_ID,
        "split_strategy": "session-based",
        "splits": {},
    }

    total_samples = 0

    for split_name, session_name in SPLIT_SESSIONS.items():
        session_dir = raw_root / session_name

        if not session_dir.is_dir():
            raise FileNotFoundError(f"Missing session directory: {session_dir}")

        csv_paths = sorted(session_dir.glob("*.csv"))
        counts = Counter()
        samples = []

        for csv_path in csv_paths:
            gesture = gesture_from_filename(csv_path)
            metadata_path = (
                metadata_root / session_name / f"{csv_path.stem}.json"
            )

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing metadata for {csv_path}: {metadata_path}"
                )

            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            if metadata.get("dataset_version") != DATASET_VERSION:
                raise ValueError(
                    f"Dataset version mismatch in {metadata_path}"
                )

            if metadata.get("gesture") != gesture:
                raise ValueError(
                    f"Gesture mismatch in {metadata_path}: "
                    f"{metadata.get('gesture')} != {gesture}"
                )

            if metadata.get("session") != session_name:
                raise ValueError(
                    f"Session mismatch in {metadata_path}: "
                    f"{metadata.get('session')} != {session_name}"
                )

            counts[gesture] += 1

            samples.append(
                {
                    "gesture": gesture,
                    "csv": relative_path(csv_path, root),
                    "metadata": relative_path(metadata_path, root),
                }
            )

        expected = EXPECTED_PER_CLASS[split_name]

        for gesture in GESTURES:
            actual = counts[gesture]
            if actual != expected:
                raise ValueError(
                    f"{split_name}/{gesture}: expected {expected}, "
                    f"got {actual}"
                )

        manifest["splits"][split_name] = {
            "session": session_name,
            "count": len(samples),
            "class_counts": {
                gesture: counts[gesture] for gesture in GESTURES
            },
            "samples": samples,
        }

        total_samples += len(samples)

    manifest["total_samples"] = total_samples

    if total_samples != 1000:
        raise ValueError(
            f"dataset-v1 must contain 1000 active samples, got {total_samples}"
        )

    return manifest


def main() -> None:
    root = project_root()
    output_path = root / "data" / "splits" / f"{DATASET_VERSION}.json"

    manifest = build_split_manifest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {manifest['dataset_version']}")
    print(f"Strategy: {manifest['split_strategy']}")

    for split_name, split in manifest["splits"].items():
        counts = ", ".join(
            f"{gesture}={count}"
            for gesture, count in split["class_counts"].items()
        )
        print(
            f"{split_name}: {split['session']} "
            f"-> {split['count']} samples ({counts})"
        )

    print(f"Total: {manifest['total_samples']}")
    print(f"Manifest: {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()