from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.features.extractor import FeatureSplit
from ml.training import train_uncertainty


def _fake_split(
    *,
    name: str,
    session: str,
) -> FeatureSplit:
    return FeatureSplit(
        name=name,
        session=session,
        feature_version="features-v1",
        features=np.zeros(
            (2, 10),
            dtype=np.float32,
        ),
        labels=np.zeros(
            2,
            dtype=np.int64,
        ),
        csv_paths=(
            Path(f"{name}_0.csv"),
            Path(f"{name}_1.csv"),
        ),
        metadata_paths=(
            Path(f"{name}_0.json"),
            Path(f"{name}_1.json"),
        ),
    )


def test_uncertainty_training_loads_only_train_and_validation(
    monkeypatch,
) -> None:
    requested: list[str] = []

    splits = {
        "train": _fake_split(
            name="train",
            session="session_01",
        ),
        "validation": _fake_split(
            name="validation",
            session="session_02",
        ),
    }

    def fake_loader(
        split_name: str,
    ) -> FeatureSplit:
        requested.append(split_name)

        if split_name == "test":
            raise AssertionError(
                "TEST split must remain locked."
            )

        return splits[split_name]

    monkeypatch.setattr(
        train_uncertainty,
        "load_feature_split",
        fake_loader,
    )

    train, validation = (
        train_uncertainty
        .load_training_splits()
    )

    assert train.name == "train"
    assert validation.name == "validation"

    assert requested == [
        "train",
        "validation",
    ]


def test_uncertainty_candidate_uses_separate_versioned_output() -> None:
    assert (
        train_uncertainty
        .UNCERTAINTY_MODEL_VERSION
        == "gesture-model-v1.1.0"
    )

    assert (
        "gesture-model-v1.1.0"
        in str(
            train_uncertainty.OUTPUT_DIR
        )
    )

    assert (
        "gesture-model-v1.0.0"
        not in str(
            train_uncertainty.OUTPUT_DIR
        )
    )

    assert (
        train_uncertainty.MODEL_FILENAME
        == "gesture-model-v1.1.0.keras"
    )


def test_uncertainty_training_rejects_wrong_sessions(
    monkeypatch,
) -> None:
    wrong_train = _fake_split(
        name="train",
        session="session_99",
    )

    validation = _fake_split(
        name="validation",
        session="session_02",
    )

    def fake_loader(
        split_name: str,
    ) -> FeatureSplit:
        if split_name == "train":
            return wrong_train

        return validation

    monkeypatch.setattr(
        train_uncertainty,
        "load_feature_split",
        fake_loader,
    )

    try:
        train_uncertainty.load_training_splits()
    except ValueError as exc:
        assert "session_01" in str(exc)
    else:
        raise AssertionError(
            "Wrong TRAIN session was accepted."
        )
