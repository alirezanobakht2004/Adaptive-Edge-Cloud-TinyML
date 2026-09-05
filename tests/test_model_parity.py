from __future__ import annotations

import hashlib

import numpy as np

from ml.export.tflite_export import (
    select_fixed_parity_vectors,
    sha256_file,
)


def test_export_sha256_file(tmp_path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"tinyml")

    expected = hashlib.sha256(
        b"tinyml"
    ).hexdigest()

    assert sha256_file(artifact) == expected


def test_fixed_parity_vector_selection() -> None:
    features = np.zeros(
        (20, 10),
        dtype=np.float32,
    )

    labels = np.repeat(
        np.arange(5, dtype=np.int64),
        4,
    )

    selected = select_fixed_parity_vectors(
        features,
        labels,
    )

    assert selected == [
        0,
        1,
        4,
        5,
        8,
        9,
        12,
        13,
        16,
        17,
    ]