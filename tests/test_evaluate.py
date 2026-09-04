from __future__ import annotations

import hashlib

from ml.training.evaluate import sha256_file


def test_sha256_file(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"gesture-model")

    expected = hashlib.sha256(
        b"gesture-model"
    ).hexdigest()

    assert sha256_file(artifact) == expected