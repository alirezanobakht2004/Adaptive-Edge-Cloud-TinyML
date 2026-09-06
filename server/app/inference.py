"""Phase 6 / M7 server-side inference for the fixed Split-3 path."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from ml.models.cloud_model import (
    CLASS_COUNT,
    CLOUD_TAIL_VERSION,
    INPUT_EMBEDDING_DIM,
    SOURCE_EDGE_MODEL_VERSION,
)


EXPECTED_SPLIT_POINT = 3
EXPECTED_MODEL_PURPOSE = "phase6-fixed-split3-cloud-tail"

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "dataset-v1"
    / "features-v1"
    / "models"
    / CLOUD_TAIL_VERSION
)

MODEL_PATH = MODEL_DIR / f"{CLOUD_TAIL_VERSION}.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"


@dataclass(frozen=True)
class InferenceResult:
    """Result returned by one server-side cloud-tail inference."""

    predicted_class: str
    confidence: float
    server_latency_ms: float
    model_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


class Split3CloudInference:
    """Validated runtime for the fixed 32-D Split-3 cloud tail."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Cloud-tail model not found: {self.model_path}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Cloud-tail metadata not found: {self.metadata_path}"
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        self._validate_metadata(metadata)

        actual_sha256 = sha256_file(self.model_path)

        if actual_sha256 != metadata["model_sha256"]:
            raise RuntimeError(
                "Cloud-tail SHA-256 mismatch: "
                f"actual={actual_sha256}, "
                f"metadata={metadata['model_sha256']}"
            )

        self.model_sha256 = actual_sha256
        self.model_version = metadata["model_version"]
        self.source_edge_model_version = metadata[
            "source_edge_model_version"
        ]
        self.split_point = int(metadata["split_point"])
        self.embedding_dimension = int(
            metadata["embedding_dimension"]
        )

        class_to_id = metadata["class_to_id"]

        self._id_to_class = {
            int(class_id): class_name
            for class_name, class_id in class_to_id.items()
        }

        expected_ids = set(range(CLASS_COUNT))

        if set(self._id_to_class) != expected_ids:
            raise RuntimeError(
                "Cloud-tail class IDs are not exactly "
                f"{sorted(expected_ids)}."
            )

        self._model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        if self._model.input_shape != (
            None,
            INPUT_EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "Unexpected cloud-tail input shape: "
                f"{self._model.input_shape}"
            )

        if self._model.output_shape != (
            None,
            CLASS_COUNT,
        ):
            raise RuntimeError(
                "Unexpected cloud-tail output shape: "
                f"{self._model.output_shape}"
            )

    @staticmethod
    def _validate_metadata(
        metadata: dict[str, object],
    ) -> None:
        required_fields = (
            "model_version",
            "model_purpose",
            "source_edge_model_version",
            "source_edge_model_sha256",
            "split_point",
            "embedding_dimension",
            "test_split_used",
            "class_to_id",
            "model_sha256",
        )

        for field in required_fields:
            if field not in metadata:
                raise RuntimeError(
                    f"Cloud-tail metadata missing field: {field}"
                )

        if metadata["model_version"] != CLOUD_TAIL_VERSION:
            raise RuntimeError(
                "Unexpected cloud-tail model version: "
                f"{metadata['model_version']}"
            )

        if metadata["model_purpose"] != EXPECTED_MODEL_PURPOSE:
            raise RuntimeError(
                "Unexpected cloud-tail model purpose: "
                f"{metadata['model_purpose']}"
            )

        if (
            metadata["source_edge_model_version"]
            != SOURCE_EDGE_MODEL_VERSION
        ):
            raise RuntimeError(
                "Unexpected source edge model version: "
                f"{metadata['source_edge_model_version']}"
            )

        if metadata["split_point"] != EXPECTED_SPLIT_POINT:
            raise RuntimeError(
                "Phase-6 runtime supports fixed Split 3 only."
            )

        if (
            metadata["embedding_dimension"]
            != INPUT_EMBEDDING_DIM
        ):
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{metadata['embedding_dimension']}"
            )

        if metadata["test_split_used"] is not False:
            raise RuntimeError(
                "Cloud-tail metadata indicates TEST usage."
            )

        if not isinstance(metadata["class_to_id"], dict):
            raise RuntimeError(
                "class_to_id must be a JSON object."
            )

    def infer(
        self,
        embedding: Sequence[float],
    ) -> InferenceResult:
        """Run one fixed Split-3 server-tail inference."""

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "embedding must contain numeric values"
            ) from exc

        if vector.shape != (INPUT_EMBEDDING_DIM,):
            raise ValueError(
                "embedding must contain exactly "
                f"{INPUT_EMBEDDING_DIM} values; "
                f"got shape {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "embedding must contain only finite values"
            )

        start_ns = time.perf_counter_ns()

        probabilities = self._model(
            vector[np.newaxis, :],
            training=False,
        ).numpy()[0]

        end_ns = time.perf_counter_ns()

        server_latency_ms = (
            end_ns - start_ns
        ) / 1_000_000.0

        if probabilities.shape != (CLASS_COUNT,):
            raise RuntimeError(
                "Unexpected cloud-tail probability shape: "
                f"{probabilities.shape}"
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Cloud-tail output contains NaN or infinity."
            )

        probability_sum = float(
            np.sum(probabilities)
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-5,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Cloud-tail probabilities do not sum to 1: "
                f"{probability_sum}"
            )

        predicted_class_id = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class_id]
        )

        return InferenceResult(
            predicted_class=self._id_to_class[
                predicted_class_id
            ],
            confidence=confidence,
            server_latency_ms=server_latency_ms,
            model_version=self.model_version,
        )
