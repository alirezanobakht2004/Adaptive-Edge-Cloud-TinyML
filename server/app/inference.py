"""Validated server-side inference for all three fixed split paths."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from ml.models.cloud_tail_split1 import (
    CLOUD_TAIL_VERSION as SPLIT1_TAIL_VERSION,
    MODEL_DIR as SPLIT1_MODEL_DIR,
    MODEL_PATH as SPLIT1_MODEL_PATH,
    validate_cloud_tail_split1,
)
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
    split: int = 3


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

    expected_version = CLOUD_TAIL_VERSION
    expected_purpose = EXPECTED_MODEL_PURPOSE
    expected_split = EXPECTED_SPLIT_POINT
    expected_dimension = INPUT_EMBEDDING_DIM

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
            self.expected_dimension,
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

    @classmethod
    def _validate_metadata(
        cls, metadata: dict[str, object],
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

        if metadata["model_version"] != cls.expected_version:
            raise RuntimeError(
                "Unexpected cloud-tail model version: "
                f"{metadata['model_version']}"
            )

        if metadata["model_purpose"] != cls.expected_purpose:
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

        if metadata["split_point"] != cls.expected_split:
            raise RuntimeError(
                f"Runtime requires split={cls.expected_split}."
            )

        if (
            metadata["embedding_dimension"]
            != cls.expected_dimension
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
        """Run one inference using this runtime's validated split contract."""

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "embedding must contain numeric values"
            ) from exc

        if vector.shape != (self.embedding_dimension,):
            raise ValueError(
                "embedding must contain exactly "
                f"{self.embedding_dimension} values; "
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
            split=self.split_point,
        )


class Split1CloudInference(Split3CloudInference):
    """64-D Split1 tail with separate version and artifact validation."""

    expected_version = SPLIT1_TAIL_VERSION

    expected_purpose = "phase7-split1-cloud-tail"
    expected_split = 1
    expected_dimension = 64

    def __init__(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        super().__init__(
            SPLIT1_MODEL_PATH if model_path is None else model_path,
            SPLIT1_MODEL_DIR / "metadata.json" if metadata_path is None else metadata_path,
        )
        validate_cloud_tail_split1(self._model)

    @classmethod
    def _validate_metadata(cls, metadata: dict[str, object]) -> None:
        from ml.dataset.loader import CLASS_TO_ID
        from ml.training.train_cloud_tail import SOURCE_MODEL_SHA256
        super()._validate_metadata(metadata)
        expected = {
            "split_id": cls.expected_split,
            "input_embedding_dim": cls.expected_dimension, "output_classes": 5,
            "dataset_version": "dataset-v1", "feature_version": "features-v1",
            "class_to_id": CLASS_TO_ID, "source_edge_model_sha256": SOURCE_MODEL_SHA256,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Invalid Split{cls.expected_split} metadata: {key}")


class Split2CloudInference(Split1CloudInference):
    """48-D continuation with the same strict provenance checks as Split1."""

    expected_version = "gesture-cloud-tail-split2-v1.0.0"
    expected_purpose = "phase7-split2-cloud-tail"
    expected_split = 2
    expected_dimension = 48

    def __init__(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        from ml.models.cloud_tail_split2 import MODEL_PATH, MODEL_DIR, validate_cloud_tail_split2
        Split3CloudInference.__init__(
            self, MODEL_PATH if model_path is None else model_path,
            MODEL_DIR / "metadata.json" if metadata_path is None else metadata_path,
        )
        validate_cloud_tail_split2(self._model)


class SplitCloudInference:
    """Explicit fixed-split routing; no adaptive policy."""

    def __init__(self) -> None:
        self.runtimes = {1: Split1CloudInference(), 2: Split2CloudInference(), 3: Split3CloudInference()}

    def infer(self, embedding: Sequence[float], split: int = 3) -> InferenceResult:
        if isinstance(split, bool) or not isinstance(split, int) or split not in self.runtimes:
            raise ValueError("Supported splits are 1, 2 and 3")
        return self.runtimes[split].infer(embedding)
