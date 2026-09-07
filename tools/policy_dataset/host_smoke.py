"""Measure real Keras calls on existing parity inputs; not an ESP32 benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .collector import initialize, collect
from .instrumentation import PolicySample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    root = Path(__file__).resolve().parents[2]
    directory = root / "data/processed/dataset-v1/features-v1/models/gesture-model-v1.1.0"
    path = directory / "gesture-model-v1.1.0.keras"
    model = tf.keras.models.load_model(path, compile=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    run = datetime.now(timezone.utc).strftime("host-smoke-%Y%m%dT%H%M%S")
    records = []
    with np.load(directory / "tflite/splits/split1_parity_vectors.npz") as vectors:
        for index, raw in enumerate(vectors["raw_features"]):
            sample = PolicySample.create(sample_id=f"host-{index}", device_id="host-smoke",
                                         window_id=str(index), run_id=run, session_id="session_02")
            sample.local(lambda value: np.asarray(value, dtype=np.float32)[None],
                         lambda value: model(value, training=False).numpy()[0].tolist(), raw)
            sample.record["provenance"].update(artifact_hashes={"source_model": digest}, notes=
                "Real host Keras instrumentation smoke only. Preprocessing is ndarray preparation; "
                "model timing includes frozen normalization. Deterministic call, not production five-pass "
                "ESP32 local behavior. Not a comparative benchmark or a policy-training campaign.")
            records.append(sample.record)
    initialize(args.output)
    print(f"Recorded {collect(args.output, records)} host-only smoke observations in {args.output}")


if __name__ == "__main__":
    main()
