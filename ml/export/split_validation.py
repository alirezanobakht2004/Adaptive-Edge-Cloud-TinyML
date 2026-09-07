"""Shared float32 validation utilities for Phase7 fixed split artifacts."""

from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data/processed/dataset-v1/features-v1/models"
SOURCE_VERSION = "gesture-model-v1.1.0"
SPLITS = MODELS / SOURCE_VERSION / "tflite/splits"


def run_float32(path: Path, inputs: np.ndarray, input_dim: int, output_dim: int) -> np.ndarray:
    interpreter = tf.lite.Interpreter(
        model_path=str(path), num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    for detail, dim in ((inp, input_dim), (out, output_dim)):
        if tuple(detail["shape"]) != (1, dim) or detail["dtype"] != np.float32:
            raise ValueError(f"Unexpected float32 tensor contract: {detail['name']}")
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != input_dim or not np.isfinite(inputs).all():
        raise ValueError("Invalid input matrix")
    outputs = []
    for row in inputs:
        interpreter.set_tensor(inp["index"], row[None])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(out["index"])[0])
    result = np.asarray(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite TFLite output")
    return result


def convert_float32(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    return converter.convert()
