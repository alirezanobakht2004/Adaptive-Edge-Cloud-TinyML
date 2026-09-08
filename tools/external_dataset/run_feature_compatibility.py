"""Evaluate native compatibility and optionally an explicit 50->100 Hz diagnostic adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.features.extractor import extract_feature_matrix
from ml.features.features_v1 import FEATURE_NAMES, FEATURE_VERSION, extract_features_v1
from .inspect_kaggle_dataset import ROOT, write_json
from .convert_kaggle_dataset import OUTPUT, load_converted, isolated_output

MODEL_DIR = ROOT / "data/processed/dataset-v1/features-v1/models/gesture-model-v1.1.0"
ADAPTER_VERSION = "linear-50-to-100hz-endpoint-hold-v1"


def resample_diagnostic(window, timestamps):
    window, timestamps = np.asarray(window), np.asarray(timestamps)
    if window.shape != (50, 6) or timestamps.shape != (50,) or not np.isfinite(window).all():
        raise ValueError("Diagnostic adapter requires exactly 50 finite six-axis samples")
    relative = timestamps - timestamps[0]
    if not np.array_equal(relative, np.arange(50) * 20):
        raise ValueError("Diagnostic adapter requires verified regular 20 ms intervals")
    target = np.arange(100) * 10
    # np.interp holds the final value at 990 ms, beyond the observed 980 ms endpoint.
    return np.column_stack([np.interp(target, relative, window[:, axis]) for axis in range(6)])


def run(directory=OUTPUT, *, diagnostic_resample=False, output=None):
    directory = isolated_output(directory)
    arrays, metadata = load_converted(directory)
    windows, native_count, rejected = [], 0, []
    labels, ids = [], []
    for index, (start, stop) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:])):
        window = arrays["sensor_values"][start:stop]
        times = arrays["timestamps_ms"][start:stop]
        native = window.shape == (100, 6) and np.array_equal(times - times[0], np.arange(100) * 10)
        native_count += int(native)
        try:
            if not native:
                if not diagnostic_resample:
                    raise ValueError("Native recording violates 100 samples at 100 Hz")
                window = resample_diagnostic(window, times)
            windows.append(window)
            labels.append(arrays["labels"][index])
            ids.append(arrays["sample_ids"][index])
        except ValueError as exc:
            rejected.append({"label": str(arrays["labels"][index]), "sample_id": str(arrays["sample_ids"][index]), "reason": str(exc)})
    units_supported = metadata["units"] == {"acceleration": "g", "gyroscope": "degrees/s"}
    report = {"source_sha256": metadata["source_sha256"], "conversion_version": metadata["conversion_version"],
              "feature_version": FEATURE_VERSION, "expected_feature_dimension": 10,
              "native_window_requirement": [100, 6], "native_sampling_hz_requirement": 100,
              "external_sampling_hz": metadata["sampling_hz_observed"],
              "external_samples_per_recording": metadata["samples_per_recording"],
              "recordings": metadata["recording_count"], "native_compatible_recordings": native_count,
              "native_compatible": native_count == metadata["recording_count"] and units_supported and metadata["axis_orientation"] != "unknown",
              "axis_orientation": metadata["axis_orientation"], "unit_evidence_available": units_supported,
              "diagnostic_adapter": ADAPTER_VERSION if diagnostic_resample else None,
              "adapter_details": "Linear interpolation at 0..990 ms every 10 ms; last 990 ms value holds observed 980 ms endpoint. No axis rotation or unit scaling.",
              "interpolation_is_not_new_measurement": True, "rejected_recordings": rejected,
              "feature_dimension": None, "evaluated_feature_recordings": 0,
              "limitations": ["Upsampling does not recover motion above the original sampling bandwidth.",
                              "Orientation and exact capture firmware/calibration remain unverified.",
                              "Normalization is frozen; dimension compatibility is not distribution compatibility."]}
    if windows and units_supported:
        features = extract_feature_matrix(np.asarray(windows))
        model_metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        mean = np.asarray(model_metadata["normalization"]["mean"])
        std = np.sqrt(np.asarray(model_metadata["normalization"]["variance"]))
        normalized = (features - mean) / np.maximum(std, 1e-7)
        if not np.isfinite(normalized).all():
            raise ValueError("Frozen normalization produced nonfinite values")
        archive = directory / "diagnostic_features_v1.npz"
        np.savez_compressed(archive, features=features, normalized_features=normalized,
                            labels=np.asarray(labels), sample_ids=np.asarray(ids))
        report.update(feature_dimension=int(features.shape[1]), evaluated_feature_recordings=len(features),
                      feature_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      model_version=model_metadata["model_version"], model_sha256=model_metadata["model_sha256"],
                      normalization_refitted=False, maximum_absolute_normalized_feature=float(np.max(np.abs(normalized))),
                      feature_statistics={name: {"mean": float(features[:, i].mean()), "std": float(features[:, i].std()),
                          "normalized_mean": float(normalized[:, i].mean()),
                          "fraction_abs_normalized_above_3": float(np.mean(np.abs(normalized[:, i]) > 3))}
                          for i, name in enumerate(FEATURE_NAMES)})
    else:
        report["blocking_reason"] = "No structurally supported windows or missing unit evidence; no feature archive generated."
    write_json(output or ROOT / "docs/evidence/external_feature_compatibility_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, default=OUTPUT)
    parser.add_argument("--diagnostic-resample", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.converted, diagnostic_resample=args.diagnostic_resample, output=args.output)
    print(json.dumps({key: report[key] for key in ("native_compatible_recordings", "evaluated_feature_recordings", "feature_dimension")}))


if __name__ == "__main__":
    main()
