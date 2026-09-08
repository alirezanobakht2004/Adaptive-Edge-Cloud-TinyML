import numpy as np
import pytest
from tools.external_dataset.evaluate_existing_model import uncertainty, describe, mapped_evaluation


def test_uncertainty_limits_and_histogram_counts():
    probabilities = np.asarray([[1., 0, 0, 0, 0], [.2] * 5], dtype=np.float32)
    metrics = uncertainty(probabilities)
    np.testing.assert_allclose(metrics["entropy"], [0, np.log(5)])
    np.testing.assert_allclose(metrics["margin"], [1, 0])
    assert sum(describe(probabilities)["confidence"]["histogram_counts"]) == 2
    assert sum(describe(probabilities)["entropy"]["histogram_counts"]) == 2


def test_unmapped_labels_never_count_as_ground_truth():
    p = np.asarray([[1., 0, 0, 0, 0], [0, 1., 0, 0, 0]])
    mapping = {"idle": {"target": "IDLE", "basis": "fixture definition"}, "wave_left": {"target": None, "basis": "unmapped"}}
    result = mapped_evaluation(p, ["idle", "wave_left"], mapping)
    assert result["accuracy_evaluated_samples"] == 1
    assert result["mapped_subset_accuracy"] == 1
    assert result["overall_external_accuracy"] is None
    with pytest.raises(ValueError, match="explicit mapping"):
        mapped_evaluation(p, ["idle", "different"], mapping)


def test_frozen_mc_inference_repeats_without_weight_updates():
    from ml.training.train_cloud_tail import load_source_model
    from tools.external_dataset.evaluate_existing_model import infer, MODEL_DIR
    model = load_source_model(MODEL_DIR / "gesture-model-v1.1.0.keras")
    model.trainable = False
    weights = [x.copy() for x in model.get_weights()]
    features = np.linspace(-1, 1, 100, dtype=np.float32).reshape(10, 10)
    first, second = infer(model, features), infer(model, features)
    np.testing.assert_array_equal(first["mc5"], second["mc5"])
    np.testing.assert_allclose(first["mc5"].sum(axis=1), 1, atol=1e-6)
    for before, after in zip(weights, model.get_weights()):
        np.testing.assert_array_equal(before, after)
