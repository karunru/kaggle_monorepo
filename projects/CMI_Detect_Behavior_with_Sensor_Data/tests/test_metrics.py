"""Tests for the metrics module."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from metrics import MetricsTracker, calculate_competition_metric, calculate_detailed_metrics


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)

    # Target gestures (BFRB)
    target_gestures = [
        "Above ear - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Neck - pinch skin",
        "Neck - scratch",
        "Cheek - pinch skin",
    ]

    # Non-target gestures
    non_target_gestures = [
        "Drink from bottle/cup",
        "Glasses on/off",
        "Pull air toward your face",
        "Text on phone",
        "Wave hello",
    ]

    all_gestures = target_gestures + non_target_gestures

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(all_gestures)

    # Generate realistic test data
    n_samples = 1000

    # True labels (60% target, 40% non-target to match real distribution)
    true_gestures = (
        np.random.choice(target_gestures, size=int(0.6 * n_samples)).tolist()
        + np.random.choice(non_target_gestures, size=int(0.4 * n_samples)).tolist()
    )
    np.random.shuffle(true_gestures)

    multiclass_true = label_encoder.transform(true_gestures)
    binary_true = np.array([1 if gesture in target_gestures else 0 for gesture in true_gestures])

    # Generate predictions with some realistic accuracy
    # Multiclass: 85% accuracy
    multiclass_preds = multiclass_true.copy()
    wrong_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    for idx in wrong_indices:
        # Wrong prediction from same category (target vs non-target)
        if binary_true[idx] == 1:  # Target
            multiclass_preds[idx] = label_encoder.transform([np.random.choice(target_gestures)])[0]
        else:  # Non-target
            multiclass_preds[idx] = label_encoder.transform([np.random.choice(non_target_gestures)])[0]

    # Binary: 90% accuracy
    binary_preds = binary_true.copy()
    wrong_binary_indices = np.random.choice(n_samples, size=int(0.10 * n_samples), replace=False)
    binary_preds[wrong_binary_indices] = 1 - binary_preds[wrong_binary_indices]

    return {
        "multiclass_preds": multiclass_preds,
        "binary_preds": binary_preds,
        "multiclass_true": multiclass_true,
        "binary_true": binary_true,
        "label_encoder": label_encoder,
        "target_gestures": target_gestures,
        "non_target_gestures": non_target_gestures,
    }


class TestCalculateCompetitionMetric:
    """Test cases for calculate_competition_metric function."""

    def test_perfect_predictions(self, sample_predictions):
        """Test with perfect predictions."""
        data = sample_predictions

        # Perfect predictions
        final_score, binary_f1, macro_f1 = calculate_competition_metric(
            data["multiclass_true"],  # Perfect multiclass
            data["binary_true"],  # Perfect binary
            data["multiclass_true"],
            data["binary_true"],
            data["label_encoder"],
        )

        assert final_score == 1.0
        assert binary_f1 == 1.0
        assert macro_f1 == 1.0

    def test_realistic_predictions(self, sample_predictions):
        """Test with realistic predictions."""
        data = sample_predictions

        final_score, binary_f1, macro_f1 = calculate_competition_metric(
            data["multiclass_preds"],
            data["binary_preds"],
            data["multiclass_true"],
            data["binary_true"],
            data["label_encoder"],
        )

        # Check that scores are reasonable
        assert 0.0 <= final_score <= 1.0
        assert 0.0 <= binary_f1 <= 1.0
        assert 0.0 <= macro_f1 <= 1.0

        # Final score should be average of binary and macro F1
        expected_final = (binary_f1 + macro_f1) / 2
        assert abs(final_score - expected_final) < 1e-10

    def test_all_wrong_predictions(self, sample_predictions):
        """Test with completely wrong predictions."""
        data = sample_predictions

        # Flip all predictions
        wrong_multiclass = np.zeros_like(data["multiclass_true"])
        wrong_binary = 1 - data["binary_true"]

        final_score, binary_f1, macro_f1 = calculate_competition_metric(
            wrong_multiclass, wrong_binary, data["multiclass_true"], data["binary_true"], data["label_encoder"]
        )

        # Scores should be low (but not necessarily 0 due to chance)
        assert 0.0 <= final_score <= 1.0
        assert 0.0 <= binary_f1 <= 1.0
        assert 0.0 <= macro_f1 <= 1.0


class TestCalculateDetailedMetrics:
    """Test cases for calculate_detailed_metrics function."""

    def test_detailed_metrics_structure(self, sample_predictions):
        """Test that detailed metrics returns correct structure."""
        data = sample_predictions

        metrics = calculate_detailed_metrics(
            data["multiclass_preds"],
            data["binary_preds"],
            data["multiclass_true"],
            data["binary_true"],
            data["label_encoder"],
        )

        # Check required keys exist
        required_keys = [
            "competition_score",
            "binary_f1",
            "macro_f1",
            "binary_accuracy",
            "binary_precision",
            "binary_recall",
            "multiclass_accuracy",
            "per_class_f1",
            "target_gesture_f1",
            "non_target_gesture_f1",
            "avg_target_f1",
            "avg_non_target_f1",
        ]

        for key in required_keys:
            assert key in metrics

    def test_detailed_metrics_values(self, sample_predictions):
        """Test that detailed metrics have reasonable values."""
        data = sample_predictions

        metrics = calculate_detailed_metrics(
            data["multiclass_preds"],
            data["binary_preds"],
            data["multiclass_true"],
            data["binary_true"],
            data["label_encoder"],
        )

        # All metrics should be between 0 and 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert 0.0 <= value <= 1.0, f"Metric {key} = {value} is out of range"

    def test_per_class_metrics(self, sample_predictions):
        """Test per-class metrics."""
        data = sample_predictions

        metrics = calculate_detailed_metrics(
            data["multiclass_preds"],
            data["binary_preds"],
            data["multiclass_true"],
            data["binary_true"],
            data["label_encoder"],
        )

        # Check that per-class F1 includes all classes
        all_classes = set(data["label_encoder"].classes_)
        per_class_keys = set(metrics["per_class_f1"].keys())
        assert all_classes == per_class_keys

        # Check target vs non-target separation
        target_classes = set(metrics["target_gesture_f1"].keys())
        non_target_classes = set(metrics["non_target_gesture_f1"].keys())

        assert target_classes.issubset(all_classes)
        assert non_target_classes.issubset(all_classes)
        assert len(target_classes & non_target_classes) == 0  # No overlap


class TestMetricsTracker:
    """Test cases for MetricsTracker class."""

    def test_init(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker()

        assert len(tracker.history) == 6
        assert all(len(values) == 0 for values in tracker.history.values())

    def test_update(self):
        """Test metrics tracker update."""
        tracker = MetricsTracker()

        # Add some epochs
        for epoch in range(5):
            tracker.update(
                epoch=epoch + 1,
                train_loss=1.0 - epoch * 0.1,
                val_loss=1.2 - epoch * 0.1,
                train_score=epoch * 0.1,
                val_score=epoch * 0.08,
                val_binary_f1=epoch * 0.09,
                val_macro_f1=epoch * 0.07,
            )

        # Check that history is updated
        assert len(tracker.history["train_loss"]) == 5
        assert len(tracker.history["val_score"]) == 5

        # Check values are stored correctly
        assert tracker.history["train_loss"][0] == 1.0
        assert tracker.history["val_score"][-1] == 4 * 0.08

    def test_best_epoch(self):
        """Test best epoch identification."""
        tracker = MetricsTracker()

        # Add epochs with varying performance
        val_scores = [0.1, 0.3, 0.8, 0.5, 0.7]
        for epoch, score in enumerate(val_scores):
            tracker.update(
                epoch=epoch + 1,
                train_loss=1.0,
                val_loss=1.0,
                train_score=score,
                val_score=score,
                val_binary_f1=score,
                val_macro_f1=score,
            )

        # Best epoch should be epoch 3 (index 2, score 0.8)
        assert tracker.get_best_epoch() == 2
        assert tracker.get_best_score() == 0.8

    def test_print_summary(self, capsys):
        """Test metrics tracker summary printing."""
        tracker = MetricsTracker()

        # Add some data
        tracker.update(1, 1.0, 1.2, 0.1, 0.08, 0.09, 0.07)
        tracker.update(2, 0.8, 1.0, 0.3, 0.25, 0.28, 0.22)

        # Print summary
        tracker.print_summary()

        # Check that something was printed
        captured = capsys.readouterr()
        assert "TRAINING SUMMARY" in captured.out
        assert "Best Epoch" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
