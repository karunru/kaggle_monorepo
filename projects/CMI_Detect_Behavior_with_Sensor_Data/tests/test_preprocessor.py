"""Tests for the preprocessor module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessor import DataPreprocessor, process_sequences, resample_sequence


@pytest.fixture
def sample_data():
    """Create sample sensor data for testing."""
    np.random.seed(42)

    # Create sample data
    data = {
        "sequence_id": ["SEQ_001"] * 50 + ["SEQ_002"] * 50,
        "sequence_counter": list(range(50)) + list(range(50)),
        "subject": ["SUBJ_001"] * 50 + ["SUBJ_002"] * 50,
        "phase": ["Transition"] * 25 + ["Gesture"] * 25 + ["Transition"] * 25 + ["Gesture"] * 25,
        "gesture": ["Above ear - pull hair"] * 50 + ["Text on phone"] * 50,
        # IMU data
        "acc_x": np.random.normal(0, 1, 100),
        "acc_y": np.random.normal(0, 1, 100),
        "acc_z": np.random.normal(9.8, 1, 100),
        "rot_w": np.random.normal(0, 0.5, 100),
        "rot_x": np.random.normal(0, 0.5, 100),
        "rot_y": np.random.normal(0, 0.5, 100),
        "rot_z": np.random.normal(0, 0.5, 100),
        # Thermopile data
        "thm_1": np.random.normal(25, 2, 100),
        "thm_2": np.random.normal(25, 2, 100),
        "thm_3": np.random.normal(25, 2, 100),
        "thm_4": np.random.normal(25, 2, 100),
        "thm_5": np.random.normal(25, 2, 100),
    }

    # Add ToF data (simplified - just first few pixels)
    for i in range(1, 6):
        for j in range(10):  # Simplified - only 10 pixels instead of 64
            data[f"tof_{i}_v{j}"] = np.random.randint(0, 255, 100)

    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    def test_init(self, preprocessor):
        """Test preprocessor initialization."""
        assert len(preprocessor.imu_cols) == 7
        assert len(preprocessor.thm_cols) == 5
        assert len(preprocessor.tof_cols) == 320

    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        # Introduce some missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[5:10, "acc_x"] = np.nan
        data_with_missing.loc[15:20, "thm_1"] = np.nan
        data_with_missing.loc[25:30, "tof_1_v0"] = -1

        result = preprocessor.handle_missing_values(data_with_missing)

        # Check that missing values are handled
        assert not result["acc_x"].isna().any()
        assert not result["thm_1"].isna().any()
        assert (result["tof_1_v0"] == -1).sum() == 0  # -1 should be replaced with 0

    def test_normalize_features(self, preprocessor, sample_data):
        """Test feature normalization."""
        result = preprocessor.normalize_features(sample_data)

        # Check IMU normalization (Z-score)
        for col in ["acc_x", "acc_y", "acc_z"]:
            assert abs(result[col].mean()) < 1e-10  # Should be close to 0
            assert abs(result[col].std() - 1) < 1e-10  # Should be close to 1

        # Check thermopile normalization (Min-Max)
        for col in ["thm_1", "thm_2"]:
            assert result[col].min() >= 0
            assert result[col].max() <= 1

    def test_feature_engineering(self, preprocessor, sample_data):
        """Test feature engineering."""
        result = preprocessor.feature_engineering(sample_data)

        # Check that magnitude features are added
        assert "acc_magnitude" in result.columns
        assert "rot_magnitude" in result.columns

        # Check thermopile statistics
        assert "thm_mean" in result.columns
        assert "thm_std" in result.columns
        assert "thm_max" in result.columns
        assert "thm_min" in result.columns

    def test_preprocess_sequence(self, preprocessor, sample_data):
        """Test full preprocessing pipeline."""
        result = preprocessor.preprocess_sequence(sample_data)

        # Check that all steps are applied
        assert len(result) == len(sample_data)
        assert "acc_magnitude" in result.columns
        assert not result.isna().any().any()


class TestResampleSequence:
    """Test cases for resample_sequence function."""

    def test_resample_same_length(self, sample_data):
        """Test resampling when target length equals current length."""
        numeric_data = sample_data.select_dtypes(include=[np.number])
        result = resample_sequence(numeric_data, target_length=len(numeric_data))

        assert result.shape[0] == len(numeric_data)
        np.testing.assert_array_equal(result, numeric_data.values)

    def test_resample_longer(self, sample_data):
        """Test resampling to longer sequence."""
        numeric_data = sample_data.select_dtypes(include=[np.number])
        target_length = len(numeric_data) * 2
        result = resample_sequence(numeric_data, target_length=target_length)

        assert result.shape[0] == target_length
        assert result.shape[1] == numeric_data.shape[1]

    def test_resample_shorter(self, sample_data):
        """Test resampling to shorter sequence."""
        numeric_data = sample_data.select_dtypes(include=[np.number])
        target_length = len(numeric_data) // 2
        result = resample_sequence(numeric_data, target_length=target_length)

        assert result.shape[0] == target_length
        assert result.shape[1] == numeric_data.shape[1]

    def test_resample_empty_data(self):
        """Test resampling with empty data."""
        empty_df = pd.DataFrame()
        result = resample_sequence(empty_df, target_length=10)

        assert result.shape[0] == 10
        assert result.shape[1] == 341  # Default number of columns


class TestProcessSequences:
    """Test cases for process_sequences function."""

    def test_process_sequences(self, sample_data, preprocessor):
        """Test full sequence processing."""
        sequences, labels, subjects = process_sequences(sample_data, preprocessor)

        # Check output shapes
        assert len(sequences) == 2  # Two unique sequences
        assert len(labels) == 2
        assert len(subjects) == 2

        # Check sequence shape (should be 200 timesteps)
        assert sequences[0].shape[0] == 200

        # Check labels and subjects
        assert "Above ear - pull hair" in labels
        assert "Text on phone" in labels
        assert "SUBJ_001" in subjects
        assert "SUBJ_002" in subjects


if __name__ == "__main__":
    pytest.main([__file__])
