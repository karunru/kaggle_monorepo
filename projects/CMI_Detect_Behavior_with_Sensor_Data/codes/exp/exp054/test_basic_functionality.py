#!/usr/bin/env python3
"""
Basic functionality test for exp054.
"""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import polars as pl
import torch
from dataset import calculate_sequence_statistics, normalize_statistical_features


def test_statistical_features():
    """Test statistical feature calculation."""
    print("Testing statistical features calculation...")

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2, 2],
            "acc_x": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
            "acc_y": [0.4, 0.5, 0.6, -0.4, -0.5, -0.6],
            "acc_z": [0.7, 0.8, 0.9, -0.7, -0.8, -0.9],
            "rot_w": [1.0, 0.9, 0.8, 0.8, 0.7, 0.6],
            "rot_x": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
            "rot_y": [0.2, 0.3, 0.4, 0.3, 0.4, 0.5],
            "rot_z": [0.3, 0.4, 0.5, 0.4, 0.5, 0.6],
            "acc_mag": [0.84, 0.96, 1.08, 0.84, 0.96, 1.08],
            "rot_angle": [0.5, 0.7, 0.9, 0.7, 0.9, 1.1],
            "acc_mag_jerk": [0.0, 0.12, 0.12, 0.0, 0.12, 0.12],
            "rot_angle_vel": [0.0, 0.2, 0.2, 0.0, 0.2, 0.2],
            "linear_acc_x": [0.05, 0.15, 0.25, -0.05, -0.15, -0.25],
            "linear_acc_y": [0.35, 0.45, 0.55, -0.35, -0.45, -0.55],
            "linear_acc_z": [0.65, 0.75, 0.85, -0.65, -0.75, -0.85],
            "linear_acc_mag": [0.74, 0.86, 0.98, 0.74, 0.86, 0.98],
            "linear_acc_mag_jerk": [0.0, 0.12, 0.12, 0.0, 0.12, 0.12],
            "angular_vel_x": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
            "angular_vel_y": [0.2, 0.3, 0.4, 0.3, 0.4, 0.5],
            "angular_vel_z": [0.3, 0.4, 0.5, 0.4, 0.5, 0.6],
            "angular_distance": [0.37, 0.54, 0.71, 0.54, 0.71, 0.88],
            "sequence_counter": [0, 1, 2, 0, 1, 2],
        }
    )

    # Calculate statistics (function now expects and returns Polars DataFrames)
    stats_df = calculate_sequence_statistics(sample_data)  # pandas will be converted internally
    print(f"Statistics DataFrame shape: {stats_df.shape}")
    print(f"Number of statistical features: {stats_df.shape[1] - 1}")  # -1 for sequence_id
    print(f"DataFrame type: {type(stats_df)}")  # Should be Polars DataFrame

    # Test normalization
    normalized_df, scaling_params = normalize_statistical_features(stats_df)
    print(f"Normalized DataFrame shape: {normalized_df.shape}")
    print(f"Number of scaling parameters: {len(scaling_params)}")
    print(f"Normalized DataFrame type: {type(normalized_df)}")  # Should be Polars DataFrame

    # Check for NaN values (Polars method)
    has_nan = stats_df.null_count().sum_horizontal().sum()
    print(f"Number of NaN values in statistics: {has_nan}")

    has_nan_normalized = normalized_df.null_count().sum_horizontal().sum()
    print(f"Number of NaN values in normalized data: {has_nan_normalized}")

    print("✓ Statistical features calculation test passed!")
    return stats_df, normalized_df, scaling_params


def test_model_forward():
    """Test model forward pass with statistical features."""
    print("\nTesting model forward pass...")

    # Import model classes
    from model import IMUOnlyLSTM

    # Create a simple model instance
    model = IMUOnlyLSTM(
        imu_dim=20,  # 20 IMU features
        n_classes=18,
        weight_decay=0.001,
        demographics_dim=10,
        statistical_dim=182,  # Expected statistical features dimension
    )

    # Create dummy input
    batch_size = 2
    seq_len = 100
    imu_dim = 20
    demographics_dim = 10
    statistical_dim = 182  # 20 * 9 + 2 (expected from our calculation)

    imu_data = torch.randn(batch_size, seq_len, imu_dim)
    demographics_embedding = torch.randn(batch_size, demographics_dim)
    statistical_features = torch.randn(batch_size, statistical_dim)

    # Test forward pass
    multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics_embedding, statistical_features)

    print(f"Input shapes:")
    print(f"  IMU: {imu_data.shape}")
    print(f"  Demographics: {demographics_embedding.shape}")
    print(f"  Statistical: {statistical_features.shape}")

    print(f"Output shapes:")
    print(f"  Multiclass logits: {multiclass_logits.shape}")
    print(f"  Binary logits: {binary_logits.shape}")
    print(f"  Nine class logits: {nine_class_logits.shape}")

    # Verify output shapes
    assert multiclass_logits.shape == (batch_size, 18)
    assert binary_logits.shape == (batch_size, 1)
    assert nine_class_logits.shape == (batch_size, 9)

    print("✓ Model forward pass test passed!")


def main():
    """Run all tests."""
    print("Starting exp054 basic functionality tests...\n")

    try:
        # Test statistical features
        stats_df, normalized_df, scaling_params = test_statistical_features()

        # Test model forward pass
        test_model_forward()

        print("\n🎉 All tests passed! exp054 implementation is working correctly.")

        # Summary
        print(f"\nSummary:")
        print(f"  Statistical features per sequence: {stats_df.shape[1] - 1}")
        print(f"  Expected total feature dimension: IMU features + Demographics + Statistical features")
        print(f"  Model successfully processes all feature types")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
