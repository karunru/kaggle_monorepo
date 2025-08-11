"""
Simple test script for exp014 functionality
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp014"))
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp013"))


def test_exp014_basic():
    """Basic functionality test for exp014"""

    try:
        # Import config
        from config import Exp014Config

        print("✓ Config import successful")

        # Create minimal config
        config = Exp014Config()
        config.minirocket.num_kernels = 10  # Small for testing
        config.minirocket.cache_enabled = False  # Disable cache for simple test
        print("✓ Config creation successful")

        # Create sample data
        np.random.seed(42)
        data = []

        for seq_id in range(2):  # 2 sequences
            seq_length = 50  # Short sequences
            for i in range(seq_length):
                data.append(
                    {
                        "sequence_id": f"test_seq_{seq_id:03d}",
                        "sequence_counter": i,
                        "gesture": f"gesture_{seq_id % 2}",
                        "subject": f"subject_{seq_id}",
                        # IMU features
                        "acc_x": np.random.randn() * 0.5,
                        "acc_y": np.random.randn() * 0.5,
                        "acc_z": np.random.randn() * 0.5 + 9.8,
                        "rot_x": np.random.randn() * 0.1,
                        "rot_y": np.random.randn() * 0.1,
                        "rot_z": np.random.randn() * 0.1,
                        "rot_w": 1.0 + np.random.randn() * 0.05,
                        # Physics features for MiniRocket
                        "linear_acc_x": np.random.randn() * 0.1,
                        "linear_acc_y": np.random.randn() * 0.1,
                        "linear_acc_z": np.random.randn() * 0.1,
                        "linear_acc_mag": np.random.randn() * 0.1,
                        "linear_acc_mag_jerk": np.random.randn() * 0.01,
                        "angular_vel_x": np.random.randn() * 0.05,
                        "angular_vel_y": np.random.randn() * 0.05,
                        "angular_vel_z": np.random.randn() * 0.05,
                        "angular_distance": np.cumsum(np.random.randn(1) * 0.01)[0],
                    }
                )

        df = pl.DataFrame(data)
        print("✓ Sample data creation successful")
        print(f"  - Data shape: {df.shape}")
        print(f"  - Sequences: {len(df.get_column('sequence_id').unique())}")

        # Test MiniRocket extractor
        from dataset import MiniRocketFeatureExtractor

        print("✓ MiniRocketFeatureExtractor import successful")

        extractor = MiniRocketFeatureExtractor(config)
        print("✓ MiniRocketFeatureExtractor creation successful")

        # Fit and transform
        features = extractor.fit_transform(df)
        print("✓ MiniRocket fit_transform successful")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Features dtype: {features.dtype}")

        # Test dataset
        from dataset import IMUDatasetWithMiniRocket

        print("✓ IMUDatasetWithMiniRocket import successful")

        dataset = IMUDatasetWithMiniRocket(df=df, config=config, minirocket_extractor=extractor, augment=False)
        print("✓ IMUDatasetWithMiniRocket creation successful")
        print(f"  - Dataset length: {len(dataset)}")

        # Test data access
        sample = dataset[0]
        print("✓ Dataset __getitem__ successful")
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - IMU shape: {sample['imu'].shape}")
        print(f"  - MiniRocket features shape: {sample['minirocket_features'].shape}")

        print("\n🎉 All exp014 basic tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== exp014 Basic Functionality Test ===")
    success = test_exp014_basic()
    if success:
        print("\n✅ exp014 implementation is working correctly!")
    else:
        print("\n❌ exp014 implementation has issues.")
