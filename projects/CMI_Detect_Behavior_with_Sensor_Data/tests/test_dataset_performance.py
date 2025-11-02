"""IMUDatasetのパフォーマンステスト."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import pytest

# Add src and exp paths
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "exp/exp002"))

from dataset import IMUDataset


def create_test_data(n_sequences: int = 100, seq_length_range: tuple = (50, 300)) -> pl.DataFrame:
    """テスト用のIMUデータを生成."""
    np.random.seed(42)

    data_rows = []
    sequence_counter = 0

    gestures = [
        "Above ear - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Neck - pinch skin",
        "Neck - scratch",
        "Cheek - pinch skin",
        "Drink from bottle/cup",
        "Glasses on/off",
        "Pull air toward your face",
        "Pinch knee/leg skin",
        "Scratch knee/leg skin",
        "Write name on leg",
        "Text on phone",
        "Feel around in tray and pull out an object",
        "Write name in air",
        "Wave hello",
    ]

    for seq_id in range(n_sequences):
        seq_length = np.random.randint(seq_length_range[0], seq_length_range[1])
        gesture = np.random.choice(gestures)

        for step in range(seq_length):
            # IMUデータ生成（7次元）
            row = {
                "sequence_id": seq_id,
                "sequence_counter": step,
                "gesture": gesture,
                "acc_x": np.random.normal(0, 1),
                "acc_y": np.random.normal(0, 1),
                "acc_z": np.random.normal(0, 1),
                "rot_w": np.random.normal(0, 0.5),
                "rot_x": np.random.normal(0, 0.5),
                "rot_y": np.random.normal(0, 0.5),
                "rot_z": np.random.normal(0, 0.5),
            }

            # ランダムに欠損値を挿入
            if np.random.random() < 0.05:  # 5%の確率で欠損
                col = np.random.choice(["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"])
                row[col] = None

            data_rows.append(row)

    return pl.DataFrame(data_rows)


def measure_memory_usage():
    """メモリ使用量を測定."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


class TestDatasetPerformance:
    """データセットパフォーマンステスト."""

    @pytest.fixture
    def small_test_data(self):
        """小規模テストデータ."""
        return create_test_data(n_sequences=50, seq_length_range=(30, 100))

    @pytest.fixture
    def medium_test_data(self):
        """中規模テストデータ."""
        return create_test_data(n_sequences=500, seq_length_range=(50, 200))

    @pytest.fixture
    def large_test_data(self):
        """大規模テストデータ."""
        return create_test_data(n_sequences=2000, seq_length_range=(50, 300))

    def test_small_dataset_performance(self, small_test_data):
        """小規模データセットのパフォーマンステスト."""
        print("\n=== Small Dataset Performance Test ===")
        print(f"Data shape: {small_test_data.shape}")

        # メモリ使用量測定開始
        initial_memory = measure_memory_usage()
        gc.collect()

        # データセット作成時間測定
        start_time = time.time()
        dataset = IMUDataset(small_test_data, target_sequence_length=200, augment=False)
        creation_time = time.time() - start_time

        # メモリ使用量測定終了
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Dataset creation time: {creation_time:.2f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        print(f"Sequences processed: {len(dataset)}")
        print(f"Time per sequence: {creation_time / len(dataset) * 1000:.2f}ms")

        # データ取得テスト
        start_time = time.time()
        sample = dataset[0]
        access_time = time.time() - start_time

        print(f"Data access time: {access_time * 1000:.2f}ms")
        print(f"Sample IMU shape: {sample['imu'].shape}")

        # 基本的な正当性チェック
        assert len(dataset) > 0
        assert sample["imu"].shape == (7, 200)  # [features, seq_len]
        assert "multiclass_label" in sample
        assert "binary_label" in sample

    def test_medium_dataset_performance(self, medium_test_data):
        """中規模データセットのパフォーマンステスト."""
        print("\n=== Medium Dataset Performance Test ===")
        print(f"Data shape: {medium_test_data.shape}")

        # メモリ使用量測定開始
        initial_memory = measure_memory_usage()
        gc.collect()

        # データセット作成時間測定
        start_time = time.time()
        dataset = IMUDataset(medium_test_data, target_sequence_length=200, augment=False)
        creation_time = time.time() - start_time

        # メモリ使用量測定終了
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Dataset creation time: {creation_time:.2f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        print(f"Sequences processed: {len(dataset)}")
        print(f"Time per sequence: {creation_time / len(dataset) * 1000:.2f}ms")

        # スループット測定
        start_time = time.time()
        for i in range(min(100, len(dataset))):
            _ = dataset[i]
        batch_access_time = time.time() - start_time

        print(f"Batch access time (100 samples): {batch_access_time:.2f}s")
        print(f"Average access time: {batch_access_time / 100 * 1000:.2f}ms")

        assert len(dataset) > 0
        assert creation_time < 30.0  # 30秒以内で完了

    def test_large_dataset_performance(self, large_test_data):
        """大規模データセットのパフォーマンステスト."""
        print("\n=== Large Dataset Performance Test ===")
        print(f"Data shape: {large_test_data.shape}")

        # メモリ使用量測定開始
        initial_memory = measure_memory_usage()
        gc.collect()

        # データセット作成時間測定
        start_time = time.time()
        dataset = IMUDataset(large_test_data, target_sequence_length=200, augment=False)
        creation_time = time.time() - start_time

        # メモリ使用量測定終了
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Dataset creation time: {creation_time:.2f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        print(f"Sequences processed: {len(dataset)}")
        print(f"Time per sequence: {creation_time / len(dataset) * 1000:.2f}ms")
        print(f"Memory per sequence: {memory_used / len(dataset):.3f}MB")

        # パフォーマンス要件チェック
        assert len(dataset) > 0
        assert creation_time < 120.0  # 2分以内で完了
        assert memory_used < 2000.0  # 2GB以内

        # ランダムアクセス性能
        start_time = time.time()
        indices = np.random.choice(len(dataset), size=50, replace=False)
        for idx in indices:
            _ = dataset[idx]
        random_access_time = time.time() - start_time

        print(f"Random access time (50 samples): {random_access_time:.2f}s")
        print(f"Average random access: {random_access_time / 50 * 1000:.2f}ms")

    def test_vectorized_vs_original_comparison(self, medium_test_data):
        """ベクトル化処理と元の処理の比較."""
        print("\n=== Vectorized vs Original Comparison ===")

        # 元の処理のシミュレーション（簡易版）
        def simulate_original_processing(df, target_length):
            """元の処理の簡易シミュレーション."""
            sequence_ids = df.get_column("sequence_id").unique().to_list()
            imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

            processed_count = 0
            for seq_id in sequence_ids:
                seq_df = df.filter(pl.col("sequence_id") == seq_id).sort("sequence_counter")
                imu_data = seq_df.select(imu_cols).to_numpy()

                # 簡単な前処理
                imu_data = np.nan_to_num(imu_data, nan=0.0)
                current_length = len(imu_data)

                if current_length != target_length:
                    if current_length < target_length:
                        padding = np.repeat(imu_data[-1:], target_length - current_length, axis=0)
                        imu_data = np.concatenate([imu_data, padding], axis=0)
                    else:
                        indices = np.linspace(0, current_length - 1, target_length)
                        interpolated_data = np.zeros((target_length, imu_data.shape[1]))
                        for i in range(imu_data.shape[1]):
                            interpolated_data[:, i] = np.interp(indices, np.arange(current_length), imu_data[:, i])
                        imu_data = interpolated_data

                processed_count += 1

            return processed_count

        # 元の処理時間測定
        print("Testing original-style processing...")
        start_time = time.time()
        original_count = simulate_original_processing(medium_test_data, 200)
        original_time = time.time() - start_time

        # ベクトル化処理時間測定
        print("Testing vectorized processing...")
        start_time = time.time()
        dataset = IMUDataset(medium_test_data, target_sequence_length=200, augment=False)
        vectorized_time = time.time() - start_time

        print(f"Original processing time: {original_time:.2f}s")
        print(f"Vectorized processing time: {vectorized_time:.2f}s")
        print(f"Speedup: {original_time / vectorized_time:.2f}x")
        print(f"Processed sequences: {len(dataset)}")

        # ベクトル化処理が高速であることを確認
        assert vectorized_time < original_time
        assert len(dataset) == original_count


if __name__ == "__main__":
    # 直接実行時のテスト
    print("Running dataset performance tests...")

    # テストデータ作成
    test_data = create_test_data(n_sequences=200, seq_length_range=(50, 200))

    # パフォーマンステスト実行
    tester = TestDatasetPerformance()
    tester.test_medium_dataset_performance(test_data)
