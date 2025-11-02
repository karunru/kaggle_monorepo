"""
test_exp014_integration.py

exp014の統合テストと性能検証
小規模データでの動作確認とメモリ使用量・処理時間測定
"""

import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import pytest
import torch
from torch.utils.data import DataLoader

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp014"))

from config import Exp014Config
from dataset import IMUDatasetWithMiniRocket, MiniRocketFeatureExtractor
from model import CMISqueezeformerHybrid


def create_small_test_data(num_sequences: int = 5, seq_length_range: tuple = (50, 100)) -> pl.DataFrame:
    """小規模なテストデータを作成"""
    np.random.seed(42)
    torch.manual_seed(42)

    sequence_data = []
    gestures = ["gesture_A", "gesture_B", "gesture_C"]

    for seq_id in range(num_sequences):
        seq_length = np.random.randint(*seq_length_range)
        gesture = gestures[seq_id % len(gestures)]
        subject = f"test_subject_{seq_id % 3}"  # 3人のsubject

        for i in range(seq_length):
            sequence_data.append(
                {
                    # 基本情報
                    "sequence_id": f"test_seq_{seq_id:03d}",
                    "sequence_counter": i,
                    "gesture": gesture,
                    "subject": subject,
                    # IMU原始データ
                    "acc_x": np.random.randn() * 0.5,
                    "acc_y": np.random.randn() * 0.5,
                    "acc_z": np.random.randn() * 0.5 + 9.8,  # 重力成分
                    "rot_x": np.random.randn() * 0.1,
                    "rot_y": np.random.randn() * 0.1,
                    "rot_z": np.random.randn() * 0.1,
                    "rot_w": 1.0 + np.random.randn() * 0.05,
                    # 物理ベース特徴量（MiniRocket用）
                    "linear_acc_x": np.random.randn() * 0.1,
                    "linear_acc_y": np.random.randn() * 0.1,
                    "linear_acc_z": np.random.randn() * 0.1,
                    "angular_vel_x": np.random.randn() * 0.05,
                    "angular_vel_y": np.random.randn() * 0.05,
                    "angular_vel_z": np.random.randn() * 0.05,
                    "angular_distance_x": np.cumsum(np.random.randn(1) * 0.01)[0],
                    "angular_distance_y": np.cumsum(np.random.randn(1) * 0.01)[0],
                    "angular_distance_z": np.cumsum(np.random.randn(1) * 0.01)[0],
                }
            )

    return pl.DataFrame(sequence_data)


def create_small_demographics_data() -> pl.DataFrame:
    """小規模なDemographicsデータを作成"""
    return pl.DataFrame(
        [
            {
                "subject": "test_subject_0",
                "age": 25,
                "sex": "Male",
                "height": 175.0,
                "shoulder_to_wrist": 65.0,
                "elbow_to_wrist": 30.0,
            },
            {
                "subject": "test_subject_1",
                "age": 30,
                "sex": "Female",
                "height": 165.0,
                "shoulder_to_wrist": 60.0,
                "elbow_to_wrist": 28.0,
            },
            {
                "subject": "test_subject_2",
                "age": 35,
                "sex": "Male",
                "height": 180.0,
                "shoulder_to_wrist": 70.0,
                "elbow_to_wrist": 32.0,
            },
        ]
    )


def create_test_config() -> Exp014Config:
    """テスト用の設定を作成"""
    config = Exp014Config()

    # 高速化のためパラメータを小さく設定
    config.minirocket.num_kernels = 100
    config.minirocket.n_jobs = 1
    config.minirocket.random_state = 42
    config.minirocket.cache_enabled = True

    # 一時ディレクトリをキャッシュディレクトリとして使用
    temp_dir = tempfile.mkdtemp()
    config.minirocket.cache_dir = temp_dir

    # モデル設定も小さく
    config.model.d_model = 32
    config.model.n_layers = 2
    config.model.n_heads = 4
    config.model.d_ff = 64
    config.model.fusion_dim = 64
    config.model.dropout = 0.1

    # 訓練設定
    config.training.batch_size = 4
    config.training.learning_rate = 1e-3
    config.training.epochs = 2  # 短時間でテスト

    # Demographics有効化
    config.demographics.enabled = True

    return config


class TestExp014Integration:
    """exp014統合テストクラス"""

    @pytest.fixture(scope="class")
    def test_setup(self):
        """テスト用セットアップ"""
        config = create_test_config()
        test_data = create_small_test_data(num_sequences=10)
        demographics_data = create_small_demographics_data()

        yield {
            "config": config,
            "test_data": test_data,
            "demographics_data": demographics_data,
        }

        # クリーンアップ
        if Path(config.minirocket.cache_dir).exists():
            shutil.rmtree(config.minirocket.cache_dir)

    def test_end_to_end_data_pipeline(self, test_setup):
        """エンドツーエンドデータパイプラインのテスト"""
        config = test_setup["config"]
        test_data = test_setup["test_data"]
        demographics_data = test_setup["demographics_data"]

        print("=== Testing End-to-End Data Pipeline ===")

        # 1. MiniRocket抽出器の学習
        start_time = time.time()
        tracemalloc.start()

        extractor = MiniRocketFeatureExtractor(config)
        extractor.fit(test_data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_time = time.time() - start_time

        print(f"MiniRocket fit time: {fit_time:.2f}s")
        print(f"MiniRocket fit memory: {peak / 1024 / 1024:.2f} MB")
        print(f"MiniRocket output dim: {extractor.output_dim}")

        # 2. データセットの作成
        start_time = time.time()
        tracemalloc.start()

        dataset = IMUDatasetWithMiniRocket(
            df=test_data,
            config=config,
            minirocket_extractor=extractor,
            augment=False,
            demographics_data=demographics_data,
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dataset_time = time.time() - start_time

        print(f"Dataset creation time: {dataset_time:.2f}s")
        print(f"Dataset creation memory: {peak / 1024 / 1024:.2f} MB")
        print(f"Dataset size: {len(dataset)} samples")

        # 3. データローダーでの反復処理
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        start_time = time.time()
        sample_count = 0

        for batch in dataloader:
            sample_count += len(batch["sequence_id"])

            # バッチの基本チェック
            assert "imu" in batch
            assert "minirocket_features" in batch
            assert "demographics" in batch
            assert "attention_mask" in batch

            # テンソルの形状チェック
            batch_size = len(batch["sequence_id"])
            assert batch["imu"].shape[0] == batch_size
            assert batch["minirocket_features"].shape[0] == batch_size

        dataloader_time = time.time() - start_time

        print(f"DataLoader iteration time: {dataloader_time:.2f}s")
        print(f"Processed samples: {sample_count}")

        assert sample_count == len(dataset)

    def test_model_forward_pass_performance(self, test_setup):
        """モデル前向き計算の性能テスト"""
        config = test_setup["config"]
        test_data = test_setup["test_data"]
        demographics_data = test_setup["demographics_data"]

        print("=== Testing Model Forward Pass Performance ===")

        # データセット作成
        dataset = IMUDatasetWithMiniRocket(
            df=test_data,
            config=config,
            minirocket_extractor=None,
            augment=False,
            demographics_data=demographics_data,
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        # モデル作成
        model = CMISqueezeformerHybrid(config=config)
        model.eval()

        # 前向き計算の測定
        inference_times = []
        memory_usages = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # メモリ追跡開始
                tracemalloc.start()
                start_time = time.time()

                # 前向き計算
                multiclass_logits, binary_logits = model(
                    imu=batch["imu"],
                    minirocket_features=batch["minirocket_features"],
                    attention_mask=batch["attention_mask"],
                    demographics=batch["demographics"],
                )

                # 時間とメモリ測定
                inference_time = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                inference_times.append(inference_time)
                memory_usages.append(peak)

                # 出力の基本チェック
                batch_size = len(batch["sequence_id"])
                assert multiclass_logits.shape == (batch_size, config.model.num_classes)
                assert binary_logits.shape == (batch_size, 1)
                assert not torch.isnan(multiclass_logits).any()
                assert not torch.isnan(binary_logits).any()

                if batch_idx >= 2:  # 最初の数バッチのみテスト
                    break

        # 統計情報の出力
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usages) / 1024 / 1024  # MB

        print(f"Average inference time per batch: {avg_inference_time:.4f}s")
        print(f"Average memory usage per batch: {avg_memory_usage:.2f} MB")
        print(f"Samples processed: {sum(len(batch['sequence_id']) for batch in [batch])}")

        # パフォーマンスのしきい値チェック（適宜調整）
        assert avg_inference_time < 1.0  # 1秒以内
        assert avg_memory_usage < 500.0  # 500MB以内

    def test_training_step_performance(self, test_setup):
        """訓練ステップの性能テスト"""
        config = test_setup["config"]
        test_data = test_setup["test_data"]
        demographics_data = test_setup["demographics_data"]

        print("=== Testing Training Step Performance ===")

        # データセット作成
        dataset = IMUDatasetWithMiniRocket(
            df=test_data,
            config=config,
            minirocket_extractor=None,
            augment=True,  # 拡張有効
            demographics_data=demographics_data,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        # モデル作成
        model = CMISqueezeformerHybrid(config=config)
        model.train()

        # オプティマイザ設定
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

        # 訓練ステップの測定
        training_times = []
        memory_usages = []
        losses = []

        for batch_idx, batch in enumerate(dataloader):
            # メモリ追跡開始
            tracemalloc.start()
            start_time = time.time()

            # 訓練ステップ
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=batch_idx)
            loss.backward()
            optimizer.step()

            # 時間とメモリ測定
            training_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            training_times.append(training_time)
            memory_usages.append(peak)
            losses.append(loss.item())

            if batch_idx >= 3:  # 最初の数ステップのみテスト
                break

        # 統計情報の出力
        avg_training_time = np.mean(training_times)
        avg_memory_usage = np.mean(memory_usages) / 1024 / 1024  # MB
        avg_loss = np.mean(losses)

        print(f"Average training time per step: {avg_training_time:.4f}s")
        print(f"Average memory usage per step: {avg_memory_usage:.2f} MB")
        print(f"Average loss: {avg_loss:.4f}")

        # 基本チェック
        assert avg_training_time < 2.0  # 2秒以内
        assert avg_memory_usage < 1000.0  # 1GB以内
        assert avg_loss > 0.0  # 損失が正の値
        assert not np.isnan(avg_loss)  # 損失がNaNでない

    def test_caching_efficiency(self, test_setup):
        """キャッシュ効率性のテスト"""
        config = test_setup["config"]
        test_data = test_setup["test_data"]

        print("=== Testing Caching Efficiency ===")

        # 初回実行（キャッシュ作成）
        start_time = time.time()
        extractor1 = MiniRocketFeatureExtractor(config)
        features1 = extractor1.fit_transform(test_data)
        first_run_time = time.time() - start_time

        # キャッシュファイルの存在確認
        cache_dir = Path(config.minirocket.cache_dir)
        cache_files_after_first = list(cache_dir.glob("*.pkl"))

        print(f"First run time: {first_run_time:.2f}s")
        print(f"Cache files created: {len(cache_files_after_first)}")

        # 2回目実行（キャッシュ読み込み）
        start_time = time.time()
        extractor2 = MiniRocketFeatureExtractor(config)
        features2 = extractor2.fit_transform(test_data)
        second_run_time = time.time() - start_time

        print(f"Second run time: {second_run_time:.2f}s")
        print(f"Speed improvement: {first_run_time / second_run_time:.1f}x")

        # 結果の一致確認
        np.testing.assert_array_equal(features1, features2)

        # キャッシュにより高速化されていることを確認
        assert second_run_time < first_run_time
        assert len(cache_files_after_first) > 0

    def test_memory_leak_detection(self, test_setup):
        """メモリリーク検出のテスト"""
        config = test_setup["config"]
        test_data = test_setup["test_data"]
        demographics_data = test_setup["demographics_data"]

        print("=== Testing Memory Leak Detection ===")

        # 初期メモリ使用量
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_memory:.2f} MB")

        memory_snapshots = [initial_memory]

        # 複数回の処理を実行してメモリ使用量を監視
        for iteration in range(5):
            # データセット作成と処理
            dataset = IMUDatasetWithMiniRocket(
                df=test_data,
                config=config,
                minirocket_extractor=None,
                augment=False,
                demographics_data=demographics_data,
            )

            model = CMISqueezeformerHybrid(config=config)
            model.eval()

            dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

            # 推論実行
            with torch.no_grad():
                for batch in dataloader:
                    multiclass_logits, binary_logits = model(
                        imu=batch["imu"],
                        minirocket_features=batch["minirocket_features"],
                        attention_mask=batch["attention_mask"],
                        demographics=batch["demographics"],
                    )

            # オブジェクト削除
            del dataset, model, dataloader, multiclass_logits, binary_logits

            # ガベージコレクション
            import gc

            gc.collect()

            # メモリ使用量測定
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append(current_memory)
            print(f"Iteration {iteration + 1} memory: {current_memory:.2f} MB")

        # メモリ増加の分析
        memory_increase = memory_snapshots[-1] - memory_snapshots[0]
        print(f"Total memory increase: {memory_increase:.2f} MB")

        # 大きなメモリリークがないことを確認（しきい値は調整可能）
        assert memory_increase < 200.0  # 200MB以内の増加は許容

    def test_scalability_with_different_sizes(self, test_setup):
        """異なるデータサイズでのスケーラビリティテスト"""
        config = test_setup["config"]

        print("=== Testing Scalability with Different Sizes ===")

        data_sizes = [5, 10, 20]  # シーケンス数
        processing_times = []
        memory_usages = []

        for size in data_sizes:
            test_data = create_small_test_data(num_sequences=size)

            # 処理時間とメモリ使用量の測定
            tracemalloc.start()
            start_time = time.time()

            dataset = IMUDatasetWithMiniRocket(
                df=test_data,
                config=config,
                minirocket_extractor=None,
                augment=False,
            )

            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

            model = CMISqueezeformerHybrid(config=config)
            model.eval()

            # 全データの処理
            with torch.no_grad():
                for batch in dataloader:
                    multiclass_logits, binary_logits = model(
                        imu=batch["imu"],
                        minirocket_features=batch["minirocket_features"],
                        attention_mask=batch["attention_mask"],
                    )

            # 測定終了
            processing_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            processing_times.append(processing_time)
            memory_usages.append(peak / 1024 / 1024)  # MB

            print(f"Size {size}: Time {processing_time:.2f}s, Memory {peak / 1024 / 1024:.2f} MB")

        # スケーラビリティの確認
        # 線形的な増加であることを確認（大まかなチェック）
        time_growth_rate = processing_times[-1] / processing_times[0]
        data_growth_rate = data_sizes[-1] / data_sizes[0]

        print(f"Time growth rate: {time_growth_rate:.2f}x")
        print(f"Data growth rate: {data_growth_rate:.2f}x")

        # スケーラビリティが極端に悪くないことを確認
        assert time_growth_rate < data_growth_rate * 2  # データ増加率の2倍以内


if __name__ == "__main__":
    # 統合テストの実行
    pytest.main([__file__, "-v", "-s"])  # -sでprintを表示
