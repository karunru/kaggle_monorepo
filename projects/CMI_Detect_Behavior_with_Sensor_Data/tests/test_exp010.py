"""exp010実装のテスト."""

import sys
from pathlib import Path

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes"))

import numpy as np
import polars as pl
import pytest
import torch

# exp010モジュールを直接インポートできるようにパスを調整
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp010"))

from config import Config
from dataset import (
    IMUDataModule,
    SingleSequenceDataset,
    calculate_angular_distance,
    calculate_angular_velocity_from_quat,
    remove_gravity_from_acc,
)
from model import CMIBERTModel


class TestFeatureEngineering:
    """特徴量エンジニアリング関数のテスト."""

    def test_remove_gravity_from_acc(self):
        """重力除去のテスト."""
        # テストデータ作成
        acc_x = np.array([0.0, 0.1, 0.2])
        acc_y = np.array([0.0, 0.1, 0.2])
        acc_z = np.array([9.81, 9.91, 10.01])  # 重力加速度を含む
        quat_w = np.array([1.0, 0.999, 0.998])
        quat_x = np.array([0.0, 0.01, 0.02])
        quat_y = np.array([0.0, 0.01, 0.02])
        quat_z = np.array([0.0, 0.01, 0.02])

        # 重力除去
        acc_no_grav = remove_gravity_from_acc(acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z)

        # 結果の確認
        assert acc_no_grav.shape == (3, 3)
        # 重力が除去されているか確認（完全にゼロにはならないが、大幅に減少するはず）
        assert np.mean(np.abs(acc_no_grav[2])) < np.mean(np.abs(acc_z))

    def test_calculate_angular_velocity_from_quat(self):
        """角速度計算のテスト."""
        # テストデータ作成（回転するクォータニオン）
        t = np.linspace(0, 1, 100)
        angle = t * np.pi / 2  # 90度回転
        quat_w = np.cos(angle / 2)
        quat_x = np.sin(angle / 2)
        quat_y = np.zeros_like(t)
        quat_z = np.zeros_like(t)

        # 角速度計算
        ang_vel = calculate_angular_velocity_from_quat(quat_w, quat_x, quat_y, quat_z)

        # 結果の確認
        assert ang_vel.shape == (3, 100)
        # X軸周りの回転なので、ang_vel[0]が非ゼロ
        assert np.mean(np.abs(ang_vel[0])) > 0

    def test_calculate_angular_distance(self):
        """角距離計算のテスト."""
        # テストデータ作成
        quat_w = np.array([1.0, 0.707, 0.0])  # 0度、90度、180度回転
        quat_x = np.array([0.0, 0.707, 1.0])
        quat_y = np.zeros(3)
        quat_z = np.zeros(3)

        # 角距離計算
        ang_dist = calculate_angular_distance(quat_w, quat_x, quat_y, quat_z)

        # 結果の確認
        assert len(ang_dist) == 3
        assert ang_dist[0] == 0.0  # 最初は常に0
        assert ang_dist[1] > 0  # 90度回転
        assert ang_dist[2] > ang_dist[1]  # さらに回転


class TestSingleSequenceDataset:
    """SingleSequenceDatasetのテスト."""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ."""
        n_samples = 100
        data = {
            "sequence_id": [1] * n_samples,
            "time_step": list(range(n_samples)),
            # IMUデータ
            "acc_x": np.random.randn(n_samples).tolist(),
            "acc_y": np.random.randn(n_samples).tolist(),
            "acc_z": np.random.randn(n_samples).tolist(),
            "gyr_x": np.random.randn(n_samples).tolist(),
            "gyr_y": np.random.randn(n_samples).tolist(),
            "gyr_z": np.random.randn(n_samples).tolist(),
            "mag_x": np.random.randn(n_samples).tolist(),
            "mag_y": np.random.randn(n_samples).tolist(),
            "mag_z": np.random.randn(n_samples).tolist(),
            "quat_w": np.random.randn(n_samples).tolist(),
            "quat_x": np.random.randn(n_samples).tolist(),
            "quat_y": np.random.randn(n_samples).tolist(),
            "quat_z": np.random.randn(n_samples).tolist(),
            # THMデータ
            "thm_1": np.random.randn(n_samples).tolist(),
            "thm_2": np.random.randn(n_samples).tolist(),
            "thm_3": np.random.randn(n_samples).tolist(),
            "thm_4": np.random.randn(n_samples).tolist(),
            "thm_5": np.random.randn(n_samples).tolist(),
        }
        # ToFデータ（340次元）
        for i in range(340):
            data[f"tof_{i + 1}"] = np.random.randn(n_samples).tolist()

        return pl.DataFrame(data)

    def test_single_sequence_dataset_init(self, sample_data):
        """SingleSequenceDatasetの初期化テスト."""
        config = Config()
        dataset = SingleSequenceDataset(sample_data, config)

        assert dataset.sequence_id == 1
        assert len(dataset) == 1  # 単一シーケンス

    def test_single_sequence_dataset_getitem(self, sample_data):
        """SingleSequenceDatasetのgetitemテスト."""
        config = Config()
        dataset = SingleSequenceDataset(sample_data, config)

        item = dataset[0]

        # IMUデータの確認
        assert "imu" in item
        assert item["imu"].shape[0] == 20  # 20特徴量（13 IMU + 3 angular_vel + 3 acc_no_grav + 1 angular_dist）

        # THMデータの確認
        assert "thm" in item
        assert item["thm"].shape[0] == 5  # 5特徴量

        # ToFデータの確認（tof_mode=16の場合）
        assert "tof" in item
        expected_tof_dim = 16 * 4  # 16領域 × 4統計量
        assert item["tof"].shape[0] == expected_tof_dim

        # attention_maskの確認
        assert "attention_mask" in item


class TestCMIBERTModel:
    """CMIBERTModelのテスト."""

    def test_model_initialization(self):
        """モデル初期化のテスト."""
        model = CMIBERTModel(
            imu_dim=20,
            thm_dim=5,
            tof_dim=64,  # 16領域 × 4統計量
            feat_dim=500,
            num_classes=18,
            bert_layers=2,  # テスト用に小さく
            bert_heads=2,
        )

        assert model.imu_dim == 20
        assert model.thm_dim == 5
        assert model.tof_dim == 64
        assert model.num_classes == 18

    def test_model_forward(self):
        """モデルの前向き計算テスト."""
        batch_size = 4
        seq_len = 100

        model = CMIBERTModel(
            imu_dim=20,
            thm_dim=5,
            tof_dim=64,
            feat_dim=256,  # テスト用に小さく
            num_classes=18,
            bert_layers=2,
            bert_heads=2,
        )

        # ダミーデータ
        imu = torch.randn(batch_size, 20, seq_len)
        thm = torch.randn(batch_size, 5, seq_len)
        tof = torch.randn(batch_size, 64, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        # 前向き計算
        multiclass_logits, binary_logits = model(imu, thm, tof, attention_mask)

        # 出力形状の確認
        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)

    def test_model_without_thm_tof(self):
        """THM/ToFなしでのモデルテスト."""
        batch_size = 2
        seq_len = 50

        model = CMIBERTModel(
            imu_dim=20,
            thm_dim=0,  # THMなし
            tof_dim=0,  # ToFなし
            feat_dim=128,
            num_classes=18,
            bert_layers=1,
            bert_heads=1,
        )

        # IMUデータのみ
        imu = torch.randn(batch_size, 20, seq_len)

        # 前向き計算
        multiclass_logits, binary_logits = model(imu)

        # 出力形状の確認
        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)

    def test_model_training_step(self):
        """訓練ステップのテスト."""
        model = CMIBERTModel(
            imu_dim=20,
            thm_dim=5,
            tof_dim=64,
            feat_dim=128,
            num_classes=18,
            bert_layers=1,
            bert_heads=1,
        )

        # ダミーバッチ
        batch = {
            "imu": torch.randn(2, 20, 50),
            "thm": torch.randn(2, 5, 50),
            "tof": torch.randn(2, 64, 50),
            "multiclass_label": torch.randint(0, 18, (2,)),
            "binary_label": torch.rand(2),
            "attention_mask": torch.ones(2, 50),
        }

        # 訓練ステップ
        loss = model.training_step(batch, 0)

        # 損失が計算されることを確認
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー


class TestIMUDataModule:
    """IMUDataModuleのテスト."""

    @pytest.fixture
    def sample_train_data(self, tmp_path):
        """テスト用訓練データ作成."""
        n_sequences = 10
        data_list = []

        for seq_id in range(n_sequences):
            n_samples = np.random.randint(50, 150)
            seq_data = {
                "sequence_id": [seq_id] * n_samples,
                "time_step": list(range(n_samples)),
                "gesture": ["Text on phone"] * n_samples,
            }

            # IMUデータ
            for col in [
                "acc_x",
                "acc_y",
                "acc_z",
                "gyr_x",
                "gyr_y",
                "gyr_z",
                "mag_x",
                "mag_y",
                "mag_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
            ]:
                seq_data[col] = np.random.randn(n_samples).tolist()

            # THMデータ
            for i in range(1, 6):
                seq_data[f"thm_{i}"] = np.random.randn(n_samples).tolist()

            # ToFデータ
            for i in range(1, 341):
                seq_data[f"tof_{i}"] = np.random.randn(n_samples).tolist()

            data_list.append(pl.DataFrame(seq_data))

        # 全データを結合
        df = pl.concat(data_list)

        # CSVとして保存
        csv_path = tmp_path / "train.csv"
        df.write_csv(csv_path)

        return csv_path

    def test_datamodule_initialization(self, sample_train_data):
        """DataModuleの初期化テスト."""
        config = Config()
        config.data.train_path = str(sample_train_data)

        datamodule = IMUDataModule(config, fold=0)

        assert datamodule.fold == 0
        assert datamodule.config == config

    def test_datamodule_setup(self, sample_train_data):
        """DataModuleのセットアップテスト."""
        config = Config()
        config.data.train_path = str(sample_train_data)
        config.val.params.n_splits = 2

        datamodule = IMUDataModule(config, fold=0)
        datamodule.setup("fit")

        # データセットが作成されることを確認
        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")
        assert len(datamodule.train_dataset) > 0
        assert len(datamodule.val_dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
