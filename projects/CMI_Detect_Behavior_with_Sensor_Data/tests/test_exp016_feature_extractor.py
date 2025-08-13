"""exp016 IMU特徴量抽出器のテスト."""

import polars as pl
import pytest
import torch
from codes.exp.exp013.dataset import (
    calculate_angular_distance_pl,
    calculate_angular_velocity_from_quat_pl,
    remove_gravity_from_acc_pl,
)
from codes.exp.exp016.model import IMUFeatureExtractor


class TestIMUFeatureExtractor:
    """IMU特徴量抽出器のテスト."""

    @pytest.fixture
    def sample_imu_data(self) -> torch.Tensor:
        """サンプルIMUデータを生成."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 100

        # 現実的なIMUデータを生成
        acc = torch.randn(batch_size, 3, seq_len) * 2.0 + torch.tensor([0.0, 0.0, 9.81]).view(1, 3, 1)

        # 正規化された四元数を生成
        quat = torch.randn(batch_size, 4, seq_len)
        quat = quat / torch.norm(quat, dim=1, keepdim=True)

        # [B, 7, T] 形式に結合 (acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z)
        imu_data = torch.cat([acc, quat], dim=1)

        return imu_data

    @pytest.fixture
    def sample_polars_data(self, sample_imu_data: torch.Tensor) -> pl.LazyFrame:
        """PyTorchデータからPolarsデータフレームを生成."""
        batch_size, _, seq_len = sample_imu_data.shape

        # バッチの最初のサンプルを使用
        data = sample_imu_data[0].T.numpy()  # [T, 7]

        df_data = {
            "acc_x": data[:, 0].tolist(),
            "acc_y": data[:, 1].tolist(),
            "acc_z": data[:, 2].tolist(),
            "rot_w": data[:, 3].tolist(),
            "rot_x": data[:, 4].tolist(),
            "rot_y": data[:, 5].tolist(),
            "rot_z": data[:, 6].tolist(),
        }

        return pl.LazyFrame(df_data)

    def test_feature_extractor_basic_functionality(self, sample_imu_data: torch.Tensor):
        """基本的な動作確認テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # 順伝播テスト
        features = extractor(sample_imu_data)

        # 出力形状の確認
        assert features.shape == (2, 16, 100), f"Expected shape (2, 16, 100), got {features.shape}"

        # NaNやInfの確認
        assert torch.isfinite(features).all(), "Features contain NaN or Inf values"

    def test_feature_extractor_output_structure(self, sample_imu_data: torch.Tensor):
        """出力特徴量の構造確認."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)
        features = extractor(sample_imu_data)

        # 各特徴量の確認
        # 0-6: 基本IMU特徴量 (7次元)
        original_data = sample_imu_data
        torch.testing.assert_close(features[:, :7, :], original_data, rtol=1e-5, atol=1e-7)

        # 7-9: 線形加速度 (3次元)
        linear_acc = features[:, 7:10, :]
        assert linear_acc.shape[1] == 3

        # 10: 線形加速度の大きさ (1次元)
        linear_acc_mag = features[:, 10:11, :]
        assert linear_acc_mag.shape[1] == 1

        # 11: 線形加速度大きさのジャーク (1次元)
        linear_acc_mag_jerk = features[:, 11:12, :]
        assert linear_acc_mag_jerk.shape[1] == 1

        # 12-14: 角速度 (3次元)
        angular_vel = features[:, 12:15, :]
        assert angular_vel.shape[1] == 3

        # 15: 角距離 (1次元)
        angular_distance = features[:, 15:16, :]
        assert angular_distance.shape[1] == 1

    def test_gravity_removal_comparison(self, sample_imu_data: torch.Tensor, sample_polars_data: pl.LazyFrame):
        """重力除去の数値比較テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # PyTorch実装
        acc = sample_imu_data[0:1, :3, :]  # 最初のバッチのみ
        quat = sample_imu_data[0:1, 3:, :]
        pytorch_linear_acc = extractor.remove_gravity(acc, quat)  # [1, 3, T]

        # Polars実装
        polars_linear_acc_df = remove_gravity_from_acc_pl(sample_polars_data, tol=1e-8).collect()
        polars_linear_acc = torch.tensor(polars_linear_acc_df.to_numpy().T, dtype=torch.float32).unsqueeze(
            0
        )  # [1, 3, T]

        # 数値比較（相対誤差5%以内）
        torch.testing.assert_close(
            pytorch_linear_acc,
            polars_linear_acc,
            rtol=0.05,
            atol=1e-3,
            msg="Gravity removal results don't match between PyTorch and Polars implementations",
        )

    def test_angular_velocity_comparison(self, sample_imu_data: torch.Tensor, sample_polars_data: pl.LazyFrame):
        """角速度計算の数値比較テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # PyTorch実装
        quat = sample_imu_data[0:1, 3:, :]  # 最初のバッチのみ
        pytorch_angular_vel = extractor.quaternion_to_angular_velocity(quat)  # [1, 3, T]

        # Polars実装
        polars_angular_vel_df = calculate_angular_velocity_from_quat_pl(
            sample_polars_data, time_delta=1.0 / 200.0, tol=1e-8
        ).collect()
        polars_angular_vel = torch.tensor(polars_angular_vel_df.to_numpy().T, dtype=torch.float32).unsqueeze(
            0
        )  # [1, 3, T]

        # 数値比較（相対誤差10%以内、角速度は数値的に不安定になりやすい）
        torch.testing.assert_close(
            pytorch_angular_vel,
            polars_angular_vel,
            rtol=0.1,
            atol=1e-2,
            msg="Angular velocity results don't match between PyTorch and Polars implementations",
        )

    def test_angular_distance_comparison(self, sample_imu_data: torch.Tensor, sample_polars_data: pl.LazyFrame):
        """角距離計算の数値比較テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # PyTorch実装
        quat = sample_imu_data[0:1, 3:, :]  # 最初のバッチのみ
        pytorch_angular_dist = extractor.calculate_angular_distance(quat)  # [1, 1, T]

        # Polars実装
        polars_angular_dist_df = calculate_angular_distance_pl(sample_polars_data, tol=1e-8).collect()
        polars_angular_dist = torch.tensor(polars_angular_dist_df.to_numpy().T, dtype=torch.float32).unsqueeze(
            0
        )  # [1, 1, T]

        # 数値比較（相対誤差5%以内）
        torch.testing.assert_close(
            pytorch_angular_dist,
            polars_angular_dist,
            rtol=0.05,
            atol=1e-3,
            msg="Angular distance results don't match between PyTorch and Polars implementations",
        )

    def test_feature_extractor_gpu_cpu_consistency(self, sample_imu_data: torch.Tensor):
        """GPU/CPU間の一貫性テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # CPU実行
        cpu_features = extractor(sample_imu_data)

        # GPU実行（利用可能な場合）
        if torch.cuda.is_available():
            device = torch.device("cuda")
            extractor_gpu = extractor.to(device)
            sample_imu_data_gpu = sample_imu_data.to(device)

            gpu_features = extractor_gpu(sample_imu_data_gpu).cpu()

            # CPU/GPU結果の比較（数値安定性のため許容範囲を広げる）
            torch.testing.assert_close(
                cpu_features, gpu_features, rtol=1e-3, atol=1e-5, msg="GPU and CPU results don't match"
            )

    def test_batch_processing(self):
        """異なるバッチサイズでの動作確認."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # 異なるバッチサイズでテスト
        for batch_size in [1, 4, 8]:
            seq_len = 50
            torch.manual_seed(42)

            # テストデータ生成
            acc = torch.randn(batch_size, 3, seq_len)
            quat = torch.randn(batch_size, 4, seq_len)
            quat = quat / torch.norm(quat, dim=1, keepdim=True)
            imu_data = torch.cat([acc, quat], dim=1)

            # 順伝播
            features = extractor(imu_data)

            # 出力形状確認
            assert features.shape == (batch_size, 16, seq_len)
            assert torch.isfinite(features).all()

    def test_numerical_stability(self):
        """数値安定性のテスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # 極値データでのテスト
        batch_size, seq_len = 2, 50

        # ゼロ四元数
        acc = torch.zeros(batch_size, 3, seq_len)
        quat = torch.zeros(batch_size, 4, seq_len)
        quat[:, 0, :] = 1.0  # (1, 0, 0, 0)
        imu_data = torch.cat([acc, quat], dim=1)

        features = extractor(imu_data)
        assert torch.isfinite(features).all(), "Features contain NaN or Inf with zero quaternions"

        # 非常に小さい四元数
        acc = torch.randn(batch_size, 3, seq_len) * 1e-6
        quat = torch.randn(batch_size, 4, seq_len) * 1e-6
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        imu_data = torch.cat([acc, quat], dim=1)

        features = extractor(imu_data)
        assert torch.isfinite(features).all(), "Features contain NaN or Inf with small quaternions"

    def test_sequence_boundary_handling(self):
        """シーケンス境界の処理テスト."""
        extractor = IMUFeatureExtractor(time_delta=1.0 / 200.0, tol=1e-8)

        # 短いシーケンス
        batch_size, seq_len = 1, 2
        torch.manual_seed(42)

        acc = torch.randn(batch_size, 3, seq_len)
        quat = torch.randn(batch_size, 4, seq_len)
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        imu_data = torch.cat([acc, quat], dim=1)

        features = extractor(imu_data)

        assert features.shape == (batch_size, 16, seq_len)
        assert torch.isfinite(features).all(), "Features contain NaN or Inf with short sequences"
