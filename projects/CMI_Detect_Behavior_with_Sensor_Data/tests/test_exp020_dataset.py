"""exp020 dataset module test cases."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp020"))

from dataset import IMUDataset, remove_gravity_from_acc_pl


def create_sample_imu_data(n_sequences: int = 2, seq_length: int = 100) -> pl.DataFrame:
    """サンプルIMUデータを作成."""
    data = []

    for seq_id in range(n_sequences):
        for counter in range(seq_length):
            row = {
                "sequence_id": f"seq_{seq_id}",
                "sequence_counter": counter,
                "subject": f"subject_{seq_id % 2}",
                "gesture": "Above ear - pull hair" if seq_id == 0 else "Drink from bottle/cup",
                "acc_x": np.random.normal(0, 1),
                "acc_y": np.random.normal(0, 1),
                "acc_z": np.random.normal(9.81, 1),  # 重力を含む
                "rot_w": np.random.normal(1, 0.1),
                "rot_x": np.random.normal(0, 0.1),
                "rot_y": np.random.normal(0, 0.1),
                "rot_z": np.random.normal(0, 0.1),
            }
            data.append(row)

    return pl.DataFrame(data)


def create_sample_demographics() -> pl.DataFrame:
    """サンプル人口統計データを作成."""
    return pl.DataFrame(
        [
            {
                "subject": "subject_0",
                "adult_child": 1,
                "age": 25,
                "sex": 1,
                "handedness": 1,
                "height_cm": 170,
                "shoulder_to_wrist_cm": 60,
                "elbow_to_wrist_cm": 25,
            },
            {
                "subject": "subject_1",
                "adult_child": 0,
                "age": 12,
                "sex": 0,
                "handedness": 0,
                "height_cm": 140,
                "shoulder_to_wrist_cm": 45,
                "elbow_to_wrist_cm": 20,
            },
        ]
    )


class TestRemoveGravity:
    """重力除去機能のテスト."""

    def test_remove_gravity_from_acc_pl(self):
        """重力除去の基本動作テスト."""
        # サンプルデータ作成
        df = create_sample_imu_data(1, 50)
        df_lazy = df.lazy()

        # 重力除去実行
        linear_acc_df = remove_gravity_from_acc_pl(df_lazy)
        result = linear_acc_df.collect()

        # 結果確認
        assert "linear_acc_x" in result.columns
        assert "linear_acc_y" in result.columns
        assert "linear_acc_z" in result.columns
        assert len(result) == 50

        # NaN値が適切に処理されているか
        assert not result["linear_acc_x"].is_null().any()
        assert not result["linear_acc_y"].is_null().any()
        assert not result["linear_acc_z"].is_null().any()


class TestIMUDatasetPhysicsFeatures:
    """IMUDatasetの物理特徴量テスト."""

    def test_advanced_physics_features_calculation(self):
        """高度な物理特徴量の計算テスト."""
        df = create_sample_imu_data(2, 50)
        demographics = create_sample_demographics()

        dataset = IMUDataset(
            df=df,
            target_sequence_length=40,
            demographics_data=demographics,
            demographics_config={
                "enabled": True,
                "age_min": 8.0,
                "age_max": 60.0,
                "height_min": 130.0,
                "height_max": 195.0,
                "shoulder_to_wrist_min": 35.0,
                "shoulder_to_wrist_max": 75.0,
                "elbow_to_wrist_min": 15.0,
                "elbow_to_wrist_max": 50.0,
            },
        )

        # データセットサイズ確認
        assert len(dataset) == 2

        # 特徴量の存在確認
        sample = dataset[0]
        imu_tensor = sample["imu"]

        # 期待される特徴量数：基本(7) + 基本物理(9) + 高度特徴量
        expected_features = [
            # 基本IMU (7)
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            # 基本物理特徴量 (9)
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "linear_acc_mag",
            "linear_acc_mag_jerk",
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            "angular_distance",
            # 累積和 (4)
            "linear_acc_x_cumsum",
            "linear_acc_y_cumsum",
            "linear_acc_z_cumsum",
            "linear_acc_mag_cumsum",
            # 差分 (7)
            "linear_acc_x_diff",
            "linear_acc_y_diff",
            "linear_acc_z_diff",
            "linear_acc_mag_diff",
            "angular_vel_x_diff",
            "angular_vel_y_diff",
            "angular_vel_z_diff",
            # 長期差分 (12)
            "linear_acc_x_diff_5",
            "linear_acc_x_diff_10",
            "linear_acc_x_diff_20",
            "linear_acc_y_diff_5",
            "linear_acc_y_diff_10",
            "linear_acc_y_diff_20",
            "linear_acc_z_diff_5",
            "linear_acc_z_diff_10",
            "linear_acc_z_diff_20",
            "linear_acc_mag_diff_5",
            "linear_acc_mag_diff_10",
            "linear_acc_mag_diff_20",
            # シフト/ラグ (12)
            "linear_acc_x_lag_1",
            "linear_acc_x_lag_3",
            "linear_acc_x_lag_5",
            "linear_acc_y_lag_1",
            "linear_acc_y_lag_3",
            "linear_acc_y_lag_5",
            "linear_acc_z_lag_1",
            "linear_acc_z_lag_3",
            "linear_acc_z_lag_5",
            "linear_acc_mag_lag_1",
            "linear_acc_mag_lag_3",
            "linear_acc_mag_lag_5",
            # 中央値差分 (4)
            "linear_acc_x_median_diff",
            "linear_acc_y_median_diff",
            "linear_acc_z_median_diff",
            "linear_acc_mag_median_diff",
            # 統計的特徴量 (6)
            "linear_acc_mag_rolling_mean",
            "linear_acc_mag_rolling_std",
            "linear_acc_energy",
            "linear_acc_x_zero_cross",
            "linear_acc_y_zero_cross",
            "linear_acc_z_zero_cross",
            # ウェーブレット変換 (16)
            "linear_acc_x_wavelet_cA",
            "linear_acc_x_wavelet_cD1",
            "linear_acc_x_wavelet_cD2",
            "linear_acc_x_wavelet_cD3",
            "linear_acc_y_wavelet_cA",
            "linear_acc_y_wavelet_cD1",
            "linear_acc_y_wavelet_cD2",
            "linear_acc_y_wavelet_cD3",
            "linear_acc_z_wavelet_cA",
            "linear_acc_z_wavelet_cD1",
            "linear_acc_z_wavelet_cD2",
            "linear_acc_z_wavelet_cD3",
            "linear_acc_mag_wavelet_cA",
            "linear_acc_mag_wavelet_cD1",
            "linear_acc_mag_wavelet_cD2",
            "linear_acc_mag_wavelet_cD3",
        ]

        # 期待される特徴量数： 7 + 9 + 4 + 7 + 12 + 12 + 4 + 6 + 16 = 77
        expected_feature_count = 77

        # 実際の特徴量数確認
        actual_feature_count = len(dataset.imu_cols)
        assert actual_feature_count == expected_feature_count, (
            f"Expected {expected_feature_count} features, got {actual_feature_count}"
        )

        # テンソル形状確認
        assert imu_tensor.shape[0] == expected_feature_count  # 特徴量数
        assert imu_tensor.shape[1] == 40  # シーケンス長

        # データ型確認
        assert imu_tensor.dtype == torch.float32

        # NaN値の確認（欠損値マスクで処理されているはず）
        assert not torch.isnan(imu_tensor).any()

    def test_cumsum_features(self):
        """累積和特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # IMU列に累積和特徴量が含まれていることを確認
        cumsum_features = [col for col in dataset.imu_cols if col.endswith("_cumsum")]
        expected_cumsum = ["linear_acc_x_cumsum", "linear_acc_y_cumsum", "linear_acc_z_cumsum", "linear_acc_mag_cumsum"]

        assert len(cumsum_features) == 4
        for feature in expected_cumsum:
            assert feature in cumsum_features

    def test_diff_features(self):
        """差分特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # 差分特徴量の確認
        diff_features = [
            col for col in dataset.imu_cols if "_diff" in col and not any(x in col for x in ["_5", "_10", "_20"])
        ]
        expected_diffs = [
            "linear_acc_x_diff",
            "linear_acc_y_diff",
            "linear_acc_z_diff",
            "linear_acc_mag_diff",
            "angular_vel_x_diff",
            "angular_vel_y_diff",
            "angular_vel_z_diff",
        ]

        assert len(diff_features) == 7
        for feature in expected_diffs:
            assert feature in diff_features

    def test_long_diff_features(self):
        """長期差分特徴量のテスト."""
        df = create_sample_imu_data(1, 30)
        dataset = IMUDataset(df=df, target_sequence_length=30)

        # 長期差分特徴量の確認
        long_diff_features = [
            col for col in dataset.imu_cols if any(x in col for x in ["_diff_5", "_diff_10", "_diff_20"])
        ]

        # 4つの基本特徴量 × 3つのラグ = 12個
        assert len(long_diff_features) == 12

    def test_lag_features(self):
        """ラグ特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # ラグ特徴量の確認
        lag_features = [col for col in dataset.imu_cols if "_lag_" in col]

        # 4つの基本特徴量 × 3つのラグ = 12個
        assert len(lag_features) == 12

    def test_median_diff_features(self):
        """中央値差分特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # 中央値差分特徴量の確認
        median_diff_features = [col for col in dataset.imu_cols if "_median_diff" in col]
        expected_median_diffs = [
            "linear_acc_x_median_diff",
            "linear_acc_y_median_diff",
            "linear_acc_z_median_diff",
            "linear_acc_mag_median_diff",
        ]

        assert len(median_diff_features) == 4
        for feature in expected_median_diffs:
            assert feature in median_diff_features

    def test_statistical_features(self):
        """統計的特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # 統計的特徴量の確認
        stat_features = [
            "linear_acc_mag_rolling_mean",
            "linear_acc_mag_rolling_std",
            "linear_acc_energy",
            "linear_acc_x_zero_cross",
            "linear_acc_y_zero_cross",
            "linear_acc_z_zero_cross",
        ]

        for feature in stat_features:
            assert feature in dataset.imu_cols

    def test_wavelet_features(self):
        """ウェーブレット変換特徴量のテスト."""
        df = create_sample_imu_data(1, 20)
        dataset = IMUDataset(df=df, target_sequence_length=20)

        # ウェーブレット変換特徴量の確認
        wavelet_features = [col for col in dataset.imu_cols if "_wavelet_" in col]

        # 4つの基本特徴量 × 4つのウェーブレット係数 = 16個
        assert len(wavelet_features) == 16

        # 各基本特徴量に対して4つの係数があることを確認
        base_features = ["linear_acc_x", "linear_acc_y", "linear_acc_z", "linear_acc_mag"]
        coeffs = ["cA", "cD1", "cD2", "cD3"]

        for base_feature in base_features:
            for coeff in coeffs:
                expected_feature = f"{base_feature}_wavelet_{coeff}"
                assert expected_feature in wavelet_features


class TestIMUDatasetDemographics:
    """Demographics統合のテスト."""

    def test_dataset_with_demographics(self):
        """Demographics付きデータセットのテスト."""
        df = create_sample_imu_data(2, 20)
        demographics = create_sample_demographics()

        dataset = IMUDataset(
            df=df,
            target_sequence_length=20,
            demographics_data=demographics,
            demographics_config={
                "enabled": True,
                "age_min": 8.0,
                "age_max": 60.0,
                "height_min": 130.0,
                "height_max": 195.0,
                "shoulder_to_wrist_min": 35.0,
                "shoulder_to_wrist_max": 75.0,
                "elbow_to_wrist_min": 15.0,
                "elbow_to_wrist_max": 50.0,
            },
        )

        sample = dataset[0]

        # Demographics特徴量が含まれていることを確認
        assert "demographics" in sample
        demographics_data = sample["demographics"]

        # 期待される特徴量が含まれていることを確認
        expected_keys = [
            "adult_child",
            "age",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ]

        for key in expected_keys:
            assert key in demographics_data

        # データ型確認
        categorical_features = ["adult_child", "sex", "handedness"]
        numerical_features = ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]

        for feature in categorical_features:
            assert demographics_data[feature].dtype == torch.long

        for feature in numerical_features:
            assert demographics_data[feature].dtype == torch.float32


class TestIMUDatasetMissingValues:
    """欠損値処理のテスト."""

    def test_missing_value_handling(self):
        """欠損値処理のテスト."""
        df = create_sample_imu_data(1, 20)

        # 一部のデータにNaN値を設定
        df = df.with_columns(
            [pl.when(pl.col("sequence_counter") % 5 == 0).then(None).otherwise(pl.col("acc_x")).alias("acc_x")]
        )

        dataset = IMUDataset(df=df, target_sequence_length=20)
        sample = dataset[0]

        # 欠損値マスクが含まれていることを確認
        assert "missing_mask" in sample
        missing_mask = sample["missing_mask"]

        # マスクのデータ型と形状確認
        assert missing_mask.dtype == torch.bool
        assert missing_mask.shape[0] == 20

        # IMUデータにNaN値が含まれていないことを確認（0で埋められているはず）
        imu_tensor = sample["imu"]
        assert not torch.isnan(imu_tensor).any()


@pytest.mark.parametrize(
    "sequence_length,target_length",
    [
        (50, 40),  # ダウンサンプリング
        (30, 40),  # アップサンプリング
        (40, 40),  # 同じ長さ
    ],
)
def test_sequence_length_normalization(sequence_length, target_length):
    """シーケンス長正規化のテスト."""
    df = create_sample_imu_data(1, sequence_length)
    dataset = IMUDataset(df=df, target_sequence_length=target_length)

    sample = dataset[0]
    imu_tensor = sample["imu"]

    # 正規化後の長さが正しいことを確認
    assert imu_tensor.shape[1] == target_length


def test_label_mappings():
    """ラベルマッピングのテスト."""
    df = create_sample_imu_data(2, 10)
    dataset = IMUDataset(df=df, target_sequence_length=10)

    # サンプル取得
    sample_0 = dataset[0]
    sample_1 = dataset[1]

    # ジェスチャーとラベルの確認
    assert sample_0["gesture"] == "Above ear - pull hair"
    assert sample_1["gesture"] == "Drink from bottle/cup"

    # バイナリラベル確認（BFRB vs non-BFRB）
    assert sample_0["binary_label"].item() == 1.0  # BFRB
    assert sample_1["binary_label"].item() == 0.0  # non-BFRB

    # マルチクラスラベル確認
    assert isinstance(sample_0["multiclass_label"].item(), int)
    assert isinstance(sample_1["multiclass_label"].item(), int)
    assert sample_0["multiclass_label"].item() != sample_1["multiclass_label"].item()


if __name__ == "__main__":
    pytest.main([__file__])
