"""Human Normalization機能のユニットテスト."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add the exp022 directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "codes" / "exp" / "exp022"))

from human_normalization import (
    HNConfig,
    compute_hn_features,
    derive_hn_channels,
    get_hn_feature_columns,
    join_subject_anthro,
)


class TestHNConfig:
    """HNConfigクラスのテスト."""

    def test_default_config(self):
        """デフォルト設定のテスト."""
        config = HNConfig()
        assert config.hn_enabled is True
        assert config.hn_eps == 1e-6
        assert config.hn_radius_min_max == (0.15, 0.9)
        assert len(config.hn_features) == 10
        assert "linear_acc_mag_per_h" in config.hn_features

    def test_custom_config(self):
        """カスタム設定のテスト."""
        config = HNConfig(
            hn_enabled=False,
            hn_eps=1e-8,
            hn_radius_min_max=(0.1, 1.0),
            hn_features=["linear_acc_mag_per_h", "linear_acc_mag_per_rS"],
        )
        assert config.hn_enabled is False
        assert config.hn_eps == 1e-8
        assert config.hn_radius_min_max == (0.1, 1.0)
        assert len(config.hn_features) == 2


class TestJoinSubjectAnthro:
    """join_subject_anthro関数のテスト."""

    def test_successful_join(self):
        """正常なjoinのテスト."""
        # テスト用IMUデータ
        imu_data = pl.DataFrame(
            {
                "subject": ["S001", "S001", "S002", "S002"],
                "sequence_id": [1, 1, 2, 2],
                "linear_acc_mag": [1.0, 2.0, 3.0, 4.0],
            }
        ).lazy()

        # テスト用demographics
        demo_df = pl.DataFrame(
            {
                "subject": ["S001", "S002"],
                "height_cm": [170.0, 180.0],
                "shoulder_to_wrist_cm": [60.0, 65.0],
                "elbow_to_wrist_cm": [25.0, 28.0],
            }
        )

        result = join_subject_anthro(imu_data, demo_df).collect()

        assert "h" in result.columns
        assert "r_elbow" in result.columns
        assert "r_shoulder" in result.columns
        assert "hn_used_fallback" in result.columns

        # 値の確認
        assert result["h"][0] == 1.7  # 170cm -> 1.7m
        assert result["r_elbow"][0] == 0.25  # 25cm -> 0.25m
        assert result["r_shoulder"][0] == 0.60  # 60cm -> 0.60m
        assert result["hn_used_fallback"][0] is False

    def test_missing_demographics(self):
        """欠損値処理のテスト."""
        # テスト用IMUデータ
        imu_data = pl.DataFrame(
            {
                "subject": ["S001", "S002", "S003"],
                "sequence_id": [1, 2, 3],
                "linear_acc_mag": [1.0, 2.0, 3.0],
            }
        ).lazy()

        # 欠損値のあるdemographics
        demo_df = pl.DataFrame(
            {
                "subject": ["S001", "S002", "S003"],
                "height_cm": [170.0, None, 180.0],
                "shoulder_to_wrist_cm": [60.0, 65.0, None],
                "elbow_to_wrist_cm": [25.0, 28.0, 30.0],
            }
        )

        result = join_subject_anthro(imu_data, demo_df).collect()

        # 欠損値フラグの確認
        assert result["hn_used_fallback"][0] is False  # S001は完全
        assert result["hn_used_fallback"][1] is True  # S002はheight_cmがnull
        assert result["hn_used_fallback"][2] is True  # S003はshoulder_to_wrist_cmがnull

        # 欠損値が中央値で埋められていることを確認
        assert not result["h"].is_null().any()
        assert not result["r_elbow"].is_null().any()
        assert not result["r_shoulder"].is_null().any()

    def test_missing_columns_error(self):
        """必須カラム不足のエラーテスト."""
        imu_data = pl.DataFrame(
            {
                "subject": ["S001"],
                "linear_acc_mag": [1.0],
            }
        ).lazy()

        # 必須カラムが不足したdemographics
        demo_df = pl.DataFrame(
            {
                "subject": ["S001"],
                "height_cm": [170.0],
                # shoulder_to_wrist_cm と elbow_to_wrist_cm が不足
            }
        )

        with pytest.raises(ValueError, match="Missing demographics columns"):
            join_subject_anthro(imu_data, demo_df).collect()


class TestDeriveHNChannels:
    """derive_hn_channels関数のテスト."""

    def test_deterministic_calculation(self):
        """決定論的計算のテスト."""
        # 既知の値でテスト
        frame = pl.DataFrame(
            {
                "sequence_id": [1, 1],
                "linear_acc_x": [1.0, 2.0],
                "linear_acc_y": [0.0, 0.0],
                "linear_acc_z": [0.0, 0.0],
                "linear_acc_mag": [1.0, 2.0],
                "angular_vel_x": [0.0, 0.0],
                "angular_vel_y": [0.0, 0.0],
                "angular_vel_z": [1.0, 2.0],  # omega = 1.0, 2.0
                "h": [2.0, 2.0],  # 2m
                "r_elbow": [0.5, 0.5],  # 0.5m
                "r_shoulder": [1.0, 1.0],  # 1.0m
            }
        ).lazy()

        result = derive_hn_channels(frame, eps=1e-6, bounds=(0.15, 0.9)).collect()

        # 身長正規化の確認
        expected_per_h = [1.0 / 2.0, 2.0 / 2.0]  # [0.5, 1.0]
        assert np.allclose(result["linear_acc_mag_per_h"].to_numpy(), expected_per_h)

        # レバー長正規化の確認
        expected_per_rS = [1.0 / 1.0, 2.0 / 1.0]  # [1.0, 2.0]
        assert np.allclose(result["linear_acc_mag_per_rS"].to_numpy(), expected_per_rS)

        # 遠心加速度計算の確認（omega^2 * r）
        # omega = [1.0, 2.0], r_shoulder = 1.0 -> a_c_shoulder = [1.0, 4.0]
        expected_acc_over_centripetal_rS = [1.0 / 1.0, 2.0 / 4.0]  # [1.0, 0.5]
        assert np.allclose(result["acc_over_centripetal_rS"].to_numpy(), expected_acc_over_centripetal_rS)

    def test_numerical_stability(self):
        """数値安定性のテスト."""
        frame = pl.DataFrame(
            {
                "sequence_id": [1, 1],
                "linear_acc_x": [1.0, 1.0],
                "linear_acc_y": [0.0, 0.0],
                "linear_acc_z": [0.0, 0.0],
                "linear_acc_mag": [1.0, 1.0],
                "angular_vel_x": [0.0, 0.0],
                "angular_vel_y": [0.0, 0.0],
                "angular_vel_z": [0.0, 0.0],  # omega = 0.0
                "h": [0.0, 2.0],  # 0と正常値
                "r_elbow": [-0.1, 0.5],  # 負値と正常値
                "r_shoulder": [0.0, 1.0],  # 0と正常値
            }
        ).lazy()

        result = derive_hn_channels(frame, eps=1e-6, bounds=(0.15, 0.9)).collect()

        # NaNやinfが発生していないことを確認
        for col in result.columns:
            if col.startswith(("linear_acc_mag_per", "acc_over_centripetal", "alpha_like", "v_over")):
                values = result[col].to_numpy()
                assert np.all(np.isfinite(values)), f"Column {col} contains non-finite values"

    def test_radius_clipping(self):
        """半径クリッピングのテスト."""
        frame = pl.DataFrame(
            {
                "sequence_id": [1, 1, 1],
                "linear_acc_x": [1.0, 1.0, 1.0],
                "linear_acc_y": [0.0, 0.0, 0.0],
                "linear_acc_z": [0.0, 0.0, 0.0],
                "linear_acc_mag": [1.0, 1.0, 1.0],
                "angular_vel_x": [0.0, 0.0, 0.0],
                "angular_vel_y": [0.0, 0.0, 0.0],
                "angular_vel_z": [1.0, 1.0, 1.0],
                "h": [2.0, 2.0, 2.0],
                "r_elbow": [0.5, 0.5, 0.5],
                "r_shoulder": [0.1, 0.5, 1.5],  # 範囲外、範囲内、範囲外
            }
        ).lazy()

        result = derive_hn_channels(frame, eps=1e-6, bounds=(0.15, 0.9)).collect()

        # r_effがクリッピングされていることを確認
        expected_r_eff = [0.15, 0.5, 0.9]  # クリッピング適用
        assert np.allclose(result["r_eff"].to_numpy(), expected_r_eff)


class TestComputeHNFeatures:
    """compute_hn_features関数のテスト."""

    def test_hn_disabled(self):
        """HN無効時のテスト."""
        frame = pl.DataFrame(
            {
                "subject": ["S001"],
                "sequence_id": [1],
                "linear_acc_mag": [1.0],
            }
        ).lazy()

        demo_df = pl.DataFrame(
            {
                "subject": ["S001"],
                "height_cm": [170.0],
                "shoulder_to_wrist_cm": [60.0],
                "elbow_to_wrist_cm": [25.0],
            }
        )

        config = HNConfig(hn_enabled=False)
        result = compute_hn_features(frame, demo_df, config).collect()

        # 元のカラムのみ存在することを確認
        assert set(result.columns) == {"subject", "sequence_id", "linear_acc_mag"}

    def test_hn_enabled(self):
        """HN有効時のテスト."""
        frame = pl.DataFrame(
            {
                "subject": ["S001", "S001"],
                "sequence_id": [1, 1],
                "linear_acc_x": [1.0, 2.0],
                "linear_acc_y": [0.0, 0.0],
                "linear_acc_z": [0.0, 0.0],
                "linear_acc_mag": [1.0, 2.0],
                "angular_vel_x": [0.0, 0.0],
                "angular_vel_y": [0.0, 0.0],
                "angular_vel_z": [1.0, 2.0],
            }
        ).lazy()

        demo_df = pl.DataFrame(
            {
                "subject": ["S001"],
                "height_cm": [200.0],
                "shoulder_to_wrist_cm": [100.0],
                "elbow_to_wrist_cm": [50.0],
            }
        )

        config = HNConfig(hn_enabled=True)
        result = compute_hn_features(frame, demo_df, config).collect()

        # HN特徴量が追加されていることを確認
        hn_features = config.hn_features
        for feature in hn_features:
            assert feature in result.columns, f"Missing HN feature: {feature}"

        # 人体測定値とフォールバックフラグが追加されていることを確認
        assert "h" in result.columns
        assert "r_elbow" in result.columns
        assert "r_shoulder" in result.columns
        assert "hn_used_fallback" in result.columns


class TestGetHNFeatureColumns:
    """get_hn_feature_columns関数のテスト."""

    def test_disabled_config(self):
        """無効設定時のテスト."""
        config = HNConfig(hn_enabled=False)
        columns = get_hn_feature_columns(config)
        assert columns == []

    def test_enabled_config(self):
        """有効設定時のテスト."""
        config = HNConfig(hn_enabled=True)
        columns = get_hn_feature_columns(config)
        assert len(columns) == 10
        assert columns == config.hn_features

    def test_custom_features(self):
        """カスタム特徴量設定のテスト."""
        custom_features = ["linear_acc_mag_per_h", "linear_acc_mag_per_rS"]
        config = HNConfig(hn_enabled=True, hn_features=custom_features)
        columns = get_hn_feature_columns(config)
        assert columns == custom_features


class TestIntegrationContract:
    """統合時の契約テスト."""

    def test_feature_dimension_increase(self):
        """HN有効時の特徴量次元増加確認."""
        # 基本フレーム
        frame = pl.DataFrame(
            {
                "subject": ["S001"] * 10,
                "sequence_id": [1] * 10,
                "linear_acc_x": np.random.randn(10),
                "linear_acc_y": np.random.randn(10),
                "linear_acc_z": np.random.randn(10),
                "linear_acc_mag": np.random.rand(10) * 2,
                "angular_vel_x": np.random.randn(10),
                "angular_vel_y": np.random.randn(10),
                "angular_vel_z": np.random.randn(10),
            }
        ).lazy()

        demo_df = pl.DataFrame(
            {
                "subject": ["S001"],
                "height_cm": [175.0],
                "shoulder_to_wrist_cm": [65.0],
                "elbow_to_wrist_cm": [27.0],
            }
        )

        # HN無効時
        config_disabled = HNConfig(hn_enabled=False)
        result_disabled = compute_hn_features(frame, demo_df, config_disabled).collect()

        # HN有効時
        config_enabled = HNConfig(hn_enabled=True)
        result_enabled = compute_hn_features(frame, demo_df, config_enabled).collect()

        # カラム数の増加確認
        # HN特徴量(10) + 人体測定値とフラグ(4) + 中間計算値(6) = 20
        expected_increase = len(config_enabled.hn_features) + 4 + 6  # HN特徴量 + 人体測定値・フラグ + 中間計算値
        actual_increase = len(result_enabled.columns) - len(result_disabled.columns)
        assert actual_increase == expected_increase

        # 形状とデータ型の確認
        assert result_enabled.shape[0] == result_disabled.shape[0]  # 行数は同じ

        # すべてのHN特徴量が有限値であることを確認
        hn_features = config_enabled.hn_features
        for feature in hn_features:
            values = result_enabled[feature].to_numpy()
            assert np.all(np.isfinite(values)), f"HN feature {feature} contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__])
