"""Test for exp013 dataset demographics integration."""

import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest
import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp013"))

from config import Config
from dataset import IMUDataset, SingleSequenceIMUDataset


class TestIMUDatasetWithDemographics:
    """Demographics統合IMUDatasetのテスト."""

    @pytest.fixture
    def sample_demographics_data(self):
        """サンプルdemographicsデータ."""
        return pl.DataFrame({
            "subject": ["SUBJ_001", "SUBJ_002", "SUBJ_003"],
            "adult_child": [1, 0, 1],
            "age": [25.0, 15.0, 35.0],
            "sex": [0, 1, 0],
            "handedness": [1, 1, 0],
            "height_cm": [170.0, 160.0, 175.0],
            "shoulder_to_wrist_cm": [55.0, 45.0, 60.0],
            "elbow_to_wrist_cm": [25.0, 22.0, 28.0],
        })

    @pytest.fixture
    def sample_imu_data(self):
        """サンプルIMUデータ."""
        return pl.DataFrame({
            "sequence_id": ["seq_001", "seq_002", "seq_003", "seq_004"],
            "subject": ["SUBJ_001", "SUBJ_001", "SUBJ_002", "SUBJ_003"],
            "sequence_counter": [0, 0, 0, 0],
            "gesture": ["Text on phone", "Wave hello", "Text on phone", "Neck - scratch"],
            "acc_x": [0.1, 0.2, 0.3, 0.4],
            "acc_y": [0.5, 0.6, 0.7, 0.8],
            "acc_z": [0.9, 1.0, 1.1, 1.2],
            "rot_w": [0.707, 0.707, 0.707, 0.707],
            "rot_x": [0.0, 0.1, 0.2, 0.3],
            "rot_y": [0.0, 0.1, 0.2, 0.3],
            "rot_z": [0.707, 0.707, 0.707, 0.707],
            "linear_acc_x": [0.1, 0.2, 0.3, 0.4],
            "linear_acc_y": [0.5, 0.6, 0.7, 0.8],
            "linear_acc_z": [0.9, 1.0, 1.1, 1.2],
            "linear_acc_mag": [1.0, 1.1, 1.2, 1.3],
            "linear_acc_mag_jerk": [0.1, 0.1, 0.1, 0.1],
            "angular_vel_x": [0.1, 0.2, 0.3, 0.4],
            "angular_vel_y": [0.5, 0.6, 0.7, 0.8],
            "angular_vel_z": [0.9, 1.0, 1.1, 1.2],
            "angular_distance": [0.1, 0.2, 0.3, 0.4],
        })

    def test_dataset_demographics_scaling_from_config(self, sample_demographics_data):
        """Demographics データのスケーリングパラメータ設定値ベース処理をテスト（簡易版）."""
        config = Config()
        config.demographics.enabled = True

        # 簡易的なテスト: demographics_configから直接パラメータが正しく設定されるかを確認
        demographics_config = config.demographics.model_dump()
        
        # 設定値が期待通りであることを確認
        assert demographics_config["age_min"] == 8.0, f"Expected age_min=8.0, got {demographics_config['age_min']}"
        assert demographics_config["age_max"] == 60.0, f"Expected age_max=60.0, got {demographics_config['age_max']}"
        assert demographics_config["height_min"] == 130.0, f"Expected height_min=130.0, got {demographics_config['height_min']}"
        assert demographics_config["height_max"] == 195.0, f"Expected height_max=195.0, got {demographics_config['height_max']}"
        assert demographics_config["shoulder_to_wrist_min"] == 35.0, f"Expected shoulder_to_wrist_min=35.0, got {demographics_config['shoulder_to_wrist_min']}"
        assert demographics_config["shoulder_to_wrist_max"] == 75.0, f"Expected shoulder_to_wrist_max=75.0, got {demographics_config['shoulder_to_wrist_max']}"
        assert demographics_config["elbow_to_wrist_min"] == 15.0, f"Expected elbow_to_wrist_min=15.0, got {demographics_config['elbow_to_wrist_min']}"
        assert demographics_config["elbow_to_wrist_max"] == 50.0, f"Expected elbow_to_wrist_max=50.0, got {demographics_config['elbow_to_wrist_max']}"

        # 実データに基づくマージン付きの範囲が適切に設定されていることを確認
        assert demographics_config["age_min"] < 10.0, "Age minimum should be below actual data minimum (10) with margin"
        assert demographics_config["age_max"] > 53.0, "Age maximum should be above actual data maximum (53) with margin"
        assert demographics_config["height_min"] < 135.0, "Height minimum should be below actual data minimum (135) with margin"
        assert demographics_config["height_max"] > 190.5, "Height maximum should be above actual data maximum (190.5) with margin"

    def test_dataset_demographics_processing(self, sample_imu_data, sample_demographics_data):
        """Demographics データの処理をテスト."""
        config = Config()
        config.demographics.enabled = True

        # データセット作成（データフレームを直接渡す）
        dataset = IMUDataset(
            df=sample_imu_data,
            target_sequence_length=50,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True}
        )

        # データアイテム取得
        item = dataset[0]  # sequence_id="seq_001", subject="SUBJ_001"

        # Demographics データが含まれることを確認
        assert "demographics" in item, "Demographics data should be included in item"

        demographics = item["demographics"]

        # カテゴリカルデータの確認
        assert "adult_child" in demographics, "adult_child should be in demographics"
        assert "sex" in demographics, "sex should be in demographics"
        assert "handedness" in demographics, "handedness should be in demographics"

        # 数値データの確認
        assert "age" in demographics, "age should be in demographics"
        assert "height_cm" in demographics, "height_cm should be in demographics"
        assert "shoulder_to_wrist_cm" in demographics, "shoulder_to_wrist_cm should be in demographics"
        assert "elbow_to_wrist_cm" in demographics, "elbow_to_wrist_cm should be in demographics"

        # データ型確認
        assert demographics["adult_child"].dtype == torch.long, "adult_child should be long tensor"
        assert demographics["age"].dtype == torch.float, "age should be float tensor"

    def test_dataset_without_demographics(self, sample_imu_data):
        """Demographics無効時のデータセットをテスト."""
        config = Config()
        config.demographics.enabled = False

        # データセット作成（Demographics無効）
        dataset = IMUDataset(
            df=sample_imu_data,
            target_sequence_length=50,
            demographics_data=None,
            demographics_config={"enabled": False}
        )

        # データアイテム取得
        item = dataset[0]

        # Demographics データが含まれないことを確認
        assert "demographics" not in item, "Demographics data should not be included when disabled"

        # 基本的なIMUデータは含まれることを確認
        assert "imu" in item, "IMU data should be included"
        assert "attention_mask" in item, "Attention mask should be included"

    def test_demographics_missing_subject_handling(self, sample_imu_data):
        """対象外subjectのdemographics処理をテスト."""
        # SUBJ_001とSUBJ_002のみのdemographicsデータ（SUBJ_003は欠損）
        limited_demographics = pl.DataFrame({
            "subject": ["SUBJ_001", "SUBJ_002"],
            "adult_child": [1, 0],
            "age": [25.0, 15.0],
            "sex": [0, 1],
            "handedness": [1, 1],
            "height_cm": [170.0, 160.0],
            "shoulder_to_wrist_cm": [55.0, 45.0],
            "elbow_to_wrist_cm": [25.0, 22.0],
        })

        config = Config()
        config.demographics.enabled = True

        # データセット作成（Demographics有効、一部データのみ）
        dataset = IMUDataset(
            df=sample_imu_data,
            target_sequence_length=50,
            demographics_data=limited_demographics,
            demographics_config={"enabled": True}
        )

        # SUBJ_003のデータ（index=3）を取得
        item = dataset[3]  # sequence_id="seq_004", subject="SUBJ_003"

        # demographicsが存在するが、値がゼロパディングされることを確認
        assert "demographics" in item, "Demographics should still be included for missing subjects"

        demographics = item["demographics"]
        # ゼロパディングされた値であることを確認
        assert demographics["adult_child"].item() == 0, "Missing subject should have zero-padded categorical data"
        assert demographics["age"].item() == 0.0, "Missing subject should have zero-padded numerical data"


class TestSingleSequenceIMUDataset:
    """SingleSequenceIMUDataset のdemographics統合テスト."""

    def test_single_sequence_dataset_with_demographics(self):
        """SingleSequence データセットでのdemographics統合をテスト."""
        # サンプルシーケンスデータ
        sequence_data = pl.DataFrame({
            "sequence_id": ["seq_test", "seq_test", "seq_test"],
            "subject": ["SUBJ_001", "SUBJ_001", "SUBJ_001"],
            "sequence_counter": [0, 1, 2],
            "acc_x": [0.1, 0.2, 0.3],
            "acc_y": [0.5, 0.6, 0.7],
            "acc_z": [0.9, 1.0, 1.1],
            "rot_w": [0.707, 0.707, 0.707],
            "rot_x": [0.0, 0.1, 0.2],
            "rot_y": [0.0, 0.1, 0.2],
            "rot_z": [0.707, 0.707, 0.707],
        })

        # Demographics データ
        demographics_data = pl.DataFrame({
            "subject": ["SUBJ_001"],
            "adult_child": [1],
            "age": [25.0],
            "sex": [0],
            "handedness": [1],
            "height_cm": [170.0],
            "shoulder_to_wrist_cm": [55.0],
            "elbow_to_wrist_cm": [25.0],
        })

        # データセット作成
        dataset = SingleSequenceIMUDataset(
            sequence_data,
            target_sequence_length=50,
            use_demographics=True,
            demographics_data=demographics_data,
            subject="SUBJ_001"  # 対象subject指定
        )

        # データアイテム取得
        item = dataset[0]

        # 基本データ確認
        assert "imu" in item, "IMU data should be included"
        assert "attention_mask" in item, "Attention mask should be included"
        assert "demographics" in item, "Demographics data should be included"

        # Demographics データの形状・型確認
        demographics = item["demographics"]
        assert demographics["adult_child"].shape == torch.Size([]), "Categorical data should be scalar"
        assert demographics["age"].shape == torch.Size([]), "Numerical data should be scalar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
