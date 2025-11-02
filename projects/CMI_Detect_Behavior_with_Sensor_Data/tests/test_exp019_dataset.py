"""exp019 IMUDatasetの利き手反転オーグメンテーション統合テスト."""

import numpy as np
import polars as pl
import pytest
import torch
from codes.exp.exp019.config import Config
from codes.exp.exp019.dataset import IMUDataset


@pytest.fixture
def sample_train_data():
    """テスト用の訓練データ."""
    np.random.seed(42)

    # 簡単なサンプルデータを作成
    data = []
    for seq_id in ["seq001", "seq002", "seq003"]:
        for counter in range(50):  # 各シーケンス50行
            data.append(
                {
                    "sequence_id": seq_id,
                    "sequence_counter": counter,
                    "subject": f"S{seq_id[-3:]}",  # seq001 -> S001
                    "gesture": "Above ear - pull hair" if seq_id == "seq001" else "Wave hello",
                    "sequence_type": "target" if seq_id == "seq001" else "non_target",
                    "behavior": "gesture",
                    "orientation": "sitting",
                    # IMUデータ
                    "acc_x": np.random.normal(0, 1),
                    "acc_y": np.random.normal(0, 1),
                    "acc_z": np.random.normal(9.8, 1),  # 重力
                    "rot_w": 1.0,
                    "rot_x": np.random.normal(0, 0.1),
                    "rot_y": np.random.normal(0, 0.1),
                    "rot_z": np.random.normal(0, 0.1),
                }
            )

    return pl.DataFrame(data)


@pytest.fixture
def sample_demographics_data():
    """テスト用のdemographicsデータ."""
    return pl.DataFrame(
        {
            "subject": ["S001", "S002", "S003"],
            "handedness": [1, 0, 1],  # 右利き、左利き、右利き
            "age": [25, 30, 35],
            "sex": [0, 1, 0],
            "adult_child": [1, 1, 1],
            "height_cm": [170.0, 165.0, 175.0],
            "shoulder_to_wrist_cm": [60.0, 58.0, 62.0],
            "elbow_to_wrist_cm": [25.0, 24.0, 26.0],
        }
    )


@pytest.fixture
def sample_config():
    """テスト用の設定."""
    config = Config()
    config.augmentation.enable_handedness_flip = True
    config.augmentation.handedness_flip_prob = 1.0  # 常に反転（テスト用）
    config.demographics.enabled = True
    return config


class TestIMUDatasetHandednessAugmentation:
    """IMUDatasetの利き手反転オーグメンテーション統合テスト."""

    def test_dataset_creation_with_handedness_aug(self, sample_train_data, sample_demographics_data):
        """利き手反転オーグメンテーション有効でのデータセット作成テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # データセットが正常に作成されている
        assert len(dataset) == 3  # 3つのシーケンス
        assert dataset.handedness_aug is not None
        assert dataset.enable_handedness_aug is True

    def test_dataset_creation_without_handedness_aug(self, sample_train_data, sample_demographics_data):
        """利き手反転オーグメンテーション無効でのデータセット作成テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=False,
            handedness_flip_prob=0.5,
        )

        # 利き手反転オーグメンテーションが無効
        assert len(dataset) == 3
        assert dataset.handedness_aug is None
        assert dataset.enable_handedness_aug is False

    def test_dataset_creation_without_demographics(self, sample_train_data):
        """demographics無しでの利き手反転オーグメンテーション設定テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=None,
            demographics_config={"enabled": False},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # demographicsが無いため利き手反転オーグメンテーションは無効
        assert len(dataset) == 3
        assert dataset.handedness_aug is None

    def test_augmentation_disabled_when_not_augmenting(self, sample_train_data, sample_demographics_data):
        """augment=Falseの時は利き手反転オーグメンテーションも無効になるテスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=False,  # データ拡張無効
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # augment=Falseのため利き手反転オーグメンテーションも無効
        assert dataset.handedness_aug is None

    def test_getitem_with_handedness_aug(self, sample_train_data, sample_demographics_data):
        """__getitem__での利き手反転オーグメンテーション適用テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,  # 常に反転
        )

        # データを取得
        item = dataset[0]

        # 期待されるキーが存在
        assert "imu" in item
        assert "multiclass_label" in item
        assert "binary_label" in item
        assert "sequence_id" in item
        assert "gesture" in item
        assert "demographics" in item
        assert "missing_mask" in item

        # IMUデータの形状確認
        assert item["imu"].shape == (16, 30)  # [features, seq_len]
        assert isinstance(item["imu"], torch.Tensor)

    def test_handedness_aug_deterministic_behavior(self, sample_train_data, sample_demographics_data):
        """利き手反転オーグメンテーションの決定的動作テスト."""
        # 常に反転するデータセット
        dataset_flip = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,  # 常に反転
        )

        # 反転しないデータセット
        dataset_no_flip = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.0,  # 反転しない
        )

        # 同じシーケンスでデータ取得
        np.random.seed(42)
        torch.manual_seed(42)
        item_flip = dataset_flip[0]

        np.random.seed(42)
        torch.manual_seed(42)
        item_no_flip = dataset_no_flip[0]

        # Y軸の値が反転されているかチェック
        imu_flip = item_flip["imu"]  # [features, seq_len]
        imu_no_flip = item_no_flip["imu"]  # [features, seq_len]

        # acc_yの比較（インデックス1）
        acc_y_flip = imu_flip[1, :]  # acc_y
        acc_y_no_flip = imu_no_flip[1, :]  # acc_y

        # 反転されているかチェック（完全一致は難しいので近似）
        # 注意: 物理特徴量計算やその他の処理が入るため、単純な符号反転でない可能性がある
        assert not torch.allclose(acc_y_flip, acc_y_no_flip, atol=1e-6), "Y軸データが反転されていない"

    def test_subject_id_passing(self, sample_train_data, sample_demographics_data):
        """被験者IDが正しく渡されているかのテスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # sequence_to_subjectマッピングが正しく作成されている
        assert dataset.sequence_to_subject is not None
        assert "seq001" in dataset.sequence_to_subject
        assert dataset.sequence_to_subject["seq001"] == "S001"

    def test_consistent_data_shape(self, sample_train_data, sample_demographics_data):
        """データ形状の一貫性テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=50,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # 複数のアイテムで一貫した形状
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["imu"].shape == (16, 50)  # [features, seq_len]
            assert isinstance(item["multiclass_label"], torch.Tensor)
            assert isinstance(item["binary_label"], torch.Tensor)
            assert "demographics" in item

    def test_no_memory_leaks(self, sample_train_data, sample_demographics_data):
        """メモリリーク防止のテスト（元データが変更されないことを確認）."""
        original_df = sample_train_data.clone()

        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,
        )

        # データセット作成後、元のDataFrameが変更されていないことを確認
        assert sample_train_data.equals(original_df)

        # 複数回アクセスしても元データは変更されない
        item1 = dataset[0]
        item2 = dataset[0]

        assert sample_train_data.equals(original_df)
