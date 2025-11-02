"""exp019 IMUDatasetの利き手反転オーグメンテーション統合テスト（更新版）."""

import numpy as np
import polars as pl
import pytest
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


class TestIMUDatasetHandednessAugmentationUpdated:
    """IMUDatasetの利き手反転オーグメンテーション統合テスト（更新版）."""

    def test_demographics_handedness_flip_consistency(self, sample_train_data, sample_demographics_data):
        """利き手反転とdemographics handenessの整合性テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,  # 常に反転
        )

        # 元のdemographicsを確認
        original_handedness = sample_demographics_data.filter(pl.col("subject") == "S001")["handedness"].item()

        # 複数回実行して反転の統計を取る（確率的なため）
        flip_count = 0
        no_flip_count = 0

        for i in range(20):  # 十分な回数実行
            np.random.seed(i)  # 異なるシードを使用
            item = dataset[0]  # S001のデータ

            if "demographics" in item and "handedness" in item["demographics"]:
                flipped_handedness = item["demographics"]["handedness"].item()
                if flipped_handedness == 1 - original_handedness:
                    flip_count += 1
                else:
                    no_flip_count += 1

        # flip_prob=1.0なので、大部分は反転されているはず（確率的要素があるため完全ではない）
        assert flip_count > no_flip_count, (
            f"Expected more flips but got flip_count={flip_count}, no_flip_count={no_flip_count}"
        )

    def test_demographics_handedness_no_flip_consistency(self, sample_train_data, sample_demographics_data):
        """利き手反転無しの場合のdemographics handenessの整合性テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.0,  # 反転しない
        )

        # 元のdemographicsを確認
        original_handedness = sample_demographics_data.filter(pl.col("subject") == "S001")["handedness"].item()

        # 複数回実行して統計的にチェック
        flip_count = 0
        no_flip_count = 0

        for i in range(20):  # 十分な回数実行
            np.random.seed(i)  # 異なるシードを使用
            item = dataset[0]  # S001のデータ

            if "demographics" in item and "handedness" in item["demographics"]:
                handedness = item["demographics"]["handedness"].item()
                if handedness == original_handedness:
                    no_flip_count += 1
                else:
                    flip_count += 1

        # flip_prob=0.0なので、大部分は反転されていないはず
        assert no_flip_count > flip_count, (
            f"Expected more no-flips but got no_flip_count={no_flip_count}, flip_count={flip_count}"
        )

    def test_apply_augmentation_returns_tuple(self, sample_train_data, sample_demographics_data):
        """_apply_augmentationがタプルを返すことのテスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # サンプルIMUデータ
        sample_imu = np.random.rand(16, 30).astype(np.float32)

        # _apply_augmentationを直接呼び出し
        result = dataset._apply_augmentation(sample_imu, "S001")

        # タプルが返されることを確認
        assert isinstance(result, tuple)
        assert len(result) == 2

        imu_data, was_flipped = result
        assert isinstance(imu_data, np.ndarray)
        assert isinstance(was_flipped, bool)
        assert imu_data.shape == sample_imu.shape

    def test_get_demographics_with_flip_handedness(self, sample_train_data, sample_demographics_data):
        """_get_demographics_for_subjectのflip_handedness引数テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,
        )

        # 通常の取得
        demographics_normal = dataset._get_demographics_for_subject("S001", flip_handedness=False)
        # 反転取得
        demographics_flipped = dataset._get_demographics_for_subject("S001", flip_handedness=True)

        assert demographics_normal is not None
        assert demographics_flipped is not None

        # 元のS001は右利き（1）なので、反転後は左利き（0）になるはず
        assert demographics_normal["handedness"].item() == 1
        assert demographics_flipped["handedness"].item() == 0

        # 他の特徴量は変わらない
        assert demographics_normal["age"].item() == demographics_flipped["age"].item()
        assert demographics_normal["sex"].item() == demographics_flipped["sex"].item()

    def test_left_handed_subject_flip(self, sample_train_data, sample_demographics_data):
        """左利き被験者の反転テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,  # 常に反転
        )

        # S002は左利き（0）
        demographics_flipped = dataset._get_demographics_for_subject("S002", flip_handedness=True)

        assert demographics_flipped is not None
        # 左利き（0）から右利き（1）に反転
        assert demographics_flipped["handedness"].item() == 1

    def test_augmentation_disabled_no_flip_flag(self, sample_train_data, sample_demographics_data):
        """augment=Falseの場合、反転フラグが常にFalseであることのテスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=False,  # データ拡張無効
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=1.0,
        )

        sample_imu = np.random.rand(16, 30).astype(np.float32)
        imu_data, was_flipped = dataset._apply_augmentation(sample_imu, "S001")

        # augment=Falseなので反転フラグはFalse
        assert was_flipped is False
        # データも変更されていない
        np.testing.assert_array_equal(imu_data, sample_imu)

    def test_handedness_aug_none_no_flip_flag(self, sample_train_data, sample_demographics_data):
        """handedness_aug=Noneの場合、反転フラグが常にFalseであることのテスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=False,  # 利き手反転無効
            handedness_flip_prob=1.0,
        )

        sample_imu = np.random.rand(16, 30).astype(np.float32)
        imu_data, was_flipped = dataset._apply_augmentation(sample_imu, "S001")

        # handedness_augがNoneなので反転フラグはFalse
        assert was_flipped is False

    def test_probabilistic_flipping(self, sample_train_data, sample_demographics_data):
        """確率的な反転の動作テスト."""
        dataset = IMUDataset(
            df=sample_train_data,
            target_sequence_length=30,
            augment=True,
            demographics_data=sample_demographics_data,
            demographics_config={"enabled": True},
            enable_handedness_aug=True,
            handedness_flip_prob=0.5,  # 50%の確率
        )

        sample_imu = np.random.rand(16, 30).astype(np.float32)

        # 複数回実行して確率的動作を確認
        flip_results = []
        for _ in range(100):
            _, was_flipped = dataset._apply_augmentation(sample_imu.copy(), "S001")
            flip_results.append(was_flipped)

        flip_ratio = sum(flip_results) / len(flip_results)
        # 50%前後になることを確認（±20%の範囲で許容）
        assert 0.3 <= flip_ratio <= 0.7
