"""
test_exp014_dataset.py

exp014のIMUDatasetWithMiniRocketクラスの単体テスト
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp014"))

from config import Exp014Config
from dataset import IMUDatasetWithMiniRocket, MiniRocketFeatureExtractor


@pytest.fixture
def sample_config():
    """テスト用の設定を作成"""
    config = Exp014Config()

    # テスト用の小さなパラメータに変更
    config.minirocket.num_kernels = 50  # 高速化のため小さく設定
    config.minirocket.n_jobs = 1
    config.minirocket.random_state = 42
    config.minirocket.cache_enabled = True

    # 一時ディレクトリをキャッシュディレクトリとして使用
    temp_dir = tempfile.mkdtemp()
    config.minirocket.cache_dir = temp_dir

    # その他のパラメータもテスト用に調整
    config.preprocessing.target_sequence_length = 100
    config.demographics.enabled = False  # シンプルにするため無効

    yield config

    # クリーンアップ
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_imu_data():
    """テスト用のIMUデータを作成"""
    np.random.seed(42)

    sequence_data = []

    # 複数のシーケンスを作成
    for seq_id in range(3):  # 3シーケンス
        seq_length = np.random.randint(80, 120)  # 80-120の範囲でランダム長
        gesture = f"gesture_{seq_id % 2}"  # 2種類のジェスチャー

        for i in range(seq_length):
            sequence_data.append(
                {
                    "sequence_id": f"test_seq_{seq_id:03d}",
                    "sequence_counter": i,
                    "gesture": gesture,
                    "subject": f"test_subject_{seq_id % 2}",  # 2人のsubject
                    # IMU特徴量 (exp013と同じ形式)
                    "acc_x": np.random.randn() * 0.5,
                    "acc_y": np.random.randn() * 0.5,
                    "acc_z": np.random.randn() * 0.5 + 9.8,  # 重力成分
                    "rot_x": np.random.randn() * 0.1,
                    "rot_y": np.random.randn() * 0.1,
                    "rot_z": np.random.randn() * 0.1,
                    "rot_w": 1.0 + np.random.randn() * 0.05,  # 正規化済み四元数
                    # 物理ベース特徴量 (MiniRocket target features)
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


@pytest.fixture
def sample_demographics_data():
    """テスト用のDemographicsデータを作成"""
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
        ]
    )


@pytest.fixture
def pretrained_minirocket_extractor(sample_config, sample_imu_data):
    """事前学習済みのMiniRocket抽出器を作成"""
    extractor = MiniRocketFeatureExtractor(sample_config)
    extractor.fit(sample_imu_data)
    return extractor


class TestIMUDatasetWithMiniRocket:
    """IMUDatasetWithMiniRocketクラスのテスト"""

    def test_initialization_without_pretrained_extractor(self, sample_config, sample_imu_data):
        """事前学習済み抽出器なしでの初期化テスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # 基本属性の確認
        assert dataset.config == sample_config
        assert dataset.minirocket_extractor is not None
        assert dataset.minirocket_extractor.is_fitted
        assert dataset.minirocket_features is not None

        # データサイズの確認
        assert len(dataset) == 3  # 3シーケンス
        assert dataset.minirocket_features.shape[0] == 3  # 3シーケンス分
        assert dataset.minirocket_features.shape[1] > 0  # 特徴量次元数

    def test_initialization_with_pretrained_extractor(
        self, sample_config, sample_imu_data, pretrained_minirocket_extractor
    ):
        """事前学習済み抽出器ありでの初期化テスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data,
            config=sample_config,
            minirocket_extractor=pretrained_minirocket_extractor,
            augment=False,
        )

        # 事前学習済み抽出器が使用されていることを確認
        assert dataset.minirocket_extractor is pretrained_minirocket_extractor
        assert dataset.minirocket_extractor.is_fitted
        assert dataset.minirocket_features is not None

        # データサイズの確認
        assert len(dataset) == 3  # 3シーケンス
        assert dataset.minirocket_features.shape[0] == 3

    def test_getitem_functionality(self, sample_config, sample_imu_data):
        """__getitem__機能のテスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # 最初のサンプルを取得
        sample = dataset[0]

        # 基本的なキーの存在確認
        expected_keys = [
            "imu",
            "multiclass_label",
            "binary_label",
            "sequence_id",
            "gesture",
            "attention_mask",
            "minirocket_features",
        ]
        for key in expected_keys:
            assert key in sample

        # データ型の確認
        assert isinstance(sample["imu"], torch.Tensor)
        assert isinstance(sample["minirocket_features"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)
        assert isinstance(sample["multiclass_label"], torch.Tensor)
        assert isinstance(sample["binary_label"], torch.Tensor)
        assert isinstance(sample["sequence_id"], str)
        assert isinstance(sample["gesture"], str)

        # テンソルの形状確認
        assert sample["imu"].dim() == 2  # [input_dim, seq_len]
        assert sample["minirocket_features"].dim() == 1  # [minirocket_dim]
        assert sample["attention_mask"].dim() == 1  # [seq_len]

        # MiniRocket特徴量の次元確認
        expected_minirocket_dim = dataset.minirocket_extractor.output_dim
        assert sample["minirocket_features"].shape[0] == expected_minirocket_dim

    def test_sequence_id_mapping(self, sample_config, sample_imu_data):
        """sequence_id -> MiniRocket特徴量のマッピングテスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # マッピングの確認
        assert dataset.sequence_id_to_minirocket_idx is not None
        assert len(dataset.sequence_id_to_minirocket_idx) == 3  # 3シーケンス

        # 各シーケンスIDが適切にマッピングされているか確認
        for i in range(len(dataset)):
            sample = dataset[i]
            seq_id = sample["sequence_id"]
            expected_idx = dataset.sequence_id_to_minirocket_idx[seq_id]

            # マッピングされたインデックスが有効か確認
            assert 0 <= expected_idx < len(dataset.minirocket_features)

    def test_data_consistency(self, sample_config, sample_imu_data):
        """データの一貫性テスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # 複数回同じインデックスを取得して一貫性を確認
        sample1 = dataset[0]
        sample2 = dataset[0]

        # 同じシーケンスIDであることを確認
        assert sample1["sequence_id"] == sample2["sequence_id"]

        # MiniRocket特徴量が同じであることを確認
        torch.testing.assert_close(sample1["minirocket_features"], sample2["minirocket_features"])

        # IMUデータが同じであることを確認（augmentationが無効の場合）
        torch.testing.assert_close(sample1["imu"], sample2["imu"])

    def test_augmentation_independence(self, sample_config, sample_imu_data):
        """拡張処理がMiniRocket特徴量に影響しないことのテスト"""
        # 拡張なしのデータセット
        dataset_no_aug = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # 拡張ありのデータセット
        dataset_with_aug = IMUDatasetWithMiniRocket(
            df=sample_imu_data,
            config=sample_config,
            minirocket_extractor=dataset_no_aug.minirocket_extractor,  # 同じ抽出器を使用
            augment=True,
        )

        sample_no_aug = dataset_no_aug[0]
        sample_with_aug = dataset_with_aug[0]

        # MiniRocket特徴量は拡張処理に関係なく同じであること
        torch.testing.assert_close(sample_no_aug["minirocket_features"], sample_with_aug["minirocket_features"])

        # sequence_idも同じであること
        assert sample_no_aug["sequence_id"] == sample_with_aug["sequence_id"]

    def test_demographics_integration(self, sample_config, sample_imu_data, sample_demographics_data):
        """Demographics統合のテスト"""
        # Demographics有効化
        sample_config.demographics.enabled = True

        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data,
            config=sample_config,
            minirocket_extractor=None,
            augment=False,
            demographics_data=sample_demographics_data,
        )

        sample = dataset[0]

        # Demographicsデータが含まれることを確認
        assert "demographics" in sample
        assert isinstance(sample["demographics"], dict)

        # Demographics特徴量のテンソルが含まれることを確認
        demographics_keys = sample["demographics"].keys()
        assert len(demographics_keys) > 0

        for key, value in sample["demographics"].items():
            assert isinstance(value, torch.Tensor)

    def test_different_sequence_lengths(self, sample_config):
        """異なるシーケンス長での動作テスト"""
        np.random.seed(42)

        # 様々な長さのシーケンスを作成
        sequence_data = []
        sequence_lengths = [50, 100, 150, 200]

        for seq_idx, length in enumerate(sequence_lengths):
            gesture = f"gesture_{seq_idx % 2}"

            for i in range(length):
                sequence_data.append(
                    {
                        "sequence_id": f"var_len_seq_{seq_idx:03d}",
                        "sequence_counter": i,
                        "gesture": gesture,
                        "subject": f"test_subject_{seq_idx % 2}",
                        # 必要な特徴量
                        "acc_x": np.random.randn() * 0.5,
                        "acc_y": np.random.randn() * 0.5,
                        "acc_z": np.random.randn() * 0.5 + 9.8,
                        "rot_x": np.random.randn() * 0.1,
                        "rot_y": np.random.randn() * 0.1,
                        "rot_z": np.random.randn() * 0.1,
                        "rot_w": 1.0 + np.random.randn() * 0.05,
                        # MiniRocket target features
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

        data = pl.DataFrame(sequence_data)

        dataset = IMUDatasetWithMiniRocket(df=data, config=sample_config, minirocket_extractor=None, augment=False)

        # 4シーケンス分のデータが作成されることを確認
        assert len(dataset) == len(sequence_lengths)
        assert dataset.minirocket_features.shape[0] == len(sequence_lengths)

        # 各サンプルが正しく取得できることを確認
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "minirocket_features" in sample
            assert sample["minirocket_features"].dim() == 1

    def test_error_handling(self, sample_config, sample_imu_data):
        """エラーハンドリングのテスト"""
        # 必要な特徴量が不足しているデータ
        incomplete_data = sample_imu_data.drop(["linear_acc_x", "linear_acc_y", "linear_acc_z"])

        # MiniRocket特徴量抽出時にエラーが発生することを確認
        with pytest.raises(Exception):  # ValueError or 他の例外
            IMUDatasetWithMiniRocket(df=incomplete_data, config=sample_config, minirocket_extractor=None, augment=False)

    def test_memory_efficiency(self, sample_config, sample_imu_data):
        """メモリ効率性のテスト"""
        dataset = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # MiniRocket特徴量が事前計算されていることを確認
        assert dataset.minirocket_features is not None

        # データサイズが妥当であることを確認
        minirocket_features_size = dataset.minirocket_features.nbytes
        assert minirocket_features_size > 0

        # メモリリークチェック: 複数回アクセスしても同じオブジェクトを使用
        features1 = dataset.minirocket_features
        features2 = dataset.minirocket_features
        assert features1 is features2  # 同じオブジェクト参照

    def test_reproducibility(self, sample_config, sample_imu_data):
        """再現性のテスト"""
        # 同じrandom_stateで2つのデータセットを作成
        dataset1 = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        dataset2 = IMUDatasetWithMiniRocket(
            df=sample_imu_data, config=sample_config, minirocket_extractor=None, augment=False
        )

        # MiniRocket特徴量が同じであることを確認
        np.testing.assert_array_equal(dataset1.minirocket_features, dataset2.minirocket_features)

        # 個別サンプルも同じであることを確認
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        torch.testing.assert_close(sample1["minirocket_features"], sample2["minirocket_features"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
