"""exp007のSingleSequenceIMUDatasetのテスト."""

import sys
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import pytest
import torch
from exp.exp007.dataset import SingleSequenceIMUDataset


class TestSingleSequenceIMUDataset:
    """SingleSequenceIMUDatasetのテスト."""

    @pytest.fixture
    def sample_single_sequence_data(self):
        """テスト用の単一シーケンスデータを作成."""
        # 欠損値を含む単一シーケンスのサンプルデータ
        data = {
            "row_id": ["seq1_0", "seq1_1", "seq1_2", "seq1_3", "seq1_4"],
            "sequence_id": ["seq1", "seq1", "seq1", "seq1", "seq1"],
            "sequence_counter": [0, 1, 2, 3, 4],
            "subject": ["subj1", "subj1", "subj1", "subj1", "subj1"],
            "gesture": [
                "Forehead - scratch",
                "Forehead - scratch",
                "Forehead - scratch",
                "Forehead - scratch",
                "Forehead - scratch",
            ],
            "acc_x": [1.0, 2.0, None, 4.0, 5.0],  # インデックス2に欠損値
            "acc_y": [1.1, 2.1, 3.1, 4.1, 5.1],
            "acc_z": [1.2, 2.2, 3.2, 4.2, 5.2],
            "rot_w": [0.9, 0.91, 0.92, 0.93, 0.94],
            "rot_x": [0.1, 0.11, None, 0.13, 0.14],  # インデックス2に欠損値
            "rot_y": [0.2, 0.21, 0.22, 0.23, 0.24],
            "rot_z": [0.3, 0.31, 0.32, 0.33, 0.34],
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def sample_single_sequence_no_missing(self):
        """欠損値のない単一シーケンスデータを作成."""
        data = {
            "row_id": ["seq2_0", "seq2_1", "seq2_2"],
            "sequence_id": ["seq2", "seq2", "seq2"],
            "sequence_counter": [0, 1, 2],
            "subject": ["subj2", "subj2", "subj2"],
            "gesture": ["Wave hello", "Wave hello", "Wave hello"],
            "acc_x": [1.0, 2.0, 3.0],
            "acc_y": [1.1, 2.1, 3.1],
            "acc_z": [1.2, 2.2, 3.2],
            "rot_w": [0.9, 0.91, 0.92],
            "rot_x": [0.1, 0.11, 0.12],
            "rot_y": [0.2, 0.21, 0.22],
            "rot_z": [0.3, 0.31, 0.32],
        }
        return pl.DataFrame(data)

    def test_initialization(self, sample_single_sequence_data):
        """初期化のテスト."""
        dataset = SingleSequenceIMUDataset(sequence_df=sample_single_sequence_data, target_sequence_length=10)

        # 初期化確認
        assert dataset.sequence_id == "seq1"
        assert dataset.target_sequence_length == 10
        assert dataset.augment is False
        assert len(dataset.imu_cols) == 7

    def test_data_preprocessing(self, sample_single_sequence_data):
        """データ前処理のテスト."""
        dataset = SingleSequenceIMUDataset(sequence_df=sample_single_sequence_data, target_sequence_length=10)

        # 前処理されたデータの形状確認
        assert dataset.imu_data.shape == (10, 7)  # [seq_len, features]
        assert len(dataset.missing_mask) == 10

        # データ型確認
        assert dataset.imu_data.dtype in [np.float32, np.float64]
        assert dataset.missing_mask.dtype == bool

    def test_missing_value_handling(self, sample_single_sequence_data):
        """欠損値処理のテスト."""
        dataset = SingleSequenceIMUDataset(
            sequence_df=sample_single_sequence_data,
            target_sequence_length=5,  # 元データと同じ長さ
        )

        # 欠損値が0で埋められていることを確認
        assert not np.isnan(dataset.imu_data).any()

        # 欠損位置のマスクが正しいことを確認（インデックス2に欠損）
        # シーケンス長正規化がない場合、元のインデックス2にTrueが設定される
        assert dataset.missing_mask[2] is True  # インデックス2が欠損
        assert dataset.missing_mask[0] is False  # インデックス0は正常
        assert dataset.missing_mask[1] is False  # インデックス1は正常

    def test_sequence_length_normalization_padding(self, sample_single_sequence_no_missing):
        """シーケンス長正規化（パディング）のテスト."""
        target_length = 10
        dataset = SingleSequenceIMUDataset(
            sequence_df=sample_single_sequence_no_missing, target_sequence_length=target_length
        )

        # 目標長になっていることを確認
        assert dataset.imu_data.shape == (target_length, 7)
        assert len(dataset.missing_mask) == target_length

        # パディング部分（元データ長3、目標長10なので4-9はパディング）
        # 元データには欠損がないので、最初の3つはFalse、パディング部分はFalse
        assert not dataset.missing_mask[:3].any()  # 元データ部分は欠損なし

    def test_sequence_length_normalization_downsampling(self, sample_single_sequence_data):
        """シーケンス長正規化（ダウンサンプリング）のテスト."""
        target_length = 3
        dataset = SingleSequenceIMUDataset(
            sequence_df=sample_single_sequence_data, target_sequence_length=target_length
        )

        # 目標長になっていることを確認
        assert dataset.imu_data.shape == (target_length, 7)
        assert len(dataset.missing_mask) == target_length

    def test_dataset_length(self, sample_single_sequence_data):
        """データセット長のテスト."""
        dataset = SingleSequenceIMUDataset(sequence_df=sample_single_sequence_data, target_sequence_length=10)

        # 単一シーケンスなので長さは1
        assert len(dataset) == 1

    def test_getitem_valid_index(self, sample_single_sequence_data):
        """__getitem__の正常ケースのテスト."""
        dataset = SingleSequenceIMUDataset(sequence_df=sample_single_sequence_data, target_sequence_length=10)

        # インデックス0でのデータ取得
        result = dataset[0]

        # 返り値の構造確認
        assert isinstance(result, dict)
        assert "imu" in result
        assert "attention_mask" in result
        assert "sequence_id" in result

        # テンソルの形状確認 [features, seq_len]
        assert isinstance(result["imu"], torch.Tensor)
        assert result["imu"].shape == (7, 10)
        assert result["imu"].dtype == torch.float32

        # attention_maskの確認
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert result["attention_mask"].shape == (10,)
        assert result["attention_mask"].dtype == torch.bool

        # sequence_idの確認
        assert result["sequence_id"] == "seq1"

    def test_getitem_invalid_index(self, sample_single_sequence_data):
        """__getitem__の異常ケースのテスト."""
        dataset = SingleSequenceIMUDataset(sequence_df=sample_single_sequence_data, target_sequence_length=10)

        # インデックス1でエラーが発生することを確認
        with pytest.raises(IndexError, match="Single sequence dataset only has one item"):
            dataset[1]

    def test_attention_mask_creation(self, sample_single_sequence_data):
        """attention_mask作成のテスト."""
        dataset = SingleSequenceIMUDataset(
            sequence_df=sample_single_sequence_data,
            target_sequence_length=5,  # 元データと同じ長さ
        )

        result = dataset[0]
        attention_mask = result["attention_mask"]

        # missing_maskの逆がattention_maskになることを確認
        # missing_mask[2] = True（欠損）→ attention_mask[2] = False
        assert attention_mask[2] is False  # 欠損位置
        assert attention_mask[0] is True  # 正常位置
        assert attention_mask[1] is True  # 正常位置

    def test_missing_imu_columns(self):
        """IMU列不足のエラーテスト."""
        # acc_xが不足したデータ
        incomplete_data = {
            "row_id": ["seq1_0"],
            "sequence_id": ["seq1"],
            "sequence_counter": [0],
            "acc_y": [1.1],  # acc_xが不足
            "acc_z": [1.2],
            "rot_w": [0.9],
            "rot_x": [0.1],
            "rot_y": [0.2],
            "rot_z": [0.3],
        }
        incomplete_df = pl.DataFrame(incomplete_data)

        # エラーが発生することを確認
        with pytest.raises(ValueError, match="Missing IMU columns"):
            SingleSequenceIMUDataset(sequence_df=incomplete_df, target_sequence_length=10)
