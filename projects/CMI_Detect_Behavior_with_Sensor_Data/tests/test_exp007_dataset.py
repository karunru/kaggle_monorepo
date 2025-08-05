"""exp007のデータセット処理のテスト."""

import numpy as np
import polars as pl
import pytest
from codes.exp.exp007.dataset import IMUDataset


class TestIMUDataset:
    """IMUDatasetのテスト."""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを作成."""
        # 欠損値を含むサンプルデータ
        data = {
            "row_id": ["seq1_0", "seq1_1", "seq1_2", "seq2_0", "seq2_1"],
            "sequence_id": ["seq1", "seq1", "seq1", "seq2", "seq2"],
            "sequence_counter": [0, 1, 2, 0, 1],
            "subject": ["subj1", "subj1", "subj1", "subj2", "subj2"],
            "gesture": [
                "Forehead - scratch",
                "Forehead - scratch",
                "Forehead - scratch",
                "Neck - scratch",
                "Neck - scratch",
            ],
            "acc_x": [1.0, 2.0, None, 3.0, 4.0],  # seq1に欠損値あり
            "acc_y": [1.1, 2.1, 3.1, 3.1, 4.1],
            "acc_z": [1.2, 2.2, 3.2, 3.2, 4.2],
            "rot_w": [0.9, 0.91, 0.92, 0.93, 0.94],
            "rot_x": [0.1, 0.11, None, 0.13, 0.14],  # seq1に欠損値あり
            "rot_y": [0.2, 0.21, 0.22, 0.23, 0.24],
            "rot_z": [0.3, 0.31, 0.32, 0.33, 0.34],
        }
        return pl.DataFrame(data)

    def test_data_type_conversion(self, sample_data):
        """データ型変換のテスト."""
        dataset = IMUDataset(
            df=sample_data,
            target_sequence_length=10,
            augment=False,
        )

        # 全てのシーケンスデータがfloat32であることを確認
        for seq_id, seq_data in dataset.sequence_data.items():
            assert seq_data["imu"].dtype == np.float32
            assert seq_data["missing_mask"].dtype == bool

    def test_missing_value_handling(self, sample_data):
        """欠損値処理のテスト."""
        dataset = IMUDataset(
            df=sample_data,
            target_sequence_length=10,
            augment=False,
        )

        # seq1のデータを確認（欠損値あり）
        seq1_data = dataset.sequence_data["seq1"]
        seq1_imu = seq1_data["imu"]
        seq1_mask = seq1_data["missing_mask"]

        # 欠損値が0で埋められていることを確認
        assert not np.isnan(seq1_imu).any()

        # 元のシーケンス長3で、インデックス2に欠損があったはず
        # （長さ正規化前の位置）
        assert seq1_data["original_length"] == 3

        # seq2のデータを確認（欠損値なし）
        seq2_data = dataset.sequence_data["seq2"]
        seq2_mask = seq2_data["missing_mask"]

        # seq2には欠損値がないことを確認
        assert seq2_data["original_length"] == 2
        # パディング部分以外はFalse
        assert not seq2_mask[:2].any()

    def test_polars_data_extraction(self, sample_data):
        """Polarsからのデータ抽出が正しく動作することを確認."""
        dataset = IMUDataset(
            df=sample_data,
            target_sequence_length=10,
            augment=False,
        )

        # 正しいシーケンス数が抽出されていることを確認
        assert len(dataset.sequence_data) == 2  # seq1, seq2

        # 各シーケンスのラベルが正しいことを確認
        seq1_data = dataset.sequence_data["seq1"]
        seq2_data = dataset.sequence_data["seq2"]

        assert seq1_data["gesture"] == "Forehead - scratch"
        assert seq2_data["gesture"] == "Neck - scratch"

        # binary_labelが正しいことを確認（両方ともtarget gesture）
        assert seq1_data["binary_label"] == 1
        assert seq2_data["binary_label"] == 1

    def test_sequence_length_normalization(self, sample_data):
        """シーケンス長の正規化が正しく動作することを確認."""
        target_length = 5
        dataset = IMUDataset(
            df=sample_data,
            target_sequence_length=target_length,
            augment=False,
        )

        # 全てのシーケンスが目標長になっていることを確認
        for seq_id, seq_data in dataset.sequence_data.items():
            assert seq_data["imu"].shape == (7, target_length)  # 7 features
            assert len(seq_data["missing_mask"]) == target_length

    def test_batch_processing(self, sample_data):
        """バッチ処理が正しく動作することを確認."""
        # より大きなデータセットを作成
        large_data = []
        for i in range(100):
            for j in range(10):
                large_data.append(
                    {
                        "row_id": f"seq{i}_{j}",
                        "sequence_id": f"seq{i}",
                        "sequence_counter": j,
                        "subject": f"subj{i % 10}",
                        "gesture": "Forehead - scratch" if i % 2 == 0 else "Wave hello",
                        "acc_x": float(i + j * 0.1),
                        "acc_y": float(i + j * 0.1 + 0.1),
                        "acc_z": float(i + j * 0.1 + 0.2),
                        "rot_w": 0.9,
                        "rot_x": 0.1,
                        "rot_y": 0.2,
                        "rot_z": 0.3,
                    }
                )

        large_df = pl.DataFrame(large_data)

        # エラーなく処理できることを確認
        dataset = IMUDataset(
            df=large_df,
            target_sequence_length=50,
            augment=False,
        )

        assert len(dataset.sequence_data) == 100

        # データ型が正しいことを確認
        for seq_id, seq_data in list(dataset.sequence_data.items())[:5]:
            assert seq_data["imu"].dtype == np.float32
