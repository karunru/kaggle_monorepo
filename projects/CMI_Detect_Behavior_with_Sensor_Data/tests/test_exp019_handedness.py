"""exp019 利き手反転オーグメンテーションのテスト."""

import numpy as np
import polars as pl
import pytest
from codes.exp.exp019.dataset import HandednessAugmentation


@pytest.fixture
def sample_demographics_df():
    """テスト用のdemographicsデータ."""
    return pl.DataFrame(
        {
            "subject": ["S001", "S002", "S003"],
            "handedness": [1, 0, 1],  # 右利き、左利き、右利き
            "age": [25, 30, 35],
            "sex": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_imu_data():
    """テスト用のIMUデータ."""
    # [features, seq_len] 形式
    seq_len = 10
    return np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # acc_x
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # acc_y
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # acc_z
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # rot_w
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # rot_x
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # rot_y
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # rot_z
            # 物理ベース特徴量
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],  # linear_acc_x
            [0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1],  # linear_acc_y
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],  # linear_acc_z
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # linear_acc_mag
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # linear_acc_mag_jerk
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # angular_vel_x
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # angular_vel_y
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # angular_vel_z
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # angular_distance
        ],
        dtype=np.float32,
    )


@pytest.fixture
def imu_cols():
    """IMU列名."""
    return [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        "linear_acc_x",
        "linear_acc_y",
        "linear_acc_z",
        "linear_acc_mag",
        "linear_acc_mag_jerk",
        "angular_vel_x",
        "angular_vel_y",
        "angular_vel_z",
        "angular_distance",
    ]


class TestHandednessAugmentation:
    """利き手反転オーグメンテーションのテスト."""

    def test_initialization(self, sample_demographics_df):
        """初期化のテスト."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=0.5)

        assert aug.flip_probability == 0.5
        assert len(aug.subject_to_handedness) == 3
        assert aug.subject_to_handedness["S001"] == 1  # 右利き
        assert aug.subject_to_handedness["S002"] == 0  # 左利き
        assert aug.subject_to_handedness["S003"] == 1  # 右利き

    def test_initialization_empty_demographics(self):
        """空のdemographicsでの初期化テスト."""
        aug = HandednessAugmentation(None, flip_probability=0.3)

        assert aug.flip_probability == 0.3
        assert len(aug.subject_to_handedness) == 0

    def test_get_handedness(self, sample_demographics_df):
        """利き手情報の取得テスト."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=0.5)

        assert aug.get_handedness("S001") == 1  # 右利き
        assert aug.get_handedness("S002") == 0  # 左利き
        assert aug.get_handedness("S999") == 1  # 存在しない被験者（デフォルト：右利き）

    def test_flip_imu_data_features_first(self, sample_imu_data, imu_cols):
        """IMUデータ反転のテスト（[features, seq_len]形式）."""
        aug = HandednessAugmentation(None, flip_probability=1.0)

        original_data = sample_imu_data.copy()
        flipped_data = aug.flip_imu_data(sample_imu_data, imu_cols)

        # Y軸データが反転されているかチェック
        acc_y_idx = imu_cols.index("acc_y")
        rot_y_idx = imu_cols.index("rot_y")
        linear_acc_y_idx = imu_cols.index("linear_acc_y")
        angular_vel_y_idx = imu_cols.index("angular_vel_y")

        # Y軸の値が反転されている
        np.testing.assert_array_equal(flipped_data[acc_y_idx], -original_data[acc_y_idx])
        np.testing.assert_array_equal(flipped_data[rot_y_idx], -original_data[rot_y_idx])
        np.testing.assert_array_equal(flipped_data[linear_acc_y_idx], -original_data[linear_acc_y_idx])
        np.testing.assert_array_equal(flipped_data[angular_vel_y_idx], -original_data[angular_vel_y_idx])

        # 他の軸は変更されていない
        acc_x_idx = imu_cols.index("acc_x")
        acc_z_idx = imu_cols.index("acc_z")
        np.testing.assert_array_equal(flipped_data[acc_x_idx], original_data[acc_x_idx])
        np.testing.assert_array_equal(flipped_data[acc_z_idx], original_data[acc_z_idx])

    def test_flip_imu_data_seq_len_first(self, sample_imu_data, imu_cols):
        """IMUデータ反転のテスト（[seq_len, features]形式）."""
        aug = HandednessAugmentation(None, flip_probability=1.0)

        # [seq_len, features]形式に変換
        input_data = sample_imu_data.T  # 転置
        original_data = input_data.copy()

        flipped_data = aug.flip_imu_data(input_data, imu_cols)

        # 出力も[seq_len, features]形式であることを確認
        assert flipped_data.shape == input_data.shape

        # Y軸データが反転されているかチェック
        acc_y_idx = imu_cols.index("acc_y")
        rot_y_idx = imu_cols.index("rot_y")

        np.testing.assert_array_equal(flipped_data[:, acc_y_idx], -original_data[:, acc_y_idx])
        np.testing.assert_array_equal(flipped_data[:, rot_y_idx], -original_data[:, rot_y_idx])

        # 他の軸は変更されていない
        acc_x_idx = imu_cols.index("acc_x")
        np.testing.assert_array_equal(flipped_data[:, acc_x_idx], original_data[:, acc_x_idx])

    def test_should_flip_probability(self, sample_demographics_df):
        """確率的反転の動作テスト."""
        # 常に反転
        aug_always = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)
        assert aug_always.should_flip() is True

        # 反転しない
        aug_never = HandednessAugmentation(sample_demographics_df, flip_probability=0.0)
        assert aug_never.should_flip() is False

        # 中間的な確率での動作確認（統計的テスト）
        aug_half = HandednessAugmentation(sample_demographics_df, flip_probability=0.5)
        flip_results = [aug_half.should_flip() for _ in range(1000)]
        flip_ratio = sum(flip_results) / len(flip_results)

        # 50%前後になることを確認（±10%の範囲）
        assert 0.4 <= flip_ratio <= 0.6

    def test_should_flip_method(self, sample_demographics_df):
        """should_flipメソッドのテスト（常に反転）."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)
        assert aug.should_flip() is True

    def test_should_not_flip_method(self, sample_demographics_df):
        """should_flipメソッドのテスト（反転なし）."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=0.0)
        assert aug.should_flip() is False

    def test_missing_imu_columns(self, sample_demographics_df, imu_cols):
        """欠損IMU列の処理テスト."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)

        # acc_yとrot_yが存在しない場合
        incomplete_imu_cols = ["acc_x", "acc_z", "rot_w", "rot_x", "rot_z"]
        incomplete_data = np.random.rand(5, 10).astype(np.float32)

        original_data = incomplete_data.copy()
        result = aug.flip_imu_data(incomplete_data, incomplete_imu_cols)

        # 反転対象の列が存在しないため、データは変更されない
        np.testing.assert_array_equal(result, original_data)

    def test_data_integrity(self, sample_demographics_df, sample_imu_data, imu_cols):
        """データの整合性チェック."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)

        original_data = sample_imu_data.copy()
        result = aug.flip_imu_data(sample_imu_data, imu_cols)

        # 形状が保持されている
        assert result.shape == original_data.shape

        # データ型が保持されている
        assert result.dtype == original_data.dtype

        # 元のデータが変更されていない（copyが正しく動作）
        np.testing.assert_array_equal(sample_imu_data, original_data)

    def test_empty_data_handling(self, sample_demographics_df, imu_cols):
        """空データの処理テスト."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)

        # 空のデータ
        empty_data = np.empty((len(imu_cols), 0), dtype=np.float32)
        result = aug.flip_imu_data(empty_data, imu_cols)

        assert result.shape == empty_data.shape
        assert result.dtype == empty_data.dtype

    def test_single_timestep_data(self, sample_demographics_df, imu_cols):
        """単一タイムステップデータの処理テスト."""
        aug = HandednessAugmentation(sample_demographics_df, flip_probability=1.0)

        # 1つのタイムステップのみのデータ
        single_data = np.random.rand(len(imu_cols), 1).astype(np.float32)
        original_data = single_data.copy()

        result = aug.flip_imu_data(single_data, imu_cols)

        # Y軸データが反転されているかチェック
        acc_y_idx = imu_cols.index("acc_y")
        rot_y_idx = imu_cols.index("rot_y")

        np.testing.assert_array_almost_equal(result[acc_y_idx], -original_data[acc_y_idx])
        np.testing.assert_array_almost_equal(result[rot_y_idx], -original_data[rot_y_idx])

        # 他の軸は変更されていない
        acc_x_idx = imu_cols.index("acc_x")
        np.testing.assert_array_almost_equal(result[acc_x_idx], original_data[acc_x_idx])
