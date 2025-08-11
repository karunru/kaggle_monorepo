"""
test_exp014_minirocket.py

exp014のMiniRocketFeatureExtractorクラスの単体テスト
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp014"))

from config import Exp014Config
from dataset import MiniRocketFeatureExtractor


@pytest.fixture
def sample_config():
    """テスト用の設定を作成"""
    config = Exp014Config()

    # テスト用の小さなパラメータに変更
    config.minirocket.num_kernels = 100  # 小さなkernel数
    config.minirocket.n_jobs = 1  # 単一プロセス
    config.minirocket.random_state = 42
    config.minirocket.cache_enabled = True

    # 一時ディレクトリをキャッシュディレクトリとして使用
    temp_dir = tempfile.mkdtemp()
    config.minirocket.cache_dir = temp_dir

    yield config

    # クリーンアップ
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_imu_data():
    """テスト用のIMUデータを作成"""
    np.random.seed(42)

    # 2シーケンス、それぞれ異なる長さのIMUデータを作成
    sequence_data = []

    # シーケンス1: 長さ100
    seq1_length = 100
    for i in range(seq1_length):
        sequence_data.append(
            {
                "sequence_id": "test_seq_001",
                "sequence_counter": i,
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

    # シーケンス2: 長さ150
    seq2_length = 150
    for i in range(seq2_length):
        sequence_data.append(
            {
                "sequence_id": "test_seq_002",
                "sequence_counter": i,
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


class TestMiniRocketFeatureExtractor:
    """MiniRocketFeatureExtractorクラスのテスト"""

    def test_initialization(self, sample_config):
        """初期化テスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        assert extractor.config == sample_config
        assert extractor.minirocket_config == sample_config.minirocket
        assert not extractor.is_fitted
        assert extractor._feature_names is None
        assert extractor._output_dim is None

    def test_data_hash_creation(self, sample_config, sample_imu_data):
        """データハッシュ作成のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # 同じデータから同じハッシュが生成されることを確認
        hash1 = extractor._compute_data_hash(sample_imu_data)
        hash2 = extractor._compute_data_hash(sample_imu_data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # MD5ハッシュの最初の16文字

        # 異なるデータから異なるハッシュが生成されることを確認
        modified_data = sample_imu_data.with_columns(pl.col("linear_acc_x") * 2)
        hash3 = extractor._compute_data_hash(modified_data)
        assert hash1 != hash3

    def test_cache_key_creation(self, sample_config, sample_imu_data):
        """キャッシュキー作成のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        data_hash = extractor._compute_data_hash(sample_imu_data)

        # fit操作のキャッシュキー
        fit_key = extractor._create_cache_key(data_hash, "fit")
        assert fit_key.startswith("minirocket_fit_")
        assert data_hash in fit_key

        # transform操作のキャッシュキー
        transform_key = extractor._create_cache_key(data_hash, "transform")
        assert transform_key.startswith("minirocket_transform_")
        assert data_hash in transform_key

        # 異なる操作で異なるキーが生成される
        assert fit_key != transform_key

    def test_polars_to_sktime_conversion(self, sample_config, sample_imu_data):
        """Polars -> sktime Panel形式変換のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        panel_data = extractor._polars_to_sktime_panel(sample_imu_data)

        # MultiIndexの確認
        assert panel_data.index.names == ["sequence_id", "time"]

        # 特徴量列の確認
        expected_features = sample_config.minirocket.target_features
        assert list(panel_data.columns) == expected_features

        # シーケンス数の確認
        unique_sequences = panel_data.index.get_level_values("sequence_id").unique()
        assert len(unique_sequences) == 2  # test_seq_001, test_seq_002

        # データ型の確認
        assert panel_data.dtypes.apply(lambda x: x.kind in ["i", "f"]).all()

    def test_fit_functionality(self, sample_config, sample_imu_data):
        """fit機能のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # fit前の状態確認
        assert not extractor.is_fitted

        # fitの実行
        fitted_extractor = extractor.fit(sample_imu_data)

        # fit後の状態確認
        assert fitted_extractor is extractor  # 自身を返す
        assert extractor.is_fitted
        assert extractor._output_dim is not None
        assert extractor._feature_names is not None
        assert len(extractor._feature_names) == extractor._output_dim

        # 特徴量名の形式確認
        for name in extractor._feature_names:
            assert name.startswith("minirocket_")

    def test_transform_functionality(self, sample_config, sample_imu_data):
        """transform機能のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # fit前にtransformを呼ぶとエラー
        with pytest.raises(ValueError, match="MiniRocket is not fitted"):
            extractor.transform(sample_imu_data)

        # fitしてからtransform
        extractor.fit(sample_imu_data)
        transformed_features = extractor.transform(sample_imu_data)

        # 出力の形状確認
        assert isinstance(transformed_features, np.ndarray)
        assert transformed_features.dtype == np.float32
        assert transformed_features.shape[0] == 2  # 2シーケンス
        assert transformed_features.shape[1] == extractor._output_dim

    def test_fit_transform_functionality(self, sample_config, sample_imu_data):
        """fit_transform機能のテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # fit_transformの実行
        transformed_features = extractor.fit_transform(sample_imu_data)

        # 状態確認
        assert extractor.is_fitted
        assert isinstance(transformed_features, np.ndarray)
        assert transformed_features.shape[0] == 2  # 2シーケンス
        assert transformed_features.shape[1] == extractor._output_dim

        # fitとtransformを別々に実行した場合と同じ結果になることを確認
        extractor2 = MiniRocketFeatureExtractor(sample_config)
        extractor2.fit(sample_imu_data)
        transformed_features2 = extractor2.transform(sample_imu_data)

        # 同じrandom_stateなので結果は同じになるはず
        np.testing.assert_array_almost_equal(transformed_features, transformed_features2)

    def test_caching_functionality(self, sample_config, sample_imu_data):
        """キャッシュ機能のテスト"""
        # キャッシュが有効な場合のテスト
        extractor1 = MiniRocketFeatureExtractor(sample_config)

        # 初回実行
        transformed1 = extractor1.fit_transform(sample_imu_data)

        # キャッシュディレクトリにファイルが作成されているか確認
        cache_dir = Path(sample_config.minirocket.cache_dir)
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) > 0

        # 2回目の実行（キャッシュから読み込み）
        extractor2 = MiniRocketFeatureExtractor(sample_config)
        transformed2 = extractor2.fit_transform(sample_imu_data)

        # 同じ結果になることを確認
        np.testing.assert_array_equal(transformed1, transformed2)

        # どちらも正しく学習済み状態になっている
        assert extractor1.is_fitted
        assert extractor2.is_fitted
        assert extractor1._output_dim == extractor2._output_dim

    def test_output_dim_property(self, sample_config, sample_imu_data):
        """output_dimプロパティのテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # fit前は近似値を返す
        output_dim_before = extractor.output_dim
        assert output_dim_before == sample_config.minirocket.num_kernels

        # fitした後は実際の値を返す
        extractor.fit(sample_imu_data)
        output_dim_after = extractor.output_dim
        assert output_dim_after == extractor._output_dim
        assert isinstance(output_dim_after, int)
        assert output_dim_after > 0

    def test_feature_names_property(self, sample_config, sample_imu_data):
        """feature_namesプロパティのテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # fit前はデフォルト名を返す
        names_before = extractor.feature_names
        assert len(names_before) == sample_config.minirocket.num_kernels
        assert all(name.startswith("minirocket_") for name in names_before)

        # fitした後は実際の特徴量名を返す
        extractor.fit(sample_imu_data)
        names_after = extractor.feature_names
        assert len(names_after) == extractor._output_dim
        assert all(name.startswith("minirocket_") for name in names_after)

    def test_error_handling(self, sample_config, sample_imu_data):
        """エラーハンドリングのテスト"""
        extractor = MiniRocketFeatureExtractor(sample_config)

        # 必要な特徴量が不足しているデータでのテスト
        incomplete_data = sample_imu_data.select(["sequence_id", "sequence_counter", "linear_acc_x", "linear_acc_y"])

        with pytest.raises(ValueError, match="Missing target features"):
            extractor._polars_to_sktime_panel(incomplete_data)

    def test_different_sequence_lengths(self, sample_config):
        """異なるシーケンス長での動作テスト"""
        # 様々な長さのシーケンスを作成
        np.random.seed(42)
        sequence_data = []

        sequence_lengths = [50, 100, 200]  # 異なる長さ

        for seq_idx, length in enumerate(sequence_lengths):
            for i in range(length):
                sequence_data.append(
                    {
                        "sequence_id": f"test_seq_{seq_idx:03d}",
                        "sequence_counter": i,
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

        extractor = MiniRocketFeatureExtractor(sample_config)
        transformed = extractor.fit_transform(data)

        # 出力は3シーケンス分
        assert transformed.shape[0] == len(sequence_lengths)
        assert transformed.shape[1] == extractor.output_dim

    def test_reproducibility(self, sample_config, sample_imu_data):
        """再現性のテスト"""
        # 同じrandom_stateで2回実行
        extractor1 = MiniRocketFeatureExtractor(sample_config)
        transformed1 = extractor1.fit_transform(sample_imu_data)

        extractor2 = MiniRocketFeatureExtractor(sample_config)
        transformed2 = extractor2.fit_transform(sample_imu_data)

        # 完全に同じ結果になることを確認
        np.testing.assert_array_equal(transformed1, transformed2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
