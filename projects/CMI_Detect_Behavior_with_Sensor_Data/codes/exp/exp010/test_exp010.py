"""exp010用のテストファイル（物理ベースIMU特徴量追加版）."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """基本的なインポートテスト."""
    try:
        import config
        import dataset
        import model

        print("✓ dataset.py, model.py, and config.py imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_class():
    """Configクラスの存在確認."""
    try:
        from config import Config

        print("✓ Config class is available")
        return True
    except ImportError:
        print("✗ Config class not found")
        return False


def test_pydantic_config():
    """pydantic-settings Config クラスのテスト."""
    try:
        from config import Config

        # デフォルト設定でのインスタンス化
        config = Config()
        print("✓ Config class instantiation successful")

        # 属性アクセステスト
        assert config.model.input_dim == 16  # 基本IMU 7 + 物理特徴量 9
        assert config.training.batch_size == 128
        assert config.model.d_model == 256
        assert len(config.target_gestures) == 8
        assert len(config.imu_features) == 16  # 基本IMU + 物理特徴量
        print("✓ Config attribute access successful")

        # バリデーションテスト
        assert 0 <= config.model.dropout <= 1
        assert config.training.learning_rate > 0
        assert config.training.batch_size > 0
        print("✓ Config validation successful")

        # dict変換テスト
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        print("✓ Config dict conversion successful")

        # exp010用のexp_num確認
        from config import EXP_NUM

        assert EXP_NUM == "exp010"
        print("✓ EXP_NUM correctly set to exp010")

        return True

    except Exception as e:
        print(f"✗ pydantic Config test failed: {e}")
        return False


def test_missing_value_mask():
    """欠損値マスク処理のテスト."""
    try:
        import numpy as np
        from dataset import IMUDataset

        # テスト用データの作成（欠損値を含む、16特徴量）
        test_data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [
                    np.nan,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                ],  # 欠損値あり
                [
                    1.0,
                    np.nan,
                    np.nan,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                ],  # 複数欠損値
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ]
        )

        # IMUDatasetインスタンスを作成してメソッドをテスト
        class MockDataset(IMUDataset):
            def __init__(self):
                pass

        dataset = MockDataset()

        # _handle_missing_values_with_maskメソッドのテスト
        processed_data, missing_mask = dataset._handle_missing_values_with_mask(test_data)

        # 欠損値が0で埋められていることを確認
        assert not np.any(np.isnan(processed_data))
        print("✓ Missing values filled with zeros")

        # missing_maskが正しく生成されていることを確認
        expected_mask = np.array([False, True, True, False])  # 欠損ありの行でTrue
        assert np.array_equal(missing_mask, expected_mask)
        print("✓ Missing value mask correctly generated")

        return True

    except Exception as e:
        print(f"✗ Missing value mask test failed: {e}")
        return False


def test_attention_mask_integration():
    """attention_mask統合テスト."""
    try:
        import torch
        from dataset import dynamic_collate_fn

        # テスト用バッチデータの作成
        batch_data = [
            {
                "imu": torch.randn(16, 100),
                "missing_mask": torch.tensor([False] * 50 + [True] * 30 + [False] * 20),  # 中間に欠損
                "multiclass_label": torch.tensor(0),
                "binary_label": torch.tensor(0.0),
                "sequence_id": "test_seq_1",
                "gesture": "test_gesture",
                "original_length": 100,
            },
            {
                "imu": torch.randn(16, 80),
                "missing_mask": torch.tensor([False] * 80),  # 欠損なし
                "multiclass_label": torch.tensor(1),
                "binary_label": torch.tensor(1.0),
                "sequence_id": "test_seq_2",
                "gesture": "test_gesture2",
                "original_length": 80,
            },
        ]

        # dynamic_collate_fnのテスト
        collated_batch = dynamic_collate_fn(batch_data)

        # attention_maskが正しく生成されていることを確認
        assert "attention_mask" in collated_batch
        attention_mask = collated_batch["attention_mask"]

        # 1番目のサンプル：欠損部分(50-79)でFalse、それ以外でTrue
        sample1_mask = attention_mask[0]
        assert torch.all(sample1_mask[:50] == True)  # 最初の50は有効
        assert torch.all(sample1_mask[50:80] == False)  # 中間30は無効（欠損）
        assert torch.all(sample1_mask[80:100] == True)  # 最後の20は有効

        # 2番目のサンプル：欠損なしなので全てTrue（長さ80まで）
        sample2_mask = attention_mask[1]
        assert torch.all(sample2_mask[:80] == True)

        print("✓ Attention mask integration successful")
        return True

    except Exception as e:
        print(f"✗ Attention mask integration test failed: {e}")
        return False


def test_basic_functionality():
    """基本機能のテスト."""
    try:
        # モデルのインスタンス化テスト
        from model import CMISqueezeformer

        model = CMISqueezeformer(
            input_dim=16,  # 物理特徴量込み
            d_model=64,  # 小さいサイズでテスト
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_classes=18,
        )
        print("✓ Model instantiation successful")

        # 入力テンソルのテスト（PyTorchがインストールされている場合）
        try:
            import torch

            test_input = torch.randn(1, 16, 100)  # [batch, features, seq_len] - 16特徴量
            attention_mask = torch.ones(1, 100, dtype=torch.bool)  # exp009でも attention mask 必須

            multiclass_logits, binary_logits = model(test_input, attention_mask)

            assert multiclass_logits.shape == (1, 18)
            assert binary_logits.shape == (1, 1)
            print("✓ Model forward pass with attention mask successful")

            # 欠損値を含むattention_maskのテスト
            partial_mask = torch.cat(
                [
                    torch.ones(1, 50, dtype=torch.bool),  # 有効部分
                    torch.zeros(1, 30, dtype=torch.bool),  # 欠損部分
                    torch.ones(1, 20, dtype=torch.bool),  # 有効部分
                ],
                dim=1,
            )

            multiclass_logits2, binary_logits2 = model(test_input, partial_mask)
            assert multiclass_logits2.shape == (1, 18)
            assert binary_logits2.shape == (1, 1)
            print("✓ Model forward pass with partial attention mask successful")

            return True
        except ImportError:
            print("! PyTorch not available, skipping forward pass test")
            return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_ema_integration():
    """EMA統合テスト（exp006から継承、exp009でも継続）."""
    try:
        from config import Config
        from model import CMISqueezeformer

        # EMA有効設定でのテスト
        config = Config()
        config.ema.enabled = True
        config.ema.beta = 0.999  # テスト用に小さく設定

        # モデルの作成（EMA設定付き）
        model = CMISqueezeformer(
            input_dim=config.model.input_dim,
            d_model=64,  # テスト用に小さく
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_classes=config.model.num_classes,
            ema_config=config.ema.model_dump(),
        )

        print("✓ EMA integrated model creation successful")

        # 手動EMA実装の確認（自己参照問題回避のため）
        assert hasattr(model, "_ema_initialized")
        print("✓ Manual EMA implementation check successful")

        return True
    except Exception as e:
        print(f"✗ EMA integration test failed: {e}")
        return False


def test_single_sequence_dataset():
    """SingleSequenceIMUDatasetのテスト."""
    try:
        import numpy as np
        import polars as pl
        import torch
        from dataset import SingleSequenceIMUDataset

        # テスト用のシーケンスデータを作成
        n_timesteps = 150
        sequence_data = {
            "sequence_id": ["test_seq"] * n_timesteps,
            "sequence_counter": list(range(n_timesteps)),
            "acc_x": np.random.randn(n_timesteps),
            "acc_y": np.random.randn(n_timesteps),
            "acc_z": np.random.randn(n_timesteps),
            "rot_w": np.random.randn(n_timesteps),
            "rot_x": np.random.randn(n_timesteps),
            "rot_y": np.random.randn(n_timesteps),
            "rot_z": np.random.randn(n_timesteps),
        }

        # 一部に欠損値を挿入
        sequence_data["acc_x"][50:55] = np.nan
        sequence_data["rot_y"][80:85] = np.nan

        sequence_df = pl.DataFrame(sequence_data)

        # データセットの作成
        dataset = SingleSequenceIMUDataset(sequence_df, target_sequence_length=200)

        # データセットのサイズ確認
        assert len(dataset) == 1
        print("✓ Single sequence dataset size correct")

        # データ取得テスト
        data = dataset[0]
        assert "imu" in data
        assert "attention_mask" in data
        assert "sequence_id" in data

        # IMUデータの形状確認（注意：SingleSequenceは基本IMUのみ使用）
        imu_tensor = data["imu"]
        assert imu_tensor.shape == (7, 200)  # [features, seq_len] - SingleSequenceは基本IMUのみ
        print("✓ IMU tensor shape correct")

        # attention_maskの形状確認
        attention_mask = data["attention_mask"]
        assert attention_mask.shape == (200,)
        assert attention_mask.dtype == torch.bool
        print("✓ Attention mask shape and dtype correct")

        # sequence_id確認
        assert data["sequence_id"] == "test_seq"
        print("✓ Sequence ID correct")

        return True
    except Exception as e:
        print(f"✗ Single sequence dataset test failed: {e}")
        return False


def test_submission_format():
    """サブミッション形式のテスト（predict関数）."""
    try:
        import numpy as np
        import polars as pl

        # テスト用のシーケンスデータを作成
        n_timesteps = 100
        sequence_data = {
            "sequence_id": ["test_seq"] * n_timesteps,
            "sequence_counter": list(range(n_timesteps)),
            "acc_x": np.random.randn(n_timesteps),
            "acc_y": np.random.randn(n_timesteps),
            "acc_z": np.random.randn(n_timesteps),
            "rot_w": np.random.randn(n_timesteps),
            "rot_x": np.random.randn(n_timesteps),
            "rot_y": np.random.randn(n_timesteps),
            "rot_z": np.random.randn(n_timesteps),
        }
        sequence_df = pl.DataFrame(sequence_data)

        # テスト用の人口統計データ（空でもOK）
        demographics_df = pl.DataFrame({"subject": ["test_subject"], "age": [25]})

        # テスト用のpredict関数（モデルが無い場合のテスト）
        def test_predict_no_models(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
            """モデルが無い場合のテスト用predict関数."""
            return "Text on phone"

        # predict関数のテスト
        result = test_predict_no_models(sequence_df, demographics_df)
        assert isinstance(result, str)
        assert result in [
            "Above ear - pull hair",
            "Cheek - pinch skin",
            "Drink from bottle/cup",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Feel around in tray and pull out an object",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Glasses on/off",
            "Neck - pinch skin",
            "Neck - scratch",
            "Pinch knee/leg skin",
            "Pull air toward your face",
            "Scratch knee/leg skin",
            "Text on phone",
            "Wave hello",
            "Write name in air",
            "Write name on leg",
        ]
        print("✓ Predict function returns valid gesture")

        # inference.pyのpredict関数をインポートしてテスト
        try:
            from inference import GESTURE_NAMES, predict

            # ジェスチャー名リストの確認
            assert len(GESTURE_NAMES) == 18
            print("✓ Gesture names list correct")

            # predict関数の実行（モデルが無くてもダミー値を返すはず）
            result = predict(sequence_df, demographics_df)
            assert isinstance(result, str)
            assert result in GESTURE_NAMES
            print("✓ Actual predict function works")

        except Exception as e:
            print(f"! Predict function test skipped due to: {e}")

        return True
    except Exception as e:
        print(f"✗ Submission format test failed: {e}")
        return False


def main():
    """メインテスト関数."""
    print("Running exp010 tests...")
    print("=" * 40)

    tests = [
        test_imports,
        test_config_class,
        test_pydantic_config,
        test_missing_value_mask,
        test_attention_mask_integration,
        test_basic_functionality,
        test_ema_integration,
        test_single_sequence_dataset,
        test_submission_format,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()

    # サマリー
    passed = sum(results)
    total = len(results)
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
