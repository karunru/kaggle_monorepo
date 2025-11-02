"""exp002用のテストファイル."""

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
        assert config.model.input_dim == 7
        assert config.training.batch_size == 32
        assert config.model.d_model == 256
        assert len(config.target_gestures) == 8
        assert len(config.imu_features) == 7
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

        return True

    except Exception as e:
        print(f"✗ pydantic Config test failed: {e}")
        return False


def test_basic_functionality():
    """基本機能のテスト."""
    try:
        # モデルのインスタンス化テスト
        from model import CMISqueezeformer

        model = CMISqueezeformer(
            input_dim=7,
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

            test_input = torch.randn(1, 7, 100)  # [batch, features, seq_len]
            multiclass_logits, binary_logits = model(test_input)

            assert multiclass_logits.shape == (1, 18)
            assert binary_logits.shape == (1, 1)
            print("✓ Model forward pass successful")
            return True
        except ImportError:
            print("! PyTorch not available, skipping forward pass test")
            return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """メインテスト関数."""
    print("Running exp002 tests...")
    print("=" * 40)

    tests = [test_imports, test_config_class, test_pydantic_config, test_basic_functionality]

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
