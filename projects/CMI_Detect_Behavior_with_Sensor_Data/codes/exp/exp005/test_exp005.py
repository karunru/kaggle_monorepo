"""exp005用のテストファイル（Length Grouping + Schedule Free統合）."""

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
        assert config.training.batch_size == 128  # exp005ではバッチサイズを128に変更
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


def test_length_grouped_sampler():
    """LengthGroupedSamplerのテスト."""
    try:
        from dataset import LengthGroupedSampler

        print("✓ LengthGroupedSampler import successful")
        return True
    except ImportError:
        print("✗ LengthGroupedSampler not found")
        return False


def test_dynamic_collate_fn():
    """動的Collate関数のテスト."""
    try:
        from dataset import dynamic_collate_fn

        print("✓ dynamic_collate_fn import successful")
        return True
    except ImportError:
        print("✗ dynamic_collate_fn not found")
        return False


def test_length_grouping_config():
    """LengthGroupingConfigのテスト."""
    try:
        from config import Config

        config = Config()

        # Length Grouping設定のテスト
        assert hasattr(config, "length_grouping")
        assert hasattr(config.length_grouping, "enabled")
        assert hasattr(config.length_grouping, "use_dynamic_padding")
        assert hasattr(config.length_grouping, "mega_batch_multiplier")
        assert hasattr(config.length_grouping, "percentile_max_length")

        # デフォルト値のテスト
        assert config.length_grouping.mega_batch_multiplier >= 1
        assert 0 < config.length_grouping.percentile_max_length <= 1

        print("✓ LengthGroupingConfig test successful")
        return True
    except Exception as e:
        print(f"✗ LengthGroupingConfig test failed: {e}")
        return False


def test_schedule_free_config():
    """ScheduleFreeConfigのテスト."""
    try:
        from config import Config

        config = Config()

        # Schedule Free設定のテスト
        assert hasattr(config, "schedule_free")
        assert hasattr(config.schedule_free, "enabled")
        assert hasattr(config.schedule_free, "optimizer_type")
        assert hasattr(config.schedule_free, "learning_rate_multiplier")
        assert hasattr(config.schedule_free, "warmup_steps")
        assert hasattr(config.schedule_free, "batch_norm_calibration_steps")

        # デフォルト値のテスト
        assert config.schedule_free.learning_rate_multiplier > 0
        assert config.schedule_free.warmup_steps >= 0
        assert config.schedule_free.batch_norm_calibration_steps >= 1
        assert config.schedule_free.optimizer_type in ["RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree"]

        print("✓ ScheduleFreeConfig test successful")
        return True
    except Exception as e:
        print(f"✗ ScheduleFreeConfig test failed: {e}")
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
            attention_mask = torch.ones(1, 100, dtype=torch.bool)  # exp005では attention mask 追加

            multiclass_logits, binary_logits = model(test_input, attention_mask)

            assert multiclass_logits.shape == (1, 18)
            assert binary_logits.shape == (1, 1)
            print("✓ Model forward pass with attention mask successful")
            return True
        except ImportError:
            print("! PyTorch not available, skipping forward pass test")
            return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_schedule_free_imports():
    """Schedule Free optimizer importのテスト."""
    try:
        # Schedule Free optimizerが利用可能かテスト
        from model import SCHEDULEFREE_AVAILABLE

        print(f"✓ Schedule Free availability: {SCHEDULEFREE_AVAILABLE}")

        if SCHEDULEFREE_AVAILABLE:
            print("✓ Schedule Free optimizers are available")
        else:
            print("! Schedule Free optimizers not available (optional dependency)")

        return True
    except Exception as e:
        print(f"✗ Schedule Free import test failed: {e}")
        return False


def test_integrated_functionality():
    """統合機能のテスト（Length Grouping + Schedule Free）."""
    try:
        from config import Config
        from model import CMISqueezeformer

        # 統合設定の作成
        config = Config()
        config.length_grouping.enabled = True
        config.length_grouping.use_dynamic_padding = True
        config.schedule_free.enabled = False  # テスト用に無効化（依存関係なしでテスト）

        # モデルの作成（統合設定付き）
        model = CMISqueezeformer(
            input_dim=config.model.input_dim,
            d_model=64,  # テスト用に小さく
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_classes=config.model.num_classes,
            schedule_free_config=config.schedule_free.model_dump(),
        )

        print("✓ Integrated model creation successful")

        # 設定の検証
        assert hasattr(config, "length_grouping")
        assert hasattr(config, "schedule_free")
        print("✓ Integrated configuration access successful")

        return True
    except Exception as e:
        print(f"✗ Integrated functionality test failed: {e}")
        return False


def main():
    """メインテスト関数."""
    print("Running exp005 tests...")
    print("=" * 40)

    tests = [
        test_imports,
        test_config_class,
        test_pydantic_config,
        test_length_grouped_sampler,
        test_dynamic_collate_fn,
        test_length_grouping_config,
        test_schedule_free_config,
        test_schedule_free_imports,
        test_basic_functionality,
        test_integrated_functionality,
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
