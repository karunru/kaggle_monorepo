#!/usr/bin/env python3
"""
EXP014 モデル単体での動作確認スクリプト

train.pyの修正が正しく動作することを、モデル単体で確認する。
"""

import sys
from pathlib import Path

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp014"))

import torch
from config import Config
from model import CMISqueezeformer


def test_train_style_model_creation():
    """train.pyスタイルでのモデル作成テスト"""
    print("=== Train.py Style Model Creation Test ===")

    config = Config()

    print(f"Config input_dim: {config.model.input_dim}")  # 元の設定値
    print(f"Effective input_dim: {config.get_effective_input_dim()}")  # 実効値

    # train.pyと同じ方法でモデル作成
    model = CMISqueezeformer(
        input_dim=config.get_effective_input_dim(),  # ← 修正後の方法
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        num_classes=config.model.num_classes,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_config={
            "type": config.training.scheduler_type,
            "min_lr": config.training.scheduler_min_lr,
            "factor": config.training.scheduler_factor,
            "patience": config.training.scheduler_patience,
        },
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
        schedule_free_config=config.schedule_free.model_dump(),
        ema_config=config.ema.model_dump(),
        target_gestures=config.target_gestures,
        non_target_gestures=config.non_target_gestures,
    )

    print("✅ Model created successfully with train.py style")
    print(f"Model input_dim: {model.input_dim}")

    return model, config


def test_forward_pass_with_correct_dimensions():
    """正しい次元での前向き計算テスト"""
    print("\n=== Forward Pass with Correct Dimensions ===")

    model, config = test_train_style_model_creation()

    # 正しい次元でのダミーデータ作成
    batch_size = 4
    input_dim = config.get_effective_input_dim()  # 352次元
    seq_len = config.preprocessing.target_sequence_length

    print(f"Creating dummy input: batch_size={batch_size}, input_dim={input_dim}, seq_len={seq_len}")

    dummy_imu = torch.randn(batch_size, input_dim, seq_len)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Demographics特徴量（オプション）
    dummy_demographics = None
    if config.demographics.enabled:
        dummy_demographics = {
            "adult_child": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            "sex": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            "handedness": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            "age": torch.rand(batch_size, dtype=torch.float32) * 50 + 10,
            "height_cm": torch.rand(batch_size, dtype=torch.float32) * 60 + 140,
            "shoulder_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 30 + 40,
            "elbow_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 25 + 20,
        }

    # 前向き計算実行
    try:
        model.eval()
        with torch.no_grad():
            multiclass_logits, binary_logits = model(dummy_imu, dummy_attention_mask, dummy_demographics)

        print("✅ Forward pass successful")
        print(f"Input shape: {dummy_imu.shape}")
        print(f"Multiclass output: {multiclass_logits.shape}")
        print(f"Binary output: {binary_logits.shape}")

        # 出力値の妥当性確認
        assert multiclass_logits.shape == (batch_size, config.model.num_classes)
        assert binary_logits.shape == (batch_size, 1)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()

        print("✅ Output validation passed")
        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dimension_compatibility():
    """次元互換性のテスト"""
    print("\n=== Dimension Compatibility Test ===")

    config = Config()

    # 基本情報の表示
    print("Feature dimensions breakdown:")
    print(f"  Base IMU features: {config.model.base_imu_features}")
    if config.rocket.enabled:
        print(f"  MiniRocket features: {config.rocket.num_kernels}")
        print(f"  MiniRocket target features: {len(config.rocket.target_features)}")
        for i, feature in enumerate(config.rocket.target_features):
            print(f"    {i + 1}. {feature}")

    total_expected = config.model.base_imu_features + (config.rocket.num_kernels if config.rocket.enabled else 0)
    effective_dim = config.get_effective_input_dim()

    print(f"  Total expected: {total_expected}")
    print(f"  Effective dimension: {effective_dim}")

    if total_expected == effective_dim:
        print("✅ Dimension calculation is consistent")
        return True
    else:
        print(f"❌ Dimension mismatch: expected {total_expected}, got {effective_dim}")
        return False


def main():
    """メインテスト実行"""
    print("EXP014 Model-Only Fix Verification Test")
    print("=" * 60)

    results = []

    # 各テスト実行
    try:
        model, config = test_train_style_model_creation()
        results.append(("Train-style Model Creation", True))
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        results.append(("Train-style Model Creation", False))
        import traceback

        traceback.print_exc()

    forward_success = test_forward_pass_with_correct_dimensions()
    results.append(("Forward Pass (Correct Dims)", forward_success))

    dimension_consistency = test_dimension_compatibility()
    results.append(("Dimension Compatibility", dimension_consistency))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("💡 The train.py input dimension fix is working correctly.")
        print("💡 The RuntimeError 'Expected size ... [128, 16] but got: [128, 352]' should be resolved.")
        return True
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
