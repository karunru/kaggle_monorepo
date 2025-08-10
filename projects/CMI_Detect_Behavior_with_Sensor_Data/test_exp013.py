#!/usr/bin/env python3
"""exp013の動作確認テストスクリプト."""

import sys
from pathlib import Path

import torch

# exp013をインポートパスに追加
sys.path.insert(0, str(Path(__file__).parent / "codes" / "exp" / "exp013"))

from config import Config
from model import CMISqueezeformer


def test_model_initialization():
    """モデル初期化テスト."""
    print("Testing model initialization...")

    # 設定を読み込み
    config = Config()

    # loss typeがsoft_f1になっているか確認
    assert config.loss.type == "soft_f1", f"Expected loss type 'soft_f1', got '{config.loss.type}'"
    print(f"✓ Loss type is correctly set to: {config.loss.type}")

    # モデルを初期化
    model = CMISqueezeformer(
        input_dim=7,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        num_classes=18,
        loss_config=config.loss.model_dump(),
        ema_config={
            "enabled": True,
            "beta": 0.9999,
            "update_after_step": 100,
            "update_every": 10,
            "update_model_with_ema_every": 1000,
        },
    )

    print("✓ Model initialized successfully")

    # setupを呼び出してEMAを初期化
    model.setup("fit")

    # EMAラッパーが初期化されているか確認
    assert hasattr(model, "ema_wrapper"), "EMA wrapper not found"
    assert model.ema_wrapper is not None, "EMA wrapper is None"
    print("✓ EMA wrapper initialized successfully")

    return model


def test_forward_pass(model):
    """前向き計算テスト."""
    print("\nTesting forward pass...")

    # テストデータ
    batch_size = 2
    seq_len = 200
    imu_data = torch.randn(batch_size, 7, seq_len)

    # 前向き計算
    model.eval()
    with torch.no_grad():
        multiclass_logits, binary_logits = model(imu_data)

    # 出力形状の確認
    assert multiclass_logits.shape == (batch_size, 18), f"Unexpected multiclass shape: {multiclass_logits.shape}"
    assert binary_logits.shape == (batch_size, 1), f"Unexpected binary shape: {binary_logits.shape}"

    print("✓ Forward pass successful")
    print(f"  - Input shape: {imu_data.shape}")
    print(f"  - Multiclass output shape: {multiclass_logits.shape}")
    print(f"  - Binary output shape: {binary_logits.shape}")


def test_training_step(model):
    """訓練ステップテスト."""
    print("\nTesting training step...")

    # テストバッチ
    batch = {
        "imu": torch.randn(2, 7, 200),
        "multiclass_label": torch.randint(0, 18, (2,)),
        "binary_label": torch.randint(0, 2, (2,)).float(),
        "attention_mask": torch.ones(2, 200, dtype=torch.bool),
    }

    # 訓練モードに設定
    model.train()

    # 訓練ステップ実行
    loss = model.training_step(batch, 0)

    assert loss is not None, "Loss is None"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"

    print("✓ Training step successful")
    print(f"  - Loss: {loss.item():.4f}")

    # EMAが更新されるか確認
    if hasattr(model, "ema_wrapper") and model.ema_wrapper is not None:
        print("✓ EMA wrapper is available for updates")


def test_soft_f1_loss():
    """SoftF1Loss動作テスト."""
    print("\nTesting SoftF1Loss...")

    from model import BinarySoftF1Loss, MulticlassSoftF1Loss

    # バイナリSoftF1Loss
    binary_loss = BinarySoftF1Loss()
    probs = torch.tensor([0.8, 0.3, 0.9, 0.2])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_val = binary_loss(probs, targets)

    assert not torch.isnan(loss_val), "Binary SoftF1Loss is NaN"
    print(f"✓ Binary SoftF1Loss: {loss_val.item():.4f}")

    # マルチクラスSoftF1Loss
    multiclass_loss = MulticlassSoftF1Loss(num_classes=3)
    probs = torch.softmax(torch.randn(4, 3), dim=-1)
    targets = torch.tensor([0, 1, 2, 1])
    loss_val = multiclass_loss(probs, targets)

    assert not torch.isnan(loss_val), "Multiclass SoftF1Loss is NaN"
    print(f"✓ Multiclass SoftF1Loss: {loss_val.item():.4f}")


def main():
    """メインテスト実行."""
    print("=" * 50)
    print("exp013 動作確認テスト")
    print("=" * 50)

    try:
        # 各テストを実行
        model = test_model_initialization()
        test_forward_pass(model)
        test_training_step(model)
        test_soft_f1_loss()

        print("\n" + "=" * 50)
        print("✅ すべてのテストが成功しました！")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ テスト失敗: {e}")
        print("=" * 50)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
