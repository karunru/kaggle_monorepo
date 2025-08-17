#!/usr/bin/env python3
"""修正後のモデル前方パスの検証."""

import sys
from pathlib import Path

import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import Config
from dataset import IMUDataModule
from model import CMISqueezeformer


def test_model_forward_pass():
    """モデルの前方パスを検証."""
    print("=== モデル前方パス検証 ===")

    # 設定読み込み
    config = Config()

    # データモジュール作成（少量のデータで高速化）
    data_module = IMUDataModule(config, fold=0)
    data_module.setup("fit")

    # 入力次元を動的に取得
    actual_input_dim = len(data_module.train_dataset.imu_cols)
    print(f"Model input_dim: {actual_input_dim}")

    # モデル作成
    model = CMISqueezeformer(
        input_dim=actual_input_dim,
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
        demographics_config=config.demographics.model_dump(),
        target_gestures=config.target_gestures,
        non_target_gestures=config.non_target_gestures,
        id_to_gesture=data_module.train_dataset.id_to_gesture
        if hasattr(data_module, "train_dataset") and hasattr(data_module.train_dataset, "id_to_gesture")
        else None,
    )

    print("Model created successfully")

    # データローダーから1バッチ取得
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")

    # モデル評価モードに設定
    model.eval()

    with torch.no_grad():
        try:
            # バッチから適切な要素を抽出
            imu_data = batch["imu"]
            attention_mask = batch.get("missing_mask", None)
            demographics = batch.get("demographics", None)

            print(f"IMU shape: {imu_data.shape}")
            print(f"Mask shape: {attention_mask.shape if attention_mask is not None else None}")

            # 前方パス実行
            multiclass_logits, binary_logits = model(imu_data, attention_mask, demographics)
            print("✅ Forward pass successful!")

            # 各出力をチェック
            print(
                f"multiclass_logits: shape={multiclass_logits.shape}, nan={torch.isnan(multiclass_logits).any()}, inf={torch.isinf(multiclass_logits).any()}"
            )
            print(
                f"binary_logits: shape={binary_logits.shape}, nan={torch.isnan(binary_logits).any()}, inf={torch.isinf(binary_logits).any()}"
            )

            if torch.isnan(multiclass_logits).any():
                print("  ⚠️ NaN detected in multiclass_logits")
            if torch.isinf(multiclass_logits).any():
                print("  ⚠️ Inf detected in multiclass_logits")
            if torch.isnan(binary_logits).any():
                print("  ⚠️ NaN detected in binary_logits")
            if torch.isinf(binary_logits).any():
                print("  ⚠️ Inf detected in binary_logits")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


if __name__ == "__main__":
    success = test_model_forward_pass()
    if success:
        print("\n✅ モデル検証成功: 訓練実行の準備完了")
    else:
        print("\n❌ モデル検証失敗: 追加修正が必要")
