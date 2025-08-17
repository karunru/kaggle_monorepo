#!/usr/bin/env python3
"""モデル内部のNaN原因デバッグ."""

import sys
from pathlib import Path

import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import Config
from dataset import IMUDataModule
from model import CMISqueezeformer


def debug_demographics_processing():
    """Demographics処理のデバッグ."""
    print("=== Demographics処理デバッグ ===")

    config = Config()
    data_module = IMUDataModule(config, fold=0)
    data_module.setup("fit")

    # バッチ取得
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    demographics = batch.get("demographics", None)
    if demographics is None:
        print("Demographics data not found")
        return

    print("Demographics data:")
    for key, value in demographics.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(f"  min={value.min().item():.6f}, max={value.max().item():.6f}")
            print(f"  nan={torch.isnan(value).any()}, inf={torch.isinf(value).any()}")

            # NaNやInfが検出された場合の詳細
            if torch.isnan(value).any():
                print(f"  ⚠️ NaN detected in {key}")
                nan_indices = torch.where(torch.isnan(value))[0]
                print(f"  NaN indices: {nan_indices[:10].tolist()}")
            if torch.isinf(value).any():
                print(f"  ⚠️ Inf detected in {key}")


def debug_model_step_by_step():
    """モデルの各ステップでNaN発生箇所を特定."""
    print("\n=== モデルステップバイステップデバッグ ===")

    config = Config()
    data_module = IMUDataModule(config, fold=0)
    data_module.setup("fit")

    # モデル作成
    actual_input_dim = len(data_module.train_dataset.imu_cols)
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
    model.eval()

    # バッチ取得
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    imu_data = batch["imu"]
    attention_mask = batch.get("missing_mask", None)
    demographics = batch.get("demographics", None)

    print("Input shapes:")
    print(f"  IMU: {imu_data.shape}")
    print(f"  Mask: {attention_mask.shape if attention_mask is not None else None}")

    with torch.no_grad():
        # ステップ1: Input transpose
        x = imu_data.transpose(1, 2)
        print(f"\nStep 1 - Transpose: {x.shape}, nan={torch.isnan(x).any()}")

        # ステップ2: Input projection
        x = model.input_projection(x)
        print(f"Step 2 - Input projection: {x.shape}, nan={torch.isnan(x).any()}")

        # ステップ3: Positional encoding
        x = model.pos_encoding(x)
        print(f"Step 3 - Positional encoding: {x.shape}, nan={torch.isnan(x).any()}")

        # ステップ4: Demographics統合（ここが怪しい）
        if model.use_demographics and demographics is not None:
            print("\nDemographics integration...")
            try:
                demo_embeddings = model.demographics_processor(demographics)
                print(f"Demo embeddings: {demo_embeddings.shape}, nan={torch.isnan(demo_embeddings).any()}")

                if torch.isnan(demo_embeddings).any():
                    print("  ⚠️ NaN detected in demographics embeddings!")
                    # 各Demographics特徴量を個別チェック
                    for key, value in demographics.items():
                        if isinstance(value, torch.Tensor):
                            if key in model.demographics_processor.categorical_embedders:
                                emb = model.demographics_processor.categorical_embedders[key](value)
                                print(f"    {key} embedding: nan={torch.isnan(emb).any()}")

                # 統合
                x = model.demographics_integrator(x, demo_embeddings)
                print(f"After demo integration: {x.shape}, nan={torch.isnan(x).any()}")

            except Exception as e:
                print(f"Error in demographics processing: {e}")
                import traceback

                traceback.print_exc()

        # ステップ5: Transformer layers
        if not torch.isnan(x).any():
            print("\nTransformer processing...")
            for i, layer in enumerate(model.transformer_layers):
                try:
                    x = layer(x, src_key_padding_mask=attention_mask)
                    print(f"Layer {i}: {x.shape}, nan={torch.isnan(x).any()}")
                    if torch.isnan(x).any():
                        print(f"  ⚠️ NaN detected in layer {i}!")
                        break
                except Exception as e:
                    print(f"Error in layer {i}: {e}")
                    break


if __name__ == "__main__":
    debug_demographics_processing()
    debug_model_step_by_step()
