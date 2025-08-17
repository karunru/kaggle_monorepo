#!/usr/bin/env python3
"""NaN問題のデバッグスクリプト."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import Config
from dataset import IMUDataModule
from human_normalization import HNConfig, compute_hn_features


def debug_hn_features():
    """HN特徴量の数値安定性を確認."""
    print("=== HN特徴量の数値安定性デバッグ ===")

    # 設定読み込み
    config = Config()

    # データモジュール作成
    data_module = IMUDataModule(config, fold=0)
    data_module.setup("fit")

    # 最初のバッチを取得
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch type: {type(batch)}")
    if isinstance(batch, dict):
        print(f"Batch keys: {list(batch.keys())}")
    # バッチ構造は大きすぎるので表示しない

    # バッチの構造を確認（辞書形式）
    if isinstance(batch, dict) and "imu" in batch:
        x_batch = batch["imu"]
        print(f"X batch shape: {x_batch.shape}")
        print(f"X batch contains NaN: {torch.isnan(x_batch).any()}")
        print(f"X batch contains Inf: {torch.isinf(x_batch).any()}")

        # 特徴量別チェック
        feature_data = x_batch[0]  # 最初のサンプル [features, sequence_length]
        imu_cols = data_module.train_dataset.imu_cols

        print(f"\n=== 特徴量別の値確認 ({len(imu_cols)}個) ===")
        for i, col_name in enumerate(imu_cols):
            values = feature_data[i]
            min_val = values.min().item()
            max_val = values.max().item()
            nan_count = torch.isnan(values).sum().item()
            inf_count = torch.isinf(values).sum().item()

            print(f"{col_name:30}: min={min_val:12.6f}, max={max_val:12.6f}, nan={nan_count:3}, inf={inf_count:3}")

            # 異常値が検出された場合の詳細情報
            if nan_count > 0 or inf_count > 0 or abs(max_val) > 1e6 or abs(min_val) > 1e6:
                print(f"  *** ANOMALY DETECTED in {col_name} ***")
                if torch.isnan(values).any():
                    nan_indices = torch.where(torch.isnan(values))[0]
                    print(f"  NaN at indices: {nan_indices[:5].tolist()}")  # 最初の5個
                if torch.isinf(values).any():
                    inf_indices = torch.where(torch.isinf(values))[0]
                    print(f"  Inf at indices: {inf_indices[:5].tolist()}")  # 最初の5個

    else:
        print(f"Batch structure: {type(batch)}")
        if isinstance(batch, dict):
            print(f"Keys: {batch.keys()}")
        print("Expected dict with 'imu' key")


def debug_raw_hn_computation():
    """生の計算での数値確認."""
    print("\n=== 生HN計算での数値確認 ===")

    config = Config()

    # サンプルデータ作成
    sample_data = pl.DataFrame(
        {
            "subject": ["S001"] * 10,
            "sequence_id": [1] * 10,
            "linear_acc_x": [1.0, 2.0, 0.0, -1.0, 5.0, 0.1, -0.1, 10.0, -10.0, 0.001],
            "linear_acc_y": [0.0, 0.0, 1.0, 2.0, -1.0, 0.2, -0.2, 0.0, 0.0, 0.0],
            "linear_acc_z": [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0],
            "linear_acc_mag": [1.0, 2.0, 1.0, 2.236, 5.196, 0.245, 0.245, 10.0, 10.0, 0.001],
            "angular_vel_x": [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, -0.1, 0.0, 0.0, 0.0],
            "angular_vel_y": [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0],
            "angular_vel_z": [1.0, 2.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0],  # ゼロも含む
        }
    ).lazy()

    # Demographics データ
    demo_df = pl.DataFrame(
        {
            "subject": ["S001"],
            "height_cm": [170.0],
            "shoulder_to_wrist_cm": [60.0],
            "elbow_to_wrist_cm": [25.0],
        }
    )

    # HN設定
    hn_config = HNConfig(
        hn_enabled=True,
        hn_eps=config.demographics.hn_eps,
        hn_radius_min_max=config.demographics.hn_radius_min_max,
        hn_features=config.demographics.hn_features,
    )

    # HN特徴量計算
    result = compute_hn_features(sample_data, demo_df, hn_config).collect()

    print("計算結果:")
    print(f"Total rows: {result.shape[0]}")
    print(f"Total cols: {result.shape[1]}")

    # 各HN特徴量の統計確認
    for feature in hn_config.hn_features:
        if feature in result.columns:
            values = result[feature].to_numpy()
            print(f"\n{feature}:")
            print(f"  min: {np.min(values):.6f}")
            print(f"  max: {np.max(values):.6f}")
            print(f"  mean: {np.mean(values):.6f}")
            print(f"  nan_count: {np.isnan(values).sum()}")
            print(f"  inf_count: {np.isinf(values).sum()}")
            print(f"  finite_count: {np.isfinite(values).sum()}")

            # NaNやInfがある場合は詳細表示
            if np.isnan(values).any() or np.isinf(values).any():
                print(f"  Values: {values}")


def debug_intermediate_values():
    """中間計算値の確認."""
    print("\n=== 中間計算値の確認 ===")

    # 極端なケースでテスト
    sample_data = pl.DataFrame(
        {
            "subject": ["S001"] * 5,
            "sequence_id": [1] * 5,
            "linear_acc_x": [0.0, 1.0, 1000.0, -1000.0, 0.001],
            "linear_acc_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "linear_acc_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "linear_acc_mag": [0.0, 1.0, 1000.0, 1000.0, 0.001],
            "angular_vel_x": [0.0, 0.0, 0.0, 0.0, 0.0],
            "angular_vel_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "angular_vel_z": [0.0, 1.0, 0.0, 0.0, 0.0],  # omega=0のケースも
        }
    ).lazy()

    demo_df = pl.DataFrame(
        {
            "subject": ["S001"],
            "height_cm": [170.0],
            "shoulder_to_wrist_cm": [60.0],
            "elbow_to_wrist_cm": [25.0],
        }
    )

    from human_normalization import derive_hn_channels, join_subject_anthro

    # Step 1: Join anthropometrics
    with_anthro = join_subject_anthro(sample_data, demo_df).collect()
    print("After join_subject_anthro:")
    print(with_anthro)

    # Step 2: Compute intermediate values manually
    eps = 1e-6
    bounds = (0.15, 0.9)

    intermediate_result = derive_hn_channels(with_anthro.lazy(), eps, bounds).collect()

    print("\n中間計算値:")
    intermediate_cols = ["omega", "v_elbow", "v_shoulder", "a_c_elbow", "a_c_shoulder"]
    for col in intermediate_cols:
        if col in intermediate_result.columns:
            values = intermediate_result[col].to_numpy()
            print(f"{col}: {values}")


if __name__ == "__main__":
    try:
        debug_hn_features()
        debug_raw_hn_computation()
        debug_intermediate_values()

    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback

        traceback.print_exc()
