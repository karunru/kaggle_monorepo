#!/usr/bin/env python3
"""
MiniRocketMultivariate動作確認スクリプト

このスクリプトは以下を確認します：
1. MiniRocketMultivariateの基本動作
2. 9つの物理特徴量を使った変換
3. polarsとsktimeのデータ変換
4. パラメータの動作確認
"""

import sys
from pathlib import Path

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import polars as pl
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.datasets import load_basic_motions

print("=== MiniRocketMultivariate 動作確認スクリプト ===\n")

# 1. 基本動作確認
print("1. 基本動作確認")
print("データセットの読み込み中...")
try:
    X_train, y_train = load_basic_motions(split="train")
    print(f"データ形状: {X_train.shape}")
    print(f"データ型: {type(X_train)}")
    print(f"最初のサンプル形状: {X_train.iloc[0, 0].shape if hasattr(X_train.iloc[0, 0], 'shape') else 'N/A'}")
    print("✓ データセット読み込み成功\n")
except Exception as e:
    print(f"✗ データセット読み込みエラー: {e}\n")
    sys.exit(1)

# 2. MiniRocketMultivariateeの基本動作
print("2. MiniRocketMultivariateの基本動作確認")
try:
    # 小さなnum_kernelsでテスト
    trf = MiniRocketMultivariate(
        num_kernels=168,  # 最小値84の倍数
        n_jobs=1,
        random_state=42
    )
    print(f"Transformer作成完了: {trf}")
    
    # 学習
    print("学習中...")
    trf.fit(X_train[:10])  # 最初の10サンプルのみで確認
    print("✓ 学習完了")
    
    # 変換
    print("変換中...")
    X_transformed = trf.transform(X_train[:10])
    print(f"変換後の形状: {X_transformed.shape}")
    print(f"変換後のデータ型: {type(X_transformed)}")
    print("✓ 基本動作確認完了\n")
    
except Exception as e:
    print(f"✗ 基本動作エラー: {e}\n")
    import traceback
    traceback.print_exc()
    print()

# 3. 9つの物理特徴量に相当するデータでの確認
print("3. CMI特徴量模擬データでの確認")

# 物理ベース特徴量の定義（exp012/dataset.pyから抽出）
physics_features = [
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

print(f"対象特徴量: {physics_features}")
print(f"特徴量数: {len(physics_features)}")

# 模擬データ生成（多変量時系列）
def create_mock_multivariate_data(n_samples=5, n_features=9, seq_length=200):
    """CMIデータに似た模擬多変量時系列データを生成"""
    data_list = []
    
    for i in range(n_samples):
        # 各特徴量ごとにシーケンスを生成
        sample_data = np.random.randn(seq_length, n_features).astype(np.float32)
        
        # 物理的な特徴量らしい傾向を追加
        # 線形加速度系(0-2): より大きな変動
        sample_data[:, 0:3] *= 2.0
        
        # 角速度系(5-7): 中程度の変動
        sample_data[:, 5:8] *= 1.0
        
        # 大きさ系(3,4,8): 常に正の値
        sample_data[:, 3] = np.abs(sample_data[:, 3])  # linear_acc_mag
        sample_data[:, 8] = np.abs(sample_data[:, 8])  # angular_distance
        
        # ジャーク(4): より小さな変動
        sample_data[:, 4] *= 0.5
        
        data_list.append(sample_data)
    
    return data_list

try:
    print("模擬データ生成中...")
    mock_data_list = create_mock_multivariate_data(n_samples=10, n_features=len(physics_features), seq_length=200)
    print(f"模擬データ形状: {len(mock_data_list)} samples, {mock_data_list[0].shape} each")
    
    # pandasのMultiIndex形式に変換（sktimeのPanel形式）
    print("sktime Panel形式への変換中...")
    
    # MultiIndex DataFrameの作成
    data_frames = []
    for i, data in enumerate(mock_data_list):
        # 時間インデックス
        time_index = range(len(data))
        
        # 各特徴量を列として持つDataFrame
        df = pd.DataFrame(data, index=time_index, columns=physics_features)
        df.index.name = 'time'
        
        # サンプルIDを追加
        df_with_id = df.copy()
        df_with_id['sample_id'] = i
        df_with_id = df_with_id.set_index('sample_id', append=True)
        df_with_id = df_with_id.reorder_levels(['sample_id', 'time'])
        
        data_frames.append(df_with_id)
    
    # MultiIndex DataFrameに結合
    panel_data = pd.concat(data_frames)
    print(f"Panel データ形状: {panel_data.shape}")
    print(f"Panel インデックス: {panel_data.index.names}")
    print("✓ Panel形式変換完了")
    
    # MiniRocketMultivariateで変換
    print("MiniRocketMultivariate変換中...")
    trf_physics = MiniRocketMultivariate(
        num_kernels=168,  # 小さな値でテスト
        n_jobs=1,
        random_state=42
    )
    
    # Panel形式データでの学習・変換
    trf_physics.fit(panel_data)
    X_physics_transformed = trf_physics.transform(panel_data)
    print(f"物理特徴量変換後の形状: {X_physics_transformed.shape}")
    print("✓ 物理特徴量変換完了\n")
    
except Exception as e:
    print(f"✗ 物理特徴量変換エラー: {e}")
    import traceback
    traceback.print_exc()
    print()

# 4. パラメータの調査
print("4. パラメータ調査")
print("推奨パラメータ:")
print("- num_kernels: 10000 (デフォルト) - 84の倍数である必要があります")
print("- n_jobs: -1 (全CPU使用)")
print("- random_state: 固定値 (再現性のため)")
print()

print("CMIプロジェクト用の推奨設定:")
print("- num_kernels: 1000-5000程度 (計算時間とのバランス)")
print("- n_jobs: -1")
print("- random_state: 42")
print()

# 5. 計算時間の確認
print("5. 計算時間確認")
import time

try:
    start_time = time.time()
    
    # より大きなnum_kernelsでテスト
    trf_large = MiniRocketMultivariate(
        num_kernels=840,  # 84 * 10
        n_jobs=-1,
        random_state=42
    )
    
    # 小さなデータセットで時間測定
    fit_start = time.time()
    trf_large.fit(panel_data)
    fit_time = time.time() - fit_start
    
    transform_start = time.time()
    X_large_transformed = trf_large.transform(panel_data)
    transform_time = time.time() - transform_start
    
    total_time = time.time() - start_time
    
    print(f"学習時間: {fit_time:.3f}秒")
    print(f"変換時間: {transform_time:.3f}秒")
    print(f"総時間: {total_time:.3f}秒")
    print(f"変換後の形状: {X_large_transformed.shape}")
    print("✓ 計算時間確認完了\n")
    
except Exception as e:
    print(f"✗ 計算時間確認エラー: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=== 動作確認完了 ===")
print()
print("次のステップ:")
print("1. exp013のdatasetクラスにMiniRocket変換を組み込む")
print("2. 変換された特徴量を既存のIMU特徴量と結合する")
print("3. モデルの入力次元を調整する")
print("4. 学習・推論パイプラインをテストする")