#!/usr/bin/env python3
"""
MiniRocketMultivariate動作確認・調査スクリプト

このスクリプトでは以下を検証する:
1. MiniRocketMultivariateの基本動作
2. 異なるnum_kernelsパラメータでの実行時間・メモリ使用量
3. Polarsデータフォーマットとの互換性
4. 実際のIMUデータ形状での動作確認
"""

import os
import sys
import time

import numpy as np
import psutil
import torch

# sktimeのインポート
try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    print("✅ sktime.transformations.panel.rocket.MiniRocketMultivariate imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MiniRocketMultivariate: {e}")
    print("Please install sktime: uv add scikit-time")
    sys.exit(1)

try:
    import pandas as pd

    print("✅ pandas imported successfully")
except ImportError:
    print("❌ pandas not available")
    sys.exit(1)


def get_memory_usage():
    """現在のメモリ使用量を取得（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def generate_sample_multivariate_time_series(
    n_samples: int = 100, n_features: int = 9, n_timepoints: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """
    サンプルの多変量時系列データを生成（pandas DataFrame形式）

    Args:
        n_samples: サンプル数
        n_features: 特徴量数（時系列の次元数）
        n_timepoints: 時系列の長さ
        random_state: 乱数シード

    Returns:
        pandas DataFrame: sktimeで使用可能な形式
    """
    np.random.seed(random_state)

    # MultiIndex DataFrame形式で作成（sktime標準形式）
    data = []

    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            # 各特徴量に対してsin波 + ノイズの時系列を生成
            t = np.linspace(0, 2 * np.pi, n_timepoints)
            frequency = np.random.uniform(0.5, 3.0)  # ランダムな周波数
            phase = np.random.uniform(0, 2 * np.pi)  # ランダムな位相
            amplitude = np.random.uniform(0.5, 2.0)  # ランダムな振幅

            # sin波にノイズを加える
            signal = amplitude * np.sin(frequency * t + phase) + np.random.normal(0, 0.1, n_timepoints)

            for time_idx, value in enumerate(signal):
                data.append(
                    {"sample": sample_idx, "feature": f"feature_{feature_idx}", "time": time_idx, "value": value}
                )

    df = pd.DataFrame(data)

    # sktimeで使用可能な形式にピボット（MultiIndex）
    df_pivot = df.pivot_table(index=["sample", "time"], columns="feature", values="value")

    return df_pivot


def test_basic_minirocket_functionality():
    """MiniRocketMultivariateの基本動作テスト"""
    print("=== 基本動作テスト開始 ===")

    # サンプルデータ生成
    print("サンプルデータ生成中...")
    df = generate_sample_multivariate_time_series(n_samples=50, n_features=9, n_timepoints=200, random_state=42)
    print(f"データ形状: {df.shape}")
    print(f"特徴量名: {list(df.columns)}")
    print(f"最初の数行:\n{df.head()}")

    # MiniRocketMultivariate初期化
    print("\nMiniRocketMultivariate初期化中...")
    transformer = MiniRocketMultivariate(
        num_kernels=84,  # 最小値でテスト
        n_jobs=1,  # シングルスレッド
        random_state=42,
    )

    # 学習
    print("变换器学習中...")
    memory_before_fit = get_memory_usage()
    start_time = time.time()

    transformer.fit(df)

    fit_time = time.time() - start_time
    memory_after_fit = get_memory_usage()

    print(f"Fit完了: {fit_time:.2f}秒")
    print(f"メモリ使用量 (fit前): {memory_before_fit:.1f}MB")
    print(f"メモリ使用量 (fit後): {memory_after_fit:.1f}MB")
    print(f"メモリ増加量: {memory_after_fit - memory_before_fit:.1f}MB")

    # 変換
    print("\n変換処理中...")
    memory_before_transform = get_memory_usage()
    start_time = time.time()

    X_transformed = transformer.transform(df)

    transform_time = time.time() - start_time
    memory_after_transform = get_memory_usage()

    print(f"Transform完了: {transform_time:.2f}秒")
    print(f"変換後データ形状: {X_transformed.shape}")
    print(f"メモリ使用量 (transform前): {memory_before_transform:.1f}MB")
    print(f"メモリ使用量 (transform後): {memory_after_transform:.1f}MB")
    print(f"メモリ増加量: {memory_after_transform - memory_before_transform:.1f}MB")

    # 結果の統計的確認
    print("\n変換結果統計:")
    if hasattr(X_transformed, "values"):
        # pandasの場合、.valuesでnumpy配列に変換
        values = X_transformed.values
    else:
        values = X_transformed

    print(f"平均: {np.mean(values):.4f}")
    print(f"標準偏差: {np.std(values):.4f}")
    print(f"最小値: {np.min(values):.4f}")
    print(f"最大値: {np.max(values):.4f}")
    print(f"NaN値数: {np.sum(np.isnan(values))}")

    return transformer, X_transformed


def test_different_num_kernels():
    """異なるnum_kernelsでのパフォーマンス比較"""
    print("\n=== num_kernelsパフォーマンス比較 ===")

    # テスト対象のnum_kernels（84の倍数）
    kernel_nums = [84, 168, 336, 672, 840]

    # サンプルデータ生成（少し大きめ）
    df = generate_sample_multivariate_time_series(n_samples=100, n_features=9, n_timepoints=200, random_state=42)

    results = []

    for num_kernels in kernel_nums:
        print(f"\n--- num_kernels = {num_kernels} ---")

        # メモリ初期化
        memory_start = get_memory_usage()

        # MiniRocketMultivariate初期化
        transformer = MiniRocketMultivariate(
            num_kernels=num_kernels,
            n_jobs=-1,  # 全CPUコア使用
            random_state=42,
        )

        # Fit時間測定
        start_time = time.time()
        transformer.fit(df)
        fit_time = time.time() - start_time

        memory_after_fit = get_memory_usage()

        # Transform時間測定
        start_time = time.time()
        X_transformed = transformer.transform(df)
        transform_time = time.time() - start_time

        memory_after_transform = get_memory_usage()

        # numpy配列に変換（pandas対応）
        if hasattr(X_transformed, "values"):
            values = X_transformed.values
        else:
            values = X_transformed

        # 結果記録
        result = {
            "num_kernels": num_kernels,
            "fit_time": fit_time,
            "transform_time": transform_time,
            "total_time": fit_time + transform_time,
            "memory_usage_mb": memory_after_transform - memory_start,
            "output_features": X_transformed.shape[1],
            "output_mean": np.mean(values),
            "output_std": np.std(values),
        }
        results.append(result)

        print(f"  Fit時間: {fit_time:.2f}秒")
        print(f"  Transform時間: {transform_time:.2f}秒")
        print(f"  合計時間: {result['total_time']:.2f}秒")
        print(f"  メモリ使用量: {result['memory_usage_mb']:.1f}MB")
        print(f"  出力特徴量数: {result['output_features']}")
        print(f"  出力統計: mean={result['output_mean']:.4f}, std={result['output_std']:.4f}")

    # 結果まとめ
    print("\n=== パフォーマンス比較結果 ===")
    print(
        f"{'num_kernels':<12} {'fit_time':<10} {'trans_time':<11} {'total_time':<11} {'memory_MB':<10} {'features':<9}"
    )
    print("-" * 70)

    for result in results:
        print(
            f"{result['num_kernels']:<12} "
            f"{result['fit_time']:<10.2f} "
            f"{result['transform_time']:<11.2f} "
            f"{result['total_time']:<11.2f} "
            f"{result['memory_usage_mb']:<10.1f} "
            f"{result['output_features']:<9}"
        )

    return results


def test_imu_realistic_data():
    """実際のIMUデータ形状での動作確認"""
    print("\n=== 実際のIMUデータ形状テスト ===")

    # 実際のexp013で使用される9つの特徴量名
    feature_names = [
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

    # よりリアルなIMUデータ生成
    print("リアルなIMUデータ生成中...")
    n_samples = 200
    n_timepoints = 200

    data = []
    np.random.seed(42)

    for sample_idx in range(n_samples):
        for feature_idx, feature_name in enumerate(feature_names):
            # IMU特有のシグナルを模擬
            t = np.linspace(0, 4 * np.pi, n_timepoints)  # 4秒相当

            if "acc" in feature_name:
                # 加速度データ: より高周波成分とジャーク
                base_signal = np.sin(2 * t) + 0.5 * np.sin(5 * t) + 0.2 * np.sin(10 * t)
                noise_level = 0.1
            elif "angular" in feature_name:
                # 角速度・角距離: より滑らかな変化
                base_signal = np.sin(0.5 * t) + 0.3 * np.sin(1.5 * t)
                noise_level = 0.05
            else:
                base_signal = np.sin(t)
                noise_level = 0.08

            # ノイズ追加
            signal = base_signal + np.random.normal(0, noise_level, n_timepoints)

            # たまに欠損値やスパイクを挿入（リアルなデータの模擬）
            if np.random.random() < 0.1:  # 10%の確率でスパイク
                spike_idx = np.random.randint(0, n_timepoints)
                signal[spike_idx] *= 5

            for time_idx, value in enumerate(signal):
                data.append({"sample": sample_idx, "feature": feature_name, "time": time_idx, "value": value})

    df = pd.DataFrame(data)

    # DataFrame形式変換
    df_pivot = df.pivot_table(index=["sample", "time"], columns="feature", values="value")

    print(f"リアルデータ形状: {df_pivot.shape}")
    print(f"特徴量: {list(df_pivot.columns)}")

    # 推奨パラメータでテスト
    recommended_num_kernels = 336  # 中間的な値

    print(f"\nMiniRocket変換テスト (num_kernels={recommended_num_kernels})...")
    transformer = MiniRocketMultivariate(num_kernels=recommended_num_kernels, n_jobs=-1, random_state=42)

    # 処理時間測定
    memory_start = get_memory_usage()
    start_time = time.time()

    transformer.fit(df_pivot)
    fit_time = time.time() - start_time

    memory_after_fit = get_memory_usage()

    start_time = time.time()
    X_transformed = transformer.transform(df_pivot)
    transform_time = time.time() - start_time

    memory_final = get_memory_usage()

    print("処理完了!")
    print(f"  Fit時間: {fit_time:.2f}秒")
    print(f"  Transform時間: {transform_time:.2f}秒")
    print(f"  変換後形状: {X_transformed.shape}")
    print(f"  メモリ使用量: {memory_final - memory_start:.1f}MB")

    # 実際のtorch tensorとの統合をテスト
    print("\nPyTorch Tensor統合テスト...")
    if hasattr(X_transformed, "values"):
        tensor_data = X_transformed.values
    else:
        tensor_data = X_transformed
    X_tensor = torch.tensor(tensor_data, dtype=torch.float32)
    print(f"  PyTorch tensor形状: {X_tensor.shape}")
    print(f"  テンソル統計: mean={X_tensor.mean().item():.4f}, std={X_tensor.std().item():.4f}")

    # 元の時系列との統合をシミュレーション
    print("\n統合シミュレーション...")
    batch_size = X_transformed.shape[0]
    original_seq_len = 200
    rocket_features = X_tensor  # [batch, num_kernels]

    # 元の時系列データを模擬（実際のexp014ではデータセットから取得）
    original_timeseries = torch.randn(batch_size, 9, original_seq_len)  # [batch, features, seq_len]

    print(f"  元時系列形状: {original_timeseries.shape}")
    print(f"  Rocket特徴量形状: {rocket_features.shape}")

    # 統合方法のオプション検討
    print("\n統合方法オプション:")

    # Option 1: Rocket特徴量を時系列次元に拡張
    # rocket_features: [batch, num_kernels] -> [batch, num_kernels, seq_len]
    rocket_expanded = rocket_features.unsqueeze(-1).expand(-1, -1, original_seq_len)
    print(f"  Rocket特徴量拡張後: {rocket_expanded.shape}")

    combined_option1 = torch.cat([original_timeseries, rocket_expanded], dim=1)
    print(f"  Option 1 - 時系列追加: {combined_option1.shape} [batch, orig_features + rocket_features, seq_len]")

    # Option 2: 別々に処理（後でfusion layer）
    print("  Option 2 - 別々処理:")
    print(f"    - 時系列部分: {original_timeseries.shape}")
    print(f"    - Rocket部分: {rocket_features.shape}")

    return (
        transformer,
        X_transformed,
        {
            "original_shape": original_timeseries.shape,
            "rocket_shape": rocket_features.shape,
            "combined_shape": combined_option1.shape,
        },
    )


def main():
    """メイン実行関数"""
    print("MiniRocketMultivariate調査スクリプト開始")
    print(f"Python プロセスID: {os.getpid()}")
    print(f"使用可能CPUコア数: {os.cpu_count()}")
    print(f"初期メモリ使用量: {get_memory_usage():.1f}MB")

    try:
        # 基本動作テスト
        transformer, X_basic = test_basic_minirocket_functionality()

        # パフォーマンス比較
        perf_results = test_different_num_kernels()

        # リアルなIMUデータでのテスト
        imu_transformer, X_imu, integration_info = test_imu_realistic_data()

        # 総括
        print("\n" + "=" * 50)
        print("調査完了総括")
        print("=" * 50)

        print("\n✅ MiniRocketMultivariateは正常に動作")
        print("✅ 推奨パラメータ: num_kernels=336 (処理時間とメモリのバランス)")
        print("✅ PyTorchテンソルとの統合が可能")
        print("✅ 実装準備完了")

        # 推奨設定
        print("\nexp014実装への推奨設定:")
        print("  - num_kernels: 336")
        print("  - n_jobs: -1 (全CPUコア使用)")
        print("  - random_state: 42 (再現性確保)")
        print("  - 統合方法: Option 1 (時系列次元に追加)")

        # 統合後の最終特徴量数
        original_features = 9
        rocket_features = 336
        total_features = original_features + rocket_features
        print(f"  - 最終特徴量数: {original_features} + {rocket_features} = {total_features}")

        return True

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 調査完了! exp014実装に進む準備ができました。")
    else:
        print("\n💥 調査中にエラーが発生しました。")
        sys.exit(1)
