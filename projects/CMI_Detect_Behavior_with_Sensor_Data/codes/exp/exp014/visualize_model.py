#!/usr/bin/env python3
"""
exp014 モデル構造可視化スクリプト

torchvizを使用してCMISqueezeformerモデルの構造を可視化する。
MiniRocket統合後のモデル構造を確認することが目的。
"""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from config import Config
from model import CMISqueezeformer

try:
    from torchviz import make_dot

    TORCHVIZ_AVAILABLE = True
except ImportError:
    print("❌ torchviz is not available. Please install: uv add torchviz")
    TORCHVIZ_AVAILABLE = False
    sys.exit(1)


def visualize_model_architecture(config: Config, output_dir: Path, include_demographics: bool = True) -> None:
    """
    モデル構造を可視化してファイルに保存.

    Args:
        config: 設定オブジェクト
        output_dir: 出力ディレクトリ
        include_demographics: Demographics特徴量を含めるか
    """
    print("Creating model for visualization...")

    # 実効的な入力次元数を取得
    effective_input_dim = config.get_effective_input_dim()
    print(f"Effective input dimension: {effective_input_dim}")
    print(f"  Base IMU features: {config.model.base_imu_features}")
    if config.rocket.enabled:
        print(f"  MiniRocket features: {config.rocket.num_kernels}")

    # モデル作成
    model = CMISqueezeformer(
        input_dim=effective_input_dim,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        num_classes=config.model.num_classes,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        demographics_config=config.demographics.model_dump() if include_demographics else None,
        target_gestures=config.target_gestures,
        non_target_gestures=config.non_target_gestures,
    )

    # モデルを評価モードに設定
    model.eval()

    # ダミーデータ作成
    batch_size = 2
    seq_len = config.preprocessing.target_sequence_length

    print(f"Creating dummy input: batch_size={batch_size}, input_dim={effective_input_dim}, seq_len={seq_len}")

    # IMUデータ（メインの入力）
    dummy_imu = torch.randn(batch_size, effective_input_dim, seq_len, requires_grad=True)

    # Demographics特徴量（オプション）
    dummy_demographics = None
    if include_demographics and config.demographics.enabled:
        dummy_demographics = {
            # カテゴリカル特徴量
            "adult_child": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            "sex": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            "handedness": torch.randint(0, 2, (batch_size,), dtype=torch.long),
            # 数値特徴量
            "age": torch.rand(batch_size, dtype=torch.float32) * 50 + 10,  # 10-60歳
            "height_cm": torch.rand(batch_size, dtype=torch.float32) * 60 + 140,  # 140-200cm
            "shoulder_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 30 + 40,  # 40-70cm
            "elbow_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 25 + 20,  # 20-45cm
        }
        print("Including demographics features in visualization")

    print("Running forward pass...")

    # 前向き計算実行
    with torch.no_grad():
        multiclass_logits, binary_logits = model(dummy_imu, demographics=dummy_demographics)

    print("Forward pass completed:")
    print(f"  Multiclass output: {multiclass_logits.shape}")
    print(f"  Binary output: {binary_logits.shape}")

    # 可視化用に勾配を有効にして再実行
    multiclass_logits, binary_logits = model(dummy_imu, demographics=dummy_demographics)

    # マルチクラス分類の可視化
    print("Generating multiclass classification graph...")
    multiclass_dot = make_dot(
        multiclass_logits, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True
    )

    # バイナリ分類の可視化
    print("Generating binary classification graph...")
    binary_dot = make_dot(binary_logits, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)

    # ファイル保存
    output_dir.mkdir(parents=True, exist_ok=True)

    # マルチクラス分類グラフ
    multiclass_path = output_dir / "model_architecture_multiclass"
    multiclass_dot.render(multiclass_path, format="png", cleanup=True)
    print(f"✅ Multiclass model architecture saved to: {multiclass_path}.png")

    # バイナリ分類グラフ
    binary_path = output_dir / "model_architecture_binary"
    binary_dot.render(binary_path, format="png", cleanup=True)
    print(f"✅ Binary model architecture saved to: {binary_path}.png")

    # モデル統計情報
    stats_path = output_dir / "model_statistics.txt"
    with open(stats_path, "w") as f:
        f.write("EXP014 Model Architecture Statistics\n")
        f.write("=" * 50 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Experiment: {config.experiment.name}\n")
        f.write(f"  Input dimension: {effective_input_dim}\n")
        f.write(f"  Model dimension: {config.model.d_model}\n")
        f.write(f"  Number of layers: {config.model.n_layers}\n")
        f.write(f"  Number of heads: {config.model.n_heads}\n")
        f.write(f"  Feed-forward dimension: {config.model.d_ff}\n")
        f.write(f"  Number of classes: {config.model.num_classes}\n")
        f.write(f"  Dropout rate: {config.model.dropout}\n")
        f.write(f"  Demographics enabled: {config.demographics.enabled}\n")
        f.write(f"  MiniRocket enabled: {config.rocket.enabled}\n")
        if config.rocket.enabled:
            f.write(f"  MiniRocket kernels: {config.rocket.num_kernels}\n")
        f.write("\n")

        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write("Parameter Statistics:\n")
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Trainable parameters: {trainable_params:,}\n")
        f.write(f"  Non-trainable parameters: {total_params - trainable_params:,}\n")
        f.write("\n")

        # 入力・出力形状
        f.write("Input/Output Shapes:\n")
        f.write(f"  Input shape: {dummy_imu.shape}\n")
        f.write(f"  Multiclass output shape: {multiclass_logits.shape}\n")
        f.write(f"  Binary output shape: {binary_logits.shape}\n")

        if dummy_demographics:
            f.write("  Demographics features:\n")
            for key, value in dummy_demographics.items():
                f.write(f"    {key}: {value.shape} ({value.dtype})\n")
        f.write("\n")

        # 特徴量内訳
        f.write("Feature Breakdown:\n")
        f.write(f"  Base IMU features: {config.model.base_imu_features}\n")
        if config.rocket.enabled:
            f.write(f"  MiniRocket features: {config.rocket.num_kernels}\n")
            f.write(f"  MiniRocket target time series: {len(config.rocket.target_features)}\n")
            for i, feature in enumerate(config.rocket.target_features):
                f.write(f"    {i + 1}. {feature}\n")
        f.write(f"  Total input features: {effective_input_dim}\n")

        if config.demographics.enabled:
            f.write(f"  Demographics embedding dimension: {config.demographics.embedding_dim}\n")

    print(f"✅ Model statistics saved to: {stats_path}")


def main():
    """メイン実行関数."""
    print("EXP014 Model Architecture Visualization")
    print("=" * 50)

    if not TORCHVIZ_AVAILABLE:
        print("❌ torchviz is required for visualization")
        return False

    try:
        # 設定読み込み
        print("Loading configuration...")
        config = Config()

        # 出力ディレクトリ
        output_dir = Path("./model_visualizations")

        # モデル可視化実行
        visualize_model_architecture(config=config, output_dir=output_dir, include_demographics=True)

        print("\n🎉 Model visualization completed successfully!")
        print(f"📁 Output directory: {output_dir.absolute()}")

        return True

    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
