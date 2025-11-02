#!/usr/bin/env python3
"""
PyTorchモデルのレイヤー構造を詳しく解析
"""

import json
from pathlib import Path

import torch


def analyze_layer_structure():
    """
    一つのモデルファイルからレイヤー構造を詳しく解析
    """
    model_path = Path("public_datasets/imu-only-datas/saved_models/fold_1_model.pth")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    print("=== レイヤー構造詳細解析 ===")
    print(f"モデルファイル: {model_path.name}")
    print(f"チェックポイントキー: {list(checkpoint.keys())}")

    # model_configの内容を確認
    if "model_config" in checkpoint:
        print("\n【モデル設定情報】")
        config = checkpoint["model_config"]
        print(json.dumps(config, indent=2, ensure_ascii=False))

    # モデルの状態辞書を詳しく調べる
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("\n【レイヤー構造詳細】")
        print(f"総レイヤー数: {len(state_dict)}")

        # レイヤー名をグループ化して表示
        layer_groups = {}
        for name in state_dict.keys():
            parts = name.split(".")
            if len(parts) > 1:
                group = parts[0]
                if group not in layer_groups:
                    layer_groups[group] = []
                layer_groups[group].append(name)
            else:
                if "root" not in layer_groups:
                    layer_groups["root"] = []
                layer_groups["root"].append(name)

        for group, layers in layer_groups.items():
            print(f"\n  【{group}】 ({len(layers)}層)")
            for layer in layers[:10]:  # 最初の10層のみ表示
                tensor = state_dict[layer]
                print(f"    - {layer}: {list(tensor.shape)} ({tensor.numel():,} params)")
            if len(layers) > 10:
                print(f"    ... 他 {len(layers) - 10} 層")

    # 特徴的なレイヤーパターンを分析
    print("\n【アーキテクチャパターン分析】")
    layer_names = list(state_dict.keys())

    # Transformer/Attentionパターンの検出
    attention_patterns = [name for name in layer_names if "attn" in name.lower() or "attention" in name.lower()]
    if attention_patterns:
        print(f"  Attentionレイヤー検出: {len(attention_patterns)}個")
        for pattern in attention_patterns[:5]:
            print(f"    - {pattern}")

    # Convolutionパターンの検出
    conv_patterns = [name for name in layer_names if "conv" in name.lower()]
    if conv_patterns:
        print(f"  Convolutionレイヤー検出: {len(conv_patterns)}個")
        for pattern in conv_patterns:
            tensor = state_dict[pattern]
            print(f"    - {pattern}: {list(tensor.shape)}")

    # Normalizationパターンの検出
    norm_patterns = [name for name in layer_names if any(x in name.lower() for x in ["norm", "bn", "batch"])]
    if norm_patterns:
        print(f"  Normalizationレイヤー検出: {len(norm_patterns)}個")
        for pattern in norm_patterns[:5]:
            print(f"    - {pattern}")

    # Linearレイヤーの検出
    linear_patterns = [name for name in layer_names if "linear" in name.lower() or "fc" in name.lower()]
    if linear_patterns:
        print(f"  Linearレイヤー検出: {len(linear_patterns)}個")
        for pattern in linear_patterns:
            tensor = state_dict[pattern]
            if "weight" in pattern:
                print(f"    - {pattern}: {list(tensor.shape)} (入力: {tensor.shape[1]}, 出力: {tensor.shape[0]})")

    # 特殊なパターンの検出
    special_patterns = [
        name for name in layer_names if any(x in name.lower() for x in ["pos", "embed", "mask", "token"])
    ]
    if special_patterns:
        print(f"  特殊レイヤー検出: {len(special_patterns)}個")
        for pattern in special_patterns:
            tensor = state_dict[pattern]
            print(f"    - {pattern}: {list(tensor.shape)}")


if __name__ == "__main__":
    analyze_layer_structure()
