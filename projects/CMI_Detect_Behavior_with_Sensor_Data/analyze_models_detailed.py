#!/usr/bin/env python3
"""
PyTorchモデルファイル詳細解析スクリプト
model_state_dictの内容も詳しく解析
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 警告を抑制
warnings.filterwarnings("ignore")


def analyze_pytorch_checkpoint_detailed(model_path: Path) -> dict[str, Any]:
    """
    PyTorchチェックポイントファイルの詳細解析
    """
    print(f"詳細分析中: {model_path}")

    try:
        # 安全でないグローバルの確認
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(model_path))

        # チェックポイントの基本情報を取得
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

        result = {
            "file_name": model_path.name,
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "unsafe_globals": list(unsafe_globals) if unsafe_globals else [],
            "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict",
            "success": True,
            "error": None,
        }

        # チェックポイントの詳細情報を取得
        if isinstance(checkpoint, dict):
            # 各キーの詳細情報
            for key in checkpoint.keys():
                if key == "model_state_dict" and isinstance(checkpoint[key], dict):
                    result["model_info"] = analyze_model_state_dict(checkpoint[key])
                elif key == "model_config":
                    result["model_config"] = checkpoint[key]
                elif key in ["epoch", "val_loss", "val_acc", "val_score"]:
                    result[key] = checkpoint[key]
                elif key == "optimizer_state_dict":
                    result["optimizer_info"] = {
                        "keys": list(checkpoint[key].keys()) if isinstance(checkpoint[key], dict) else "Not a dict"
                    }

        return result

    except Exception as e:
        return {"file_name": model_path.name, "success": False, "error": str(e)}


def analyze_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    モデルのstate_dictを詳しく解析してアーキテクチャ情報を抽出
    """
    layer_info = []
    total_params = 0
    trainable_params = 0

    # レイヤーの分類
    conv_layers = []
    linear_layers = []
    norm_layers = []
    attention_layers = []
    embedding_layers = []
    other_layers = []

    for name, tensor in state_dict.items():
        param_count = tensor.numel()
        total_params += param_count

        is_trainable = tensor.requires_grad if hasattr(tensor, "requires_grad") else True
        if is_trainable:
            trainable_params += param_count

        layer_detail = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "param_count": param_count,
            "requires_grad": is_trainable,
            "device": str(tensor.device) if hasattr(tensor, "device") else "Unknown",
        }

        layer_info.append(layer_detail)

        # レイヤータイプの分類
        name_lower = name.lower()
        if "conv" in name_lower:
            conv_layers.append(layer_detail)
        elif "linear" in name_lower or "fc" in name_lower:
            linear_layers.append(layer_detail)
        elif "norm" in name_lower or "bn" in name_lower or "batch_norm" in name_lower or "layer_norm" in name_lower:
            norm_layers.append(layer_detail)
        elif "attention" in name_lower or "attn" in name_lower:
            attention_layers.append(layer_detail)
        elif "embedding" in name_lower or "embed" in name_lower:
            embedding_layers.append(layer_detail)
        else:
            other_layers.append(layer_detail)

    # アーキテクチャの推定
    architecture_info = estimate_detailed_architecture(
        layer_info, conv_layers, linear_layers, norm_layers, attention_layers, embedding_layers
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "num_layers": len(layer_info),
        "layer_categories": {
            "convolutional": len(conv_layers),
            "linear": len(linear_layers),
            "normalization": len(norm_layers),
            "attention": len(attention_layers),
            "embedding": len(embedding_layers),
            "other": len(other_layers),
        },
        "layer_details": layer_info,
        "architecture_analysis": architecture_info,
        "conv_layers": conv_layers,
        "linear_layers": linear_layers,
        "norm_layers": norm_layers,
        "attention_layers": attention_layers,
        "embedding_layers": embedding_layers,
    }


def estimate_detailed_architecture(
    layer_info, conv_layers, linear_layers, norm_layers, attention_layers, embedding_layers
):
    """
    詳細なアーキテクチャ分析
    """
    layer_names = [layer["name"] for layer in layer_info]

    analysis = {
        "primary_type": "Unknown",
        "sub_components": [],
        "input_size_estimate": None,
        "output_size_estimate": None,
        "depth_estimate": 0,
        "special_features": [],
    }

    # 主要なアーキテクチャタイプの判定
    if attention_layers:
        analysis["primary_type"] = "Attention-based (Transformer-like)"
        analysis["sub_components"].append("Self-Attention")
    elif conv_layers and linear_layers:
        analysis["primary_type"] = "Hybrid CNN-MLP"
        analysis["sub_components"].extend(["Convolutional", "Fully Connected"])
    elif conv_layers:
        analysis["primary_type"] = "Convolutional Neural Network"
        analysis["sub_components"].append("Convolutional")
    elif linear_layers:
        analysis["primary_type"] = "Fully Connected Network"
        analysis["sub_components"].append("Fully Connected")

    # 特殊機能の検出
    layer_names_str = " ".join(layer_names).lower()
    if "dropout" in layer_names_str:
        analysis["special_features"].append("Dropout Regularization")
    if "batch" in layer_names_str or "layer_norm" in layer_names_str:
        analysis["special_features"].append("Normalization")
    if "residual" in layer_names_str or "skip" in layer_names_str:
        analysis["special_features"].append("Residual Connections")
    if "positional" in layer_names_str or "pos" in layer_names_str:
        analysis["special_features"].append("Positional Encoding")

    # 深さの推定
    unique_layer_prefixes = set()
    for name in layer_names:
        parts = name.split(".")
        if len(parts) > 1:
            unique_layer_prefixes.add(parts[0])
    analysis["depth_estimate"] = len(unique_layer_prefixes)

    # 入出力サイズの推定
    if linear_layers:
        # 最初と最後のlinearレイヤーから推定
        first_linear = min(linear_layers, key=lambda x: x["name"])
        last_linear = max(linear_layers, key=lambda x: x["name"])

        if "weight" in first_linear["name"] and len(first_linear["shape"]) == 2:
            analysis["input_size_estimate"] = first_linear["shape"][1]
        if "weight" in last_linear["name"] and len(last_linear["shape"]) == 2:
            analysis["output_size_estimate"] = last_linear["shape"][0]

    return analysis


def print_detailed_results(results):
    """
    詳細結果の表示
    """
    print("\n" + "=" * 80)
    print("詳細解析結果")
    print("=" * 80)

    # PyTorchモデル詳細
    if results["pytorch_models"]:
        print("\n【PyTorchモデル解析】")
        for i, model in enumerate(results["pytorch_models"], 1):
            print(f"\n{i}. {model['file_name']}")
            print(f"   ファイルサイズ: {model['file_size_mb']:.2f} MB")
            print(f"   安全でないグローバル: {model['unsafe_globals']}")

            if "model_info" in model:
                info = model["model_info"]
                arch = info["architecture_analysis"]

                print(f"   総パラメータ数: {info['total_parameters']:,}")
                print(f"   学習可能パラメータ数: {info['trainable_parameters']:,}")
                print(f"   アーキテクチャタイプ: {arch['primary_type']}")
                print(f"   推定深さ: {arch['depth_estimate']}")
                print(f"   特殊機能: {', '.join(arch['special_features']) if arch['special_features'] else 'なし'}")

                if arch["input_size_estimate"]:
                    print(f"   推定入力サイズ: {arch['input_size_estimate']}")
                if arch["output_size_estimate"]:
                    print(f"   推定出力サイズ: {arch['output_size_estimate']}")

                print("   レイヤー構成:")
                for layer_type, count in info["layer_categories"].items():
                    if count > 0:
                        print(f"     - {layer_type}: {count}層")

            if "epoch" in model:
                print(f"   学習エポック数: {model['epoch']}")
            if "val_acc" in model:
                print(f"   検証精度: {model['val_acc']:.4f}")
            if "val_score" in model:
                print(f"   検証スコア: {model['val_score']:.4f}")

    # 特徴量スケーラー詳細
    if results["feature_scaler"] and results["feature_scaler"]["success"]:
        scaler = results["feature_scaler"]
        print("\n【特徴量スケーラー解析】")
        print(f"   スケーラータイプ: {scaler['scaler_type']}")
        print(f"   特徴量数: {scaler['n_features']}")

        if "mean_" in scaler:
            mean_array = np.array(scaler["mean_"])
            print(
                f"   平均値統計: min={mean_array.min():.4f}, max={mean_array.max():.4f}, mean={mean_array.mean():.4f}"
            )

        if "scale_" in scaler:
            scale_array = np.array(scaler["scale_"])
            print(
                f"   スケール統計: min={scale_array.min():.4f}, max={scale_array.max():.4f}, mean={scale_array.mean():.4f}"
            )

    # K-fold結果詳細
    if results["kfold_results"] and results["kfold_results"]["success"]:
        kfold = results["kfold_results"]["content"]
        print("\n【K-fold結果解析】")
        print(f"   フォールド数: {kfold['n_folds']}")
        print(f"   平均検証精度: {kfold['summary']['val_acc_mean']:.4f} ± {kfold['summary']['val_acc_std']:.4f}")
        print(f"   平均検証スコア: {kfold['summary']['val_score_mean']:.4f} ± {kfold['summary']['val_score_std']:.4f}")
        print(f"   平均テスト精度: {kfold['summary']['test_acc_mean']:.4f} ± {kfold['summary']['test_acc_std']:.4f}")
        print(
            f"   平均テストスコア: {kfold['summary']['test_score_mean']:.4f} ± {kfold['summary']['test_score_std']:.4f}"
        )

    # サブミッション詳細
    if results["submission"] and results["submission"]["success"]:
        sub = results["submission"]
        print("\n【サブミッションファイル解析】")
        print(f"   データ形状: {sub['shape'][0]} 行 × {sub['shape'][1]} 列")
        print(f"   メモリ使用量: {sub['memory_usage_mb']:.4f} MB")
        print(f"   予測されたジェスチャー: {list(sub['gesture_stats']['most_common'].keys())}")


def main():
    """
    メイン詳細解析処理
    """
    base_path = Path("public_datasets/imu-only-datas")
    saved_models_path = base_path / "saved_models"

    results = {"pytorch_models": [], "feature_scaler": None, "kfold_results": None, "submission": None}

    print("=== PyTorchモデルファイル詳細解析開始 ===")

    # 1. PyTorchモデルファイルの詳細解析
    print("\n1. PyTorchモデルファイルの詳細解析")
    model_files = list(saved_models_path.glob("fold_*_model.pth"))

    for model_file in sorted(model_files):
        result = analyze_pytorch_checkpoint_detailed(model_file)
        results["pytorch_models"].append(result)
        print(f"  - {result['file_name']}: {'成功' if result['success'] else '失敗'}")
        if not result["success"]:
            print(f"    エラー: {result['error']}")

    # 以前の関数を再利用
    from analyze_models import analyze_feature_scaler, analyze_kfold_results, analyze_submission_parquet

    # 2. 特徴量スケーラーの解析
    print("\n2. 特徴量スケーラー解析")
    scaler_file = saved_models_path / "feature_scaler.joblib"
    if scaler_file.exists():
        results["feature_scaler"] = analyze_feature_scaler(scaler_file)

    # 3. K-fold結果の解析
    print("\n3. K-fold結果解析")
    kfold_file = saved_models_path / "kfold_results.json"
    if kfold_file.exists():
        results["kfold_results"] = analyze_kfold_results(kfold_file)

    # 4. サブミッションファイルの解析
    print("\n4. サブミッションファイル解析")
    submission_file = base_path / "submission.parquet"
    if submission_file.exists():
        results["submission"] = analyze_submission_parquet(submission_file)

    # 詳細結果の表示
    print_detailed_results(results)

    # 結果を保存
    output_path = Path("outputs/claude/model_analysis_detailed_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n詳細解析結果を {output_path} に保存しました")

    return results


if __name__ == "__main__":
    results = main()
