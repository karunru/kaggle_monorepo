#!/usr/bin/env python3
"""
PyTorchモデルファイル解析スクリプト
"""

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

# 警告を抑制
warnings.filterwarnings("ignore")


def analyze_pytorch_checkpoint_safety(model_path: Path) -> dict[str, Any]:
    """
    PyTorchチェックポイントファイルの安全性を解析
    """
    print(f"分析中: {model_path}")

    try:
        # 安全でないグローバルの確認
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(model_path))

        # チェックポイントの基本情報を取得 (weights_only=Falseを指定して古い形式に対応)
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

        result = {
            "file_name": model_path.name,
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "unsafe_globals": list(unsafe_globals) if unsafe_globals else [],
            "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict",
            "has_state_dict": "state_dict" in checkpoint if isinstance(checkpoint, dict) else False,
            "success": True,
            "error": None,
        }

        # モデルの状態辞書が存在する場合は詳細情報を取得
        if result["has_state_dict"]:
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            result["model_info"] = analyze_model_state_dict(state_dict)
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # 直接state_dictの場合
            result["model_info"] = analyze_model_state_dict(checkpoint)

        return result

    except Exception as e:
        return {"file_name": model_path.name, "success": False, "error": str(e)}


def analyze_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    モデルのstate_dictを解析してアーキテクチャ情報を抽出
    """
    layer_info = []
    total_params = 0

    for name, tensor in state_dict.items():
        param_count = tensor.numel()
        total_params += param_count

        layer_info.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "param_count": param_count,
                "requires_grad": tensor.requires_grad if hasattr(tensor, "requires_grad") else "Unknown",
            }
        )

    # レイヤータイプの推定
    layer_types = set()
    for layer in layer_info:
        name = layer["name"]
        if "conv" in name.lower():
            layer_types.add("Convolutional")
        elif "linear" in name.lower() or "fc" in name.lower():
            layer_types.add("Linear")
        elif "bn" in name.lower() or "batch_norm" in name.lower():
            layer_types.add("BatchNorm")
        elif "dropout" in name.lower():
            layer_types.add("Dropout")
        elif "attention" in name.lower() or "attn" in name.lower():
            layer_types.add("Attention")

    return {
        "total_parameters": total_params,
        "num_layers": len(layer_info),
        "layer_types": list(layer_types),
        "layer_details": layer_info,
        "estimated_architecture": estimate_architecture(layer_info),
    }


def estimate_architecture(layer_info: list[dict]) -> str:
    """
    レイヤー情報からアーキテクチャを推定
    """
    layer_names = [layer["name"] for layer in layer_info]

    # 一般的なアーキテクチャパターンの検出
    if any("transformer" in name.lower() for name in layer_names):
        return "Transformer-based"
    elif any("conv" in name.lower() for name in layer_names):
        if any("attention" in name.lower() for name in layer_names):
            return "ConvNet with Attention"
        else:
            return "Convolutional Neural Network"
    elif any("lstm" in name.lower() or "gru" in name.lower() for name in layer_names):
        return "Recurrent Neural Network"
    elif all(
        "linear" in name.lower() or "fc" in name.lower() or "bias" in name.lower() or "weight" in name.lower()
        for name in layer_names
    ):
        return "Fully Connected Network"
    else:
        return "Unknown/Mixed Architecture"


def analyze_feature_scaler(scaler_path: Path) -> dict[str, Any]:
    """
    特徴量スケーラーファイルを解析
    """
    try:
        scaler = joblib.load(str(scaler_path))

        result = {"file_name": scaler_path.name, "scaler_type": type(scaler).__name__, "success": True, "error": None}

        # スケーラーの詳細情報を取得
        if hasattr(scaler, "scale_"):
            result["scale_"] = scaler.scale_.tolist() if hasattr(scaler.scale_, "tolist") else str(scaler.scale_)
        if hasattr(scaler, "mean_"):
            result["mean_"] = scaler.mean_.tolist() if hasattr(scaler.mean_, "tolist") else str(scaler.mean_)
        if hasattr(scaler, "var_"):
            result["var_"] = scaler.var_.tolist() if hasattr(scaler.var_, "tolist") else str(scaler.var_)
        if hasattr(scaler, "min_"):
            result["min_"] = scaler.min_.tolist() if hasattr(scaler.min_, "tolist") else str(scaler.min_)
        if hasattr(scaler, "max_"):
            result["max_"] = scaler.max_.tolist() if hasattr(scaler.max_, "tolist") else str(scaler.max_)
        if hasattr(scaler, "data_min_"):
            result["data_min_"] = (
                scaler.data_min_.tolist() if hasattr(scaler.data_min_, "tolist") else str(scaler.data_min_)
            )
        if hasattr(scaler, "data_max_"):
            result["data_max_"] = (
                scaler.data_max_.tolist() if hasattr(scaler.data_max_, "tolist") else str(scaler.data_max_)
            )
        if hasattr(scaler, "data_range_"):
            result["data_range_"] = (
                scaler.data_range_.tolist() if hasattr(scaler.data_range_, "tolist") else str(scaler.data_range_)
            )

        # 特徴量数を推定
        if hasattr(scaler, "n_features_in_"):
            result["n_features"] = scaler.n_features_in_
        elif hasattr(scaler, "scale_") and hasattr(scaler.scale_, "__len__"):
            result["n_features"] = len(scaler.scale_)
        elif hasattr(scaler, "mean_") and hasattr(scaler.mean_, "__len__"):
            result["n_features"] = len(scaler.mean_)

        return result

    except Exception as e:
        return {"file_name": scaler_path.name, "success": False, "error": str(e)}


def analyze_kfold_results(json_path: Path) -> dict[str, Any]:
    """
    K-fold結果JSONファイルを解析
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            results = json.load(f)

        return {"file_name": json_path.name, "content": results, "success": True, "error": None}

    except Exception as e:
        return {"file_name": json_path.name, "success": False, "error": str(e)}


def analyze_submission_parquet(parquet_path: Path) -> dict[str, Any]:
    """
    サブミッションParquetファイルを解析
    """
    try:
        df = pd.read_parquet(str(parquet_path))

        result = {
            "file_name": parquet_path.name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "success": True,
            "error": None,
        }

        # 各列の統計情報
        for col in df.columns:
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                result[f"{col}_stats"] = {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "null_count": df[col].isnull().sum(),
                }
            else:
                result[f"{col}_stats"] = {
                    "unique_count": df[col].nunique(),
                    "null_count": df[col].isnull().sum(),
                    "most_common": df[col].value_counts().head().to_dict() if not df[col].empty else {},
                }

        return result

    except Exception as e:
        return {"file_name": parquet_path.name, "success": False, "error": str(e)}


def main():
    """
    メイン解析処理
    """
    base_path = Path("public_datasets/imu-only-datas")
    saved_models_path = base_path / "saved_models"

    results = {"pytorch_models": [], "feature_scaler": None, "kfold_results": None, "submission": None}

    print("=== PyTorchモデルファイル解析開始 ===")

    # 1. PyTorchモデルファイルの解析
    print("\n1. PyTorchモデルファイルの安全性とアーキテクチャ解析")
    model_files = list(saved_models_path.glob("fold_*_model.pth"))

    for model_file in sorted(model_files):
        result = analyze_pytorch_checkpoint_safety(model_file)
        results["pytorch_models"].append(result)
        print(f"  - {result['file_name']}: {'成功' if result['success'] else '失敗'}")
        if not result["success"]:
            print(f"    エラー: {result['error']}")

    # 2. 特徴量スケーラーの解析
    print("\n2. 特徴量スケーラー解析")
    scaler_file = saved_models_path / "feature_scaler.joblib"
    if scaler_file.exists():
        results["feature_scaler"] = analyze_feature_scaler(scaler_file)
        print(
            f"  - {results['feature_scaler']['file_name']}: {'成功' if results['feature_scaler']['success'] else '失敗'}"
        )
        if not results["feature_scaler"]["success"]:
            print(f"    エラー: {results['feature_scaler']['error']}")
    else:
        print("  - feature_scaler.joblib が見つかりません")

    # 3. K-fold結果の解析
    print("\n3. K-fold結果解析")
    kfold_file = saved_models_path / "kfold_results.json"
    if kfold_file.exists():
        results["kfold_results"] = analyze_kfold_results(kfold_file)
        print(
            f"  - {results['kfold_results']['file_name']}: {'成功' if results['kfold_results']['success'] else '失敗'}"
        )
        if not results["kfold_results"]["success"]:
            print(f"    エラー: {results['kfold_results']['error']}")
    else:
        print("  - kfold_results.json が見つかりません")

    # 4. サブミッションファイルの解析
    print("\n4. サブミッションファイル解析")
    submission_file = base_path / "submission.parquet"
    if submission_file.exists():
        results["submission"] = analyze_submission_parquet(submission_file)
        print(f"  - {results['submission']['file_name']}: {'成功' if results['submission']['success'] else '失敗'}")
        if not results["submission"]["success"]:
            print(f"    エラー: {results['submission']['error']}")
    else:
        print("  - submission.parquet が見つかりません")

    # 結果を保存
    output_path = Path("outputs/claude/model_analysis_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n解析結果を {output_path} に保存しました")

    return results


if __name__ == "__main__":
    results = main()
