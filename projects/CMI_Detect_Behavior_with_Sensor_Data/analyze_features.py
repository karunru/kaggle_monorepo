#!/usr/bin/env python3
"""特徴量詳細解析スクリプト"""

import json
import pickle
from pathlib import Path

import numpy as np


def analyze_feature_cols(dataset_path):
    """feature_cols.npyを解析"""
    dataset_path = Path(dataset_path)
    feature_file = dataset_path / "feature_cols.npy"

    if not feature_file.exists():
        return None

    try:
        cols = np.load(feature_file, allow_pickle=True)
        return {
            "shape": cols.shape,
            "dtype": str(cols.dtype),
            "total_features": len(cols),
            "features": cols.tolist() if cols.size < 1000 else cols[:50].tolist() + ["... and more"],
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_scaler(dataset_path):
    """scaler.pklを解析"""
    dataset_path = Path(dataset_path)
    scaler_file = dataset_path / "scaler.pkl"

    if not scaler_file.exists():
        return None

    try:
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

        result = {"type": type(scaler).__name__, "module": type(scaler).__module__}

        # StandardScalerの場合
        if hasattr(scaler, "mean_"):
            result["n_features"] = len(scaler.mean_)
            result["mean_sample"] = scaler.mean_[:5].tolist() if len(scaler.mean_) > 5 else scaler.mean_.tolist()
            result["scale_sample"] = scaler.scale_[:5].tolist() if len(scaler.scale_) > 5 else scaler.scale_.tolist()
            result["var_sample"] = scaler.var_[:5].tolist() if hasattr(scaler, "var_") and len(scaler.var_) > 5 else []

        return result
    except Exception as e:
        return {"error": str(e)}


def analyze_static_scaler(dataset_path):
    """static_scaler.pklを解析"""
    dataset_path = Path(dataset_path)
    scaler_file = dataset_path / "static_scaler.pkl"

    if not scaler_file.exists():
        return None

    try:
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

        result = {"type": type(scaler).__name__, "module": type(scaler).__module__}

        if hasattr(scaler, "mean_"):
            result["n_features"] = len(scaler.mean_)

        return result
    except Exception as e:
        return {"error": str(e)}


def analyze_gesture_classes(dataset_path):
    """gesture_classes.npyを解析"""
    dataset_path = Path(dataset_path)
    gesture_file = dataset_path / "gesture_classes.npy"

    if not gesture_file.exists():
        return None

    try:
        classes = np.load(gesture_file, allow_pickle=True)
        return {"shape": classes.shape, "dtype": str(classes.dtype), "classes": classes.tolist()}
    except Exception as e:
        return {"error": str(e)}


def main():
    base_path = Path("public_datasets")
    datasets = [
        "cmi-imu-model",
        "cmi-fullfeats-models",
        "s-offline-0-8254-15fold",
        "cmi-imu-only-models",
        "b-offline-0-8855-specialprocess",
        "imu-only-datas",
        "kdxf-collisiondetect",
    ]

    results = {}

    for dataset in datasets:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            continue

        print(f"\n=== {dataset} ===")

        results[dataset] = {
            "feature_cols": analyze_feature_cols(dataset_path),
            "scaler": analyze_scaler(dataset_path),
            "static_scaler": analyze_static_scaler(dataset_path),
            "gesture_classes": analyze_gesture_classes(dataset_path),
        }

        # 特徴量数の表示
        if results[dataset]["feature_cols"]:
            print(f"Features: {results[dataset]['feature_cols'].get('total_features', 'N/A')}")
            if "features" in results[dataset]["feature_cols"]:
                features = results[dataset]["feature_cols"]["features"]
                if len(features) > 10:
                    print(f"First 10: {features[:10]}")
                else:
                    print(f"All: {features}")

        # スケーラー情報の表示
        if results[dataset]["scaler"]:
            print(f"Scaler: {results[dataset]['scaler'].get('type', 'N/A')}")
            print(f"N_features in scaler: {results[dataset]['scaler'].get('n_features', 'N/A')}")

        # ジェスチャークラスの表示
        if results[dataset]["gesture_classes"]:
            classes = results[dataset]["gesture_classes"].get("classes", [])
            if classes:
                print(f"Gesture classes ({len(classes)}): {classes[:5]}...")

    # 結果をJSONで保存
    output_file = Path("outputs/claude/feature_analysis_detailed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
