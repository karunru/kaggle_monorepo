#!/usr/bin/env python3
"""
CMI PyTorchモデルアーキテクチャ比較解析スクリプト v2

weights_only=False を使用した詳細解析版
セキュリティリスクを承知の上で、アーキテクチャの詳細を抽出します。
"""

import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.serialization import get_unsafe_globals_in_checkpoint

warnings.filterwarnings("ignore")


class CMIModelAnalyzerV2:
    """CMIモデルの詳細解析クラス（v2）"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.datasets = [
            "cmi-imu-model",
            "cmi-fullfeats-models",
            "s-offline-0-8254-15fold",
            "cmi-imu-only-models",
            "b-offline-0-8855-specialprocess",
        ]
        self.results = {}

    def check_model_security(self, model_path: str) -> dict[str, Any]:
        """モデルファイルのセキュリティチェック"""
        try:
            unsafe_globals = get_unsafe_globals_in_checkpoint(model_path)
            return {
                "safe": len(unsafe_globals) == 0,
                "unsafe_globals": list(unsafe_globals)[:10],  # 最初の10個のみ表示
                "total_unsafe_count": len(unsafe_globals),
                "file_size": os.path.getsize(model_path),
            }
        except Exception as e:
            return {"safe": False, "error": str(e), "file_size": os.path.getsize(model_path)}

    def analyze_model_architecture_safe(self, model_path: str) -> dict[str, Any]:
        """
        安全性を考慮したアーキテクチャ解析
        weights_only=Falseを使用してモデルを読み込み
        """
        try:
            # まずは最低限の情報を取得
            file_size = os.path.getsize(model_path)

            # セキュリティリスクを承知でweights_only=Falseで読み込み
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            architecture_info = {
                "file_size": file_size,
                "checkpoint_keys": list(checkpoint.keys()),
                "model_present": False,
                "total_parameters": 0,
                "layers": [],
                "layer_types": {},
                "model_structure": {},
            }

            # checkpointからモデル部分を抽出
            model_data = None
            if "model" in checkpoint:
                model_data = checkpoint["model"]
                architecture_info["model_present"] = True
            elif "state_dict" in checkpoint:
                model_data = checkpoint["state_dict"]
                architecture_info["model_present"] = True
            elif isinstance(checkpoint, dict) and any("weight" in k or "bias" in k for k in checkpoint.keys()):
                model_data = checkpoint
                architecture_info["model_present"] = True

            if model_data is not None:
                architecture_info.update(self._analyze_state_dict(model_data))

            # その他の情報
            if "epoch" in checkpoint:
                architecture_info["training_epoch"] = checkpoint["epoch"]
            if "best_score" in checkpoint:
                architecture_info["best_score"] = checkpoint["best_score"]

            return architecture_info

        except Exception as e:
            return {"error": str(e), "file_size": os.path.getsize(model_path)}

    def _analyze_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """state_dictの詳細解析"""
        analysis = {
            "total_parameters": 0,
            "layers": [],
            "layer_types": defaultdict(int),
            "layer_shapes": {},
            "model_structure": {},
        }

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                layer_info = {
                    "name": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "num_params": value.numel(),
                    "requires_grad": value.requires_grad if hasattr(value, "requires_grad") else None,
                }
                analysis["layers"].append(layer_info)
                analysis["total_parameters"] += value.numel()

                # レイヤータイプの分類
                layer_type = self._classify_layer_type(key)
                analysis["layer_types"][layer_type] += 1

                # 形状情報の記録
                analysis["layer_shapes"][key] = list(value.shape)

        # モデル構造の推定
        analysis["model_structure"] = self._infer_model_structure_detailed(state_dict)

        return analysis

    def _classify_layer_type(self, layer_name: str) -> str:
        """レイヤー名の詳細分類"""
        name_lower = layer_name.lower()

        if "conv1d" in name_lower or "conv" in name_lower:
            return "convolution"
        elif "linear" in name_lower or "fc" in name_lower:
            return "linear"
        elif "bn" in name_lower or "batchnorm" in name_lower:
            return "batch_norm"
        elif "layernorm" in name_lower:
            return "layer_norm"
        elif "lstm" in name_lower:
            return "lstm"
        elif "gru" in name_lower:
            return "gru"
        elif "attention" in name_lower or "attn" in name_lower:
            return "attention"
        elif "transformer" in name_lower:
            return "transformer"
        elif "embed" in name_lower:
            return "embedding"
        elif "weight" in name_lower:
            return "weight"
        elif "bias" in name_lower:
            return "bias"
        else:
            return "other"

    def _infer_model_structure_detailed(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """詳細なモデル構造推定"""
        structure = {
            "architecture_type": "unknown",
            "has_transformer": False,
            "has_lstm": False,
            "has_gru": False,
            "has_attention": False,
            "has_cnn": False,
            "num_linear_layers": 0,
            "num_conv_layers": 0,
            "estimated_depth": 0,
            "input_size": None,
            "output_size": None,
        }

        layer_names = list(state_dict.keys())

        # アーキテクチャタイプの推定
        if any("binet" in name.lower() or "bi_net" in name.lower() for name in layer_names):
            structure["architecture_type"] = "BiNet"
        elif any("singlenet" in name.lower() or "single_net" in name.lower() for name in layer_names):
            structure["architecture_type"] = "singleNet"
        elif any("branch" in name.lower() or "tower" in name.lower() for name in layer_names):
            structure["architecture_type"] = "multi_branch"
        else:
            structure["architecture_type"] = "single_branch"

        # 各コンポーネントの存在チェック
        structure["has_transformer"] = any("transformer" in name.lower() for name in layer_names)
        structure["has_lstm"] = any("lstm" in name.lower() for name in layer_names)
        structure["has_gru"] = any("gru" in name.lower() for name in layer_names)
        structure["has_attention"] = any("attention" in name.lower() or "attn" in name.lower() for name in layer_names)
        structure["has_cnn"] = any("conv" in name.lower() for name in layer_names)

        # レイヤー数のカウント
        structure["num_linear_layers"] = sum(
            1 for name in layer_names if "linear" in name.lower() and "weight" in name.lower()
        )
        structure["num_conv_layers"] = sum(
            1 for name in layer_names if "conv" in name.lower() and "weight" in name.lower()
        )

        # 深さの推定
        max_layer_num = 0
        for name in layer_names:
            import re

            numbers = re.findall(r"\d+", name)
            if numbers:
                max_layer_num = max(max_layer_num, max(int(num) for num in numbers))
        structure["estimated_depth"] = max_layer_num

        # 入力・出力サイズの推定
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                if "embedding" in name.lower() or (name.startswith("0.") and "weight" in name):
                    if len(tensor.shape) >= 2:
                        structure["input_size"] = tensor.shape[-1]
                elif "classifier" in name.lower() or "output" in name.lower() or name.endswith(".weight"):
                    if len(tensor.shape) >= 2:
                        structure["output_size"] = tensor.shape[0]

        return structure

    def analyze_features(self, dataset_path: str) -> dict[str, Any]:
        """特徴量情報の詳細解析"""
        feature_info = {}

        # feature_cols.npy の解析
        feature_cols_path = Path(dataset_path) / "feature_cols.npy"
        if feature_cols_path.exists():
            try:
                feature_cols = np.load(feature_cols_path, allow_pickle=True)
                if isinstance(feature_cols, np.ndarray):
                    if feature_cols.dtype == "object":
                        cols_list = feature_cols.tolist()
                    else:
                        cols_list = feature_cols.tolist()
                else:
                    cols_list = list(feature_cols)

                feature_info["feature_columns"] = {
                    "count": len(cols_list),
                    "sample_columns": cols_list[:10],  # 最初の10個のみ
                    "feature_types": self._analyze_feature_types(cols_list),
                }
            except Exception as e:
                feature_info["feature_columns"] = {"error": str(e)}

        # scaler.pkl の詳細解析
        scaler_path = Path(dataset_path) / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                feature_info["scaler"] = {
                    "type": str(type(scaler).__name__),
                    "n_features": getattr(scaler, "n_features_in_", "unknown"),
                    "feature_names": getattr(scaler, "feature_names_in_", None),
                    "scale_": getattr(scaler, "scale_", None),
                    "mean_": getattr(scaler, "mean_", None),
                }

                # scalerの詳細情報
                if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                    feature_info["scaler"]["scale_stats"] = {
                        "min": float(np.min(scaler.scale_)),
                        "max": float(np.max(scaler.scale_)),
                        "mean": float(np.mean(scaler.scale_)),
                    }

            except Exception as e:
                feature_info["scaler"] = {"error": str(e)}

        # 他のファイルの解析
        self._analyze_other_files(dataset_path, feature_info)

        return feature_info

    def _analyze_feature_types(self, feature_columns: list[str]) -> dict[str, int]:
        """特徴量の種類を分析"""
        feature_types = defaultdict(int)

        for col in feature_columns:
            col_lower = col.lower()
            if "accel" in col_lower or "acc_" in col_lower:
                feature_types["accelerometer"] += 1
            elif "gyro" in col_lower:
                feature_types["gyroscope"] += 1
            elif "mag" in col_lower:
                feature_types["magnetometer"] += 1
            elif "linear" in col_lower:
                feature_types["linear_acceleration"] += 1
            elif "gravity" in col_lower:
                feature_types["gravity"] += 1
            elif "rotation" in col_lower:
                feature_types["rotation"] += 1
            elif any(stat in col_lower for stat in ["mean", "std", "var", "min", "max"]):
                feature_types["statistical"] += 1
            elif any(freq in col_lower for freq in ["freq", "fft", "spectral"]):
                feature_types["frequency"] += 1
            else:
                feature_types["other"] += 1

        return dict(feature_types)

    def _analyze_other_files(self, dataset_path: str, feature_info: dict[str, Any]) -> None:
        """その他のファイルの解析"""
        files_to_analyze = [
            ("static_scaler.pkl", "static_scaler"),
            ("sequence_maxlen.npy", "sequence_maxlen"),
            ("gesture_classes.npy", "gesture_classes"),
        ]

        for filename, key in files_to_analyze:
            file_path = Path(dataset_path) / filename
            if file_path.exists():
                try:
                    if filename.endswith(".pkl"):
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)
                        feature_info[key] = {
                            "type": str(type(data).__name__),
                            "attributes": [attr for attr in dir(data) if not attr.startswith("_")][:10],
                        }
                    elif filename.endswith(".npy"):
                        data = np.load(file_path, allow_pickle=True)
                        feature_info[key] = {
                            "shape": data.shape if hasattr(data, "shape") else None,
                            "dtype": str(data.dtype) if hasattr(data, "dtype") else None,
                            "data": data.tolist() if data.size < 100 else f"Array too large: {data.size} elements",
                        }
                except Exception as e:
                    feature_info[key] = {"error": str(e)}

    def run_comprehensive_analysis(self) -> dict[str, Any]:
        """包括的解析の実行"""
        results = {}

        for dataset in self.datasets:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset} not found")
                continue

            print(f"\\nAnalyzing dataset: {dataset}")
            dataset_results = {"models": {}, "features": {}, "summary": {}}

            # モデル解析
            model_files = list(dataset_path.glob("*.pt"))
            for i, model_file in enumerate(model_files):
                print(f"  [{i + 1}/{len(model_files)}] {model_file.name}")

                security_info = self.check_model_security(str(model_file))
                architecture_info = self.analyze_model_architecture_safe(str(model_file))

                dataset_results["models"][model_file.name] = {
                    "security": security_info,
                    "architecture": architecture_info,
                }

            # 特徴量解析
            dataset_results["features"] = self.analyze_features(str(dataset_path))

            # サマリー作成
            dataset_results["summary"] = self._create_detailed_summary(dataset_results)

            results[dataset] = dataset_results

        return results

    def _create_detailed_summary(self, dataset_results: dict[str, Any]) -> dict[str, Any]:
        """詳細サマリーの作成"""
        summary = {
            "total_models": len(dataset_results["models"]),
            "architecture_analysis": {},
            "security_analysis": {},
            "model_statistics": {},
        }

        architectures = []
        total_params = []
        safe_count = 0

        for model_name, model_info in dataset_results["models"].items():
            # セキュリティ
            if model_info["security"].get("safe", False):
                safe_count += 1

            # アーキテクチャ
            arch_info = model_info["architecture"]
            if "model_structure" in arch_info:
                arch_type = arch_info["model_structure"].get("architecture_type", "unknown")
                architectures.append(arch_type)

            # パラメータ数
            if "total_parameters" in arch_info:
                total_params.append(arch_info["total_parameters"])

        # 統計情報
        summary["security_analysis"] = {
            "safe_models": safe_count,
            "total_models": len(dataset_results["models"]),
            "safety_ratio": safe_count / len(dataset_results["models"]) if dataset_results["models"] else 0,
        }

        if architectures:
            from collections import Counter

            arch_counts = Counter(architectures)
            summary["architecture_analysis"] = {
                "primary_architecture": arch_counts.most_common(1)[0][0],
                "architecture_distribution": dict(arch_counts),
            }

        if total_params:
            summary["model_statistics"] = {
                "avg_parameters": int(np.mean(total_params)),
                "min_parameters": int(np.min(total_params)),
                "max_parameters": int(np.max(total_params)),
                "std_parameters": int(np.std(total_params)),
            }

        return summary

    def generate_detailed_comparison(self, results: dict[str, Any]) -> dict[str, Any]:
        """詳細比較分析"""
        comparison = {"architecture_comparison": {}, "feature_comparison": {}, "performance_insights": {}}

        # アーキテクチャ比較
        singlenet_datasets = []
        binet_datasets = []

        for dataset_name, data in results.items():
            primary_arch = data["summary"].get("architecture_analysis", {}).get("primary_architecture", "unknown")

            if "singleNet" in primary_arch or "single" in primary_arch.lower():
                singlenet_datasets.append(
                    {
                        "dataset": dataset_name,
                        "avg_params": data["summary"].get("model_statistics", {}).get("avg_parameters", 0),
                        "feature_count": data["features"].get("feature_columns", {}).get("count", 0),
                    }
                )
            elif "BiNet" in primary_arch or "bi" in primary_arch.lower():
                binet_datasets.append(
                    {
                        "dataset": dataset_name,
                        "avg_params": data["summary"].get("model_statistics", {}).get("avg_parameters", 0),
                        "feature_count": data["features"].get("feature_columns", {}).get("count", 0),
                    }
                )

        comparison["architecture_comparison"] = {
            "singleNet": {
                "datasets": singlenet_datasets,
                "avg_parameters": np.mean([d["avg_params"] for d in singlenet_datasets]) if singlenet_datasets else 0,
                "avg_features": np.mean([d["feature_count"] for d in singlenet_datasets]) if singlenet_datasets else 0,
            },
            "BiNet": {
                "datasets": binet_datasets,
                "avg_parameters": np.mean([d["avg_params"] for d in binet_datasets]) if binet_datasets else 0,
                "avg_features": np.mean([d["feature_count"] for d in binet_datasets]) if binet_datasets else 0,
            },
        }

        return comparison


def main():
    """メイン実行"""
    data_root = (
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/data/cmi_models"
    )

    analyzer = CMIModelAnalyzerV2(data_root)

    print("=== CMI PyTorchモデル詳細解析 v2 開始 ===")
    print("注意: セキュリティリスクを承知でweights_only=Falseを使用します")

    # 解析実行
    results = analyzer.run_comprehensive_analysis()

    # 比較分析
    comparison = analyzer.generate_detailed_comparison(results)

    # 結果保存
    output_dir = Path(
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/outputs/claude"
    )
    output_dir.mkdir(exist_ok=True)

    import json

    with open(output_dir / "cmi_detailed_analysis_v2.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "comparison": comparison}, f, ensure_ascii=False, indent=2, default=str)

    print("\\n=== 解析完了 ===")
    print(f"詳細結果: {output_dir / 'cmi_detailed_analysis_v2.json'}")


if __name__ == "__main__":
    main()
