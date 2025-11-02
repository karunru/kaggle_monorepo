#!/usr/bin/env python3
"""
CMI PyTorchモデルアーキテクチャ比較解析スクリプト

このスクリプトは指定された5つのCMIデータセットのPyTorchモデルを解析し、
以下の情報を抽出・比較します：
1. モデルのセキュリティチェック
2. アーキテクチャの詳細構造
3. singleNet vs BiNet の比較
4. 特徴量情報の比較
5. 性能との関係分析
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.serialization import get_unsafe_globals_in_checkpoint

warnings.filterwarnings("ignore")


class CMIModelAnalyzer:
    """CMIモデルの包括的解析クラス"""

    def __init__(self, data_root: str):
        """
        初期化

        Args:
            data_root: データセットのルートディレクトリパス
        """
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
        """
        PyTorchモデルファイルのセキュリティチェック

        Args:
            model_path: モデルファイルのパス

        Returns:
            セキュリティチェック結果
        """
        try:
            unsafe_globals = get_unsafe_globals_in_checkpoint(model_path)
            return {
                "safe": len(unsafe_globals) == 0,
                "unsafe_globals": list(unsafe_globals),
                "file_size": os.path.getsize(model_path),
            }
        except Exception as e:
            return {"safe": False, "error": str(e), "file_size": os.path.getsize(model_path)}

    def analyze_model_architecture(self, model_path: str) -> dict[str, Any]:
        """
        モデルアーキテクチャの詳細解析

        Args:
            model_path: モデルファイルのパス

        Returns:
            アーキテクチャ情報
        """
        try:
            # CPU上でのロードを強制
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

            architecture_info = {
                "total_parameters": 0,
                "layers": [],
                "layer_types": {},
                "input_shapes": {},
                "output_shapes": {},
                "model_structure": {},
            }

            # state_dictの解析
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # レイヤー情報の抽出
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    layer_info = {
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "num_params": tensor.numel(),
                    }
                    architecture_info["layers"].append(layer_info)
                    architecture_info["total_parameters"] += tensor.numel()

                    # レイヤータイプの分類
                    layer_type = self._classify_layer_type(key)
                    if layer_type not in architecture_info["layer_types"]:
                        architecture_info["layer_types"][layer_type] = 0
                    architecture_info["layer_types"][layer_type] += 1

            # モデル構造の推定
            architecture_info["model_structure"] = self._infer_model_structure(state_dict)

            return architecture_info

        except Exception as e:
            return {"error": str(e)}

    def _classify_layer_type(self, layer_name: str) -> str:
        """レイヤー名からレイヤータイプを分類"""
        layer_name_lower = layer_name.lower()

        if "conv" in layer_name_lower:
            return "convolution"
        elif "linear" in layer_name_lower or "fc" in layer_name_lower:
            return "linear"
        elif "bn" in layer_name_lower or "batchnorm" in layer_name_lower:
            return "batch_norm"
        elif "lstm" in layer_name_lower or "gru" in layer_name_lower:
            return "recurrent"
        elif "attention" in layer_name_lower or "attn" in layer_name_lower:
            return "attention"
        elif "embed" in layer_name_lower:
            return "embedding"
        elif "weight" in layer_name_lower:
            return "weight"
        elif "bias" in layer_name_lower:
            return "bias"
        else:
            return "other"

    def _infer_model_structure(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """state_dictからモデル構造を推定"""
        structure = {
            "is_single_net": True,
            "is_bi_net": False,
            "has_lstm": False,
            "has_attention": False,
            "num_linear_layers": 0,
            "num_conv_layers": 0,
            "model_depth": 0,
        }

        layer_names = list(state_dict.keys())

        # BiNetの特徴：複数のネットワーク分岐があるかチェック
        branch_indicators = ["branch", "net1", "net2", "path1", "path2", "tower"]
        for indicator in branch_indicators:
            if any(indicator in name.lower() for name in layer_names):
                structure["is_bi_net"] = True
                structure["is_single_net"] = False
                break

        # LSTM/GRUの存在チェック
        structure["has_lstm"] = any("lstm" in name.lower() or "gru" in name.lower() for name in layer_names)

        # Attentionの存在チェック
        structure["has_attention"] = any("attention" in name.lower() or "attn" in name.lower() for name in layer_names)

        # レイヤー数のカウント
        structure["num_linear_layers"] = sum(
            1 for name in layer_names if "linear" in name.lower() or "fc" in name.lower()
        )
        structure["num_conv_layers"] = sum(1 for name in layer_names if "conv" in name.lower())

        # モデルの深さ推定
        max_layer_num = 0
        for name in layer_names:
            # レイヤー番号の抽出を試行
            import re

            numbers = re.findall(r"\d+", name)
            if numbers:
                max_layer_num = max(max_layer_num, max(int(num) for num in numbers))

        structure["model_depth"] = max_layer_num

        return structure

    def analyze_features(self, dataset_path: str) -> dict[str, Any]:
        """
        特徴量情報の解析

        Args:
            dataset_path: データセットディレクトリのパス

        Returns:
            特徴量情報
        """
        feature_info = {}

        # feature_cols.npy の解析
        feature_cols_path = Path(dataset_path) / "feature_cols.npy"
        if feature_cols_path.exists():
            try:
                feature_cols = np.load(feature_cols_path, allow_pickle=True)
                feature_info["feature_columns"] = {
                    "count": len(feature_cols),
                    "columns": feature_cols.tolist() if hasattr(feature_cols, "tolist") else list(feature_cols),
                }
            except Exception as e:
                feature_info["feature_columns"] = {"error": str(e)}

        # scaler.pkl の解析
        scaler_path = Path(dataset_path) / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                feature_info["scaler"] = {
                    "type": str(type(scaler)),
                    "n_features": getattr(scaler, "n_features_in_", "unknown"),
                    "feature_names": getattr(scaler, "feature_names_in_", None),
                }
            except Exception as e:
                feature_info["scaler"] = {"error": str(e)}

        # static_scaler.pkl の解析
        static_scaler_path = Path(dataset_path) / "static_scaler.pkl"
        if static_scaler_path.exists():
            try:
                with open(static_scaler_path, "rb") as f:
                    static_scaler = pickle.load(f)
                feature_info["static_scaler"] = {
                    "type": str(type(static_scaler)),
                    "n_features": getattr(static_scaler, "n_features_in_", "unknown"),
                }
            except Exception as e:
                feature_info["static_scaler"] = {"error": str(e)}

        # sequence_maxlen.npy の解析
        sequence_maxlen_path = Path(dataset_path) / "sequence_maxlen.npy"
        if sequence_maxlen_path.exists():
            try:
                sequence_maxlen = np.load(sequence_maxlen_path)
                feature_info["sequence_maxlen"] = sequence_maxlen.tolist()
            except Exception as e:
                feature_info["sequence_maxlen"] = {"error": str(e)}

        # gesture_classes.npy の解析
        gesture_classes_path = Path(dataset_path) / "gesture_classes.npy"
        if gesture_classes_path.exists():
            try:
                gesture_classes = np.load(gesture_classes_path, allow_pickle=True)
                feature_info["gesture_classes"] = {"count": len(gesture_classes), "classes": gesture_classes.tolist()}
            except Exception as e:
                feature_info["gesture_classes"] = {"error": str(e)}

        return feature_info

    def run_comprehensive_analysis(self) -> dict[str, Any]:
        """包括的な解析の実行"""
        comprehensive_results = {}

        for dataset in self.datasets:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset} not found at {dataset_path}")
                continue

            print(f"Analyzing dataset: {dataset}")
            dataset_results = {"models": {}, "features": {}, "summary": {}}

            # モデルファイルの検索と解析
            model_files = list(dataset_path.glob("*.pt"))

            for model_file in model_files:
                model_name = model_file.name
                print(f"  - Analyzing model: {model_name}")

                # セキュリティチェック
                security_info = self.check_model_security(str(model_file))

                # アーキテクチャ解析
                architecture_info = self.analyze_model_architecture(str(model_file))

                dataset_results["models"][model_name] = {"security": security_info, "architecture": architecture_info}

            # 特徴量解析
            dataset_results["features"] = self.analyze_features(str(dataset_path))

            # データセットサマリー
            dataset_results["summary"] = self._create_dataset_summary(dataset_results)

            comprehensive_results[dataset] = dataset_results

        return comprehensive_results

    def _create_dataset_summary(self, dataset_results: dict[str, Any]) -> dict[str, Any]:
        """データセットのサマリー作成"""
        summary = {
            "total_models": len(dataset_results["models"]),
            "model_types": {},
            "total_parameters_avg": 0,
            "safe_models": 0,
            "architecture_types": set(),
        }

        total_params = []

        for model_name, model_info in dataset_results["models"].items():
            # セキュリティ
            if model_info["security"].get("safe", False):
                summary["safe_models"] += 1

            # モデルタイプ
            if "singleNet" in model_name:
                summary["model_types"]["singleNet"] = summary["model_types"].get("singleNet", 0) + 1
                summary["architecture_types"].add("singleNet")
            elif "BiNet" in model_name:
                summary["model_types"]["BiNet"] = summary["model_types"].get("BiNet", 0) + 1
                summary["architecture_types"].add("BiNet")

            # パラメータ数
            if "architecture" in model_info and "total_parameters" in model_info["architecture"]:
                total_params.append(model_info["architecture"]["total_parameters"])

        if total_params:
            summary["total_parameters_avg"] = int(np.mean(total_params))
            summary["total_parameters_min"] = int(np.min(total_params))
            summary["total_parameters_max"] = int(np.max(total_params))

        summary["architecture_types"] = list(summary["architecture_types"])

        return summary

    def compare_architectures(self, results: dict[str, Any]) -> dict[str, Any]:
        """アーキテクチャの比較分析"""
        comparison = {
            "singleNet_vs_BiNet": {
                "singleNet": {"datasets": [], "avg_params": 0, "characteristics": {}},
                "BiNet": {"datasets": [], "avg_params": 0, "characteristics": {}},
            },
            "feature_comparison": {},
            "performance_correlation": {},
        }

        singlenet_params = []
        binet_params = []

        for dataset_name, dataset_data in results.items():
            summary = dataset_data["summary"]

            if "singleNet" in summary["architecture_types"]:
                comparison["singleNet_vs_BiNet"]["singleNet"]["datasets"].append(dataset_name)
                if "total_parameters_avg" in summary:
                    singlenet_params.append(summary["total_parameters_avg"])

            if "BiNet" in summary["architecture_types"]:
                comparison["singleNet_vs_BiNet"]["BiNet"]["datasets"].append(dataset_name)
                if "total_parameters_avg" in summary:
                    binet_params.append(summary["total_parameters_avg"])

        if singlenet_params:
            comparison["singleNet_vs_BiNet"]["singleNet"]["avg_params"] = int(np.mean(singlenet_params))

        if binet_params:
            comparison["singleNet_vs_BiNet"]["BiNet"]["avg_params"] = int(np.mean(binet_params))

        # 特徴量比較
        for dataset_name, dataset_data in results.items():
            features = dataset_data.get("features", {})
            if "feature_columns" in features and "count" in features["feature_columns"]:
                comparison["feature_comparison"][dataset_name] = features["feature_columns"]["count"]

        return comparison

    def generate_report(self, results: dict[str, Any], comparison: dict[str, Any]) -> str:
        """詳細レポートの生成"""
        report = []
        report.append("# CMI PyTorchモデルアーキテクチャ比較解析レポート\n")

        # 概要
        report.append("## 解析概要")
        report.append(f"解析対象データセット数: {len(results)}")
        total_models = sum(data["summary"]["total_models"] for data in results.values())
        report.append(f"総モデル数: {total_models}")

        # セキュリティチェック結果
        report.append("\n## セキュリティチェック結果")
        for dataset_name, dataset_data in results.items():
            safe_count = dataset_data["summary"]["safe_models"]
            total_count = dataset_data["summary"]["total_models"]
            report.append(f"- {dataset_name}: {safe_count}/{total_count} モデルが安全")

        # アーキテクチャ比較
        report.append("\n## アーキテクチャ比較")

        singlenet_info = comparison["singleNet_vs_BiNet"]["singleNet"]
        binet_info = comparison["singleNet_vs_BiNet"]["BiNet"]

        report.append("### singleNet アーキテクチャ")
        report.append(f"- 使用データセット: {', '.join(singlenet_info['datasets'])}")
        report.append(f"- 平均パラメータ数: {singlenet_info['avg_params']:,}")

        report.append("\n### BiNet アーキテクチャ")
        report.append(f"- 使用データセット: {', '.join(binet_info['datasets'])}")
        report.append(f"- 平均パラメータ数: {binet_info['avg_params']:,}")

        # 特徴量比較
        report.append("\n## 特徴量情報比較")
        for dataset_name, feature_count in comparison["feature_comparison"].items():
            report.append(f"- {dataset_name}: {feature_count} 特徴量")

        # 詳細分析
        report.append("\n## 詳細分析結果")
        for dataset_name, dataset_data in results.items():
            report.append(f"\n### {dataset_name}")

            summary = dataset_data["summary"]
            report.append(f"- モデル数: {summary['total_models']}")
            report.append(f"- アーキテクチャタイプ: {', '.join(summary['architecture_types'])}")

            if "total_parameters_avg" in summary:
                report.append(f"- 平均パラメータ数: {summary['total_parameters_avg']:,}")

            # 特徴量情報
            features = dataset_data.get("features", {})
            if "feature_columns" in features and "count" in features["feature_columns"]:
                report.append(f"- 特徴量数: {features['feature_columns']['count']}")

        return "\n".join(report)


def main():
    """メイン実行関数"""
    # データセットのパスを設定
    data_root = (
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/data/cmi_models"
    )

    # 解析器の初期化
    analyzer = CMIModelAnalyzer(data_root)

    print("=== CMI PyTorchモデルアーキテクチャ比較解析開始 ===\n")

    # 包括的解析の実行
    results = analyzer.run_comprehensive_analysis()

    # アーキテクチャ比較
    comparison = analyzer.compare_architectures(results)

    # レポート生成
    report = analyzer.generate_report(results, comparison)

    # 結果の保存
    output_dir = Path(
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/outputs/claude"
    )
    output_dir.mkdir(exist_ok=True)

    # 詳細結果をJSONで保存
    import json

    with open(output_dir / "cmi_analysis_detailed_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "comparison": comparison}, f, ensure_ascii=False, indent=2, default=str)

    # レポートをMarkdownで保存
    with open(output_dir / "cmi_architecture_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("=== 解析完了 ===")
    print(f"詳細結果: {output_dir / 'cmi_analysis_detailed_results.json'}")
    print(f"レポート: {output_dir / 'cmi_architecture_analysis_report.md'}")
    print("\n=== レポート概要 ===")
    print(report)


if __name__ == "__main__":
    main()
