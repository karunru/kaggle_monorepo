#!/usr/bin/env python3
"""
PyTorchモデル解析ツール

このスクリプトはダウンロードしたKaggleデータセットのモデルファイルを解析し、
モデルアーキテクチャ、特徴量情報、パフォーマンス指標を抽出します。
"""

import json
import pickle
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

warnings.filterwarnings("ignore")


@dataclass
class ModelAnalysis:
    """モデル解析結果を格納するデータクラス"""

    dataset_name: str
    model_files: list[str]
    model_architecture: dict[str, Any]
    feature_info: dict[str, Any]
    preprocessing_info: dict[str, Any]
    performance_metrics: dict[str, Any]
    file_sizes: dict[str, int]
    errors: list[str]


class ModelAnalyzer:
    """PyTorchモデル解析クラス"""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_path.name

    def analyze_pytorch_model(self, model_path: Path) -> dict[str, Any]:
        """PyTorchモデルファイルを解析"""
        try:
            # チェックポイント内のunsafeな情報を取得
            unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(model_path))

            # モデルを読み込み（CPUにマップ）
            checkpoint = torch.load(model_path, map_location="cpu")

            analysis = {
                "file_name": model_path.name,
                "file_size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
                "unsafe_globals": list(unsafe_globals) if unsafe_globals else [],
                "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            }

            # モデル状態辞書の解析
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif all(k.count(".") > 0 for k in checkpoint.keys() if isinstance(k, str)):
                    # 直接state_dictの場合
                    state_dict = checkpoint
                else:
                    state_dict = None

                # その他の情報を抽出
                if "epoch" in checkpoint:
                    analysis["epoch"] = checkpoint["epoch"]
                if "loss" in checkpoint:
                    analysis["final_loss"] = checkpoint["loss"]
                if "optimizer" in checkpoint:
                    analysis["has_optimizer"] = True
            else:
                # 直接state_dictの場合
                state_dict = checkpoint

            # レイヤー情報の解析
            if state_dict:
                analysis["layer_analysis"] = self._analyze_state_dict(state_dict)
                analysis["total_parameters"] = self._count_parameters(state_dict)

            return analysis

        except Exception as e:
            return {"file_name": model_path.name, "error": str(e), "traceback": traceback.format_exc()}

    def _analyze_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """State dictからレイヤー情報を解析"""
        layers = {}
        layer_types = set()

        for key, tensor in state_dict.items():
            # レイヤー名を抽出（最初のドットまで）
            layer_name = key.split(".")[0] if "." in key else key

            if layer_name not in layers:
                layers[layer_name] = {"parameters": [], "shapes": [], "total_params": 0}

            layers[layer_name]["parameters"].append(key)
            layers[layer_name]["shapes"].append(list(tensor.shape))
            layers[layer_name]["total_params"] += tensor.numel()

            # レイヤータイプを推定
            if "conv" in layer_name.lower():
                layer_types.add("Convolutional")
            elif "linear" in layer_name.lower() or "fc" in layer_name.lower():
                layer_types.add("Linear")
            elif "lstm" in layer_name.lower():
                layer_types.add("LSTM")
            elif "gru" in layer_name.lower():
                layer_types.add("GRU")
            elif "norm" in layer_name.lower():
                layer_types.add("Normalization")
            elif "embed" in layer_name.lower():
                layer_types.add("Embedding")

        return {
            "layers": dict(list(layers.items())[:10]),  # 最初の10レイヤーのみ
            "total_layers": len(layers),
            "estimated_layer_types": list(layer_types),
            "input_shape_hints": self._infer_input_shape(state_dict),
            "output_shape_hints": self._infer_output_shape(state_dict),
        }

    def _count_parameters(self, state_dict: dict[str, torch.Tensor]) -> int:
        """総パラメータ数をカウント"""
        return sum(tensor.numel() for tensor in state_dict.values())

    def _infer_input_shape(self, state_dict: dict[str, torch.Tensor]) -> list[int] | None:
        """入力形状を推定"""
        # 最初のレイヤーの重みから推定
        first_keys = [k for k in state_dict.keys() if "weight" in k]
        if first_keys:
            first_weight = state_dict[first_keys[0]]
            if len(first_weight.shape) >= 2:
                return list(first_weight.shape[1:])  # バッチ次元を除く
        return None

    def _infer_output_shape(self, state_dict: dict[str, torch.Tensor]) -> list[int] | None:
        """出力形状を推定"""
        # 最後のレイヤーの重みから推定
        last_keys = [k for k in state_dict.keys() if "weight" in k]
        if last_keys:
            last_weight = state_dict[last_keys[-1]]
            if len(last_weight.shape) >= 1:
                return [last_weight.shape[0]]  # 出力次元
        return None

    def load_preprocessing_info(self) -> dict[str, Any]:
        """前処理情報を読み込み"""
        preprocessing_info = {}

        # scaler.pklファイルを探す
        for scaler_file in self.dataset_path.glob("*.pkl"):
            try:
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
                preprocessing_info[scaler_file.name] = {
                    "type": type(scaler).__name__,
                    "attributes": [
                        attr for attr in dir(scaler) if not attr.startswith("_") and not callable(getattr(scaler, attr))
                    ][:10],
                }

                # StandardScaler/MinMaxScalerの詳細情報
                if hasattr(scaler, "mean_"):
                    preprocessing_info[scaler_file.name]["feature_count"] = len(scaler.mean_)
                    preprocessing_info[scaler_file.name]["mean_stats"] = {
                        "min": float(np.min(scaler.mean_)),
                        "max": float(np.max(scaler.mean_)),
                        "std": float(np.std(scaler.mean_)),
                    }
                if hasattr(scaler, "scale_"):
                    preprocessing_info[scaler_file.name]["scale_stats"] = {
                        "min": float(np.min(scaler.scale_)),
                        "max": float(np.max(scaler.scale_)),
                        "std": float(np.std(scaler.scale_)),
                    }

            except Exception as e:
                preprocessing_info[scaler_file.name] = {"error": str(e)}

        # .joblibファイルを探す
        for joblib_file in self.dataset_path.glob("*.joblib"):
            try:
                scaler = joblib.load(joblib_file)
                preprocessing_info[joblib_file.name] = {
                    "type": type(scaler).__name__,
                    "attributes": [
                        attr for attr in dir(scaler) if not attr.startswith("_") and not callable(getattr(scaler, attr))
                    ][:10],
                }
            except Exception as e:
                preprocessing_info[joblib_file.name] = {"error": str(e)}

        return preprocessing_info

    def load_feature_info(self) -> dict[str, Any]:
        """特徴量情報を読み込み"""
        feature_info = {}

        # .npyファイルを探す
        for npy_file in self.dataset_path.glob("*.npy"):
            try:
                data = np.load(npy_file, allow_pickle=True)
                feature_info[npy_file.name] = {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "size_mb": round(npy_file.stat().st_size / 1024 / 1024, 3),
                }

                # 内容の詳細（小さなファイルのみ）
                if data.size < 10000:
                    if data.dtype.kind in ["U", "S", "O"]:  # 文字列やオブジェクト
                        feature_info[npy_file.name]["sample_values"] = data.tolist()[:10]
                    else:
                        feature_info[npy_file.name]["statistics"] = {
                            "min": float(np.min(data)) if data.size > 0 else None,
                            "max": float(np.max(data)) if data.size > 0 else None,
                            "mean": float(np.mean(data)) if data.size > 0 else None,
                            "std": float(np.std(data)) if data.size > 0 else None,
                        }

            except Exception as e:
                feature_info[npy_file.name] = {"error": str(e)}

        return feature_info

    def load_performance_metrics(self) -> dict[str, Any]:
        """パフォーマンス指標を読み込み"""
        metrics = {}

        # JSONファイルを探す
        for json_file in self.dataset_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                metrics[json_file.name] = data
            except Exception as e:
                metrics[json_file.name] = {"error": str(e)}

        return metrics

    def get_file_sizes(self) -> dict[str, int]:
        """ファイルサイズ情報を取得"""
        file_sizes = {}
        for file in self.dataset_path.iterdir():
            if file.is_file():
                file_sizes[file.name] = file.stat().st_size
        return file_sizes

    def analyze_dataset(self) -> ModelAnalysis:
        """データセット全体を解析"""
        print(f"Analyzing dataset: {self.dataset_name}")

        errors = []
        model_files = []
        model_architecture = {}

        # PyTorchモデルファイルを検索・解析
        for model_ext in ["*.pt", "*.pth"]:
            for model_file in self.dataset_path.glob(model_ext):
                model_files.append(model_file.name)
                print(f"  Analyzing model: {model_file.name}")

                try:
                    analysis = self.analyze_pytorch_model(model_file)
                    model_architecture[model_file.name] = analysis
                except Exception as e:
                    error_msg = f"Error analyzing {model_file.name}: {e!s}"
                    errors.append(error_msg)
                    print(f"    {error_msg}")

        # その他の情報を読み込み
        try:
            feature_info = self.load_feature_info()
        except Exception as e:
            feature_info = {}
            errors.append(f"Error loading feature info: {e!s}")

        try:
            preprocessing_info = self.load_preprocessing_info()
        except Exception as e:
            preprocessing_info = {}
            errors.append(f"Error loading preprocessing info: {e!s}")

        try:
            performance_metrics = self.load_performance_metrics()
        except Exception as e:
            performance_metrics = {}
            errors.append(f"Error loading performance metrics: {e!s}")

        file_sizes = self.get_file_sizes()

        return ModelAnalysis(
            dataset_name=self.dataset_name,
            model_files=model_files,
            model_architecture=model_architecture,
            feature_info=feature_info,
            preprocessing_info=preprocessing_info,
            performance_metrics=performance_metrics,
            file_sizes=file_sizes,
            errors=errors,
        )


def analyze_all_datasets(base_path: str = "public_datasets") -> dict[str, ModelAnalysis]:
    """全データセットを解析"""
    base_path = Path(base_path)
    results = {}

    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            analyzer = ModelAnalyzer(dataset_dir)
            results[dataset_dir.name] = analyzer.analyze_dataset()

    return results


if __name__ == "__main__":
    print("Starting model analysis...")
    results = analyze_all_datasets()

    print(f"\nAnalysis completed for {len(results)} datasets.")

    # 簡単なサマリーを表示
    for dataset_name, analysis in results.items():
        print(f"\n=== {dataset_name} ===")
        print(f"Model files: {len(analysis.model_files)}")
        print(f"Feature files: {len(analysis.feature_info)}")
        print(f"Preprocessing files: {len(analysis.preprocessing_info)}")
        if analysis.errors:
            print(f"Errors: {len(analysis.errors)}")

    # 結果をJSONで保存
    output_file = Path("outputs/claude/model_analysis_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # DataClassをJSONシリアライズ可能な形式に変換
    json_results = {}
    for dataset_name, analysis in results.items():
        json_results[dataset_name] = {
            "dataset_name": analysis.dataset_name,
            "model_files": analysis.model_files,
            "model_architecture": analysis.model_architecture,
            "feature_info": analysis.feature_info,
            "preprocessing_info": analysis.preprocessing_info,
            "performance_metrics": analysis.performance_metrics,
            "file_sizes": analysis.file_sizes,
            "errors": analysis.errors,
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")
