#!/usr/bin/env python3
"""
CMI特徴量詳細分析スクリプト

各データセットの特徴量構成を詳細に解析し、
センサータイプ、統計的特徴量、前処理方法の違いを明確にします。
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class CMIFeatureAnalyzer:
    """CMI特徴量の詳細解析クラス"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.datasets = [
            "cmi-imu-model",
            "cmi-fullfeats-models",
            "s-offline-0-8254-15fold",
            "cmi-imu-only-models",
            "b-offline-0-8855-specialprocess",
        ]

    def analyze_feature_columns(self, dataset_path: Path) -> dict[str, Any]:
        """特徴量カラムの詳細解析"""
        feature_cols_path = dataset_path / "feature_cols.npy"

        if not feature_cols_path.exists():
            return {"error": "feature_cols.npy not found"}

        try:
            feature_cols = np.load(feature_cols_path, allow_pickle=True)

            # リスト形式に変換
            if isinstance(feature_cols, np.ndarray):
                cols_list = feature_cols.tolist()
            else:
                cols_list = list(feature_cols)

            analysis = {
                "total_features": len(cols_list),
                "feature_samples": cols_list[:20],  # 最初の20個
                "sensor_types": self._classify_sensor_types(cols_list),
                "statistical_features": self._classify_statistical_features(cols_list),
                "coordinate_systems": self._analyze_coordinate_systems(cols_list),
                "time_domain_features": self._classify_time_domain_features(cols_list),
                "frequency_domain_features": self._classify_frequency_domain_features(cols_list),
            }

            return analysis

        except Exception as e:
            return {"error": str(e)}

    def _classify_sensor_types(self, features: list[str]) -> dict[str, int]:
        """センサータイプの分類"""
        sensor_counts = defaultdict(int)

        for feature in features:
            feature_lower = feature.lower()

            if any(acc in feature_lower for acc in ["accel", "acc_"]):
                sensor_counts["accelerometer"] += 1
            elif "gyro" in feature_lower:
                sensor_counts["gyroscope"] += 1
            elif "mag" in feature_lower or "magnetic" in feature_lower:
                sensor_counts["magnetometer"] += 1
            elif "linear" in feature_lower:
                sensor_counts["linear_acceleration"] += 1
            elif "gravity" in feature_lower:
                sensor_counts["gravity"] += 1
            elif "rotation" in feature_lower or "orient" in feature_lower:
                sensor_counts["rotation_vector"] += 1
            else:
                sensor_counts["other"] += 1

        return dict(sensor_counts)

    def _classify_statistical_features(self, features: list[str]) -> dict[str, int]:
        """統計的特徴量の分類"""
        stat_counts = defaultdict(int)

        stat_keywords = {
            "mean": ["mean", "avg"],
            "std": ["std", "stddev"],
            "var": ["var", "variance"],
            "min": ["min"],
            "max": ["max"],
            "median": ["median"],
            "quantile": ["q25", "q75", "quantile", "percentile"],
            "skewness": ["skew"],
            "kurtosis": ["kurt"],
            "energy": ["energy", "power"],
            "entropy": ["entropy"],
            "correlation": ["corr"],
            "covariance": ["cov"],
        }

        for feature in features:
            feature_lower = feature.lower()
            found = False

            for stat_type, keywords in stat_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    stat_counts[stat_type] += 1
                    found = True
                    break

            if not found:
                stat_counts["raw_or_other"] += 1

        return dict(stat_counts)

    def _analyze_coordinate_systems(self, features: list[str]) -> dict[str, int]:
        """座標系の分析"""
        coord_counts = defaultdict(int)

        for feature in features:
            feature_lower = feature.lower()

            if "_x" in feature_lower or "x_" in feature_lower:
                coord_counts["x_axis"] += 1
            elif "_y" in feature_lower or "y_" in feature_lower:
                coord_counts["y_axis"] += 1
            elif "_z" in feature_lower or "z_" in feature_lower:
                coord_counts["z_axis"] += 1
            elif "magnitude" in feature_lower or "mag" in feature_lower:
                coord_counts["magnitude"] += 1
            else:
                coord_counts["other"] += 1

        return dict(coord_counts)

    def _classify_time_domain_features(self, features: list[str]) -> dict[str, int]:
        """時間領域特徴量の分類"""
        time_counts = defaultdict(int)

        time_keywords = {
            "window_stats": ["window", "rolling"],
            "lag_features": ["lag", "shift"],
            "trend": ["trend", "slope"],
            "peak": ["peak", "valley"],
            "crossing": ["cross", "zero"],
            "duration": ["duration", "length"],
        }

        for feature in features:
            feature_lower = feature.lower()

            for time_type, keywords in time_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    time_counts[time_type] += 1
                    break

        return dict(time_counts)

    def _classify_frequency_domain_features(self, features: list[str]) -> dict[str, int]:
        """周波数領域特徴量の分類"""
        freq_counts = defaultdict(int)

        freq_keywords = {
            "fft": ["fft", "fourier"],
            "spectral": ["spectral", "spectrum"],
            "frequency": ["freq", "hz"],
            "wavelet": ["wavelet", "dwt"],
            "periodogram": ["psd", "periodogram"],
        }

        for feature in features:
            feature_lower = feature.lower()

            for freq_type, keywords in freq_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    freq_counts[freq_type] += 1
                    break

        return dict(freq_counts)

    def analyze_scaler_info(self, dataset_path: Path) -> dict[str, Any]:
        """スケーラー情報の詳細解析"""
        scaler_analysis = {}

        # 通常のscaler.pkl
        scaler_path = dataset_path / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

                scaler_info = {
                    "type": str(type(scaler).__name__),
                    "n_features": getattr(scaler, "n_features_in_", None),
                }

                # StandardScalerの場合
                if hasattr(scaler, "mean_") and scaler.mean_ is not None:
                    scaler_info["mean_stats"] = {
                        "min": float(np.min(scaler.mean_)),
                        "max": float(np.max(scaler.mean_)),
                        "mean": float(np.mean(scaler.mean_)),
                        "std": float(np.std(scaler.mean_)),
                    }

                if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                    scaler_info["scale_stats"] = {
                        "min": float(np.min(scaler.scale_)),
                        "max": float(np.max(scaler.scale_)),
                        "mean": float(np.mean(scaler.scale_)),
                        "std": float(np.std(scaler.scale_)),
                    }

                # MinMaxScalerの場合
                if hasattr(scaler, "data_min_") and scaler.data_min_ is not None:
                    scaler_info["data_range"] = {
                        "min_value": float(np.min(scaler.data_min_)),
                        "max_value": float(np.max(scaler.data_max_)),
                        "range_span": float(np.mean(scaler.data_max_ - scaler.data_min_)),
                    }

                scaler_analysis["main_scaler"] = scaler_info

            except Exception as e:
                scaler_analysis["main_scaler"] = {"error": str(e)}

        # static_scaler.pkl
        static_scaler_path = dataset_path / "static_scaler.pkl"
        if static_scaler_path.exists():
            try:
                with open(static_scaler_path, "rb") as f:
                    static_scaler = pickle.load(f)

                scaler_analysis["static_scaler"] = {
                    "type": str(type(static_scaler).__name__),
                    "n_features": getattr(static_scaler, "n_features_in_", None),
                }

            except Exception as e:
                scaler_analysis["static_scaler"] = {"error": str(e)}

        return scaler_analysis

    def analyze_gesture_classes(self, dataset_path: Path) -> dict[str, Any]:
        """ジェスチャークラス情報の解析"""
        gesture_path = dataset_path / "gesture_classes.npy"

        if not gesture_path.exists():
            return {"error": "gesture_classes.npy not found"}

        try:
            gesture_classes = np.load(gesture_path, allow_pickle=True)

            if isinstance(gesture_classes, np.ndarray):
                classes_list = gesture_classes.tolist()
            else:
                classes_list = list(gesture_classes)

            return {
                "num_classes": len(classes_list),
                "classes": classes_list,
                "class_distribution": Counter(classes_list) if len(classes_list) < 100 else "Too many classes to show",
            }

        except Exception as e:
            return {"error": str(e)}

    def run_comprehensive_feature_analysis(self) -> dict[str, Any]:
        """包括的特徴量解析の実行"""
        results = {}

        print("=== CMI特徴量詳細解析開始 ===\\n")

        for dataset in self.datasets:
            dataset_path = self.data_root / dataset

            if not dataset_path.exists():
                print(f"Warning: {dataset} not found")
                continue

            print(f"Analyzing features for: {dataset}")

            dataset_analysis = {
                "feature_columns": self.analyze_feature_columns(dataset_path),
                "scaler_info": self.analyze_scaler_info(dataset_path),
                "gesture_classes": self.analyze_gesture_classes(dataset_path),
            }

            results[dataset] = dataset_analysis

        return results

    def generate_feature_comparison_report(self, results: dict[str, Any]) -> str:
        """特徴量比較レポートの生成"""
        report = []
        report.append("# CMI特徴量詳細比較レポート\\n")

        # 概要
        report.append("## 1. 特徴量数比較")
        for dataset, data in results.items():
            feature_count = data["feature_columns"].get("total_features", "N/A")
            report.append(f"- **{dataset}**: {feature_count} 特徴量")

        # センサータイプ比較
        report.append("\\n## 2. センサータイプ分布")
        for dataset, data in results.items():
            report.append(f"\\n### {dataset}")
            sensor_types = data["feature_columns"].get("sensor_types", {})
            for sensor, count in sensor_types.items():
                report.append(f"- {sensor}: {count}")

        # 統計特徴量比較
        report.append("\\n## 3. 統計特徴量分布")
        for dataset, data in results.items():
            report.append(f"\\n### {dataset}")
            stats = data["feature_columns"].get("statistical_features", {})
            for stat_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {stat_type}: {count}")

        # スケーラー情報
        report.append("\\n## 4. 前処理スケーラー情報")
        for dataset, data in results.items():
            report.append(f"\\n### {dataset}")
            scaler_info = data.get("scaler_info", {})

            if "main_scaler" in scaler_info:
                main_scaler = scaler_info["main_scaler"]
                if "type" in main_scaler:
                    report.append(f"- メインスケーラー: {main_scaler['type']}")
                    report.append(f"- 特徴量数: {main_scaler.get('n_features', 'N/A')}")

            if "static_scaler" in scaler_info:
                static_scaler = scaler_info["static_scaler"]
                if "type" in static_scaler:
                    report.append(f"- 静的スケーラー: {static_scaler['type']}")

        # ジェスチャークラス
        report.append("\\n## 5. ジェスチャークラス情報")
        for dataset, data in results.items():
            gesture_info = data.get("gesture_classes", {})
            if "num_classes" in gesture_info:
                report.append(f"- **{dataset}**: {gesture_info['num_classes']} クラス")

        return "\\n".join(report)


def main():
    """メイン実行"""
    data_root = (
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/data/cmi_models"
    )

    analyzer = CMIFeatureAnalyzer(data_root)

    # 解析実行
    results = analyzer.run_comprehensive_feature_analysis()

    # レポート生成
    report = analyzer.generate_feature_comparison_report(results)

    # 結果保存
    output_dir = Path(
        "/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/outputs/claude"
    )

    # 詳細結果
    with open(output_dir / "cmi_feature_analysis_detailed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # レポート
    with open(output_dir / "cmi_feature_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\\n=== 特徴量解析完了 ===")
    print(f"詳細結果: {output_dir / 'cmi_feature_analysis_detailed.json'}")
    print(f"レポート: {output_dir / 'cmi_feature_comparison_report.md'}")


if __name__ == "__main__":
    main()
