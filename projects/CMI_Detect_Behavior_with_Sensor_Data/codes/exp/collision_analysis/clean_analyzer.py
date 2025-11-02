#!/usr/bin/env python3
"""
Clean Collision Detection Dataset Analyzer

KDXF Collision Detection データセットのクリーンな分析スクリプト
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from codes.src.utils.logger import create_logger


class CleanCollisionAnalyzer:
    """シンプルで確実なCollision Detection データセット解析クラス"""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train_data.csv"
        self.test_path = self.data_dir / "test_data.csv"
        self.logger = create_logger(__name__)

    def analyze_sensor_types(self, columns: list) -> dict[str, Any]:
        """センサー種別の分類"""
        sensor_types = {
            "acceleration": [col for col in columns if "acc" in col.lower()],
            "rotation": [col for col in columns if "rot" in col.lower()],
            "thermal": [col for col in columns if "thm" in col.lower()],
            "time_of_flight": [col for col in columns if "tof" in col.lower()],
            "metadata": [col for col in columns if col in ["row_id", "sequence_id", "risk_score"]],
            "other": [],
        }

        # otherカテゴリを計算
        classified = set()
        for sensor_list in sensor_types.values():
            classified.update(sensor_list)
        sensor_types["other"] = [col for col in columns if col not in classified]

        return {
            "sensor_types": sensor_types,
            "sensor_counts": {k: len(v) for k, v in sensor_types.items()},
            "total_sensors": len(columns),
        }

    def analyze_target_variable(self, sample_size: int = 50000) -> dict[str, Any]:
        """risk_score の分析"""
        self.logger.info("Analyzing target variable (risk_score)")

        try:
            df = pd.read_csv(self.train_path, nrows=sample_size)

            if "risk_score" not in df.columns:
                return {"error": "risk_score column not found"}

            risk_col = df["risk_score"]

            # 基本統計
            basic_stats = {
                "count": len(risk_col),
                "mean": float(risk_col.mean()),
                "std": float(risk_col.std()),
                "min": float(risk_col.min()),
                "max": float(risk_col.max()),
                "median": float(risk_col.median()),
                "unique_values": int(risk_col.nunique()),
            }

            # 分布情報
            value_counts = risk_col.value_counts().head(10)
            distribution = {
                "top_values": {str(k): int(v) for k, v in value_counts.items()},
                "positive_samples": int((risk_col > 0).sum()),
                "zero_samples": int((risk_col == 0).sum()),
                "negative_samples": int((risk_col < 0).sum()),
            }

            return {"basic_stats": basic_stats, "distribution": distribution, "sample_size": len(df)}

        except Exception as e:
            self.logger.error(f"Error in target analysis: {e}")
            return {"error": str(e)}

    def analyze_sequences(self, sample_size: int = 30000) -> dict[str, Any]:
        """シーケンス特性の分析"""
        self.logger.info("Analyzing sequence characteristics")

        try:
            df = pd.read_csv(self.train_path, nrows=sample_size)

            if "sequence_id" not in df.columns:
                return {"error": "sequence_id not found"}

            # シーケンス統計
            seq_lengths = df["sequence_id"].value_counts()

            length_stats = {
                "total_sequences": len(seq_lengths),
                "avg_length": float(seq_lengths.mean()),
                "median_length": float(seq_lengths.median()),
                "min_length": int(seq_lengths.min()),
                "max_length": int(seq_lengths.max()),
                "std_length": float(seq_lengths.std()),
            }

            # 長さ分布（上位10位）
            length_distribution = seq_lengths.head(10).to_dict()
            length_distribution = {str(k): int(v) for k, v in length_distribution.items()}

            return {"length_stats": length_stats, "length_distribution": length_distribution, "sample_size": len(df)}

        except Exception as e:
            self.logger.error(f"Error in sequence analysis: {e}")
            return {"error": str(e)}

    def analyze_sensor_patterns(self, sample_size: int = 10000) -> dict[str, Any]:
        """センサーデータパターンの分析"""
        self.logger.info("Analyzing sensor data patterns")

        try:
            df = pd.read_csv(self.train_path, nrows=sample_size)

            # センサー種別取得
            sensor_analysis = self.analyze_sensor_types(df.columns.tolist())
            sensor_types = sensor_analysis["sensor_types"]

            pattern_results = {}

            for sensor_type, columns in sensor_types.items():
                if not columns or sensor_type == "metadata":
                    continue

                # 最初の3カラムのみ分析（メモリとプロセス効率のため）
                sample_cols = columns[:3]
                sensor_data = df[sample_cols]

                stats = {}
                for col in sample_cols:
                    col_data = sensor_data[col]
                    stats[col] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "missing": int(col_data.isnull().sum()),
                        "zeros": int((col_data == 0).sum()),
                        "negative_ones": int((col_data == -1).sum()),
                    }

                pattern_results[sensor_type] = {
                    "total_columns": len(columns),
                    "analyzed_columns": sample_cols,
                    "column_stats": stats,
                }

            return {"pattern_analysis": pattern_results, "sample_size": sample_size}

        except Exception as e:
            self.logger.error(f"Error in sensor pattern analysis: {e}")
            return {"error": str(e)}

    def compare_datasets(self) -> dict[str, Any]:
        """train と test の比較"""
        self.logger.info("Comparing train and test datasets")

        comparison = {}

        for dataset_name, file_path in [("train", self.train_path), ("test", self.test_path)]:
            try:
                # ファイルサイズ
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB

                # 基本構造
                sample_df = pd.read_csv(file_path, nrows=1000)

                # 行数カウント（効率的）
                with open(file_path) as f:
                    total_rows = sum(1 for _ in f) - 1  # ヘッダー除く

                comparison[dataset_name] = {
                    "file_size_mb": float(file_size),
                    "total_rows": total_rows,
                    "total_columns": len(sample_df.columns),
                    "columns": sample_df.columns.tolist(),
                    "has_risk_score": "risk_score" in sample_df.columns,
                }

            except Exception as e:
                comparison[dataset_name] = {"error": str(e)}

        return comparison

    def generate_comprehensive_analysis(self) -> dict[str, Any]:
        """包括的分析の実行"""
        self.logger.info("Starting comprehensive collision detection analysis")

        results = {
            "analysis_metadata": {
                "dataset_name": "KDXF Collision Detection",
                "analysis_type": "comprehensive_structure_analysis",
            }
        }

        # 各分析を順次実行
        try:
            results["dataset_comparison"] = self.compare_datasets()

            # trainデータがある場合のみ詳細分析
            if self.train_path.exists():
                # 基本カラム情報から
                sample_df = pd.read_csv(self.train_path, nrows=100)
                columns = sample_df.columns.tolist()

                results["sensor_analysis"] = self.analyze_sensor_types(columns)
                results["target_analysis"] = self.analyze_target_variable()
                results["sequence_analysis"] = self.analyze_sequences()
                results["pattern_analysis"] = self.analyze_sensor_patterns()

            results["status"] = "completed_successfully"

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            results["status"] = "error"
            results["error_message"] = str(e)

        return results

    def generate_cmi_comparison(self) -> dict[str, Any]:
        """CMIデータセットとの比較分析"""
        return {
            "collision_dataset_characteristics": {
                "domain": "collision_detection",
                "data_type": "multi_sensor_time_series",
                "target_variable": "continuous_risk_score",
                "sensor_types": ["acceleration", "rotation", "thermal", "time_of_flight"],
                "multi_user_data": True,
                "temporal_structure": "sequence_based",
            },
            "cmi_dataset_characteristics": {
                "domain": "behavior_detection",
                "data_type": "sensor_time_series",
                "target_variable": "behavioral_classification",
                "sensor_types": ["accelerometer", "gyroscope", "etc"],
                "temporal_structure": "time_series_based",
            },
            "comparison_analysis": {
                "similarities": [
                    "Both involve sensor-based time series data",
                    "Both require temporal pattern recognition",
                    "Both can benefit from sequence modeling",
                    "Similar preprocessing requirements",
                ],
                "differences": [
                    "Collision risk (continuous) vs behavior classification (categorical)",
                    "Multi-user collision scenarios vs individual behavior",
                    "Safety-critical vs behavioral analysis",
                    "Different sensor modalities and fusion requirements",
                ],
                "transferable_techniques": [
                    "Time series feature engineering",
                    "Sequence-to-sequence modeling",
                    "LSTM/Transformer architectures",
                    "Sensor data normalization",
                    "Cross-validation for time series",
                ],
                "feature_engineering_opportunities": [
                    "Statistical features (mean, std, skewness)",
                    "Temporal features (lag, rolling statistics)",
                    "Frequency domain features (FFT)",
                    "Sensor fusion features",
                    "Sequential pattern features",
                ],
            },
        }


def main():
    """メイン実行関数"""

    # パス設定
    project_root = Path(__file__).parents[3]
    data_dir = project_root / "public_datasets" / "kdxf-collisiondetect"
    output_dir = project_root / "outputs" / "claude"

    # 解析実行
    analyzer = CleanCollisionAnalyzer(data_dir)

    print("🚀 Starting Collision Detection Dataset Analysis...")
    comprehensive_results = analyzer.generate_comprehensive_analysis()

    print("🔍 Generating CMI comparison...")
    cmi_comparison = analyzer.generate_cmi_comparison()

    # 結果を統合
    final_results = {**comprehensive_results, "cmi_comparison": cmi_comparison}

    # 結果保存
    output_path = output_dir / "collision_dataset_clean_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # サマリー表示
    print("\n" + "=" * 80)
    print("COLLISION DETECTION DATASET - CLEAN ANALYSIS RESULTS")
    print("=" * 80)

    # データセット比較結果
    if "dataset_comparison" in final_results:
        comparison = final_results["dataset_comparison"]

        print("\n📊 DATASET OVERVIEW:")
        for dataset_name in ["train", "test"]:
            if dataset_name in comparison and "error" not in comparison[dataset_name]:
                info = comparison[dataset_name]
                print(
                    f"  {dataset_name.upper()}: {info['total_rows']:,} rows, {info['total_columns']} cols, {info['file_size_mb']:.1f} MB"
                )

    # センサー分析結果
    if "sensor_analysis" in final_results:
        sensor_info = final_results["sensor_analysis"]
        sensor_counts = sensor_info.get("sensor_counts", {})

        print("\n🎛️  SENSOR BREAKDOWN:")
        for sensor_type, count in sensor_counts.items():
            if count > 0:
                print(f"  {sensor_type.replace('_', ' ').title()}: {count} columns")

    # ターゲット変数結果
    if "target_analysis" in final_results and "error" not in final_results["target_analysis"]:
        target_info = final_results["target_analysis"]
        basic_stats = target_info.get("basic_stats", {})
        distribution = target_info.get("distribution", {})

        print("\n🎯 TARGET VARIABLE (risk_score):")
        print(f"  Range: {basic_stats.get('min', 'N/A'):.3f} to {basic_stats.get('max', 'N/A'):.3f}")
        print(f"  Mean: {basic_stats.get('mean', 'N/A'):.3f} ± {basic_stats.get('std', 'N/A'):.3f}")
        print(f"  Unique values: {basic_stats.get('unique_values', 'N/A')}")
        print(f"  Positive samples: {distribution.get('positive_samples', 'N/A')}")
        print(f"  Zero samples: {distribution.get('zero_samples', 'N/A')}")

    # シーケンス分析結果
    if "sequence_analysis" in final_results and "error" not in final_results["sequence_analysis"]:
        seq_info = final_results["sequence_analysis"]
        length_stats = seq_info.get("length_stats", {})

        print("\n📈 SEQUENCE CHARACTERISTICS:")
        print(f"  Total sequences: {length_stats.get('total_sequences', 'N/A')}")
        print(f"  Avg sequence length: {length_stats.get('avg_length', 'N/A'):.1f}")
        print(f"  Length range: {length_stats.get('min_length', 'N/A')} to {length_stats.get('max_length', 'N/A')}")

    # CMI比較結果
    if "cmi_comparison" in final_results:
        comparison = final_results["cmi_comparison"]["comparison_analysis"]

        print("\n🔄 CMI DATASET COMPARISON HIGHLIGHTS:")
        similarities = comparison.get("similarities", [])
        differences = comparison.get("differences", [])

        print("  Key Similarities:")
        for sim in similarities[:2]:
            print(f"    • {sim}")

        print("  Key Differences:")
        for diff in differences[:2]:
            print(f"    • {diff}")

        print("  Transferable Techniques:")
        techniques = comparison.get("transferable_techniques", [])
        for tech in techniques[:3]:
            print(f"    • {tech}")

    print(f"\n💾 Complete results saved to: {output_path}")
    print("✅ Analysis completed successfully!")


if __name__ == "__main__":
    main()
