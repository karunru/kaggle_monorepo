#!/usr/bin/env python3
"""
Collision Detection Dataset Structure Analyzer

このスクリプトは、KDXF Collision Detection データセットの基本構造を解析します。
メモリ効率を考慮して、大きなCSVファイルを段階的に読み込んで分析を行います。
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from codes.src.utils.logger import create_logger


class CollisionDataAnalyzer:
    """Collision Detection データセット解析クラス"""

    def __init__(self, data_dir: str | Path):
        """
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train_data.csv"
        self.test_path = self.data_dir / "test_data.csv"
        self.logger = create_logger(__name__)

    def get_file_info(self) -> dict[str, Any]:
        """ファイル基本情報を取得"""
        info = {}

        for name, path in [("train", self.train_path), ("test", self.test_path)]:
            if path.exists():
                stat = path.stat()
                info[name] = {"exists": True, "size_mb": stat.st_size / (1024 * 1024), "path": str(path)}
            else:
                info[name] = {"exists": False, "size_mb": 0, "path": str(path)}

        return info

    def analyze_csv_structure(self, file_path: Path, sample_rows: int = 1000) -> dict[str, Any]:
        """CSV構造を解析（サンプルデータで効率的に）"""
        self.logger.info(f"Analyzing structure of {file_path.name}")

        try:
            # まずヘッダーのみ読み込み
            header_df = pd.read_csv(file_path, nrows=0)
            columns = header_df.columns.tolist()

            # サンプルデータで型推定
            sample_df = pd.read_csv(file_path, nrows=sample_rows)

            # 全行数をカウント（効率的な方法）
            with open(file_path) as f:
                total_rows = sum(1 for line in f) - 1  # ヘッダー除く

            # データ型情報
            dtypes_info = {}
            for col in columns:
                dtypes_info[col] = {
                    "dtype": str(sample_df[col].dtype),
                    "null_count": sample_df[col].isnull().sum(),
                    "unique_count": sample_df[col].nunique(),
                    "sample_values": sample_df[col].dropna().head(5).tolist(),
                }

            return {
                "total_rows": total_rows,
                "total_columns": len(columns),
                "columns": columns,
                "dtypes_info": dtypes_info,
                "memory_usage_mb": sample_df.memory_usage(deep=True).sum() / (1024 * 1024),
                "estimated_full_memory_mb": (sample_df.memory_usage(deep=True).sum() / sample_rows * total_rows)
                / (1024 * 1024),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return {"error": str(e)}

    def get_basic_statistics(
        self, file_path: Path, numeric_only: bool = True, chunk_size: int = 10000
    ) -> dict[str, Any]:
        """基本統計量をチャンク処理で取得"""
        self.logger.info(f"Computing basic statistics for {file_path.name}")

        stats_accumulator = {}
        total_rows = 0

        try:
            # チャンクごとに処理
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                if numeric_only:
                    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                    chunk = chunk[numeric_cols]

                total_rows += len(chunk)

                if chunk_idx == 0:
                    # 初回：統計情報初期化
                    stats_accumulator = {
                        "count": chunk.count(),
                        "sum": chunk.sum(),
                        "sum_sq": (chunk**2).sum(),
                        "min": chunk.min(),
                        "max": chunk.max(),
                        "columns": chunk.columns.tolist(),
                    }
                else:
                    # 累積統計更新
                    stats_accumulator["count"] += chunk.count()
                    stats_accumulator["sum"] += chunk.sum()
                    stats_accumulator["sum_sq"] += (chunk**2).sum()
                    stats_accumulator["min"] = pd.concat([stats_accumulator["min"], chunk.min()], axis=1).min(axis=1)
                    stats_accumulator["max"] = pd.concat([stats_accumulator["max"], chunk.max()], axis=1).max(axis=1)

                if chunk_idx % 10 == 0:
                    self.logger.info(f"Processed {chunk_idx + 1} chunks ({total_rows:,} rows)")

            # 最終統計計算
            count = stats_accumulator["count"]
            mean = stats_accumulator["sum"] / count
            variance = (stats_accumulator["sum_sq"] / count) - (mean**2)
            std = np.sqrt(variance)

            result = {}
            for col in stats_accumulator["columns"]:
                result[col] = {
                    "count": int(count[col]),
                    "mean": float(mean[col]) if not pd.isna(mean[col]) else None,
                    "std": float(std[col]) if not pd.isna(std[col]) else None,
                    "min": float(stats_accumulator["min"][col]) if not pd.isna(stats_accumulator["min"][col]) else None,
                    "max": float(stats_accumulator["max"][col]) if not pd.isna(stats_accumulator["max"][col]) else None,
                }

            return {"total_rows": total_rows, "statistics": result}

        except Exception as e:
            self.logger.error(f"Error computing statistics for {file_path}: {e}")
            return {"error": str(e)}

    def detect_time_series_structure(
        self, file_path: Path, time_columns: list = None, sample_size: int = 10000
    ) -> dict[str, Any]:
        """時系列構造を検出"""
        self.logger.info(f"Detecting time series structure in {file_path.name}")

        try:
            # サンプルデータで時系列構造を調査
            sample_df = pd.read_csv(file_path, nrows=sample_size)

            # 時間関連カラムを自動検出
            if time_columns is None:
                time_columns = []
                for col in sample_df.columns:
                    col_lower = col.lower()
                    if any(
                        keyword in col_lower for keyword in ["time", "timestamp", "date", "step", "frame", "sequence"]
                    ):
                        time_columns.append(col)

            time_info = {}

            for col in time_columns:
                if col in sample_df.columns:
                    try:
                        if sample_df[col].dtype == "object":
                            # 文字列の場合、日時解析を試行
                            parsed_times = pd.to_datetime(sample_df[col], errors="coerce")
                            if not parsed_times.isna().all():
                                time_info[col] = {
                                    "type": "datetime",
                                    "valid_parse_rate": (1 - parsed_times.isna().mean()),
                                    "time_range": {"start": str(parsed_times.min()), "end": str(parsed_times.max())},
                                    "sample_values": sample_df[col].head(5).tolist(),
                                }
                        else:
                            # 数値の場合（タイムステップ等）
                            time_info[col] = {
                                "type": "numeric",
                                "range": {"min": float(sample_df[col].min()), "max": float(sample_df[col].max())},
                                "is_sequential": self._check_sequential(sample_df[col]),
                                "sample_values": sample_df[col].head(10).tolist(),
                            }
                    except Exception as e:
                        time_info[col] = {"error": str(e)}

            # シーケンス構造の検出
            sequence_info = self._detect_sequence_structure(sample_df)

            return {
                "time_columns": time_columns,
                "time_info": time_info,
                "sequence_info": sequence_info,
                "total_sample_rows": len(sample_df),
            }

        except Exception as e:
            self.logger.error(f"Error detecting time series structure: {e}")
            return {"error": str(e)}

    def _check_sequential(self, series: pd.Series) -> dict[str, Any]:
        """数値系列の連続性をチェック"""
        try:
            diff = series.diff().dropna()

            return {
                "is_monotonic": series.is_monotonic_increasing,
                "mean_diff": float(diff.mean()) if len(diff) > 0 else None,
                "std_diff": float(diff.std()) if len(diff) > 0 else None,
                "constant_step": bool(diff.nunique() == 1) if len(diff) > 0 else False,
            }
        except Exception:
            return {"error": "Unable to check sequential pattern"}

    def _detect_sequence_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """シーケンス構造を検出"""
        try:
            # ID関連カラムの検出
            id_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ["id", "user", "session", "trial", "subject"]):
                    id_columns.append(col)

            sequence_info = {"id_columns": id_columns}

            # 各IDカラムについて、レコード数分布を調査
            for col in id_columns[:3]:  # 最大3列まで
                if col in df.columns:
                    counts = df[col].value_counts()
                    sequence_info[f"{col}_stats"] = {
                        "unique_count": len(counts),
                        "record_counts": {
                            "min": int(counts.min()),
                            "max": int(counts.max()),
                            "mean": float(counts.mean()),
                            "median": float(counts.median()),
                        },
                        "sample_ids": counts.head(5).index.tolist(),
                    }

            return sequence_info

        except Exception as e:
            return {"error": str(e)}

    def run_full_analysis(self) -> dict[str, Any]:
        """完全な構造解析を実行"""
        self.logger.info("Starting full collision detection dataset analysis")

        results = {}
        import time

        start_time = time.time()

        # ファイル基本情報
        results["file_info"] = self.get_file_info()

        # 各ファイルの構造解析
        for dataset_name in ["train", "test"]:
            if results["file_info"][dataset_name]["exists"]:
                file_path = self.train_path if dataset_name == "train" else self.test_path

                results[f"{dataset_name}_structure"] = self.analyze_csv_structure(file_path)
                results[f"{dataset_name}_statistics"] = self.get_basic_statistics(file_path)
                results[f"{dataset_name}_timeseries"] = self.detect_time_series_structure(file_path)

        results["analysis_duration"] = time.time() - start_time

        return results


def main():
    """メイン実行関数"""

    # データディレクトリパス
    data_dir = Path(__file__).parents[3] / "public_datasets" / "kdxf-collisiondetect"

    # 出力ディレクトリ
    output_dir = Path(__file__).parents[3] / "outputs" / "claude"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析実行
    analyzer = CollisionDataAnalyzer(data_dir)
    results = analyzer.run_full_analysis()

    # 結果を保存（NumpyオブジェクトをJSON互換に変換）
    import json

    def convert_numpy_types(obj):
        """NumpyデータタイプをPython基本型に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # NumpyタイプをJSONシリアライズ可能に変換
    results_converted = convert_numpy_types(results)

    output_path = output_dir / "collision_dataset_structure_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)

    # 結果サマリー表示
    print("=" * 60)
    print("Collision Detection Dataset Structure Analysis Results")
    print("=" * 60)

    if "file_info" in results:
        for dataset_name in ["train", "test"]:
            info = results["file_info"][dataset_name]
            if info["exists"]:
                print(f"\n{dataset_name.upper()} Dataset:")
                print(f"  File size: {info['size_mb']:.1f} MB")

                if f"{dataset_name}_structure" in results:
                    struct = results[f"{dataset_name}_structure"]
                    if "error" not in struct:
                        print(f"  Rows: {struct['total_rows']:,}")
                        print(f"  Columns: {struct['total_columns']}")
                        print(f"  Estimated memory: {struct.get('estimated_full_memory_mb', 0):.1f} MB")

                        # カラム名をいくつか表示
                        columns = struct.get("columns", [])
                        print(f"  Sample columns: {', '.join(columns[:10])}")
                        if len(columns) > 10:
                            print(f"    ... and {len(columns) - 10} more columns")

    print(f"\nAnalysis completed in {results.get('analysis_duration', 0):.2f} seconds")
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
