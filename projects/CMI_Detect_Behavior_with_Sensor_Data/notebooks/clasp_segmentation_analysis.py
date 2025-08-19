#!/usr/bin/env python3
"""
CLaSP Segmentation + Supervised Segmentation Analysis
CMIコンペティションのセンサーデータに対するセグメンテーション分析

教師なし（CLaSP）と教師あり（Window-based Classification）の両手法を実装し、
性能を比較する包括的な分析スクリプト。
"""

import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# sktime関連のインポート
try:
    from sktime.annotation.clasp import ClaSPSegmentation

    print("✅ sktime CLaSPモジュールのインポート成功")
except ImportError as e:
    print(f"❌ sktime CLaSPモジュールのインポート失敗: {e}")
    print("以下のコマンドでインストールしてください: uv add scikit-time")
    sys.exit(1)

# プロジェクトパスの設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

# 出力ディレクトリの作成
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 警告を非表示
warnings.filterwarnings("ignore")

# matplotlibの設定
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 12

print("=" * 60)
print("CLaSP + Supervised Segmentation Analysis")
print("=" * 60)

# =============================================================================
# IMU特徴量計算関数
# =============================================================================


def remove_gravity_from_acc_pl(df: pl.DataFrame) -> pl.DataFrame:
    """重力を除去した線形加速度を計算"""
    return df.with_columns(
        [
            (pl.col("acc_x") - 2 * (pl.col("rot_x") * pl.col("rot_z") - pl.col("rot_w") * pl.col("rot_y"))).alias(
                "linear_acc_x"
            ),
            (pl.col("acc_y") - 2 * (pl.col("rot_y") * pl.col("rot_z") + pl.col("rot_w") * pl.col("rot_x"))).alias(
                "linear_acc_y"
            ),
            (
                pl.col("acc_z")
                - (pl.col("rot_w") ** 2 - pl.col("rot_x") ** 2 - pl.col("rot_y") ** 2 + pl.col("rot_z") ** 2)
            ).alias("linear_acc_z"),
        ]
    )


def calculate_angular_velocity_from_quat_pl(df: pl.DataFrame) -> pl.DataFrame:
    """四元数から角速度を計算"""
    return df.with_columns(
        [
            (2 * (pl.col("rot_w") * pl.col("rot_x") + pl.col("rot_y") * pl.col("rot_z"))).alias("angular_vel_x"),
            (2 * (pl.col("rot_w") * pl.col("rot_y") - pl.col("rot_z") * pl.col("rot_x"))).alias("angular_vel_y"),
            (2 * (pl.col("rot_w") * pl.col("rot_z") + pl.col("rot_x") * pl.col("rot_y"))).alias("angular_vel_z"),
        ]
    )


def calculate_angular_distance_pl(df: pl.DataFrame) -> pl.DataFrame:
    """角距離を計算"""
    return df.with_columns(
        [
            (2 * pl.col("rot_w").clip(-1, 1).map_elements(lambda x: np.arccos(x), return_dtype=pl.Float64)).alias(
                "angular_distance"
            )
        ]
    )


def compute_imu_physics_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    19個のIMU物理特徴量を計算

    Args:
        df: IMUデータを含むPolars DataFrame

    Returns:
        物理特徴量が追加されたDataFrame
    """
    print("IMU物理特徴量を計算中...")

    # 基本的な大きさの計算
    df = df.with_columns(
        [
            (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2).sqrt().alias("acc_mag"),
            (2 * pl.col("rot_w").clip(-1, 1).map_elements(lambda x: np.arccos(x), return_dtype=pl.Float64)).alias(
                "rot_angle"
            ),
        ]
    )

    # Jerk（加速度の変化率）の計算
    df = df.with_columns(
        [pl.col("acc_mag").diff().abs().alias("acc_mag_jerk"), pl.col("rot_angle").diff().abs().alias("rot_angle_vel")]
    )

    # 重力除去
    df = remove_gravity_from_acc_pl(df)

    # 線形加速度の大きさとJerk
    df = df.with_columns(
        [
            (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
            .sqrt()
            .alias("linear_acc_mag")
        ]
    )

    df = df.with_columns([pl.col("linear_acc_mag").diff().abs().alias("linear_acc_mag_jerk")])

    # 角速度と角距離
    df = calculate_angular_velocity_from_quat_pl(df)
    df = calculate_angular_distance_pl(df)

    # NaNを0で埋める
    df = df.fill_nan(0).fill_null(0)

    print("✅ 19個のIMU物理特徴量計算完了")

    return df


# =============================================================================
# CLaSPSegmentationAnalyzer クラス
# =============================================================================


class CLaSPSegmentationAnalyzer:
    """
    CLaSP (Greedy Gaussian Segmentation)を使用した時系列セグメンテーション分析クラス
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def analyze_single_sequence(
        self, sequence_data: pd.DataFrame, sequence_id: str, n_cps: int = 2, auto_window_size: bool = True
    ) -> dict[str, Any]:
        """
        単一シーケンスのCLaSPセグメンテーション分析

        Args:
            sequence_data: シーケンスデータ
            sequence_id: シーケンスID
            n_cps: 検出する変化点数
            auto_window_size: ウィンドウサイズ自動検出

        Returns:
            分析結果の辞書
        """
        print(f"\nシーケンス {sequence_id} の分析開始...")

        if len(sequence_data) < 10:
            return {
                "sequence_id": sequence_id,
                "error": "シーケンス長が短すぎます",
                "change_points": {},
                "segments": {},
            }

        # 利用可能な特徴量のフィルタリング
        available_features = [col for col in self.feature_cols if col in sequence_data.columns]

        if len(available_features) == 0:
            return {
                "sequence_id": sequence_id,
                "error": "利用可能な特徴量がありません",
                "change_points": {},
                "segments": {},
            }

        print(f"  利用可能特徴量: {len(available_features)}")

        # ウィンドウサイズの決定
        if auto_window_size:
            window_size = min(max(len(sequence_data) // 4, 5), 20)
        else:
            window_size = 10

        change_points_by_feature = {}

        # 各特徴量に対してCLaSPを実行
        for feature in available_features:
            try:
                feature_data = sequence_data[feature].values

                if np.all(np.isnan(feature_data)) or len(np.unique(feature_data)) < 2:
                    continue

                # CLaSPセグメンテーション実行
                clasp = ClaSPSegmentation(period_length=window_size, n_cps=n_cps)
                change_points_raw = clasp.fit_predict(feature_data)

                # 型安全な変化点変換
                change_points_list = []
                if hasattr(change_points_raw, "values"):
                    raw_values = change_points_raw.values.flatten()
                elif hasattr(change_points_raw, "tolist"):
                    raw_values = change_points_raw.tolist()
                else:
                    raw_values = list(change_points_raw)

                for cp in raw_values:
                    try:
                        change_points_list.append(int(float(cp)))
                    except (ValueError, TypeError):
                        continue  # 無効な値をスキップ

                # 有効な変化点のみ保持
                valid_change_points = [cp for cp in change_points_list if 0 <= cp < len(sequence_data)]

                if len(valid_change_points) > 0:
                    change_points_by_feature[feature] = valid_change_points

            except Exception as e:
                print(f"  {feature}: エラー - {e}")
                continue

        # 変化点の統合（コンセンサス）
        all_change_points = []
        for cps in change_points_by_feature.values():
            all_change_points.extend(cps)

        # 変化点のクラスタリング
        consensus_change_points = self._clustering_based_consensus(
            all_change_points, threshold=max(1, len(sequence_data) * 0.05)
        )

        # セグメント情報の生成
        segments = self._create_segments(consensus_change_points, len(sequence_data))

        result = {
            "sequence_id": sequence_id,
            "sequence_length": len(sequence_data),
            "window_size": window_size,
            "n_features_analyzed": len(change_points_by_feature),
            "change_points": {"by_feature": change_points_by_feature, "consensus": consensus_change_points},
            "segments": segments,
        }

        print(f"  ✅ 完了: {len(consensus_change_points)}個の変化点検出")

        return result

    def _clustering_based_consensus(self, change_points: list[int], threshold: int) -> list[int]:
        """変化点のクラスタリングベースコンセンサス"""
        if not change_points:
            return []

        change_points = sorted(list(set(change_points)))
        clusters = []
        current_cluster = [change_points[0]]

        for cp in change_points[1:]:
            if cp - current_cluster[-1] <= threshold:
                current_cluster.append(cp)
            else:
                clusters.append(current_cluster)
                current_cluster = [cp]

        clusters.append(current_cluster)

        # 各クラスターの代表点（中央値）を取る
        consensus_points = [int(np.median(cluster)) for cluster in clusters]

        return sorted(consensus_points)

    def _create_segments(self, change_points: list[int], sequence_length: int) -> dict[str, Any]:
        """セグメント情報の作成"""
        if not change_points:
            return {"ranges": [(0, sequence_length)], "phase_labels": ["Segment_0"], "n_segments": 1}

        # セグメント範囲の計算
        segments = []
        phase_labels = []

        # 最初のセグメント
        segments.append((0, change_points[0]))
        phase_labels.append("Transition")

        # 中間のセグメント
        for i in range(len(change_points) - 1):
            segments.append((change_points[i], change_points[i + 1]))
            if i == 0 and len(change_points) > 1:
                phase_labels.append("Pause")
            else:
                phase_labels.append(f"Segment_{i + 1}")

        # 最後のセグメント
        segments.append((change_points[-1], sequence_length))
        if len(change_points) == 1:
            phase_labels.append("Gesture")
        elif len(change_points) == 2:
            phase_labels.append("Gesture")
        else:
            phase_labels.append("Final_Segment")

        return {"ranges": segments, "phase_labels": phase_labels, "n_segments": len(segments)}


# =============================================================================
# SupervisedSegmentationAnalyzer クラス
# =============================================================================


class SupervisedSegmentationAnalyzer:
    """
    教師ありWindow-based Segmentation分析クラス
    behaviorラベルを使用して学習し、より高精度なセグメンテーションを実現
    """

    def __init__(self, feature_cols: list[str], window_size: int = 20, stride: int = 10):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.stride = stride
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_importance_ = None

    def _extract_window_features(self, sequence_data: pd.DataFrame) -> np.ndarray:
        """時系列データからウィンドウベースの特徴量を抽出"""
        available_features = [col for col in self.feature_cols if col in sequence_data.columns]

        if len(available_features) == 0:
            raise ValueError("利用可能な特徴量がありません")

        feature_data = sequence_data[available_features].values
        window_features = []

        for i in range(0, len(feature_data) - self.window_size + 1, self.stride):
            window = feature_data[i : i + self.window_size]

            # 統計的特徴量を計算
            features = []

            for feature_idx in range(window.shape[1]):
                feature_values = window[:, feature_idx]

                # 基本統計量
                features.extend(
                    [
                        np.mean(feature_values),
                        np.std(feature_values),
                        np.min(feature_values),
                        np.max(feature_values),
                        np.median(feature_values),
                    ]
                )

                # 追加統計量
                features.extend(
                    [np.percentile(feature_values, 25), np.percentile(feature_values, 75), np.var(feature_values)]
                )

            # ウィンドウ全体の特徴量
            features.extend(
                [
                    np.mean(np.std(window, axis=0)),  # 平均標準偏差
                    np.mean(np.diff(window, axis=0).flatten()),  # 平均変化率
                    len(np.where(np.diff(feature_data[i : i + self.window_size, 0]) > 0)[0]),  # 上昇点数
                ]
            )

            window_features.append(features)

        return np.array(window_features)

    def _extract_window_labels(self, sequence_data: pd.DataFrame) -> list[str]:
        """ウィンドウごとのbehaviorラベルを抽出（多数決）"""
        if "behavior" not in sequence_data.columns:
            raise ValueError("behaviorカラムが見つかりません")

        behaviors = sequence_data["behavior"].values
        window_labels = []

        for i in range(0, len(behaviors) - self.window_size + 1, self.stride):
            window_behaviors = behaviors[i : i + self.window_size]

            # 多数決でウィンドウラベルを決定
            majority_label = Counter(window_behaviors).most_common(1)[0][0]
            window_labels.append(majority_label)

        return window_labels

    def fit(self, sequences_data: dict[str, pd.DataFrame]) -> "SupervisedSegmentationAnalyzer":
        """複数シーケンスのデータでモデルを学習"""
        print("教師ありセグメンテーションモデルを学習中...")
        print(f"ウィンドウサイズ: {self.window_size}, ストライド: {self.stride}")

        X_all = []
        y_all = []

        for seq_id, seq_data in sequences_data.items():
            try:
                # ウィンドウ特徴量の抽出
                X_windows = self._extract_window_features(seq_data)
                y_windows = self._extract_window_labels(seq_data)

                if len(X_windows) == len(y_windows) and len(X_windows) > 0:
                    X_all.append(X_windows)
                    y_all.extend(y_windows)
                    print(f"  {seq_id}: {len(X_windows)}個のウィンドウを抽出")

            except Exception as e:
                print(f"  {seq_id}: エラー - {e}")
                continue

        if len(X_all) == 0:
            raise ValueError("学習用データが抽出できませんでした")

        # データの結合
        X_combined = np.vstack(X_all)
        y_combined = np.array(y_all)

        print("\n学習データ準備完了:")
        print(f"  ウィンドウ数: {len(X_combined)}")
        print(f"  特徴量次元: {X_combined.shape[1]}")
        print(f"  ラベル分布: {Counter(y_combined)}")

        # ラベルエンコーディング
        y_encoded = self.label_encoder.fit_transform(y_combined)

        # モデル学習
        self.classifier.fit(X_combined, y_encoded)
        self.is_fitted = True

        # 特徴量重要度の保存
        self.feature_importance_ = self.classifier.feature_importances_

        # Cross-validation評価
        cv_scores = cross_val_score(
            self.classifier,
            X_combined,
            y_encoded,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1_macro",
        )

        print("\n交差検証結果:")
        print(f"  F1スコア (macro): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        return self

    def predict_sequence(self, sequence_data: pd.DataFrame, sequence_id: str) -> dict:
        """単一シーケンスのセグメンテーション予測"""
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。fit()を実行してください。")

        try:
            # ウィンドウ特徴量の抽出
            X_windows = self._extract_window_features(sequence_data)

            if len(X_windows) == 0:
                return {
                    "sequence_id": sequence_id,
                    "error": "ウィンドウ特徴量を抽出できませんでした",
                    "predictions": [],
                    "change_points": [],
                    "segments": {},
                }

            # 予測実行
            y_pred_encoded = self.classifier.predict(X_windows)
            y_pred_proba = self.classifier.predict_proba(X_windows)

            # ラベルデコーディング
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

            # 各ウィンドウの開始位置を計算
            window_positions = []
            for i in range(0, len(sequence_data) - self.window_size + 1, self.stride):
                window_positions.append(i + self.window_size // 2)  # ウィンドウ中央位置

            # セグメント境界の検出
            change_points = []
            segments = []

            if len(y_pred) > 1:
                current_phase = y_pred[0]
                start_pos = 0

                for i in range(1, len(y_pred)):
                    if y_pred[i] != current_phase:
                        # フェーズ変化を検出
                        change_point = window_positions[i]
                        change_points.append(change_point)

                        segments.append(
                            {
                                "start": start_pos,
                                "end": change_point,
                                "phase": current_phase,
                                "confidence": np.mean(y_pred_proba[start_pos:i].max(axis=1)),
                            }
                        )

                        current_phase = y_pred[i]
                        start_pos = i

                # 最後のセグメント
                segments.append(
                    {
                        "start": start_pos,
                        "end": len(sequence_data),
                        "phase": current_phase,
                        "confidence": np.mean(y_pred_proba[start_pos:].max(axis=1)),
                    }
                )

            result = {
                "sequence_id": sequence_id,
                "sequence_length": len(sequence_data),
                "n_windows": len(X_windows),
                "window_predictions": y_pred.tolist(),
                "window_probabilities": y_pred_proba.tolist(),
                "window_positions": window_positions,
                "change_points": change_points,
                "segments": segments,
                "phase_labels": [seg["phase"] for seg in segments],
            }

            print(f"シーケンス {sequence_id}: {len(change_points)}個の変化点を検出")
            print(f"  セグメント: {[seg['phase'] for seg in segments]}")

            return result

        except Exception as e:
            return {"sequence_id": sequence_id, "error": str(e), "predictions": [], "change_points": [], "segments": {}}


# =============================================================================
# SegmentationComparisonTool クラス
# =============================================================================


class SegmentationComparisonTool:
    """教師ありセグメンテーションとCLaSPの比較可視化ツール"""

    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_comparison(
        self,
        sequence_data: pd.DataFrame,
        clasp_result: dict,
        supervised_result: dict,
        features_to_plot: list[str] = None,
        figsize: tuple[int, int] = (20, 15),
    ) -> None:
        """CLaSPと教師ありセグメンテーションの比較可視化"""
        if features_to_plot is None:
            features_to_plot = ["acc_mag", "linear_acc_mag", "angular_vel_x"][:3]
            features_to_plot = [f for f in features_to_plot if f in sequence_data.columns]

        sequence_id = clasp_result.get("sequence_id", supervised_result.get("sequence_id", "Unknown"))
        time_points = np.arange(len(sequence_data))

        n_features = len(features_to_plot)
        fig, axes = plt.subplots(n_features + 2, 1, figsize=figsize, sharex=True)

        # 各特徴量のプロット
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]

            # 時系列データ
            ax.plot(time_points, sequence_data[feature], "b-", alpha=0.7, linewidth=1.5, label="IMU Data")

            # CLaSPの変化点
            clasp_cps = clasp_result.get("change_points", {}).get("consensus", [])
            for cp in clasp_cps:
                ax.axvline(
                    x=cp,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label="CLaSP" if cp == clasp_cps[0] else "",
                )

            # 教師ありの変化点
            supervised_cps = supervised_result.get("change_points", [])
            for cp in supervised_cps:
                ax.axvline(
                    x=cp,
                    color="green",
                    linestyle=":",
                    alpha=0.8,
                    linewidth=2,
                    label="Supervised" if cp == supervised_cps[0] else "",
                )

            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            if i == 0:
                ax.set_title(
                    f"Segmentation Comparison - Sequence {sequence_id}\n"
                    f"CLaSP: {len(clasp_cps)} change points, Supervised: {len(supervised_cps)} change points",
                    fontsize=14,
                    fontweight="bold",
                )

        # CLaSPセグメンテーション結果
        ax_clasp = axes[-2]
        clasp_segments = clasp_result.get("segments", {}).get("ranges", [])
        clasp_labels = clasp_result.get("segments", {}).get("phase_labels", [])

        for j, ((start, end), label) in enumerate(zip(clasp_segments, clasp_labels)):
            color = self.colors[j % len(self.colors)]
            ax_clasp.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.7, label=label)

        ax_clasp.set_ylabel("CLaSP\nSegments")
        ax_clasp.set_ylim(-0.5, 0.5)
        ax_clasp.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # 教師ありセグメンテーション結果
        ax_supervised = axes[-1]
        supervised_segments = supervised_result.get("segments", [])

        for j, segment in enumerate(supervised_segments):
            start = segment.get("start", 0)
            end = segment.get("end", len(sequence_data))
            phase = segment.get("phase", f"Segment_{j}")
            confidence = segment.get("confidence", 0.5)

            color = self.colors[j % len(self.colors)]
            alpha = 0.5 + 0.5 * confidence  # 確信度で透明度を調整
            ax_supervised.barh(
                0, end - start, left=start, height=0.5, color=color, alpha=alpha, label=f"{phase} ({confidence:.2f})"
            )

        ax_supervised.set_ylabel("Supervised\nSegments")
        ax_supervised.set_xlabel("Time Points")
        ax_supervised.set_ylim(-0.5, 0.5)
        ax_supervised.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Ground Truth（利用可能な場合）
        if "behavior" in sequence_data.columns:
            # Ground truthの表示をオーバーレイ
            unique_behaviors = sequence_data["behavior"].unique()
            behavior_map = {b: i for i, b in enumerate(unique_behaviors)}
            behavior_numeric = sequence_data["behavior"].map(behavior_map)

            # 最上段のサブプロットにGround Truthを追加
            ax_gt = axes[0].twinx()
            ax_gt.plot(time_points, behavior_numeric, "k-", linewidth=3, alpha=0.5, label="Ground Truth")
            ax_gt.set_ylabel("Ground Truth", color="black")
            ax_gt.set_yticks(range(len(unique_behaviors)))
            ax_gt.set_yticklabels(unique_behaviors)
            ax_gt.legend(loc="upper left")

        plt.tight_layout()
        
        # ファイル保存
        output_file = OUTPUT_DIR / f"segmentation_comparison_{sequence_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 比較プロットを保存: {output_file}")

    def compute_comparison_metrics(
        self, sequence_data: pd.DataFrame, clasp_result: dict, supervised_result: dict
    ) -> dict:
        """CLaSPと教師ありセグメンテーションの比較メトリクス計算"""
        if "behavior" not in sequence_data.columns:
            return {"error": "Ground truth (behavior) が利用できません"}

        # Ground truthの変化点検出
        behaviors = sequence_data["behavior"].values
        true_change_points = []
        for i in range(1, len(behaviors)):
            if behaviors[i] != behaviors[i - 1]:
                true_change_points.append(i)

        # 各手法の変化点取得
        clasp_cps = clasp_result.get("change_points", {}).get("consensus", [])
        supervised_cps = supervised_result.get("change_points", [])

        tolerance = max(1, len(sequence_data) * 0.05)  # 5%の許容誤差

        def calculate_metrics(predicted_cps, true_cps, tolerance):
            if len(true_cps) == 0:
                return {"precision": 0, "recall": 0, "f1": 0}
            if len(predicted_cps) == 0:
                return {"precision": 0, "recall": 0, "f1": 0}

            matched = 0
            for pred_cp in predicted_cps:
                for true_cp in true_cps:
                    if abs(pred_cp - true_cp) <= tolerance:
                        matched += 1
                        break

            precision = matched / len(predicted_cps) if predicted_cps else 0
            recall = matched / len(true_cps) if true_cps else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {"precision": precision, "recall": recall, "f1": f1}

        clasp_metrics = calculate_metrics(clasp_cps, true_change_points, tolerance)
        supervised_metrics = calculate_metrics(supervised_cps, true_change_points, tolerance)

        return {
            "ground_truth_change_points": len(true_change_points),
            "clasp": {"change_points": len(clasp_cps), "metrics": clasp_metrics},
            "supervised": {"change_points": len(supervised_cps), "metrics": supervised_metrics},
            "tolerance": tolerance,
        }

    def plot_performance_comparison(self, comparison_results: list[dict]) -> None:
        """複数シーケンスでの性能比較を可視化"""
        clasp_f1s = []
        supervised_f1s = []
        clasp_precisions = []
        supervised_precisions = []
        clasp_recalls = []
        supervised_recalls = []

        for result in comparison_results:
            if "error" not in result:
                clasp_f1s.append(result["clasp"]["metrics"]["f1"])
                supervised_f1s.append(result["supervised"]["metrics"]["f1"])
                clasp_precisions.append(result["clasp"]["metrics"]["precision"])
                supervised_precisions.append(result["supervised"]["metrics"]["precision"])
                clasp_recalls.append(result["clasp"]["metrics"]["recall"])
                supervised_recalls.append(result["supervised"]["metrics"]["recall"])

        if len(clasp_f1s) == 0:
            print("比較可能なデータがありません")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # F1スコア比較
        axes[0].boxplot([clasp_f1s, supervised_f1s], labels=["CLaSP", "Supervised"])
        axes[0].set_title("F1 Score Comparison")
        axes[0].set_ylabel("F1 Score")
        axes[0].grid(True, alpha=0.3)

        # Precision比較
        axes[1].boxplot([clasp_precisions, supervised_precisions], labels=["CLaSP", "Supervised"])
        axes[1].set_title("Precision Comparison")
        axes[1].set_ylabel("Precision")
        axes[1].grid(True, alpha=0.3)

        # Recall比較
        axes[2].boxplot([clasp_recalls, supervised_recalls], labels=["CLaSP", "Supervised"])
        axes[2].set_title("Recall Comparison")
        axes[2].set_ylabel("Recall")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        
        # ファイル保存
        output_file = OUTPUT_DIR / "performance_comparison_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 性能比較プロットを保存: {output_file}")

        # 統計サマリー
        print("\n=== 性能比較サマリー ===")
        print(f"評価シーケンス数: {len(clasp_f1s)}")
        print("\nF1 Score:")
        print(f"  CLaSP: {np.mean(clasp_f1s):.3f} (±{np.std(clasp_f1s):.3f})")
        print(f"  Supervised: {np.mean(supervised_f1s):.3f} (±{np.std(supervised_f1s):.3f})")
        print("\nPrecision:")
        print(f"  CLaSP: {np.mean(clasp_precisions):.3f} (±{np.std(clasp_precisions):.3f})")
        print(f"  Supervised: {np.mean(supervised_precisions):.3f} (±{np.std(supervised_precisions):.3f})")
        print("\nRecall:")
        print(f"  CLaSP: {np.mean(clasp_recalls):.3f} (±{np.std(clasp_recalls):.3f})")
        print(f"  Supervised: {np.mean(supervised_recalls):.3f} (±{np.std(supervised_recalls):.3f})")


# =============================================================================
# メイン実行部分
# =============================================================================


def main():
    """メイン実行関数"""
    print("\n1. データ読み込みとIMU特徴量計算")
    print("-" * 50)

    # データファイルパス（サンプル - 実際のパスに変更してください）
    data_path = PROJECT_ROOT / "data" / "train.csv"

    if not data_path.exists():
        print("❌ データファイルが見つかりません")
        print(f"想定パス: {data_path}")
        print("データファイルを配置するか、パスを修正してください")
        return

    # データ読み込み（最初の10,000行のみ：デモ用）
    df = pl.read_csv(data_path).head(10000)
    print(f"データ読み込み完了: {df.shape}")

    # IMU特徴量計算
    df_with_features = compute_imu_physics_features(df)

    # 利用可能な特徴量リスト（19個）
    available_features = [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        "acc_mag",
        "rot_angle",
        "acc_mag_jerk",
        "rot_angle_vel",
        "linear_acc_x",
        "linear_acc_y",
        "linear_acc_z",
        "linear_acc_mag",
        "linear_acc_mag_jerk",
        "angular_vel_x",
        "angular_vel_y",
        "angular_vel_z",
        "angular_distance",
    ]

    print(f"IMU特徴量: {len(available_features)}個")

    print("\n2. CLaSPセグメンテーション分析")
    print("-" * 50)

    # CLaSP分析器の初期化
    clasp_analyzer = CLaSPSegmentationAnalyzer(available_features)

    # 分析対象シーケンスの選択（最初の5つ）
    sequence_ids = df_with_features["sequence_id"].unique().to_list()[:5]
    print(f"分析対象: {len(sequence_ids)}シーケンス")

    # CLaSP分析実行
    clasp_results = {}
    for seq_id in sequence_ids:
        seq_data = df_with_features.filter(pl.col("sequence_id") == seq_id).to_pandas()
        if len(seq_data) > 30:  # 十分な長さのシーケンスのみ
            result = clasp_analyzer.analyze_single_sequence(seq_data, seq_id)
            if "error" not in result:
                clasp_results[seq_id] = result

    print(f"CLaSP分析完了: {len(clasp_results)}シーケンス")

    print("\n3. 教師ありセグメンテーション")
    print("-" * 50)

    # 教師ありセグメンテーション分析器の初期化
    supervised_analyzer = SupervisedSegmentationAnalyzer(feature_cols=available_features, window_size=20, stride=10)

    # 学習用データの準備
    train_sequence_ids = df_with_features["sequence_id"].unique().to_list()[:20]
    train_sequences = {}

    for seq_id in train_sequence_ids:
        seq_data = df_with_features.filter(pl.col("sequence_id") == seq_id).to_pandas()
        if len(seq_data) > 30 and "behavior" in seq_data.columns:
            train_sequences[seq_id] = seq_data

    print(f"学習用データ: {len(train_sequences)}シーケンス")

    # モデル学習
    if len(train_sequences) > 0:
        supervised_analyzer.fit(train_sequences)

        # 教師ありセグメンテーション実行
        supervised_results = {}
        for seq_id in list(clasp_results.keys()):
            seq_data = df_with_features.filter(pl.col("sequence_id") == seq_id).to_pandas()
            result = supervised_analyzer.predict_sequence(seq_data, seq_id)
            if "error" not in result:
                supervised_results[seq_id] = result

        print(f"教師ありセグメンテーション完了: {len(supervised_results)}シーケンス")

        print("\n4. 比較分析")
        print("-" * 50)

        # 比較ツールの初期化
        comparison_tool = SegmentationComparisonTool()

        # 共通シーケンスでの比較
        common_sequence_ids = list(set(clasp_results.keys()) & set(supervised_results.keys()))

        if len(common_sequence_ids) > 0:
            print(f"比較対象: {len(common_sequence_ids)}シーケンス")

            comparison_results = []

            for seq_id in common_sequence_ids[:3]:  # 最初の3シーケンス
                print(f"\n--- シーケンス {seq_id} の比較 ---")

                # データ取得
                seq_data = df_with_features.filter(pl.col("sequence_id") == seq_id).to_pandas()
                clasp_result = clasp_results[seq_id]
                supervised_result = supervised_results[seq_id]

                # 比較可視化
                comparison_tool.plot_comparison(seq_data, clasp_result, supervised_result)

                # メトリクス計算
                metrics = comparison_tool.compute_comparison_metrics(seq_data, clasp_result, supervised_result)
                comparison_results.append(metrics)

                if "error" not in metrics:
                    print(f"\n{seq_id} 性能比較:")
                    print(f"  Ground Truth変化点数: {metrics['ground_truth_change_points']}")
                    print(f"  CLaSP F1: {metrics['clasp']['metrics']['f1']:.3f}")
                    print(f"  Supervised F1: {metrics['supervised']['metrics']['f1']:.3f}")

            # 全体性能比較
            if comparison_results:
                print("\n" + "=" * 50)
                print("全体性能比較")
                print("=" * 50)
                comparison_tool.plot_performance_comparison(comparison_results)

        else:
            print("❌ 比較可能なシーケンスがありません")

    else:
        print("❌ 学習用データが不足しています")

    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
