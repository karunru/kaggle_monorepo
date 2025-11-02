# SupervisedSegmentationAnalyzer修正用コード
# ノートブックで比較処理の前に実行してください

# 1. supervised_analyzerが定義されているかチェック
if "supervised_analyzer" not in locals():
    print("SupervisedSegmentationAnalyzerを初期化しています...")

    # 必要なライブラリのインポート
    import warnings
    from collections import Counter

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    warnings.filterwarnings("ignore")

    # SupervisedSegmentationAnalyzerの定義（ノートブックにすでに定義されている場合はスキップ）
    if "SupervisedSegmentationAnalyzer" not in locals():

        class SupervisedSegmentationAnalyzer:
            """
            教師ありWindow-based Segmentation分析クラス
            behaviorラベルを使用して学習し、より高精度なセグメンテーションを実現
            """

            def __init__(self, feature_cols: List[str], window_size: int = 20, stride: int = 10):
                self.feature_cols = feature_cols
                self.window_size = window_size
                self.stride = stride
                self.classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
                )
                self.label_encoder = LabelEncoder()
                self.is_fitted = False
                self.feature_importance_ = None

            def _extract_window_features(self, sequence_data: pd.DataFrame) -> np.ndarray:
                """ウィンドウベースの特徴量を抽出"""
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
                            [
                                np.percentile(feature_values, 25),
                                np.percentile(feature_values, 75),
                                np.var(feature_values),
                            ]
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

            def _extract_window_labels(self, sequence_data: pd.DataFrame) -> List[str]:
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

            def fit(self, sequences_data):
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

                print("\\n学習データ準備完了:")
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

                print("\\n交差検証結果:")
                print(f"  F1スコア (macro): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

                return self

            def predict_sequence(self, sequence_data, sequence_id):
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
                    return {
                        "sequence_id": sequence_id,
                        "error": str(e),
                        "predictions": [],
                        "change_points": [],
                        "segments": {},
                    }

    # available_featuresが定義されているかチェック
    if "available_features" not in locals():
        available_features = [
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
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
        ]

    # SupervisedSegmentationAnalyzerの初期化
    supervised_analyzer = SupervisedSegmentationAnalyzer(
        feature_cols=available_features,
        window_size=20,  # ウィンドウサイズ
        stride=10,  # ストライド（重複度）
    )

    print("✅ supervised_analyzerを初期化しました")

    # 学習データが利用可能な場合は学習
    if "df_with_features" in locals():
        # 学習用データの準備
        train_sequence_ids = df_with_features["sequence_id"].unique().to_list()[:20]  # 20シーケンスで学習

        # 学習用データセットの作成
        train_sequences = {}
        for seq_id in train_sequence_ids:
            seq_data = df_with_features.filter(pl.col("sequence_id") == seq_id).to_pandas()
            if len(seq_data) > 30 and "behavior" in seq_data.columns:  # 十分な長さのシーケンスのみ
                train_sequences[seq_id] = seq_data

        print(f"学習用シーケンス数: {len(train_sequences)}")

        if len(train_sequences) > 0:
            # モデル学習
            supervised_analyzer.fit(train_sequences)
            print("✅ 教師ありモデルの学習完了")
        else:
            print("❌ 学習用データが不足しています")
    else:
        print("⚠️ df_with_featuresが定義されていません。後で学習してください。")

else:
    print("✅ supervised_analyzerは既に定義済みです")


# 安全な比較実行関数
def safe_comparison_execution():
    """安全に比較を実行する関数"""
    # 必要な変数の存在チェック
    missing_vars = []

    if "supervised_analyzer" not in locals() and "supervised_analyzer" not in globals():
        missing_vars.append("supervised_analyzer")

    if "sample_results" not in locals() and "sample_results" not in globals():
        missing_vars.append("sample_results")

    if "comparison_tool" not in locals() and "comparison_tool" not in globals():
        missing_vars.append("comparison_tool")

    if missing_vars:
        print(f"❌ 以下の変数が定義されていません: {', '.join(missing_vars)}")
        print("必要なセルを先に実行してください")
        return False

    # supervised_analyzerの学習状態チェック
    analyzer = locals().get("supervised_analyzer") or globals().get("supervised_analyzer")
    if not hasattr(analyzer, "is_fitted") or not analyzer.is_fitted:
        print("❌ supervised_analyzerが学習されていません")
        print("先に学習セルを実行してください")
        return False

    return True


print("\\n=== 安全な比較実行のために ===")
print("比較処理を実行する前に以下を確認してください:")
print("1. SupervisedSegmentationAnalyzerクラスが定義済み")
print("2. supervised_analyzerがインスタンス化済み")
print("3. supervised_analyzerが学習済み (is_fitted=True)")
print("4. sample_results (CLaSP結果) が存在")
print("5. comparison_tool (比較ツール) が存在")
print("\\n比較実行前に safe_comparison_execution() を呼び出してチェックしてください")
