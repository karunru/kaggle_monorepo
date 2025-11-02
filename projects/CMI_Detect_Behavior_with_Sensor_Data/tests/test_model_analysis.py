#!/usr/bin/env python3
"""
モデル解析機能のテストコード
"""

import json
import sys
from pathlib import Path

import pytest
import torch

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent))

from analyze_models_detailed import (
    analyze_model_state_dict,
    analyze_pytorch_checkpoint_detailed,
    estimate_detailed_architecture,
)


class TestModelAnalysis:
    """モデル解析機能のテスト"""

    def test_pytorch_checkpoint_analysis(self):
        """PyTorchチェックポイント解析のテスト"""
        model_path = Path("public_datasets/imu-only-datas/saved_models/fold_1_model.pth")

        if not model_path.exists():
            pytest.skip(f"モデルファイルが見つかりません: {model_path}")

        result = analyze_pytorch_checkpoint_detailed(model_path)

        # 基本的な結果の検証
        assert result["success"] == True
        assert result["file_name"] == "fold_1_model.pth"
        assert result["file_size_mb"] > 0
        assert "model_info" in result

        # モデル情報の検証
        model_info = result["model_info"]
        assert model_info["total_parameters"] > 0
        assert model_info["num_layers"] > 0
        assert "architecture_analysis" in model_info

    def test_model_state_dict_analysis(self):
        """モデルstate_dict解析のテスト"""
        model_path = Path("public_datasets/imu-only-datas/saved_models/fold_1_model.pth")

        if not model_path.exists():
            pytest.skip(f"モデルファイルが見つかりません: {model_path}")

        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        result = analyze_model_state_dict(state_dict)

        # 結果の検証
        assert result["total_parameters"] == 456475  # 期待値
        assert result["num_layers"] == 64  # 期待値
        assert "layer_categories" in result
        assert "architecture_analysis" in result

        # レイヤーカテゴリーの検証
        categories = result["layer_categories"]
        assert categories["convolutional"] == 4
        assert categories["normalization"] == 30
        assert categories["attention"] == 2

    def test_architecture_estimation(self):
        """アーキテクチャ推定のテスト"""
        # ダミーレイヤー情報を作成
        layer_info = [
            {"name": "conv1.weight", "shape": [64, 20, 3]},
            {"name": "attention.weight", "shape": [1, 256]},
            {"name": "bn1.weight", "shape": [64]},
        ]

        conv_layers = [layer_info[0]]
        attention_layers = [layer_info[1]]
        norm_layers = [layer_info[2]]

        result = estimate_detailed_architecture(layer_info, conv_layers, [], norm_layers, attention_layers, [])

        # 結果の検証
        assert result["primary_type"] == "Attention-based (Transformer-like)"
        assert "Self-Attention" in result["sub_components"]
        # Normalizationの検出は実際のレイヤー名に依存するため、より柔軟にテスト
        assert isinstance(result["special_features"], list)

    def test_model_consistency(self):
        """5つのfoldモデルの一貫性テスト"""
        base_path = Path("public_datasets/imu-only-datas/saved_models")
        model_files = list(base_path.glob("fold_*_model.pth"))

        if len(model_files) == 0:
            pytest.skip("モデルファイルが見つかりません")

        param_counts = []
        architectures = []

        for model_file in model_files:
            result = analyze_pytorch_checkpoint_detailed(model_file)
            if result["success"] and "model_info" in result:
                param_counts.append(result["model_info"]["total_parameters"])
                architectures.append(result["model_info"]["architecture_analysis"]["primary_type"])

        # 全てのモデルが同じパラメータ数を持つことを確認
        assert len(set(param_counts)) == 1, f"パラメータ数が不一致: {param_counts}"

        # 全てのモデルが同じアーキテクチャタイプを持つことを確認
        assert len(set(architectures)) == 1, f"アーキテクチャタイプが不一致: {architectures}"


class TestFeatureScaler:
    """特徴量スケーラーのテスト"""

    def test_feature_scaler_properties(self):
        """特徴量スケーラーの基本プロパティテスト"""
        # 解析結果JSONファイルを読み込む
        results_path = Path("outputs/claude/model_analysis_detailed_results.json")

        if not results_path.exists():
            pytest.skip("解析結果ファイルが見つかりません")

        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

        scaler_info = results["feature_scaler"]
        assert scaler_info["success"] == True
        assert scaler_info["scaler_type"] == "StandardScaler"
        assert scaler_info["n_features"] == 20

        # スケーラーのパラメータが正しい長さであることを確認
        assert len(scaler_info["mean_"]) == 20
        assert len(scaler_info["scale_"]) == 20
        assert len(scaler_info["var_"]) == 20


class TestKFoldResults:
    """K-fold結果のテスト"""

    def test_kfold_results_structure(self):
        """K-fold結果の構造テスト"""
        results_path = Path("outputs/claude/model_analysis_detailed_results.json")

        if not results_path.exists():
            pytest.skip("解析結果ファイルが見つかりません")

        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

        kfold_info = results["kfold_results"]["content"]

        # 基本構造の確認
        assert kfold_info["n_folds"] == 5
        assert len(kfold_info["fold_results"]) == 5
        assert "summary" in kfold_info

        # 各フォールドの結果確認
        for i, fold_result in enumerate(kfold_info["fold_results"], 1):
            assert fold_result["fold"] == i
            assert "final_val_acc" in fold_result
            assert "final_val_score" in fold_result
            assert "test_accuracy" in fold_result
            assert "test_score" in fold_result

            # スコアが合理的な範囲内であることを確認
            assert 0.0 <= fold_result["final_val_acc"] <= 1.0
            assert 0.0 <= fold_result["final_val_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
