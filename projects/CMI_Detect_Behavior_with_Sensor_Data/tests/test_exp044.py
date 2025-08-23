"""Test code for exp044 implementation."""

import sys
from pathlib import Path

import torch

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / "codes" / "exp" / "exp044"))

from config import Config
from model import IMUOnlyLSTM, _safe_softmax, fuse_heads


class TestProbabilityFusion:
    """確率融合機能のテスト."""

    def test_safe_softmax(self):
        """_safe_softmax関数のテスト."""
        # ロジット入力
        logits = torch.randn(10, 5)
        result = _safe_softmax(logits, dim=-1, T=1.0)

        assert result.shape == (10, 5)
        assert torch.allclose(result.sum(dim=-1), torch.ones(10), atol=1e-3)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

        # 確率入力（和が1）
        probs = torch.softmax(torch.randn(10, 5), dim=-1)
        result = _safe_softmax(probs, dim=-1, T=1.0)

        assert result.shape == (10, 5)
        assert torch.allclose(result.sum(dim=-1), torch.ones(10), atol=1e-3)

    def test_fuse_heads_basic(self):
        """fuse_heads関数の基本テスト."""
        batch_size = 32

        # サンプル確率分布
        multiclass_probs = torch.softmax(torch.randn(batch_size, 18), dim=-1)
        binary_probs = torch.softmax(torch.randn(batch_size, 2), dim=-1)
        nine_class_probs = torch.softmax(torch.randn(batch_size, 9), dim=-1)

        # インデックス設定
        target_idx18 = [0, 1, 2, 3, 4, 5, 6, 7]  # 8つのターゲット
        nontarget_idx18 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 10の非ターゲット
        nine_non_target_id = 8  # 9クラス目

        # 融合実行
        final_probs, preds = fuse_heads(
            multiclass_probs, binary_probs, nine_class_probs, target_idx18, nontarget_idx18, nine_non_target_id
        )

        # 出力検証
        assert final_probs.shape == (batch_size, 18)
        assert preds.shape == (batch_size,)
        assert torch.allclose(final_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-3)
        assert torch.all(final_probs >= 0)
        assert torch.all(final_probs <= 1)
        assert torch.all(preds >= 0)
        assert torch.all(preds < 18)

    def test_fuse_heads_edge_cases(self):
        """fuse_heads関数のエッジケースのテスト."""
        batch_size = 10

        # 極端な確率（ほぼ0と1）
        multiclass_probs = torch.zeros(batch_size, 18)
        multiclass_probs[:, 0] = 1.0  # 全て最初のクラスに確率1

        binary_probs = torch.zeros(batch_size, 2)
        binary_probs[:, 1] = 1.0  # 全てターゲット

        nine_class_probs = torch.zeros(batch_size, 9)
        nine_class_probs[:, 0] = 1.0  # 全て最初のターゲットクラス

        target_idx18 = [0, 1, 2, 3, 4, 5, 6, 7]
        nontarget_idx18 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        nine_non_target_id = 8

        # 融合実行
        final_probs, preds = fuse_heads(
            multiclass_probs, binary_probs, nine_class_probs, target_idx18, nontarget_idx18, nine_non_target_id
        )

        # 数値安定性の確認
        assert not torch.any(torch.isnan(final_probs))
        assert not torch.any(torch.isinf(final_probs))
        assert not torch.any(torch.isnan(preds))


class TestIMUOnlyLSTMConfig:
    """IMUOnlyLSTMの設定テスト."""

    def test_model_config_integration(self):
        """ModelConfigとの統合テスト."""
        config = Config()
        model_config = config.model

        # 設定値の検証
        assert model_config.imu_block1_out_channels == 64
        assert model_config.imu_block2_out_channels == 128
        assert model_config.bigru_hidden_size == 128
        assert model_config.dense1_out_features == 256
        assert model_config.dense2_out_features == 128

        # 確率融合パラメータの検証
        assert model_config.fusion_temperature_18 == 1.0
        assert model_config.fusion_temperature_binary == 1.0
        assert model_config.fusion_temperature_9 == 1.0
        assert model_config.fusion_weight_hierarchical == 0.6
        assert model_config.fusion_tau_threshold == 0.5

    def test_imu_model_with_config(self):
        """IMUOnlyLSTMモデルのconfig使用テスト."""
        config = Config()
        model_config = config.model

        # カスタム設定でモデル作成
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=16, model_config=model_config)

        # モデル構造の検証
        assert hasattr(model, "dense_layers")
        assert isinstance(model.dense_layers, torch.nn.Sequential)

        # フォワードテスト
        batch_size = 8
        seq_len = 100
        imu_data = torch.randn(batch_size, seq_len, 20)
        demographics = torch.randn(batch_size, 16)

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)

    def test_imu_model_without_config(self):
        """IMUOnlyLSTMモデルのconfig未使用（後方互換性）テスト."""
        # config未指定でも動作することを確認
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=0, model_config=None)

        # デフォルト値での動作確認
        batch_size = 4
        seq_len = 50
        imu_data = torch.randn(batch_size, seq_len, 20)

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)


class TestCSVOutput:
    """CSV出力機能のテスト."""

    def test_save_validation_results_to_csv(self):
        """CSV保存機能のテスト."""
        config = Config()

        # CMISqueezeformerのモック作成（最低限のプロパティのみ）
        class MockCMISqueezeformer:
            def __init__(self):
                self.paths_config = config.paths
                self.current_epoch = 5

            def save_validation_results_to_csv(
                self,
                sequence_ids,
                binary_probs,
                multiclass_probs,
                nine_class_probs,
                final_probs,
                pred_gestures,
                true_gestures,
                fold,
            ):
                """検証結果をCSVに保存（polarsとconfig経由のパス使用）."""
                from pathlib import Path

                import polars as pl

                # 出力ディレクトリの作成（configから取得）
                output_dir = Path(self.paths_config.output_dir) / "validation_results"
                output_dir.mkdir(parents=True, exist_ok=True)

                # データの準備
                data = {
                    "sequence_id": sequence_ids,
                    "binary_probs": binary_probs.cpu().numpy(),
                    "pred_gesture": pred_gestures,
                    "true_gesture": true_gestures,
                    "fold": fold,
                }

                # 18クラス確率の追加（multiclass_probs_[1-18]）
                for i in range(18):
                    data[f"multiclass_probs_{i + 1}"] = multiclass_probs[:, i].cpu().numpy()

                # 9クラス確率の追加（nine_class_probs_[1-9]）
                for i in range(9):
                    data[f"nine_class_probs_{i + 1}"] = nine_class_probs[:, i].cpu().numpy()

                # 最終確率の追加（final_probs_[1-18]）
                for i in range(18):
                    data[f"final_probs_{i + 1}"] = final_probs[:, i].cpu().numpy()

                # Polars DataFrameの作成と保存
                df = pl.DataFrame(data)
                csv_path = output_dir / f"validation_results_fold_{fold}_epoch_{self.current_epoch}.csv"
                df.write_csv(csv_path)

                return csv_path

        mock_model = MockCMISqueezeformer()

        # サンプルデータの作成
        batch_size = 10
        sequence_ids = [f"SEQ_{i:06d}" for i in range(batch_size)]
        binary_probs = torch.rand(batch_size)
        multiclass_probs = torch.softmax(torch.randn(batch_size, 18), dim=-1)
        nine_class_probs = torch.softmax(torch.randn(batch_size, 9), dim=-1)
        final_probs = torch.softmax(torch.randn(batch_size, 18), dim=-1)
        pred_gestures = ["Above ear - pull hair"] * batch_size
        true_gestures = ["Above ear - pull hair"] * batch_size
        fold = 1

        # CSV保存実行
        csv_path = mock_model.save_validation_results_to_csv(
            sequence_ids,
            binary_probs,
            multiclass_probs,
            nine_class_probs,
            final_probs,
            pred_gestures,
            true_gestures,
            fold,
        )

        # ファイル存在確認
        assert csv_path.exists()

        # Polarsで読み込んで内容確認
        import polars as pl

        df = pl.read_csv(csv_path)

        assert df.shape[0] == batch_size
        assert "sequence_id" in df.columns
        assert "binary_probs" in df.columns
        assert "pred_gesture" in df.columns
        assert "true_gesture" in df.columns
        assert "fold" in df.columns

        # 18クラス確率列の確認
        for i in range(1, 19):
            assert f"multiclass_probs_{i}" in df.columns

        # 9クラス確率列の確認
        for i in range(1, 10):
            assert f"nine_class_probs_{i}" in df.columns

        # 最終確率列の確認
        for i in range(1, 19):
            assert f"final_probs_{i}" in df.columns

        # クリーンアップ
        csv_path.unlink()
        csv_path.parent.rmdir()


class TestFoldValidationResults:
    """Fold終了時CSV保存機能のテスト."""

    def test_clear_fold_results(self):
        """fold結果クリア機能のテスト."""
        config = Config()

        class MockModel:
            def __init__(self):
                self.fold_validation_results = [{"epoch": 1, "cmi_score": 0.5}]

            def clear_fold_results(self):
                self.fold_validation_results.clear()
                print("Cleared fold validation results")

        mock_model = MockModel()
        assert len(mock_model.fold_validation_results) == 1

        mock_model.clear_fold_results()
        assert len(mock_model.fold_validation_results) == 0

    def test_save_fold_validation_results_to_csv(self):
        """fold終了時CSV保存機能のテスト."""
        config = Config()

        class MockCMISqueezeformer:
            def __init__(self):
                self.paths_config = config.paths
                self.fold_validation_results = [
                    {
                        "epoch": 1,
                        "sequence_ids": [f"SEQ_{i:06d}" for i in range(5)],
                        "binary_probs": torch.rand(5),
                        "multiclass_probs": torch.softmax(torch.randn(5, 18), dim=-1),
                        "nine_class_probs": torch.softmax(torch.randn(5, 9), dim=-1),
                        "final_probs": torch.softmax(torch.randn(5, 18), dim=-1),
                        "pred_gestures": ["Above ear - pull hair"] * 5,
                        "true_gestures": ["Above ear - pull hair"] * 5,
                        "cmi_score": 0.3,
                    },
                    {
                        "epoch": 2,
                        "sequence_ids": [f"SEQ_{i:06d}" for i in range(5)],
                        "binary_probs": torch.rand(5),
                        "multiclass_probs": torch.softmax(torch.randn(5, 18), dim=-1),
                        "nine_class_probs": torch.softmax(torch.randn(5, 9), dim=-1),
                        "final_probs": torch.softmax(torch.randn(5, 18), dim=-1),
                        "pred_gestures": ["Above ear - pull hair"] * 5,
                        "true_gestures": ["Above ear - pull hair"] * 5,
                        "cmi_score": 0.8,  # 最高スコア
                    },
                ]

            def save_fold_validation_results_to_csv(self, fold: int):
                """Fold終了時に最良エポックの検証結果をCSVに保存."""
                if not self.fold_validation_results:
                    print(f"No validation results to save for fold {fold}")
                    return

                # CMIスコアが最も高いエポックを選択
                best_result = max(self.fold_validation_results, key=lambda x: x.get("cmi_score", 0.0))
                best_epoch = best_result["epoch"]

                print(
                    f"Saving fold {fold} results (best epoch: {best_epoch}, CMI score: {best_result['cmi_score']:.4f})"
                )

                # CSV保存（簡易版）
                return self.save_validation_results_to_csv(
                    sequence_ids=best_result["sequence_ids"],
                    binary_probs=best_result["binary_probs"],
                    multiclass_probs=best_result["multiclass_probs"],
                    nine_class_probs=best_result["nine_class_probs"],
                    final_probs=best_result["final_probs"],
                    pred_gestures=best_result["pred_gestures"],
                    true_gestures=best_result["true_gestures"],
                    fold=fold,
                    epoch=best_epoch,
                )

            def save_validation_results_to_csv(
                self,
                sequence_ids,
                binary_probs,
                multiclass_probs,
                nine_class_probs,
                final_probs,
                pred_gestures,
                true_gestures,
                fold,
                epoch=None,
            ):
                """検証結果をCSVに保存."""
                import polars as pl
                from pathlib import Path

                # 出力ディレクトリの作成
                output_dir = Path(self.paths_config.output_dir) / "validation_results"
                output_dir.mkdir(parents=True, exist_ok=True)

                # データの準備
                data = {
                    "sequence_id": sequence_ids,
                    "binary_probs": binary_probs.cpu().numpy(),
                    "pred_gesture": pred_gestures,
                    "true_gesture": true_gestures,
                    "fold": fold,
                }

                # 18クラス確率の追加
                for i in range(18):
                    data[f"multiclass_probs_{i + 1}"] = multiclass_probs[:, i].cpu().numpy()

                # 9クラス確率の追加
                for i in range(9):
                    data[f"nine_class_probs_{i + 1}"] = nine_class_probs[:, i].cpu().numpy()

                # 最終確率の追加
                for i in range(18):
                    data[f"final_probs_{i + 1}"] = final_probs[:, i].cpu().numpy()

                # CSV保存
                df = pl.DataFrame(data)
                epoch_num = epoch if epoch is not None else 0
                csv_path = output_dir / f"validation_results_fold_{fold}_epoch_{epoch_num}.csv"
                df.write_csv(csv_path)

                return csv_path

        mock_model = MockCMISqueezeformer()
        fold = 1

        # CSV保存実行
        csv_path = mock_model.save_fold_validation_results_to_csv(fold)

        # ファイル存在確認
        assert csv_path.exists()

        # ファイル名に最良エポック(2)が含まれていることを確認
        assert "epoch_2.csv" in str(csv_path)

        # クリーンアップ
        csv_path.unlink()
        csv_path.parent.rmdir()


class TestConfigValidation:
    """設定ファイルの検証テスト."""

    def test_exp044_config_values(self):
        """exp044固有の設定値テスト."""
        config = Config()

        # 実験設定の確認
        assert config.experiment.exp_num == "exp044"
        assert "Hierarchical Bayesian fusion" in config.experiment.description
        assert "hierarchical_bayesian_fusion" in config.experiment.tags
        assert "configurable_imu" in config.experiment.tags
        assert "csv_output" in config.experiment.tags

        # ログ設定の確認
        assert "hierarchical_bayesian_fusion" in config.logging.wandb_tags
        assert "configurable_imu" in config.logging.wandb_tags
        assert "csv_output" in config.logging.wandb_tags

    def test_model_config_defaults(self):
        """ModelConfigのデフォルト値テスト."""
        config = Config()
        model_config = config.model

        # IMU設定のデフォルト値
        assert model_config.imu_block1_out_channels == 64
        assert model_config.imu_block2_out_channels == 128
        assert model_config.bigru_hidden_size == 128
        assert model_config.dense1_out_features == 256
        assert model_config.dense2_out_features == 128
        assert model_config.dense1_dropout == 0.5
        assert model_config.dense2_dropout == 0.3
        assert model_config.gru_dropout == 0.4
        assert model_config.imu_block_dropout == 0.3

        # 確率融合設定のデフォルト値
        assert model_config.fusion_temperature_18 == 1.0
        assert model_config.fusion_temperature_binary == 1.0
        assert model_config.fusion_temperature_9 == 1.0
        assert model_config.fusion_weight_hierarchical == 0.6
        assert model_config.fusion_tau_threshold == 0.5


if __name__ == "__main__":
    # 基本テストの実行
    test_fusion = TestProbabilityFusion()
    test_fusion.test_safe_softmax()
    test_fusion.test_fuse_heads_basic()
    test_fusion.test_fuse_heads_edge_cases()
    print("✓ Probability fusion tests passed")

    test_model = TestIMUOnlyLSTMConfig()
    test_model.test_model_config_integration()
    test_model.test_imu_model_with_config()
    test_model.test_imu_model_without_config()
    print("✓ IMUOnlyLSTM config tests passed")

    test_config = TestConfigValidation()
    test_config.test_exp044_config_values()
    test_config.test_model_config_defaults()
    print("✓ Config validation tests passed")

    print("All tests passed successfully!")
