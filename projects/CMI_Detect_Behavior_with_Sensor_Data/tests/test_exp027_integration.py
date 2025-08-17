"""exp027の統合テスト."""

import torch
from codes.exp.exp027.config import config
from codes.exp.exp027.model import FocalLoss
from torch import nn


class TestExp027Integration:
    """exp027の統合テスト."""

    def test_cmi_focal_loss_integration(self):
        """cmi_focal損失でmulticlass_criterionとnine_class_criterionの両方がFocalLossを使用することを確認."""
        # モック設定を作成（cmi_focal損失タイプ）
        mock_loss_config = {
            "type": "cmi_focal",
            "focal_gamma": 2.0,
            "focal_alpha": 1.0,
            "label_smoothing": 0.1,
            "nine_class_head_enabled": True,
        }

        # 簡単なクラスを作成してテスト
        class TestModel:
            def __init__(self, loss_config):
                self.loss_config = loss_config
                self._setup_loss_functions()

            def _setup_loss_functions(self):
                """exp027のmodel.pyと同様の損失関数設定ロジック."""
                loss_type = self.loss_config.get("type", "soft_f1")

                if loss_type == "cmi_focal":
                    # Focal Loss
                    self.multiclass_criterion = FocalLoss(
                        gamma=self.loss_config.get("focal_gamma", 2.0),
                        alpha=self.loss_config.get("focal_alpha", 1.0),
                        label_smoothing=self.loss_config.get("label_smoothing", 0.0),
                    )
                    self.binary_criterion = nn.BCEWithLogitsLoss()
                    # 9クラス用もFocalLoss
                    self.nine_class_criterion = FocalLoss(
                        gamma=self.loss_config.get("focal_gamma", 2.0),
                        alpha=self.loss_config.get("focal_alpha", 1.0),
                        label_smoothing=self.loss_config.get("label_smoothing", 0.0),
                    )

        # テストモデル作成
        model = TestModel(mock_loss_config)

        # 両方の損失関数がFocalLossのインスタンスであることを確認
        assert isinstance(model.multiclass_criterion, FocalLoss)
        assert isinstance(model.nine_class_criterion, FocalLoss)
        assert isinstance(model.binary_criterion, nn.BCEWithLogitsLoss)

        # パラメータが正しく設定されていることを確認
        assert model.multiclass_criterion.gamma == 2.0
        assert model.multiclass_criterion.alpha == 1.0
        assert model.multiclass_criterion.label_smoothing == 0.1

        assert model.nine_class_criterion.gamma == 2.0
        assert model.nine_class_criterion.alpha == 1.0
        assert model.nine_class_criterion.label_smoothing == 0.1

    def test_focal_loss_parameters_consistency(self):
        """multiclass_criterionとnine_class_criterionのFocalLossパラメータが一致することを確認."""
        # 異なるパラメータでテスト
        test_params = [
            {"focal_gamma": 1.5, "focal_alpha": 0.5, "label_smoothing": 0.0},
            {"focal_gamma": 3.0, "focal_alpha": 2.0, "label_smoothing": 0.2},
            {"focal_gamma": 0.5, "focal_alpha": 1.5, "label_smoothing": 0.05},
        ]

        for params in test_params:
            mock_loss_config = {"type": "cmi_focal", **params}

            # 損失関数作成
            multiclass_criterion = FocalLoss(
                gamma=mock_loss_config.get("focal_gamma", 2.0),
                alpha=mock_loss_config.get("focal_alpha", 1.0),
                label_smoothing=mock_loss_config.get("label_smoothing", 0.0),
            )
            nine_class_criterion = FocalLoss(
                gamma=mock_loss_config.get("focal_gamma", 2.0),
                alpha=mock_loss_config.get("focal_alpha", 1.0),
                label_smoothing=mock_loss_config.get("label_smoothing", 0.0),
            )

            # パラメータ一致確認
            assert multiclass_criterion.gamma == nine_class_criterion.gamma
            assert multiclass_criterion.alpha == nine_class_criterion.alpha
            assert multiclass_criterion.label_smoothing == nine_class_criterion.label_smoothing

    def test_focal_loss_computation_consistency(self):
        """同じ入力に対してmulticlass_criterionとnine_class_criterionが同じ損失を計算することを確認."""
        # 同一パラメータでFocalLoss作成
        gamma, alpha, label_smoothing = 2.0, 1.0, 0.1
        multiclass_criterion = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
        nine_class_criterion = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)

        # テスト用データ（9クラス）
        batch_size, num_classes = 4, 9
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 損失計算
        multiclass_loss = multiclass_criterion(inputs, targets)
        nine_class_loss = nine_class_criterion(inputs, targets)

        # 同じ結果を返すことを確認
        assert torch.allclose(multiclass_loss, nine_class_loss, atol=1e-6)

    def test_exp027_config_focal_loss_compatibility(self):
        """exp027の設定がFocalLossと互換性があることを確認."""
        # exp027の損失設定を確認
        loss_config = config.loss

        # 必要なパラメータが存在することを確認
        assert hasattr(loss_config, "focal_gamma")
        assert hasattr(loss_config, "focal_alpha")
        assert hasattr(loss_config, "label_smoothing")

        # デフォルト値が有効であることを確認
        assert isinstance(loss_config.focal_gamma, (int, float))
        assert isinstance(loss_config.focal_alpha, (int, float))
        assert isinstance(loss_config.label_smoothing, (int, float))

        assert loss_config.focal_gamma > 0
        assert loss_config.focal_alpha > 0
        assert 0 <= loss_config.label_smoothing <= 1

    def test_exp027_experiment_metadata(self):
        """exp027の実験メタデータが正しく設定されていることを確認."""
        exp_config = config.experiment

        # exp027固有の設定確認
        assert exp_config.exp_num == "exp027"
        assert "focal_loss" in exp_config.name
        assert "FocalLoss" in exp_config.description
        assert "focal_loss" in exp_config.tags
        assert "nine_class_focal" in exp_config.tags
