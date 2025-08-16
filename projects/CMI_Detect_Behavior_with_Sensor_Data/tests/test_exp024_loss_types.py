"""Test different loss types for exp024."""

import pytest
import torch
from codes.exp.exp024.config import Config
from codes.exp.exp024.model import CMISqueezeformer


@pytest.mark.parametrize("loss_type", [
    "cmi",
    "cmi_focal",
    "soft_f1",
    "acls",
    "label_smoothing",
    "mbls"
])
def test_loss_type_initialization(loss_type):
    """各loss typeでモデルが正しく初期化されることを確認."""
    config = Config()
    config.loss.type = loss_type
    config.demographics.enabled = False  # シンプルにするため

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
    )

    # 3つの損失関数が設定されていることを確認
    assert hasattr(model, 'multiclass_criterion')
    assert hasattr(model, 'binary_criterion')
    assert hasattr(model, 'nine_class_criterion')
    assert model.multiclass_criterion is not None
    assert model.binary_criterion is not None
    assert model.nine_class_criterion is not None


@pytest.mark.parametrize("loss_type", [
    "cmi",
    "cmi_focal",
    "soft_f1",
    "acls",
    "label_smoothing",
    "mbls"
])
def test_training_step_with_different_loss_types(loss_type):
    """各loss typeでtraining_stepが正常に動作することを確認."""
    config = Config()
    config.loss.type = loss_type
    config.demographics.enabled = False

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
    )

    # ダミーバッチ
    batch = {
        "imu": torch.randn(2, 7, 50),  # [batch, input_dim, seq_len]
        "multiclass_label": torch.randint(0, 18, (2,)),
        "binary_label": torch.randint(0, 2, (2,)).float(),  # BCEWithLogitsLossはfloat型が必要
        "nine_class_label": torch.randint(0, 9, (2,)),
    }

    # training_stepが正常に動作することを確認
    loss = model.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss > 0
    assert loss.requires_grad  # 勾配計算可能


def test_nine_class_criterion_consistency():
    """nine_class_criterionが適切に設定されることを確認."""
    config = Config()
    config.demographics.enabled = False

    # label_smoothingの場合
    config.loss.type = "label_smoothing"
    model_ls = CMISqueezeformer(
        input_dim=7, d_model=64, num_classes=18, learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
    )

    # LabelSmoothingCrossEntropyが設定されていることを確認
    assert model_ls.nine_class_criterion.__class__.__name__ == "LabelSmoothingCrossEntropy"

    # mblsの場合
    config.loss.type = "mbls"
    model_mbls = CMISqueezeformer(
        input_dim=7, d_model=64, num_classes=18, learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
    )

    # MbLSが設定されていることを確認
    assert model_mbls.nine_class_criterion.__class__.__name__ == "MbLS"


def test_helper_methods():
    """ヘルパーメソッドが正しく動作することを確認."""
    config = Config()
    config.demographics.enabled = False

    model = CMISqueezeformer(
        input_dim=7, d_model=64, num_classes=18, learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
        acls_config=config.acls.model_dump(),
    )

    # ACLS criterionの作成テスト
    acls_criterion = model._create_acls_criterion(10)
    assert acls_criterion.__class__.__name__ == "ACLS"

    # SoftF1 criterionの作成テスト
    soft_f1_criterion = model._create_soft_f1_criterion(10)
    assert soft_f1_criterion.__class__.__name__ == "MulticlassSoftF1Loss"

    # Binary SoftF1 criterionの作成テスト
    binary_soft_f1_criterion = model._create_binary_soft_f1_criterion()
    assert binary_soft_f1_criterion.__class__.__name__ == "BinarySoftF1Loss"

    # ACLS Binary criterionの作成テスト
    acls_binary_criterion = model._create_acls_binary_criterion()
    assert acls_binary_criterion.__class__.__name__ == "ACLSBinary"


def test_kl_loss_with_different_loss_types():
    """異なるloss typeでKL lossが正常に動作することを確認."""
    config = Config()
    config.loss.kl_weight = 0.1
    config.demographics.enabled = False

    for loss_type in ["cmi", "soft_f1", "acls", "label_smoothing", "mbls"]:
        config.loss.type = loss_type

        model = CMISqueezeformer(
            input_dim=7, d_model=64, num_classes=18, learning_rate=1e-4,
            loss_config=config.loss.model_dump(),
            acls_config=config.acls.model_dump(),
        )

        # KL loss計算
        multiclass_logits = torch.randn(2, 18)
        nine_class_logits = torch.randn(2, 9)
        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        assert torch.isfinite(kl_loss)
        assert kl_loss >= 0


if __name__ == "__main__":
    pytest.main([__file__])
