"""Test KL divergence loss functionality for exp024."""

import pytest
import torch
from codes.exp.exp024.config import Config
from codes.exp.exp024.model import CMISqueezeformer


def test_kl_divergence_computation():
    """KL divergence計算の動作確認."""
    # 設定
    config = Config()
    config.loss.kl_weight = 0.1
    config.loss.kl_temperature = 1.0

    # モデル初期化
    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    # ダミーのlogits
    batch_size = 4
    multiclass_logits = torch.randn(batch_size, 18)
    nine_class_logits = torch.randn(batch_size, 9)

    # KL loss計算
    kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

    # 検証
    assert kl_loss.shape == torch.Size([])  # スカラー
    assert kl_loss >= 0  # KL divergenceは非負
    assert torch.isfinite(kl_loss)  # 有限値


def test_perfect_alignment():
    """完全に一致する分布でKL lossが小さいことを確認."""
    config = Config()
    config.loss.kl_weight = 0.1
    config.loss.kl_temperature = 1.0

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    # 18クラスと9クラスが整合する場合
    batch_size = 2
    multiclass_logits = torch.zeros(batch_size, 18)
    multiclass_logits[:, 0] = 10.0  # class 0に高い確率

    nine_class_logits = torch.zeros(batch_size, 9)
    nine_class_logits[:, 0] = 10.0  # class 0に高い確率

    # KL lossは非常に小さいはず
    kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

    # 完全整合の場合、KL lossは小さい値になる
    assert kl_loss < 0.1
    assert torch.isfinite(kl_loss)


def test_mapping_consistency():
    """18→9クラスマッピングの整合性確認."""
    config = Config()
    config.loss.kl_weight = 0.1
    config.loss.kl_temperature = 1.0

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    batch_size = 3

    # Target classesでのテスト（0-7）
    for target_class in range(8):
        multiclass_logits = torch.zeros(batch_size, 18)
        multiclass_logits[:, target_class] = 10.0  # 特定のtarget classに高い確率

        nine_class_logits = torch.zeros(batch_size, 9)
        nine_class_logits[:, target_class] = 10.0  # 対応する9-class位置

        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        # 整合性がある場合はKL lossが小さい
        assert kl_loss < 0.5
        assert torch.isfinite(kl_loss)

    # Non-target classesでのテスト（8-17 → 8）
    for non_target_class in range(8, 18):
        multiclass_logits = torch.zeros(batch_size, 18)
        multiclass_logits[:, non_target_class] = 10.0  # 特定のnon-target classに高い確率

        nine_class_logits = torch.zeros(batch_size, 9)
        nine_class_logits[:, 8] = 10.0  # non-target集約位置（index 8）

        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        # 整合性がある場合はKL lossが小さい
        assert kl_loss < 0.5
        assert torch.isfinite(kl_loss)


def test_kl_loss_disabled():
    """KL weight=0の場合、KL lossが0になることを確認."""
    config = Config()
    config.loss.kl_weight = 0.0  # 無効化
    config.loss.kl_temperature = 1.0

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    batch_size = 2
    multiclass_logits = torch.randn(batch_size, 18)
    nine_class_logits = torch.randn(batch_size, 9)

    # KL weightが0の場合
    assert model.kl_weight == 0.0

    # compute_kl_lossは依然として計算可能
    kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)
    assert torch.isfinite(kl_loss)


def test_temperature_effect():
    """温度パラメータの効果を確認."""
    config = Config()
    config.loss.kl_weight = 0.1

    # モデル初期化
    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    batch_size = 2
    multiclass_logits = torch.randn(batch_size, 18) * 5  # 大きなlogitsで差を明確に
    nine_class_logits = torch.randn(batch_size, 9) * 5

    # 異なる温度でのKL loss計算
    model.kl_temperature = 1.0
    kl_loss_temp1 = model.compute_kl_loss(multiclass_logits, nine_class_logits)

    model.kl_temperature = 3.0
    kl_loss_temp3 = model.compute_kl_loss(multiclass_logits, nine_class_logits)

    # 両方とも有限値
    assert torch.isfinite(kl_loss_temp1)
    assert torch.isfinite(kl_loss_temp3)

    # 温度が高いとソフトな分布になり、KL lossが変化する
    # （具体的な大小関係は入力によるが、少なくとも計算は成功する）


def test_model_integration():
    """モデル全体でのKL loss統合テスト."""
    config = Config()
    config.loss.kl_weight = 0.1
    config.loss.kl_temperature = 1.0
    config.demographics.enabled = False  # シンプルにするため

    model = CMISqueezeformer(
        input_dim=7,
        d_model=64,
        num_classes=18,
        learning_rate=1e-4,
        loss_config=config.loss.model_dump(),
    )

    # ダミーバッチ
    batch = {
        "imu": torch.randn(2, 7, 100),  # [batch, input_dim, seq_len]
        "multiclass_label": torch.randint(0, 18, (2,)),
        "binary_label": torch.randint(0, 2, (2,)),
        "nine_class_label": torch.randint(0, 9, (2,)),
    }

    # training_stepでKL lossが含まれることを確認
    loss = model.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__])
