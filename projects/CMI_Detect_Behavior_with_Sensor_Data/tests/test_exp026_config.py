"""exp026設定のテストコード."""

import torch
from codes.exp.exp026.config import Config
from codes.exp.exp026.model import CMISqueezeformer


class TestExp026Config:
    """exp026の設定変更をテストするクラス."""

    def test_loss_config_default_type(self):
        """LossConfig.typeのデフォルト値がsoft_f1であることを確認."""
        config = Config()
        assert config.loss.type == "soft_f1", f"Expected 'soft_f1', but got '{config.loss.type}'"

    def test_soft_f1_parameters(self):
        """SoftF1損失関数のパラメータが正しく設定されることを確認."""
        config = Config()
        assert config.loss.soft_f1_beta == 1.0, f"Expected beta=1.0, but got {config.loss.soft_f1_beta}"
        assert config.loss.soft_f1_eps == 1e-6, f"Expected eps=1e-6, but got {config.loss.soft_f1_eps}"

    def test_model_initialization_with_soft_f1(self):
        """SoftF1損失を使用してモデルが正常に初期化されることを確認."""
        config = Config()
        
        # モデル作成
        model = CMISqueezeformer(
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            num_classes=config.model.num_classes,
            loss_config=dict(config.loss)
        )

        # 損失関数が正しく設定されていることを確認
        assert hasattr(model, "multiclass_criterion"), "multiclass_criterion not found"
        assert hasattr(model, "binary_criterion"), "binary_criterion not found"
        assert hasattr(model, "nine_class_criterion"), "nine_class_criterion not found"

    def test_soft_f1_loss_computation(self):
        """SoftF1損失の計算が正常に実行されることを確認."""
        config = Config()
        
        # 小さなモデルで計算テスト
        model = CMISqueezeformer(
            input_dim=7,  # IMUデータのみ
            d_model=64,
            n_layers=1,
            n_heads=4,
            d_ff=256,
            num_classes=config.model.num_classes,
            loss_config=dict(config.loss)
        )
        
        # ダミーデータでテスト
        batch_size = 2
        seq_len = 50
        
        # IMUデータのみのテスト（簡素化）
        imu_data = torch.randn(batch_size, 7, seq_len)  # [batch, input_dim, seq_len]
        
        # ターゲットデータ
        multiclass_targets = torch.randint(0, config.model.num_classes, (batch_size,))
        binary_targets = torch.randint(0, 2, (batch_size,))
        nine_class_targets = torch.randint(0, 9, (batch_size,))
        
        # フォワードパス
        multiclass_logits, binary_logits, nine_class_logits = model(imu_data)
        
        # 出力の構造を確認
        assert multiclass_logits.shape == (batch_size, config.model.num_classes), f"Expected shape {(batch_size, config.model.num_classes)}, got {multiclass_logits.shape}"
        assert binary_logits.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {binary_logits.shape}"
        assert nine_class_logits.shape == (batch_size, 9), f"Expected shape {(batch_size, 9)}, got {nine_class_logits.shape}"
        
        # SoftF1損失の計算テスト
        import torch.nn.functional as F
        multiclass_probs = F.softmax(multiclass_logits, dim=-1)
        binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
        nine_class_probs = F.softmax(nine_class_logits, dim=-1)
        
        multiclass_loss = model.multiclass_criterion(multiclass_probs, multiclass_targets)
        binary_loss = model.binary_criterion(binary_probs, binary_targets.float())
        nine_class_loss = model.nine_class_criterion(nine_class_probs, nine_class_targets)
        
        # 損失値が有効であることを確認
        assert torch.isfinite(multiclass_loss), f"Multiclass loss is not finite: {multiclass_loss}"
        assert torch.isfinite(binary_loss), f"Binary loss is not finite: {binary_loss}"
        assert torch.isfinite(nine_class_loss), f"Nine class loss is not finite: {nine_class_loss}"
        assert multiclass_loss.item() >= 0, f"Multiclass loss is negative: {multiclass_loss.item()}"
        assert binary_loss.item() >= 0, f"Binary loss is negative: {binary_loss.item()}"
        assert nine_class_loss.item() >= 0, f"Nine class loss is negative: {nine_class_loss.item()}"

    def test_config_differences_from_exp025(self):
        """exp026がexp025から適切に変更されていることを確認."""
        # このテストはexp025との差分を確認するためのもの
        config = Config()

        # 主要な変更点：損失関数タイプがsoft_f1であること
        assert config.loss.type == "soft_f1", "Loss type should be changed to soft_f1"

        # その他の設定は変更されていないことを確認
        assert config.loss.alpha == 0.5, "Alpha should remain unchanged"
        assert config.loss.nine_class_head_enabled is True, "Nine class head should remain enabled"
        assert config.loss.auto_weighting == "uncertainty", "Auto weighting should remain unchanged"
