"""exp027のFocalLoss実装テスト."""

import torch
from codes.exp.exp027.model import FocalLoss
from torch import nn


class TestExp027FocalLoss:
    """exp027のFocalLoss実装のテスト."""

    def test_focal_loss_instantiation(self):
        """FocalLossが正しくインスタンス化できることを確認."""
        focal_loss = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.1)
        assert isinstance(focal_loss, nn.Module)
        assert focal_loss.gamma == 2.0
        assert focal_loss.alpha == 1.0
        assert focal_loss.label_smoothing == 0.1

    def test_focal_loss_forward(self):
        """FocalLossの前向き計算が正しく動作することを確認."""
        focal_loss = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)

        # テスト用データ
        batch_size, num_classes = 4, 9
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 損失計算
        loss = focal_loss(inputs, targets)

        # 結果確認
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー値
        assert loss.item() >= 0  # 損失は非負

    def test_focal_loss_with_label_smoothing(self):
        """Label smoothingありのFocalLossが正しく動作することを確認."""
        focal_loss = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.1)

        # テスト用データ
        batch_size, num_classes = 4, 9
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 損失計算
        loss = focal_loss(inputs, targets)

        # 結果確認
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_focal_loss_gradient(self):
        """FocalLossの勾配計算が正しく動作することを確認."""
        focal_loss = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)

        # テスト用データ（勾配計算が必要）
        batch_size, num_classes = 4, 9
        inputs = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 損失計算と逆伝播
        loss = focal_loss(inputs, targets)
        loss.backward()

        # 勾配確認
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape

    def test_focal_loss_vs_cross_entropy(self):
        """gamma=0の時、FocalLossがCrossEntropyと近似することを確認."""
        # FocalLoss (gamma=0, alpha=1, no label smoothing)
        focal_loss = FocalLoss(gamma=0.0, alpha=1.0, label_smoothing=0.0)

        # CrossEntropyLoss
        ce_loss = nn.CrossEntropyLoss()

        # テスト用データ
        batch_size, num_classes = 8, 9
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 損失計算
        focal_result = focal_loss(inputs, targets)
        ce_result = ce_loss(inputs, targets)

        # 近似確認（gamma=0の時、Focal LossはCross Entropyとほぼ同じ）
        assert torch.allclose(focal_result, ce_result, atol=1e-5)

    def test_focal_loss_difficult_examples(self):
        """Focal Lossが困難な例により大きな重みを与えることを確認."""
        focal_loss = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)

        # 簡単な例（高い確率で正解）
        easy_inputs = torch.tensor([[10.0, 0.0, 0.0]])  # 正解ラベル0で高い確率
        easy_targets = torch.tensor([0])

        # 困難な例（低い確率で正解）
        hard_inputs = torch.tensor([[1.0, 0.9, 0.8]])  # 正解ラベル0だが確率は低い
        hard_targets = torch.tensor([0])

        # 損失計算
        easy_loss = focal_loss(easy_inputs, easy_targets)
        hard_loss = focal_loss(hard_inputs, hard_targets)

        # 困難な例の方が大きな損失を持つことを確認
        assert hard_loss.item() > easy_loss.item()
