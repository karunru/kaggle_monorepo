"""exp021のloss関数のテスト."""

import torch
from codes.exp.exp021.model import FocalLoss
from torch import nn


class TestFocalLoss:
    """FocalLossのテストクラス."""

    def test_focal_loss_initialization(self):
        """FocalLossの初期化テスト."""
        # デフォルトパラメータ
        loss = FocalLoss()
        assert loss.gamma == 2.0
        assert loss.alpha == 1.0
        assert loss.label_smoothing == 0.0

        # カスタムパラメータ
        loss = FocalLoss(gamma=1.5, alpha=0.8, label_smoothing=0.1)
        assert loss.gamma == 1.5
        assert loss.alpha == 0.8
        assert loss.label_smoothing == 0.1

    def test_focal_loss_forward(self):
        """FocalLossのforward計算テスト."""
        batch_size = 4
        num_classes = 3

        # データ準備
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # label_smoothing無しでのテスト
        loss_fn = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)
        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー値
        assert loss.item() >= 0.0  # 損失は非負

    def test_focal_loss_with_label_smoothing(self):
        """label_smoothing有りでのFocalLossテスト."""
        batch_size = 4
        num_classes = 3

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # label_smoothing有りでのテスト
        loss_fn = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.1)
        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_label_smoothing_effect(self):
        """label_smoothingの効果テスト."""
        batch_size = 4
        num_classes = 3

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # label_smoothing無し
        loss_fn_no_smooth = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)
        loss_no_smooth = loss_fn_no_smooth(inputs, targets)

        # label_smoothing有り
        loss_fn_smooth = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.1)
        loss_smooth = loss_fn_smooth(inputs, targets)

        # 損失値が異なることを確認（同じ入力でもlabel_smoothingで結果が変わる）
        assert loss_no_smooth.item() != loss_smooth.item()

    def test_backward_compatibility(self):
        """後方互換性テスト（label_smoothing=0.0で従来と同様の動作）."""
        batch_size = 4
        num_classes = 3

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 旧実装相当（コンストラクタでlabel_smoothingを指定しない）
        loss_fn_old = FocalLoss(gamma=2.0, alpha=1.0)
        loss_old = loss_fn_old(inputs, targets)

        # 新実装でlabel_smoothing=0.0を明示
        loss_fn_new = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.0)
        loss_new = loss_fn_new(inputs, targets)

        # 同じ結果になることを確認
        torch.testing.assert_close(loss_old, loss_new)


class TestCrossEntropyLossWithLabelSmoothing:
    """CrossEntropyLossのlabel_smoothingテスト."""

    def test_cross_entropy_with_label_smoothing(self):
        """label_smoothing付きCrossEntropyLossのテスト."""
        batch_size = 4
        num_classes = 3

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # label_smoothing無し
        criterion_no_smooth = nn.CrossEntropyLoss(label_smoothing=0.0)
        loss_no_smooth = criterion_no_smooth(inputs, targets)

        # label_smoothing有り
        criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_smooth = criterion_smooth(inputs, targets)

        assert isinstance(loss_no_smooth, torch.Tensor)
        assert isinstance(loss_smooth, torch.Tensor)
        assert loss_no_smooth.item() != loss_smooth.item()
