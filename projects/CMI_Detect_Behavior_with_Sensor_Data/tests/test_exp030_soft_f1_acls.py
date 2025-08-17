"""exp030 SoftF1ACLSの動作テスト."""

import pytest
import torch

# exp030のlossesからインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../codes/exp/exp030'))

from losses import MulticlassSoftF1ACLS, BinarySoftF1ACLS


class TestMulticlassSoftF1ACLS:
    """MulticlassSoftF1ACLSのテストクラス."""

    def test_initialization(self):
        """初期化のテスト."""
        criterion = MulticlassSoftF1ACLS(
            num_classes=5,
            beta=1.0,
            eps=1e-6,
            pos_lambda=1.0,
            neg_lambda=0.1,
            alpha=0.1,
            margin=10.0,
        )
        
        assert criterion.num_classes == 5
        assert criterion.beta == 1.0
        assert criterion.eps == 1e-6
        assert criterion.pos_lambda == 1.0
        assert criterion.neg_lambda == 0.1
        assert criterion.alpha == 0.1
        assert criterion.margin == 10.0

    def test_forward_with_logits(self):
        """ロジット入力でのフォワードテスト."""
        criterion = MulticlassSoftF1ACLS(num_classes=3)
        
        # テストデータ
        batch_size = 4
        logits = torch.randn(batch_size, 3, requires_grad=True)
        targets = torch.randint(0, 3, (batch_size,))
        
        # 損失計算
        loss = criterion(logits, targets)
        
        # 基本的な検証
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0  # スカラー
        assert loss.item() >= 0  # 損失は非負

    def test_forward_with_probs(self):
        """確率入力でのフォワードテスト."""
        criterion = MulticlassSoftF1ACLS(num_classes=3)
        
        # テストデータ（確率値）
        batch_size = 4
        probs = torch.softmax(torch.randn(batch_size, 3), dim=-1)
        targets = torch.randint(0, 3, (batch_size,))
        
        # 損失計算
        loss = criterion(probs, targets)
        
        # 基本的な検証
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # スカラー
        assert loss.item() >= 0  # 損失は非負

    def test_zero_regularization(self):
        """正則化項がゼロの場合のテスト."""
        criterion = MulticlassSoftF1ACLS(num_classes=3, alpha=0.0)
        
        batch_size = 4
        logits = torch.randn(batch_size, 3, requires_grad=True)
        targets = torch.randint(0, 3, (batch_size,))
        
        loss = criterion(logits, targets)
        
        # alphaが0の場合、SoftF1Lossのみが計算される
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """勾配計算のテスト."""
        criterion = MulticlassSoftF1ACLS(num_classes=3)
        
        batch_size = 4
        logits = torch.randn(batch_size, 3, requires_grad=True)
        targets = torch.randint(0, 3, (batch_size,))
        
        loss = criterion(logits, targets)
        loss.backward()
        
        # 勾配が計算されているか確認
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestBinarySoftF1ACLS:
    """BinarySoftF1ACLSのテストクラス."""

    def test_initialization(self):
        """初期化のテスト."""
        criterion = BinarySoftF1ACLS(
            beta=1.0,
            eps=1e-6,
            pos_lambda=1.0,
            neg_lambda=0.1,
            alpha=0.1,
            margin=10.0,
        )
        
        assert criterion.beta == 1.0
        assert criterion.eps == 1e-6
        assert criterion.pos_lambda == 1.0
        assert criterion.neg_lambda == 0.1
        assert criterion.alpha == 0.1
        assert criterion.margin == 10.0

    def test_forward_with_logits(self):
        """ロジット入力でのフォワードテスト."""
        criterion = BinarySoftF1ACLS()
        
        # テストデータ
        batch_size = 4
        logits = torch.randn(batch_size, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        # 損失計算
        loss = criterion(logits, targets)
        
        # 基本的な検証
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0  # スカラー
        assert loss.item() >= 0  # 損失は非負

    def test_forward_with_probs(self):
        """確率入力でのフォワードテスト."""
        criterion = BinarySoftF1ACLS()
        
        # テストデータ（確率値）
        batch_size = 4
        probs = torch.sigmoid(torch.randn(batch_size))
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        # 損失計算
        loss = criterion(probs, targets)
        
        # 基本的な検証
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # スカラー
        assert loss.item() >= 0  # 損失は非負

    def test_zero_regularization(self):
        """正則化項がゼロの場合のテスト."""
        criterion = BinarySoftF1ACLS(alpha=0.0)
        
        batch_size = 4
        logits = torch.randn(batch_size, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = criterion(logits, targets)
        
        # alphaが0の場合、SoftF1Lossのみが計算される
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """勾配計算のテスト."""
        criterion = BinarySoftF1ACLS()
        
        batch_size = 4
        logits = torch.randn(batch_size, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = criterion(logits, targets)
        loss.backward()
        
        # 勾配が計算されているか確認
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_2d_input_handling(self):
        """2次元入力（[batch, 1]）の処理テスト."""
        criterion = BinarySoftF1ACLS()
        
        batch_size = 4
        logits_2d = torch.randn(batch_size, 1, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = criterion(logits_2d, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestSoftF1ACLSComparison:
    """SoftF1ACLSと他の損失の比較テスト."""

    def test_compared_to_cross_entropy(self):
        """CrossEntropyとの比較テスト."""
        # SoftF1ACLS
        soft_f1_acls = MulticlassSoftF1ACLS(num_classes=3, alpha=0.0)  # ACLS正則化なし
        
        # CrossEntropy
        ce_loss = torch.nn.CrossEntropyLoss()
        
        # テストデータ
        batch_size = 4
        logits = torch.randn(batch_size, 3, requires_grad=True)
        targets = torch.randint(0, 3, (batch_size,))
        
        # 損失計算
        sf1_loss = soft_f1_acls(logits, targets)
        ce_loss_val = ce_loss(logits, targets)
        
        # どちらも計算できることを確認
        assert isinstance(sf1_loss, torch.Tensor)
        assert isinstance(ce_loss_val, torch.Tensor)
        
        # 値の妥当性確認
        assert sf1_loss.item() >= 0
        assert ce_loss_val.item() >= 0

    def test_regularization_effect(self):
        """正則化項の効果テスト."""
        # 正則化なし
        criterion_no_reg = MulticlassSoftF1ACLS(num_classes=3, alpha=0.0)
        
        # 正則化あり
        criterion_with_reg = MulticlassSoftF1ACLS(num_classes=3, alpha=0.1)
        
        # テストデータ
        batch_size = 4
        logits = torch.randn(batch_size, 3, requires_grad=True)
        targets = torch.randint(0, 3, (batch_size,))
        
        # 損失計算
        loss_no_reg = criterion_no_reg(logits, targets)
        loss_with_reg = criterion_with_reg(logits, targets)
        
        # 正則化項がある場合、通常は損失が大きくなる
        assert isinstance(loss_no_reg, torch.Tensor)
        assert isinstance(loss_with_reg, torch.Tensor)
        assert loss_no_reg.item() >= 0
        assert loss_with_reg.item() >= 0


if __name__ == "__main__":
    """直接実行時のテスト."""
    # 簡単な動作確認
    print("Testing MulticlassSoftF1ACLS...")
    
    # Multiclass test
    criterion = MulticlassSoftF1ACLS(num_classes=5)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    loss = criterion(logits, targets)
    print(f"Multiclass SoftF1ACLS loss: {loss.item():.4f}")
    
    # Binary test
    print("\nTesting BinarySoftF1ACLS...")
    bin_criterion = BinarySoftF1ACLS()
    bin_logits = torch.randn(8, requires_grad=True)
    bin_targets = torch.randint(0, 2, (8,)).float()
    bin_loss = bin_criterion(bin_logits, bin_targets)
    print(f"Binary SoftF1ACLS loss: {bin_loss.item():.4f}")
    
    # Gradient test
    print("\nTesting gradients...")
    loss.backward()
    print(f"Logits grad norm: {logits.grad.norm().item():.4f}")
    
    bin_loss.backward()
    print(f"Binary logits grad norm: {bin_logits.grad.norm().item():.4f}")
    
    print("\nAll tests completed successfully!")