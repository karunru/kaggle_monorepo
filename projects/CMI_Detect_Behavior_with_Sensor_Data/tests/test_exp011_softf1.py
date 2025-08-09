"""exp011用のSoftF1Lossのテストコード."""

import pytest
import torch
import torch.nn.functional as F

# exp011からインポート
import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "codes" / "exp" / "exp011"))

from model import BinarySoftF1Loss, MulticlassSoftF1Loss, CMISqueezeformer


class TestBinarySoftF1Loss:
    """BinarySoftF1Lossのテストクラス."""

    def test_binary_soft_f1_loss_init(self):
        """初期化のテスト."""
        loss_fn = BinarySoftF1Loss()
        assert loss_fn.beta == 1.0
        assert loss_fn.eps == 1e-6

        loss_fn_custom = BinarySoftF1Loss(beta=2.0, eps=1e-8)
        assert loss_fn_custom.beta == 2.0
        assert loss_fn_custom.eps == 1e-8

    def test_binary_soft_f1_loss_perfect_prediction(self):
        """完全予測時のテスト."""
        loss_fn = BinarySoftF1Loss()
        
        # 完全予測（F1=1.0、Loss=0.0）
        probs = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        
        loss = loss_fn(probs, targets)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_binary_soft_f1_loss_worst_prediction(self):
        """最悪予測時のテスト."""
        loss_fn = BinarySoftF1Loss()
        
        # 最悪予測（F1=0.0、Loss=1.0）
        probs = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        
        loss = loss_fn(probs, targets)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_binary_soft_f1_loss_gradient(self):
        """勾配計算のテスト."""
        loss_fn = BinarySoftF1Loss()
        
        probs = torch.tensor([0.8, 0.6, 0.3, 0.2], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        
        loss = loss_fn(probs, targets)
        loss.backward()
        
        # 勾配が計算されていることを確認
        assert probs.grad is not None
        assert not torch.allclose(probs.grad, torch.zeros_like(probs.grad))

    def test_binary_soft_f1_loss_range(self):
        """損失値の範囲テスト."""
        loss_fn = BinarySoftF1Loss()
        
        # ランダムな予測値でテスト
        torch.manual_seed(42)
        for _ in range(10):
            probs = torch.rand(20, dtype=torch.float32)
            targets = torch.randint(0, 2, (20,), dtype=torch.float32)
            
            loss = loss_fn(probs, targets)
            
            # 損失値が0-1の範囲内であることを確認
            assert 0.0 <= loss.item() <= 1.0


class TestMulticlassSoftF1Loss:
    """MulticlassSoftF1Lossのテストクラス."""

    def test_multiclass_soft_f1_loss_init(self):
        """初期化のテスト."""
        loss_fn = MulticlassSoftF1Loss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert loss_fn.beta == 1.0
        assert loss_fn.eps == 1e-6

        loss_fn_custom = MulticlassSoftF1Loss(num_classes=10, beta=2.0, eps=1e-8)
        assert loss_fn_custom.num_classes == 10
        assert loss_fn_custom.beta == 2.0
        assert loss_fn_custom.eps == 1e-8

    def test_multiclass_soft_f1_loss_perfect_prediction(self):
        """完全予測時のテスト."""
        num_classes = 3
        loss_fn = MulticlassSoftF1Loss(num_classes=num_classes)
        
        # 完全予測（one-hot）
        probs = torch.tensor([
            [1.0, 0.0, 0.0],  # クラス0
            [0.0, 1.0, 0.0],  # クラス1
            [0.0, 0.0, 1.0],  # クラス2
        ], dtype=torch.float32)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        
        loss = loss_fn(probs, targets)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_multiclass_soft_f1_loss_gradient(self):
        """勾配計算のテスト."""
        num_classes = 4
        loss_fn = MulticlassSoftF1Loss(num_classes=num_classes)
        
        # Softmax後の確率値
        logits = torch.randn(8, num_classes, requires_grad=True)
        probs = F.softmax(logits, dim=-1)
        targets = torch.randint(0, num_classes, (8,), dtype=torch.long)
        
        loss = loss_fn(probs, targets)
        loss.backward()
        
        # 勾配が計算されていることを確認
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_multiclass_soft_f1_loss_range(self):
        """損失値の範囲テスト."""
        num_classes = 5
        loss_fn = MulticlassSoftF1Loss(num_classes=num_classes)
        
        # ランダムな予測値でテスト
        torch.manual_seed(42)
        for _ in range(10):
            logits = torch.randn(16, num_classes)
            probs = F.softmax(logits, dim=-1)
            targets = torch.randint(0, num_classes, (16,), dtype=torch.long)
            
            loss = loss_fn(probs, targets)
            
            # 損失値が0-1の範囲内であることを確認
            assert 0.0 <= loss.item() <= 1.0

    def test_multiclass_soft_f1_loss_shape_validation(self):
        """入力形状の検証テスト."""
        num_classes = 3
        loss_fn = MulticlassSoftF1Loss(num_classes=num_classes)
        
        # 正しい形状
        probs = torch.rand(5, num_classes)
        probs = F.softmax(probs, dim=-1)  # 確率化
        targets = torch.randint(0, num_classes, (5,))
        
        loss = loss_fn(probs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー


class TestCMISqueezeformerSoftF1Integration:
    """CMISqueezeformerとSoftF1Lossの統合テスト."""

    def test_model_with_soft_f1_loss(self):
        """SoftF1Loss使用時のモデル動作テスト."""
        # SoftF1Loss設定
        loss_config = {
            "type": "soft_f1",
            "alpha": 0.5,
            "soft_f1_beta": 1.0,
            "soft_f1_eps": 1e-6,
        }
        
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_classes=18,
            loss_config=loss_config,
        )
        
        # テストデータ
        batch_size = 4
        seq_len = 100
        imu_data = torch.randn(batch_size, 7, seq_len)
        multiclass_labels = torch.randint(0, 18, (batch_size,))
        binary_labels = torch.randint(0, 2, (batch_size,)).float()
        
        batch = {
            "imu": imu_data,
            "multiclass_label": multiclass_labels,
            "binary_label": binary_labels,
        }
        
        # 訓練ステップ
        loss = model.training_step(batch, 0)
        
        # 損失値の検証
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー
        assert loss.item() >= 0.0  # 非負
        
        # 勾配計算の確認
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_loss_function_setup(self):
        """損失関数設定のテスト."""
        # SoftF1Loss設定
        loss_config = {
            "type": "soft_f1",
            "alpha": 0.7,
            "soft_f1_beta": 2.0,
            "soft_f1_eps": 1e-8,
        }
        
        model = CMISqueezeformer(
            input_dim=7,
            num_classes=18,
            loss_config=loss_config,
        )
        
        # 損失関数の型チェック
        assert isinstance(model.multiclass_criterion, MulticlassSoftF1Loss)
        assert isinstance(model.binary_criterion, BinarySoftF1Loss)
        
        # パラメータチェック
        assert model.multiclass_criterion.beta == 2.0
        assert model.multiclass_criterion.eps == 1e-8
        assert model.binary_criterion.beta == 2.0
        assert model.binary_criterion.eps == 1e-8
        assert model.loss_alpha == 0.7

    def test_model_validation_step_soft_f1(self):
        """検証ステップでのSoftF1Loss動作テスト."""
        loss_config = {"type": "soft_f1"}
        
        model = CMISqueezeformer(
            input_dim=7,
            d_model=32,
            n_layers=1,
            num_classes=18,
            loss_config=loss_config,
        )
        
        # テストデータ
        batch = {
            "imu": torch.randn(2, 7, 50),
            "multiclass_label": torch.randint(0, 18, (2,)),
            "binary_label": torch.randint(0, 2, (2,)).float(),
            "sequence_id": ["seq1", "seq2"],
            "gesture": ["gesture1", "gesture2"],
        }
        
        # 検証ステップ
        output = model.validation_step(batch, 0)
        
        # 出力の検証
        assert isinstance(output, dict)
        assert "val_loss" in output
        assert "multiclass_probs" in output
        assert "binary_probs" in output
        
        # 確率値の範囲チェック
        multiclass_probs = output["multiclass_probs"]
        binary_probs = output["binary_probs"]
        
        assert torch.all(multiclass_probs >= 0.0) and torch.all(multiclass_probs <= 1.0)
        assert torch.all(binary_probs >= 0.0) and torch.all(binary_probs <= 1.0)
        assert torch.allclose(multiclass_probs.sum(dim=-1), torch.ones(2))  # softmax確率合計=1


class TestSoftF1LossComparison:
    """SoftF1Lossと他の損失関数の比較テスト."""

    def test_loss_consistency_binary(self):
        """バイナリ分類での損失関数一貫性テスト."""
        # テストデータ
        probs = torch.tensor([0.9, 0.8, 0.3, 0.1])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        
        # SoftF1Loss
        soft_f1_loss = BinarySoftF1Loss()
        loss_soft_f1 = soft_f1_loss(probs, targets)
        
        # 損失値が妥当な範囲にあることを確認
        assert 0.0 <= loss_soft_f1.item() <= 1.0
        
        # 完全予測で損失が0に近いことを確認
        perfect_probs = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss_perfect = soft_f1_loss(perfect_probs, targets)
        assert loss_perfect.item() < 0.01

    def test_loss_sensitivity(self):
        """損失関数の感度テスト."""
        loss_fn = BinarySoftF1Loss()
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        # 異なる予測確率での損失値
        probs_good = torch.tensor([0.9, 0.1, 0.8, 0.2])  # 良い予測
        probs_bad = torch.tensor([0.3, 0.7, 0.4, 0.6])   # 悪い予測
        
        loss_good = loss_fn(probs_good, targets)
        loss_bad = loss_fn(probs_bad, targets)
        
        # 良い予測の方が損失が小さいことを確認
        assert loss_good.item() < loss_bad.item()


if __name__ == "__main__":
    # 簡単な実行テスト
    print("Running basic SoftF1Loss tests...")
    
    # BinarySoftF1Lossテスト
    binary_loss = BinarySoftF1Loss()
    test_probs = torch.tensor([0.8, 0.3, 0.9, 0.1])
    test_targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    binary_result = binary_loss(test_probs, test_targets)
    print(f"Binary SoftF1Loss: {binary_result.item():.4f}")
    
    # MulticlassSoftF1Lossテスト
    multiclass_loss = MulticlassSoftF1Loss(num_classes=3)
    test_probs_mc = F.softmax(torch.randn(4, 3), dim=-1)
    test_targets_mc = torch.tensor([0, 1, 2, 1])
    multiclass_result = multiclass_loss(test_probs_mc, test_targets_mc)
    print(f"Multiclass SoftF1Loss: {multiclass_result.item():.4f}")
    
    print("Basic tests completed successfully!")