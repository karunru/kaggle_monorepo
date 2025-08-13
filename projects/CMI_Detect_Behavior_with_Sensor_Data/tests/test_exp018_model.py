"""exp018のモデルテスト."""

import pytest
import torch

from codes.exp.exp018.config import Config
from codes.exp.exp018.model import CMISqueezeformer


class TestCMISqueezeformerBERT:
    """CMISqueezeformer with BERTのテスト."""

    @pytest.fixture
    def config(self):
        """テスト用設定."""
        config = Config()
        # テスト用に小さなパラメータを設定
        config.model.input_dim = 16
        config.model.d_model = 32
        config.model.n_layers = 2
        config.model.n_heads = 4
        config.model.d_ff = 128
        config.model.num_classes = 18
        config.model.dropout = 0.1
        
        # BERT設定
        config.bert.hidden_size = 32
        config.bert.num_layers = 2
        config.bert.num_heads = 4
        config.bert.intermediate_size = 128
        
        # Demographics無効化（テスト簡略化のため）
        config.demographics.enabled = False
        
        return config

    @pytest.fixture
    def model(self, config):
        """テスト用モデル."""
        model = CMISqueezeformer(
            input_dim=config.model.input_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            num_classes=config.model.num_classes,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            bert_config=config.bert.model_dump(),
            demographics_config=config.demographics.model_dump(),
        )
        model.eval()
        return model

    @pytest.fixture
    def sample_input(self, config):
        """テスト用入力データ."""
        batch_size = 4
        seq_len = 100
        input_dim = config.model.input_dim
        
        imu = torch.randn(batch_size, input_dim, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        return imu, attention_mask

    def test_model_initialization(self, model):
        """モデルの初期化テスト."""
        assert model is not None
        assert hasattr(model, 'cls_token')
        assert hasattr(model, 'bert')
        assert hasattr(model, 'multiclass_head')
        assert hasattr(model, 'binary_head')
        
        # CLSトークンの形状チェック
        assert model.cls_token.shape == (1, 1, 32)

    def test_forward_pass_basic(self, model, sample_input):
        """基本的なforward passテスト."""
        imu, attention_mask = sample_input
        
        with torch.no_grad():
            multiclass_logits, binary_logits = model(imu, attention_mask)
        
        # 出力形状の確認
        assert multiclass_logits.shape == (4, 18)  # [batch_size, num_classes]
        assert binary_logits.shape == (4, 1)       # [batch_size, 1]
        
        # 出力がNaNでないことを確認
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()

    def test_forward_pass_without_attention_mask(self, model, sample_input):
        """attention_maskなしのforward passテスト."""
        imu, _ = sample_input
        
        with torch.no_grad():
            multiclass_logits, binary_logits = model(imu, attention_mask=None)
        
        assert multiclass_logits.shape == (4, 18)
        assert binary_logits.shape == (4, 1)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()

    def test_gradient_flow(self, model, sample_input):
        """勾配が正しく流れることを確認."""
        model.train()
        imu, attention_mask = sample_input
        
        multiclass_logits, binary_logits = model(imu, attention_mask)
        
        # ダミー損失を計算
        multiclass_target = torch.randint(0, 18, (4,))
        binary_target = torch.randint(0, 2, (4, 1)).float()
        
        multiclass_loss = torch.nn.CrossEntropyLoss()(multiclass_logits, multiclass_target)
        binary_loss = torch.nn.BCEWithLogitsLoss()(binary_logits, binary_target)
        total_loss = multiclass_loss + binary_loss
        
        # 勾配計算
        total_loss.backward()
        
        # 勾配が存在することを確認
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_different_sequence_lengths(self, model, config):
        """異なる系列長でのテスト."""
        batch_size = 2
        input_dim = config.model.input_dim
        
        # 異なる系列長
        for seq_len in [50, 100, 200]:
            imu = torch.randn(batch_size, input_dim, seq_len)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            
            with torch.no_grad():
                multiclass_logits, binary_logits = model(imu, attention_mask)
            
            assert multiclass_logits.shape == (batch_size, 18)
            assert binary_logits.shape == (batch_size, 1)

    def test_masked_sequences(self, model, sample_input):
        """マスクされた系列のテスト."""
        imu, _ = sample_input
        batch_size, input_dim, seq_len = imu.shape
        
        # 部分的にマスクされたattention_mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, 50:] = False  # 最初のサンプルの後半をマスク
        attention_mask[1, 80:] = False  # 2番目のサンプルの後半をマスク
        
        with torch.no_grad():
            multiclass_logits, binary_logits = model(imu, attention_mask)
        
        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()

    def test_bert_config_variations(self, config):
        """異なるBERT設定でのテスト."""
        # 異なるBERT設定パターン
        bert_configs = [
            {"hidden_size": 32, "num_layers": 1, "num_heads": 2, "intermediate_size": 64},
            {"hidden_size": 64, "num_layers": 3, "num_heads": 8, "intermediate_size": 256},
        ]
        
        for bert_config in bert_configs:
            model = CMISqueezeformer(
                input_dim=config.model.input_dim,
                d_model=config.model.d_model,
                n_layers=config.model.n_layers,
                n_heads=config.model.n_heads,
                d_ff=config.model.d_ff,
                num_classes=config.model.num_classes,
                kernel_size=config.model.kernel_size,
                dropout=config.model.dropout,
                bert_config=bert_config,
                demographics_config=config.demographics.model_dump(),
            )
            
            # 簡単なforward pass
            imu = torch.randn(2, config.model.input_dim, 50)
            attention_mask = torch.ones(2, 50, dtype=torch.bool)
            
            with torch.no_grad():
                multiclass_logits, binary_logits = model(imu, attention_mask)
            
            assert multiclass_logits.shape == (2, 18)
            assert binary_logits.shape == (2, 1)

    def test_model_device_consistency(self, model, sample_input):
        """モデルとデータのデバイス一貫性テスト."""
        imu, attention_mask = sample_input
        
        # CPUでのテスト
        model = model.to('cpu')
        imu = imu.to('cpu')
        attention_mask = attention_mask.to('cpu')
        
        with torch.no_grad():
            multiclass_logits, binary_logits = model(imu, attention_mask)
        
        assert multiclass_logits.device.type == 'cpu'
        assert binary_logits.device.type == 'cpu'

    def test_model_memory_efficiency(self, config):
        """メモリ効率のテスト（大きなモデルの場合）."""
        # より大きなモデル設定
        large_config = config.model_copy()
        large_config.bert.hidden_size = 128
        large_config.bert.num_layers = 4
        
        model = CMISqueezeformer(
            input_dim=large_config.model.input_dim,
            d_model=large_config.model.d_model,
            n_layers=large_config.model.n_layers,
            n_heads=large_config.model.n_heads,
            d_ff=large_config.model.d_ff,
            num_classes=large_config.model.num_classes,
            kernel_size=large_config.model.kernel_size,
            dropout=large_config.model.dropout,
            bert_config=large_config.bert.model_dump(),
            demographics_config=large_config.demographics.model_dump(),
        )
        
        # パラメータ数の確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # 全パラメータが訓練可能であることを確認
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # テストの実行例
    pytest.main([__file__, "-v"])