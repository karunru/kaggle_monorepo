"""Tests for the model module."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import CMISqueezeformer, ConvolutionModule, PositionalEncoding, SqueezeformerBlock


class TestPositionalEncoding:
    """Test cases for PositionalEncoding."""

    def test_init(self):
        """Test positional encoding initialization."""
        pe = PositionalEncoding(d_model=128, dropout=0.1)
        assert pe.pe.shape[1] == 128

    def test_forward(self):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(d_model=128)
        x = torch.randn(2, 50, 128)  # [batch, seq_len, d_model]

        output = pe(x)
        assert output.shape == x.shape


class TestConvolutionModule:
    """Test cases for ConvolutionModule."""

    def test_init(self):
        """Test convolution module initialization."""
        conv_module = ConvolutionModule(d_model=256, kernel_size=31)
        assert isinstance(conv_module.pointwise_conv1, nn.Conv1d)
        assert isinstance(conv_module.depthwise_conv, nn.Conv1d)

    def test_forward(self):
        """Test convolution module forward pass."""
        conv_module = ConvolutionModule(d_model=256)
        x = torch.randn(2, 50, 256)  # [batch, seq_len, d_model]

        output = conv_module(x)
        assert output.shape == x.shape


class TestSqueezeformerBlock:
    """Test cases for SqueezeformerBlock."""

    def test_init(self):
        """Test squeezeformer block initialization."""
        block = SqueezeformerBlock(d_model=256, n_heads=8, d_ff=1024)
        assert isinstance(block.self_attn, nn.MultiheadAttention)
        assert isinstance(block.conv_module, ConvolutionModule)

    def test_forward(self):
        """Test squeezeformer block forward pass."""
        block = SqueezeformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 50, 256)  # [batch, seq_len, d_model]

        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_mask(self):
        """Test squeezeformer block forward pass with attention mask."""
        block = SqueezeformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 50, 256)
        mask = torch.zeros(50, 50)  # No masking

        output = block(x, attention_mask=mask)
        assert output.shape == x.shape


class TestCMISqueezeformer:
    """Test cases for CMISqueezeformer."""

    @pytest.fixture
    def model_config(self):
        """Default model configuration."""
        return {
            "input_dim": 100,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 1024,
            "num_classes": 18,
            "dropout": 0.1,
        }

    def test_init(self, model_config):
        """Test model initialization."""
        model = CMISqueezeformer(**model_config)

        assert model.input_dim == 100
        assert model.d_model == 256
        assert model.num_classes == 18
        assert len(model.blocks) == 4

    def test_forward_full_model(self, model_config):
        """Test forward pass with full model."""
        model = CMISqueezeformer(**model_config)
        x = torch.randn(2, 100, 200)  # [batch, features, seq_len]

        multiclass_logits, binary_logits = model(x, imu_only=False)

        assert multiclass_logits.shape == (2, 18)  # [batch, num_classes]
        assert binary_logits.shape == (2, 1)  # [batch, 1]

    def test_forward_imu_only(self, model_config):
        """Test forward pass with IMU-only mode."""
        model = CMISqueezeformer(**model_config)
        x = torch.randn(2, 100, 200)  # [batch, features, seq_len]

        multiclass_logits, binary_logits = model(x, imu_only=True)

        assert multiclass_logits.shape == (2, 18)  # [batch, num_classes]
        assert binary_logits.shape == (2, 1)  # [batch, 1]

    def test_different_input_sizes(self, model_config):
        """Test model with different input sizes."""
        # Test with different sequence lengths
        model = CMISqueezeformer(**model_config)

        # Short sequence
        x_short = torch.randn(2, 100, 50)
        mc_logits, bin_logits = model(x_short)
        assert mc_logits.shape == (2, 18)
        assert bin_logits.shape == (2, 1)

        # Long sequence
        x_long = torch.randn(2, 100, 500)
        mc_logits, bin_logits = model(x_long)
        assert mc_logits.shape == (2, 18)
        assert bin_logits.shape == (2, 1)

    def test_batch_sizes(self, model_config):
        """Test model with different batch sizes."""
        model = CMISqueezeformer(**model_config)

        # Single sample
        x_single = torch.randn(1, 100, 200)
        mc_logits, bin_logits = model(x_single)
        assert mc_logits.shape == (1, 18)
        assert bin_logits.shape == (1, 1)

        # Large batch
        x_large = torch.randn(16, 100, 200)
        mc_logits, bin_logits = model(x_large)
        assert mc_logits.shape == (16, 18)
        assert bin_logits.shape == (16, 1)

    def test_model_parameters(self, model_config):
        """Test model parameter count."""
        model = CMISqueezeformer(**model_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

    def test_gradient_flow(self, model_config):
        """Test that gradients flow properly through the model."""
        model = CMISqueezeformer(**model_config)
        x = torch.randn(2, 100, 200, requires_grad=True)

        multiclass_logits, binary_logits = model(x)
        loss = multiclass_logits.sum() + binary_logits.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode(self, model_config):
        """Test model in evaluation mode."""
        model = CMISqueezeformer(**model_config)
        x = torch.randn(2, 100, 200)

        # Train mode
        model.train()
        out1_mc, out1_bin = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            out2_mc, out2_bin = model(x)

        # Outputs should be different due to dropout
        model.train()
        out3_mc, out3_bin = model(x)

        # In eval mode, outputs should be deterministic
        model.eval()
        with torch.no_grad():
            out4_mc, out4_bin = model(x)

        torch.testing.assert_close(out2_mc, out4_mc)
        torch.testing.assert_close(out2_bin, out4_bin)


if __name__ == "__main__":
    pytest.main([__file__])
