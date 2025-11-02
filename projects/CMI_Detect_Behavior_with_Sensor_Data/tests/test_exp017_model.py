"""Test model module for exp017."""

import torch
from codes.exp.exp017.model import CMISqueezeformer


def test_model_creation():
    """Test that CMISqueezeformer can be created."""
    config = {
        "model": {
            "input_dim": 7,  # 6 IMU + 1 physics feature
            "hidden_dim": 128,
            "num_classes": 18,  # 18 gesture classes
            "num_blocks": 4,
            "num_heads": 8,
            "conv_expansion_factor": 2,
            "feed_forward_expansion_factor": 4,
            "conv_kernel_size": 31,
            "feed_forward_dropout": 0.1,
            "attention_dropout": 0.1,
            "conv_dropout": 0.1,
            "use_demographics": False,
        },
        "loss": {
            "type": "soft_f1",
            "alpha": 0.5,
            "soft_f1_beta": 1.0,
            "soft_f1_eps": 1e-6,
        },
        "acls": {
            "acls_pos_lambda": 1.0,
            "acls_neg_lambda": 0.1,
            "acls_alpha": 0.1,
            "acls_margin": 10.0,
            "label_smoothing_alpha": 0.1,
            "mbls_margin": 10.0,
            "mbls_alpha": 0.1,
            "mbls_schedule": None,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
        },
        "schedulefree": {
            "enabled": False,
        },
        "ema": {
            "enabled": False,
        },
        "demographics": {
            "embedding_dim": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
        },
    }

    model = CMISqueezeformer(config)
    assert model is not None

    # Test forward pass
    batch_size = 2
    seq_len = 100
    input_dim = 7

    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    multiclass_logits, binary_logits = model(x, mask)

    assert multiclass_logits.shape == (batch_size, 18)
    assert binary_logits.shape == (batch_size, 1)


def test_model_loss_type():
    """Test that model uses soft_f1 loss by default."""
    config = {
        "model": {
            "input_dim": 7,
            "hidden_dim": 128,
            "num_classes": 18,
            "num_blocks": 4,
            "num_heads": 8,
            "conv_expansion_factor": 2,
            "feed_forward_expansion_factor": 4,
            "conv_kernel_size": 31,
            "feed_forward_dropout": 0.1,
            "attention_dropout": 0.1,
            "conv_dropout": 0.1,
            "use_demographics": False,
        },
        "loss": {
            "type": "soft_f1",
            "alpha": 0.5,
            "soft_f1_beta": 1.0,
            "soft_f1_eps": 1e-6,
        },
        "acls": {
            "acls_pos_lambda": 1.0,
            "acls_neg_lambda": 0.1,
            "acls_alpha": 0.1,
            "acls_margin": 10.0,
            "label_smoothing_alpha": 0.1,
            "mbls_margin": 10.0,
            "mbls_alpha": 0.1,
            "mbls_schedule": None,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
        },
        "schedulefree": {
            "enabled": False,
        },
        "ema": {
            "enabled": False,
        },
        "demographics": {
            "embedding_dim": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
        },
    }

    model = CMISqueezeformer(config)

    # Check that loss functions are correctly initialized
    from codes.exp.exp017.model import BinarySoftF1Loss, MulticlassSoftF1Loss

    assert isinstance(model.multiclass_criterion, MulticlassSoftF1Loss)
    assert isinstance(model.binary_criterion, BinarySoftF1Loss)
