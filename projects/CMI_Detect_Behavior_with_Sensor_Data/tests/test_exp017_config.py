"""Test configuration for exp017."""

from codes.exp.exp017.config import EXP_NUM, Config


def test_exp_num():
    """Test that EXP_NUM is correctly set to exp017."""
    assert EXP_NUM == "exp017"


def test_default_loss_type():
    """Test that default loss type is soft_f1."""
    config = Config()
    assert config.loss.type == "soft_f1"


def test_config_creation():
    """Test that Config can be created with default values."""
    config = Config()
    assert config is not None
    assert config.loss.type == "soft_f1"
    assert config.loss.alpha == 0.5  # Default value
    assert config.loss.soft_f1_beta == 1.0  # Default value
    assert config.loss.soft_f1_eps == 1e-6  # Default value


def test_loss_config_types():
    """Test that all loss types are valid."""
    valid_types = ["cmi", "cmi_focal", "soft_f1", "acls", "label_smoothing", "mbls"]
    config = Config()
    # Defaultはsoft_f1
    assert config.loss.type in valid_types
