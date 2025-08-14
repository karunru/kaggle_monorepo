"""exp021のconfig設定のテスト."""

from codes.exp.exp021.config import LossConfig


class TestLossConfig:
    """LossConfigのテストクラス."""

    def test_default_values(self):
        """デフォルト値のテスト."""
        config = LossConfig()
        assert config.type == "acls"
        assert config.alpha == 0.5
        assert config.focal_gamma == 2.0
        assert config.focal_alpha == 1.0
        assert config.soft_f1_beta == 1.0
        assert config.soft_f1_eps == 1e-6
        assert config.label_smoothing == 0.0

    def test_label_smoothing_parameter(self):
        """label_smoothingパラメータのテスト."""
        # デフォルト値
        config = LossConfig()
        assert config.label_smoothing == 0.0

        # カスタム値
        config = LossConfig(label_smoothing=0.1)
        assert config.label_smoothing == 0.1

        # 上限値
        config = LossConfig(label_smoothing=1.0)
        assert config.label_smoothing == 1.0

    def test_cmi_loss_type(self):
        """cmi loss typeのテスト."""
        config = LossConfig(type="cmi", label_smoothing=0.05)
        assert config.type == "cmi"
        assert config.label_smoothing == 0.05

    def test_cmi_focal_loss_type(self):
        """cmi_focal loss typeのテスト."""
        config = LossConfig(type="cmi_focal", focal_gamma=1.5, focal_alpha=0.8, label_smoothing=0.1)
        assert config.type == "cmi_focal"
        assert config.focal_gamma == 1.5
        assert config.focal_alpha == 0.8
        assert config.label_smoothing == 0.1
