"""exp027の設定テスト."""


from codes.exp.exp027.config import Config, config


class TestExp027Config:
    """exp027設定のテスト."""

    def test_exp_num(self):
        """実験番号がexp027であることを確認."""
        assert config.experiment.exp_num == "exp027"
        assert config.experiment.name == "exp027_focal_loss_nine_class"

    def test_description(self):
        """実験説明が正しく設定されていることを確認."""
        expected_description = "FocalLoss for both multiclass and nine-class heads"
        assert config.experiment.description == expected_description

    def test_tags(self):
        """実験タグが正しく設定されていることを確認."""
        expected_tags = [
            "focal_loss",
            "nine_class_focal",
            "multiclass_consistency",
            "squeezeformer",
            "pytorch_lightning",
        ]
        assert config.experiment.tags == expected_tags

    def test_wandb_tags(self):
        """WandBタグが正しく設定されていることを確認."""
        expected_tags = ["exp027", "focal_loss", "nine_class_focal", "multiclass_consistency", "squeezeformer"]
        assert config.logging.wandb_tags == expected_tags

    def test_wandb_name(self):
        """WandB実行名が正しく設定されていることを確認."""
        assert config.logging.wandb_name == "exp027"

    def test_loss_config(self):
        """損失関数設定が正しく設定されていることを確認."""
        # デフォルトでcmi_focalが使用されることを確認
        assert hasattr(config.loss, "type")
        assert hasattr(config.loss, "focal_gamma")
        assert hasattr(config.loss, "focal_alpha")
        assert hasattr(config.loss, "label_smoothing")

    def test_config_instantiation(self):
        """Config クラスが正しくインスタンス化できることを確認."""
        test_config = Config()
        assert test_config is not None
        assert hasattr(test_config, "experiment")
        assert hasattr(test_config, "loss")
        assert hasattr(test_config, "logging")
