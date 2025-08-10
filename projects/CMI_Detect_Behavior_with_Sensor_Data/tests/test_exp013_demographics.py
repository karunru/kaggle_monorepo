"""Test for exp013 demographics integration."""

import sys
from pathlib import Path

import pytest
import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp013"))

from config import Config, DemographicsConfig
from model import CMISqueezeformer, DemographicsEmbedding


class TestDemographicsEmbedding:
    """DemographicsEmbedding クラスのテスト."""

    @pytest.fixture
    def demographics_config(self):
        """テスト用Demographics設定."""
        return DemographicsConfig(
            enabled=True,
            embedding_dim=16,
            categorical_embedding_dims={"adult_child": 2, "sex": 2, "handedness": 2},
            numerical_features=["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
            age_min=10.0,
            age_max=60.0,
            height_min=130.0,
            height_max=200.0,
            shoulder_to_wrist_min=35.0,
            shoulder_to_wrist_max=75.0,
            elbow_to_wrist_min=15.0,
            elbow_to_wrist_max=45.0,
        )

    @pytest.fixture
    def demographics_embedding(self, demographics_config):
        """テスト用DemographicsEmbedding."""
        return DemographicsEmbedding(
            categorical_features=demographics_config.categorical_features,
            numerical_features=demographics_config.numerical_features,
            categorical_embedding_dims=demographics_config.categorical_embedding_dims,
            embedding_dim=demographics_config.embedding_dim,
        )

    @pytest.fixture
    def sample_demographics_data(self):
        """サンプルdemographicsデータ."""
        return {
            "adult_child": torch.tensor([1], dtype=torch.long),  # Adult
            "sex": torch.tensor([0], dtype=torch.long),  # Female
            "handedness": torch.tensor([1], dtype=torch.long),  # Right-handed
            "age": torch.tensor([25.0], dtype=torch.float),
            "height_cm": torch.tensor([170.0], dtype=torch.float),
            "shoulder_to_wrist_cm": torch.tensor([55.0], dtype=torch.float),
            "elbow_to_wrist_cm": torch.tensor([25.0], dtype=torch.float),
        }

    def test_demographics_embedding_forward(self, demographics_embedding, sample_demographics_data):
        """Demographics埋め込みの順伝播をテスト."""
        output = demographics_embedding(sample_demographics_data)

        # 出力テンソルの形状確認
        assert output.shape == (1, 16), f"Expected shape (1, 16), got {output.shape}"

        # 出力値の範囲確認（全て有限値であること）
        assert torch.isfinite(output).all(), "Output contains non-finite values"

        # Gradientが流れることを確認
        demographics_embedding.train()
        output = demographics_embedding(sample_demographics_data)
        loss = output.sum()
        loss.backward()

        # 埋め込み層のgradientが存在することを確認
        for param in demographics_embedding.parameters():
            assert param.grad is not None, "Gradient not computed for demographics embedding"

    def test_demographics_embedding_range_clipping(self, demographics_embedding):
        """範囲外値のクリッピングをテスト."""
        # 範囲外の値を持つデータ
        out_of_range_data = {
            "adult_child": torch.tensor([1], dtype=torch.long),
            "sex": torch.tensor([0], dtype=torch.long),
            "handedness": torch.tensor([1], dtype=torch.long),
            "age": torch.tensor([150.0], dtype=torch.float),  # 範囲外（高すぎる）
            "height_cm": torch.tensor([50.0], dtype=torch.float),  # 範囲外（低すぎる）
            "shoulder_to_wrist_cm": torch.tensor([100.0], dtype=torch.float),  # 範囲外（高すぎる）
            "elbow_to_wrist_cm": torch.tensor([5.0], dtype=torch.float),  # 範囲外（低すぎる）
        }

        output = demographics_embedding(out_of_range_data)

        # 出力が有限値であることを確認
        assert torch.isfinite(output).all(), "Output contains non-finite values after clipping"
        assert output.shape == (1, 16), f"Expected shape (1, 16), got {output.shape}"


class TestCMISqueezeformerWithDemographics:
    """Demographics統合CMISqueezeformerのテスト."""

    @pytest.fixture
    def config(self):
        """テスト用設定."""
        config = Config()
        config.demographics.enabled = True
        return config

    @pytest.fixture
    def model(self, config):
        """テスト用モデル."""
        return CMISqueezeformer(
            input_dim=config.model.input_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            num_classes=config.model.num_classes,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            demographics_config={
                "enabled": config.demographics.enabled,
                "categorical_features": config.demographics.categorical_features,
                "numerical_features": config.demographics.numerical_features,
                "categorical_embedding_dims": config.demographics.categorical_embedding_dims,
                "embedding_dim": config.demographics.embedding_dim,
            },
            target_gestures=config.target_gestures,
            non_target_gestures=config.non_target_gestures,
        )

    @pytest.fixture
    def sample_batch(self):
        """サンプルバッチデータ."""
        batch_size = 2
        seq_len = 50
        input_dim = 16

        return {
            "imu": torch.randn(batch_size, input_dim, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "demographics": {
                "adult_child": torch.tensor([1, 0], dtype=torch.long),
                "sex": torch.tensor([0, 1], dtype=torch.long),
                "handedness": torch.tensor([1, 1], dtype=torch.long),
                "age": torch.tensor([25.0, 15.0], dtype=torch.float),
                "height_cm": torch.tensor([170.0, 160.0], dtype=torch.float),
                "shoulder_to_wrist_cm": torch.tensor([55.0, 45.0], dtype=torch.float),
                "elbow_to_wrist_cm": torch.tensor([25.0, 22.0], dtype=torch.float),
            },
        }

    def test_model_forward_with_demographics(self, model, sample_batch):
        """Demographics有効時のモデル順伝播をテスト."""
        model.eval()

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                sample_batch["imu"], sample_batch["attention_mask"], demographics=sample_batch["demographics"]
            )

        # 出力形状確認
        batch_size = sample_batch["imu"].shape[0]
        num_classes = 18

        assert multiclass_logits.shape == (batch_size, num_classes), (
            f"Expected multiclass shape ({batch_size}, {num_classes}), got {multiclass_logits.shape}"
        )
        assert binary_logits.shape == (batch_size, 1), (
            f"Expected binary shape ({batch_size}, 1), got {binary_logits.shape}"
        )

        # 出力値の確認
        assert torch.isfinite(multiclass_logits).all(), "Multiclass logits contain non-finite values"
        assert torch.isfinite(binary_logits).all(), "Binary logits contain non-finite values"

    def test_model_forward_without_demographics(self, model, sample_batch):
        """Demographics無効時のモデル順伝播をテスト."""
        model.eval()

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                sample_batch["imu"], sample_batch["attention_mask"], demographics=None
            )

        # 出力形状確認
        batch_size = sample_batch["imu"].shape[0]
        num_classes = 18

        assert multiclass_logits.shape == (batch_size, num_classes), (
            f"Expected multiclass shape ({batch_size}, {num_classes}), got {multiclass_logits.shape}"
        )
        assert binary_logits.shape == (batch_size, 1), (
            f"Expected binary shape ({batch_size}, 1), got {binary_logits.shape}"
        )

    def test_model_training_step_with_demographics(self, model, sample_batch):
        """Demographics統合時のtraining_stepをテスト."""
        # ターゲットデータを追加
        sample_batch["multiclass_label"] = torch.randint(0, 18, (2,), dtype=torch.long)
        sample_batch["binary_label"] = torch.randint(0, 2, (2,), dtype=torch.float)

        model.train()
        loss = model.training_step(sample_batch, batch_idx=0)

        # 損失値の確認
        assert torch.isfinite(loss), f"Training loss is not finite: {loss}"
        assert loss > 0, f"Training loss should be positive: {loss}"

    def test_model_validation_step_with_demographics(self, model, sample_batch):
        """Demographics統合時のvalidation_stepをテスト."""
        # ターゲットデータを追加
        sample_batch["multiclass_label"] = torch.randint(0, 18, (2,), dtype=torch.long)
        sample_batch["binary_label"] = torch.randint(0, 2, (2,), dtype=torch.float)
        sample_batch["sequence_id"] = ["test_seq_1", "test_seq_2"]
        sample_batch["gesture"] = ["Text on phone", "Wave hello"]

        model.eval()
        with torch.no_grad():
            output = model.validation_step(sample_batch, batch_idx=0)

        # 出力の確認
        assert isinstance(output, dict), "Validation step should return a dict"
        assert "val_loss" in output, "Output should contain val_loss"

        # 損失値の確認
        val_loss = output["val_loss"]
        assert torch.isfinite(val_loss), f"Validation loss is not finite: {val_loss}"
        assert val_loss > 0, f"Validation loss should be positive: {val_loss}"


class TestConfigDemographics:
    """Demographics設定のテスト."""

    def test_demographics_config_default(self):
        """デフォルトDemographics設定をテスト."""
        config = DemographicsConfig()

        assert config.enabled is True, "Demographics should be enabled by default"
        assert config.embedding_dim == 16, "Default embedding dim should be 16"
        assert "adult_child" in config.categorical_features, "adult_child should be in categorical features"
        assert "age" in config.numerical_features, "age should be in numerical features"

    def test_main_config_demographics_integration(self):
        """メイン設定でのDemographics統合をテスト."""
        config = Config()

        assert hasattr(config, "demographics"), "Config should have demographics attribute"
        assert config.demographics.enabled is True, "Demographics should be enabled by default"
        assert hasattr(config.data, "demographics_train_path"), "Config should have demographics data paths"
        assert hasattr(config.data, "demographics_test_path"), "Config should have demographics test data path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
