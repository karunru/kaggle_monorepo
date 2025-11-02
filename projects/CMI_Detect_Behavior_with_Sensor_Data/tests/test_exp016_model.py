"""exp016 モデル統合テスト."""

import pytest
import torch
import torch.nn.functional as F
from codes.exp.exp016.model import CMISqueezeformer


class TestExp016Model:
    """exp016 CMISqueezeformerモデルのテスト."""

    @pytest.fixture
    def sample_imu_data(self) -> torch.Tensor:
        """サンプルIMUデータを生成."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 100

        # 現実的なIMUデータを生成
        acc = torch.randn(batch_size, 3, seq_len) * 2.0 + torch.tensor([0.0, 0.0, 9.81]).view(1, 3, 1)

        # 正規化された四元数を生成
        quat = torch.randn(batch_size, 4, seq_len)
        quat = quat / torch.norm(quat, dim=1, keepdim=True)

        # [B, 7, T] 形式に結合 (acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z)
        imu_data = torch.cat([acc, quat], dim=1)

        return imu_data

    @pytest.fixture
    def sample_attention_mask(self) -> torch.Tensor:
        """サンプルアテンションマスクを生成."""
        batch_size, seq_len = 2, 100
        # 最後の10フレームをマスクアウト
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, -10:] = False
        return mask

    @pytest.fixture
    def sample_demographics(self) -> torch.Tensor:
        """サンプルdemographicsデータを生成."""
        batch_size = 2
        # [adult_child, sex, handedness, age, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm]
        demographics = torch.tensor(
            [
                [1, 0, 1, 25.0, 165.0, 55.0, 30.0],  # 大人、女性、右利き
                [0, 1, 1, 12.0, 140.0, 45.0, 25.0],  # 子供、男性、右利き
            ],
            dtype=torch.float32,
        )
        return demographics

    def test_model_forward_without_demographics(
        self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor
    ):
        """Demographics無しでのモデル推論テスト."""
        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.eval()

        # 順伝播
        with torch.no_grad():
            multiclass_logits, binary_logits = model(sample_imu_data, sample_attention_mask)

        # 出力形状の確認
        batch_size = sample_imu_data.shape[0]
        assert multiclass_logits.shape == (batch_size, 18), (
            f"Expected multiclass shape ({batch_size}, 18), got {multiclass_logits.shape}"
        )
        assert binary_logits.shape == (batch_size, 1), (
            f"Expected binary shape ({batch_size}, 1), got {binary_logits.shape}"
        )

        # NaN/Infの確認
        assert torch.isfinite(multiclass_logits).all(), "Multiclass logits contain NaN or Inf"
        assert torch.isfinite(binary_logits).all(), "Binary logits contain NaN or Inf"

    def test_model_forward_with_demographics(
        self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor, sample_demographics: torch.Tensor
    ):
        """Demographics有りでのモデル推論テスト."""
        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {
                "enabled": True,
                "embedding_dim": 16,
                "categorical_features": ["adult_child", "sex", "handedness"],
                "numerical_features": ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
                "categorical_embedding_dims": {"adult_child": 2, "sex": 2, "handedness": 2},
            },
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.eval()

        # 順伝播
        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                sample_imu_data, sample_attention_mask, demographics=sample_demographics
            )

        # 出力形状の確認
        batch_size = sample_imu_data.shape[0]
        assert multiclass_logits.shape == (batch_size, 18), (
            f"Expected multiclass shape ({batch_size}, 18), got {multiclass_logits.shape}"
        )
        assert binary_logits.shape == (batch_size, 1), (
            f"Expected binary shape ({batch_size}, 1), got {binary_logits.shape}"
        )

        # NaN/Infの確認
        assert torch.isfinite(multiclass_logits).all(), "Multiclass logits contain NaN or Inf"
        assert torch.isfinite(binary_logits).all(), "Binary logits contain NaN or Inf"

    def test_feature_extraction_dimension(self, sample_imu_data: torch.Tensor):
        """特徴量抽出の次元確認テスト."""
        config = {"input_dim": 7, "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8}}

        model = CMISqueezeformer(**config)

        # 特徴量抽出器の出力確認
        features = model.imu_feature_extractor(sample_imu_data)

        batch_size, seq_len = sample_imu_data.shape[0], sample_imu_data.shape[2]
        expected_shape = (batch_size, 16, seq_len)
        assert features.shape == expected_shape, f"Expected feature shape {expected_shape}, got {features.shape}"

    def test_model_training_step(self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor):
        """訓練ステップのテスト."""
        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.train()

        # サンプルターゲット
        batch_size = sample_imu_data.shape[0]
        multiclass_targets = torch.randint(0, 18, (batch_size,))
        binary_targets = torch.randint(0, 2, (batch_size,)).float()

        batch = {
            "imu": sample_imu_data,
            "attention_mask": sample_attention_mask,
            "multiclass_target": multiclass_targets,
            "binary_target": binary_targets,
            "demographics": None,
        }

        # 訓練ステップ実行
        loss = model.training_step(batch, 0)

        # 損失の確認
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_model_validation_step(self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor):
        """検証ステップのテスト."""
        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.eval()

        # サンプルターゲット
        batch_size = sample_imu_data.shape[0]
        multiclass_targets = torch.randint(0, 18, (batch_size,))
        binary_targets = torch.randint(0, 2, (batch_size,)).float()

        batch = {
            "imu": sample_imu_data,
            "attention_mask": sample_attention_mask,
            "multiclass_target": multiclass_targets,
            "binary_target": binary_targets,
            "demographics": None,
        }

        # 検証ステップ実行
        result = model.validation_step(batch, 0)

        # 結果の確認
        assert "val_loss" in result, "Validation result should contain val_loss"
        assert torch.isfinite(result["val_loss"]), "Validation loss should be finite"

    def test_gradient_flow(self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor):
        """勾配フローのテスト."""
        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.train()

        # サンプルターゲット
        batch_size = sample_imu_data.shape[0]
        multiclass_targets = torch.randint(0, 18, (batch_size,))
        binary_targets = torch.randint(0, 2, (batch_size,)).float()

        # 順伝播
        multiclass_logits, binary_logits = model(sample_imu_data, sample_attention_mask)

        # 損失計算
        multiclass_loss = F.cross_entropy(multiclass_logits, multiclass_targets)
        binary_loss = F.binary_cross_entropy_with_logits(binary_logits.squeeze(), binary_targets)
        total_loss = multiclass_loss + binary_loss

        # 逆伝播
        total_loss.backward()

        # 勾配の確認
        has_gradient = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradient = True
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"

        assert has_gradient, "No gradients found in model parameters"

    def test_different_batch_sizes(self):
        """異なるバッチサイズでの動作確認."""
        config = {
            "input_dim": 7,
            "d_model": 64,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 128,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        model = CMISqueezeformer(**config)
        model.eval()

        # 異なるバッチサイズでテスト
        for batch_size in [1, 4, 8]:
            seq_len = 50
            torch.manual_seed(42)

            # テストデータ生成
            acc = torch.randn(batch_size, 3, seq_len)
            quat = torch.randn(batch_size, 4, seq_len)
            quat = quat / torch.norm(quat, dim=1, keepdim=True)
            imu_data = torch.cat([acc, quat], dim=1)

            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            # 順伝播
            with torch.no_grad():
                multiclass_logits, binary_logits = model(imu_data, attention_mask)

            # 出力形状確認
            assert multiclass_logits.shape == (batch_size, 18)
            assert binary_logits.shape == (batch_size, 1)
            assert torch.isfinite(multiclass_logits).all()
            assert torch.isfinite(binary_logits).all()

    def test_gpu_compatibility(self, sample_imu_data: torch.Tensor, sample_attention_mask: torch.Tensor):
        """GPU互換性テスト."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = {
            "input_dim": 7,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "num_classes": 18,
            "demographics_config": {"enabled": False},
            "feature_extractor_config": {"time_delta": 1.0 / 200.0, "tol": 1e-8},
        }

        device = torch.device("cuda")
        model = CMISqueezeformer(**config).to(device)
        model.eval()

        sample_imu_data_gpu = sample_imu_data.to(device)
        sample_attention_mask_gpu = sample_attention_mask.to(device)

        # 順伝播
        with torch.no_grad():
            multiclass_logits, binary_logits = model(sample_imu_data_gpu, sample_attention_mask_gpu)

        # 出力確認
        assert multiclass_logits.device == device
        assert binary_logits.device == device
        assert torch.isfinite(multiclass_logits).all()
        assert torch.isfinite(binary_logits).all()
