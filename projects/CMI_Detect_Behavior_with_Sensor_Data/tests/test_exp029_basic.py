"""exp029の基本機能テスト."""

import sys
from pathlib import Path

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp029"))

import pytest
import torch
from model import CMISqueezeformer, DemographicsEmbedding, IMUOnlyLSTM


class TestDemographicsEmbedding:
    """DemographicsEmbedding クラスのテスト."""

    def test_demographics_embedding_basic(self):
        """基本的な埋め込み処理のテスト."""
        embedding = DemographicsEmbedding(
            categorical_features=["adult_child", "sex", "handedness"],
            numerical_features=["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
            categorical_embedding_dims={"adult_child": 2, "sex": 2, "handedness": 2},
            embedding_dim=16,
            dropout=0.1,
        )

        batch_size = 3
        demographics_data = {
            "adult_child": torch.randint(0, 2, (batch_size,)),
            "sex": torch.randint(0, 2, (batch_size,)),
            "handedness": torch.randint(0, 2, (batch_size,)),
            "age": torch.randn(batch_size) * 10 + 25,
            "height_cm": torch.randn(batch_size) * 20 + 170,
            "shoulder_to_wrist_cm": torch.randn(batch_size) * 10 + 55,
            "elbow_to_wrist_cm": torch.randn(batch_size) * 5 + 30,
        }

        result = embedding(demographics_data)

        assert result.shape == (batch_size, 16)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_demographics_embedding_scaling(self):
        """年齢のスケーリング処理のテスト."""
        embedding = DemographicsEmbedding(
            categorical_features=["adult_child"],
            numerical_features=["age"],
            categorical_embedding_dims={"adult_child": 2},
            embedding_dim=8,
            age_min=18.0,
            age_max=65.0,
        )

        # 範囲外の年齢をテスト
        demographics_data = {
            "adult_child": torch.tensor([1]),
            "age": torch.tensor([100.0]),  # 上限を超える年齢
        }

        result = embedding(demographics_data)

        assert result.shape == (1, 8)
        assert not torch.isnan(result).any()


class TestIMUOnlyLSTM:
    """IMUOnlyLSTM クラスのテスト."""

    def test_imu_lstm_forward_without_demographics(self):
        """Demographics無しでの前向き計算テスト."""
        model = IMUOnlyLSTM(
            imu_dim=20,
            n_classes=18,
            demographics_dim=0,
            dropout=0.1,
        )

        batch_size = 2
        seq_len = 150
        imu_data = torch.randn(batch_size, seq_len, 20)

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(nine_class_logits).any()

    def test_imu_lstm_forward_with_demographics(self):
        """Demographics統合での前向き計算テスト."""
        model = IMUOnlyLSTM(
            imu_dim=20,
            n_classes=18,
            demographics_dim=16,
            dropout=0.1,
        )

        batch_size = 2
        seq_len = 150
        imu_data = torch.randn(batch_size, seq_len, 20)
        demographics_embedding = torch.randn(batch_size, 16)

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics_embedding)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(nine_class_logits).any()


class TestCMISqueezeformer:
    """CMISqueezeformer クラスのテスト."""

    def test_model_forward_basic(self):
        """基本的な前向き計算のテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "focal", "kl_weight": 0.1},
        )

        batch_size = 2
        seq_len = 200
        imu_data = torch.randn(batch_size, seq_len, 20)

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(nine_class_logits).any()

    def test_model_forward_with_demographics(self):
        """Demographics統合での前向き計算のテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={
                "enabled": True,
                "embedding_dim": 16,
                "categorical_features": ["adult_child", "sex", "handedness"],
                "numerical_features": ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
                "categorical_embedding_dims": {"adult_child": 2, "sex": 2, "handedness": 2},
            },
            loss_config={"type": "focal", "kl_weight": 0.1},
        )

        batch_size = 2
        seq_len = 200
        imu_data = torch.randn(batch_size, seq_len, 20)

        demographics_data = {
            "adult_child": torch.randint(0, 2, (batch_size,)),
            "sex": torch.randint(0, 2, (batch_size,)),
            "handedness": torch.randint(0, 2, (batch_size,)),
            "age": torch.randn(batch_size) * 10 + 25,
            "height_cm": torch.randn(batch_size) * 20 + 170,
            "shoulder_to_wrist_cm": torch.randn(batch_size) * 10 + 55,
            "elbow_to_wrist_cm": torch.randn(batch_size) * 5 + 30,
        }

        multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics=demographics_data)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(nine_class_logits).any()

    def test_kl_loss_computation(self):
        """KL divergence lossの計算テスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "focal", "kl_weight": 0.1, "kl_temperature": 1.0},
        )

        batch_size = 2
        multiclass_logits = torch.randn(batch_size, 18)
        nine_class_logits = torch.randn(batch_size, 9)

        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.dim() == 0  # scalar
        assert kl_loss >= 0  # KL divergence is non-negative
        assert not torch.isnan(kl_loss)
        assert not torch.isinf(kl_loss)

    def test_training_step(self):
        """訓練ステップのテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "focal", "kl_weight": 0.1},
        )

        batch_size = 2
        seq_len = 200
        batch = {
            "imu": torch.randn(batch_size, seq_len, 20),
            "multiclass_label": torch.randint(0, 18, (batch_size,)),
            "binary_label": torch.randint(0, 2, (batch_size,)).float(),
            "nine_class_label": torch.randint(0, 9, (batch_size,)),
        }

        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_parameter_count(self):
        """パラメータ数の確認テスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

        # 期待されるパラメータ数の範囲チェック
        assert 400_000 <= total_params <= 600_000  # exp029のモデルサイズ想定

    def test_input_shape_flexibility(self):
        """入力形状の柔軟性テスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
        )
        model.eval()  # BatchNormエラー回避のため評価モードに設定

        batch_size = 2  # BatchNormエラー回避のためbatch_sizeを増加
        seq_len = 100

        # [batch, seq_len, features] 形式
        imu_data_1 = torch.randn(batch_size, seq_len, 20)
        output_1 = model(imu_data_1)

        # [batch, features, seq_len] 形式
        imu_data_2 = torch.randn(batch_size, 20, seq_len)
        output_2 = model(imu_data_2)

        # 両方とも同じ出力形状であることを確認
        assert output_1[0].shape == output_2[0].shape  # multiclass
        assert output_1[1].shape == output_2[1].shape  # binary
        assert output_1[2].shape == output_2[2].shape  # nine_class


class TestLossFunctions:
    """新しい損失関数タイプのテスト."""

    def test_cmi_loss_type(self):
        """CMI損失関数タイプのテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "cmi"},
        )
        
        # 損失関数が正しく設定されているか確認
        assert hasattr(model, "multiclass_criterion")
        assert hasattr(model, "binary_criterion")
        assert hasattr(model, "nine_class_criterion")

    def test_acls_loss_type(self):
        """ACLS損失関数タイプのテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "acls"},
            acls_config={
                "acls_pos_lambda": 1.0,
                "acls_neg_lambda": 0.1,
                "acls_alpha": 0.1,
                "acls_margin": 10.0,
            },
        )
        
        # ACLS損失関数が正しく設定されているか確認
        assert hasattr(model, "multiclass_criterion")
        assert hasattr(model, "binary_criterion")
        assert hasattr(model, "nine_class_criterion")

    def test_label_smoothing_loss_type(self):
        """Label Smoothing損失関数タイプのテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "label_smoothing"},
            acls_config={"label_smoothing_alpha": 0.1},
        )
        
        # Label Smoothing損失関数が正しく設定されているか確認
        assert hasattr(model, "multiclass_criterion")
        assert hasattr(model, "binary_criterion")
        assert hasattr(model, "nine_class_criterion")

    def test_mbls_loss_type(self):
        """MbLS損失関数タイプのテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "mbls"},
            acls_config={
                "mbls_margin": 10.0,
                "mbls_alpha": 0.1,
                "mbls_schedule": None,
            },
        )
        
        # MbLS損失関数が正しく設定されているか確認
        assert hasattr(model, "multiclass_criterion")
        assert hasattr(model, "binary_criterion")
        assert hasattr(model, "nine_class_criterion")

    def test_learnable_weights_direct(self):
        """学習可能重み（direct）のテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "cmi_focal", "auto_weighting": "direct"},
        )
        
        # 学習可能パラメータが正しく設定されているか確認
        assert hasattr(model, "alpha_raw")
        assert hasattr(model, "w9_raw")
        assert hasattr(model, "wkl_raw")
        assert isinstance(model.alpha_raw, torch.nn.Parameter)
        assert isinstance(model.w9_raw, torch.nn.Parameter)
        assert isinstance(model.wkl_raw, torch.nn.Parameter)

    def test_learnable_weights_uncertainty(self):
        """学習可能重み（uncertainty）のテスト."""
        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config={"enabled": False},
            loss_config={"type": "cmi_focal", "auto_weighting": "uncertainty"},
        )
        
        # uncertainty用パラメータが正しく設定されているか確認
        assert hasattr(model, "loss_s_params")
        assert "multiclass" in model.loss_s_params
        assert "binary" in model.loss_s_params
        assert "nine_class" in model.loss_s_params
        assert "kl" in model.loss_s_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
