"""exp025学習可能重み付け機能のテストモジュール."""

import pytest
import torch
from codes.exp.exp025.config import LossConfig
from codes.exp.exp025.model import CMISqueezeformer
from torch import nn


class TestExp025LearnableWeights:
    """exp025の学習可能重み付け機能のテストクラス."""

    @pytest.fixture
    def base_config(self):
        """基本的なLossConfig設定."""
        return {
            "type": "cmi",
            "alpha": 0.5,
            "nine_class_loss_weight": 0.2,
            "kl_weight": 0.1,
            "auto_weighting": "none",
        }

    @pytest.fixture
    def uncertainty_config(self):
        """不確かさベース自動重み付け設定."""
        return {
            "type": "cmi",
            "alpha": 0.5,
            "nine_class_loss_weight": 0.2,
            "kl_weight": 0.1,
            "auto_weighting": "uncertainty",
            "uncertainty_init_value": 0.0,
            "auto_weight_clamp": (-10.0, 10.0),
        }

    @pytest.fixture
    def direct_config(self):
        """直接的な学習可能重み設定."""
        return {
            "type": "cmi",
            "alpha": 0.5,
            "nine_class_loss_weight": 0.2,
            "kl_weight": 0.1,
            "auto_weighting": "direct",
            "auto_weight_clamp": (-10.0, 10.0),
        }

    @pytest.fixture
    def sample_batch(self):
        """テスト用のサンプルバッチデータ."""
        batch_size = 4
        seq_len = 50
        return {
            "imu": torch.randn(batch_size, 7, seq_len),
            "multiclass_label": torch.randint(0, 18, (batch_size,)),
            "binary_label": torch.randint(0, 2, (batch_size,)).float(),
            "nine_class_label": torch.randint(0, 9, (batch_size,)),
            "sequence_id": [f"seq_{i}" for i in range(batch_size)],
            "gesture": [f"gesture_{i}" for i in range(batch_size)],
        }

    def test_uncertainty_weighting_initialization(self, uncertainty_config):
        """不確かさベース重み付けの初期化テスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # 学習可能パラメータが正しく初期化されているかチェック
        assert hasattr(model, "loss_s_params")
        assert "multiclass" in model.loss_s_params
        assert "binary" in model.loss_s_params
        assert "nine_class" in model.loss_s_params
        assert "kl" in model.loss_s_params

        # 初期値のチェック
        for param in model.loss_s_params.values():
            assert param.data.item() == 0.0
            assert param.requires_grad is True

    def test_direct_weighting_initialization(self, direct_config):
        """直接的な重み付けの初期化テスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=direct_config,
        )

        # 学習可能パラメータが正しく初期化されているかチェック
        assert hasattr(model, "alpha_raw")
        assert hasattr(model, "w9_raw")
        assert hasattr(model, "wkl_raw")

        # 値が妥当な範囲にあるかチェック
        alpha = torch.sigmoid(model.alpha_raw)
        assert 0.0 < alpha.item() < 1.0
        assert model.w9_raw.requires_grad is True
        assert model.wkl_raw.requires_grad is True

    def test_none_weighting_backward_compatibility(self, base_config):
        """auto_weighting="none"の下位互換性テスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=base_config,
        )

        # 学習可能パラメータが存在しないことを確認
        assert not hasattr(model, "loss_s_params")
        assert not hasattr(model, "alpha_raw")

    def test_uncertainty_weighted_loss_computation(self, uncertainty_config, sample_batch):
        """不確かさ重み付き損失計算のテスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = model(sample_batch["imu"])

        # 個別損失の計算
        multiclass_loss = nn.CrossEntropyLoss()(multiclass_logits, sample_batch["multiclass_label"])
        binary_loss = nn.BCEWithLogitsLoss()(binary_logits.squeeze(-1), sample_batch["binary_label"].float())
        nine_class_loss = nn.CrossEntropyLoss()(nine_class_logits, sample_batch["nine_class_label"])
        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        # 不確かさ重み付き損失の計算
        total_loss = model._compute_uncertainty_weighted_loss(multiclass_loss, binary_loss, nine_class_loss, kl_loss)

        # 損失が正の値であることを確認
        assert total_loss.item() > 0.0
        assert total_loss.requires_grad is True

    def test_direct_weighted_loss_computation(self, direct_config, sample_batch):
        """直接的重み付き損失計算のテスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=direct_config,
        )

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = model(sample_batch["imu"])

        # 個別損失の計算
        multiclass_loss = nn.CrossEntropyLoss()(multiclass_logits, sample_batch["multiclass_label"])
        binary_loss = nn.BCEWithLogitsLoss()(binary_logits.squeeze(-1), sample_batch["binary_label"].float())
        nine_class_loss = nn.CrossEntropyLoss()(nine_class_logits, sample_batch["nine_class_label"])
        kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)

        # 直接的重み付き損失の計算
        total_loss = model._compute_direct_weighted_loss(multiclass_loss, binary_loss, nine_class_loss, kl_loss)

        # 損失が正の値であることを確認
        assert total_loss.item() > 0.0
        assert total_loss.requires_grad is True

    def test_training_step_with_uncertainty_weighting(self, uncertainty_config, sample_batch):
        """不確かさ重み付きでのtraining_stepテスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # training_stepの実行
        loss = model.training_step(sample_batch, 0)

        # 損失が正の値であることを確認
        assert loss.item() > 0.0
        assert loss.requires_grad is True

    def test_training_step_with_direct_weighting(self, direct_config, sample_batch):
        """直接的重み付きでのtraining_stepテスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=direct_config,
        )

        # training_stepの実行
        loss = model.training_step(sample_batch, 0)

        # 損失が正の値であることを確認
        assert loss.item() > 0.0
        assert loss.requires_grad is True

    def test_gradient_flow_for_learnable_parameters(self, uncertainty_config, sample_batch):
        """学習可能パラメータの勾配計算テスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # 損失計算
        loss = model.training_step(sample_batch, 0)

        # 勾配計算
        loss.backward()

        # 学習可能重みパラメータに勾配が計算されているかチェック
        for name, param in model.loss_s_params.items():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_parameter_update_after_optimizer_step(self, uncertainty_config, sample_batch):
        """オプティマイザステップ後のパラメータ更新テスト."""
        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # 初期パラメータ値を保存
        initial_params = {name: param.data.clone() for name, param in model.loss_s_params.items()}

        # オプティマイザの設定
        optimizer = model.configure_optimizers()

        # 損失計算と勾配計算
        loss = model.training_step(sample_batch, 0)
        loss.backward()

        # オプティマイザステップ
        optimizer.step()

        # パラメータが更新されているかチェック
        for name, param in model.loss_s_params.items():
            if param.grad is not None and param.grad.abs().sum() > 0:
                assert not torch.equal(param.data, initial_params[name])

    def test_config_validation(self):
        """LossConfig設定値の検証テスト."""
        # 有効な設定
        valid_config = LossConfig(auto_weighting="uncertainty")
        assert valid_config.auto_weighting == "uncertainty"

        # 無効な設定（例外が発生することを確認）
        with pytest.raises(ValueError):
            LossConfig(auto_weighting="invalid_type")

    def test_clamping_functionality(self, uncertainty_config):
        """パラメータクランプ機能のテスト."""
        # 極端なクランプ範囲を設定
        uncertainty_config["auto_weight_clamp"] = (-1.0, 1.0)

        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=uncertainty_config,
        )

        # パラメータを極端な値に設定
        with torch.no_grad():
            model.loss_s_params["multiclass"].data.fill_(-20.0)

        # 損失計算時にクランプが効いているかチェック
        dummy_losses = [torch.tensor(1.0) for _ in range(4)]
        loss = model._compute_uncertainty_weighted_loss(*dummy_losses)

        # 極端な値でも有限の損失値が返されることを確認
        assert torch.isfinite(loss)

    def test_nine_class_disabled_configuration(self):
        """9クラスヘッド無効時の設定テスト."""
        config = {
            "type": "cmi",
            "nine_class_head_enabled": False,
            "auto_weighting": "uncertainty",
        }

        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=config,
        )

        # 9クラス用パラメータが存在しないことを確認
        assert "nine_class" not in model.loss_s_params

    def test_kl_weight_zero_configuration(self):
        """KL重み0設定時のテスト."""
        config = {
            "type": "cmi",
            "kl_weight": 0.0,
            "auto_weighting": "uncertainty",
        }

        model = CMISqueezeformer(
            input_dim=7,
            d_model=64,
            n_layers=2,
            num_classes=18,
            loss_config=config,
        )

        # KL用パラメータが存在しないことを確認
        assert "kl" not in model.loss_s_params
