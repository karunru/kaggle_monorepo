"""
test_exp014_model.py

exp014のCMISqueezeformerHybridモデルの単体テスト
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "codes" / "exp" / "exp014"))

from config import Exp014Config
from model import CMISqueezeformerHybrid, HybridFeatureFusion, MiniRocketBranch


@pytest.fixture
def sample_config():
    """テスト用の設定を作成"""
    config = Exp014Config()

    # テスト用のパラメータ調整
    config.model.input_dim = 15  # IMU特徴量数
    config.model.minirocket_input_dim = 100  # MiniRocket特徴量数（小さく設定）
    config.model.d_model = 64  # モデル次元数（小さく設定）
    config.model.n_layers = 2  # レイヤー数（小さく設定）
    config.model.n_heads = 4  # ヘッド数
    config.model.d_ff = 128  # FFN次元数
    config.model.num_classes = 18  # ジェスチャー数
    config.model.fusion_dim = 128  # 融合後の次元数
    config.model.fusion_method = "concatenation"  # テスト用
    config.model.dropout = 0.1

    # Demographics無効化（シンプルにするため）
    config.demographics.enabled = False

    # その他の設定
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-4
    config.loss.type = "cmi"
    config.loss.alpha = 0.5

    return config


@pytest.fixture
def sample_batch_data():
    """テスト用のバッチデータを作成"""
    batch_size = 4
    seq_len = 100
    imu_dim = 15
    minirocket_dim = 100

    # テンソルデータを作成
    imu = torch.randn(batch_size, imu_dim, seq_len)
    minirocket_features = torch.randn(batch_size, minirocket_dim)
    attention_mask = torch.ones(batch_size, seq_len)  # 全て有効

    # ラベル
    multiclass_labels = torch.randint(0, 18, (batch_size,))
    binary_labels = torch.randint(0, 2, (batch_size,)).float()

    return {
        "imu": imu,
        "minirocket_features": minirocket_features,
        "attention_mask": attention_mask,
        "multiclass_labels": multiclass_labels,
        "binary_labels": binary_labels,
    }


@pytest.fixture
def sample_demographics_batch():
    """テスト用のDemographicsバッチを作成"""
    batch_size = 4

    return {
        "age": torch.randn(batch_size, 1),  # 正規化済み年齢
        "sex_Male": torch.randint(0, 2, (batch_size,)).float(),
        "height": torch.randn(batch_size, 1),  # 正規化済み身長
        "shoulder_to_wrist": torch.randn(batch_size, 1),
        "elbow_to_wrist": torch.randn(batch_size, 1),
    }


class TestMiniRocketBranch:
    """MiniRocketBranchクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        branch = MiniRocketBranch(
            input_dim=100, output_dim=64, hidden_dims=[256, 128, 64], dropout=0.1, activation="silu"
        )

        assert branch.network is not None
        assert isinstance(branch.activation, torch.nn.SiLU)

    def test_forward_pass(self):
        """前向き計算のテスト"""
        batch_size = 4
        input_dim = 100
        output_dim = 64

        branch = MiniRocketBranch(input_dim=input_dim, output_dim=output_dim)
        x = torch.randn(batch_size, input_dim)

        output = branch(x)

        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_activations(self):
        """異なる活性化関数のテスト"""
        input_dim = 50
        output_dim = 32
        x = torch.randn(2, input_dim)

        for activation in ["silu", "relu", "gelu"]:
            branch = MiniRocketBranch(input_dim=input_dim, output_dim=output_dim, activation=activation)
            output = branch(x)
            assert output.shape == (2, output_dim)

    def test_default_hidden_dims(self):
        """デフォルト隠れ層次元のテスト"""
        branch = MiniRocketBranch(input_dim=200, output_dim=64)
        x = torch.randn(3, 200)
        output = branch(x)
        assert output.shape == (3, 64)


class TestHybridFeatureFusion:
    """HybridFeatureFusionクラスのテスト"""

    def test_concatenation_fusion(self):
        """結合融合のテスト"""
        imu_dim = 64
        minirocket_dim = 100
        output_dim = 128
        batch_size = 4

        fusion = HybridFeatureFusion(
            imu_dim=imu_dim, minirocket_dim=minirocket_dim, output_dim=output_dim, fusion_method="concatenation"
        )

        imu_features = torch.randn(batch_size, imu_dim)
        minirocket_features = torch.randn(batch_size, minirocket_dim)

        output = fusion(imu_features, minirocket_features)

        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()

    def test_addition_fusion(self):
        """加算融合のテスト"""
        imu_dim = 64
        minirocket_dim = 100
        output_dim = 128
        batch_size = 4

        fusion = HybridFeatureFusion(
            imu_dim=imu_dim, minirocket_dim=minirocket_dim, output_dim=output_dim, fusion_method="addition"
        )

        imu_features = torch.randn(batch_size, imu_dim)
        minirocket_features = torch.randn(batch_size, minirocket_dim)

        output = fusion(imu_features, minirocket_features)

        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()

    def test_attention_fusion(self):
        """アテンション融合のテスト"""
        imu_dim = 64
        minirocket_dim = 64  # 同じ次元にする
        output_dim = 64
        batch_size = 4

        fusion = HybridFeatureFusion(
            imu_dim=imu_dim, minirocket_dim=minirocket_dim, output_dim=output_dim, fusion_method="attention"
        )

        imu_features = torch.randn(batch_size, imu_dim)
        minirocket_features = torch.randn(batch_size, minirocket_dim)

        output = fusion(imu_features, minirocket_features)

        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()

    def test_invalid_fusion_method(self):
        """無効な融合方法のテスト"""
        with pytest.raises(ValueError, match="Unknown fusion method"):
            HybridFeatureFusion(imu_dim=64, minirocket_dim=64, output_dim=64, fusion_method="invalid_method")


class TestCMISqueezeformerHybrid:
    """CMISqueezeformerHybridクラスのテスト"""

    def test_model_initialization(self, sample_config):
        """モデル初期化のテスト"""
        model = CMISqueezeformerHybrid(
            config=sample_config,
            target_gestures=["gesture_A", "gesture_B"],
            non_target_gestures=["gesture_C"],
        )

        # 基本属性の確認
        assert model.config == sample_config
        assert model.model_config == sample_config.model

        # 分岐ネットワークの確認
        assert model.imu_input_projection is not None
        assert model.minirocket_branch is not None
        assert model.feature_fusion is not None

        # 分類ヘッドの確認
        assert model.multiclass_head is not None
        assert model.binary_head is not None

        # 損失関数の確認
        assert model.multiclass_criterion is not None
        assert model.binary_criterion is not None

    def test_forward_pass_without_demographics(self, sample_config, sample_batch_data):
        """Demographics無しでの前向き計算テスト"""
        model = CMISqueezeformerHybrid(config=sample_config)
        model.eval()

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=sample_batch_data["imu"],
                minirocket_features=sample_batch_data["minirocket_features"],
                attention_mask=sample_batch_data["attention_mask"],
            )

        batch_size = sample_batch_data["imu"].shape[0]
        num_classes = sample_config.model.num_classes

        assert multiclass_logits.shape == (batch_size, num_classes)
        assert binary_logits.shape == (batch_size, 1)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()

    def test_forward_pass_with_demographics(self, sample_config, sample_batch_data, sample_demographics_batch):
        """Demographics有りでの前向き計算テスト"""
        # Demographics有効化
        sample_config.demographics.enabled = True

        model = CMISqueezeformerHybrid(config=sample_config)
        model.eval()

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=sample_batch_data["imu"],
                minirocket_features=sample_batch_data["minirocket_features"],
                attention_mask=sample_batch_data["attention_mask"],
                demographics=sample_demographics_batch,
            )

        batch_size = sample_batch_data["imu"].shape[0]
        num_classes = sample_config.model.num_classes

        assert multiclass_logits.shape == (batch_size, num_classes)
        assert binary_logits.shape == (batch_size, 1)

    def test_training_step(self, sample_config, sample_batch_data):
        """訓練ステップのテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)

        batch = {
            "imu": sample_batch_data["imu"],
            "minirocket_features": sample_batch_data["minirocket_features"],
            "multiclass_label": sample_batch_data["multiclass_labels"],
            "binary_label": sample_batch_data["binary_labels"],
            "attention_mask": sample_batch_data["attention_mask"],
        }

        loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # スカラー
        assert loss.item() > 0  # 正の値
        assert not torch.isnan(loss)

    def test_validation_step(self, sample_config, sample_batch_data):
        """検証ステップのテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)

        batch = {
            "imu": sample_batch_data["imu"],
            "minirocket_features": sample_batch_data["minirocket_features"],
            "multiclass_label": sample_batch_data["multiclass_labels"],
            "binary_label": sample_batch_data["binary_labels"],
            "attention_mask": sample_batch_data["attention_mask"],
            "sequence_id": ["seq_001", "seq_002", "seq_003", "seq_004"],
            "gesture": ["gesture_A", "gesture_B", "gesture_A", "gesture_B"],
        }

        output = model.validation_step(batch, batch_idx=0)

        # 必要なキーが含まれていることを確認
        expected_keys = [
            "val_loss",
            "val_multiclass_loss",
            "val_binary_loss",
            "multiclass_probs",
            "binary_probs",
            "multiclass_labels",
            "binary_labels",
            "sequence_ids",
            "gestures",
        ]
        for key in expected_keys:
            assert key in output

        # テンソルの形状確認
        batch_size = len(batch["sequence_id"])
        num_classes = sample_config.model.num_classes

        assert output["multiclass_probs"].shape == (batch_size, num_classes)
        assert output["binary_probs"].shape == (batch_size,)

    def test_different_fusion_methods(self, sample_config, sample_batch_data):
        """異なる融合方法のテスト"""
        fusion_methods = ["concatenation", "addition", "attention"]

        for fusion_method in fusion_methods:
            sample_config.model.fusion_method = fusion_method

            model = CMISqueezeformerHybrid(config=sample_config)
            model.eval()

            with torch.no_grad():
                multiclass_logits, binary_logits = model(
                    imu=sample_batch_data["imu"],
                    minirocket_features=sample_batch_data["minirocket_features"],
                    attention_mask=sample_batch_data["attention_mask"],
                )

            batch_size = sample_batch_data["imu"].shape[0]
            num_classes = sample_config.model.num_classes

            assert multiclass_logits.shape == (batch_size, num_classes)
            assert binary_logits.shape == (batch_size, 1)
            assert not torch.isnan(multiclass_logits).any()

    def test_attention_mask_handling(self, sample_config, sample_batch_data):
        """アテンションマスク処理のテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)
        model.eval()

        # 部分的にマスクされたattention_maskを作成
        attention_mask = sample_batch_data["attention_mask"].clone()
        attention_mask[:, 50:] = 0  # 後半をマスク

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=sample_batch_data["imu"],
                minirocket_features=sample_batch_data["minirocket_features"],
                attention_mask=attention_mask,
            )

        batch_size = sample_batch_data["imu"].shape[0]
        num_classes = sample_config.model.num_classes

        assert multiclass_logits.shape == (batch_size, num_classes)
        assert binary_logits.shape == (batch_size, 1)
        assert not torch.isnan(multiclass_logits).any()

    def test_optimizer_configuration(self, sample_config):
        """オプティマイザ設定のテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)

        # 通常のオプティマイザ
        sample_config.schedule_free.enabled = False
        sample_config.training.scheduler_type = "cosine"

        optimizer_config = model.configure_optimizers()

        assert isinstance(optimizer_config, list)
        assert len(optimizer_config) == 2  # [optimizer], [scheduler]

    def test_gradient_flow(self, sample_config, sample_batch_data):
        """勾配フローのテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)
        model.train()

        batch = {
            "imu": sample_batch_data["imu"],
            "minirocket_features": sample_batch_data["minirocket_features"],
            "multiclass_label": sample_batch_data["multiclass_labels"],
            "binary_label": sample_batch_data["binary_labels"],
            "attention_mask": sample_batch_data["attention_mask"],
        }

        # 前向き計算と損失計算
        loss = model.training_step(batch, batch_idx=0)

        # 逆伝播
        loss.backward()

        # 勾配が計算されていることを確認
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                grad_norms.append(grad_norm)

        # ほとんどのパラメータに勾配が計算されていることを確認
        assert len(grad_norms) > 0
        assert sum(g > 0 for g in grad_norms) > len(grad_norms) * 0.8  # 80%以上

    def test_model_output_ranges(self, sample_config, sample_batch_data):
        """モデル出力の範囲テスト"""
        model = CMISqueezeformerHybrid(config=sample_config)
        model.eval()

        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=sample_batch_data["imu"],
                minirocket_features=sample_batch_data["minirocket_features"],
                attention_mask=sample_batch_data["attention_mask"],
            )

            # ソフトマックス確率が正しい範囲にあることを確認
            multiclass_probs = F.softmax(multiclass_logits, dim=-1)
            assert torch.all(multiclass_probs >= 0)
            assert torch.all(multiclass_probs <= 1)
            assert torch.allclose(multiclass_probs.sum(dim=-1), torch.ones(multiclass_probs.shape[0]))

            # シグモイド確率が正しい範囲にあることを確認
            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
            assert torch.all(binary_probs >= 0)
            assert torch.all(binary_probs <= 1)

    def test_model_device_compatibility(self, sample_config, sample_batch_data):
        """デバイス互換性のテスト"""
        model = CMISqueezeformerHybrid(config=sample_config)

        # CPUでのテスト
        model.eval()
        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=sample_batch_data["imu"],
                minirocket_features=sample_batch_data["minirocket_features"],
                attention_mask=sample_batch_data["attention_mask"],
            )

        assert multiclass_logits.device == torch.device("cpu")
        assert binary_logits.device == torch.device("cpu")

        # GPUが利用可能な場合のテスト
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch_data.items()}

            with torch.no_grad():
                multiclass_logits, binary_logits = model(
                    imu=batch_gpu["imu"],
                    minirocket_features=batch_gpu["minirocket_features"],
                    attention_mask=batch_gpu["attention_mask"],
                )

            assert multiclass_logits.device == device
            assert binary_logits.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
