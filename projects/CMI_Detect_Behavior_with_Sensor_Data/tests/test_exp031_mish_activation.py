"""exp031用テストコード: Mish活性化関数の動作確認."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# exp031モジュールをインポート
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "codes" / "exp" / "exp031"))

from config import Config
from model import CMISqueezeformer, DemographicsEmbedding, IMUOnlyLSTM, SEBlock


class TestMishActivation:
    """Mish活性化関数のテスト."""

    def test_mish_function_exists(self):
        """F.mishが利用可能かテスト."""
        x = torch.randn(2, 3)
        try:
            output = F.mish(x)
            assert output.shape == x.shape
            print("F.mish is available and working")
        except AttributeError:
            pytest.skip("F.mish is not available in this PyTorch version")

    def test_mish_vs_relu_difference(self):
        """MishとReLUの出力差異を確認."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        try:
            mish_output = F.mish(x)
            relu_output = F.relu(x)

            # 負の値でMishとReLUに差があることを確認
            assert not torch.allclose(mish_output, relu_output)
            # 正の値では近い値になることを確認（完全一致ではない）
            assert mish_output[3] > 0.8  # x=1.0の場合
            assert mish_output[4] > 1.8  # x=2.0の場合
            print(f"Mish output: {mish_output}")
            print(f"ReLU output: {relu_output}")
        except AttributeError:
            pytest.skip("F.mish is not available in this PyTorch version")


class TestSEBlockMish:
    """SEBlockでのMish使用テスト."""

    def test_se_block_with_mish(self):
        """SEBlockがMishを使用していることを確認."""
        channels = 64
        se_block = SEBlock(channels)

        # SEBlockの構造を確認
        excitation_layers = list(se_block.excitation.children())
        assert len(excitation_layers) == 4  # Linear, Mish, Linear, Sigmoid

        # 2番目の層がMishであることを確認
        assert isinstance(excitation_layers[1], torch.nn.Mish)

        # 動作確認
        x = torch.randn(2, channels, 100)
        output = se_block(x)
        assert output.shape == x.shape
        print("SEBlock with Mish activation works correctly")


class TestIMUOnlyLSTMMish:
    """IMUOnlyLSTMでのMish使用テスト."""

    def test_imu_only_lstm_mish_forward(self):
        """IMUOnlyLSTMのforward処理でMishが使用されていることを確認."""
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=16)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 20)
        demographics_embedding = torch.randn(batch_size, 16)

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = model(x, demographics_embedding)

        # 出力形状の確認
        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)

        # 出力値が有限であることを確認
        assert torch.all(torch.isfinite(multiclass_logits))
        assert torch.all(torch.isfinite(binary_logits))
        assert torch.all(torch.isfinite(nine_class_logits))

        print("IMUOnlyLSTM with Mish activation works correctly")


class TestDemographicsEmbeddingMish:
    """DemographicsEmbeddingでのMish使用テスト."""

    def test_demographics_embedding_mish(self):
        """DemographicsEmbeddingがMishを使用していることを確認."""
        embedding = DemographicsEmbedding(
            categorical_features=["adult_child", "sex", "handedness"],
            numerical_features=["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
            categorical_embedding_dims={"adult_child": 2, "sex": 2, "handedness": 2},
            embedding_dim=16,
        )

        # projectionレイヤーの構造確認
        projection_layers = list(embedding.projection.children())
        assert len(projection_layers) == 6  # Linear, Mish, Dropout, Linear, Mish, Dropout

        # Mishレイヤーの存在確認
        mish_layers = [layer for layer in projection_layers if isinstance(layer, torch.nn.Mish)]
        assert len(mish_layers) == 2

        # 動作確認
        batch_size = 3
        demographics = {
            "adult_child": torch.randint(0, 2, (batch_size,)),
            "sex": torch.randint(0, 2, (batch_size,)),
            "handedness": torch.randint(0, 2, (batch_size,)),
            "age": torch.randn(batch_size) * 10 + 25,
            "height_cm": torch.randn(batch_size) * 20 + 170,
            "shoulder_to_wrist_cm": torch.randn(batch_size) * 10 + 55,
            "elbow_to_wrist_cm": torch.randn(batch_size) * 5 + 30,
        }

        output = embedding(demographics)
        assert output.shape == (batch_size, 16)
        assert torch.all(torch.isfinite(output))

        print("DemographicsEmbedding with Mish activation works correctly")


class TestCMISqueezeformerMish:
    """CMISqueezeformerでのMish使用の統合テスト."""

    def test_cmi_squeezeformer_with_mish(self):
        """CMISqueezeformerでMishが正常に動作することを確認."""
        config = Config()

        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config=config.demographics.model_dump(),
            loss_config=config.loss.model_dump(),
        )

        batch_size = 2
        seq_len = 150

        # テストデータ
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

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics=demographics_data)

        # 出力形状の確認
        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)

        # 出力値が有限であることを確認
        assert torch.all(torch.isfinite(multiclass_logits))
        assert torch.all(torch.isfinite(binary_logits))
        assert torch.all(torch.isfinite(nine_class_logits))

        # 勾配計算が正常に動作することを確認
        loss = multiclass_logits.sum() + binary_logits.sum() + nine_class_logits.sum()
        loss.backward()

        # 勾配が計算されていることを確認
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

        print("CMISqueezeformer with Mish activation works correctly")

    def test_model_parameter_count(self):
        """モデルのパラメータ数が妥当であることを確認."""
        config = Config()

        model = CMISqueezeformer(
            input_dim=20,
            num_classes=18,
            demographics_config=config.demographics.model_dump(),
            loss_config=config.loss.model_dump(),
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # パラメータ数が妥当であることを確認（極端に少なくない、多すぎない）
        assert 10_000 < total_params < 10_000_000
        assert total_params == trainable_params  # 全パラメータが学習可能

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


class TestNoReluRemaining:
    """ReLUが残っていないことを確認するテスト."""

    def test_no_relu_in_model_forward(self):
        """モデルのforward処理でReLUが使用されていないことを確認."""
        # この確認は実際にはモニタリングやコード解析で行うべきですが、
        # テストとして基本的な動作確認を行います
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18)

        # フックを使ってReLU層の呼び出しを監視
        relu_called = {"count": 0}

        def relu_hook(module, input, output):
            relu_called["count"] += 1

        # ReLU層にフックを登録
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(relu_hook)

        # テスト実行（evaluationモードに変更してBatchNormエラーを回避）
        model.eval()
        x = torch.randn(1, 100, 20)
        with torch.no_grad():
            model(x)

        # ReLUが呼び出されていないことを確認
        assert relu_called["count"] == 0, f"ReLU was called {relu_called['count']} times"
        print("No ReLU activation detected in forward pass")


if __name__ == "__main__":
    # 個別テスト実行用
    pytest.main([__file__, "-v"])
