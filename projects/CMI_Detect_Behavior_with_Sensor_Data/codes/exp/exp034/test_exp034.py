"""exp034のテストコード - AFNOMixer1DとIMUOnlyLSTMの動作確認."""

import pytest
import torch
from model import AFNOMixer1D, IMUOnlyLSTM


class TestAFNOMixer1D:
    """AFNOMixer1Dのテスト."""

    def test_forward_basic(self):
        """基本的な前向き計算のテスト."""
        batch_size, seq_len, hidden_dim = 4, 100, 256
        mixer = AFNOMixer1D(hidden_dim=hidden_dim)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = mixer(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_different_sizes(self):
        """異なるサイズでの前向き計算のテスト."""
        hidden_dim = 128
        mixer = AFNOMixer1D(hidden_dim=hidden_dim)

        # 異なるバッチサイズとシーケンス長でテスト
        test_cases = [
            (1, 50, hidden_dim),
            (2, 200, hidden_dim),
            (8, 300, hidden_dim),
        ]

        for batch_size, seq_len, hidden_dim in test_cases:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            output = mixer(x)

            assert output.shape == (batch_size, seq_len, hidden_dim)
            assert not torch.isnan(output).any()

    def test_residual_connection(self):
        """残差接続の動作確認."""
        batch_size, seq_len, hidden_dim = 2, 50, 64
        mixer = AFNOMixer1D(hidden_dim=hidden_dim, ff_mult=1, dropout=0.0)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = mixer(x)

        # ドロップアウトが0なので、完全にゼロになることはない
        assert not torch.allclose(output, torch.zeros_like(output))


class TestIMUOnlyLSTM:
    """IMUOnlyLSTMのテスト."""

    def test_forward_without_demographics(self):
        """Demographics無しでの前向き計算のテスト."""
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=0)

        batch_size, seq_len, imu_dim = 4, 100, 20
        x = torch.randn(batch_size, seq_len, imu_dim)

        multiclass_logits, binary_logits, nine_class_logits = model(x)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(nine_class_logits).any()

    def test_forward_with_demographics(self):
        """Demographics有りでの前向き計算のテスト."""
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=16)

        batch_size, seq_len, imu_dim = 4, 100, 20
        x = torch.randn(batch_size, seq_len, imu_dim)
        demographics_embedding = torch.randn(batch_size, 16)

        multiclass_logits, binary_logits, nine_class_logits = model(x, demographics_embedding)

        assert multiclass_logits.shape == (batch_size, 18)
        assert binary_logits.shape == (batch_size, 1)
        assert nine_class_logits.shape == (batch_size, 9)

    def test_afno_components_exist(self):
        """AFNOコンポーネントが存在することを確認."""
        model = IMUOnlyLSTM(imu_dim=20, n_classes=18)

        # AFNOコンポーネントが存在することを確認
        assert hasattr(model, "token_proj")
        assert hasattr(model, "afno1")
        assert hasattr(model, "afno2")
        assert hasattr(model, "seq_pool")

        # BiGRUとAttentionが削除されていることを確認
        assert not hasattr(model, "bigru")
        assert not hasattr(model, "gru_dropout")
        assert not hasattr(model, "attention")

    def test_different_input_dimensions(self):
        """異なる入力次元でのテスト."""
        test_cases = [
            (16, 18),  # 入力次元16、クラス数18
            (20, 18),  # 入力次元20、クラス数18
            (32, 10),  # 入力次元32、クラス数10
        ]

        for imu_dim, n_classes in test_cases:
            model = IMUOnlyLSTM(imu_dim=imu_dim, n_classes=n_classes)

            batch_size, seq_len = 2, 50
            x = torch.randn(batch_size, seq_len, imu_dim)

            multiclass_logits, binary_logits, nine_class_logits = model(x)

            assert multiclass_logits.shape == (batch_size, n_classes)
            assert binary_logits.shape == (batch_size, 1)
            assert nine_class_logits.shape == (batch_size, 9)


def test_integration():
    """統合テスト - 全体的な動作確認."""
    # モデルの作成
    model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=16)

    # 入力データの作成
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, 20)
    demographics_embedding = torch.randn(batch_size, 16)

    # 前向き計算
    model.eval()
    with torch.no_grad():
        multiclass_logits, binary_logits, nine_class_logits = model(x, demographics_embedding)

    # 出力の確認
    assert multiclass_logits.shape == (batch_size, 18)
    assert binary_logits.shape == (batch_size, 1)
    assert nine_class_logits.shape == (batch_size, 9)

    # NaNやInfがないことを確認
    assert not torch.isnan(multiclass_logits).any()
    assert not torch.isnan(binary_logits).any()
    assert not torch.isnan(nine_class_logits).any()
    assert not torch.isinf(multiclass_logits).any()
    assert not torch.isinf(binary_logits).any()
    assert not torch.isinf(nine_class_logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
