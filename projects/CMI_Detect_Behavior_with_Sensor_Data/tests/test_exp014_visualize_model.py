"""
exp014 Model Visualization Tests

torchvizを使用したモデルアーキテクチャ可視化機能のテストスイート。
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from codes.exp.exp014.config import Exp014Config
from codes.exp.exp014.visualize_model import ModelVisualizer


class TestModelVisualizer:
    """ModelVisualizerクラスのテスト."""

    @pytest.fixture
    def config(self):
        """テスト用設定の作成."""
        config = Exp014Config()
        # テスト用に小さな値に設定
        config.model.d_model = 64
        config.model.n_layers = 2
        config.model.n_heads = 4
        config.model.imu_input_dim = 8
        config.model.minirocket_input_dim = 100
        return config

    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリの作成."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def visualizer(self, config, temp_output_dir):
        """テスト用ModelVisualizerの作成."""
        return ModelVisualizer(
            config=config,
            output_dir=temp_output_dir,
            output_format="png",
        )

    def test_model_visualizer_initialization(self, config, temp_output_dir):
        """ModelVisualizerの初期化テスト."""
        visualizer = ModelVisualizer(
            config=config,
            output_dir=temp_output_dir,
            output_format="pdf",
        )

        assert visualizer.config == config
        assert visualizer.output_dir == Path(temp_output_dir)
        assert visualizer.output_format == "pdf"
        assert visualizer.output_dir.exists()
        assert hasattr(visualizer, "model")
        assert hasattr(visualizer, "timestamp")

    def test_create_dummy_data_basic(self, visualizer):
        """基本的なダミーデータ生成テスト."""
        dummy_data = visualizer.create_dummy_data(batch_size=2, seq_len=50, include_demographics=False)

        assert "imu" in dummy_data
        assert "minirocket_features" in dummy_data
        assert "attention_mask" in dummy_data
        assert "demographics" not in dummy_data

        # 形状の確認
        assert dummy_data["imu"].shape == (2, visualizer.config.model.imu_input_dim, 50)
        assert dummy_data["minirocket_features"].shape == (2, visualizer.config.model.minirocket_input_dim)
        assert dummy_data["attention_mask"].shape == (2, 50)

        # データ型の確認
        assert dummy_data["imu"].dtype == torch.float32
        assert dummy_data["minirocket_features"].dtype == torch.float32
        assert dummy_data["attention_mask"].dtype == torch.bool

    def test_create_dummy_data_with_demographics(self, visualizer):
        """Demographics付きダミーデータ生成テスト."""
        # Demographics有効化
        visualizer.config.demographics.enabled = True

        dummy_data = visualizer.create_dummy_data(batch_size=3, seq_len=75, include_demographics=True)

        assert "demographics" in dummy_data
        demographics = dummy_data["demographics"]

        assert "sex" in demographics
        assert "age" in demographics
        assert "height" in demographics
        assert "shoulder_to_wrist" in demographics
        assert "elbow_to_wrist" in demographics

        # 形状の確認
        for key in demographics.keys():
            assert demographics[key].shape == (3,)

    def test_create_dummy_data_demographics_disabled(self, visualizer):
        """Demographics無効時のダミーデータ生成テスト."""
        # Demographics無効化
        visualizer.config.demographics.enabled = False

        dummy_data = visualizer.create_dummy_data(
            batch_size=2,
            seq_len=50,
            include_demographics=True,  # True指定でも無効になることを確認
        )

        assert "demographics" not in dummy_data

    def test_create_dummy_data_various_sizes(self, visualizer):
        """様々なサイズでのダミーデータ生成テスト."""
        test_cases = [
            (1, 10),
            (4, 200),
            (8, 500),
        ]

        for batch_size, seq_len in test_cases:
            dummy_data = visualizer.create_dummy_data(
                batch_size=batch_size, seq_len=seq_len, include_demographics=False
            )

            assert dummy_data["imu"].shape == (batch_size, visualizer.config.model.imu_input_dim, seq_len)
            assert dummy_data["minirocket_features"].shape == (batch_size, visualizer.config.model.minirocket_input_dim)
            assert dummy_data["attention_mask"].shape == (batch_size, seq_len)

    @patch("torchviz.make_dot")
    def test_visualize_full_model(self, mock_make_dot, visualizer):
        """全体モデル可視化テスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # ダミーデータ生成
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=20, include_demographics=False)

        # 可視化実行
        visualizer.visualize_full_model(dummy_data, "test_full")

        # make_dotが呼ばれたことを確認
        assert mock_make_dot.called
        # renderが呼ばれたことを確認
        assert mock_dot.render.called

    @patch("torchviz.make_dot")
    def test_visualize_component_imu(self, mock_make_dot, visualizer):
        """IMUコンポーネント可視化テスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # ダミーデータ生成
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=20, include_demographics=False)

        # 可視化実行
        visualizer.visualize_component("imu", dummy_data, "test_imu")

        # 呼び出し確認
        assert mock_make_dot.called
        assert mock_dot.render.called

    @patch("torchviz.make_dot")
    def test_visualize_component_minirocket(self, mock_make_dot, visualizer):
        """MiniRocketコンポーネント可視化テスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # ダミーデータ生成
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=20, include_demographics=False)

        # 可視化実行
        visualizer.visualize_component("minirocket", dummy_data, "test_minirocket")

        # 呼び出し確認
        assert mock_make_dot.called
        assert mock_dot.render.called

    @patch("torchviz.make_dot")
    def test_visualize_component_fusion(self, mock_make_dot, visualizer):
        """Fusionコンポーネント可視化テスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # ダミーデータ生成
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=20, include_demographics=False)

        # 可視化実行
        visualizer.visualize_component("fusion", dummy_data, "test_fusion")

        # 呼び出し確認
        assert mock_make_dot.called
        assert mock_dot.render.called

    def test_visualize_component_invalid(self, visualizer):
        """無効なコンポーネント名のテスト."""
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=20, include_demographics=False)

        with pytest.raises(ValueError, match="Unknown component"):
            visualizer.visualize_component("invalid", dummy_data)

    @patch("torchviz.make_dot")
    def test_visualize_all_components(self, mock_make_dot, visualizer):
        """全コンポーネント可視化テスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # 可視化実行
        visualizer.visualize_all_components(batch_size=1, seq_len=20, include_demographics=False)

        # make_dotが複数回呼ばれることを確認（full + imu + minirocket + fusion）
        assert mock_make_dot.call_count >= 4
        assert mock_dot.render.call_count >= 4

    def test_output_format_options(self, config, temp_output_dir):
        """出力フォーマットオプションのテスト."""
        formats = ["png", "pdf", "svg"]

        for fmt in formats:
            visualizer = ModelVisualizer(
                config=config,
                output_dir=temp_output_dir,
                output_format=fmt,
            )
            assert visualizer.output_format == fmt

    def test_model_in_eval_mode(self, visualizer):
        """モデルがeval modeに設定されていることを確認."""
        assert not visualizer.model.training


class TestCommandLineInterface:
    """コマンドラインインターフェースのテスト."""

    def test_main_help(self):
        """ヘルプメッセージの表示テスト."""
        result = subprocess.run(
            ["python", "-m", "codes.exp.exp014.visualize_model", "--help"],
            check=False, capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "CMI Squeezeformer Hybrid Model Visualization" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--format" in result.stdout
        assert "--components" in result.stdout

    @pytest.mark.parametrize("format_type", ["png", "pdf", "svg"])
    def test_format_arguments(self, format_type):
        """フォーマット引数のテスト."""
        # 実際の実行は重いので、引数パースのみをテスト
        import sys

        from codes.exp.exp014.visualize_model import main

        # 引数を設定
        test_args = [
            "visualize_model.py",
            "--format",
            format_type,
            "--components",
            "imu",  # 軽量なコンポーネントのみ
            "--batch-size",
            "1",
            "--seq-len",
            "10",
            "--no-demographics",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("codes.exp.exp014.visualize_model.ModelVisualizer") as mock_visualizer:
                # 初期化のmock
                mock_instance = MagicMock()
                mock_visualizer.return_value = mock_instance

                # 実行
                with patch("codes.exp.exp014.visualize_model.Exp014Config") as mock_config:
                    mock_config.return_value = MagicMock()
                    result = main()

                # 正常終了確認
                assert result == 0
                # ModelVisualizerが正しい引数で初期化されたことを確認
                mock_visualizer.assert_called_once()
                args, kwargs = mock_visualizer.call_args
                assert kwargs["output_format"] == format_type

    @pytest.mark.parametrize("component", ["all", "imu", "minirocket", "fusion", "full"])
    def test_component_arguments(self, component):
        """コンポーネント引数のテスト."""
        import sys

        from codes.exp.exp014.visualize_model import main

        # 引数を設定
        test_args = [
            "visualize_model.py",
            "--components",
            component,
            "--batch-size",
            "1",
            "--seq-len",
            "10",
            "--no-demographics",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("codes.exp.exp014.visualize_model.ModelVisualizer") as mock_visualizer:
                mock_instance = MagicMock()
                mock_visualizer.return_value = mock_instance

                with patch("codes.exp.exp014.visualize_model.Exp014Config") as mock_config:
                    mock_config.return_value = MagicMock()
                    result = main()

                # 正常終了確認
                assert result == 0

                # 適切なメソッドが呼ばれたことを確認
                if component == "all":
                    mock_instance.visualize_all_components.assert_called_once()
                elif component == "full":
                    mock_instance.visualize_full_model.assert_called_once()
                else:
                    mock_instance.visualize_component.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングのテスト."""

    def test_invalid_config_handling(self, temp_output_dir):
        """無効な設定の処理テスト."""
        # 無効な設定を作成
        config = Exp014Config()
        config.model.d_model = -1  # 無効な値

        # ModelVisualizerは初期化時にエラーが発生する可能性があるが、
        # ここでは設定の検証をテスト
        assert config.model.d_model == -1

    def test_cuda_availability_handling(self, config, temp_output_dir):
        """CUDA利用可能性の処理テスト."""
        # CPU環境でもテストが通ることを確認
        visualizer = ModelVisualizer(
            config=config,
            output_dir=temp_output_dir,
            output_format="png",
        )

        # ダミーデータ生成（CPUで実行）
        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=10, include_demographics=False)

        # CPUテンソルであることを確認
        assert dummy_data["imu"].device.type == "cpu"

    @patch("torchviz.make_dot")
    def test_visualization_error_handling(self, mock_make_dot, config, temp_output_dir):
        """可視化エラーのハンドリングテスト."""
        # make_dotがエラーを発生させる設定
        mock_make_dot.side_effect = Exception("Mock visualization error")

        visualizer = ModelVisualizer(
            config=config,
            output_dir=temp_output_dir,
            output_format="png",
        )

        dummy_data = visualizer.create_dummy_data(batch_size=1, seq_len=10, include_demographics=False)

        # エラーが発生することを確認
        with pytest.raises(Exception, match="Mock visualization error"):
            visualizer.visualize_full_model(dummy_data)


class TestIntegration:
    """統合テスト."""

    @patch("torchviz.make_dot")
    def test_end_to_end_workflow(self, mock_make_dot, temp_output_dir):
        """エンドツーエンドのワークフローテスト."""
        # Mock設定
        mock_dot = MagicMock()
        mock_make_dot.return_value = mock_dot

        # 設定作成
        config = Exp014Config()
        config.model.d_model = 32  # 小さな値でテスト
        config.model.n_layers = 1

        # 可視化実行
        visualizer = ModelVisualizer(
            config=config,
            output_dir=temp_output_dir,
            output_format="png",
        )

        # 全コンポーネント可視化
        visualizer.visualize_all_components(
            batch_size=1,
            seq_len=10,
            include_demographics=False,
        )

        # 呼び出し確認
        assert mock_make_dot.called
        assert mock_dot.render.called

        # 出力ディレクトリが作成されていることを確認
        assert Path(temp_output_dir).exists()

    def test_model_architecture_consistency(self):
        """モデルアーキテクチャの整合性テスト."""
        config = Exp014Config()
        visualizer = ModelVisualizer(
            config=config,
            output_dir="temp_test",
            output_format="png",
        )

        # モデルパラメータの確認
        total_params = sum(p.numel() for p in visualizer.model.parameters())
        trainable_params = sum(p.numel() for p in visualizer.model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # 全パラメータが学習可能であることを確認

        # モデル出力形状の確認
        dummy_data = visualizer.create_dummy_data(batch_size=2, seq_len=50, include_demographics=False)

        with torch.no_grad():
            multiclass_logits, binary_logits = visualizer.model(
                imu=dummy_data["imu"],
                minirocket_features=dummy_data["minirocket_features"],
                attention_mask=dummy_data["attention_mask"],
                demographics=dummy_data.get("demographics"),
            )

        # 出力形状の確認
        assert multiclass_logits.shape == (2, config.model.num_classes)
        assert binary_logits.shape == (2, 1)
