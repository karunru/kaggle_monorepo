"""exp015用テストコード - PyTorch特徴量抽出統合版."""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "codes" / "exp" / "exp015"))

import numpy as np
import polars as pl
import torch
from codes.exp.exp015.config import Config
from codes.exp.exp015.dataset import IMUDataset
from codes.exp.exp015.model import CMISqueezeformer, IMUFeatureExtractor


class TestIMUFeatureExtractor:
    """IMU特徴量抽出器のテスト."""

    def setup_method(self):
        """テスト用のセットアップ."""
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 7  # acc_x/y/z, rot_w/x/y/z
        self.feature_extractor = IMUFeatureExtractor(sampling_rate=100.0, cutoff_freq=10.0, filter_order=15)

    def test_feature_extractor_initialization(self):
        """特徴量抽出器の初期化テスト."""
        extractor = IMUFeatureExtractor()
        assert extractor.sampling_rate == 100.0
        assert extractor.cutoff_freq == 10.0
        assert extractor.filter_size == 15

        # フィルターの重みが正しく初期化されているか確認
        assert extractor.lpf_acc.weight.requires_grad is False
        assert extractor.lpf_ang_vel.weight.requires_grad is False

    def test_feature_extractor_output_shape(self):
        """特徴量抽出器の出力形状テスト."""
        # ランダム入力データ生成
        imu_data = torch.randn(self.batch_size, self.input_dim, self.seq_len)

        # 特徴量抽出
        features = self.feature_extractor(imu_data)

        # 出力形状の確認 (39次元の拡張特徴量)
        expected_output_dim = 39
        assert features.shape == (self.batch_size, expected_output_dim, self.seq_len)

        # NaN値が含まれていないことを確認
        assert not torch.isnan(features).any()

    def test_remove_gravity_functionality(self):
        """重力除去機能のテスト."""
        # テスト用の加速度と四元数データ
        acc = torch.tensor([[[0.0, 0.0, 9.81]]], dtype=torch.float32).transpose(-1, -2)  # [1, 3, 1]
        quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32).transpose(-1, -2)  # [1, 4, 1]

        linear_acc = self.feature_extractor.remove_gravity(acc, quat)

        # 形状確認
        assert linear_acc.shape == (1, 3, 1)

        # 重力が除去されているか確認（完全にゼロにはならないが近い値になる）
        assert linear_acc.abs().max() < 1.0  # 許容誤差

    def test_quaternion_to_angular_velocity(self):
        """四元数から角速度への変換テスト."""
        # 単位四元数のテストデータ
        quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]]], dtype=torch.float32).transpose(
            -1, -2
        )  # [1, 4, 2]

        angular_vel = self.feature_extractor.quaternion_to_angular_velocity(quat)

        # 形状確認
        assert angular_vel.shape == (1, 3, 2)

        # 有限値であることを確認
        assert torch.isfinite(angular_vel).all()

    def test_calculate_angular_distance(self):
        """角距離計算のテスト."""
        # 単位四元数のテストデータ
        quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]]], dtype=torch.float32).transpose(
            -1, -2
        )  # [1, 4, 2]

        angular_distance = self.feature_extractor.calculate_angular_distance(quat)

        # 形状確認
        assert angular_distance.shape == (1, 1, 2)

        # 0以上の値であることを確認
        assert (angular_distance >= 0).all()

        # 有限値であることを確認
        assert torch.isfinite(angular_distance).all()

    def test_gradient_flow(self):
        """勾配の流れるテスト."""
        # 勾配計算が必要な入力テンソル
        imu_data = torch.randn(2, 7, 50, requires_grad=True)

        # 特徴量抽出
        features = self.feature_extractor(imu_data)

        # 損失計算（単純な合計）
        loss = features.sum()

        # 逆伝播
        loss.backward()

        # 勾配が流れていることを確認
        assert imu_data.grad is not None
        assert not torch.isnan(imu_data.grad).any()


class TestCMISqueezeformerIntegration:
    """CMISqueezeformerとIMU特徴量抽出器の統合テスト."""

    def setup_method(self):
        """テスト用のセットアップ."""
        self.config = Config()
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 7
        self.num_classes = 18

        # モデル作成
        self.model = CMISqueezeformer(
            input_dim=self.input_dim,
            feature_extractor_config=self.config.feature_extractor.model_dump(),
            d_model=256,
            n_layers=2,  # テスト用に小さく
            n_heads=4,  # テスト用に小さく
            d_ff=512,  # テスト用に小さく
            num_classes=self.num_classes,
            dropout=0.1,
        )

    def test_model_initialization(self):
        """モデル初期化のテスト."""
        assert self.model.input_dim == 7
        assert self.model.extracted_feature_dim == 39
        assert self.model.num_classes == self.num_classes

        # 特徴量抽出器が正しく初期化されているか確認
        assert isinstance(self.model.imu_feature_extractor, IMUFeatureExtractor)
        assert self.model.input_projection.in_features == 39  # 拡張特徴量の次元

    def test_forward_pass_basic(self):
        """基本的な順伝播テスト."""
        # ランダム入力データ
        imu_data = torch.randn(self.batch_size, self.input_dim, self.seq_len)

        # 順伝播
        multiclass_logits, binary_logits = self.model(imu_data)

        # 出力形状の確認
        assert multiclass_logits.shape == (self.batch_size, self.num_classes)
        assert binary_logits.shape == (self.batch_size, 1)

        # 有限値であることを確認
        assert torch.isfinite(multiclass_logits).all()
        assert torch.isfinite(binary_logits).all()

    def test_forward_pass_with_attention_mask(self):
        """Attention Maskありの順伝播テスト."""
        # ランダム入力データ
        imu_data = torch.randn(self.batch_size, self.input_dim, self.seq_len)

        # Attention Mask（最初の半分を有効に）
        attention_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        attention_mask[:, : self.seq_len // 2] = True

        # 順伝播
        multiclass_logits, binary_logits = self.model(imu_data, attention_mask=attention_mask)

        # 出力形状の確認
        assert multiclass_logits.shape == (self.batch_size, self.num_classes)
        assert binary_logits.shape == (self.batch_size, 1)

        # 有限値であることを確認
        assert torch.isfinite(multiclass_logits).all()
        assert torch.isfinite(binary_logits).all()

    def test_forward_pass_with_demographics(self):
        """Demographics特徴量ありの順伝播テスト."""
        # Demographics統合を有効にしたモデル
        model_with_demo = CMISqueezeformer(
            input_dim=self.input_dim,
            feature_extractor_config=self.config.feature_extractor.model_dump(),
            demographics_config={"enabled": True, "embedding_dim": 16},
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            num_classes=self.num_classes,
        )

        # ランダム入力データ
        imu_data = torch.randn(self.batch_size, self.input_dim, self.seq_len)

        # ダミーDemographics特徴量
        demographics = {
            "adult_child": torch.randint(0, 2, (self.batch_size,)),
            "sex": torch.randint(0, 2, (self.batch_size,)),
            "handedness": torch.randint(0, 2, (self.batch_size,)),
            "age": torch.randint(18, 65, (self.batch_size,)).float(),
            "height_cm": torch.randint(150, 190, (self.batch_size,)).float(),
            "shoulder_to_wrist_cm": torch.randint(40, 70, (self.batch_size,)).float(),
            "elbow_to_wrist_cm": torch.randint(20, 40, (self.batch_size,)).float(),
        }

        # 順伝播
        multiclass_logits, binary_logits = model_with_demo(imu_data, demographics=demographics)

        # 出力形状の確認
        assert multiclass_logits.shape == (self.batch_size, self.num_classes)
        assert binary_logits.shape == (self.batch_size, 1)

        # 有限値であることを確認
        assert torch.isfinite(multiclass_logits).all()
        assert torch.isfinite(binary_logits).all()

    def test_parameter_count(self):
        """パラメータ数の確認."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # パラメータが存在することを確認
        assert total_params > 0
        assert trainable_params > 0

        # 訓練可能パラメータが全パラメータ以下であることを確認
        assert trainable_params <= total_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def test_training_mode(self):
        """訓練モードのテスト."""
        self.model.train()

        # ランダム入力データ
        imu_data = torch.randn(self.batch_size, self.input_dim, self.seq_len)

        # 順伝播
        multiclass_logits, binary_logits = self.model(imu_data)

        # 損失計算
        target_multiclass = torch.randint(0, self.num_classes, (self.batch_size,))
        target_binary = torch.randint(0, 2, (self.batch_size,)).float()

        loss_multiclass = torch.nn.functional.cross_entropy(multiclass_logits, target_multiclass)
        loss_binary = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits.squeeze(), target_binary)

        total_loss = loss_multiclass + loss_binary

        # 逆伝播のテスト
        total_loss.backward()

        # 勾配が計算されていることを確認
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient not found for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient found for {name}"


class TestDatasetIntegration:
    """データセット統合テスト."""

    def setup_method(self):
        """テスト用のセットアップ."""
        # ダミーデータ作成
        self.df = self._create_dummy_data()
        self.config = Config()

        # データセット作成
        self.dataset = IMUDataset(self.df, target_sequence_length=100, augment=False, use_dynamic_padding=False)

    def _create_dummy_data(self) -> pl.DataFrame:
        """ダミーデータ作成."""
        sequences = []
        gestures = ["Above ear - pull hair", "Forehead - pull hairline", "Drink from bottle/cup", "Glasses on/off"]

        for seq_id in range(10):
            seq_len = np.random.randint(50, 150)

            data = {
                "sequence_id": [seq_id] * seq_len,
                "sequence_counter": list(range(seq_len)),
                "gesture": [gestures[seq_id % len(gestures)]] * seq_len,
                "acc_x": np.random.randn(seq_len),
                "acc_y": np.random.randn(seq_len),
                "acc_z": np.random.randn(seq_len),
                "rot_w": np.random.randn(seq_len),
                "rot_x": np.random.randn(seq_len),
                "rot_y": np.random.randn(seq_len),
                "rot_z": np.random.randn(seq_len),
            }

            sequences.append(pl.DataFrame(data))

        return pl.concat(sequences)

    def test_dataset_initialization(self):
        """データセット初期化テスト."""
        assert len(self.dataset) == 10  # 10シーケンス
        assert len(self.dataset.imu_cols) == 7  # 基本IMU特徴量のみ
        assert self.dataset.num_classes == 4  # 4つのジェスチャー

    def test_dataset_getitem(self):
        """データセット取得テスト."""
        sample = self.dataset[0]

        # キーの確認
        expected_keys = {"imu", "multiclass_label", "binary_label", "sequence_id", "gesture", "missing_mask"}
        assert set(sample.keys()) == expected_keys

        # IMUデータの形状確認（7次元の基本特徴量）
        assert sample["imu"].shape[0] == 7  # 基本IMU特徴量
        assert sample["imu"].shape[1] == 100  # target_sequence_length

        # ラベルの確認
        assert isinstance(sample["multiclass_label"], torch.Tensor)
        assert isinstance(sample["binary_label"], torch.Tensor)
        assert isinstance(sample["missing_mask"], torch.Tensor)

        # データ型の確認
        assert sample["imu"].dtype == torch.float32
        assert sample["multiclass_label"].dtype == torch.long
        assert sample["binary_label"].dtype == torch.float32
        assert sample["missing_mask"].dtype == torch.bool


class TestEndToEndPipeline:
    """エンドツーエンドパイプラインテスト."""

    def test_complete_pipeline(self):
        """完全なパイプラインテスト."""
        # 設定
        config = Config()

        # ダミーデータ作成
        df = self._create_dummy_data()

        # データセット作成
        dataset = IMUDataset(df, target_sequence_length=50, augment=False)

        # モデル作成
        model = CMISqueezeformer(
            input_dim=7,
            feature_extractor_config=config.feature_extractor.model_dump(),
            d_model=128,  # テスト用に小さく
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_classes=len(dataset.gesture_to_id),
            dropout=0.1,
        )

        # データローダー
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 1バッチの処理
        batch = next(iter(dataloader))

        # モデル予測
        model.eval()
        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                batch["imu"],
                attention_mask=~batch["missing_mask"],  # missing_maskの逆
            )

        # 出力の確認
        assert multiclass_logits.shape[0] == batch["imu"].shape[0]  # バッチサイズ
        assert multiclass_logits.shape[1] == len(dataset.gesture_to_id)  # クラス数
        assert binary_logits.shape == (batch["imu"].shape[0], 1)

        # 有限値であることを確認
        assert torch.isfinite(multiclass_logits).all()
        assert torch.isfinite(binary_logits).all()

        print("✓ End-to-end pipeline test passed!")

    def _create_dummy_data(self) -> pl.DataFrame:
        """ダミーデータ作成."""
        sequences = []
        gestures = ["Above ear - pull hair", "Forehead - pull hairline", "Drink from bottle/cup", "Glasses on/off"]

        for seq_id in range(8):
            seq_len = np.random.randint(30, 80)

            data = {
                "sequence_id": [seq_id] * seq_len,
                "sequence_counter": list(range(seq_len)),
                "gesture": [gestures[seq_id % len(gestures)]] * seq_len,
                "acc_x": np.random.randn(seq_len),
                "acc_y": np.random.randn(seq_len),
                "acc_z": np.random.randn(seq_len),
                "rot_w": np.random.randn(seq_len),
                "rot_x": np.random.randn(seq_len),
                "rot_y": np.random.randn(seq_len),
                "rot_z": np.random.randn(seq_len),
            }

            sequences.append(pl.DataFrame(data))

        return pl.concat(sequences)


if __name__ == "__main__":
    # テストの実行
    test_classes = [
        TestIMUFeatureExtractor,
        TestCMISqueezeformerIntegration,
        TestDatasetIntegration,
        TestEndToEndPipeline,
    ]

    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")

        instance = test_class()
        instance.setup_method()

        # テストメソッドを実行
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                print(f"Running {method_name}...")
                try:
                    getattr(instance, method_name)()
                    print(f"✓ {method_name} passed")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
                    raise

    print("\n🎉 All tests passed successfully!")
