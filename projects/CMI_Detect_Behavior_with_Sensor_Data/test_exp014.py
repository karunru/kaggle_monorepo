#!/usr/bin/env python3
"""
EXP014 基本動作テストスクリプト

MiniRocketMultivariate統合後の基本的な動作確認を行う。
- データセットの初期化テスト
- モデルの初期化テスト
- 前向き計算テスト
- MiniRocket変換テスト
"""

import sys
from pathlib import Path

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp014"))

import numpy as np
import polars as pl
import torch
from config import Config
from dataset import IMUDataset
from model import CMISqueezeformer

def create_dummy_data(n_samples: int = 100, n_timepoints: int = 200) -> pl.DataFrame:
    """ダミーのIMUデータを作成."""
    print(f"Creating dummy data: {n_samples} samples, {n_timepoints} timepoints")
    
    data = []
    for seq_idx in range(n_samples):
        for time_idx in range(n_timepoints):
            # 基本IMUデータ
            row = {
                "sequence_id": seq_idx,
                "sequence_counter": time_idx,
                "subject": f"subject_{seq_idx % 10}",  # 10人の被験者
                "gesture": f"Gesture_{seq_idx % 18}",  # 18クラス
                
                # 基本IMUデータ
                "acc_x": np.random.normal(0, 1),
                "acc_y": np.random.normal(0, 1), 
                "acc_z": np.random.normal(9.8, 1),  # 重力込み
                "rot_w": np.random.normal(0, 0.1),
                "rot_x": np.random.normal(0, 0.1),
                "rot_y": np.random.normal(0, 0.1),
                "rot_z": np.random.normal(0, 0.1),
            }
            data.append(row)
    
    df = pl.DataFrame(data)
    print(f"Generated dataframe shape: {df.shape}")
    return df


def create_dummy_demographics(n_subjects: int = 10) -> pl.DataFrame:
    """ダミーのdemographicsデータを作成."""
    print(f"Creating dummy demographics data for {n_subjects} subjects")
    
    data = []
    for i in range(n_subjects):
        row = {
            "subject": f"subject_{i}",
            "adult_child": np.random.randint(0, 2),  # 0=child, 1=adult
            "age": np.random.randint(10, 60),
            "sex": np.random.randint(0, 2),  # 0=female, 1=male
            "handedness": np.random.randint(0, 2),  # 0=left, 1=right
            "height_cm": np.random.normal(165, 15),
            "shoulder_to_wrist_cm": np.random.normal(55, 8),
            "elbow_to_wrist_cm": np.random.normal(30, 5),
        }
        data.append(row)
    
    df = pl.DataFrame(data)
    print(f"Generated demographics shape: {df.shape}")
    return df


def test_dataset_initialization():
    """データセット初期化のテスト."""
    print("\n=== Dataset Initialization Test ===")
    
    # ダミーデータ作成
    df = create_dummy_data(n_samples=50, n_timepoints=150)
    demographics_df = create_dummy_demographics(n_subjects=10)
    
    # 設定読み込み
    config = Config()
    
    try:
        # データセット初期化（MiniRocket有効）
        dataset = IMUDataset(
            df=df,
            target_sequence_length=config.preprocessing.target_sequence_length,
            demographics_data=demographics_df,
            demographics_config=config.demographics.model_dump(),
            rocket_config=config.rocket.model_dump(),
        )
        
        print(f"✅ Dataset initialized successfully")
        print(f"   - Number of sequences: {len(dataset)}")
        print(f"   - Use demographics: {dataset.use_demographics}")
        print(f"   - Use MiniRocket: {dataset.use_rocket}")
        
        # サンプルデータ取得テスト
        sample = dataset[0]
        print(f"   - Sample IMU shape: {sample['imu'].shape}")
        print(f"   - Sample multiclass label: {sample['multiclass_label']}")
        print(f"   - Sample binary label: {sample['binary_label']}")
        
        if 'demographics' in sample:
            print(f"   - Demographics features: {list(sample['demographics'].keys())}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """モデル初期化のテスト."""
    print("\n=== Model Initialization Test ===")
    
    config = Config()
    effective_input_dim = config.get_effective_input_dim()
    
    try:
        # モデル初期化
        model = CMISqueezeformer(
            input_dim=effective_input_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            num_classes=config.model.num_classes,
            demographics_config=config.demographics.model_dump(),
        )
        
        print(f"✅ Model initialized successfully")
        print(f"   - Input dimension: {effective_input_dim}")
        print(f"   - Model dimension: {config.model.d_model}")
        print(f"   - Number of layers: {config.model.n_layers}")
        
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """前向き計算のテスト."""
    print("\n=== Forward Pass Test ===")
    
    if model is None:
        print("❌ Model is None, skipping forward pass test")
        return False
    
    config = Config()
    effective_input_dim = config.get_effective_input_dim()
    
    try:
        # ダミー入力データ
        batch_size = 4
        seq_len = config.preprocessing.target_sequence_length
        
        dummy_imu = torch.randn(batch_size, effective_input_dim, seq_len)
        
        # Demographics特徴量（オプション）
        dummy_demographics = None
        if config.demographics.enabled:
            dummy_demographics = {
                "adult_child": torch.randint(0, 2, (batch_size,), dtype=torch.long),
                "sex": torch.randint(0, 2, (batch_size,), dtype=torch.long),
                "handedness": torch.randint(0, 2, (batch_size,), dtype=torch.long),
                "age": torch.rand(batch_size, dtype=torch.float32) * 50 + 10,
                "height_cm": torch.rand(batch_size, dtype=torch.float32) * 60 + 140,
                "shoulder_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 30 + 40,
                "elbow_to_wrist_cm": torch.rand(batch_size, dtype=torch.float32) * 25 + 20,
            }
        
        # 前向き計算
        model.eval()
        with torch.no_grad():
            multiclass_logits, binary_logits = model(dummy_imu, demographics=dummy_demographics)
        
        print(f"✅ Forward pass completed successfully")
        print(f"   - Input shape: {dummy_imu.shape}")
        print(f"   - Multiclass output: {multiclass_logits.shape}")
        print(f"   - Binary output: {binary_logits.shape}")
        print(f"   - Demographics included: {dummy_demographics is not None}")
        
        # 出力値の妥当性チェック
        assert multiclass_logits.shape == (batch_size, config.model.num_classes)
        assert binary_logits.shape == (batch_size, 1)
        assert not torch.isnan(multiclass_logits).any()
        assert not torch.isnan(binary_logits).any()
        
        print(f"   - Output validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """統合テスト."""
    print("\n=== Integration Test ===")
    
    try:
        # データ作成
        df = create_dummy_data(n_samples=20, n_timepoints=100)
        demographics_df = create_dummy_demographics(n_subjects=10)
        
        # 設定
        config = Config()
        
        # データセット作成
        dataset = IMUDataset(
            df=df,
            target_sequence_length=config.preprocessing.target_sequence_length,
            demographics_data=demographics_df,
            demographics_config=config.demographics.model_dump(),
            rocket_config=config.rocket.model_dump(),
        )
        
        # モデル作成
        model = CMISqueezeformer(
            input_dim=config.get_effective_input_dim(),
            d_model=config.model.d_model,
            n_layers=2,  # テスト用に軽量化
            n_heads=4,
            d_ff=512,
            num_classes=config.model.num_classes,
            demographics_config=config.demographics.model_dump(),
        )
        
        # データセットからのサンプル取得
        sample = dataset[0]
        
        # モデルに入力
        model.eval()
        with torch.no_grad():
            imu_input = sample['imu'].unsqueeze(0)  # バッチ次元追加
            demographics_input = {k: v.unsqueeze(0) for k, v in sample.get('demographics', {}).items()} if 'demographics' in sample else None
            
            multiclass_logits, binary_logits = model(imu_input, demographics=demographics_input)
        
        print(f"✅ Integration test passed")
        print(f"   - Dataset → Model pipeline works correctly")
        print(f"   - Input shape: {imu_input.shape}")
        print(f"   - Output shapes: multiclass={multiclass_logits.shape}, binary={binary_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行."""
    print("EXP014 Basic Operation Test")
    print("=" * 50)
    
    results = []
    
    # 各テストの実行
    results.append(("Dataset Initialization", test_dataset_initialization()))
    
    model = test_model_initialization()
    results.append(("Model Initialization", model is not None))
    
    results.append(("Forward Pass", test_forward_pass(model)))
    results.append(("Integration", test_integration()))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EXP014 implementation is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)