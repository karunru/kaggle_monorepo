#!/usr/bin/env python3
"""
EXP014 train.py修正後の動作確認スクリプト

修正されたtrain.pyで正しく入力次元（352次元）が使用されるかを確認する。
"""

import sys
from pathlib import Path

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp014"))

import torch
import pytorch_lightning as pl
from config import Config
from model import CMISqueezeformer
from dataset import IMUDataModule

def test_model_initialization_with_config():
    """設定ファイルからモデル初期化テスト"""
    print("=== Model Initialization Test with Config ===")
    
    config = Config()
    
    # 実効入力次元の確認
    effective_input_dim = config.get_effective_input_dim()
    print(f"Effective input dim: {effective_input_dim}")
    print(f"Base IMU features: {config.model.base_imu_features}")
    print(f"MiniRocket enabled: {config.rocket.enabled}")
    print(f"MiniRocket kernels: {config.rocket.num_kernels}")
    
    # モデル作成（train.pyと同じ方法）
    try:
        model = CMISqueezeformer(
            input_dim=config.get_effective_input_dim(),  # 修正後の方法
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            num_classes=config.model.num_classes,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            scheduler_config={
                "type": config.training.scheduler_type,
                "min_lr": config.training.scheduler_min_lr,
                "factor": config.training.scheduler_factor,
                "patience": config.training.scheduler_patience,
            },
            loss_config=config.loss.model_dump(),
            acls_config=config.acls.model_dump(),
            schedule_free_config=config.schedule_free.model_dump(),
            ema_config=config.ema.model_dump(),
            target_gestures=config.target_gestures,
            non_target_gestures=config.non_target_gestures,
        )
        
        print("✅ Model created successfully")
        print(f"Model input_dim: {model.input_dim}")
        
        # パラメータ数確認
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_module_initialization():
    """DataModuleの初期化テスト"""
    print("\n=== DataModule Initialization Test ===")
    
    config = Config()
    
    try:
        # DataModuleの作成（train.pyと同じ方法）
        data_module = IMUDataModule(config, fold=0)
        print("✅ DataModule created successfully")
        
        # セットアップ
        data_module.setup("fit")
        print("✅ DataModule setup completed")
        
        # データローダーの取得
        train_loader = data_module.train_dataloader()
        print(f"Train loader created - batches: {len(train_loader)}")
        
        # サンプルバッチの形状確認
        sample_batch = next(iter(train_loader))
        imu_shape = sample_batch["imu"].shape
        print(f"Sample batch IMU shape: {imu_shape}")
        
        # 期待される形状との比較
        expected_features = config.get_effective_input_dim()
        if imu_shape[1] == expected_features:
            print(f"✅ Input dimension matches expected: {expected_features}")
        else:
            print(f"❌ Dimension mismatch - expected: {expected_features}, got: {imu_shape[1]}")
            
        return data_module, sample_batch
        
    except Exception as e:
        print(f"❌ DataModule initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass_compatibility():
    """前向き計算の互換性テスト"""
    print("\n=== Forward Pass Compatibility Test ===")
    
    config = Config()
    
    # モデル作成
    model = test_model_initialization_with_config()
    if model is None:
        print("❌ Cannot test forward pass - model creation failed")
        return False
    
    # データモジュール作成
    data_module, sample_batch = test_data_module_initialization()
    if data_module is None or sample_batch is None:
        print("❌ Cannot test forward pass - data module creation failed")
        return False
    
    try:
        # モデルを評価モードに設定
        model.eval()
        
        # サンプルバッチで前向き計算
        with torch.no_grad():
            imu = sample_batch["imu"]
            attention_mask = sample_batch.get("attention_mask")
            demographics = sample_batch.get("demographics")
            
            print(f"Forward pass input shapes:")
            print(f"  IMU: {imu.shape}")
            print(f"  Attention mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            
            # 前向き計算実行
            multiclass_logits, binary_logits = model(imu, attention_mask, demographics)
            
            print(f"✅ Forward pass successful")
            print(f"  Multiclass output: {multiclass_logits.shape}")
            print(f"  Binary output: {binary_logits.shape}")
            
            # 出力値の妥当性確認
            assert not torch.isnan(multiclass_logits).any()
            assert not torch.isnan(binary_logits).any()
            
            print(f"✅ Output validation passed")
            return True
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("EXP014 Train.py Fix Verification Test")
    print("=" * 50)
    
    # 各テスト実行
    results = []
    
    model = test_model_initialization_with_config()
    results.append(("Model Initialization", model is not None))
    
    data_module, sample_batch = test_data_module_initialization()  
    results.append(("DataModule Initialization", data_module is not None and sample_batch is not None))
    
    forward_pass_success = test_forward_pass_compatibility()
    results.append(("Forward Pass Compatibility", forward_pass_success))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! train.py fix is working correctly.")
        print("💡 The input dimension issue has been resolved.")
        return True
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)