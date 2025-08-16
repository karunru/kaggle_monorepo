#!/usr/bin/env python3
"""BERT無効化でのモデル検証."""

import sys
from pathlib import Path

import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import Config
from dataset import IMUDataModule


def test_simple_model():
    """BERT無しの簡単なモデルで検証."""
    print("=== 簡単なモデル検証（BERT無し） ===")

    config = Config()
    config.demographics.hn_enabled = True  # HN有効化
    config.demographics.enabled = True     # Demographics有効化

    print(f"HN enabled: {config.demographics.hn_enabled}")
    print(f"Demographics enabled: {config.demographics.enabled}")

    # データモジュール作成
    data_module = IMUDataModule(config, fold=0)
    data_module.setup("fit")

    # 入力次元を動的に取得
    actual_input_dim = len(data_module.train_dataset.imu_cols)
    print(f"Model input_dim: {actual_input_dim}")

    # 簡単なモデルを作成（BERTを使わずに）
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim, num_classes=18):
            super().__init__()
            self.input_projection = torch.nn.Linear(input_dim, 256)
            self.dropout = torch.nn.Dropout(0.1)
            self.multiclass_head = torch.nn.Linear(256, num_classes)
            self.binary_head = torch.nn.Linear(256, 1)

            # 重み初期化
            torch.nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
            torch.nn.init.constant_(self.input_projection.bias, 0.0)
            torch.nn.init.xavier_uniform_(self.multiclass_head.weight, gain=0.5)
            torch.nn.init.constant_(self.multiclass_head.bias, 0.0)
            torch.nn.init.xavier_uniform_(self.binary_head.weight, gain=0.5)
            torch.nn.init.constant_(self.binary_head.bias, 0.0)

        def forward(self, imu_data, attention_mask=None, demographics=None):
            # [batch, features, seq_len] -> [batch, seq_len, features]
            x = imu_data.transpose(1, 2)

            # Input projection
            x = self.input_projection(x)
            x = torch.relu(x)
            x = self.dropout(x)

            # Global average pooling
            if attention_mask is not None:
                # マスクを適用
                mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
                x = x * mask  # マスク適用
                x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # 平均
            else:
                x = x.mean(dim=1)  # 単純平均

            # Classification heads
            multiclass_logits = self.multiclass_head(x)
            binary_logits = self.binary_head(x)

            return multiclass_logits, binary_logits

    model = SimpleModel(actual_input_dim)
    model.eval()

    print("Simple model created successfully")

    # データローダーから1バッチ取得
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")

    with torch.no_grad():
        try:
            # バッチから適切な要素を抽出
            imu_data = batch['imu']
            attention_mask = batch.get('missing_mask', None)
            demographics = batch.get('demographics', None)

            print(f"IMU shape: {imu_data.shape}")
            print(f"Mask shape: {attention_mask.shape if attention_mask is not None else None}")

            # 前方パス実行
            multiclass_logits, binary_logits = model(imu_data, attention_mask, demographics)
            print("✅ Forward pass successful!")

            # 各出力をチェック
            print(f"multiclass_logits: shape={multiclass_logits.shape}, nan={torch.isnan(multiclass_logits).any()}, inf={torch.isinf(multiclass_logits).any()}")
            print(f"binary_logits: shape={binary_logits.shape}, nan={torch.isnan(binary_logits).any()}, inf={torch.isinf(binary_logits).any()}")

            if torch.isnan(multiclass_logits).any():
                print("  ⚠️ NaN detected in multiclass_logits")
                return False
            if torch.isinf(multiclass_logits).any():
                print("  ⚠️ Inf detected in multiclass_logits")
                return False
            if torch.isnan(binary_logits).any():
                print("  ⚠️ NaN detected in binary_logits")
                return False
            if torch.isinf(binary_logits).any():
                print("  ⚠️ Inf detected in binary_logits")
                return False

            print("✅ No NaN/Inf detected - Simple model is stable")
            return True

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("\n✅ 簡単なモデルでは正常動作: BERT/Transformerが原因")
        print("対策: BERT処理またはTransformer attention機構を調査")
    else:
        print("\n❌ 簡単なモデルでもNaN発生: データ自体に問題")
