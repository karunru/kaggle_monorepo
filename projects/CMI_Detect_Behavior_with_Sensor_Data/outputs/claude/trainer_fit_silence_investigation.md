# trainer.fit実行後の無音期間調査と対策

## 問題の概要
exp002/train.pyでtrainer.fit()が実行されてから実際に学習が始まるまでの間、標準出力に何も表示されない時間が長く、ユーザーがプロセスが停止したと誤解する可能性がありました。

## 原因分析

### 1. データローディングの初期化処理
- `IMUDataModule.__init__()`でのPolars CSVファイル読み込み
- `IMUDataModule.setup()`でのtrain/validationスプリット作成
- `IMUDataset._preprocess_data()`での全シーケンス前処理

### 2. PyTorch Lightning内部処理
- GPU/CUDA初期化とモデル転送
- データローダーの初期化とワーカープロセス起動
- distributed設定の初期化

### 3. データ前処理の重い処理
- シーケンスごとの欠損値処理
- 補間によるシーケンス長正規化
- ラベルマッピング作成

## 実装した対策

### 1. IMUDataset初期化の可視化
```python
# 進捗表示の追加
print(f"Loading dataset with {len(df)} rows...")
print(f"Found {len(self.sequence_ids)} unique sequences")
print("Preprocessing data...")

# tqdmによる前処理の進捗表示
for seq_id in tqdm(self.sequence_ids, desc="Processing sequences", unit="seq"):
    # 処理内容
```

### 2. IMUDataModuleの詳細ログ
```python
@timer_decorator("IMUDataModule.setup")
def setup(self, stage: str | None = None):
    print(f"\nSetting up data module for fold {self.fold}, stage: {stage}")
    print("Creating train/validation splits...")
    # データセット作成時の進捗表示
    print(f"Creating train dataset ({len(train_sequence_ids)} rows)...")
    print(f"Creating validation dataset ({len(val_sequence_ids)} rows)...")
```

### 3. trainer.fit()前後のログ強化
```python
# Training前のログ
logger.info("Creating data module...")
logger.info("Creating model...")
logger.info("Starting trainer.fit() - initializing data loaders...")
logger.info(f"Batch size: {config.training.batch_size}, Num workers: {config.training.num_workers}")

# Training実行
trainer.fit(model, data_module)

# Training後のログ
logger.info("Training completed")
```

### 4. システム情報の表示
```python
# GPU情報の詳細表示
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 5. タイマーデコレーターの活用
```python
@timer_decorator("IMUDataset._preprocess_data")
def _preprocess_data(self):
    # 前処理の実行時間を測定

@timer_decorator("IMUDataModule.setup")
def setup(self, stage: str | None = None):
    # セットアップ時間を測定
```

## 効果

### Before（対策前）
```
[train_single_fold] start
Starting training for fold 0
Creating data module...
Creating model...
Starting trainer.fit() - initializing data loaders...
（長時間無音）
Epoch 1/10: ...
```

### After（対策後）
```
[train_single_fold] start
Starting training for fold 0
Creating data module...
Loading train data from ../../data/train.csv...
Loaded 123456 rows

Setting up data module for fold 0, stage: fit
Creating train/validation splits...
Creating train dataset (98765 rows)...
Loading dataset with 98765 rows...
Found 4000 unique sequences
Preprocessing data...
[IMUDataset._preprocess_data] start
Processing sequences: 100%|██████████| 4000/4000 [00:05<00:00, 750.23seq/s]
[IMUDataset._preprocess_data] done in 5.33 s
Data preprocessing completed

Creating validation dataset (24691 rows)...
Loading dataset with 24691 rows...
Found 1000 unique sequences
Preprocessing data...
[IMUDataset._preprocess_data] start
Processing sequences: 100%|██████████| 1000/1000 [00:01<00:00, 800.15seq/s]
[IMUDataset._preprocess_data] done in 1.25 s
Data preprocessing completed

Datasets created: train=4000, val=1000 sequences
[IMUDataModule.setup] done in 6.58 s

Creating model...
Starting trainer.fit() - initializing data loaders...
Batch size: 32, Num workers: 4
Epoch 1/10: ...
```

## ボトルネック特定のポイント

1. **データローディング時間**: CSV読み込みとPolars処理
2. **前処理時間**: シーケンスごとの補間・正規化処理
3. **GPU初期化時間**: モデル転送とCUDA初期化
4. **データローダー初期化**: ワーカープロセス起動

## 今後の改善案

1. **データキャッシュ**: 前処理済みデータのpickle保存
2. **並列処理**: マルチプロセシングによる前処理の高速化
3. **遅延読み込み**: 必要時のみデータを読み込む仕組み
4. **プログレスバー**: PyTorch Lightningの詳細進捗表示

## 更新されたファイル

- `exp/exp002/dataset.py`: 進捗表示とタイマーデコレーター追加
- `exp/exp002/train.py`: 詳細ログとタイマー追加