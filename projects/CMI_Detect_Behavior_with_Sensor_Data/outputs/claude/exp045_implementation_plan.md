# exp045 実装計画書

## 概要
exp036をベースに、以下の3つの改善を実装する：
1. model.pyのDense layersをnn.Sequentialでリファクタリング
2. IMUOnlyLSTMのマジックナンバーをModelConfigに移動して設定可能にする
3. train時はphase="Gesture"のみ、validation/inference時は全phase（Transition+Gesture）を使用する

## 実装タスク詳細

### タスク1: exp036をexp045にコピー
**ファイル:**
- `codes/exp/exp036/` から `codes/exp/exp045/` へコピー
- test_*.pyファイルは除外
- コピー対象：
  - config.py
  - dataset.py
  - losses.py
  - model.py
  - __init__.py
  - human_normalization.py
  - train.py
  - inference.py

### タスク2: config.pyの更新
**ファイル:** `codes/exp/exp045/config.py`

**変更内容:**
1. EXP_NUM = "045"に変更
2. ExperimentConfig.descriptionを更新：
   ```python
   description="exp036 + Dense layers refactoring + configurable LSTM params + phase-based filtering"
   ```
3. ExperimentConfig.tagsを更新：
   ```python
   tags=["lstm", "phase_filtering", "gesture_only_train", "refactored"]
   ```
4. LoggingConfig.wandb_tagsを更新：
   ```python
   wandb_tags=["exp045", "phase_filtering", "gesture_only_train"]
   ```
5. ModelConfigに新しいパラメータを追加：
   ```python
   # LSTM architecture parameters
   lstm_hidden_dim: int = Field(default=128, description="LSTM隠れ層の次元数")
   lstm_num_layers: int = Field(default=1, description="LSTMレイヤー数")
   lstm_bidirectional: bool = Field(default=True, description="双方向LSTMを使用")
   
   # Dense layer parameters
   dense1_dim: int = Field(default=256, description="第1全結合層の次元数")
   dense2_dim: int = Field(default=128, description="第2全結合層の次元数")
   dense1_dropout: float = Field(default=0.5, description="第1全結合層のドロップアウト率")
   dense2_dropout: float = Field(default=0.3, description="第2全結合層のドロップアウト率")
   
   # CNN block parameters
   cnn_block1_out_channels: int = Field(default=64, description="第1CNNブロックの出力チャンネル数")
   cnn_block2_out_channels: int = Field(default=128, description="第2CNNブロックの出力チャンネル数")
   cnn_block1_kernel_size: int = Field(default=3, description="第1CNNブロックのカーネルサイズ")
   cnn_block2_kernel_size: int = Field(default=5, description="第2CNNブロックのカーネルサイズ")
   cnn_dropout: float = Field(default=0.3, description="CNNブロックのドロップアウト率")
   gru_dropout: float = Field(default=0.4, description="GRUレイヤーのドロップアウト率")
   ```

6. DataConfigに新しいパラメータを追加：
   ```python
   filter_gesture_only_train: bool = Field(default=True, description="訓練時にGestureフェーズのみをフィルタリング")
   ```

### タスク3: model.pyのリファクタリング
**ファイル:** `codes/exp/exp045/model.py`

**変更内容:**

1. IMUOnlyLSTMクラスの__init__メソッドを更新：
   - ModelConfigからパラメータを受け取るようにする
   - Dense layersをnn.Sequentialでまとめる

```python
def __init__(
    self,
    imu_dim=20,
    n_classes=18,
    weight_decay=1e-4,
    demographics_dim=0,
    dropout=0.1,
    model_config=None,  # 新規追加
):
    super().__init__()
    self.imu_dim = imu_dim
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.demographics_dim = demographics_dim
    
    # ModelConfigからパラメータを取得（デフォルト値も設定）
    if model_config is None:
        # デフォルト値（後方互換性のため）
        lstm_hidden_dim = 128
        cnn_block1_out = 64
        cnn_block2_out = 128
        cnn_block1_kernel = 3
        cnn_block2_kernel = 5
        cnn_dropout = 0.3
        gru_dropout = 0.4
        dense1_dim = 256
        dense2_dim = 128
        dense1_dropout = 0.5
        dense2_dropout = 0.3
    else:
        lstm_hidden_dim = model_config.lstm_hidden_dim
        cnn_block1_out = model_config.cnn_block1_out_channels
        cnn_block2_out = model_config.cnn_block2_out_channels
        cnn_block1_kernel = model_config.cnn_block1_kernel_size
        cnn_block2_kernel = model_config.cnn_block2_kernel_size
        cnn_dropout = model_config.cnn_dropout
        gru_dropout = model_config.gru_dropout
        dense1_dim = model_config.dense1_dim
        dense2_dim = model_config.dense2_dim
        dense1_dropout = model_config.dense1_dropout
        dense2_dropout = model_config.dense2_dropout

    # IMU deep branch with ResidualSE-CNN blocks
    self.imu_block1 = ResidualSECNNBlock(imu_dim, cnn_block1_out, cnn_block1_kernel, dropout=cnn_dropout, weight_decay=weight_decay)
    self.imu_block2 = ResidualSECNNBlock(cnn_block1_out, cnn_block2_out, cnn_block2_kernel, dropout=cnn_dropout, weight_decay=weight_decay)

    # BiGRU
    self.bigru = nn.GRU(cnn_block2_out, lstm_hidden_dim, bidirectional=True, batch_first=True)
    self.gru_dropout = nn.Dropout(gru_dropout)

    # Attention
    self.attention = AttentionLayer(lstm_hidden_dim * 2)  # *2 for bidirectional

    # Dense layers (基本特徴量抽出) - nn.Sequentialにリファクタリング
    self.dense_layers = nn.Sequential(
        nn.Linear(lstm_hidden_dim * 2, dense1_dim, bias=False),
        nn.BatchNorm1d(dense1_dim),
        nn.Mish(),
        nn.Dropout(dense1_dropout),
        nn.Linear(dense1_dim, dense2_dim, bias=False),
        nn.BatchNorm1d(dense2_dim),
        nn.Mish(),
        nn.Dropout(dense2_dropout)
    )

    # Classification heads
    classification_input_dim = dense2_dim + self.demographics_dim
    # ... (rest remains the same)
```

2. forwardメソッドの更新：
```python
def forward(self, x, demographics_embedding=None):
    # ... (前半部分は同じ)
    
    # Dense layers (基本特徴量抽出) - nn.Sequential使用
    imu_features = self.dense_layers(attended)
    
    # ... (後半部分は同じ)
```

3. CMISqueezeformerクラスの__init__でmodel_configを渡す：
```python
self.imu_model = IMUOnlyLSTM(
    imu_dim=actual_input_dim,
    n_classes=self.config.model.num_classes,
    weight_decay=1e-4,
    demographics_dim=demographics_dim,
    dropout=self.config.model.dropout,
    model_config=self.config.model  # 追加
)
```

### タスク4: dataset.pyのphaseベースフィルタリング実装
**ファイル:** `codes/exp/exp045/dataset.py`

**変更内容:**

1. IMUDatasetクラスの__init__にパラメータ追加：
```python
def __init__(
    self,
    df: pl.DataFrame,
    target_sequence_length: int = 200,
    augment: bool = False,
    augmentation_config: AugmentationConfig | None = None,
    use_dynamic_padding: bool = False,
    demographics_data: pl.DataFrame | None = None,
    demographics_config: DemographicsConfig | None = None,
    enable_handedness_aug: bool = False,
    handedness_flip_prob: float = 0.5,
    target_gestures: list[str] | None = None,
    non_target_gestures: list[str] | None = None,
    filter_gesture_only: bool = False,  # 新規追加
):
    # ... 既存のコード ...
    self.filter_gesture_only = filter_gesture_only
    # ... 既存のコード ...
```

2. _preprocess_dataメソッドを更新してphaseフィルタリングを追加：
```python
def _preprocess_data(self):
    """データの前処理."""
    # 基本IMU列の存在確認
    basic_imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    missing_cols = [col for col in basic_imu_cols if col not in self.df.columns]
    if missing_cols:
        raise ValueError(f"Missing basic IMU columns: {missing_cols}")
    
    # phaseフィルタリング（trainのみ）
    df_to_process = self.df
    if self.filter_gesture_only and "phase" in self.df.columns:
        print("Filtering data to Gesture phase only...")
        df_to_process = self.df.filter(pl.col("phase") == "Gesture")
        print(f"Filtered from {len(self.df)} to {len(df_to_process)} rows (Gesture phase only)")
    
    # Polarsベクトル化処理を使用（欠損値マスク付き）
    self.sequence_data = self._preprocess_data_vectorized_with_mask(df_to_process)
    
    # ... 既存のコード ...
```

3. _preprocess_data_vectorized_with_maskメソッドのシグネチャを更新：
```python
def _preprocess_data_vectorized_with_mask(self, df: pl.DataFrame | None = None):
    """ベクトル化された前処理（欠損値マスク付き）."""
    if df is None:
        df = self.df
    
    # ... 既存のコードでself.dfをdfに置き換え ...
```

4. IMUDataModuleクラスのsetupメソッドを更新：
```python
def setup(self, stage: str | None = None):
    """データセットのセットアップ."""
    print(f"\nSetting up data module for fold {self.fold}, stage: {stage}")
    if stage == "fit" or stage is None:
        # ... 既存のコード ...
        
        print(f"Creating train dataset ({len(train_sequence_ids)} rows)...")
        self.train_dataset = IMUDataset(
            train_sequence_ids,
            target_sequence_length=self.target_sequence_length,
            augment=True,
            augmentation_config=self.augmentation_config,
            use_dynamic_padding=self.use_dynamic_padding,
            demographics_data=self.demographics_data,
            demographics_config=self.config.demographics,
            enable_handedness_aug=self.config.augmentation.enable_handedness_flip,
            handedness_flip_prob=self.config.augmentation.handedness_flip_prob,
            filter_gesture_only=self.config.data.filter_gesture_only_train,  # 追加
        )

        print(f"Creating validation dataset ({len(val_sequence_ids)} rows)...")
        self.val_dataset = IMUDataset(
            val_sequence_ids,
            target_sequence_length=self.target_sequence_length,
            augment=False,
            use_dynamic_padding=self.use_dynamic_padding,
            demographics_data=self.demographics_data,
            demographics_config=self.config.demographics,
            enable_handedness_aug=False,
            handedness_flip_prob=0.0,
            filter_gesture_only=False,  # validationでは全データを使用
        )
        # ... 既存のコード ...
```

### タスク5: inferenceスクリプトの更新
**ファイル:** `codes/exp/exp045/inference.py`

1. 推論時はphaseフィルタリングを無効にすることを確認（filter_gesture_only=False）

### タスク6: テストと検証
1. 静的解析の実行：
   - `mise run format`
   - `mise run lint`
   - `mise run type-check`

2. 動作確認：
   - `cd codes/exp/exp045 && uv run python train.py`で訓練が開始されることを確認
   - データのフィルタリングが正しく動作することを確認（ログで確認）

## 注意事項
- exp036からのインポートは禁止
- 差分は指示されたもの以外最小限にする
- 他のexp directory配下のファイルの編集は禁止

## 期待される効果
1. **コードの保守性向上**: Dense layersがnn.Sequentialにまとめられることで、レイヤー構造が明確になる
2. **設定の柔軟性向上**: LSTMのパラメータがconfigから設定可能になる
3. **訓練の効率化**: Gestureフェーズのみで訓練することで、実際のジェスチャー認識に集中した学習が可能
4. **推論精度の維持**: validationとinferenceでは全データを使用することで、実運用時の性能を正確に評価