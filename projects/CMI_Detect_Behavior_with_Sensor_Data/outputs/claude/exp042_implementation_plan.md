# EXP042 実装計画

## 概要

exp042は、exp041をベースにしてIMUOnlyLSTMモデルのマジックナンバーをModelConfigで設定可能にし、ScheduleFree Optimizerを有効化する実装です。

### 目的
1. **設定可能性の向上**: IMUOnlyLSTMのハードコードされたパラメータを設定ファイルから制御可能にする
2. **最適化手法の改善**: ScheduleFree Optimizerを有効にして学習率スケジューリングを自動化する
3. **保守性の向上**: モデルアーキテクチャの調整を容易にする

## 変更内容

### 1. プロジェクト構造
```
codes/exp/exp042/
├── __init__.py                  # exp041からコピー
├── config.py                   # 更新（exp042用設定）
├── dataset.py                  # exp041からコピー
├── human_normalization.py      # exp041からコピー  
├── inference.py                # exp041からコピー
├── losses.py                   # exp041からコピー
├── model.py                    # 更新（ModelConfig使用）
└── train.py                    # exp041からコピー
```

**注意**: test_*.pyファイルは指示通り除外します。

### 2. config.py の変更

#### 2.1 ExperimentConfig
- `exp_num`: "exp041" → "exp042"
- `name`: "exp041_orient_auxiliary_task" → "exp042_configurable_imu_lstm" 
- `description`: "ConvNeXt1D with SE attention for IMU gesture recognition" → "Configurable IMU-LSTM with Schedule-Free optimization"
- `tags`: Schedule-Free関連タグを追加

#### 2.2 ModelConfig 拡張
IMUOnlyLSTMのマジックナンバーを設定可能にするため、以下のフィールドを追加：

```python
class ModelConfig(BaseModel):
    # 既存フィールド（省略）
    
    # IMUOnlyLSTM固有設定
    imu_block1_out_channels: int = Field(default=64, description="IMU Block1出力チャンネル数")
    imu_block1_kernel_size: int = Field(default=3, description="IMU Block1カーネルサイズ")
    imu_block1_num_blocks: int = Field(default=2, description="IMU Block1ブロック数")
    imu_block1_drop_path: float = Field(default=0.1, description="IMU Block1 DropPath率")
    imu_block1_dropout: float = Field(default=0.3, description="IMU Block1ドロップアウト率")
    
    imu_block2_out_channels: int = Field(default=128, description="IMU Block2出力チャンネル数")
    imu_block2_kernel_size: int = Field(default=5, description="IMU Block2カーネルサイズ")
    imu_block2_num_blocks: int = Field(default=2, description="IMU Block2ブロック数")
    imu_block2_drop_path: float = Field(default=0.15, description="IMU Block2 DropPath率")
    imu_block2_dropout: float = Field(default=0.3, description="IMU Block2ドロップアウト率")
    
    bigru_hidden_size: int = Field(default=128, description="BiGRU隠れ層サイズ")
    gru_dropout: float = Field(default=0.4, description="GRUドロップアウト率")
    
    dense1_hidden_size: int = Field(default=256, description="Dense Layer 1隠れ層サイズ")
    dense2_hidden_size: int = Field(default=128, description="Dense Layer 2隠れ層サイズ")
    dense1_dropout: float = Field(default=0.5, description="Dense Layer 1ドロップアウト率")
    dense2_dropout: float = Field(default=0.3, description="Dense Layer 2ドロップアウト率")
    
    multiclass_head_hidden: int = Field(default=64, description="マルチクラスヘッド隠れ層サイズ")
    binary_head_hidden: int = Field(default=32, description="バイナリヘッド隠れ層サイズ")
    nine_class_head_hidden: int = Field(default=64, description="9クラスヘッド隠れ層サイズ")
    orientation_head_hidden: int = Field(default=32, description="Orientationヘッド隠れ層サイズ")
```

#### 2.3 ScheduleFreeConfig
- `enabled`: False → True（デフォルト値を変更）

#### 2.4 LoggingConfig
- `wandb_tags`: exp042関連タグを追加、schedule_free関連タグを追加

### 3. model.py の変更

#### 3.1 IMUOnlyLSTMクラス更新
`__init__`メソッドでModelConfigを受け取り、設定値を使用するよう変更：

```python
class IMUOnlyLSTM(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,  # 新規追加
        imu_dim=20,
        n_classes=18,
        weight_decay=1e-4,
        demographics_dim=0,
        dropout=0.1,
    ):
        super().__init__()
        # 設定値を使用してモデル構築
        self.imu_block1 = ResidualSEConvNeXtBlock(
            in_channels=imu_dim,
            out_channels=model_config.imu_block1_out_channels,  # 設定から取得
            kernel_size=model_config.imu_block1_kernel_size,    # 設定から取得
            # ... 他の設定値も同様
        )
        # ... 残りの層も設定値を使用
```

#### 3.2 CMISqueezeformerクラス更新
IMUOnlyLSTMのインスタンス化時にmodel_configを渡すよう変更：

```python
class CMISqueezeformer(pl.LightningModule):
    def __init__(self, ...):
        # ...
        self.imu_model = IMUOnlyLSTM(
            model_config=config.model,  # ModelConfigを渡す
            imu_dim=config.model.input_dim,
            # ... 他のパラメータ
        )
```

## 実装手順

### フェーズ1: プロジェクト準備
1. ✅ 実装計画ドキュメント作成
2. exp041をexp042にコピー（test_*.pyは除外）
3. 基本設定の更新（EXP_NUM、実験名など）

### フェーズ2: 設定拡張
4. ModelConfigにIMUOnlyLSTM設定フィールド追加
5. ScheduleFreeConfig.enabledをTrueに変更
6. ExperimentConfig、LoggingConfigのメタデータ更新

### フェーズ3: モデル実装
7. IMUOnlyLSTMクラスの__init__メソッド更新
8. CMISqueezeformerクラスでのModelConfig利用

### フェーズ4: 検証
9. 設定の妥当性確認
10. モデルの動作確認
11. 静的解析・テスト実行

## 期待される効果

### 1. 柔軟性の向上
- モデルアーキテクチャの調整が設定ファイルから可能
- ハイパーパラメータの実験が容易

### 2. 最適化の改善
- ScheduleFree Optimizerによる学習率の自動調整
- 手動スケジューリングからの脱却

### 3. 保守性の向上
- マジックナンバー排除によるコードの可読性向上
- 設定の一元管理

## 技術仕様

### 依存関係
- Pydantic Settings: 設定管理
- PyTorch: モデル実装
- PyTorch Lightning: 訓練フレームワーク
- ScheduleFree Optimizer: 最適化ライブラリ

### 互換性
- exp041との設定互換性維持（デフォルト値使用）
- 既存のデータローダー・損失関数との互換性維持

## 実装完了条件

1. exp042ディレクトリが正常に作成される
2. 設定ファイルが適切に更新される
3. モデルがModelConfigを使用して構築される
4. ScheduleFree Optimizerが有効になる
5. 静的解析（ruff、mypy）が通る
6. 基本動作確認が完了する

---

**実装者**: Claude Code
**作成日**: 2025-08-21
**ベース実験**: exp041
**実験タイプ**: Configuration Enhancement + Schedule-Free Optimization