# exp033実装完了報告

## 実装概要
exp032のFNOアーキテクチャとexp031のResidualSECNNアーキテクチャを並列処理で結合し、異なるスケールの時系列特徴を同時に学習する実装を完了しました。

## 実装内容

### 1. ファイル構成
```
codes/exp/exp033/
├── __init__.py
├── config.py           # ✅ 更新済み (exp033設定、並列処理タグ追加)
├── dataset.py          # exp032からコピー
├── human_normalization.py # exp032からコピー
├── inference.py        # exp032からコピー
├── losses.py           # exp032からコピー
├── model.py            # ✅ 並列処理実装追加
└── train.py            # exp032からコピー
```

### 2. 主要な変更点

#### config.py更新項目
- `EXP_NUM`: "exp032" → "exp033"
- `ExperimentConfig.name`: "exp033_residual_fno_parallel"
- `ExperimentConfig.description`: "Parallel ResidualSECNN and FNO-1D blocks with concatenation for multi-scale temporal feature extraction"
- `ExperimentConfig.tags`: ["residual_se_cnn", "fno", "fourier_neural_operator", "parallel_fusion", "multi_scale"]追加
- `LoggingConfig.wandb_tags`: 同様に更新

#### model.py新規実装

##### IMUOnlyLSTMクラスの並列アーキテクチャ
```python
# ResidualSECNN ブランチ（exp031ベース）
self.residual_branch = nn.Sequential(
    ResidualSECNNBlock(imu_dim, 64, 3, dropout=0.3),
    ResidualSECNNBlock(64, 128, 5, dropout=0.3)
)

# FNO ブランチ（exp032ベース）
self.fno_proj = nn.Sequential(
    nn.Conv1d(imu_dim, 128, 1),
    nn.BatchNorm1d(128),
    nn.Mish()
)
self.fno_branch = nn.Sequential(
    FNOBlock1D(128, modes=32, dropout=0.2) × 4,
    SEBlock(128),
    nn.MaxPool1d(2),
    nn.Dropout(0.3)
)

# 特徴量融合層（128+128=256次元）
self.fusion_conv = nn.Sequential(
    nn.Conv1d(256, 256, 1),
    nn.BatchNorm1d(256),
    nn.Mish(),
    nn.Dropout(0.2)
)

# BiGRU（入力次元を256に変更）
self.bigru = nn.GRU(256, 128, bidirectional=True)
```

##### forwardメソッドの並列処理
```python
# 並列ブランチ処理
residual_features = self.residual_branch(imu)  # [batch, 128, seq_len//4]
fno_input = self.fno_proj(imu)
fno_features = self.fno_branch(fno_input)      # [batch, 128, seq_len//2]

# 系列長調整
target_length = residual_features.size(-1)
if fno_features.size(-1) != target_length:
    fno_features = F.adaptive_avg_pool1d(fno_features, target_length)

# 特徴量結合と融合
concatenated = torch.cat([residual_features, fno_features], dim=1)  # [batch, 256, target_length]
merged_features = self.fusion_conv(concatenated)
```

### 3. アーキテクチャの技術的特徴

#### 並列処理の利点
- **ResidualSECNN**: 局所的な時系列パターンを効率的に捕捉（CNN + SE Attention）
- **FNO**: FFTベースでグローバルな長距離依存関係を学習（O(N log N)計算）
- **並列fusion**: 異なるスケールの特徴を同時に活用

#### パラメータ効率性
- 総パラメータ数: 4,910,717（学習可能）
- exp032（4,591,613）比で約7%増加
- 2つのブランチで多様な特徴抽出能力を獲得

#### 特徴量統合戦略
1. **適応的長さ調整**: `F.adaptive_avg_pool1d`で系列長を統一
2. **チャネル結合**: 128+128=256次元でconcat
3. **融合変換**: 1×1 Convで特徴量を統合
4. **下流統合**: BiGRU（256次元入力）で時系列モデリング

### 4. 実装検証結果

#### ✅ 基本動作確認
- SpectralConv1d、FNOBlock1D、ResidualSECNNBlock、IMUOnlyLSTMの正常なインスタンス化
- Config設定の正常な読み込み

#### ✅ フォワードパス確認
- 入力: [8, 100, 20] (batch_size, seq_len, imu_dim)
- 出力: 
  - multiclass_logits: [8, 18]
  - binary_logits: [8, 1] 
  - nine_class_logits: [8, 9]
- パラメータ数: 4,910,717 (学習可能)

#### ✅ インポート確認
- train.pyの正常なインポート
- 依存関係の問題なし

### 5. 期待される効果

#### 表現学習の向上
1. **マルチスケール特徴**: 局所パターンとグローバルパターンの同時学習
2. **相補的な特徴抽出**: CNNの詳細性とFNOの効率性を併用
3. **robust特徴表現**: 異なるアーキテクチャによるアンサンブル効果

#### 性能向上の理論的根拠
- **局所-グローバル統合**: 短期的なジェスチャー特徴と長期的な運動パターンの同時モデリング
- **計算効率**: FNOのO(N log N)とCNNの局所計算の最適な組み合わせ
- **汎化能力**: 複数の特徴抽出戦略によるロバスト性向上

### 6. 実行方法

実装が完了し、基本動作確認も完了しています。以下のコマンドで実際の訓練を開始できます：

```bash
cd codes/exp/exp033
uv run python train.py
```

### 7. 次のステップ

1. **ハイパーパラメータ調整**: モード数、ドロップアウト率の最適化
2. **融合戦略の改善**: Attention機構やGated融合の検討
3. **実験結果分析**: exp031、exp032との性能比較
4. **アブレーション研究**: 各ブランチの貢献度分析

実装は完了し、exp033として独立した並列処理実験環境が構築されました。