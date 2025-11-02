# exp050実装レポート

## 実装概要
exp049からexp050への移行により、ResidualSECNNBlockをResidualSEConvNeXtBlockに置き換えました。ConvNeXtアーキテクチャの導入により、より効率的で表現力の高い特徴抽出が可能になります。

## 実装内容

### 1. ファイル構成
```
codes/exp/exp050/
├── __init__.py
├── config.py          # 設定ファイル（EXP_NUM、description、tags更新）
├── dataset.py          # データセットクラス
├── human_normalization.py  # 人間正規化
├── inference.py        # 推論スクリプト
├── losses.py          # 損失関数
├── model.py           # モデル定義（ConvNeXt統合）
└── train.py           # 訓練スクリプト
```

### 2. 主要変更点

#### 2.1 model.py への新規クラス追加
- `stochastic_depth`: DropPath実装関数
- `LayerNorm1d`: 1D向けLayerNormクラス
- `ConvNeXt1DBlock`: ConvNeXtの基本ブロック
- `ResidualSEConvNeXtBlock`: ResidualSECNNBlockの置き換えクラス

#### 2.2 モデル統合
`IMUOnlyLSTM`クラス内で以下の変更を実施：
```python
# 変更前
self.imu_block1 = ResidualSECNNBlock(...)
self.imu_block2 = ResidualSECNNBlock(...)

# 変更後
self.imu_block1 = ResidualSEConvNeXtBlock(...)
self.imu_block2 = ResidualSEConvNeXtBlock(...)
```

#### 2.3 設定ファイル更新
```python
# EXP_NUM
EXP_NUM = "exp050"

# ExperimentConfig
description = "ConvNeXt1D block integration with SE attention"
tags = [..., "residual_se_convnext", "convnext"]

# LoggingConfig
wandb_tags = [..., "residual_se_convnext", "convnext"]
```

### 3. ConvNeXtブロックの特徴

#### 3.1 ResidualSEConvNeXtBlockの構成
- **入力射影**: チャンネル数変更時の1x1畳み込み
- **ConvNeXt1Dブロック**: 複数のConvNeXtブロック（num_blocks=2）
- **SEブロック**: Squeeze-and-Excitation注意機構
- **ショートカット接続**: 残差接続
- **プーリング・ドロップアウト**: MaxPool1D + Dropout

#### 3.2 ConvNeXt1DBlockの構成
- **Depthwise Conv1d**: グループ畳み込み（大きなカーネル）
- **LayerNorm**: チャンネル方向の正規化
- **PointwiseMLP**: 1x1畳み込みによるMLPブロック
- **LayerScale**: チャンネル別スケーリング係数
- **StochasticDepth**: 確率的な深度（ドロップパス）

### 4. 技術的改善点

#### 4.1 効率性向上
- **Depthwise Convolution**: 計算量削減とパラメータ効率化
- **Large Kernel**: 大きな受容野による特徴抽出能力向上
- **Progressive DropPath**: ブロック毎に調整される確率的ドロップアウト

#### 4.2 学習安定性
- **LayerScale**: 初期化時の微小な重み係数による安定した学習
- **LayerNorm**: バッチサイズに依存しない正規化
- **Residual Connection**: 勾配消失問題の軽減

### 5. 実装検証

#### 5.1 インポート検証
```bash
# モデルインポート
✅ from model import IMUOnlyLSTM

# 設定インポート  
✅ from config import Config, EXP_NUM
```

#### 5.2 静的解析結果
- 新規追加コードに問題なし
- 既存コードのlintエラーは維持（影響なし）

### 6. 期待される効果

#### 6.1 性能向上
- より効率的な特徴抽出による精度向上
- 大きな受容野による時系列パターン認識能力強化
- SE注意機構とConvNeXtの組み合わせによるチャンネル間相関学習

#### 6.2 訓練効率
- LayerScaleとStochasticDepthによる安定した学習
- Depthwise Convolutionによる計算効率化
- 勾配フローの改善

## 次のステップ
1. **訓練実行**: `cd codes/exp/exp050 && uv run python train.py`
2. **性能評価**: exp049との比較実験
3. **パラメータ調整**: drop_path、layer_scale_init等のハイパーパラメータ最適化
4. **アブレーション研究**: ConvNeXtブロック数の影響調査

## 関連ファイル
- 実装計画: `outputs/claude/exp050_implementation_plan.md`（このファイル）
- ソースコード: `codes/exp/exp050/`
- 元実験: `codes/exp/exp049/`
- 参考実装: `codes/exp/exp043/model.py`