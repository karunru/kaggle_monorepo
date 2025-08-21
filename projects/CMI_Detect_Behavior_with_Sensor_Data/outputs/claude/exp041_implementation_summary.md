# exp041: ConvNeXt1D Implementation Summary

## 概要
exp040のCNNアーキテクチャをConvNeXt1Dに置き換えたIMU手振り認識モデルの実装

## 実装内容

### 1. アーキテクチャの変更
- **変更前**: ResidualSECNNBlock (Conv1d + BatchNorm + SE + Residual)
- **変更後**: ResidualSEConvNeXtBlock (ConvNeXt1D + SE + Residual)

### 2. 新規追加クラス

#### LayerNorm1d
- 1D ConvLayersでのLayerNorm対応
- Channel-last形式でのLayerNorm適用

#### ConvNeXt1DBlock  
- Depthwise Conv1d (groups=channels, large kernel)
- LayerNorm (channels-last)
- Pointwise MLP: Conv1d(C → 4C) + GELU + Conv1d(4C → C)  
- LayerScale (γパラメータによる学習可能スケーリング)
- StochasticDepth (DropPath) による正則化

#### ResidualSEConvNeXtBlock
- 複数のConvNeXt1Dブロックを積み重ね
- SEBlock（Squeeze-and-Excitation）を維持
- Residual接続とMaxPooling
- Progressive drop pathによる正則化強度調整

### 3. モデル構造
```python
IMUOnlyLSTM(
  # ConvNeXt特徴抽出部
  imu_block1: ResidualSEConvNeXtBlock(20→64, kernel=3, blocks=2, drop_path=0.1)
  imu_block2: ResidualSEConvNeXtBlock(64→128, kernel=5, blocks=2, drop_path=0.15)
  
  # 系列モデリング部（変更なし）
  bigru: Bidirectional GRU(128→256)
  attention: AttentionLayer(256)
  
  # 分類ヘッド部（変更なし）
  multiclass_head: Linear(128+demo→18classes)
  binary_head: Linear(128+demo→1) 
  nine_class_head: Linear(128+demo→9)
  orientation_head: Linear(128+demo→4)
)
```

### 4. 主要な改善点

#### ConvNeXtの利点
- **より効率的な畳み込み**: Depthwise + Pointwise分離で計算効率向上
- **LayerNorm**: BatchNormより安定した学習
- **LayerScale**: 深いモデルでの勾配問題を緩和
- **StochasticDepth**: 正則化効果による過学習抑制

#### SE Attentionの維持
- Channel attention機構を保持
- ConvNeXt + SE の組み合わせで表現力向上

### 5. 設定変更
```python
# config.py
EXP_NUM = "exp041"
description = "ConvNeXt1D with SE attention for IMU gesture recognition"
tags = [..., "residual_se_convnext", "convnext", ...]
wandb_tags = [..., "residual_se_convnext", "convnext", ...]
```

### 6. 技術仕様

#### DropPath Scheduling
- block1: drop_path=0.1
- block2: drop_path=0.15  
- Progressive scheduling: drop_path * (i+1) / num_blocks

#### LayerScale初期化
- layer_scale_init=1e-6 (ConvNeXtデフォルト値)

#### Fallback実装
- torchvision.ops.stochastic_depthのフォールバック実装を追加
- 環境依存性を軽減

## 期待される効果

1. **精度向上**: ConvNeXtの最新アーキテクチャによる表現力向上
2. **安定学習**: LayerNormとLayerScaleによる学習安定性
3. **正則化**: StochasticDepthによる汎化性能向上  
4. **計算効率**: Depthwise畳み込みによる効率化

## 実装状況
- ✅ exp040→exp041コピー完了
- ✅ config.py更新完了
- ✅ ConvNeXt1D関連クラス追加完了
- ✅ ResidualSEConvNeXtBlock実装完了
- ✅ IMUOnlyLSTM更新完了
- ✅ 不要クラス削除完了
- ✅ 動作確認完了

## 注意事項
- SEBlockは従来通り維持
- BiGRU以降の構造は変更なし  
- Demographics統合機能も維持
- 損失関数とスコア融合ロジックは変更なし