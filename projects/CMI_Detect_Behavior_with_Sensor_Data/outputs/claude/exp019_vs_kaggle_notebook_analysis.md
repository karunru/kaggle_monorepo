# exp019 vs Kaggle Notebook (lb0-79-cmi-imu-only) 差分調査

## 概要

本ドキュメントは、codes/exp/exp019の実装とKaggleノートブック「lb0-79-cmi-imu-only」の差分を詳細に調査した結果をまとめています。

調査対象：
- **exp019**: ローカル実装（Squeezeformer + BERT アーキテクチャ）
- **Kaggle Notebook**: https://www.kaggle.com/code/beijijun/lb0-79-cmi-imu-only (LB 0.79スコア)

## 1. モデルアーキテクチャ

### exp019: Squeezeformer + BERT
```python
# 全体構成
Input → Squeezeformer Blocks → BERT → Classification Heads
       → Demographics Embedding ↗
```

**SqueezeformerBlock構成:**
1. FeedForward Module 1
2. Multi-Head Self-Attention (8 heads)  
3. Convolution Module
4. FeedForward Module 2

**ConvolutionModule詳細:**
- LayerNorm → Pointwise Conv1d (d_model→2*d_model) → GLU → Depthwise Conv1d (kernel=31) → BatchNorm1d → SiLU → Pointwise Conv1d (d_model) → Dropout

**BERT統合:**
- CLS token追加
- Demographics embedding結合
- 二重分類ヘッド（マルチクラス + バイナリ）

**モデル次元:**
- d_model: 256
- n_layers: 8  
- n_heads: 8
- d_ff: 1024

### Kaggle Notebook: CNN + BiGRU + Attention
```python  
# 全体構成
IMU Input → ResidualCNN → BiGRU → MLP Attention → Dense Layers → Classification
```

**ResidualCNNBlock構成:**
1. Conv1d (in→out channels) + BatchNorm1d + ReLU
2. Conv1d (out→out channels) + BatchNorm1d  
3. **CoordinateAttention** (key innovation)
4. Shortcut connection + ReLU
5. MaxPool1d + Dropout

**BiGRU + Attention:**
- Bidirectional GRU (hidden_dim=128)
- **MLP Attention** for context aggregation

**モデル構成:**
- CNN: 20→64→128 channels
- BiGRU: 128 hidden units (bidirectional → 256)
- Dense: 256→128→18 classes

## 2. 特徴量エンジニアリング

### 共通の物理ベース特徴量

両手法とも類似の物理計算を実装：

**基本IMU特徴量 (7次元):**
- 加速度: acc_x, acc_y, acc_z
- 四元数: rot_w, rot_x, rot_y, rot_z

**物理派生特徴量:**
- 加速度モーション: `sqrt(acc_x² + acc_y² + acc_z²)`
- 回転角度: `2 * arccos(rot_w)`
- 重力除去後の線形加速度 (3次元)
- 四元数からの角速度 (3次元)  
- 角距離計算
- Jerk計算（微分ベース）

### 実装の違い

**exp019:**
- **Polars**ベースの効率的な実装
- vectorized計算でシーケンス境界を尊重
- より数値安定的な計算（tolerance設定）

**Kaggle Notebook:**
- **Pandas/NumPy**ベースの実装
- SciPy.Rotation使用で実装が簡潔
- 行単位でのループ計算（やや非効率）

### 最終特徴量次元

**exp019:** 16次元（基本IMU 7 + 物理派生 9）
**Kaggle:** 20次元（基本IMU 7 + 物理派生 13）

## 3. ハイパーパラメータ

| 項目 | exp019 | Kaggle Notebook |
|------|---------|-----------------|
| **バッチサイズ** | 128 | 32 |
| **学習率** | 3e-4 | 1e-3 |
| **エポック数** | 100 | 100 |
| **早期停止** | 15 epochs | 25 epochs |
| **最大系列長** | 設定可能 | 256 |
| **シード** | 42 | 2 |
| **ワーカー数** | 4 | 4 |

### 学習率スケジューラ

**exp019:**
- Cosine Annealing LR / ReduceLROnPlateau
- 設定可能なmin_lr (1e-6)
- Schedule Free optimizerオプション

**Kaggle Notebook:**
- **手動実装**: Linear Warmup (3 epochs) + Cosine Annealing
- warmup後にコサイン減衰（eta_min=1e-6）
- バッチレベルでの細かい制御

## 4. データ前処理・拡張

### exp019: 高度なデータ処理

**前処理パイプライン:**
- Polarsベースの高速処理
- 動的パディング対応
- Length-grouped sampling（系列長によるグループ化）

**データ拡張:**
- HandednessAugmentation（左右反転）
- 設定可能な拡張パラメータ

**正規化:**
- 分離されたスケーラー設計
- Demographics統合対応

### Kaggle Notebook: 多様な拡張手法

**正規化:**
- **ToFScaler**: ToF特徴量専用（-1→10, 0→-10 マッピング）
- StandardScaler: その他特徴量用

**データ拡張（訓練時のみ）:**
1. **Jitter**: ガウシアンノイズ (σ=0.05, 70%確率)
2. **Time Masking**: 時間軸マスキング (最大20frames, 50%確率)
3. **Feature Masking**: 特徴量軸マスキング (最大15features, 50%確率)  
4. **Motion Drift**: IMU drift simulation (50%確率)

## 5. 損失関数・最適化

### exp019: 多様な損失関数対応

**サポート損失関数:**
- CMI (CrossEntropy)
- CMI Focal (Focal Loss) 
- Soft F1 Loss
- **ACLS** (Adaptive and Conditional Label Smoothing)
- Label Smoothing CrossEntropy
- **MBLS** (Margin-based Label Smoothing)

**最適化:**
- AdamW (デフォルト)
- **Schedule Free optimizers** (RAdamScheduleFree/AdamWScheduleFree/SGDScheduleFree)
- **EMA** (Exponential Moving Average) サポート
- Switch EMA機能

**予測ヘッド:**
- Multiclass head (18クラス)  
- Binary head (BFRB vs Non-BFRB)
- 統合損失: α * multiclass + (1-α) * binary

### Kaggle Notebook: シンプルな設計

**損失関数:**
- CrossEntropy Loss (multiclass only)
- **Label Smoothing** (α=0.1) with KL Divergence
- 単一の18クラス分類のみ

**最適化:**
- Adam optimizer (weight_decay=1e-4)  
- 手動実装の学習率スケジューリング
- Gradient clipping (max_norm=1.0)

## 6. 推論・予測戦略

### exp019: エンドツーエンド統合
- PyTorch Lightning framework
- Multi-fold ensemble対応
- Demographics統合推論
- CMI score計算内蔵

### Kaggle Notebook: シンプルな ensemble
- 5-fold model ensemble
- Softmax probability averaging
- Padding対応 (40 frames)
- シンプルな予測パイプライン

## 7. 計算効率・スケーラビリティ

### exp019
**利点:**
- Polarsによる高速データ処理
- Mixed precision training
- 効率的なメモリ管理
- 設定駆動アーキテクチャ

**課題:**
- 複雑な設定構造
- 重いTransformer計算
- BERT統合のオーバーヘッド

### Kaggle Notebook  
**利点:**
- 軽量なCNN+RNN構造
- 実装が理解しやすい
- 効果的なAttention機構

**課題:**
- Pandas処理のボトルネック
- 手動実装の保守性
- スケーラビリティの制限

## 8. 主要な技術的差異

| 技術要素 | exp019 | Kaggle Notebook |
|----------|--------|-----------------|
| **コア技術** | Transformer + BERT | CNN + RNN |
| **Attention** | Multi-Head Self-Attention | Coordinate + MLP Attention |
| **特徴量処理** | Polars vectorized | NumPy/Pandas looped |
| **正規化** | 統一StandardScaler | 分離型スケーラー |
| **損失戦略** | Multi-head prediction | Single-head prediction |
| **フレームワーク** | PyTorch Lightning | Pure PyTorch |
| **設定管理** | Pydantic BaseSettings | ハードコード変数 |

## 9. パフォーマンス比較

**Kaggle Notebook実績:**
- LB Score: 0.79
- 平均CV F1: 0.7858
- Fold別スコア: 0.7734, 0.8004, 0.7856, 0.7701, 0.7994

**exp019の特徴:**
- より現代的なTransformer architecture
- 多様な損失関数とEMAサポート
- Demographics統合能力
- より拡張性の高い設計

## 10. 実装推奨事項

### Kaggle Notebookから学べる点:
1. **CoordinateAttention**: 効果的なsequence attention
2. **手動学習率スケジューリング**: 細かい制御が可能
3. **多様なデータ拡張**: Jitter, Masking, Drift simulation
4. **ToF特徴量の特別扱い**: 専用スケーラーで性能向上

### exp019の改善に活かせる技術:
1. CoordinateAttentionをSqueezeformerに統合
2. Motion driftシミュレーション追加
3. ToF特徴量の専用処理
4. より積極的なLabel smoothing

## 結論

- **Kaggle Notebook**: シンプルで実用的、効果的なAttention機構
- **exp019**: 最新技術統合、高い拡張性、複雑な設計

両手法ともIMU-onlyで高いスコアを達成しており、異なるアプローチの有効性を示している。exp019はより研究指向で拡張性が高く、Kaggleノートブックは実用性と理解しやすさに優れている。