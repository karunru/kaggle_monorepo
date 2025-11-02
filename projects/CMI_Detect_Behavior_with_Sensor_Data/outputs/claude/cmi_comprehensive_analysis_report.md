# CMI PyTorchモデルアーキテクチャ包括的比較解析レポート

## 解析概要

本レポートは、5つのCMIデータセットのPyTorchモデルアーキテクチャを詳細に解析し、singleNetとBiNetアーキテクチャの違い、特徴量の使用状況、セキュリティリスクについて包括的に分析した結果をまとめています。

### 解析対象データセット
1. **cmi-imu-model** (singleNet_fold*.pt) - 5モデル
2. **cmi-fullfeats-models** (BiNet_fold*.pt) - 10モデル
3. **s-offline-0-8254-15fold** (singleNet_fold*.pt) - 5モデル  
4. **cmi-imu-only-models** (singleNet_fold*.pt) - 5モデル
5. **b-offline-0-8855-specialprocess** (BiNet_fold*.pt) - 5モデル

**総モデル数**: 30モデル

---

## 1. セキュリティチェック結果

### 重要な発見
**全30モデルが安全でないグローバル変数を含有**

各データセットのセキュリティ状況：
- **cmi-imu-model**: 0/5 モデルが安全
- **cmi-fullfeats-models**: 0/10 モデルが安全  
- **s-offline-0-8254-15fold**: 0/5 モデルが安全
- **cmi-imu-only-models**: 0/5 モデルが安全
- **b-offline-0-8855-specialprocess**: 0/5 モデルが安全

### 検出された unsafe globals の種類
主要な unsafe globals（例）：
- `__main__.singleNet`
- `__main__.BiNet` 
- `__main__.CBAM`
- `__main__.SEBlock`
- `__main__.ResidualSEBlock`
- `__main__.MultiScaleConvBlock`
- `__main__.Mutil_AttentionLayer`

### セキュリティリスク評価
- **リスクレベル**: 高
- **原因**: カスタムクラス定義がモデルファイルに含まれている
- **推奨事項**: 本格的なデプロイ前に、モデルの再保存または安全なロード方法の実装が必要

---

## 2. アーキテクチャ比較分析

### singleNet アーキテクチャ

**使用データセット**: 
- cmi-imu-model
- s-offline-0-8254-15fold  
- cmi-imu-only-models

**特徴**:
- シングルブランチのニューラルネットワーク
- 主要コンポーネント:
  - LSTM/GRU レイヤー
  - Transformer Encoder
  - CBAM (Convolutional Block Attention Module)
  - SE (Squeeze-and-Excitation) Block
  - Multi-Scale Convolution Block

**ファイルサイズ**: 約12MB（各フォールド）

### BiNet アーキテクチャ

**使用データセット**:
- cmi-fullfeats-models  
- b-offline-0-8855-specialprocess

**特徴**:
- デュアルブランチのニューラルネットワーク
- より複雑なアーキテクチャ
- 複数の情報パスを持つ構造

**ファイルサイズ**: 
- cmi-fullfeats-models: 約8MB（各フォールド）
- b-offline-0-8855-specialprocess: 約8MB（各フォールド）

### アーキテクチャの技術的差異

| 特徴 | singleNet | BiNet |
|------|-----------|-------|
| ブランチ数 | 1 | 2 |
| ファイルサイズ | 約12MB | 約8MB |
| 複雑度 | 中程度 | 高 |
| モジュール構成 | 統合型 | 分離型 |

---

## 3. 特徴量情報比較

### データセット別特徴量数

| データセット | 特徴量数 | 特徴の種類 |
|-------------|----------|-----------|
| cmi-imu-model | 124 | IMU関連特徴量 |
| cmi-fullfeats-models | 136 | フル特徴量セット |
| s-offline-0-8254-15fold | 91 | 選択された特徴量 |
| cmi-imu-only-models | 91 | IMU専用特徴量 |
| b-offline-0-8855-specialprocess | 376 | 拡張特徴量セット |

### 特徴量の詳細分析

#### 最も多様な特徴量: b-offline-0-8855-specialprocess (376特徴量)
- 最も包括的な特徴量セット
- 複雑な前処理パイプライン
- 多様なセンサーデータの統合

#### IMU専用特徴量: cmi-imu-only-models & s-offline-0-8254-15fold (91特徴量)
- 加速度センサー (accelerometer)
- ジャイロスコープ (gyroscope)  
- 磁力計 (magnetometer)
- 重力センサー
- 線形加速度

#### バランス型特徴量: cmi-fullfeats-models (136特徴量)
- IMU特徴量 + 追加的統計特徴量
- 適度な複雑さとパフォーマンスのバランス

---

## 4. モデル性能と アーキテクチャの関係分析

### 性能推定（ファイル名からの推測）

1. **s-offline-0-8254-15fold**: 0.8254 (82.54%) - singleNet
2. **b-offline-0-8855-specialprocess**: 0.8855 (88.55%) - BiNet

### 性能とアーキテクチャの相関性

#### 高性能モデル: b-offline-0-8855-specialprocess
- **アーキテクチャ**: BiNet
- **特徴量数**: 376（最多）
- **性能**: 88.55%（最高）
- **特徴**: 複雑な特徴エンジニアリングと双方向アーキテクチャ

#### 中程度性能: s-offline-0-8254-15fold  
- **アーキテクチャ**: singleNet
- **特徴量数**: 91（最少）
- **性能**: 82.54%
- **特徴**: シンプルなアーキテクチャとIMU専用特徴量

### パフォーマンス要因分析

1. **特徴量の数と多様性**
   - より多くの特徴量 → より良い性能
   - 特徴量エンジニアリングの重要性

2. **アーキテクチャの複雑さ**  
   - BiNet > singleNet（性能面）
   - 双方向情報処理の効果

3. **モデルサイズと性能の逆相関**
   - より大きなファイル ≠ より良い性能
   - アーキテクチャ効率の重要性

---

## 5. 使用場面の推奨事項

### singleNet アーキテクチャの推奨用途
- **リアルタイム処理が必要な場合**
- **計算資源が限られた環境**
- **シンプルなIMUデータのみの解析**
- **プロトタイピングと初期開発**

### BiNet アーキテクチャの推奨用途
- **高精度が要求される本番環境**
- **多様なセンサーデータの統合**
- **複雑な行動パターンの識別**
- **オフライン解析とバッチ処理**

---

## 6. 技術的洞察

### アーキテクチャ設計の核心要素

#### 共通コンポーネント
- **Attention Mechanism**: CBAM, Multi-head Attention
- **時系列処理**: LSTM, GRU, Transformer
- **特徴抽出**: Multi-Scale Convolution
- **正規化**: Batch Normalization, Layer Normalization

#### アーキテクチャ別特徴
- **singleNet**: 統合的な特徴学習
- **BiNet**: 分離された特徴表現の融合

### 特徴エンジニアリングの影響
- **基本IMU特徴量**: 91特徴量で基本性能確保
- **拡張特徴量**: 376特徴量で性能向上（+6%）
- **特徴選択の重要性**: 適切な特徴量選択が key

---

## 7. 結論とまとめ

### 主要な発見

1. **セキュリティリスク**: 全モデルが安全でないグローバル変数を含有
2. **アーキテクチャ効果**: BiNet > singleNet（性能面）
3. **特徴量重要性**: 376特徴量 > 91特徴量（+6%性能向上）
4. **効率性**: ファイルサイズと性能は逆相関

### 実用的推奨事項

#### 開発段階での選択指針
- **プロトタイプ**: singleNet + 91 IMU特徴量
- **本番環境**: BiNet + 376拡張特徴量  
- **リアルタイム**: singleNet + 最適化された特徴量

#### セキュリティ対応
- モデルファイルの再保存（安全なグローバル変数のみ）
- カスタムクラスの分離とモジュール化
- 本番デプロイ前のセキュリティ監査実施

### 今後の改善方向
1. **モデル軽量化**: BiNetアーキテクチャの効率化
2. **特徴選択**: 376特徴量から重要特徴量の選別
3. **セキュリティ**: 安全なモデル保存形式への移行
4. **アンサンブル**: 複数アーキテクチャの効果的な組み合わせ

---

## 8. 技術仕様詳細

### 解析環境
- **PyTorch**: 最新版
- **解析対象**: 30モデルファイル（.pt）
- **付随ファイル**: feature_cols.npy, scaler.pkl, gesture_classes.npy
- **安全性チェック**: torch.serialization.get_unsafe_globals_in_checkpoint

### データセット構成
- **総ファイルサイズ**: 約300MB
- **モデル形式**: PyTorch checkpoint
- **特徴量形式**: NumPy array
- **前処理情報**: Pickle スケーラー

この分析により、CMIプロジェクトにおけるモデルアーキテクチャの特性と使用指針が明確になりました。