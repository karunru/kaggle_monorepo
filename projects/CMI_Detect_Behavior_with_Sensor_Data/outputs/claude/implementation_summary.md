# CMI Detect Behavior with Sensor Data - 実装完了レポート

## 実装概要

2025年6月28日、CMI - Detect Behavior with Sensor Dataコンペティション用のSqueezeformerベースソリューションの実装が完了しました。

## 実装されたモジュール

### 1. データ前処理 (`src/preprocessor.py`)
- **マルチモーダルセンサー対応**: IMU、Thermopile、Time-of-Flight
- **センサー別正規化**: IMU（Z-score）、Thermopile（Min-Max）、ToF（0-254正規化）
- **特徴量エンジニアリング**: 加速度magnitude、thermopile統計量、ToF空間特徴
- **シーケンス長正規化**: 200タイムステップ（100 transition + 100 gesture）

### 2. データセット (`src/dataset.py`)
- **マルチタスク学習**: Binary分類（Target vs Non-Target）+ 18クラス分類
- **データ拡張**: ガウシアンノイズ、時間軸スケーリング、ToF部分マスキング
- **ロバストネス**: IMU-onlyデータ対応

### 3. モデル (`src/model.py`)
- **Squeezeformerアーキテクチャ**: 8層、8ヘッド、256次元
- **コンボリューションモジュール**: Depthwise separable convolution + GLU
- **デュアルヘッド**: マルチクラス分類 + バイナリ分類
- **IMUブランチ**: テストセットのIMU-onlyデータ対応

### 4. クロスバリデーション (`src/cv_strategy.py`)
- **StratifiedGroupKFold**: 被験者ベースでのグループ分割
- **データリークなし**: 同一被験者が訓練・検証に跨らない
- **Binary層化**: Target vs Non-Target比率を維持

### 5. 評価指標 (`src/metrics.py`)
- **コンペティション指標**: (Binary F1 + Macro F1) / 2
- **詳細メトリクス**: クラス別F1、精度、再現率
- **可視化**: 訓練進捗トラッキング

### 6. 損失関数 (`src/loss.py`)
- **CMILoss**: 基本マルチタスク損失（α重み付き）
- **CMIFocalLoss**: 困難サンプルに対するFocal Loss
- **CMIAdvancedLoss**: ラベルスムージング + 一貫性正則化

### 7. 訓練パイプライン (`exp/exp001/train.py`)
- **5-fold CV**: 全foldでの訓練・評価
- **Early Stopping**: 検証スコアベース
- **Mixed Precision**: メモリ効率化
- **Hydra設定**: 実験管理

### 8. 推論パイプライン (`src/inference.py`)
- **TTA**: Test-Time Augmentation対応
- **アンサンブル**: 複数foldのMajority Voting
- **提出ファイル生成**: Kaggle形式のCSV出力

## 主要設計決定

### データ処理
- **シーケンス長**: 200タイムステップ（resampling）
- **重複処理**: stride=50での滑動窓
- **欠損値処理**: 線形補間（IMU）、前方埋め（其他）

### モデルアーキテクチャ
- **入力次元**: 332（IMU:7 + engineered:5 + Thermopile:5 + ToF:320）
- **Position Encoding**: 学習可能な位置埋め込み
- **Global Pooling**: AdaptiveAvgPool1d

### 訓練設定
- **バッチサイズ**: 32
- **学習率**: 3e-4（Cosineスケジューラ）
- **オプティマイザ**: AdamW（weight_decay=1e-4）
- **エポック数**: 100（Early Stopping付き）

## ファイル構造

```
src/
├── preprocessor.py    # データ前処理
├── dataset.py         # PyTorchデータセット
├── model.py          # Squeezeformerモデル
├── cv_strategy.py    # CV戦略
├── metrics.py        # 評価指標
├── loss.py           # 損失関数
└── inference.py      # 推論パイプライン

exp/exp001/
├── config.yaml       # 実験設定
├── train.py         # 訓練スクリプト
├── inference.py     # Hydra推論スクリプト
└── run_inference.py # 簡易推論スクリプト

tests/
├── test_preprocessor.py
├── test_model.py
└── test_metrics.py
```

## 使用方法

### 訓練実行
```bash
cd exp/exp001
python train.py
```

### 推論実行
```bash
# Hydra版（設定ファイル使用）
python inference.py

# 簡易版
python run_inference.py
```

## 技術的特徴

### 1. マルチモーダル対応
- 3種類のセンサーデータを統合処理
- センサー別の適切な前処理・正規化
- IMU-onlyフォールバック機能

### 2. ロバスト設計
- データ拡張による汎化性能向上
- 複数の損失関数バリエーション
- Test-Time Augmentation

### 3. 実験管理
- Hydraによる設定管理
- 自動fold管理とキャッシュ
- 詳細なメトリクス追跡

## テスト状況

全主要モジュールにユニットテストを実装：
- データ前処理の動作確認
- モデルの入出力形状確認
- 評価指標の計算精度確認

## 次のステップ

1. **実データでの訓練実行**
2. **ハイパーパラメータチューニング**
3. **アンサンブル手法の拡張**
4. **推論速度の最適化**

## まとめ

Squeezeformerアーキテクチャを用いたエンドツーエンドの解法を実装しました。マルチモーダルセンサーデータに対する包括的な前処理、ロバストなモデル設計、適切なクロスバリデーション戦略により、コンペティションの要求に対応可能なソリューションが完成しました。