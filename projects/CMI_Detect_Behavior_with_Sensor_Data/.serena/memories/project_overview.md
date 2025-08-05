# CMI Detect Behavior with Sensor Data - プロジェクト概要

## プロジェクト目的
このプロジェクトは、CMI（Child Mind Institute）のKaggleコンペティション「Detect Behavior with Sensor Data」への参加を目的としています。センサーデータを用いた行動検出がメインタスクです。

## 競技の詳細
- コンペティション：Child Mind Institute - Detect Behavior with Sensor Data
- データ：IMUセンサーデータ（加速度・回転センサー）
- タスク：センサーデータからの行動パターン検出・分類

## 技術スタック
- **言語**: Python 3.12
- **パッケージマネージャー**: uv (0.6.9)
- **深層学習フレームワーク**: PyTorch, PyTorch Lightning
- **データ処理**: Polars（高速データフレーム処理）
- **モデル**: Squeezeformer（音響モデルベース）
- **設定管理**: Pydantic Settings, Hydra, OmegaConf
- **実験管理**: 各実験ディレクトリ（exp001-exp007）
- **評価**: カスタム評価指標（Concordance Index for EEFS）

## プロジェクト構造の特徴
1. **実験ベース設計**: exp001-exp007で段階的な改善を実施
2. **共通コードベース**: src/配下に再利用可能なモジュール
3. **Kaggle統合**: 自動的なデータセット・コードアップロード機能
4. **GPU最適化**: CUDA 12.x対応、CuPy使用

## 現在の実験状況
- exp002-007: Squeezeformerベースの実装
- exp007: 欠損値処理にAttention Mask適用（最新）
- 各実験でモデル改善を段階的に実施