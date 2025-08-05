# コードベースアーキテクチャ

## 全体構造
```
CMI_Detect_Behavior_with_Sensor_Data/
├── data/                    # Kaggleデータセット
├── outputs/                 # 実験出力・ドキュメント
│   ├── claude/             # Claude作成ドキュメント
│   ├── exp002-exp007/      # 各実験の出力
├── codes/                   # メインコードベース
│   ├── src/                # 共通ライブラリ
│   └── exp/                # 実験別コード
├── tests/                   # テストコード
└── [設定ファイル群]
```

## codes/src/ 共通ライブラリ構造

### データ処理・特徴量
- **features/**: 特徴量エンジニアリング（Polars/Arrow対応）
  - `base.py`: 抽象基底クラス（自動保存/ロード、圧縮）
  - `statistics.py`: 統計的特徴量
  - `category_*.py`: カテゴリカル変数処理
  - `kaplan_meier_target.py`: 生存分析特徴量

### 機械学習
- **models/**: 機械学習モデル（GPU最適化対応）
  - `base.py`: 抽象基底クラス（CV、前処理、事後処理統合）
  - `factory.py`: モデルファクトリ
  - `lightgbm.py`, `xgb.py`, `cat.py`: 勾配ブースティング
  - `tabnet.py`, `gnn.py`, `litnn.py`: ニューラルネット系

### 評価・最適化
- **evaluation/**: 評価・最適化関連
  - `metrics.py`: 競技用評価指標（Concordance Index for EEFS）
  - `optimization.py`: ハイパーパラメータ最適化
  - `cat.py`, `lgbm.py`: モデル別評価ラッパー

### データ分割・サンプリング
- **validation/**: クロスバリデーション戦略
  - `factory.py`: 時系列対応CV（滑動窓、日付ベース）
- **sampling/**: 不均衡データ対応
  - `factory.py`: SMOTE、アンダーサンプリング（GPU対応）

### アンサンブル
- **ensemble/**: アンサンブル手法
  - `blending.py`: 重み付きアンサンブル最適化（ランダムサーチ）

### Kaggle評価システム
- **kaggle_evaluation/**: Kaggle評価用gRPCインターフェース
  - `cmi_gateway.py`: CMIコンペ用ゲートウェイ
  - `cmi_inference_server.py`: CMI推論サーバー
  - `core/`: 評価コア機能とgRPCプロトコル

### ユーティリティ
- **utils/**: 共通ユーティリティ
  - `config.py`: YAML設定管理（階層化設定）
  - `logger.py`: 集約ログ管理
  - `seed_everything.py`: 再現性確保
  - `visualization.py`: データ可視化

## codes/exp/ 実験構造

### 実験進化の流れ
- **exp001**: 初期実験（旧フレームワーク）
- **exp002**: Squeezeformer実装（PyTorch Lightning）
- **exp003**: 損失関数改善
- **exp004**: 最適化手法改善
- **exp005**: 長さグループ化
- **exp006**: Switch EMA実装
- **exp007**: 欠損値attention mask処理（最新）

### 各実験の標準構造
```
exp{番号}/
├── config.py          # Pydantic設定
├── model.py           # モデル定義
├── dataset.py         # データセット・データローダー
├── train.py           # 訓練スクリプト
├── inference.py       # 推論スクリプト
└── test_exp{番号}.py  # テストコード
```

## 設計パターン

### 設定管理
- **Pydantic Settings**: 型安全な設定管理
- **階層化設定**: 実験・モデル・データ・パスの分離

### 抽象化パターン
- **Factory Pattern**: モデル・検証・サンプリング戦略
- **Strategy Pattern**: 特徴量エンジニアリング
- **Template Method**: 機械学習パイプライン

### GPU最適化
- **CuPy統合**: NumPy互換GPU加速
- **PyTorch Lightning**: 分散訓練対応
- **混合精度**: メモリ効率化

## データフロー
1. **データ取得**: Kaggle API → data/
2. **前処理**: features/ → 特徴量生成
3. **分割**: validation/ → CV戦略
4. **訓練**: models/ → GPU訓練
5. **評価**: evaluation/ → 競技指標
6. **出力**: outputs/ → 結果保存
7. **提出**: Kaggle API → サブミッション