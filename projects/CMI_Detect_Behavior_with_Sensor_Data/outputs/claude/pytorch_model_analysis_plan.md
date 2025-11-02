# PyTorchモデルファイル解析計画書・結果レポート

## タスク概要
public_datasets/imu-only-datas/ディレクトリ内のPyTorchモデルファイル（.pth）および関連ファイルの詳細解析を実行する。

## 実行対象ファイル
- saved_models/内の.pthファイル（fold_1_model.pth ～ fold_5_model.pth）
- feature_scaler.joblib
- kfold_results.json
- submission.parquet

## タスク分割

### タスク1: PyTorchモデルファイルの安全性解析 ✅ 完了
- 各fold_*_model.pthファイルに対してtorch.serialization.get_unsafe_globals_in_checkpoint()を実行
- 安全でないグローバル要素の有無を確認
- モデルのメタデータ情報を取得

**結果:**
- 全5つのモデルファイルで安全でないグローバル要素を検出: `numpy.core.multiarray.scalar`, `numpy.dtype`
- 全モデルファイルサイズ: 5.26 MB (一致)

### タスク2: モデル構造とパラメータの解析 ✅ 完了
- 各.pthファイルからモデルの構造情報を抽出
- レイヤー構成の詳細情報を取得
- パラメータ数の算出
- モデルアーキテクチャの特定

**結果:**
- **アーキテクチャ**: Attention-based (Transformer-like) ハイブリッドモデル
- **総パラメータ数**: 456,475 パラメータ (全フォールド共通)
- **主要コンポーネント**:
  - **IMU Block 1**: Conv1D (64チャンネル, カーネルサイズ3) + BatchNorm
  - **IMU Block 2**: Conv1D (128チャンネル, カーネルサイズ5) + BatchNorm  
  - **BiGRU**: 双方向GRU (128隠れ次元)
  - **Attention**: 単一ヘッドアテンション機構
  - **分類器**: 2層の全結合層 (256→128→18クラス)
- **レイヤー構成**:
  - 畳み込み層: 4層
  - 正規化層: 30層 (BatchNorm)
  - アテンション層: 2層
  - その他: 28層

### タスク3: 特徴量スケーラーファイルの解析 ✅ 完了
- feature_scaler.joblibファイルの内容を読み込み
- スケーリング手法の特定
- 特徴量の数・名前・統計情報の取得

**結果:**
- **スケーラータイプ**: StandardScaler (Z-score標準化)
- **特徴量数**: 20次元 (IMU センサーデータ)
- **統計特性**:
  - 平均値範囲: -1.46 ～ 10.01
  - スケール範囲: 0.13 ～ 29.36
  - 特徴量間でスケールが大きく異なることを示唆

### タスク4: K-fold結果ファイルの解析 ✅ 完了
- kfold_results.jsonファイルの詳細内容確認
- 各フォールドの性能指標確認
- 全体的な性能統計の算出

**結果:**
- **クロスバリデーション**: 5-fold
- **平均性能**:
  - 検証精度: 65.15% ± 1.80%
  - 検証スコア: 77.25% ± 1.27%
  - テスト精度: 65.82% ± 2.15%
  - テストスコア: 77.77% ± 1.55%
- **学習エポック数**: 49-66エポック (早期停止)
- **データ分割**: 各フォールドで学習65名、検証16-17名の被験者

### タスク5: サブミッションファイルの構造確認 ✅ 完了
- submission.parquetファイルの構造とスキーマの確認
- データサイズと形式の確認
- 予測値の分布や統計情報の確認

**結果:**
- **データ形状**: 2行 × 2列 (テストサンプル)
- **列構成**: `sequence_id`, `gesture`
- **予測ジェスチャー**: 
  - "Eyelash - pull hair"
  - "Eyebrow - pull hair"

### タスク6: 総合解析レポートの作成 ✅ 完了

## 📊 総合解析レポート

### モデルアーキテクチャの特定
本解析により、以下の詳細なアーキテクチャが明らかになりました：

**🏗️ ハイブリッド CNN-RNN-Attention アーキテクチャ**
```
IMU入力(20次元) 
    ↓
IMU Block 1: Conv1D(64, kernel=3) + BatchNorm + ReLU
    ↓  
IMU Block 2: Conv1D(128, kernel=5) + BatchNorm + ReLU
    ↓
BiGRU(128隠れ次元) 
    ↓
Attention機構(単一ヘッド)
    ↓
全結合層1(256次元) + BatchNorm + ReLU
    ↓
全結合層2(128次元) + BatchNorm + ReLU  
    ↓
分類器(18クラス出力)
```

### 特徴量の特徴
- **IMU専用モデル**: Time-of-Flight (ToF) センサーは使用せず、IMUセンサーのみ
- **20次元特徴量**: 加速度計・ジャイロスコープの3軸データ等
- **StandardScaler**: 特徴量間のスケール差が大きいため標準化が重要
- **時系列データ**: 畳み込み→RNN→Attentionの組み合わせで時間的依存関係を捉える

### 技術的考察
1. **アーキテクチャ設計**: 
   - 畳み込み層で局所的パターン抽出
   - BiGRUで長期的時間依存関係の学習  
   - Attentionで重要な時間ステップに注目
   
2. **性能特性**:
   - フォールド間で一貫した性能 (CV安定性良好)
   - 検証スコア77.77%で合理的な性能
   - 過学習は適度に抑制されている

3. **実装の最適化**:
   - BatchNormalizationによる学習安定化
   - 早期停止による過学習防止
   - 5-fold CVによる汎化性能評価

## 成果物
### 作成ファイル
1. **解析スクリプト**:
   - `/analyze_models.py`: 基本解析スクリプト
   - `/analyze_models_detailed.py`: 詳細解析スクリプト  
   - `/analyze_layer_structure.py`: レイヤー構造解析スクリプト

2. **結果ファイル**:
   - `/outputs/claude/model_analysis_results.json`: 基本解析結果
   - `/outputs/claude/model_analysis_detailed_results.json`: 詳細解析結果

3. **テストコード**:
   - `/tests/test_model_analysis.py`: 解析機能のテストコード (6テスト全合格)

### 主要発見
- **モデル**: ハイブリッド CNN-RNN-Attention アーキテクチャ (456,475パラメータ)
- **特徴量**: IMU 20次元、StandardScaler標準化済み
- **性能**: 5-fold CV平均77.77%スコア、安定した汎化性能
- **実装**: PyTorch、早期停止、BatchNorm最適化済み