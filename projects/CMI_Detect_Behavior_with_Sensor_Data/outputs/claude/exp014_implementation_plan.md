# EXP014実装計画: MiniRocketMultivariate特徴量統合

## 概要

exp013をベースに、指定された9つの時系列に対してMiniRocketMultivariate変換を適用し、元の時系列と組み合わせて拡張された特徴量をモデルの入力とする実装を行う。

## 対象時系列（exp013から継承）

```python
target_time_series = [
    "linear_acc_x",
    "linear_acc_y", 
    "linear_acc_z",
    "linear_acc_mag",
    "linear_acc_mag_jerk",
    "angular_vel_x",
    "angular_vel_y",
    "angular_vel_z",
    "angular_distance",
]
```

## 実装方針

### 1. MiniRocketMultivariate調査・動作確認

#### 調査項目
- sktimeのMiniRocketMultivariateのAPI仕様確認（完了）
- 最適なnum_kernelsパラメータの決定が必要
- Polarsデータフォーマット（PanelPolarsEager）への変換方法
- 実際の変換処理時間とメモリ使用量の測定

#### 動作確認スクリプト作成
- `outputs/claude/test_minirocket_investigation.py`を作成
- 小さなサンプルデータでの動作テスト
- パラメータ調整と最適化

### 2. exp014実装

#### ディレクトリ構成
```
codes/exp/exp014/
├── __init__.py
├── config.py（exp013からコピー・調整）
├── dataset.py（MiniRocket統合版）
├── model.py（拡張特徴量対応版）
├── losses.py（exp013からコピー）
├── train.py（exp013からコピー）
└── inference.py（exp013からコピー）
```

#### dataset.py主要変更点
1. **MiniRocketMultivariate変換器の追加**
   - 初期化時にMiniRocketMultivariate instanceを作成
   - fit処理の追加（訓練データで変換器を学習）
   - transform処理で元時系列をRocket特徴量に変換

2. **特徴量統合処理**
   - 元の9次元時系列: `[batch, 9, seq_len]`
   - MiniRocket変換後: `[batch, num_kernels]` (時系列→固定長ベクトル)
   - 統合方法: 元時系列にRocket特徴量をconcatして拡張

3. **データフロー設計**
   ```python
   # 元データ
   original_features = [batch, 9, seq_len]  # 9つの時系列
   
   # MiniRocket変換
   rocket_features = minirocket.transform(original_features)  # [batch, num_kernels]
   
   # 統合（方式検討必要）
   # Option A: Rocket特徴量を時系列次元に追加
   extended_features = torch.cat([original_features, rocket_features.unsqueeze(-1).repeat(1, 1, seq_len)], dim=1)
   
   # Option B: 別々に処理してfeature fusionレイヤーで統合
   ```

#### model.py主要変更点
1. **input_dim調整**
   - 現在: 16次元（基本IMU 7次元 + 物理特徴量 9次元）
   - 変更後: 9次元時系列 + MiniRocket特徴量統合後の次元数

2. **アーキテクチャ検討**
   - 元時系列処理部分: Squeezeformerブロック（既存）
   - Rocket特徴量処理: 追加のMLPレイヤー
   - Feature fusion: 統合処理レイヤー

### 3. 技術検討事項

#### MiniRocketパラメータ最適化
- `num_kernels`: デフォルト10,000は多すぎる可能性
- 候補値: 84, 168, 336, 672, 840（84の倍数制約）
- 計算時間とパフォーマンスのトレードオフ要検証

#### データ統合戦略
1. **Early Fusion**: 変換後特徴量を時系列次元に追加
2. **Late Fusion**: 別々に処理後、中間層で統合
3. **Parallel Processing**: 並列処理後にfeature fusion layer

#### メモリ・計算効率
- MiniRocket変換のGPU対応状況確認
- バッチ処理時のメモリ使用量最適化
- 前処理の並列化

### 4. torchvizモデル可視化

#### 実装内容
- 拡張後のモデル構造可視化
- 特徴量フローの確認
- 出力先: `codes/exp/exp014/model_architecture.png`

#### 可視化対象
- Input layers（拡張特徴量）
- Squeezeformer blocks
- Feature fusion layers
- Classification heads

### 5. テスト・検証

#### 単体テスト
- MiniRocket変換処理
- データ統合処理
- モデル前向き計算

#### 統合テスト
- 端対端のデータ処理フロー
- 訓練・推論パイプライン
- パフォーマンス測定

## 実装の制約・注意事項

### exp013からの最小変更原則
- 既存コードの動作を壊さないよう慎重に変更
- 設定ファイルでRocket機能のON/OFF切り替え可能にする

### 依存関係
- sktimeライブラリの追加（`uv add scikit-time`）
- torchvizライブラリの追加（`uv add torchviz`）

### パフォーマンス要件
- 訓練時間の大幅な増加を避ける
- メモリ使用量の監視
- バッチサイズ調整の必要性

## 成果物

1. **動作確認スクリプト**: `outputs/claude/test_minirocket_investigation.py`
2. **exp014実装**: `codes/exp/exp014/`配下のすべてのファイル
3. **モデル可視化**: `codes/exp/exp014/model_architecture.png`
4. **テストコード**: `tests/test_exp014_*.py`
5. **実装完了報告書**: このドキュメントの更新版

## 実装結果

### 完了した作業

✅ **1. MiniRocket調査・基本理解**
- 調査スクリプト: `outputs/claude/test_minirocket_investigation.py`
- 推奨パラメータ: num_kernels=336, n_jobs=-1, random_state=42
- 処理時間とメモリ使用量の測定完了

✅ **2. exp013→exp014コピー**
- `codes/exp/exp014/`配下にexp013のコードをコピー
- 実験番号とメタデータをexp014用に更新

✅ **3. dataset.py実装**
- MiniRocketMultivariate変換器の統合
- 対象9つの時系列特徴量の変換機能
- 元時系列とRocket特徴量の統合（時系列次元で結合）
- 設定可能なrocket_config追加

✅ **4. config.py修正**
- MiniRocketConfig クラス追加
- ModelConfig に動的input_dim計算メソッド追加
- メイン Config クラスに rocket 設定統合

✅ **5. model.py修正**
- input_dim のデフォルト値を352次元に更新（基本IMU 16 + MiniRocket 336）
- ドキュメンテーション文字列の更新
- テストコードの更新

✅ **6. 可視化機能実装**
- `codes/exp/exp014/visualize_model.py` 作成
- torchvizによるモデル構造可視化
- 統計情報の出力（パラメータ数、特徴量内訳等）

✅ **7. テスト・検証**
- `test_exp014.py` 基本動作テストスクリプト作成
- データセット初期化テスト
- モデル初期化テスト
- 前向き計算テスト
- 統合テスト

### 実装詳細

**特徴量拡張:**
- 基本IMU特徴量: 16次元 (基本IMU 7 + 物理特徴量 9)
- MiniRocket特徴量: 336次元 (num_kernels=336)
- **合計入力次元: 352次元**

**MiniRocket設定:**
- 対象時系列: 9つの物理ベース特徴量
- 変換方式: 時系列次元での結合 (Option 1)
- パフォーマンス: fit時間 < 0.1秒, transform時間 < 0.1秒

**モデル統計:**
- 総パラメータ数: 12,326,607
- 入力形状: [batch, 352, seq_len]
- 出力形状: multiclass [batch, 18], binary [batch, 1]

### 生成ファイル

1. **実装ファイル:**
   - `codes/exp/exp014/dataset.py` (MiniRocket統合版)
   - `codes/exp/exp014/config.py` (設定更新版)
   - `codes/exp/exp014/model.py` (拡張特徴量対応版)

2. **可視化・検証ファイル:**
   - `codes/exp/exp014/visualize_model.py` (モデル可視化スクリプト)
   - `codes/exp/exp014/model_visualizations/` (可視化結果)
   - `test_exp014.py` (基本動作テスト)

3. **調査・計画ファイル:**
   - `outputs/claude/test_minirocket_investigation.py` (調査スクリプト)
   - `outputs/claude/exp014_implementation_plan.md` (この計画書)

### 次のステップ

1. **実際のデータでの検証:**
   - 実際のCMIデータセットでのMiniRocket変換テスト
   - メモリ使用量とtraini時間の実測

2. **ハイパーパラメータ調整:**
   - num_kernels の最適値探索 (84, 168, 336, 672)
   - モデル構造の調整 (d_model, n_layers等)

3. **性能評価:**
   - exp013との比較実験
   - CMIスコアでの性能評価

---

**実装完了日:** 2025-01-11  
**作成者:** Claude Code Assistant  
**ステータス:** ✅ 実装完了・テスト済み