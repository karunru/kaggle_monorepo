# exp030実装報告書 - SoftF1ACLS損失関数の実装

## 概要

exp030では、exp029を基にSoftF1LossにACLS（Adaptive and Conditional Label Smoothing）を適用した新しい損失関数「SoftF1ACLS」を実装しました。

## 実装内容

### 1. プロジェクト構成

```
codes/exp/exp030/
├── __init__.py
├── config.py           # 設定ファイル（exp030用に更新）
├── dataset.py          # データセット処理（exp029から継承）
├── human_normalization.py  # ヒューマンノーマライゼーション（exp029から継承）
├── inference.py        # 推論処理（exp029から継承）
├── losses.py           # 損失関数（新規SoftF1ACLSを追加）
├── model.py            # モデル実装（SoftF1ACLS対応を追加）
├── submission.parquet  # サブミッションファイル（exp029から継承）
└── train.py            # 訓練スクリプト（exp029から継承）

tests/
└── test_exp030_soft_f1_acls.py  # SoftF1ACLSテストスクリプト（新規作成）
```

### 2. 新規実装したファイル

#### 2.1 SoftF1ACLS損失関数（losses.py）

**MulticlassSoftF1ACLS**
- SoftF1LossにACLS正則化を組み合わせた損失関数
- パラメータ：
  - `num_classes`: クラス数
  - `beta`: F-beta scoreのパラメータ（デフォルト: 1.0）
  - `eps`: 数値安定性のepsilon（デフォルト: 1e-6）
  - `pos_lambda`: 正サンプル正則化重み（デフォルト: 1.0）
  - `neg_lambda`: 負サンプル正則化重み（デフォルト: 0.1）
  - `alpha`: 正則化項の全体重み（デフォルト: 0.1）
  - `margin`: 距離計算のマージン（デフォルト: 10.0）

**BinarySoftF1ACLS**
- バイナリ分類用のSoftF1ACLS
- MulticlassSoftF1ACLSと同様のパラメータ構成

**アルゴリズム**
1. SoftF1Lossを計算（1 - F1スコア）
2. 予測クラスと真のクラスを比較
3. 正しく分類されたサンプル（positive）と誤分類されたサンプル（negative）で異なる正則化を適用
4. 最終損失 = SoftF1Loss + alpha × regularization_term

#### 2.2 テストスクリプト（test_exp030_soft_f1_acls.py）

**テスト内容**
- 初期化テスト
- ロジット入力でのフォワードテスト
- 確率入力でのフォワードテスト
- 正則化項ゼロの場合のテスト
- 勾配計算テスト
- 2次元入力の処理テスト
- CrossEntropyとの比較テスト
- 正則化項の効果テスト

**テスト結果**: 13個のテストすべてがパス

### 3. 更新されたファイル

#### 3.1 config.py

**主な変更点**：
- `EXP_NUM`: "exp029" → "exp030"
- `ExperimentConfig.name`: "exp030_softf1_acls"
- `ExperimentConfig.description`: "SoftF1ACLS implementation with IMU-only LSTM, human normalization, and demographics"
- `ExperimentConfig.tags`: "soft_f1_acls"タグを追加
- `LossConfig.type`: "soft_f1_acls"オプションを追加、デフォルト値を"soft_f1_acls"に設定
- `LoggingConfig.wandb_tags`: "soft_f1_acls"タグを追加

#### 3.2 model.py

**主な変更点**：
- インポートに`MulticlassSoftF1ACLS`と`BinarySoftF1ACLS`を追加
- `_setup_loss_functions()`に"soft_f1_acls"処理を追加
- `_create_soft_f1_acls_criterion()`メソッドを追加
- `_create_binary_soft_f1_acls_criterion()`メソッドを追加
- `training_step()`に"soft_f1_acls"の損失計算処理を追加

## 技術的詳細

### SoftF1ACLSの設計理念

1. **ベース損失**: SoftF1Loss（F1スコアの微分可能な近似）
2. **ACLS正則化**: 予測の信頼度に基づく適応的な正則化
   - 正しく分類されたサンプルに対する正則化（pos_lambda）
   - 誤分類されたサンプルに対する正則化（neg_lambda）
3. **マージンベース距離計算**: 最大ロジットと真のクラスロジットの差分を用いた正則化

### パラメータ設定

実装では以下のデフォルト値を使用：
- `pos_lambda`: 1.0（正サンプル正則化重み）
- `neg_lambda`: 0.1（負サンプル正則化重み）
- `alpha`: 0.1（正則化項全体の重み）
- `margin`: 10.0（距離計算マージン）
- `beta`: 1.0（F-betaスコアパラメータ）
- `eps`: 1e-6（数値安定性）

これらのパラメータは既存のACLSConfigの設定値を流用しており、config.pyでの調整が可能です。

## 実装上の特徴

### 1. 柔軟な入力処理
- ロジットと確率値の両方に対応
- 自動的に入力タイプを判別して適切に処理

### 2. 勾配計算の安定性
- 数値安定性のためのepsilonを適切に配置
- log計算時のゼロ除算回避

### 3. 既存フレームワークとの互換性
- exp029の既存の実装パターンに従った設計
- 既存のACLSConfigを流用した設定管理

## 動作確認

### テスト結果
```bash
============================= test session starts ==============================
../../../tests/test_exp030_soft_f1_acls.py::TestMulticlassSoftF1ACLS::test_initialization PASSED [  7%]
../../../tests/test_exp030_soft_f1_acls.py::TestMulticlassSoftF1ACLS::test_forward_with_logits PASSED [ 15%]
../../../tests/test_exp030_soft_f1_acls.py::TestMulticlassSoftF1ACLS::test_forward_with_probs PASSED [ 23%]
../../../tests/test_exp030_soft_f1_acls.py::TestMulticlassSoftF1ACLS::test_zero_regularization PASSED [ 30%]
../../../tests/test_exp030_soft_f1_acls.py::TestMulticlassSoftF1ACLS::test_gradient_flow PASSED [ 38%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_initialization PASSED [ 46%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_forward_with_logits PASSED [ 53%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_forward_with_probs PASSED [ 61%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_zero_regularization PASSED [ 69%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_gradient_flow PASSED [ 76%]
../../../tests/test_exp030_soft_f1_acls.py::TestBinarySoftF1ACLS::test_2d_input_handling PASSED [ 84%]
../../../tests/test_exp030_soft_f1_acls.py::TestSoftF1ACLSComparison::test_compared_to_cross_entropy PASSED [ 92%]
../../../tests/test_exp030_soft_f1_acls.py::TestSoftF1ACLSComparison::test_regularization_effect PASSED [100%]

============================== 13 passed in 1.21s
```

### 直接実行結果
```bash
Testing MulticlassSoftF1ACLS...
Multiclass SoftF1ACLS loss: 0.7867

Testing BinarySoftF1ACLS...
Binary SoftF1ACLS loss: 0.3835

Testing gradients...
Logits grad norm: 0.0426
Binary logits grad norm: 0.0916

All tests completed successfully!
```

## 使用方法

### 1. 基本設定

config.pyで損失関数タイプを設定：
```python
loss:
  type: "soft_f1_acls"
```

### 2. パラメータ調整

ACLSConfigセクションでパラメータを調整：
```python
acls:
  acls_pos_lambda: 1.0     # 正サンプル正則化重み
  acls_neg_lambda: 0.1     # 負サンプル正則化重み  
  acls_alpha: 0.1          # 正則化項全体重み
  acls_margin: 10.0        # 距離計算マージン
```

### 3. 訓練実行

```bash
cd codes/exp/exp030
uv run python train.py
```

## まとめ

exp030では、SoftF1LossにACLSの適応的正則化を組み合わせた新しい損失関数を成功的に実装しました。実装は既存のフレームワークと完全に互換性があり、テストも13個すべてがパスしています。

**主な成果**：
1. ✅ SoftF1ACLSの設計と実装
2. ✅ exp029からexp030への差分最小化移行
3. ✅ 包括的なテストスイートの作成
4. ✅ 既存フレームワークとの完全な互換性確保
5. ✅ 設定ベースのパラメータ調整機能

この実装により、F1スコア最適化と適応的ラベルスムージングの両方の利点を活用した損失関数を利用できるようになりました。