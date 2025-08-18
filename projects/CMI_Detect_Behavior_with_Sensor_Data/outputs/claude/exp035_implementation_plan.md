# exp035 実装計画

## 概要
exp031（Mish activation function実装）をベースに、exp027を参照してheadの手前にBERTを挟む実装を追加する。

## 実装手順

### 1. exp031をexp035にコピー
- codes/exp/exp031のすべてのファイルをcodes/exp/exp035にコピー
- test_*.pyファイルは除外

### 2. config.pyの更新
- EXP_NUM を "exp035" に変更
- ExperimentConfig.description: "BERT integration with Mish activation - head前にBERT層を追加"
- ExperimentConfig.tags: ["imu_only", "lstm", "bigru", "attention", "residual_se_cnn", "human_normalization", "demographics", "soft_f1_acls", "mish", "bert"]
- LoggingConfig.wandb_tags: 同上

### 3. model.pyへのBERT統合実装
CMISqueezeformerクラスの変更：

#### 3.1. インポートの追加
```python
from transformers import BertConfig, BertModel
```

#### 3.2. __init__メソッドの拡張
- bert_config パラメータを追加（BertConfigクラスと同じ設定を使用）
- BERTモデルの初期化
- CLSトークンの初期化
- 投影層の追加（d_modelとbert_hidden_sizeが異なる場合）

#### 3.3. forwardメソッドの修正
現在のモデル処理の流れ：
1. IMU処理 → LSTM/BiGRU → Attention → 分類ヘッド

新しい処理の流れ：
1. IMU処理 → LSTM/BiGRU → Attention → BERT → 分類ヘッド

具体的な変更：
- Attentionの出力後、BERTに入力する前に投影層を通す
- CLSトークンを追加
- BERTで処理
- CLSトークンの出力を取得
- Demographics特徴量と結合
- 分類ヘッドに入力

### 4. 最小限の差分維持
- exp031の既存実装は極力変更しない
- BERT層の追加は既存のアーキテクチャの最後、分類ヘッドの直前に挿入
- 他のファイル（dataset.py, train.py, inference.py等）は変更不要

### 5. 実装後の確認
- static analysis (ruff format, ruff check, mypy)
- 簡単な動作確認スクリプトの実行

## 期待される効果
- BERTの文脈理解能力により、時系列パターンの理解が向上
- CLSトークンによる系列全体の集約表現の改善
- 既存のMish activationとの相乗効果

## 実装状況

### 完了したタスク
1. ✅ 実装計画ドキュメントの作成
2. ✅ exp031をexp035にコピー  
3. ✅ config.pyの更新（description, tags, wandb_tags）
4. ✅ model.pyにBERT統合の実装を追加
5. ✅ BERTによる変更が他のファイルに影響しないか確認
6. ✅ 静的解析の実行 (ruff format, ruff check, mypy)
7. ✅ 簡単な動作確認スクリプトの実行

## 実装完了の要約

### 主な変更点
1. **config.py**
   - EXP_NUM を "exp035" に変更
   - description を "BERT integration with Mish activation - head前にBERT層を追加" に更新
   - tags と wandb_tags に "bert" を追加

2. **model.py**
   - transformers から BertConfig, BertModel をインポート
   - CMISqueezeformer クラスに bert_config パラメータを追加
   - BERT統合の実装:
     - CLSトークンの初期化
     - BiGRU出力からBERT hidden_sizeへの投影層
     - BERTモデルの初期化
     - 新しいBERT統合分類ヘッド（multiclass, binary, nine_class）
   - forward メソッドの変更:
     - IMUOnlyLSTMのattention出力を取得
     - BERTで処理してCLSトークンの出力を取得
     - Demographics特徴量と結合
     - 新しい分類ヘッドで予測

3. **train.py**
   - CMISqueezeformer 初期化時に demographics_config と bert_config を追加

### 動作確認結果
- モデルのテストが正常に完了
- 入力形状: torch.Size([2, 200, 20])
- 出力形状: multiclass(18), binary(1), nine_class(9)
- 総パラメータ数: 11,790,072（BERT統合により大幅増加）
- 比較: IMUOnlyLSTMのみでは478,077パラメータ

### 期待される効果
- BERTの自己注意機構により、時系列パターンの文脈理解が向上
- CLSトークンによる系列全体の集約表現の改善
- 既存のMish activationとの相乗効果により性能向上が期待される