# exp018 実装計画

## 概要
exp013のモデルにBERTアテンション層を組み込み、分類性能の向上を図る。

## 参考資料
- 参考ノートブック: https://www.kaggle.com/code/cody11null/public-bert-training-attempt
- ベースコード: codes/exp/exp013

## 実装方針

### 1. BERT統合の設計
- **挿入位置**: Squeezeformerブロックの後、Global poolingの代わりにBERTを使用
- **理由**: 
  - Squeezeformerで抽出された時系列特徴量に対してBERTの自己注意機構を適用
  - CLSトークンを使用することで、より効果的な特徴集約が可能

### 2. アーキテクチャの変更点

#### 現在のexp013の流れ:
```
IMU入力 → Input Projection → Positional Encoding → Squeezeformer Blocks 
→ Global Pooling → Demographics結合 → multiclass_head/binary_head
```

#### exp018での新しい流れ:
```
IMU入力 → Input Projection → Positional Encoding → Squeezeformer Blocks 
→ [CLS]トークン追加 → BERT → CLS出力取得 → Demographics結合 
→ multiclass_head/binary_head
```

### 3. 実装タスク

#### タスク1: exp013をexp018にコピー
- codes/exp/exp013を codes/exp/exp018にコピー
- __init__.pyの内容を更新

#### タスク2: BERTの追加 (model.py)
- transformersライブラリからBertConfig、BertModelをインポート
- CMISqueezeformerクラスの変更:
  - CLSトークンのパラメータ定義を追加
  - BERTモデルの初期化を追加
  - forward関数でGlobal PoolingをBERT処理に置き換え

#### タスク3: 設定の追加 (config.py)
- BERT関連のハイパーパラメータを追加:
  - bert_hidden_size (デフォルト: d_model)
  - bert_num_layers (デフォルト: 4)
  - bert_num_heads (デフォルト: 8)
  - bert_intermediate_size (デフォルト: d_model * 4)

#### タスク4: テストコードの作成
- tests/test_exp018_model.py を作成
- 以下をテスト:
  - モデルが正常に初期化されること
  - forward passが正常に動作すること
  - 出力形状が正しいこと
  - 勾配が正常に流れること

### 4. 実装の詳細

#### BERTの初期化コード:
```python
from transformers import BertConfig, BertModel

# __init__内
# CLSトークンの定義
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

# BERT設定
bert_config = BertConfig(
    hidden_size=config.bert_hidden_size or d_model,
    num_hidden_layers=config.bert_num_layers,
    num_attention_heads=config.bert_num_heads,
    intermediate_size=config.bert_intermediate_size or d_model * 4,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
self.bert = BertModel(bert_config)
```

#### forward関数の変更:
```python
# Squeezeformer blocks後、Global Poolingを削除してBERT処理に置き換え
# CLSトークンを追加
batch_size = x.size(0)
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1+seq_len, d_model]

# attention_maskの調整（CLSトークン分を追加）
if attention_mask is not None:
    cls_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
    bert_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
else:
    bert_attention_mask = None

# BERT処理
bert_output = self.bert(inputs_embeds=x, attention_mask=bert_attention_mask)
x = bert_output.last_hidden_state[:, 0, :]  # CLSトークンの出力を取得 [batch, d_model]

# Demographics特徴量の統合（既存のコードを維持）
if self.use_demographics and demographics is not None:
    demographics_embedding = self.demographics_embedding(demographics)
    x = torch.cat([x, demographics_embedding], dim=-1)
elif self.use_demographics and demographics is None:
    batch_size = x.size(0)
    demographics_padding = torch.zeros(batch_size, self.demographics_dim, device=x.device)
    x = torch.cat([x, demographics_padding], dim=-1)
```

### 5. 依存関係の更新
- pyproject.tomlにtransformersライブラリを追加（まだ含まれていない場合）

### 6. 注意点
- BERTの追加により計算量が増加するため、メモリ使用量とトレーニング時間に注意
- gradient checkpointingの使用を検討（メモリ削減のため）
- attention_maskの適切な処理を確保

### 7. 期待される効果
- BERTの自己注意機構により、時系列データ内の長距離依存関係をより効果的に捉えられる
- CLSトークンによる特徴集約がGlobal poolingよりも柔軟で効果的
- 参考ノートブックでの検証済みアプローチであり、性能向上が期待できる

## 実装スケジュール
1. exp013のコピーとセットアップ (5分)
2. model.pyへのBERT統合 (20分)
3. config.pyの更新 (5分)
4. テストコードの作成と実行 (15分)
5. 動作確認と調整 (10分)

合計予想時間: 約55分