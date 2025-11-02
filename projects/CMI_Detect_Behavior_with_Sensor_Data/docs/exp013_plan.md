# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
  - `codes/exp/exp012` をベースとして `codes/exp/exp013` を実装すること
  - 差分は指示されたもの以外最小限にすること 
# 新規実装アイデア
- `model.py` の
```python
        # Classification heads
        multiclass_logits = self.multiclass_head(x)
        binary_logits = self.binary_head(x)
```
の前に、demographicsの情報を埋め込んだベクトルをconcatしてclassification headに入力する
  - https://chatgpt.com/s/dr_68989d445ac0819195fc10506d2865fe を参照して実装計画を立てること
  - `age`, `height_cm`などの特徴量は推論時にtrainにない範囲の値がきても大丈夫なようにscalingをすること