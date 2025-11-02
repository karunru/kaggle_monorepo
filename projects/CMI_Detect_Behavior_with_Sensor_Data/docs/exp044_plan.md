# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp036` を `codes/exp/exp044` にコピーしてからexp044を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp036` からのインポートは禁止
- `codes/exp/exp044` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- model.pyの
```python
        # Dense layers (基本特徴量抽出)
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
```
は `nn.Sequential` を使ってまとめて一つで定義して
- IMUOnlyLSTMのマジックナンバーをModelConfigで設定できるようにして
- docs/chatgpt_fusion_heads_without_orientation.md を参考にして確率の統合と精度向上を実装して
- sequence_id, binary_probs, multiclass_probs_[1-18], nine_class_probs_[1-9], final_probs_[1-n], pred_gesture, true_gesture, foldをcolumnにもつcsvを最終的にoutputするようにして
