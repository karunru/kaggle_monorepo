# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp049` を `codes/exp/exp050` にコピーしてからexp050を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp049` からのインポートは禁止
- `codes/exp/exp050` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- @codes/exp/exp043/model.pyを参考にして、 ResidualSECNNBlock を ResidualSEConvNeXtBlock に置き換えて