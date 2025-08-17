# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp032` を `codes/exp/exp033` にコピーしてからexp033を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp032` からのインポートは禁止
- `codes/exp/exp033` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- exp032では exp031の ResidualSECNNBlock を FNOBlock1D などに置き換えたが、exp033では ResidualSECNNBlock と FNOBlock1D の両方に通してconcatするようにして
  - これらのドキュメントを見て理解して
  - outputs/claude/exp032_implementation_summary.md
  - docs/exp032_plan.md
  - codes/exp/exp031/model.py