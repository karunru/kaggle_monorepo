# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp031` を `codes/exp/exp032` にコピーしてからexp032を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp031` からのインポートは禁止
- `codes/exp/exp032` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- docs/chatgpt_Fourier_Neural_Operator.mdを参照して、実装プランAのFNO-1DブロックをCNNブロックの代わりにを実装して