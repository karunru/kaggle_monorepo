# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp028` を `codes/exp/exp029` にコピーしてからexp029を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp028` からのインポートは禁止
- `codes/exp/exp029` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- `codes/exp/exp027` にある項目をexp029に加えて
  - human normalization
  - DemographicsEmbedding
  - nine_class, multi_class, binary の3つのheadとkl_divを含むloss
  - 上記以外にもあれば実装して
  - モデルアーキテクチャと特徴量以外はexp027と同じになるようにしたい
- model.pyをはじめとしてexp029内でDRYになるように、動作が変わらない範囲でリファクタリングして
  - 重複したconfigとかもありそう