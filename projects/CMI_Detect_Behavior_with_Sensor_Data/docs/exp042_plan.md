# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp041` を `codes/exp/exp042` にコピーしてからexp042を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp041` からのインポートは禁止
- `codes/exp/exp042` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- IMUOnlyLSTMのマジックナンバーをModelConfigで設定できるようにして
- model.py で orientation_criterion もsoft_f1, acls とかと同じif blockにいれるようにして
- ScheduleFreeConfig.enabledをTrueにして
  - https://github.com/facebookresearch/schedule_free を参照して正しく動くようになっているか確認して eval のタイミングとか
- config.pyで使ってないConfigや変数があれば消して