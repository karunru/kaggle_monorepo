# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp022` を `codes/exp/exp023` にコピーしてからexp023を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp022` からのインポートは禁止
- `codes/exp/exp023` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
# 新規実装アイデア
- outputs/codex/model_architecture_investigation_exp021.md の 9 クラス補助ヘッドの追加 を実装して
- config.pyのhn_enabledはFalseにして
- LossConfig.typeは"acls"にして
