# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp024` を `codes/exp/exp025` にコピーしてからexp025を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp024` からのインポートは禁止
- `codes/exp/exp025` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
# 新規実装アイデア
- @docs/chatgpt_loss_plan.md , @outputs/codex/exp024_learnable_loss_weights.md を参照して total_loss の重みを学習可能なパラメータにして