# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp023` を `codes/exp/exp024` にコピーしてからexp024を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp023` からのインポートは禁止
- `codes/exp/exp024` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
# 新規実装アイデア
- @docs/chatgpt_loss_plan.md の議論を元に、multiclassの分布とnine_classの分布が近づく用にKLダイバージェンスをlossに加えて
  - multiclassとnine_classの順番とか対応付けに気をつけて
