# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp013` を `codes/exp/exp018` にコピーしてからexp018を実装すること
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp013` からのインポートは禁止
# 新規実装アイデア
- https://www.kaggle.com/code/cody11null/public-bert-training-attempt を参考に、modelの `multiclass_head` と `binary_head` の手前にbertを挟み込んで