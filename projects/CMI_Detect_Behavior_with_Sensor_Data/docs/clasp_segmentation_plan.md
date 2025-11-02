# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- serenaのmemoryに従うこと
- notebooks/ 配下に notebookを作成して
# 新規実装アイデア
- https://www.sktime.net/en/stable/examples/annotation/segmentation_with_clasp.html を参考に、sequence毎にphase (Transition or Gesture) をsegmentationして
  - codes/exp/exp036/dataset.py を参照してimuに関する 19個の特徴量を入力にして
  - 元の波形と予測結果をそれぞれplotするようにして