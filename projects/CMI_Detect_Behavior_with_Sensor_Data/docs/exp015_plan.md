# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp013` を `codes/exp/exp015` にコピーしてからexp14を実装すること
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp013` からのインポートは禁止
# 新規実装アイデア
- https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds を参考に、今dataset.pyで実装している特徴量生成をmodel側でpytorchで実装し直して
  - `codes/exp/exp013/dataset.py` と 参考notebookの特徴量の差分を調査して、重複は省いて両方の特徴量をpytorchで実装して
  - 不要になったコードやconfigは削除して
  - TOF/Thermalなどは使ってないからいじらないで
  - モデルアーキテクチャは今のままにして
  - 特徴量生成以外は変更しないで