# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp019` を `codes/exp/exp020` にコピーしてからexp020を実装すること
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp019` からのインポートは禁止
# 新規実装アイデア
- https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583023#3249995 には
```
I'm curious what features beyond public notebooks you've tried that were actually helpful. I've tried csums, diffs, longer ranged diffs, shifts, diffs from sequence median, wavelets, and some others. 
```
と記載があります。
精度を向上させるために、上記を含むremove gravity されたimuのデータを使った特徴量を考えて実装して