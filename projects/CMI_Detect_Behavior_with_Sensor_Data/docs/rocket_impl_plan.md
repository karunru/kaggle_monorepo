# コンペの基本概念として以下のファイルを参照すること
- `docs/competition_overview.md`: コンペティション概要の確認
- `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `outputs/claude/`に実装計画のmarkdownを作成すること
# 新規実装アイデア
- `codes/exp/exp012/dataset.py` を参照して、以下の時系列を入力として `MiniRocketMultivariate` で変換した時系列もモデルの入力にする
```python
[
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "linear_acc_mag",
    "linear_acc_mag_jerk",
    "angular_vel_x",
    "angular_vel_y",
    "angular_vel_z",
    "angular_distance",
]
```
- `MiniRocketMultivariate` については https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.rocket.MiniRocketMultivariate.html を参照すること
  - n_jobは-1, random_stateはNone以外の適切な値を設定すること
  - num_kernelsは適切な値が何かを調査、思考して決定すること
  - sktimeでのデータの持ち方は https://www.sktime.net/en/stable/api_reference/data_format.html 及び配下のページを再帰的に参照して調査すること
    - 元のデータはpolarsで読み込んでいるから、できれば `TablePolarsEager`, `PanelPolarsEager` あたりを使って
  - sktimeやMiniRocketMultivariateの動作確認をしたい場合は別途 `outputs/claude/` にスクリプトを作成して実行すること
  - MiniRocketの派生元であるRocket及び派生モデルなどは https://chatgpt.com/share/6898c695-b6d4-8003-a2cd-baa163d25371 を参照すること
  - MiniRocketの技術的な背景は https://arxiv.org/abs/2012.08791 を参照すること