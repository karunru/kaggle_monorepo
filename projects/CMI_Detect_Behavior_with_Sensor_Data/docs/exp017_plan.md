# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp013` を `codes/exp/exp017` にコピーしてからexp17を実装すること
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp013` からのインポートは禁止
# 新規実装アイデア
-     type: Literal["cmi", "cmi_focal", "soft_f1", "acls", "label_smoothing", "mbls"] = Field(
        default="acls", description="損失関数タイプ"
    )
 を `default="soft_l1"` にして
- その他013用の説明文を017用に書き換えて