# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp019` を `codes/exp/exp021` にコピーしてからexp021を実装すること
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp019` からのインポートは禁止
# 新規実装アイデア
- loss_type == "cmi_focal" と loss_type == "cmi" で、nn.CrossEntropyLoss, F.cross_entropy(inputs, targets, reduction="none") にlabel_smoothingという引数を渡せるようにconfigとmodelの実装を修正して
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 公式ドキュメントを確認して