# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp026` を `codes/exp/exp027` にコピーしてからexp027を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp026` からのインポートは禁止
- `codes/exp/exp027` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- LossConfig.typeをcmi_focalにして
```python
        elif loss_type == "cmi_focal":
            # Focal Loss
            self.multiclass_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )
            self.binary_criterion = nn.BCEWithLogitsLoss()
            # 9クラス用はデフォルトのCrossEntropy
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```
を
```python
        elif loss_type == "cmi_focal":
            # Focal Loss
            self.multiclass_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )
            self.binary_criterion = nn.BCEWithLogitsLoss()
            # 9クラス用はデフォルトのCrossEntropy
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.nine_class_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )
```
にして