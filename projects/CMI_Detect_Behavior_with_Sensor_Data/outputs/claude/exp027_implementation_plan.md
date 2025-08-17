# Exp027実装計画: FocalLoss for Nine-Class Head

## 実装日時
2025-08-16

## 実装概要
exp026をベースに、nine_class_criterionでもFocalLossを使用するよう変更するexp027を実装します。

## 実装タスク

### タスク1: exp026のコピーとセットアップ ⏳
- `codes/exp/exp026`を`codes/exp/exp027`にコピー
- test_*.pyファイルは除外
- `__pycache__`ディレクトリは除外

### タスク2: config.py更新 ⏳
- `ExperimentConfig.description`: "FocalLoss for both multiclass and nine-class heads"
- `ExperimentConfig.tags`: ["focal_loss", "nine_class_focal", "multiclass_consistency", "squeezeformer", "pytorch_lightning"]
- `LoggingConfig.wandb_tags`: 上記と同様のタグに更新

### タスク3: 損失関数実装変更 ⏳
- `model.py`のcmi_focal損失実装部分を変更
- `nine_class_criterion`を`nn.CrossEntropyLoss`から`FocalLoss`に変更
- パラメータ設定: `focal_gamma`, `focal_alpha`, `label_smoothing`を使用

### タスク4: テストコード作成 ⏳
- exp027の基本機能テスト
- 損失関数の動作確認テスト

### タスク5: 静的解析・テスト実行 ⏳
- ruff format/lint実行
- mypy型チェック実行  
- pytestテスト実行

### タスク6: 実装完了文書更新 ⏳

## 技術詳細

### 変更箇所
```python
# 変更前 (exp026)
self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# 変更後 (exp027)  
self.nine_class_criterion = FocalLoss(
    gamma=self.loss_config.get("focal_gamma", 2.0),
    alpha=self.loss_config.get("focal_alpha", 1.0),
    label_smoothing=self.loss_config.get("label_smoothing", 0.0),
)
```

### 期待される効果
- nine_class_criterionでもFocalLossの恩恵（難しいサンプルに集中）を得られる
- multiclass_criterionとnine_class_criterionの整合性向上

## 実装ステータス
- [x] タスク1: exp026のコピー
- [x] タスク2: config.py更新  
- [x] タスク3: 損失関数変更
- [x] タスク4: テストコード
- [x] タスク5: 静的解析・テスト
- [x] タスク6: 文書更新

## 成果物

### 実装完了したファイル
- `codes/exp/exp027/`: exp027実装コード一式
  - `config.py`: 実験設定（description, tags, wandb_tags更新済み）
  - `model.py`: nine_class_criterionでFocalLoss使用に変更済み
  - `dataset.py`: データセット実装
  - `losses.py`: 損失関数実装
  - `train.py`: 訓練スクリプト
  - `inference.py`: 推論スクリプト
  - `human_normalization.py`: Human Normalization実装
  - `__init__.py`: モジュール初期化

### テストコード
- `tests/test_exp027_config.py`: 設定テスト（7テスト、全て通過）
- `tests/test_exp027_focal_loss.py`: FocalLoss実装テスト（6テスト、全て通過）
- `tests/test_exp027_integration.py`: 統合テスト（5テスト、全て通過）

### 実装変更詳細

#### config.pyの変更
```python
# 実験メタデータ
EXP_NUM = "exp027"
name = "exp027_focal_loss_nine_class"
description = "FocalLoss for both multiclass and nine-class heads"
tags = ["focal_loss", "nine_class_focal", "multiclass_consistency", "squeezeformer", "pytorch_lightning"]
wandb_tags = ["exp027", "focal_loss", "nine_class_focal", "multiclass_consistency", "squeezeformer"]
```

#### model.pyの変更
```python
# 変更前 (exp026)
self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# 変更後 (exp027)  
self.nine_class_criterion = FocalLoss(
    gamma=self.loss_config.get("focal_gamma", 2.0),
    alpha=self.loss_config.get("focal_alpha", 1.0),
    label_smoothing=self.loss_config.get("label_smoothing", 0.0),
)
```

### テスト結果
- **総テスト数**: 18テスト
- **成功**: 18テスト
- **失敗**: 0テスト
- **警告**: 1件（Pydantic deprecation warning、機能に影響なし）

### 品質確認
- **静的解析**: ruff format/lint実行済み（コード整形済み）
- **型チェック**: 構造上のmypy問題は全体コードベースの問題（exp027固有の問題なし）
- **テスト**: 全テスト通過

## 技術的達成内容
1. **FocalLoss統一**: multiclass_criterionとnine_class_criterionの両方でFocalLossを使用
2. **パラメータ一致**: 両損失関数で同一のgamma, alpha, label_smoothingパラメータ使用
3. **設定整合性**: 実験メタデータの正確な更新
4. **テスト充実**: 設定、損失関数、統合の3方面でのテスト実装