# WandB Configuration Update for exp004 and exp005

## 概要
exp004およびexp005のtrain.pyファイルに対して、WandB設定の更新を実施しました。exp006とexp007で使用されているパターンを適用し、より詳細なロギングと設定保存を実現しました。

## 実施内容

### 1. exp004/train.py の更新

#### setup_loggers関数の変更
- `wandb_config_dict = config.model_dump()` を追加し、全設定をWandBに保存
- `experiment_type` を `"schedule_free"` に設定
- `config` パラメータをWandbLoggerに明示的に渡すように変更
- モデルパラメータとデータ統計の追加メタデータをログ

#### train_single_fold関数の変更
- モデル作成後にパラメータ数を計算し、WandBに記録
  - `total_params`: 総パラメータ数
  - `trainable_params`: 学習可能パラメータ数
  - `model_size_mb`: モデルサイズ（MB）
- より詳細なvalidation結果の収集
  - `val_multiclass_loss` と `val_binary_loss` を追加
- 各foldの最終結果をWandBにログ
- Summary table用のデータ保存

### 2. exp005/train.py の更新

#### setup_loggers関数の変更
- exp004と同様の変更を実施
- `experiment_type` を `"length_grouping_schedule_free"` に設定

#### train_single_fold関数の変更
- exp004と同様の変更を実施
- モデルパラメータ数のログ
- 詳細なvalidation結果の収集とログ

## 主な変更点

### WandB設定の拡張
```python
# 設定の保存
wandb_config_dict = config.model_dump()
wandb_config_dict["fold"] = fold
wandb_config_dict["experiment_type"] = "schedule_free"  # または "length_grouping_schedule_free"

# WandBロガーの作成時にconfigを渡す
wandb_logger = WandbLogger(
    project=wandb_config.project,
    name=f"{wandb_config.name}_fold_{fold}",
    tags=wandb_config.tags + [f"fold_{fold}"],
    save_dir=config.paths.output_dir,
    config=wandb_config_dict,  # configを明示的に保存
)
```

### モデルパラメータのログ
```python
# モデルパラメータ数を計算してWandBにログ
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# WandBに記録
logger_instance.experiment.config.update({
    "model_params": {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
    }
})
```

### 詳細なメトリクスのログ
```python
# Fold固有のメトリクスをログ
logger_instance.experiment.log({
    f"fold_{fold}/final_val_loss": fold_results["val_loss"],
    f"fold_{fold}/final_val_cmi_score": fold_results["val_cmi_score"],
    f"fold_{fold}/final_val_multiclass_loss": fold_results["val_multiclass_loss"],
    f"fold_{fold}/final_val_binary_loss": fold_results["val_binary_loss"],
})
```

## 効果
- WandBダッシュボードでより詳細な実験追跡が可能に
- 各実験の設定とメタデータが完全に保存される
- モデルサイズとパラメータ数の可視化
- 各foldの詳細な結果の追跡
- 実験タイプ（`experiment_type`）による実験の分類が容易に

## 影響を受けるファイル
- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/exp/exp004/train.py`
- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/exp/exp005/train.py`