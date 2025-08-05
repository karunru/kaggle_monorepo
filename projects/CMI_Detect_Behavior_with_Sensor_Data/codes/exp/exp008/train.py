"""IMU-only Squeezeformer訓練スクリプト - exp008用(欠損値をattention_maskで処理)."""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from config import Config
from dataset import IMUDataModule
from model import CMISqueezeformer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Imports from project root
from src.utils.logger import create_logger
from src.utils.seed_everything import seed_everything

# Warnings suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_callbacks(config: Config) -> list[pl.Callback]:
    """コールバックの設定."""
    callbacks = []

    # Model checkpoint
    checkpoint_config = config.lightning.callbacks.model_checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
        filename=checkpoint_config.filename,
        auto_insert_metric_name=checkpoint_config.auto_insert_metric_name,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping_config = config.lightning.callbacks.early_stopping
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_config.monitor,
        mode=early_stopping_config.mode,
        patience=config.training.early_stopping_patience,
        min_delta=early_stopping_config.min_delta,
        verbose=early_stopping_config.verbose,
    )
    callbacks.append(early_stopping_callback)

    # Learning rate monitor
    lr_monitor_callback = LearningRateMonitor(logging_interval=config.lightning.callbacks.lr_monitor.logging_interval)
    callbacks.append(lr_monitor_callback)

    # Rich progress bar
    if config.lightning.callbacks.rich_progress_bar.enable:
        rich_progress_callback = RichProgressBar()
        callbacks.append(rich_progress_callback)

    return callbacks


def setup_loggers(config: Config, fold: int, wandb_logger=None) -> list:
    """ロガーの設定."""
    loggers = []

    # CSV Logger
    csv_logger = CSVLogger(save_dir=config.paths.output_dir, name=f"fold_{fold}", version=None)
    loggers.append(csv_logger)

    # WandB Logger (if provided)
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    return loggers


def create_trainer(config: Config, callbacks: list[pl.Callback], loggers: list, fold: int) -> pl.Trainer:
    """PyTorch Lightning Trainerの作成."""
    trainer_config = config.lightning.trainer

    # フォールド別のディレクトリ設定
    fold_dir = os.path.join(config.paths.output_dir, f"fold_{fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        max_epochs=config.training.epochs,  # 動的参照
        gradient_clip_val=config.training.gradient_clip_val,  # 動的参照
        accumulate_grad_batches=config.training.accumulate_grad_batches,  # 動的参照
        deterministic=trainer_config.deterministic,
        benchmark=trainer_config.benchmark,
        enable_checkpointing=trainer_config.enable_checkpointing,
        default_root_dir=fold_dir,
        enable_progress_bar=True,
        log_every_n_steps=trainer_config.log_every_n_steps,
        check_val_every_n_epoch=trainer_config.check_val_every_n_epoch,
        val_check_interval=trainer_config.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
    )

    return trainer


def train_single_fold(config: Config, fold: int, logger, wandb_logger) -> dict:
    """単一フォールドの訓練."""
    logger.info(f"Starting training for fold {fold}")

    # WandBロガーのprefixを設定(foldごとのログ分離)
    if wandb_logger is not None:
        wandb_logger._prefix = f"fold_{fold}"

    # Data module
    data_module = IMUDataModule(config, fold=fold)

    # Model
    model = CMISqueezeformer(
        input_dim=config.model.input_dim,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        num_classes=config.model.num_classes,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_config={
            "type": config.training.scheduler_type,
            "min_lr": config.training.scheduler_min_lr,
            "factor": config.training.scheduler_factor,
            "patience": config.training.scheduler_patience,
        },
        loss_config=config.loss.model_dump(),
        schedule_free_config=config.schedule_free.model_dump(),
        ema_config=config.ema.model_dump(),
    )

    # Callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config, fold, wandb_logger)

    # Trainer
    trainer = create_trainer(config, callbacks, loggers, fold)

    # Training
    trainer.fit(model, data_module)

    # Best model loading
    if hasattr(trainer.checkpoint_callback, "best_model_path"):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Loading best model from: {best_model_path}")
            model = CMISqueezeformer.load_from_checkpoint(best_model_path)

    # Validation
    val_results = trainer.validate(model, data_module)

    # より詳細なvalidation結果を収集
    val_result = val_results[0]
    fold_results = {
        "fold": fold,
        "val_loss": val_result["val_loss"],
        "val_cmi_score": val_result.get("val_cmi_score", 0.0),
        "val_multiclass_loss": val_result.get("val_multiclass_loss", 0.0),
        "val_binary_loss": val_result.get("val_binary_loss", 0.0),
        "best_model_path": trainer.checkpoint_callback.best_model_path
        if hasattr(trainer.checkpoint_callback, "best_model_path")
        else None,
    }

    logger.info(f"Fold {fold} results: {fold_results}")

    # WandBにfold結果をログ(_prefixによりfold_{fold}/メトリクス名の形式になる)
    if wandb_logger is not None:
        # Fold固有のメトリクスをログ(_prefixが自動的に付与される)
        wandb_logger.experiment.log(
            {
                "final_val_loss": fold_results["val_loss"],
                "final_val_cmi_score": fold_results["val_cmi_score"],
                "final_val_multiclass_loss": fold_results["val_multiclass_loss"],
                "final_val_binary_loss": fold_results["val_binary_loss"],
            }
        )

        # Summary table用のデータも保存
        wandb_logger.experiment.config.update({f"fold_{fold}_results": fold_results}, allow_val_change=True)

    return fold_results


def train_cross_validation(config: Config) -> dict:
    """クロスバリデーション訓練."""
    # ロガーの設定
    log_dir = Path(config.paths.output_dir) / "logs"
    logger = create_logger(__name__, log_dir / "train.log")

    logger.info("Starting cross-validation training")
    logger.info(f"Configuration: {config.model_dump()}")

    # WandBロガーの作成(foldに関係なく単一のロガー)
    wandb_logger = None
    if config.logging.wandb_enabled:
        # WandBに保存する設定を準備
        wandb_config_dict = config.model_dump()

        wandb_logger = WandbLogger(
            project=config.logging.wandb_project,
            name=config.logging.wandb_name,
            tags=config.logging.wandb_tags,
            save_dir=config.paths.output_dir,
            config=wandb_config_dict,
        )

    # シード設定
    seed_everything(config.training.seed)

    # 各フォールドの訓練
    fold_results = []
    n_splits = config.val.params.n_splits

    for fold in range(n_splits):
        try:
            fold_result = train_single_fold(config, fold, logger, wandb_logger)
            fold_results.append(fold_result)

            # メモリクリーンアップ
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception:
            logger.exception(f"Error in fold {fold}")
            raise

    # 結果の集計
    val_losses = [result["val_loss"] for result in fold_results]
    val_cmi_scores = [result["val_cmi_score"] for result in fold_results]

    cv_results = {
        "mean_val_loss": np.mean(val_losses),
        "std_val_loss": np.std(val_losses),
        "mean_val_cmi_score": np.mean(val_cmi_scores),
        "std_val_cmi_score": np.std(val_cmi_scores),
        "fold_results": fold_results,
    }

    logger.info("Cross-validation completed")
    logger.info(f"Mean CV Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    logger.info(f"Mean CV CMI Score: {cv_results['mean_val_cmi_score']:.4f} ± {cv_results['std_val_cmi_score']:.4f}")

    # 結果をCSVで保存
    results_df = pd.DataFrame(fold_results)
    results_path = Path(config.paths.output_dir) / "cv_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")

    # WandBにCV全体の結果をログ(各foldで最初に有効化されたWandBインスタンスを使用)
    if wandb_logger is not None:
        wandb_logger.experiment.config.update(cv_results, allow_val_change=True)
        logger.info("CV results logged to WandB")
    else:
        logger.info("WandB disabled, skipping CV results logging")

    return cv_results


def main():
    """メイン関数."""

    # 設定の読み込み
    config = Config()

    # 出力ディレクトリの作成
    config.create_output_dirs()

    # データパスの存在確認
    if not config.validate_paths():
        msg = f"Train data not found: {config.data.train_path}"
        raise FileNotFoundError(msg)

    # GPU使用可能性の確認
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    # クロスバリデーション訓練の実行
    cv_results = train_cross_validation(config)

    print("\n" + "=" * 50)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(f"Mean Validation Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    print(f"Mean CMI Score: {cv_results['mean_val_cmi_score']:.4f} ± {cv_results['std_val_cmi_score']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
