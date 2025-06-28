import atexit
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from dataset import BirdCLEFDataModule  # noqa
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from models import Model  # noqa
from omegaconf import OmegaConf
from sampling import get_sampling
from timm import create_model
from utils import (
    birdclef_roc_auc,
    find_exp_num,
    parse_args,
    remove_abnormal_exp,
    seed_everything,
)
from validation import get_validation

warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(
        remove_abnormal_exp,
        log_path=config.log_path,
        config_path=config.config_path,
    )
    seed_everything(config.seed)

    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("medium")

    exp_num = find_exp_num(config_path=config.config_path)
    exp_num = str(exp_num).zfill(3)
    config.exp_num = exp_num

    wandb_logger = WandbLogger(
        project="BirdCLEF2025",
        name=f"exp_{exp_num}",
    )

    config.weight_path = str(Path(config.weight_path) / f"exp_{exp_num}")
    os.makedirs(config.weight_path, exist_ok=True)
    config.pred_path = str(Path(config.pred_path) / f"exp_{exp_num}")
    os.makedirs(config.pred_path, exist_ok=True)
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(
            config,
            resolve=True,
        )
    )

    df = pl.read_csv(
        Path(config.data_path) / "train.csv",
        null_values={"latitude": "None", "longitude": "None"},
        infer_schema_length=10_0000,
    )
    labels = df["primary_label"].unique().sort().to_list()
    oof_pred_cols = [f"oof_{target_col}" for target_col in labels]
    df = df.with_columns([pl.lit(None).alias(col) for col in oof_pred_cols])
    num_features = create_model(
        config.model.backbone,
        pretrained=False,
        num_classes=0,
        in_chans=1,
    ).num_features
    emb_cols = [f"emb_{i:04}" for i in range(num_features)]
    df = df.with_columns([pl.lit(None).alias(col) for col in emb_cols])

    splits = get_validation(df, config)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold not in config.train_folds:
            continue

        train_df = df.filter(pl.int_range(len(df)).is_in(train_idx))
        train_df, _ = get_sampling(train_df, train_df[config.val.params.target], config)
        print("sampled:")
        print(train_df[config.val.params.target].value_counts(sort=True))

        val_df = df.filter(pl.int_range(len(df)).is_in(val_idx))
        datamodule = eval(config.datamodule)(  # noqa: S307
            train_df=train_df,
            val_df=val_df,
            labels=labels,
            cfg=config,
        )
        model = eval(config.model.name)(config)  # noqa: S307
        model = torch.compile(model)

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping.patience,
        )
        swa = callbacks.StochasticWeightAveraging(
            swa_lrs=config.stochastic_weight_avg.swa_lrs,
            swa_epoch_start=config.stochastic_weight_avg.swa_epoch_start,
            annealing_epochs=config.stochastic_weight_avg.annealing_epochs,
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=config.weight_path,
            filename=f"fold_{fold}_best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            save_weights_only=True,
        )

        wandb_logger._prefix = f"fold_{fold}"
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=config.epoch,
            callbacks=[
                early_stopping,
                swa,
                lr_monitor,
                loss_checkpoint,
            ],
            **config.trainer,
        )

        trainer.fit(model, datamodule=datamodule)

        output = trainer.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            return_predictions=True,
            ckpt_path="best",
        )
        oof_pred = []
        emb = []
        for _output in output:
            oof_pred.append(_output[0])
            emb.append(_output[1])

        oof_pred = torch.cat(oof_pred).to(torch.float32).numpy()
        emb = torch.cat(emb).to(torch.float32).numpy()

        # Update oof predictions using polars
        for i, col in enumerate(oof_pred_cols):
            # 全体の長さのリストを作成し、val_idxの位置に値を設定
            full_pred = [None] * len(df)
            for j, idx in enumerate(val_idx):
                full_pred[idx] = oof_pred[j, i]

            df = df.with_columns(pl.col(col).fill_null(pl.Series(full_pred)))

        # Update embeddings using polars
        for i, col in enumerate(emb_cols):
            full_emb = [None] * len(df)
            for j, idx in enumerate(val_idx):
                full_emb[idx] = emb[j, i]

            df = df.with_columns(pl.col(col).fill_null(pl.Series(full_emb)))

        oof_df = pd.DataFrame(oof_pred.copy())
        oof_df["id"] = np.arange(len(oof_df))

        # Get target columns from validation data using polars
        val_target_data = val_df["primary_label"].to_dummies().to_numpy()
        true_df = pd.DataFrame(val_target_data)
        true_df["id"] = np.arange(len(true_df))

        print(oof_df)
        score = birdclef_roc_auc(
            solution=true_df,
            submission=oof_df,
            row_id_column_name="id",
        )
        scores.append(score)
        wandb_logger.experiment.config.update({f"fold_{fold}/best_score": score})

        del (
            trainer,
            model,
            oof_pred,
        )
        gc.collect()
        torch.cuda.empty_cache()
        garbage_collection_cuda()

    wandb_logger.experiment.config.update({"mean_cv": np.mean(scores)})
    df = df.with_columns(pl.int_range(len(df)).alias("id"))

    # Calculate OOF score using polars filtering
    filtered_df = df.filter(pl.col(oof_pred_cols[0]).is_not_null())

    # Create solution_df with one-hot encoded labels
    target_data = filtered_df["primary_label"].to_dummies().to_numpy()
    solution_df = pd.DataFrame(target_data, columns=labels)
    solution_df["id"] = np.arange(len(solution_df))

    # Create submission_df with predictions
    submission_df = filtered_df.select([*oof_pred_cols]).to_pandas()
    submission_df = submission_df.rename(columns=dict(zip(oof_pred_cols, labels)))
    submission_df["id"] = np.arange(len(submission_df))

    wandb_logger.experiment.config.update(
        {
            "oof_score": birdclef_roc_auc(
                solution=solution_df,
                submission=submission_df,
                row_id_column_name="id",
            )
        }
    )

    save_cols = ["filename", "primary_label", *oof_pred_cols, *emb_cols]

    df.select(save_cols).write_csv(Path(config.pred_path) / "oof_pred.csv")

    OmegaConf.save(config, Path(config.config_path) / f"exp_{exp_num}.yaml")


if __name__ == "__main__":
    main()
