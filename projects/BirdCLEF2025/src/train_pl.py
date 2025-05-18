import atexit
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from timm import create_model

from dataset import HMSDataModule, HMSConvTranDataModule
from models import Model, ChrisModel, ConvTranModule, FreezeUnfreezeCallback # noqa
from sampling import get_sampling
from utils import (
    find_exp_num,
    parse_args,
    remove_abnormal_exp,
    seed_everything,
    kaggle_kl_div_score,
)
from validation import get_validation
from lightning.pytorch.tuner import Tuner
from torch import nn


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

    df = pd.read_csv(Path(config.data_path) / "train.csv")
    target_cols = df.columns[-6:].tolist()
    oof_pred_cols = [f"oof_{target_col}" for target_col in target_cols]
    df[oof_pred_cols] = -1
    num_features = (
        create_model(
            config.model.backbone,
            pretrained=False,
            num_classes=0,
            in_chans=3,
        ).num_features
    )
    emb_cols = [f"emb_{i}" for i in range(num_features)]
    df.loc[:, emb_cols] = -1

    splits = get_validation(df, config)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold not in config.train_folds:
            continue

        train_df = df.loc[train_idx].reset_index(drop=True)
        train_df, _ = get_sampling(train_df, train_df[config.val.params.target], config)
        print("sampled:")
        print(train_df.groupby(config.val.params.target)["eeg_id"].count())

        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = eval(config.datamodule)(
            train_df=train_df,
            val_df=val_df,
            target_cols=target_cols,
            cfg=config,
        )
        model = eval(config.model.name)(config)
        if config.load_weight.do:
            weight_paths = sorted(
                list(
                    Path(
                        str(config.weight_path).replace(
                            str(exp_num), config.load_weight.exp
                        )
                    ).glob(f"fold_{fold}_best_loss*")
                )
            )
            last_weight_path = (
                weight_paths[-2] if len(weight_paths) >= 2 else weight_paths[0]
            )
            print(f"load {last_weight_path}")
            model.load_state_dict(torch.load(last_weight_path)["state_dict"])
        # model = torch.compile(model)

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
        freeze_epoch = FreezeUnfreezeCallback(
            freeze_epoch=config.load_weight.freeze_epoch
        )

        if config.optimizer.use_SAM:
            config.trainer.accumulate_grad_batches = 1

        wandb_logger._prefix = f"fold_{fold}"
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=config.epoch,
            callbacks=[
                early_stopping,
                swa,
                lr_monitor,
                loss_checkpoint,
                # freeze_epoch,
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
        df.loc[val_idx, oof_pred_cols] = oof_pred
        df.loc[val_idx, emb_cols] = emb

        oof_df = pd.DataFrame(oof_pred.copy())
        oof_df["id"] = np.arange(len(oof_df))
        true_df = pd.DataFrame(df.loc[val_idx, target_cols].to_numpy())
        true_df["id"] = np.arange(len(true_df))
        print(oof_df)
        score = kaggle_kl_div_score(
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
        pl.utilities.memory.garbage_collection_cuda()

    wandb_logger.experiment.config.update({"mean_cv": np.mean(scores)})
    df["id"] = np.arange(len(df))
    wandb_logger.experiment.config.update(
        {
            "oof_score": kaggle_kl_div_score(
                solution=df.loc[df[oof_pred_cols[0]] >= 0, target_cols + ["id"]],
                submission=df.loc[
                    df[oof_pred_cols[0]] >= 0, oof_pred_cols + ["id"]
                ].rename(columns=dict(zip(oof_pred_cols, target_cols))),
                row_id_column_name="id",
            )
        }
    )

    save_cols = (
        [
            "eeg_id",
            "patient_id",
        ]
        + oof_pred_cols
        + target_cols
        + emb_cols
    )

    df[save_cols].to_csv(Path(config.pred_path) / "oof_pred.csv", index=False)

    OmegaConf.save(config, Path(config.config_path) / f"exp_{exp_num}.yaml")


if __name__ == "__main__":
    main()
