import datetime
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import cudf
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from xfeat.utils import is_cudf


def validation_refinement_by_day_of_week(
    df: pd.DataFrame, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pred_day = config["val"]["params"]["pred_day"]
    splits = slide_window_split_by_day(df["date"], config)

    split_refinement_by_day_of_week = []

    for trn_idx, val_idx in splits:
        val_days = df.loc[val_idx, "date"].unique()
        val_first_day = val_days[pred_day - 1]
        val_wday = df.query("date == @val_first_day")["wday"].iloc[0]
        refinement_val_idx = df.loc[val_idx].query("wday == @val_wday").index

        split_refinement_by_day_of_week.append((trn_idx, refinement_val_idx))

    return split_refinement_by_day_of_week


# https://eng.uber.com/omphalos/
# https://www.kaggle.com/harupy/m5-baseline?scriptVersionId=30229819
def slide_window_split_by_day(
    day_series: pd.Series, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    max_day = np.max(day_series)

    def _train_first_day(
        i: int,
        max_day: np.datetime64,
        n_split: int,
        train_days: int,
        valid_days: int,
        slide_step_days: int,
    ) -> np.datetime64:
        return max_day - np.timedelta64(
            (n_split - (i + 1)) * slide_step_days + train_days + valid_days - 1, "D"
        )

    def _valid_first_day(
        train_first_day: np.datetime64, train_days: int
    ) -> np.datetime64:
        return train_first_day + np.timedelta64(train_days, "D")

    def _valid_last_day(
        valid_first_day: np.datetime64, valid_days: int
    ) -> np.datetime64:
        return valid_first_day + np.timedelta64(valid_days - 1, "D")

    split = []

    for i in range(params["n_split"]):
        train_first_day = _train_first_day(
            i,
            max_day,
            n_split=params["n_split"],
            train_days=params["train_days"],
            valid_days=params["valid_days"],
            slide_step_days=params["slide_step_days"],
        )
        valid_first_day = _valid_first_day(
            train_first_day, train_days=params["train_days"]
        )
        valid_last_day = _valid_last_day(
            valid_first_day, valid_days=params["valid_days"]
        )

        is_trn = (day_series >= train_first_day) & (day_series < valid_first_day)
        is_val = (day_series >= valid_first_day) & (day_series <= valid_last_day)
        trn_idx = day_series[is_trn].index
        val_idx = day_series[is_val].index

        split.append((trn_idx, val_idx))

    return split


def date_hold_out(
    df: pd.DataFrame, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    date_col = params["date_col"]
    threshold_date = params["threshold_date"]
    threshold_date = datetime.datetime.strptime(threshold_date, "%Y-%m-%d")

    split = []

    df = df.reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    is_trn = df[date_col] < threshold_date
    is_val = df[date_col] >= threshold_date
    trn_idx = df[is_trn].index
    val_idx = df[is_val].index

    split.append((trn_idx, val_idx))

    return split


def kfold(
    df: pd.DataFrame, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    kf = KFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )
    split = []
    for trn_idx, val_idx in kf.split(df):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def group_kfold(
    df: pd.DataFrame, groups: pd.Series, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    kf = KFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )
    uniq_groups = groups.unique()
    split = []
    for trn_grp_idx, val_grp_idx in kf.split(uniq_groups):
        trn_grp = uniq_groups[trn_grp_idx]
        val_grp = uniq_groups[val_grp_idx]
        trn_idx = df[df[params["group"]].isin(trn_grp)].index.values
        val_idx = df[df[params["group"]].isin(val_grp)].index.values
        split.append((trn_idx, val_idx))

    return split


def stratified_kfold(
    df: pd.DataFrame, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    skf = StratifiedKFold(
        n_splits=params["n_splits"], random_state=params["random_state"], shuffle=True
    )

    y = (
        df[params["target"]].to_numpy()
        if is_cudf(df)
        else np.array(df[params["target"]])
    )
    X_col = [col for col in df.columns.to_list() if col is not params["target"]]
    split = []
    for trn_idx, val_idx in skf.split(df[X_col], y):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def stratified_group_kfold(
    df: pd.DataFrame, groups: pd.Series, config: Union[DictConfig, ListConfig]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]

    sgkf = StratifiedGroupKFold(
        n_splits=params["n_splits"],
        random_state=params["random_state"],
        shuffle=True,
    )

    split = []
    for trn_idx, val_idx in sgkf.split(
        df,
        y=df[params["target"]],
        groups=df[params["group"]],
    ):
        split.append((np.asarray(trn_idx), np.asarray(val_idx)))
    return split


def get_validation(
    df: pd.DataFrame, config: Union[DictConfig, ListConfig], is_pseudo_label=False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    fold_file_path = Path(config["data_path"]) / (
        config["val"]["name"]
        + "_n_splits_"
        + str(config["val"]["params"]["n_splits"])
        + "_target_"
        + str(config["val"]["params"]["target"])
        + "_random_state_"
        + str(config["val"]["params"]["random_state"])
        + f"{'_pseudo_label' if is_pseudo_label else ''}"
        + ".csv"
    )
    _df = df.copy()

    if fold_file_path.exists() and not config["val"]["params"]["force_recreate"]:
        print(f"load {fold_file_path}")

        fold_df = cudf.read_csv(fold_file_path)

        split = []
        for fold in range(config["val"]["params"]["n_splits"]):
            trn_ids = fold_df.loc[
                fold_df["fold"] != fold, config["val"]["params"]["id"]
            ]
            trn_idx = _df[
                _df[config["val"]["params"]["id"]].isin(trn_ids.to_pandas())
            ].index.to_numpy()
            val_ids = fold_df.loc[
                fold_df["fold"] == fold, config["val"]["params"]["id"]
            ]
            val_idx = _df[
                _df[config["val"]["params"]["id"]].isin(val_ids.to_pandas())
            ].index.to_numpy()
            if is_cudf(_df):
                split.append((np.asarray(trn_idx.get()), np.asarray(val_idx.get())))
            else:
                split.append((np.asarray(trn_idx), np.asarray(val_idx)))

    else:
        print(f"make {fold_file_path}")

        name: str = config["val"]["name"]

        func = globals().get(name)
        if func is None:
            raise NotImplementedError

        if "group" in name:
            groups_col = config["val"]["params"]["group"]
            groups = _df[groups_col]
            split = func(_df, groups, config)
        else:
            split = func(_df, config)

        _df["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(split):
            _df.loc[val_idx, "fold"] = fold

        if isinstance(config["val"]["params"]["id"], list):
            _df[config["val"]["params"]["id"] + ["fold"]].to_csv(
                fold_file_path, index=False
            )
        else:
            _df[[config["val"]["params"]["id"], "fold"]].to_csv(
                fold_file_path, index=False
            )

    return split
