import abc
import gc
import inspect
import logging
from pathlib import Path
from typing import Any

import polars as pl
from src.utils import timer


class Feature(metaclass=abc.ABCMeta):
    prefix = ""
    suffix = ""
    save_dir = "features"
    is_feature = True
    config: dict[str, Any] | None = None

    def __init__(self):
        self.name = self.__class__.__name__
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self.train = pl.DataFrame()
        self.valid = pl.DataFrame()
        self.test = pl.DataFrame()
        self.train_path = Path(self.save_dir) / f"{self.name}_train.arrow"
        self.test_path = Path(self.save_dir) / f"{self.name}_test.arrow"

    def run(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame = None,
        log: bool = False,
        config: dict[str, Any] | None = None,
    ):
        self.config = config
        with timer(self.name, log=log):
            self.create_features(train_df, test_df=test_df)
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = self.suffix + "_" if self.suffix else ""
            self.train = self.train.rename(lambda column_name: f"{prefix}{column_name}{suffix}")
            self.valid = self.valid.rename(lambda column_name: f"{prefix}{column_name}{suffix}")
            self.test = self.test.rename(lambda column_name: f"{prefix}{column_name}{suffix}")
        return self

    @abc.abstractmethod
    def create_features(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame = None,
    ):
        raise NotImplementedError

    def save(self):
        self.train.select(pl.all().shrink_dtype()).write_ipc(self.train_path, compression="zstd")
        self.test.select(pl.all().shrink_dtype()).write_ipc(self.test_path, compression="zstd")


def is_feature(klass) -> bool:
    return "is_feature" in set(dir(klass))


def get_features(namespace: dict):
    for v in namespace.values():
        if inspect.isclass(v) and is_feature(v) and not inspect.isabstract(v):
            yield v()


def generate_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    namespace: dict,
    required: list,
    overwrite: bool,
    log: bool = False,
    config: dict[str, Any] | None = None,
):
    for f in get_features(namespace):
        if (f.name not in required) or (f.train_path.exists() and f.test_path.exists() and not overwrite):
            if not log:
                print(f.name, "was skipped")
            else:
                logging.info(f"{f.name} was skipped")
        else:
            f.run(train_df, test_df, log, config).save()

            gc.collect()


def load_features(config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    feature_path = config["dataset"]["feature_dir"]

    with timer("load train"):
        x_train = pl.read_ipc(f"{feature_path}/{config['features'][0]}_train.arrow")
        for feature in config["features"][1:]:
            if Path(f"{feature_path}/{feature}_train.arrow").exists():
                x_train = x_train.join(
                    pl.read_ipc(f"{feature_path}/{feature}_train.arrow"),
                    how="left",
                    on=config["index"],
                )

    with timer("load test"):
        x_test = pl.read_ipc(f"{feature_path}/{config['features'][0]}_test.arrow")
        for feature in config["features"][1:]:
            if Path(f"{feature_path}/{feature}_test.arrow").exists():
                x_test = x_test.join(
                    pl.read_ipc(f"{feature_path}/{feature}_test.arrow"),
                    how="left",
                    on=config["index"],
                )

    return x_train, x_test
