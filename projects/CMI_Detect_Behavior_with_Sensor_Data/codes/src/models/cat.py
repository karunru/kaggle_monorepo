import gc
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .base import BaseModel

CatModel = Union[CatBoostClassifier, CatBoostRegressor]
AoD = Union[np.ndarray, pd.DataFrame, pl.DataFrame]
AoS = Union[np.ndarray, pd.Series, pl.Series]


class CatBoost(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        self.mode = mode

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        if mode == "regression":
            model = CatBoostRegressor(
                cat_features=self.config["categorical_cols"],
                **model_params,
            )
        else:
            model = CatBoostClassifier(
                cat_features=self.config["categorical_cols"],
                **model_params,
            )

        # Convert to pandas if it's a polars DataFrame
        _x_train = x_train.to_pandas() if isinstance(x_train, pl.DataFrame) else x_train.copy()
        _x_valid = x_valid.to_pandas() if isinstance(x_valid, pl.DataFrame) else x_valid.copy()

        _x_train[self.config["categorical_cols"]] = _x_train[self.config["categorical_cols"]].astype(str)
        _x_valid[self.config["categorical_cols"]] = _x_valid[self.config["categorical_cols"]].astype(str)

        train_pool = Pool(
            data=_x_train,
            label=y_train,
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=_x_train.columns.tolist(),
        )
        valid_pool = Pool(
            data=_x_valid,
            label=y_valid,
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=_x_valid.columns.tolist(),
        )
        del _x_train, _x_valid, x_train, x_valid
        gc.collect()
        torch.cuda.empty_cache()

        model.fit(
            X=train_pool,
            eval_set=valid_pool,
        )
        best_score = model.best_score_["validation"][model_params["eval_metric"]]
        return model, best_score

    def get_best_iteration(self, model: CatModel) -> int:
        return model.best_iteration_

    def predict(
        self,
        model: CatModel,
        features: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        # Convert to pandas if it's a polars DataFrame
        _features = features.to_pandas() if isinstance(features, pl.DataFrame) else features.copy()
        _features[self.config["categorical_cols"]] = _features[self.config["categorical_cols"]].astype(str)
        data_pool = Pool(
            data=_features,
            cat_features=self.config["categorical_cols"],
            text_features=None,
            embedding_features=None,
            timestamp=None,
            feature_names=_features.columns.tolist(),
        )
        if self.mode == "binary":
            return model.predict_proba(data_pool)[:, 1]
        else:
            return model.predict(data_pool)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return dict(zip(model.feature_names_, model.get_feature_importance(), strict=False))
