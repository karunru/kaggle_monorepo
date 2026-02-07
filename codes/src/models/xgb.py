from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from .base import BaseModel

XGBModel = Union[xgb.XGBClassifier, xgb.XGBRegressor]
AoD = Union[np.ndarray, pd.DataFrame, pl.DataFrame]
AoS = Union[np.ndarray, pd.Series, pl.Series]


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, features=None, target=None, batch_size=256 * 256):
        self.features = features
        self.target = target
        self.it = 0  # set iterator to 0
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(features) / self.batch_size))
        super().__init__()

    def reset(self):
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data):
        """Yield next batch of data."""
        if self.it == self.batches:
            return 0  # Return 0 when there's no more batch.

        a = self.it * self.batch_size
        b = min((self.it + 1) * self.batch_size, len(self.features))
        data = self.features.iloc[a:b]
        label = self.target[a:b]
        input_data(data=data, label=label)  # , weight=dt['weight'])
        self.it += 1
        return 1


class XGBoost(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> tuple[XGBModel, dict]:
        xgb.set_config(use_rmm=True)
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        dtrain = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(data=x_valid, label=y_valid, enable_categorical=True)

        callbacks = None

        # TRAIN MODEL FOLD K
        model = xgb.train(
            model_params,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=callbacks,
            **train_params,
        )
        # model.save_model(f"XGB_v{VER}_fold{fold}.xgb")
        best_score = model.best_score
        return model, best_score

    def get_best_iteration(self, model: XGBModel):
        return model.best_iteration

    def predict(
        self,
        model: XGBModel,
        features: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        return (
            pd.Series(
                model.predict(
                    xgb.DMatrix(data=features),
                ),
            )
            # .rank(pct=True)
            .to_numpy()
        )

    def get_feature_importance(self, model: XGBModel) -> np.ndarray:
        return model.get_score(importance_type="gain")
