from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge as SklearnRidge

from .base import BaseModel

RidgeModel = Union[SklearnRidge]
AoD = Union[np.ndarray, pd.DataFrame, pl.DataFrame]
AoS = Union[np.ndarray, pd.Series, pl.Series]


class Ridge(BaseModel):
    config = dict()
    is_linear_model = True

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> tuple[RidgeModel, dict]:
        model_params = config["model"]["model_params"]

        self.config["categorical_cols"] = config["categorical_cols"]
        mode = config["model"]["mode"]
        self.mode = mode

        self.num_feats = len(x_train.columns)

        model = SklearnRidge(**model_params)

        # Convert to numpy arrays if needed
        x_train_values = x_train.to_numpy() if hasattr(x_train, "to_numpy") else x_train
        y_train_values = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        x_valid_values = x_valid.to_numpy() if hasattr(x_valid, "to_numpy") else x_valid
        y_valid_values = y_valid.to_numpy() if hasattr(y_valid, "to_numpy") else y_valid

        model.fit(x_train_values, y_train_values)
        best_score = {"valid_score": model.score(x_valid_values, y_valid_values)}

        return model, best_score

    def predict(self, model: RidgeModel, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Convert to numpy array if needed
        features_values = features.to_numpy() if hasattr(features, "to_numpy") else features
        return model.predict(features_values)

    def get_best_iteration(self, model: RidgeModel) -> int:
        return 0

    def get_feature_importance(self, model: RidgeModel) -> np.ndarray:
        return np.zeros(self.num_feats)
