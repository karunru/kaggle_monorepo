from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from cuml import SVC, SVR

from .base import BaseModel

SVMModel = Union[SVR, SVC]
AoD = Union[np.ndarray, pd.DataFrame, pl.DataFrame]
AoS = Union[np.ndarray, pd.Series, pl.Series]


class SVM(BaseModel):
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
    ) -> tuple[SVMModel, dict]:
        model_params = config["model"]["model_params"]

        self.config["categorical_cols"] = config["categorical_cols"]
        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            model = SVR(**model_params)
        else:
            model = SVC(probability=True, **model_params)

        self.num_feats = len(x_train.columns)

        model.fit(
            x_train,
            y_train,
        )
        best_score = model.score(x_valid, y_valid)
        return model, best_score

    def predict(self, model: SVMModel, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Convert to numpy array if needed
        features_values = features.to_numpy() if hasattr(features, "to_numpy") else features

        if self.mode == "regression":
            return model.predict(features_values)
        else:
            return model.predict_proba(features_values)[:, 1]

    def get_best_iteration(self, model: SVMModel) -> int:
        return 0

    def get_feature_importance(self, model: SVMModel) -> dict:
        # Create a dictionary with feature indices as keys and zeros as values
        # This ensures compatibility with pd.DataFrame.from_dict
        feature_importance_dict = {}
        for i in range(self.num_feats):
            feature_importance_dict[f"feature_{i}"] = 0.0
        return feature_importance_dict
