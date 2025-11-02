import gc
import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch.cuda
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from src.evaluation import calc_metric
from src.sampling import get_sampling
from src.utils import timer

# Define a generic Model type
Model = Any


class BaseModel:
    is_linear_model = False
    is_survival_model = False
    is_cat_embed_model = False

    @abstractmethod
    def fit(
        self,
        x_train: pl.DataFrame,
        y_train: pl.Series | np.ndarray,
        x_valid: pl.DataFrame,
        y_valid: pl.Series | np.ndarray,
        config: dict,
        **kwargs,
    ) -> tuple[Model, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self, model: Model) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model, features: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self, model: Model) -> dict:
        raise NotImplementedError

    def pre_process_for_liner_model(
        self,
        cat_cols: list,
        x_train: pl.DataFrame,
        x_valid: pl.DataFrame | None,
        x_valid2: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        num_cols = [col for col in x_train.columns if col not in cat_cols]

        # Convert to pandas for one-hot encoding
        x_train_pd = x_train.to_pandas()
        x_valid_pd = x_valid.to_pandas() if x_valid is not None else None
        x_valid2_pd = x_valid2.to_pandas() if x_valid2 is not None else None

        # One hot encoding
        x_train_pd = pd.get_dummies(
            x_train_pd,
            dummy_na=True,
            columns=cat_cols,
        )
        x_valid_pd = (
            pd.get_dummies(
                x_valid_pd,
                dummy_na=True,
                columns=cat_cols,
            )
            if x_valid_pd is not None
            else None
        )
        x_valid2_pd = (
            pd.get_dummies(
                x_valid2_pd,
                dummy_na=True,
                columns=cat_cols,
            )
            if x_valid2_pd is not None
            else None
        )

        # rank gauss transform
        transformer = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution="normal")

        # Use sklearn's SimpleImputer instead of cuml
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer()

        x_train_num_imputed = imputer.fit_transform(x_train_pd[num_cols])
        transformer.fit(x_train_num_imputed)

        x_train_pd[num_cols] = transformer.transform(x_train_num_imputed)

        use_cols = x_train_pd.columns.tolist()
        if x_valid_pd is not None:
            x_valid_num_imputed = imputer.transform(x_valid_pd[num_cols])
            x_valid_pd[num_cols] = transformer.transform(x_valid_num_imputed)
            use_cols = list(set(use_cols) & set(x_valid_pd.columns))

        if x_valid2_pd is not None:
            x_valid2_num_imputed = imputer.transform(x_valid2_pd[num_cols])
            x_valid2_pd[num_cols] = transformer.transform(x_valid2_num_imputed)
            use_cols = list(set(use_cols) & set(x_valid2_pd.columns))

        # Convert back to polars
        x_train_result = pl.from_pandas(x_train_pd[use_cols])
        x_valid_result = pl.from_pandas(x_valid_pd[use_cols]) if x_valid_pd is not None else None
        x_valid2_result = pl.from_pandas(x_valid2_pd[use_cols]) if x_valid2_pd is not None else None

        return (
            x_train_result,
            x_valid_result,
            x_valid2_result,
        )

    def pre_process_for_cat_embed_model(
        self,
        categorical_col: list[str],
        x_train: pl.DataFrame,
        x_valid: pl.DataFrame | None,
        x_valid2: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        self.categorical_cols = list(
            set(
                categorical_col + [col for col in x_train.columns if x_train.get_column(col).n_unique() < 25],
            ),
        )
        self.numerical_cols = list(set(x_train.columns) - set(self.categorical_cols))
        return (
            x_train,
            x_valid if x_valid is not None else None,
            x_valid2 if x_valid2 is not None else None,
        )

    def post_process(
        self,
        oof_preds: np.ndarray,
        test_preds: np.ndarray,
        valid_preds: np.ndarray | None,
        y_train: np.ndarray,
        y_valid: np.ndarray | None,
        train_features: pd.DataFrame | None,
        test_features: pd.DataFrame | None,
        valid_features: pd.DataFrame | None,
        target_scaler: MinMaxScaler | None,
        config: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        _y_train = y_train
        _oof_preds = oof_preds
        _test_preds = test_preds
        _y_valid = y_valid if y_valid is not None else None
        _valid_preds = valid_preds if valid_preds is not None else None

        # Check for Cox objective in either XGBoost or CatBoost parameters
        objective = config["model"]["model_params"].get("objective", "")
        loss_function = config["model"]["model_params"].get("loss_function", "")

        is_cox = "cox" in objective.lower() or "cox" in loss_function.lower()
        if is_cox and config["target"] != "efs_time":
            _y_train = -rankdata(y_train)
            _oof_preds = -rankdata(_oof_preds)
            _test_preds = -rankdata(_test_preds)

        # _oof_preds[_oof_preds < 0] = 0
        # _oof_preds[_oof_preds > 1] = 1
        # _test_preds[_test_preds < 0] = 0
        # _test_preds[_test_preds > 1] = 1
        # if _valid_preds is not None:
        #     _valid_preds[_valid_preds < 0] = 0
        #     _valid_preds[_valid_preds > 1] = 1

        return (
            _y_train,
            _oof_preds,
            _test_preds,
            _y_valid if y_valid is not None else None,
            _valid_preds if y_valid is not None else None,
        )

    def cv(
        self,
        y_train: pl.Series | np.ndarray,
        train_features: pl.DataFrame,
        test_features: pl.DataFrame,
        y_valid: pl.Series | np.ndarray | None,
        valid_features: pl.DataFrame | None,
        feature_name: list[str],
        folds_ids: list[tuple[np.ndarray, np.ndarray]],
        target_scaler: MinMaxScaler | None,
        config: dict,
        log: bool = True,
        efs_train: np.ndarray = None,
    ) -> tuple[list[Model], np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame, dict]:
        # initialize
        valid_exists = True if valid_features is not None else False
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if valid_exists:
            valid_preds = np.zeros(len(valid_features))
        else:
            valid_preds = None
        best_iteration = 0.0
        cv_score_list: list[dict] = []
        models: list[Model] = []

        with timer("make X"):
            X_train = train_features.clone()
            X_test = test_features.clone()
            X_valid = valid_features.clone() if valid_features is not None else None

            # Pre-process for linear models at the beginning
            if self.is_linear_model:
                X_train, X_valid, X_test = self.pre_process_for_liner_model(
                    cat_cols=config["categorical_cols"],
                    x_train=X_train,
                    x_valid=X_valid,
                    x_valid2=X_test,
                )

            if self.is_cat_embed_model:
                X_train, X_valid, X_test = self.pre_process_for_cat_embed_model(
                    categorical_col=config["categorical_cols"],
                    x_train=X_train,
                    x_valid=X_valid,
                    x_valid2=X_test,
                )

        with timer("make y"):
            y = y_train.to_numpy() if isinstance(y_train, pl.Series) else y_train
            y_valid = y_valid.to_numpy() if isinstance(y_valid, pl.Series) else y_valid

        importances = pd.DataFrame(index=feature_name)

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            if i_fold not in config["train_folds"]:
                continue

            with timer(f"fold {i_fold}"):
                self.fold = i_fold
                with timer("get train data and valid data"):
                    # get train data and valid data
                    # Create a row number column for filtering
                    X_train_with_idx = X_train.with_row_index("__row_idx")
                    # Filter using row indices
                    x_trn = X_train_with_idx.filter(pl.col("__row_idx").is_in(trn_idx)).drop("__row_idx")
                    y_trn = y[trn_idx]
                    x_val = X_train_with_idx.filter(pl.col("__row_idx").is_in(val_idx)).drop("__row_idx")
                    y_val = y[val_idx]

                logging.info(f"train size: {x_trn.shape}, valid size: {x_val.shape}")

                with timer("get sampling"):
                    x_trn, y_trn = get_sampling(x_trn, y_trn, config)
                    logging.info(f"sampled train size: {x_trn.shape}")

                with timer("train model"):
                    # train model
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Pass event indicators to the fit method if available
                    if hasattr(self, "is_survival_model") and self.is_survival_model and efs_train is not None:
                        model, best_score = self.fit(
                            x_trn.to_pandas(),
                            y_trn,
                            x_val.to_pandas(),
                            y_val,
                            config=config,
                            efs_train=efs_train[trn_idx],
                            efs_valid=efs_train[val_idx],
                        )
                    else:
                        model, best_score = self.fit(
                            x_trn.to_pandas(),
                            y_trn,
                            x_val.to_pandas(),
                            y_val,
                            config=config,
                        )
                    cv_score_list.append(best_score)
                    models.append(model)
                    best_iteration += self.get_best_iteration(model) / len(folds_ids)
                    del x_trn, y_trn, y_val
                    gc.collect()
                    torch.cuda.empty_cache()

                with timer("predict oof and test"):
                    # predict oof and test
                    oof_preds[val_idx] = self.predict(model, x_val.to_pandas()).reshape(-1)
                    test_preds += self.predict(model, X_test.to_pandas()).reshape(-1) / len(folds_ids)

                    if valid_exists:
                        valid_preds += self.predict(model, X_valid.to_pandas()).reshape(-1) / len(folds_ids)

                with timer("get feature importance"):
                    # get feature importances
                    importances_tmp = pd.DataFrame.from_dict(
                        self.get_feature_importance(model),
                        orient="index",
                        columns=[f"gain_{i_fold + 1}"],
                    )
                    importances = importances.join(importances_tmp, how="inner")

                del model, x_val
                gc.collect()
                torch.cuda.empty_cache()

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds
        self.raw_valid_preds = valid_preds

        # post_process (if you have any)
        y, oof_preds, test_preds, y_valid, valid_preds = self.post_process(
            oof_preds=oof_preds,
            test_preds=test_preds,
            valid_preds=valid_preds,
            y_train=y_train,
            y_valid=y_valid,
            train_features=train_features,
            test_features=test_features,
            valid_features=valid_features,
            target_scaler=target_scaler,
            config=config,
        )

        evals_results = {
            "evals_result": {
                "cv_score": {f"cv{i + 1}": cv_score for i, cv_score in enumerate(cv_score_list)},
                "mean_cv_score": np.array(cv_score_list).mean(),
                "n_data": len(X_train),
                "best_iteration": best_iteration,
                "n_features": len(feature_name),
                "feature_importance": feature_importance.sort_values(ascending=False).to_dict(),
            },
        }

        return (
            models,
            oof_preds,
            test_preds,
            valid_preds,
            feature_importance,
            evals_results,
        )

    def cv_with_pseudo_label(
        self,
        y_train: pl.Series | np.ndarray,
        train_features: pl.DataFrame,
        test_features: pl.DataFrame,
        y_valid: pl.Series | np.ndarray | None,
        valid_features: pl.DataFrame | None,
        pseudo_label: pl.DataFrame,
        feature_name: list[str],
        folds_ids: list[tuple[np.ndarray, np.ndarray]],
        pseudo_label_folds_ids: list[tuple[np.ndarray, np.ndarray]],
        target_scaler: MinMaxScaler | None,
        config: dict,
        log: bool = True,
    ) -> tuple[list[Model], np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame, dict]:
        # initialize
        valid_exists = True if valid_features is not None else False
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if valid_exists:
            valid_preds = np.zeros(len(valid_features))
        else:
            valid_preds = None
        best_iteration = 0.0
        cv_score_list: list[dict] = []
        models: list[Model] = []

        with timer("make X"):
            X_train = train_features.clone()
            X_test = test_features.clone()
            X_valid = valid_features.clone() if valid_features is not None else None

        with timer("make y"):
            y = y_train.to_numpy() if isinstance(y_train, pl.Series) else y_train
            y_valid = y_valid.to_numpy() if isinstance(y_valid, pl.Series) else y_valid

        importances = pd.DataFrame(index=feature_name)

        for i_fold, (
            (trn_idx, val_idx),
            (pseudo_label_trn_idx, pseudo_label_val_idx),
        ) in enumerate(zip(folds_ids, pseudo_label_folds_ids, strict=False)):
            if i_fold not in config["train_folds"]:
                continue

            with timer(f"fold {i_fold}"):
                self.fold = i_fold

                with timer("get pseudo label train data and valid data"):
                    pseudo_label_trn_ID = pseudo_label.filter(pl.Series(pseudo_label_trn_idx).cast(pl.UInt32))[
                        config["val"]["params"]["id"]
                    ].clone()

                    pseudo_label_x_trn = X_test.filter(
                        pl.col(config["val"]["params"]["id"]).is_in(pseudo_label_trn_ID),
                    ).select([col for col in X_test.columns if col != config["val"]["params"]["id"]])

                    pseudo_label_y_trn = pseudo_label.filter(
                        pl.col(config["val"]["params"]["id"]).is_in(pseudo_label_trn_ID),
                    )[config["target"]].to_numpy()

                    logging.info(f"pseudo label train size: {pseudo_label_x_trn.shape}")

                with timer("get train data and valid data"):
                    # get train data and valid data
                    # Create a row number column for filtering
                    X_train_with_idx = X_train.with_row_index("__row_idx")
                    # Filter using row indices
                    x_trn = (
                        X_train_with_idx.filter(pl.col("__row_idx").is_in(trn_idx))
                        .drop("__row_idx")
                        .select(
                            [col for col in X_train.columns if col != config["val"]["params"]["id"]],
                        )
                    )
                    y_trn = y[trn_idx]
                    x_val = (
                        X_train_with_idx.filter(pl.col("__row_idx").is_in(val_idx))
                        .drop("__row_idx")
                        .select(
                            [col for col in X_train.columns if col != config["val"]["params"]["id"]],
                        )
                    )
                    y_val = y[val_idx]

                with timer("concat train and pseudo label"):
                    x_trn = pl.concat([x_trn, pseudo_label_x_trn], how="vertical")
                    y_trn = np.concatenate((y_trn, pseudo_label_y_trn), axis=0)
                    del pseudo_label_x_trn, pseudo_label_y_trn
                    gc.collect()
                    torch.cuda.empty_cache()

                logging.info(f"train size: {x_trn.shape}, valid size: {x_val.shape}")

                with timer("get sampling"):
                    x_trn, y_trn = get_sampling(x_trn, y_trn, config)
                    logging.info(f"sampled train size: {x_trn.shape}")

                with timer("train model"):
                    # train model
                    gc.collect()
                    torch.cuda.empty_cache()

                    model, best_score = self.fit(
                        x_trn,
                        y_trn,
                        x_val,
                        y_val,
                        config=config,
                    )
                    cv_score_list.append(best_score)
                    models.append(model)
                    best_iteration += self.get_best_iteration(model) / len(folds_ids)
                    del (x_trn, y_trn, y_val)
                    gc.collect()
                    torch.cuda.empty_cache()

                with timer("predict oof and test"):
                    # predict oof and test
                    oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)

                    # For test predictions, ensure we're using the right columns
                    test_features_for_pred = X_test
                    if config["val"]["params"]["id"] in X_test.columns:
                        test_features_for_pred = X_test.select(
                            [col for col in X_test.columns if col != config["val"]["params"]["id"]],
                        )

                    test_preds += self.predict(model, test_features_for_pred).reshape(-1) / len(folds_ids)

                    if valid_exists:
                        valid_preds += self.predict(model, X_valid).reshape(-1) / len(folds_ids)

                with timer("get feature importance"):
                    # get feature importances
                    importances_tmp = pd.DataFrame.from_dict(
                        self.get_feature_importance(model),
                        orient="index",
                        columns=[f"gain_{i_fold + 1}"],
                    )
                    importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds
        self.raw_valid_preds = valid_preds

        # post_process (if you have any)
        y, oof_preds, test_preds, y_valid, valid_preds = self.post_process(
            oof_preds=oof_preds,
            test_preds=test_preds,
            valid_preds=valid_preds,
            y_train=y_train,
            y_valid=y_valid,
            train_features=train_features,
            test_features=test_features,
            valid_features=valid_features,
            target_scaler=target_scaler,
            config=config,
        )

        # print oof score
        oof_score = calc_metric(y, oof_preds)
        logging.info(f"oof score: {oof_score:.5f}")

        if valid_exists:
            valid_score = calc_metric(y_valid, valid_preds)
            logging.info(f"valid score: {valid_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")
            if valid_exists:
                logging.info(f"valid score: {valid_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score": oof_score,
                "cv_score": {f"cv{i + 1}": cv_score for i, cv_score in enumerate(cv_score_list)},
                "n_data": X_train.shape[0],
                "best_iteration": best_iteration,
                "n_features": X_train.shape[1],
                "feature_importance": feature_importance.sort_values(ascending=False).to_dict(),
            },
        }

        if valid_exists:
            evals_results["valid_score"] = valid_score
        return (
            models,
            oof_preds,
            test_preds,
            valid_preds,
            feature_importance,
            evals_results,
        )
