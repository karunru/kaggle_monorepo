# Import the factory function first to avoid circular imports
# Define model type hints
from typing import Union

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from KTBoost.KTBoost import BoostingClassifier, BoostingRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from rgf.sklearn import FastRGFClassifier, FastRGFRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.svm import SVC, SVR

from .factory import get_model

# Define type aliases
CatModel = Union[CatBoostClassifier, CatBoostRegressor]
ERTModel = Union[ExtraTreesClassifier, ExtraTreesRegressor]
KTModel = Union[BoostingClassifier, BoostingRegressor]
RGFModel = Union[FastRGFClassifier, FastRGFRegressor]
SVMModel = Union[SVC, SVR]
XGBModel = Union[xgb.XGBClassifier, xgb.XGBRegressor]
RidgeModel = Union[SklearnRidge]
TabNetModel = Union[TabNetClassifier, TabNetRegressor]
LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]

# Now import the model classes
from .cat import CatBoost
from .ert import ExtremelyRandomizedTrees
from .ktb import KTBoost
from .lightgbm import LightGBM
from .rgf import RegularizedGreedyForest
from .ridge import Ridge
from .svm import SVM
from .tabnet import TabNet
from .xgb import XGBoost
