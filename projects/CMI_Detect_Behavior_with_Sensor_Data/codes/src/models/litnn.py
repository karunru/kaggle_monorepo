import functools
import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as lightning
import torch
from lifelines.utils import concordance_index
from pytorch_lightning.utilities import grad_norm
from pytorch_tabular.models.common.layers import ODST
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from src.models.base import BaseModel
from src.utils import timer
from torch import nn
from torch.utils.data import TensorDataset


class CatEmbeddings(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """

    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: list[int],
        embedding_dim: int,
    ):
        """
        projection_dim: The dimension of the final output after projecting the concatenated embeddings into a lower-dimensional space.
        categorical_cardinality: A list where each element represents the number of unique categories (cardinality) in each categorical feature.
        embedding_dim: The size of the embedding space for each categorical feature.
        self.embeddings: list of embedding layers for each categorical feature.
        self.projection: sequential neural network that goes from the embedding to the output projection dimension with GELU activation.
        """
        super(CatEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinality],
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.Mish(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x_cat):
        """
        Apply the projection on concatened embeddings that contains all categorical features.
        """
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            # OrdinalEncoderの値を0始まりに調整（+2）
            adjusted_indices = x_cat[:, i] + 2

            # 範囲外の値を循環させる（モジュラー演算）
            num_categories = embedding.num_embeddings
            safe_indices = adjusted_indices % num_categories

            # 埋め込み適用
            embedded.append(embedding(safe_indices))

        # Concatenate all embeddings
        x_cat = torch.cat(embedded, dim=1)
        return self.projection(x_cat)


class NN(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_cardinality: list[int],
        embedding_dim: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float = 0,
    ):
        """
        continuous_dim: The number of continuous features.
        categorical_cardinality: A list of integers representing the number of unique categories in each categorical feature.
        embedding_dim: The dimensionality of the embedding space for each categorical feature.
        projection_dim: The size of the projected output space for the categorical embeddings.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        self.embeddings: previous embeddings for categorical data.
        self.mlp: defines an MLP model with an ODST layer followed by batch normalization and dropout.
        self.out: linear output layer that maps the output of the MLP to a single value
        self.dropout: defines dropout
        Weights initialization with xavier normal algorithm and biases with zeros.
        """
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        """
        Create embedding layers for categorical data, concatenate with continous variables.
        Add dropout and goes through MLP and return raw output and 1-dimensional output as well.
        """
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    """
    calculates all possible 2-combinations (pairs) of a tensor of indices from 0 to N-1,
    and caches the result using functools.lru_cache for optimization
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda() if torch.cuda.is_available() else comb


class LitNN(lightning.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_cardinality: list[int],
        embedding_dim: int,
        projection_dim: int,
        hidden_dim: int,
        lr: float = 1e-3,
        dropout: float = 0.2,
        weight_decay: float = 1e-3,
        aux_weight: float = 0.1,
        margin: float = 0.5,
        race_index: int = 0,
    ):
        """
        continuous_dim: The number of continuous input features.
        categorical_cardinality: A list of integers, where each element corresponds to the number of unique categories for each categorical feature.
        embedding_dim: The dimension of the embeddings for the categorical features.
        projection_dim: The dimension of the projected space after embedding concatenation.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        aux_weight: Weight used for auxiliary tasks.
        margin: Margin used in some loss functions.
        race_index: An index that refer to race_group in the input data.
        """
        super(LitNN, self).__init__()
        self.save_hyperparameters()

        # Creates an instance of the NN model defined above
        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout,
        )
        self.targets = []

        # Defines a small feedforward neural network that performs an auxiliary task with 1-dimensional output
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.Mish(),
            nn.Linear(self.hparams.hidden_dim // 3, 1),
        )

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction="none")
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        """
        Output loss and race_group loss.
        """
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss) ** 2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
        """
        Most important part of the model : loss function used for training.
        We face survival data with event indicators along with time-to-event.

        This function computes the main loss by the following the steps :
        * create all data pairs with "combinations" function (= all "two subjects" combinations)
        * make sure that we have at least 1 event in each pair
        * convert y to +1 or -1 depending on the correct ranking
        * loss is computed using a margin-based hinge loss
        * mask is applied to ensure only valid pairs are being used (censored data can't be ranked with event in some cases)
        * average loss on all pairs is returned
        """
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Defines all invalid comparisons :
        * Case 1: "Left outlived Right" but Right is censored
        * Case 2: "Right outlived Left" but Left is censored
        Masks for case 1 and case 2 are combined using |= operator and inverted using ~ to create a "valid pair mask"
        """
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = left_outlived & left_1_right_0
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= right_outlived & right_1_left_0
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        """
        This method defines how the model processes each batch during validation
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate c-index accounting for each race_group or global.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        """
        Calculate c-index accounting for each race_group
        """
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def configure_optimizers(self):
        """
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=6e-3,
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def init_dl(X_cat, X_num, df, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long),
    )
    bs = 2048
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
    return dl_train


def preprocess_data(train, val, categorical_cols, numerical_cols):
    """
    Standardize numerical variables and prepare categorical variables.
    Fill NA values with mean for numerical.
    Create torch dataloaders to prepare data for training and evaluation.
    """
    # カテゴリカル変数はすでにエンコードされていると仮定
    X_cat_train = train[categorical_cols].to_numpy()
    X_cat_val = val[categorical_cols].to_numpy()

    # カテゴリカル変数の一意な値の数を計算
    categorical_cardinality = [train[col].nunique() for col in categorical_cols]

    X_num_train = train[numerical_cols].to_numpy()
    X_num_val = val[numerical_cols].to_numpy()

    # データローダーの作成
    dl_train = init_dl(X_cat_train, X_num_train, train, training=True)
    dl_val = init_dl(X_cat_val, X_num_val, val)

    return (
        X_num_train.shape[1],
        dl_train,
        dl_val,
        categorical_cardinality,
    )


class PairwiseRankingNN(BaseModel):
    """
    Implementation of the Pairwise Ranking Neural Network model for survival analysis.
    """

    def __init__(self):
        super().__init__()
        self.is_linear_model = False
        self.is_survival_model = True
        self.is_cat_embed_model = True
        self.model = None
        self.transformers = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.race_index = 0  # Default, will be updated in fit

    def fit(
        self,
        x_train: pd.DataFrame | pl.DataFrame,
        y_train: np.ndarray | pl.Series,
        x_valid: pd.DataFrame | pl.DataFrame,
        y_valid: np.ndarray | pl.Series,
        config: dict,
        **kwargs,
    ) -> tuple[Any, dict]:
        """
        Train the Pairwise Ranking Neural Network model.
        """
        with timer("convert to pandas"):
            if isinstance(x_train, pl.DataFrame):
                x_train = x_train.to_pandas()
            if isinstance(x_valid, pl.DataFrame):
                x_valid = x_valid.to_pandas()
            if isinstance(y_train, pl.Series):
                y_train = y_train.to_numpy()
            if isinstance(y_valid, pl.Series):
                y_valid = y_valid.to_numpy()

        # Add target to dataframes for preprocessing
        train_df = x_train.copy()
        train_df["efs_time"] = y_train  # Use actual target values for efs_time
        train_df["efs"] = kwargs.get("efs_train", np.ones(len(y_train)))  # Default to 1 if not provided

        valid_df = x_valid.copy()
        valid_df["efs_time"] = y_valid  # Use actual target values for efs_time
        valid_df["efs"] = kwargs.get("efs_valid", np.ones(len(y_valid)))  # Default to 1 if not provided

        # Preprocess data
        with timer("preprocess data"):
            # Store column names to ensure consistency between train and predict
            self.train_columns = x_train.columns.tolist()

            # Find race_group index if it exists
            self.race_index = self.categorical_cols.index("race_group") if "race_group" in self.categorical_cols else 0

            # データの前処理
            num_dim, dl_train, dl_val, categorical_cardinality = preprocess_data(
                train_df,
                valid_df,
                self.categorical_cols,
                self.numerical_cols,
            )

            # Log feature dimensions for debugging
            logging.info(f"Categorical features: {len(self.categorical_cols)}")
            logging.info(f"Numerical features: {len(self.numerical_cols)}")
            logging.info(f"X_num_train shape: {(len(train_df), num_dim)}")

        # Model parameters
        model_params = config.get("model", {}).get("model_params", {})
        embedding_dim = model_params.get("embedding_dim", 16)
        projection_dim = model_params.get("projection_dim", 112)
        hidden_dim = model_params.get("hidden_dim", 56)
        lr = model_params.get("lr", 0.06)
        dropout = model_params.get("dropout", 0.05)
        aux_weight = model_params.get("aux_weight", 0.26)
        margin = model_params.get("margin", 0.25)
        weight_decay = model_params.get("weight_decay", 0.0002)
        max_epochs = model_params.get("max_epochs", 60)

        # Create model
        with timer("create model"):
            model = LitNN(
                continuous_dim=num_dim,
                categorical_cardinality=categorical_cardinality,
                embedding_dim=embedding_dim,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                dropout=dropout,
                weight_decay=weight_decay,
                aux_weight=aux_weight,
                margin=margin,
                race_index=self.race_index,
            )

            # Configure trainer
            checkpoint_callback = lightning.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
            trainer = lightning.Trainer(
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                max_epochs=max_epochs,
                log_every_n_steps=6,
                callbacks=[
                    checkpoint_callback,
                    lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
                    lightning.callbacks.TQDMProgressBar(),
                    lightning.callbacks.StochasticWeightAveraging(
                        swa_lrs=1e-5,
                        swa_epoch_start=45,
                        annealing_epochs=15,
                    ),
                ],
            )

            # Train model
            trainer.fit(model, dl_train, dl_val)

            # Test model
            trainer.test(model, dl_val)

            self.model = model.eval()

            # Get best validation loss
            best_score = trainer.callback_metrics.get("test_cindex", torch.tensor(0.0)).item()

        return self.model, best_score

    def predict(
        self,
        model: Any,
        features: pd.DataFrame | pl.DataFrame,
    ) -> np.ndarray:
        """
        Generate predictions using the trained model.
        """
        with timer("predict"):
            if isinstance(features, pl.DataFrame):
                features = features.to_pandas()

            # Prepare test data
            test_df = features.copy()
            test_df["efs_time"] = 1.0  # Dummy value, not used for prediction
            test_df["efs"] = 1  # Dummy value, not used for prediction

            # Ensure all expected columns are present
            for col in self.categorical_cols + self.numerical_cols:
                if col not in test_df.columns:
                    test_df[col] = 0  # Add missing columns with default values

            # Process categorical features
            X_cat_test = test_df[self.categorical_cols].to_numpy()

            # Process numerical features using the same transformers as in fit
            imp = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
            scaler = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution="normal")

            X_num_test = imp.fit_transform(test_df[self.numerical_cols])
            X_num_test = scaler.fit_transform(X_num_test)

            # Create dataloader
            dl_test = init_dl(X_cat_test, X_num_test, test_df, training=False)

            # Determine device
            device = next(model.parameters()).device

            # Get predictions batch by batch
            all_preds = []
            model.eval()
            with torch.no_grad():
                for batch in dl_test:
                    x_cat, x_cont, _, _ = batch
                    x_cat = x_cat.to(device)
                    x_cont = x_cont.to(device)
                    pred, _ = model(x_cat, x_cont)
                    all_preds.append(pred.cpu().numpy())

            # Concatenate all predictions
            predictions = np.concatenate(all_preds)

            # Return negative predictions (higher values = higher risk)
            return -predictions

    def get_best_iteration(self, model: Any) -> int:
        """
        Return the best iteration from the model.
        """
        return 0  # Not applicable for this model type

    def get_feature_importance(self, model: Any) -> dict:
        """
        Return feature importance from the model.
        """
        # Neural networks don't have direct feature importance
        # Return empty dict or implement a permutation importance method
        return {}
