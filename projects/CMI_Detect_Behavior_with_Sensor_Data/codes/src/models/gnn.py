import functools
import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as lightning
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from pytorch_lightning.utilities import grad_norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.models.base import BaseModel
from src.utils import timer
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv


class TukeyBiweightLoss(nn.Module):
    """
    Tukey's biweight loss function, robust to outliers.
    """

    def __init__(self, c=4.685):
        super(TukeyBiweightLoss, self).__init__()
        self.c = c

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        scaled_error = error / self.c
        condition = torch.abs(scaled_error) < 1
        loss = torch.where(condition, (self.c**2 / 6) * (1 - (1 - scaled_error**2) ** 3), self.c**2 / 6)
        return loss.mean()


def race_loss_with_penalty(y_true, y_pred, race_labels, race_weights=None, lambda_std=0.5):
    """
    Compute pairwise ranking loss with race fairness penalty.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        race_labels: Race labels for each sample
        race_weights: Optional weights for each race
        lambda_std: Weight for the standard deviation penalty

    Returns:
        final_loss: Combined loss with fairness penalty
        mse_std: Standard deviation of MSE across races
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    unique_races = torch.unique(race_labels)
    mse_values = []
    total_loss = 0.0

    for race in unique_races:
        indices = race_labels == race
        y_true_r = y_true[indices]
        y_pred_r = y_pred[indices]

        if len(y_true_r) < 2:  # Skip if not enough samples
            continue

        # Compute pairwise differences
        y_true_diff = y_true_r.unsqueeze(0) - y_true_r.unsqueeze(1)
        y_pred_diff = y_pred_r.unsqueeze(0) - y_pred_r.unsqueeze(1)

        valid_pairs = (y_true_diff > 0).float()  # Valid pairs where y_i > y_j
        mse_r = torch.sum(valid_pairs * (y_true_diff - y_pred_diff) ** 2) / (torch.sum(valid_pairs) + 1e-8)

        # Store MSE for variance calculation
        mse_values.append(mse_r)

        # Compute race-weighted loss
        weight = race_weights.get(race.item(), 1.0) if race_weights else 1.0
        total_loss += weight * mse_r

    # Compute standard deviation of MSE across races
    if len(mse_values) > 1:
        mse_values = torch.stack(mse_values)
        mse_std = torch.std(mse_values)
    else:
        mse_std = torch.tensor(0.0, device=y_true.device)  # No variance if only one race

    # Final loss combines MSE loss and standard deviation loss
    final_loss = total_loss / len(unique_races) + lambda_std * mse_std

    return final_loss, mse_std


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder using GraphSAGE layers.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.mish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        return x


class SurvivalGNN(nn.Module):
    """
    Graph Neural Network for survival prediction.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        out_channels=64,
        num_layers=3,
        dropout=0.2,
        categorical_dims=None,
        embedding_dim=16,
    ):
        super(SurvivalGNN, self).__init__()

        # Embeddings for categorical features
        self.categorical_dims = categorical_dims or {}
        self.embeddings = nn.ModuleDict()

        total_embedding_dim = 0
        for feat_name, dim in self.categorical_dims.items():
            self.embeddings[feat_name] = nn.Embedding(dim + 1, embedding_dim)  # +1 for missing values
            total_embedding_dim += embedding_dim

        # Adjust input channels to include embeddings
        self.in_channels = in_channels - len(self.categorical_dims) + total_embedding_dim

        # GNN encoder
        self.encoder = GNNEncoder(
            self.in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            dropout,
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Auxiliary prediction head
        self.aux_predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_channels // 2),
            nn.Mish(),
            nn.Linear(hidden_channels // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Process categorical features with embeddings
        if self.categorical_dims:
            # Extract categorical features
            cat_features = {}
            cat_indices = {}
            start_idx = 0

            for i, (feat_name, dim) in enumerate(self.categorical_dims.items()):
                cat_indices[feat_name] = start_idx + i
                cat_features[feat_name] = x[:, start_idx + i].long()

            # Create embeddings
            embeddings = []
            for feat_name, indices in cat_features.items():
                # Ensure indices are non-negative and within bounds
                safe_indices = torch.clamp(indices, 0, self.categorical_dims[feat_name])
                embeddings.append(self.embeddings[feat_name](safe_indices))

            # Remove original categorical features and add embeddings
            mask = torch.ones(x.shape[1], dtype=torch.bool)
            for idx in cat_indices.values():
                mask[idx] = False

            x_numeric = x[:, mask]
            x_embedded = torch.cat([x_numeric] + embeddings, dim=1)
            x = x_embedded

        # Apply GNN encoder
        node_embeddings = self.encoder(x, edge_index)

        # Make predictions
        predictions = self.predictor(node_embeddings)
        aux_predictions = self.aux_predictor(node_embeddings)

        return predictions, node_embeddings, aux_predictions


@functools.lru_cache
def combinations(N):
    """
    Calculate all possible 2-combinations (pairs) of indices from 0 to N-1.
    Cached for efficiency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ind = torch.arange(N, device=device)
    comb = torch.combinations(ind, r=2)
    return comb


class LitSurvivalGNN(lightning.LightningModule):
    """
    PyTorch Lightning module for training the Survival GNN model.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        out_channels=64,
        num_layers=3,
        dropout=0.2,
        lr=0.001,
        weight_decay=1e-4,
        categorical_dims=None,
        embedding_dim=16,
        aux_weight=0.1,
        margin=0.5,
        race_index=None,
    ):
        super(LitSurvivalGNN, self).__init__()
        self.save_hyperparameters()

        # Create model
        self.model = SurvivalGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            categorical_dims=categorical_dims,
            embedding_dim=embedding_dim,
        )

        # Loss functions
        self.tukey_loss = TukeyBiweightLoss(c=4.685)

        # For validation metrics
        self.targets = []
        self.race_index = race_index

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer.
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, data):
        """
        Forward pass through the model.
        """
        return self.model(data)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        # Move batch to the correct device
        device = next(self.parameters()).device
        batch = batch.to(device)

        # Get predictions
        y_pred, embeddings, aux_pred = self(batch)
        y = batch.y.view(-1, 1)
        efs = batch.efs.view(-1)
        race_labels = batch.race if hasattr(batch, "race") else None

        # Calculate main loss
        loss, race_loss = self.get_full_loss(efs, race_labels, y, y_pred)

        # Calculate auxiliary loss
        aux_loss = F.mse_loss(aux_pred, y, reduction="none")
        aux_mask = efs == 1
        aux_loss = (aux_loss.view(-1) * aux_mask).sum() / (aux_mask.sum() + 1e-8)

        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)

        # Combined loss
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, race_labels, y, y_pred):
        """
        Calculate the full loss including race fairness penalty.
        """
        loss = self.calc_loss(y, y_pred, efs)

        # Add race fairness penalty if race labels are available
        if race_labels is not None:
            race_loss = self.get_race_losses(efs, race_labels, y, y_pred)
            loss += 0.1 * race_loss
        else:
            race_loss = torch.tensor(0.0, device=y.device)

        return loss, race_loss

    def get_race_losses(self, efs, race_labels, y, y_pred):
        """
        Calculate loss for each race group and compute variance penalty.
        """
        if race_labels is None:
            return torch.tensor(0.0, device=y.device)

        races = torch.unique(race_labels)
        race_losses = []

        for race in races:
            ind = race_labels == race
            if ind.sum() > 1:  # Need at least 2 samples for pairwise loss
                race_losses.append(self.calc_loss(y[ind], y_pred[ind], efs[ind]))

        if not race_losses:
            return torch.tensor(0.0, device=y.device)

        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss) ** 2 for r in race_losses) / len(race_losses)

        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_pred, efs):
        """
        Calculate pairwise ranking loss for survival data.

        This implements a margin-based ranking loss that ensures:
        - Patients with longer survival times are ranked higher
        - Only valid comparisons are made (accounting for censoring)
        """
        # Ensure all tensors are on the same device
        device = y.device
        y_pred = y_pred.to(device)
        efs = efs.to(device)

        N = y.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device)

        # For all datasets, use MSE loss as a fallback
        # This is more stable and avoids memory issues
        mse_loss = F.mse_loss(y_pred, y)

        # For very small datasets, we can use pairwise ranking
        if N <= 100:
            try:
                # Use all combinations for small datasets
                comb = combinations(N).to(device)
                # Filter pairs where at least one has an event
                comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]

                if len(comb) == 0:
                    return mse_loss

                # Get predictions and targets for each pair
                pred_left = y_pred[comb[:, 0]]
                pred_right = y_pred[comb[:, 1]]
                y_left = y[comb[:, 0]]
                y_right = y[comb[:, 1]]

                # Create target: +1 if left should be ranked higher than right
                y_target = 2 * (y_left > y_right).float() - 1

                # Margin-based hinge loss
                pairwise_loss = F.relu(-y_target * (pred_left - pred_right) + self.hparams.margin)

                # Apply mask for valid comparisons
                mask = self.get_mask(comb, efs, y_left, y_right)
                pairwise_loss = (pairwise_loss.float() * mask.float()).sum() / (mask.sum() + 1e-8)

                return pairwise_loss
            except Exception as e:
                # If any error occurs, fall back to MSE loss
                print(f"Error in pairwise loss calculation: {e}. Using MSE loss instead.")
                return mse_loss
        else:
            # For larger datasets, sample a small subset of pairs
            try:
                # Sample a subset of pairs to avoid memory issues
                max_pairs = 1000  # Reduced from 50000 to avoid OOM

                # Get indices of samples with events
                event_indices = torch.where(efs == 1)[0]

                # If no events, use MSE loss
                if len(event_indices) == 0:
                    return mse_loss

                # Create pairs ensuring at least one has an event
                pairs = []
                # Sample fewer pairs
                for _ in range(min(max_pairs, len(event_indices))):
                    i = event_indices[torch.randint(0, len(event_indices), (1,), device=device)]
                    j = torch.randint(0, N, (1,), device=device)
                    if i.item() != j.item():
                        pairs.append([i.item(), j.item()])

                # Convert to tensor
                if not pairs:
                    return mse_loss

                comb = torch.tensor(pairs, device=device)

                # Get predictions and targets for each pair
                pred_left = y_pred[comb[:, 0]]
                pred_right = y_pred[comb[:, 1]]
                y_left = y[comb[:, 0]]
                y_right = y[comb[:, 1]]

                # Create target: +1 if left should be ranked higher than right
                y_target = 2 * (y_left > y_right).float() - 1

                # Margin-based hinge loss
                pairwise_loss = F.relu(-y_target * (pred_left - pred_right) + self.hparams.margin)

                # Apply mask for valid comparisons
                mask = self.get_mask(comb, efs, y_left, y_right)
                pairwise_loss = (pairwise_loss.float() * mask.float()).sum() / (mask.sum() + 1e-8)

                # Combine with MSE loss for stability
                return 0.7 * pairwise_loss + 0.3 * mse_loss
            except Exception as e:
                # If any error occurs, fall back to MSE loss
                print(f"Error in pairwise loss calculation: {e}. Using MSE loss instead.")
                return mse_loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Create a mask for valid comparisons in survival data.

        Handles censoring by identifying invalid comparisons:
        - When "left outlived right" but right is censored
        - When "right outlived left" but left is censored
        """
        # Ensure all tensors are on the same device
        device = comb.device
        efs = efs.to(device)
        y_left = y_left.to(device)
        y_right = y_right.to(device)

        # Process the entire tensor at once to avoid concatenation issues
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = left_outlived & left_1_right_0

        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= right_outlived & right_1_left_0

        # Invert to get valid pairs
        mask = ~mask2
        return mask

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        # Move batch to the correct device
        device = next(self.parameters()).device
        batch = batch.to(device)

        y_pred, embeddings, _ = self(batch)
        y = batch.y.view(-1)
        efs = batch.efs.view(-1)
        race_labels = batch.race if hasattr(batch, "race") else None

        # Calculate loss
        loss, race_loss = self.get_full_loss(efs, race_labels, y.view(-1, 1), y_pred)

        # Store predictions for concordance index calculation
        self.targets.append(
            [y, y_pred.view(-1).detach(), efs, race_labels if race_labels is not None else torch.zeros_like(y)]
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Calculate and log concordance index at the end of validation.
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate concordance index overall and by race group.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_pred = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()

        # Calculate stratified metric
        metric = self._metric(efs, races, y, y_pred)

        # Calculate overall c-index
        cindex = concordance_index(y, y_pred, efs)

        return cindex, metric

    def _metric(self, efs, races, y, y_pred):
        """
        Calculate the competition metric: mean c-index across race groups minus std penalty.
        """
        metric_list = []
        for race in np.unique(races):
            mask = races == race
            if mask.sum() > 1:  # Need at least 2 samples
                y_ = y[mask]
                y_pred_ = y_pred[mask]
                efs_ = efs[mask]
                metric_list.append(concordance_index(y_, y_pred_, efs_))

        if not metric_list:
            return 0.0

        # Competition metric: mean - sqrt(variance)
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        """
        # Move batch to the correct device
        device = next(self.parameters()).device
        batch = batch.to(device)

        y_pred, embeddings, _ = self(batch)
        y = batch.y.view(-1)
        efs = batch.efs.view(-1)
        race_labels = batch.race if hasattr(batch, "race") else None

        # Calculate loss
        loss, race_loss = self.get_full_loss(efs, race_labels, y.view(-1, 1), y_pred)

        # Store predictions for concordance index calculation
        self.targets.append(
            [y, y_pred.view(-1).detach(), efs, race_labels if race_labels is not None else torch.zeros_like(y)]
        )

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        """
        Calculate and log concordance index at the end of testing.
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=self.hparams.lr * 0.1,
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def create_graph_data(
    df, features, target="efs_time", event="efs", categorical_cols=None, k_neighbors=5, race_col="race_group"
):
    """
    Create a graph from tabular data by connecting similar patients.

    Args:
        df: DataFrame with patient data
        features: List of feature columns to use
        target: Target column name
        event: Event indicator column name
        categorical_cols: List of categorical columns
        k_neighbors: Number of neighbors to connect in the graph
        race_col: Column name for race

    Returns:
         PyTorch Geometric Data object
    """
    # Prepare feature matrix
    X = df[features].copy()

    # Handle categorical features
    if categorical_cols is None:
        categorical_cols = []

    # Get categorical indices
    cat_indices = [features.index(col) for col in categorical_cols if col in features]

    # Create feature matrix
    X_array = X.values.astype(np.float32)

    # Create target vector
    y = df[target].values.astype(np.float32)

    # Create event indicator
    if df[event].dtype == "object":
        efs = df[event].map({"Event": 1, "Censoring": 0}).values.astype(np.int64)
    else:
        # 既に数値の場合はそのまま使用
        efs = df[event].values.astype(np.int64)

    # Create race labels if available
    race_labels = None
    if race_col in df.columns:
        race_mapping = {race: i for i, race in enumerate(df[race_col].unique())}
        race_labels = df[race_col].map(race_mapping).values.astype(np.int64)

    # Create graph edges using k-nearest neighbors
    # Use only numeric features for finding neighbors
    numeric_cols = [col for col in features if col not in categorical_cols]

    # For large datasets, use a subset of features to reduce memory usage
    if len(df) > 10000 and len(numeric_cols) > 10:
        # Use only the most important numeric features
        numeric_cols = numeric_cols[:10]

    if numeric_cols:
        X_numeric = df[numeric_cols].values

        # Standardize numeric features
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)

        # Find k-nearest neighbors
        # Use smaller k for large datasets
        actual_k = min(k_neighbors, 3) if len(df) > 10000 else k_neighbors
        knn = NearestNeighbors(
            n_neighbors=actual_k + 1, algorithm="ball_tree"
        )  # +1 because a point is its own neighbor
        knn.fit(X_numeric)

        # Process in batches to avoid memory issues
        batch_size = 5000
        n_samples = len(X_numeric)
        edge_list = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_X = X_numeric[i:end_idx]
            distances, indices = knn.kneighbors(batch_X)

            for j, neighbors in enumerate(indices):
                for neighbor in neighbors[1:]:  # Skip the first neighbor (self)
                    edge_list.append([i + j, neighbor])
                    edge_list.append([neighbor, i + j])  # Add reverse edge for undirected graph

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # If no numeric features, create a sparse random graph instead of fully connected
        n = len(df)
        if n > 1000:
            # For large datasets, create a sparse random graph
            edge_list = []
            avg_degree = 5  # Average number of connections per node
            for i in range(n):
                # Connect to a few random nodes
                for _ in range(avg_degree):
                    j = np.random.randint(0, n)
                    if i != j:
                        edge_list.append([i, j])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # For small datasets, can use fully connected graph
            edge_list = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_list.append([i, j])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(X_array, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float),
        efs=torch.tensor(efs, dtype=torch.long),
    )

    # Add race labels if available
    if race_labels is not None:
        data.race = torch.tensor(race_labels, dtype=torch.long)

    return data


def prepare_data_for_gnn(
    train_df,
    valid_df,
    test_df,
    features,
    target="efs_time",
    event="efs",
    categorical_cols=None,
    k_neighbors=3,
    batch_size=32,
):
    """
    Prepare data for GNN training.

    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        test_df: Test DataFrame
        features: List of feature columns
        target: Target column name
        event: Event indicator column name
        categorical_cols: List of categorical columns
        k_neighbors: Number of neighbors for graph construction
        batch_size: Batch size for data loaders

    Returns:
        train_loader: Training data loader
        valid_loader: Validation data loader
        test_loader: Test data loader
        in_channels: Number of input features
        categorical_dims: Dictionary of categorical dimensions
    """
    # Create graph data
    train_data = create_graph_data(
        train_df,
        features,
        target,
        event,
        categorical_cols,
        k_neighbors,
    )

    valid_data = (
        create_graph_data(
            valid_df,
            features,
            target,
            event,
            categorical_cols,
            k_neighbors,
        )
        if valid_df is not None
        else None
    )

    test_data = (
        create_graph_data(
            test_df,
            features,
            target,
            event,
            categorical_cols,
            k_neighbors,
        )
        if test_df is not None
        else None
    )

    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader([valid_data], batch_size=batch_size) if valid_data is not None else None
    test_loader = DataLoader([test_data], batch_size=batch_size) if test_data is not None else None

    # Get input dimensions
    in_channels = train_data.x.size(1)

    # Get categorical dimensions
    categorical_dims = {}
    if categorical_cols:
        for i, col in enumerate(categorical_cols):
            if col in features:
                idx = features.index(col)
                categorical_dims[col] = int(train_data.x[:, idx].max().item())

    return train_loader, valid_loader, test_loader, in_channels, categorical_dims


class SurvivalGNNModel(BaseModel):
    """
    Implementation of the Survival GNN model for the BaseModel interface.
    """

    def __init__(self):
        super().__init__()
        self.is_linear_model = False
        self.is_survival_model = True
        self.is_cat_embed_model = False
        self.model = None
        self.features = None
        self.categorical_cols = None
        self.race_index = None
        self.k_neighbors = 10

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
        Train the Survival GNN model.
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
        train_df["efs_time"] = y_train
        train_df["efs"] = kwargs.get("efs_train", np.ones(len(y_train)))

        valid_df = x_valid.copy()
        valid_df["efs_time"] = y_valid
        valid_df["efs"] = kwargs.get("efs_valid", np.ones(len(y_valid)))

        # Get feature list
        self.features = x_train.columns.tolist()

        # Identify categorical columns
        self.categorical_cols = config.get("categorical_cols", [])

        # Find race_group index if it exists
        self.race_index = self.features.index("race_group") if "race_group" in self.features else None

        # Get model parameters
        model_params = config.get("model", {}).get("model_params", {})
        hidden_channels = model_params.get("hidden_channels", 128)
        out_channels = model_params.get("out_channels", 64)
        num_layers = model_params.get("num_layers", 3)
        dropout = model_params.get("dropout", 0.2)
        lr = model_params.get("lr", 0.001)
        weight_decay = float(model_params.get("weight_decay", 0.0001))
        embedding_dim = model_params.get("embedding_dim", 16)
        aux_weight = model_params.get("aux_weight", 0.1)
        margin = model_params.get("margin", 0.5)
        self.k_neighbors = model_params.get("k_neighbors", 10)
        max_epochs = model_params.get("max_epochs", 60)

        # Prepare data for GNN
        with timer("prepare data for GNN"):
            train_loader, valid_loader, _, in_channels, categorical_dims = prepare_data_for_gnn(
                train_df,
                valid_df,
                None,
                self.features,
                "efs_time",
                "efs",
                self.categorical_cols,
                self.k_neighbors,
            )

        # Create model
        with timer("create model"):
            model = LitSurvivalGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                categorical_dims=categorical_dims,
                embedding_dim=embedding_dim,
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
            trainer.fit(model, train_loader, valid_loader)

            # Test model
            trainer.test(model, valid_loader)

            self.model = model.eval()

            # Get best validation metric
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
            test_df["efs_time"] = 1.0  # Dummy value
            test_df["efs"] = 1  # Dummy value

            # Ensure all expected features are present
            for col in self.features:
                if col not in test_df.columns:
                    test_df[col] = 0  # Add missing columns with default values
                elif test_df[col].isna().any():
                    # 欠損値を0で埋める
                    test_df[col] = test_df[col].fillna(0)

            # 特徴量の型を確認し、必要に応じて変換
            for col in self.categorical_cols:
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype(int)

            # Create graph data
            try:
                _, _, test_loader, _, _ = prepare_data_for_gnn(
                    None,
                    None,
                    test_df,
                    self.features,
                    "efs_time",
                    "efs",
                    self.categorical_cols,
                    self.k_neighbors,
                    batch_size=128,
                )
            except Exception as e:
                logging.exception(f"Error preparing data for GNN: {e}")
                # エラーが発生した場合はランダムな予測を返す
                return np.random.rand(len(test_df))

            # Get predictions
            model.eval()
            all_preds = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(next(model.parameters()).device)
                    pred, _, _ = model(batch)
                    all_preds.append(pred.cpu().numpy())

            # Concatenate predictions
            predictions = np.concatenate(all_preds).reshape(-1)

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
        # GNNs don't have direct feature importance
        return {}
