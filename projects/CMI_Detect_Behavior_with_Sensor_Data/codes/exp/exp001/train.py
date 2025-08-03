"""Training script for CMI Detect Behavior with Sensor Data - Experiment 001."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import hydra
import numpy as np
import pandas as pd
import torch
from cv_strategy import get_validation_splits
from dataset import create_dataloaders
from loss import get_loss_function
from metrics import MetricsTracker, calculate_competition_metric
from model import CMISqueezeformer
from omegaconf import DictConfig, OmegaConf

# Import our modules
from preprocessor import DataPreprocessor, process_sequences
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        sequences = batch["sequence"].to(device)
        multiclass_labels = batch["multiclass_label"].squeeze().to(device)
        binary_labels = batch["binary_label"].squeeze().to(device)

        # Forward pass
        multiclass_logits, binary_logits = model(sequences)

        # Compute loss
        loss, mc_loss, bin_loss = criterion(multiclass_logits, binary_logits, multiclass_labels, binary_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} (MC: {mc_loss.item():.4f}, Binary: {bin_loss.item():.4f})"
            )

    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_multiclass_preds = []
    all_binary_preds = []
    all_multiclass_true = []
    all_binary_true = []

    with torch.no_grad():
        for batch in val_loader:
            sequences = batch["sequence"].to(device)
            multiclass_labels = batch["multiclass_label"].squeeze().to(device)
            binary_labels = batch["binary_label"].squeeze().to(device)

            # Forward pass
            multiclass_logits, binary_logits = model(sequences)

            # Compute loss
            loss, _, _ = criterion(multiclass_logits, binary_logits, multiclass_labels, binary_labels)
            total_loss += loss.item()

            # Collect predictions
            multiclass_preds = torch.argmax(multiclass_logits, dim=1)
            binary_preds = (torch.sigmoid(binary_logits.squeeze()) > 0.5).long()

            all_multiclass_preds.extend(multiclass_preds.cpu().numpy())
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_multiclass_true.extend(multiclass_labels.cpu().numpy())
            all_binary_true.extend(binary_labels.cpu().numpy())

    # Calculate metrics
    val_loss = total_loss / len(val_loader)

    multiclass_preds = np.array(all_multiclass_preds)
    binary_preds = np.array(all_binary_preds)
    multiclass_true = np.array(all_multiclass_true)
    binary_true = np.array(all_binary_true)

    competition_score, binary_f1, macro_f1 = calculate_competition_metric(
        multiclass_preds, binary_preds, multiclass_true, binary_true, label_encoder
    )

    return val_loss, competition_score, binary_f1, macro_f1


def train_fold(
    fold: int,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    train_subjects: np.ndarray,
    val_sequences: np.ndarray,
    val_labels: np.ndarray,
    val_subjects: np.ndarray,
    label_encoder: LabelEncoder,
    config: DictConfig,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train a single fold."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING FOLD {fold + 1}")
    print(f"{'=' * 60}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_sequences,
        train_labels,
        train_subjects,
        val_sequences,
        val_labels,
        val_subjects,
        label_encoder,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    input_dim = train_sequences.shape[2]  # Number of features
    model = CMISqueezeformer(
        input_dim=input_dim,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        num_classes=len(label_encoder.classes_),
        dropout=config.model.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    train_labels_tensor = torch.from_numpy(label_encoder.transform(train_labels))
    criterion = get_loss_function(config.loss, train_labels_tensor)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    if config.training.scheduler.type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.scheduler.min_lr,
        )
    elif config.training.scheduler.type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.training.scheduler.factor,
            patience=config.training.scheduler.patience,
            verbose=True,
        )
    else:
        scheduler = None

    # Training loop
    best_score = 0.0
    best_model_state = None
    patience_counter = 0
    metrics_tracker = MetricsTracker()

    for epoch in range(config.training.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)

        # Validate
        val_loss, val_score, val_binary_f1, val_macro_f1 = validate_epoch(
            model, val_loader, criterion, label_encoder, device
        )

        # Calculate training score for tracking
        train_score = 0.0  # Could implement this if needed

        # Update metrics tracker
        metrics_tracker.update(epoch + 1, train_loss, val_loss, train_score, val_score, val_binary_f1, val_macro_f1)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        # Early stopping and best model saving
        if val_score > best_score:
            best_score = val_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save best model for this fold
            model_save_path = Path(config.training.output_dir) / f"fold_{fold}_best_model.pth"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "config": OmegaConf.to_yaml(config),
                    "label_encoder": label_encoder,
                    "best_score": best_score,
                    "epoch": epoch + 1,
                },
                model_save_path,
            )

        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    metrics_tracker.print_summary()

    # Final validation with detailed metrics
    print(f"\n🎯 FINAL VALIDATION RESULTS - FOLD {fold + 1}")
    val_loss, val_score, val_binary_f1, val_macro_f1 = validate_epoch(
        model, val_loader, criterion, label_encoder, device
    )

    fold_results = {
        "fold": fold,
        "best_score": best_score,
        "final_score": val_score,
        "binary_f1": val_binary_f1,
        "macro_f1": val_macro_f1,
        "model_path": str(model_save_path),
    }

    return model, fold_results


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    print("🚀 Starting CMI Sensor Data Training")
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Set random seed
    set_seed(config.training.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv(config.data.train_path)
    print(f"Loaded {len(train_df)} training samples")

    # Data preprocessing
    print("🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    sequences, labels, subjects = process_sequences(train_df, preprocessor)

    print(f"Processed {len(sequences)} sequences")
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Unique labels: {len(np.unique(labels))}")
    print(f"Unique subjects: {len(np.unique(subjects))}")

    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    print(f"Label classes: {list(label_encoder.classes_)}")

    # Cross-validation splits
    print("📊 Creating CV splits...")
    cv_splits = get_validation_splits(train_df, config)

    # Training loop
    fold_results = []
    all_models = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits[: config.training.n_folds]):
        # Split data
        train_sequences = sequences[train_idx]
        train_labels = labels[train_idx]
        train_subjects = subjects[train_idx]

        val_sequences = sequences[val_idx]
        val_labels = labels[val_idx]
        val_subjects = subjects[val_idx]

        # Train fold
        model, results = train_fold(
            fold,
            train_sequences,
            train_labels,
            train_subjects,
            val_sequences,
            val_labels,
            val_subjects,
            label_encoder,
            config,
            device,
        )

        fold_results.append(results)
        all_models.append(model)

        print(f"Fold {fold + 1} Score: {results['best_score']:.4f}")

    # Final results
    print(f"\n{'=' * 60}")
    print("🏆 CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")

    scores = [result["best_score"] for result in fold_results]
    binary_f1s = [result["binary_f1"] for result in fold_results]
    macro_f1s = [result["macro_f1"] for result in fold_results]

    print(f"CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Binary F1: {np.mean(binary_f1s):.4f} ± {np.std(binary_f1s):.4f}")
    print(f"Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_path = Path(config.training.output_dir) / "cv_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    print("✅ Training completed!")


if __name__ == "__main__":
    main()
