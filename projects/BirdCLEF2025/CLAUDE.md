# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Data Preprocessing
```bash
# 1. Download competition data
kaggle competitions download -c birdclef-2025 -p ./data/
unzip ./data/birdclef-2025.zip -d ./data/
rm ./data/birdclef-2025.zip

# 2. Create mel-spectrograms from audio files
# Run notebook: notebooks/001_create_melspectrograms.ipynb
# This converts audio to mel-spectrograms with 5-second windows, 2.5-second shift
```

### Training
```bash
# Run training with base configuration
uv run python train_pl.py --config config/base.yaml

# Training with custom parameters
uv run python train_pl.py --config config/base.yaml model.backbone=resnet50 train_loader.batch_size=4

# Training specific folds
uv run python train_pl.py --config config/base.yaml train_folds=[0,1,2]
```

### Key Training Parameters
- Config uses OmegaConf dot notation for overrides
- Experiment numbering is automatic (exp_001, exp_002, etc.)
- Failed experiments are automatically cleaned up
- Outputs: weights/exp_XXX/, preds/exp_XXX/, config/exp_XXX.yaml

## High-Level Architecture

### Competition Approach
This is a bird sound classification competition (BirdCLEF 2025) that converts audio to images:
1. **Audio → Mel-spectrogram**: Preprocesses raw audio files into normalized mel-spectrograms
2. **Image Classification**: Uses computer vision models on spectrograms
3. **Multi-label Classification**: Predicts bird species from 5-second audio clips

### Key Components

#### Data Pipeline
1. **Audio Processing** (notebooks/001_create_melspectrograms.ipynb):
   - Parameters: SR=32000, N_FFT=1024, HOP_LENGTH=512, N_MELS=128
   - 5-second windows with 2.5-second overlap
   - Saves as normalized float32 numpy arrays in `data/train_melspectrograms/{stem}/{slice_id}.npy`

2. **Dataset** (src/dataset/dataset.py):
   - PyTorch Dataset loading preprocessed mel-spectrograms
   - Random slice selection during training
   - Supports train/val/test modes

3. **Data Augmentation** (src/augmentations/):
   - InAugment: Custom spectrogram augmentation
   - XY masking on spectrograms
   - ResizeMix for strong augmentation
   - Albumentations for standard transforms

#### Model Architecture (src/models/model.py)
- Uses timm (PyTorch Image Models) for backbone
- Single-channel input (grayscale spectrograms)
- Configurable pooling (avgmax by default)
- Softmax output for species classification
- Embedding extraction support

#### Training Pipeline (src/train_pl.py)
1. **K-Fold Cross-Validation**:
   - Stratified splits on primary_label
   - OOF predictions and embeddings tracking
   - Per-fold and overall KL divergence scoring

2. **PyTorch Lightning Integration**:
   - Mixed precision training (bf16)
   - Gradient accumulation (effective batch size = 16)
   - Early stopping (patience=4)
   - Stochastic Weight Averaging
   - Learning rate scheduling (cosine with warmup)

3. **Experiment Management**:
   - Automatic experiment numbering
   - Weights & Biases logging
   - Config versioning
   - Failed experiment cleanup

#### Configuration System (src/config/base.yaml)
- Hierarchical YAML configuration
- Runtime overrides via CLI
- Key settings:
  - 5-fold CV
  - 20 epochs
  - BCE loss
  - AdamW optimizer
  - Cosine LR schedule

### Directory Structure
```
data/
  train_audio/         # Raw audio files
  train_melspectrograms/  # Preprocessed spectrograms
  train.csv            # Labels and metadata
  
src/
  augmentations/       # Data augmentation
  config/              # Configuration files
  dataset/             # Dataset and DataModule
  models/              # Model architecture
  sampling/            # Data sampling strategies
  utils/               # Utilities
  validation/          # CV strategies
  train_pl.py          # Main training script
  
weights/exp_XXX/       # Model checkpoints
preds/exp_XXX/         # Predictions
config/exp_XXX.yaml    # Experiment configs
```

### Important Patterns
- Uses Polars for fast dataframe operations
- Explicit memory management between folds
- torch.compile() for model optimization
- Anomaly detection enabled for debugging
- KL divergence as primary metric