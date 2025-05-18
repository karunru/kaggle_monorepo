from pathlib import Path
import random
from typing import Literal, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_default_transforms


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        mel_spec_dir: Union[str, Path],
        transform=None,
    ):
        """
        Dataset for BirdCLEF that loads mel spectrograms
        
        Args:
            csv_path: Path to train.csv
            mel_spec_dir: Directory containing the mel spectrogram .npy files
            transform: Optional transforms to apply to the mel spectrograms
        """
        self.df = pd.read_csv(csv_path)
        self.mel_spec_dir = Path(mel_spec_dir)
        self.transform = transform
        
        # Create a dictionary mapping each filename to its slices
        self.file_to_slices: Dict[str, List[Path]] = {}
        self.file_to_label: Dict[str, int] = {}
        
        # Get unique class labels and create a mapping
        self.labels = sorted(self.df['primary_label'].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        
        # Find all available slices for each filename
        for _, row in self.df.iterrows():
            filename = row['filename']
            primary_label = row['primary_label']
            
            # Get all slice paths for this filename (excluding file extension)
            base_name = Path(filename).stem
            slices = list(self.mel_spec_dir.glob(f"{base_name}_*.npy"))
            
            if slices:
                self.file_to_slices[filename] = slices
                self.file_to_label[filename] = self.label_to_idx[primary_label]
        
        # Create list of filenames that have valid slices
        self.valid_filenames = list(self.file_to_slices.keys())
    
    def __len__(self):
        return len(self.valid_filenames)
    
    def __getitem__(self, idx):
        """
        Get a random slice for the filename at idx
        
        Returns:
            mel_spec: Mel spectrogram as a tensor
            label: Class label as a tensor
        """
        filename = self.valid_filenames[idx]
        
        # Randomly select one of the slices for this file
        slice_path = random.choice(self.file_to_slices[filename])
        
        # Load the mel spectrogram from the .npy file
        mel_spec = np.load(slice_path)
        
        # Get the label for this file
        label = self.file_to_label[filename]
        
        # Apply transforms if available
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        # Convert to tensor if not already
        if not isinstance(mel_spec, torch.Tensor):
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
            
        # If the mel_spec is 2D, add a channel dimension
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension (1, H, W)
            
        return mel_spec, torch.tensor(label, dtype=torch.long)


class HMSDataset(Dataset):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        df: pd.DataFrame,
        target_col: str,
        eeg_spectrogram_path: Path,
        transform=None,
    ):
        self.mode = mode
        self.df = df

        self.target_col = target_col

        self.eeg_spectrogram = np.load(
            eeg_spectrogram_path,
            allow_pickle=True,
        ).item()

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[idx]

        r = (
            0
            if self.mode == "test"
            else int(
                (
                    row["spectrogram_label_offset_seconds_min"]
                    + row["spectrogram_label_offset_seconds_max"]
                )
                // 4
            )
        )

        if self.mode != "test":
            y = row[self.target_cols].values.astype(np.float32)

        img = self.eeg_spectrogram[row["eeg_id"]]
        X[:, :, 4:] = img

        for region in range(4):
            img = self.spectrogram[row["spectrogram_id"]][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        return self.transform(image=X)["image"], y


class HMSDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_cols: list[str],
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self.target_cols = target_cols
        self._cfg = cfg

        self.eeg_spectrograms_path = (
            Path(self._cfg.data_path)
            / f"train_eeg_spectrograms_{self._cfg.eeg_spec}/eeg_specs.npy"
        )
        self.spectrograms_path = (
            Path(self._cfg.data_path) / f"train_spectrograms/all_spectrograms.npy"
        )

        self.transform = get_default_transforms()

    def __create_dataset(self, train=True):
        return (
            HMSDataset(
                mode="train",
                df=self._train_df,
                target_cols=self.target_cols,
                eeg_spectrogram_path=self.eeg_spectrograms_path,
                spectrogram_path=self.spectrograms_path,
                transform=self.transform["albu_train"],
            )
            if train
            else HMSDataset(
                mode="val",
                df=self._val_df,
                target_cols=self.target_cols,
                eeg_spectrogram_path=self.eeg_spectrograms_path,
                spectrogram_path=self.spectrograms_path,
                transform=self.transform["albu_val"],
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class HMSInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        target_cols: list[str],
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self.target_cols = target_cols
        self._cfg = cfg
        self.eeg_spectrograms_path = (
            Path(self._cfg.data_path)
            / f"test_eeg_spectrograms_{self._cfg.eeg_spec}/eeg_specs.npy"
        )
        self.spectrograms_path = (
            Path(self._cfg.data_path) / f"test_spectrograms/all_spectrograms.npy"
        )
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = HMSDataset(
            mode="test",
            df=self._test_df,
            target_cols=self.target_cols,
            eeg_spectrogram_path=self.eeg_spectrograms_path,
            spectrogram_path=self.spectrograms_path,
            transform=A.NoOp(always_apply=True),
        )
        return DataLoader(dataset, **self._cfg.val_loader)
