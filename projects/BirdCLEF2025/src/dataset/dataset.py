import random
from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import polars as pl
from augmentations import get_default_transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        df: pl.DataFrame,
        labels: list[str],
        mel_spec_path: Path,
        transform=None,
    ):
        self.mode = mode
        self.df = df

        self.file_to_slices: dict[str, list[Path]] = {}
        self.file_to_label: dict[str, int] = {}

        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.labels = np.identity(len(labels), dtype=np.float32)[
            [self.label_to_idx[label] for label in df["primary_label"].to_list()]
        ]

        self.filenames = df["filename"].to_list()
        primary_labels = df["primary_label"].to_list()
        for filename, primary_label in zip(self.filenames, primary_labels):
            base_name = Path(filename).stem
            slices = sorted((mel_spec_path / base_name).glob("*.npy"))

            self.file_to_slices[filename] = slices
            self.file_to_label[filename] = self.label_to_idx[primary_label]

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        slice_path = random.choice(self.file_to_slices[filename])

        mel_spec = np.load(slice_path)

        label = self.labels[idx]

        if self.transform:
            # Albumentations expects (H, W) or (H, W, C) format
            mel_spec = self.transform(image=mel_spec)["image"]
            # Add channel dimension after augmentation for PyTorch
            mel_spec = mel_spec[np.newaxis, ...]  # (H, W) -> (1, H, W)
        return mel_spec, label


class BirdCLEFDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        labels: list[str],
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self.labels = labels
        self._cfg = cfg

        self.mel_spec_path = Path(self._cfg.data_path) / "train_melspectrograms"
        self.transform = get_default_transforms()

    def __create_dataset(self, train=True):
        return (
            BirdCLEFDataset(
                mode="train",
                df=self._train_df,
                labels=self.labels,
                mel_spec_path=self.mel_spec_path,
                transform=self.transform["albu_train"],
            )
            if train
            else BirdCLEFDataset(
                mode="val",
                df=self._val_df,
                labels=self.labels,
                mel_spec_path=self.mel_spec_path,
                transform=self.transform["albu_val"],
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class BirdCLEFInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df: pl.DataFrame,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg
        self.mel_spec_path = Path(self._cfg.data_path) / "test_mel_specs"

    def predict_dataloader(self):
        dataset = BirdCLEFDataset(
            mode="test",
            df=self._test_df,
            mel_spec_path=self.mel_spec_path,
            transform=A.NoOp(always_apply=True),
        )
        return DataLoader(dataset, **self._cfg.val_loader)
