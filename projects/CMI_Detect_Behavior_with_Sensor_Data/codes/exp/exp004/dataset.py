"""IMU専用データセットモジュール - exp004用（欠損値をattention_maskで処理 + Polars最適化）."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning as lightning
import torch
from scipy import interpolate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
# Local imports
from config import Config
from utils.timer import timer_decorator
from validation.factory import get_validation


def dynamic_collate_fn(batch: list[dict], percentile_max_length: float = 0.95) -> dict:
    """
    動的パディング用のcollate関数.

    Args:
        batch: バッチデータのリスト
        percentile_max_length: 最大長の決定に使用するパーセンタイル

    Returns:
        パディング済みバッチデータ
    """
    # シーケンス長を取得
    lengths = [item["imu"].size(1) for item in batch]
    max_length = int(np.percentile(lengths, percentile_max_length * 100))

    # バッチサイズ取得
    batch_size = len(batch)
    features = batch[0]["imu"].size(0)

    # テンソルの初期化
    imu_batch = torch.zeros(batch_size, features, max_length)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    multiclass_labels = torch.zeros(batch_size, dtype=torch.long)
    binary_labels = torch.zeros(batch_size, dtype=torch.float32)
    sequence_ids = []
    gestures = []

    # データをパディング
    for i, item in enumerate(batch):
        imu_len = item["imu"].size(1)
        actual_len = min(imu_len, max_length)

        imu_batch[i, :, :actual_len] = item["imu"][:, :actual_len]
        attention_mask[i, :actual_len] = item.get("attention_mask", torch.ones(actual_len, dtype=torch.bool))[
            :actual_len
        ]

        multiclass_labels[i] = item["multiclass_label"]
        binary_labels[i] = item["binary_label"]
        sequence_ids.append(item["sequence_id"])
        gestures.append(item["gesture"])

        lengths[i] = actual_len

    return {
        "imu": imu_batch,
        "attention_mask": attention_mask,
        "multiclass_label": multiclass_labels,
        "binary_label": binary_labels,
        "sequence_id": sequence_ids,
        "gesture": gestures,
        "max_length": max_length,  # デバッグ用
        "original_lengths": lengths,  # デバッグ用
    }


class IMUDataset(Dataset):
    """IMU専用データセット（acc_x/y/z + rot_w/x/y/z）."""

    def __init__(
        self,
        df: pl.DataFrame,
        target_sequence_length: int = 200,
        augment: bool = False,
        augmentation_config: dict | None = None,
        use_dynamic_padding: bool = False,
    ):
        """
        初期化.

        Args:
            df: Polarsデータフレーム
            target_sequence_length: 目標シーケンス長（固定長モード時）
            augment: データ拡張フラグ
            augmentation_config: データ拡張設定
            use_dynamic_padding: 動的パディング使用フラグ
        """
        self.df = df
        self.target_sequence_length = target_sequence_length
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.use_dynamic_padding = use_dynamic_padding

        # IMU列の定義
        self.imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

        # ユニークシーケンスIDの取得
        print(f"Loading dataset with {len(df)} rows...")
        self.sequence_ids = df.get_column("sequence_id").unique().to_list()
        print(f"Found {len(self.sequence_ids)} unique sequences")

        # ジェスチャー情報の取得
        gesture_df = df.group_by("sequence_id").agg(pl.col("gesture").first()).sort("sequence_id")
        self.gestures = dict(zip(gesture_df.get_column("sequence_id"), gesture_df.get_column("gesture")))

        # ラベルマッピングの作成
        unique_gestures = sorted(df.get_column("gesture").unique().to_list())
        self.gesture_to_id = {gesture: i for i, gesture in enumerate(unique_gestures)}
        self.num_classes = len(unique_gestures)

        # バイナリ分類用のターゲットジェスチャー定義
        target_gestures = {
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        }
        self.is_target_gesture = {gesture: 1 if gesture in target_gestures else 0 for gesture in unique_gestures}

        print("Preprocessing data...")
        # データ前処理
        self._preprocess_data()
        print("Data preprocessing completed")

    @timer_decorator("IMUDataset._preprocess_data")
    def _preprocess_data(self):
        """データの前処理."""
        # IMU列の存在確認
        missing_cols = [col for col in self.imu_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing IMU columns: {missing_cols}")

        # Polarsベクトル化処理を使用（欠損値マスク付き）
        self.sequence_data = self._preprocess_data_vectorized_with_mask()

    @timer_decorator("IMUDataset._preprocess_data_vectorized_with_mask")
    def _preprocess_data_vectorized_with_mask(self) -> dict:
        """Polarsを使ったベクトル化前処理（欠損値マスク付き）."""
        print("Starting vectorized preprocessing with missing value mask...")

        # シーケンスごとの前処理済みデータを一括取得（欠損値処理前）
        processed_data = (
            self.df.sort(["sequence_id", "sequence_counter"])
            .group_by("sequence_id")
            .agg(
                [
                    # IMUデータをリストで集約（欠損値はそのまま）
                    *[pl.col(col) for col in self.imu_cols],
                    # ラベル（シーケンス内で一意）
                    pl.col("gesture").first().alias("gesture"),
                ]
            )
            .sort("sequence_id")
        )

        print(f"Collected {len(processed_data)} sequences for processing")

        # 並列でシーケンス長正規化（欠損値マスク付き）
        sequence_data = self._normalize_sequences_parallel_with_mask(processed_data)

        return sequence_data

    def _normalize_sequences_parallel_with_mask(self, processed_data: pl.DataFrame) -> dict:
        """並列でシーケンス長正規化（欠損値マスク付き）."""
        sequence_data = {}

        # 並列処理用の関数
        def process_sequence_batch(batch_data: list[tuple]) -> list[tuple]:
            batch_results = []
            for row_data in batch_data:
                seq_id = row_data[0]
                # IMUデータを取得（各列のリスト）
                imu_lists = row_data[1:-1]  # 最後はgesture
                gesture = row_data[-1]

                # 各特徴量のリストからNumPy配列に変換（float32型を明示）
                imu_array = np.array(imu_lists, dtype=np.float32)  # [n_features, seq_len]

                # 元の系列長を保存
                original_length = imu_array.shape[1]

                # 欠損値処理（attention_mask用）
                imu_data_processed, missing_mask = self._handle_missing_values_with_mask_vectorized(imu_array)

                # シーケンス長正規化（動的パディング使用時はスキップ）
                if not self.use_dynamic_padding:
                    normalized_imu = self._normalize_sequence_length_vectorized(imu_data_processed)
                    # missing_maskも同様に正規化
                    normalized_mask = self._normalize_missing_mask(missing_mask, original_length)
                else:
                    normalized_imu = imu_data_processed
                    normalized_mask = missing_mask

                # ラベル情報
                multiclass_label = self.gesture_to_id[gesture]
                binary_label = self.is_target_gesture[gesture]

                batch_results.append(
                    (seq_id, normalized_imu, normalized_mask, original_length, multiclass_label, binary_label, gesture)
                )

            return batch_results

        # データを小さなバッチに分割
        batch_size = max(1, len(processed_data) // 4)  # 4つのバッチに分割
        batches = []

        for i in range(0, len(processed_data), batch_size):
            batch = processed_data.slice(i, min(batch_size, len(processed_data) - i))

            # Polarsの効率的なメソッドを使用してデータを取得
            batch_dict = batch.to_dict(as_series=False)

            # バッチデータを構築
            batch_data = []
            for j in range(len(batch)):
                row_data = [batch_dict["sequence_id"][j]]
                # IMUデータを追加
                for col in self.imu_cols:
                    row_data.append(batch_dict[col][j])
                # gestureを追加
                row_data.append(batch_dict["gesture"][j])
                batch_data.append(row_data)

            batches.append(batch_data)

        print(f"Processing {len(batches)} batches in parallel...")

        # 並列処理実行
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = list(
                tqdm(executor.map(process_sequence_batch, batches), desc="Normalizing sequences", total=len(batches))
            )

        # 結果をマージ
        for batch_result in batch_results:
            for (
                seq_id,
                imu_data,
                missing_mask,
                original_length,
                multiclass_label,
                binary_label,
                gesture,
            ) in batch_result:
                sequence_data[seq_id] = {
                    "imu": imu_data,
                    "missing_mask": missing_mask,
                    "original_length": original_length,
                    "multiclass_label": multiclass_label,
                    "binary_label": binary_label,
                    "gesture": gesture,
                }

        print(f"Completed processing {len(sequence_data)} sequences")
        return sequence_data

    def _handle_missing_values_with_mask_vectorized(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """欠損値の処理（attention_mask用、ベクトル化版）.

        Args:
            data: IMUデータ [n_features, seq_len]

        Returns:
            tuple: (処理済みデータ, missing_mask)
                - 処理済みデータ: 欠損値を0で埋めたデータ [n_features, seq_len]
                - missing_mask: [seq_len] 欠損位置のマスク (True=欠損あり, False=正常)
        """
        # データ型を確保（念のため）
        data = data.astype(np.float32)

        # 各タイムステップでの欠損値の存在を記録
        # 1つでも特徴量が欠損していればそのタイムステップは欠損とみなす
        missing_mask = np.isnan(data).any(axis=0)  # [seq_len]

        # 欠損値を0で埋める（attention_maskで無視されるため、値は重要でない）
        data_processed = np.nan_to_num(data, nan=0.0)

        return data_processed, missing_mask

    def _normalize_missing_mask(self, missing_mask: np.ndarray, original_length: int) -> np.ndarray:
        """欠損値マスクの長さ正規化."""
        if len(missing_mask) == self.target_sequence_length:
            return missing_mask
        elif len(missing_mask) < self.target_sequence_length:
            # パディング部分は正常（欠損なし）として扱う
            padded_mask = np.zeros(self.target_sequence_length, dtype=bool)
            padded_mask[: len(missing_mask)] = missing_mask
            return padded_mask
        else:
            # ダウンサンプリング時は比例的に欠損マスクも調整
            indices = np.linspace(0, len(missing_mask) - 1, self.target_sequence_length).astype(int)
            return missing_mask[indices]

    def _normalize_sequence_length_vectorized(self, data: np.ndarray) -> np.ndarray:
        """ベクトル化されたシーケンス長正規化."""
        current_length = data.shape[1]

        if current_length == self.target_sequence_length:
            return data
        elif current_length < self.target_sequence_length:
            # パディング（ゼロ埋め）
            padding = np.zeros((data.shape[0], self.target_sequence_length - current_length))
            return np.hstack([data, padding])
        else:
            # ダウンサンプリング（線形補間）
            original_indices = np.arange(current_length)
            target_indices = np.linspace(0, current_length - 1, self.target_sequence_length)

            normalized_data = np.zeros((data.shape[0], self.target_sequence_length))
            for feature_idx in range(data.shape[0]):
                f = interpolate.interp1d(original_indices, data[feature_idx], kind="linear")
                normalized_data[feature_idx] = f(target_indices)

            return normalized_data

    def _handle_missing_values_with_mask(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """欠損値の処理（attention_mask用）.

        Args:
            data: IMUデータ [seq_len, features]

        Returns:
            tuple: (処理済みデータ, missing_mask)
                - 処理済みデータ: 欠損値を0で埋めたデータ
                - missing_mask: [seq_len] 欠損位置のマスク (True=欠損あり, False=正常)
        """
        # コピーを作成（読み取り専用エラー回避）
        data = data.copy()

        # 各タイムステップでの欠損値の存在を記録
        # 1つでも特徴量が欠損していればそのタイムステップは欠損とみなす
        missing_mask = np.isnan(data).any(axis=1)  # [seq_len]

        # 欠損値を0で埋める（attention_maskで無視されるため、値は重要でない）
        data = np.nan_to_num(data, nan=0.0)

        return data, missing_mask

    def _normalize_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """シーケンス長を目標長に正規化."""
        current_length = len(data)

        if current_length == self.target_sequence_length:
            return data
        elif current_length < self.target_sequence_length:
            # パディング（ゼロ埋め）
            padding = np.zeros((self.target_sequence_length - current_length, data.shape[1]))
            return np.vstack([data, padding])
        else:
            # ダウンサンプリング（線形補間）
            original_indices = np.arange(current_length)
            target_indices = np.linspace(0, current_length - 1, self.target_sequence_length)

            normalized_data = np.zeros((self.target_sequence_length, data.shape[1]))
            for feature_idx in range(data.shape[1]):
                f = interpolate.interp1d(original_indices, data[:, feature_idx], kind="linear")
                normalized_data[:, feature_idx] = f(target_indices)

            return normalized_data

    def _apply_augmentation(self, imu_data: np.ndarray) -> np.ndarray:
        """データ拡張の適用."""
        if not self.augment:
            return imu_data

        # ガウシアンノイズ
        if self.augmentation_config.get("gaussian_noise", {}).get("probability", 0) > 0:
            if np.random.random() < self.augmentation_config["gaussian_noise"]["probability"]:
                std = self.augmentation_config["gaussian_noise"]["std"]
                noise = np.random.normal(0, std, imu_data.shape)
                imu_data = imu_data + noise

        # 時間スケーリング
        if self.augmentation_config.get("time_scaling", {}).get("probability", 0) > 0:
            if np.random.random() < self.augmentation_config["time_scaling"]["probability"]:
                scale_range = self.augmentation_config["time_scaling"]["scale_range"]
                scale = np.random.uniform(scale_range[0], scale_range[1])
                seq_len = imu_data.shape[-1]
                new_len = int(seq_len * scale)
                if new_len > 0:
                    # リサイズ
                    indices = np.linspace(0, seq_len - 1, new_len)
                    resized_data = np.zeros((imu_data.shape[0], new_len))
                    for i in range(imu_data.shape[0]):
                        f = interpolate.interp1d(
                            np.arange(seq_len), imu_data[i], kind="linear", bounds_error=False, fill_value=0
                        )
                        resized_data[i] = f(indices)

                    # 元の長さに戻す
                    if new_len < seq_len:
                        # パディング
                        padding = np.zeros((imu_data.shape[0], seq_len - new_len))
                        imu_data = np.hstack([resized_data, padding])
                    else:
                        # 切り取り
                        imu_data = resized_data[:, :seq_len]

        # 部分マスキング
        if self.augmentation_config.get("partial_masking", {}).get("probability", 0) > 0:
            if np.random.random() < self.augmentation_config["partial_masking"]["probability"]:
                mask_ratio = self.augmentation_config["partial_masking"]["mask_ratio"]
                mask_length_range = self.augmentation_config["partial_masking"]["mask_length_range"]

                seq_len = imu_data.shape[-1]
                mask_length = np.random.randint(mask_length_range[0], min(mask_length_range[1], seq_len))
                start_idx = np.random.randint(0, max(1, seq_len - mask_length))

                # マスクを適用
                imu_data[:, start_idx : start_idx + mask_length] = 0

        return imu_data

    def __len__(self) -> int:
        """データ数を返す."""
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """データ取得."""
        seq_id = self.sequence_ids[idx]
        data = self.sequence_data[seq_id]

        # IMUデータの取得とデータ拡張
        imu_data = data["imu"].copy()
        imu_data = self._apply_augmentation(imu_data)

        # テンソル化 [seq_len, features] -> [features, seq_len]
        # 最適化版では既に[features, seq_len]なのでそのまま使用
        if imu_data.shape[0] == len(self.imu_cols):  # [features, seq_len]
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
        else:  # [seq_len, features]
            imu_tensor = torch.tensor(imu_data.T, dtype=torch.float32)

        result = {
            "imu": imu_tensor,
            "multiclass_label": torch.tensor(data["multiclass_label"], dtype=torch.long),
            "binary_label": torch.tensor(data["binary_label"], dtype=torch.float32),
            "sequence_id": seq_id,
            "gesture": data["gesture"],
        }

        # attention_maskの追加（欠損値がある場合）
        if "missing_mask" in data:
            # missing_maskの逆がattention_mask（正常位置でTrue、欠損位置でFalse）
            attention_mask = torch.tensor(~data["missing_mask"], dtype=torch.bool)
            result["attention_mask"] = attention_mask

        return result


class IMUDataModule(lightning.LightningDataModule):
    """IMU用のLightning DataModule."""

    def __init__(
        self,
        config: Config,
        fold: int = 0,
    ):
        """
        初期化.

        Args:
            config: 設定辞書
            fold: CVのfold番号
        """
        super().__init__()
        self.config = config
        self.fold = fold
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers

        # データ読み込み
        print(f"Loading train data from {config.data.train_path}...")
        self.train_df = pl.read_csv(config.data.train_path)
        print(f"Loaded {len(self.train_df)} rows")

        # CVスプリットの取得
        self.splits = get_validation(self.train_df, config.model_dump())
        self.train_idx, self.val_idx = self.splits[fold]

        # データセット用パラメータ
        self.target_sequence_length = config.preprocessing.target_sequence_length
        self.augmentation_config = config.augmentation.model_dump()

        # 長さグループ化設定（exp004では無効化）
        self.use_length_grouping = False
        self.use_dynamic_padding = False

    @timer_decorator("IMUDataModule.setup")
    def setup(self, stage: str | None = None):
        """データセットのセットアップ."""
        print(f"\nSetting up data module for fold {self.fold}, stage: {stage}")

        if stage in ("fit", None):
            # 訓練・検証用データの作成
            print("Creating train/validation splits...")
            train_sequence_ids = [self.train_df.get_column("sequence_id")[i] for i in self.train_idx]
            val_sequence_ids = [self.train_df.get_column("sequence_id")[i] for i in self.val_idx]

            train_df = self.train_df.filter(pl.col("sequence_id").is_in(train_sequence_ids))
            val_df = self.train_df.filter(pl.col("sequence_id").is_in(val_sequence_ids))

            print(f"Creating train dataset ({len(train_df)} rows)...")
            self.train_dataset = IMUDataset(
                train_df,
                target_sequence_length=self.target_sequence_length,
                augment=True,
                augmentation_config=self.augmentation_config,
                use_dynamic_padding=self.use_dynamic_padding,
            )

            print(f"Creating validation dataset ({len(val_df)} rows)...")
            self.val_dataset = IMUDataset(
                val_df,
                target_sequence_length=self.target_sequence_length,
                augment=False,
                use_dynamic_padding=self.use_dynamic_padding,
            )

            print(f"Datasets created: train={len(self.train_dataset)}, val={len(self.val_dataset)} sequences")

    def train_dataloader(self) -> DataLoader:
        """訓練用データローダー."""
        # 動的パディング使用時
        if self.use_dynamic_padding:
            collate_fn = lambda batch: dynamic_collate_fn(batch)
        else:
            collate_fn = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """検証用データローダー."""
        # 動的パディング使用時
        if self.use_dynamic_padding:
            collate_fn = lambda batch: dynamic_collate_fn(batch)
        else:
            collate_fn = None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class SingleSequenceIMUDataset(Dataset):
    """単一シーケンス用のIMUデータセット（推論用）."""

    def __init__(self, sequence_df: pl.DataFrame, target_sequence_length: int = 200):
        """
        初期化.

        Args:
            sequence_df: 単一シーケンスのPolarsデータフレーム
            target_sequence_length: 目標シーケンス長
        """
        self.df = sequence_df
        self.target_sequence_length = target_sequence_length
        self.augment = False  # 推論時は拡張なし

        # IMU列の定義
        self.imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

        # シーケンスIDを取得(単一シーケンスなので最初の値)
        self.sequence_id = sequence_df.get_column("sequence_id")[0]

        # データ前処理
        self._preprocess_data()

    def _preprocess_data(self):
        """データの前処理(単一シーケンス用)."""
        # IMU列の存在確認
        missing_cols = [col for col in self.imu_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing IMU columns: {missing_cols}")

        # シーケンスを時間順でソート
        seq_df = self.df.sort("sequence_counter")

        # IMUデータの抽出
        imu_data = seq_df.select(self.imu_cols).to_numpy()

        # 欠損値処理(attention_mask用)
        imu_data, missing_mask = self._handle_missing_values_with_mask(imu_data)

        # シーケンス長の正規化
        imu_data = self._normalize_sequence_length(imu_data)

        # missing_maskも長さ調整(シーケンス長の正規化に対応)
        if len(missing_mask) != self.target_sequence_length:
            if len(missing_mask) < self.target_sequence_length:
                # パディング部分は正常(欠損なし)として扱う
                padded_mask = np.zeros(self.target_sequence_length, dtype=bool)
                padded_mask[: len(missing_mask)] = missing_mask
                missing_mask = padded_mask
            else:
                # ダウンサンプリング時は比例的に欠損マスクも調整
                indices = np.linspace(0, len(missing_mask) - 1, self.target_sequence_length).astype(int)
                missing_mask = missing_mask[indices]

        # データの格納
        self.imu_data = imu_data
        self.missing_mask = missing_mask

    def _handle_missing_values_with_mask(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """欠損値の処理（attention_mask用）.

        Args:
            data: IMUデータ [seq_len, features]

        Returns:
            tuple: (処理済みデータ, missing_mask)
                - 処理済みデータ: 欠損値を0で埋めたデータ
                - missing_mask: [seq_len] 欠損位置のマスク (True=欠損あり, False=正常)
        """
        # コピーを作成（読み取り専用エラー回避）
        data = data.copy()

        # 各タイムステップでの欠損値の存在を記録
        # 1つでも特徴量が欠損していればそのタイムステップは欠損とみなす
        missing_mask = np.isnan(data).any(axis=1)  # [seq_len]

        # 欠損値を0で埋める（attention_maskで無視されるため、値は重要でない）
        data = np.nan_to_num(data, nan=0.0)

        return data, missing_mask

    def _normalize_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """シーケンス長を目標長に正規化."""
        current_length = len(data)

        if current_length == self.target_sequence_length:
            return data
        elif current_length < self.target_sequence_length:
            # パディング（ゼロ埋め）
            padding = np.zeros((self.target_sequence_length - current_length, data.shape[1]))
            return np.vstack([data, padding])
        else:
            # ダウンサンプリング（線形補間）
            original_indices = np.arange(current_length)
            target_indices = np.linspace(0, current_length - 1, self.target_sequence_length)

            normalized_data = np.zeros((self.target_sequence_length, data.shape[1]))
            for feature_idx in range(data.shape[1]):
                f = interpolate.interp1d(original_indices, data[:, feature_idx], kind="linear")
                normalized_data[:, feature_idx] = f(target_indices)

            return normalized_data

    def __len__(self) -> int:
        """データ数を返す(単一シーケンスなので常に1)."""
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """データ取得(単一シーケンス用)."""
        if idx != 0:
            raise IndexError("Single sequence dataset only has one item")

        # IMUデータの取得
        imu_data = self.imu_data.copy()

        # テンソル化 [seq_len, features] -> [features, seq_len]
        imu_tensor = torch.tensor(imu_data.T, dtype=torch.float32)

        # missing_maskからattention_maskを作成(欠損位置を無視)
        attention_mask = torch.tensor(~self.missing_mask, dtype=torch.bool)  # 欠損位置でFalse、正常位置でTrue

        return {"imu": imu_tensor, "attention_mask": attention_mask, "sequence_id": self.sequence_id}


if __name__ == "__main__":
    """テスト実行."""
    # テスト用設定
    config = Config()

    # データモジュールのテスト
    dm = IMUDataModule(config, fold=0)
    dm.setup("fit")

    # データローダーのテスト
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Number of classes: {dm.train_dataset.num_classes}")

    # サンプルデータの確認
    sample = next(iter(train_loader))
    print(f"IMU shape: {sample['imu'].shape}")
    print(f"Multiclass label shape: {sample['multiclass_label'].shape}")
    print(f"Binary label shape: {sample['binary_label'].shape}")
