"""IMU専用データセットモジュール - exp007用（欠損値をattention_maskで処理 + Polars最適化）."""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning as lightning
import torch
from config import Config
from scipy import interpolate

# Imports from project root
from src.utils.timer import timer_decorator
from src.validation.factory import get_validation
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


def get_length_grouped_indices(
    lengths: list[int], batch_size: int, mega_batch_mult: int = 8, generator: torch.Generator | None = None
) -> list[int]:
    """
    HuggingFace風の長さグループ化インデックス生成.

    Args:
        lengths: 各サンプルの系列長リスト
        batch_size: バッチサイズ
        mega_batch_mult: メガバッチの倍率
        generator: 乱数生成器

    Returns:
        グループ化されたインデックスリスト
    """
    # メガバッチサイズ
    mega_batch_size = batch_size * mega_batch_mult

    # インデックスをランダムシャッフル
    indices = list(range(len(lengths)))
    if generator is not None:
        # PyTorchのGeneratorを使用
        g = torch.Generator()
        g.manual_seed(generator.initial_seed())
        indices = torch.randperm(len(lengths), generator=g).tolist()
    else:
        random.shuffle(indices)

    # メガバッチに分割
    mega_batches = [indices[i : i + mega_batch_size] for i in range(0, len(indices), mega_batch_size)]

    # 各メガバッチ内で長さによるソート
    for mega_batch in mega_batches:
        mega_batch.sort(key=lambda i: lengths[i], reverse=True)

    # メガバッチをバッチサイズに分割
    batched_indices = []
    for mega_batch in mega_batches:
        for i in range(0, len(mega_batch), batch_size):
            batch = mega_batch[i : i + batch_size]
            if len(batch) == batch_size:  # 完全なバッチのみ使用
                batched_indices.extend(batch)

    return batched_indices


class LengthGroupedSampler(Sampler):
    """
    HuggingFace風のLengthGroupedSampler.
    系列長でグループ化してバッチ効率を向上させる。
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: list[int] | None = None,
        mega_batch_mult: int = 8,
        generator: torch.Generator | None = None,
    ):
        """
        初期化.

        Args:
            dataset: データセット
            batch_size: バッチサイズ
            lengths: 各サンプルの系列長リスト（Noneの場合はdatasetから取得）
            mega_batch_mult: メガバッチ倍率
            generator: 乱数生成器
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.mega_batch_mult = mega_batch_mult
        self.generator = generator

        # 系列長の取得
        if lengths is None:
            self.lengths = self._get_lengths_from_dataset()
        else:
            self.lengths = lengths

        if len(self.lengths) != len(dataset):
            raise ValueError(f"Length mismatch: lengths={len(self.lengths)}, dataset={len(dataset)}")

    def _get_lengths_from_dataset(self) -> list[int]:
        """データセットから系列長を取得."""
        lengths = []
        for i in range(len(self.dataset)):
            if hasattr(self.dataset, "sequence_ids") and hasattr(self.dataset, "sequence_data"):
                seq_id = self.dataset.sequence_ids[i]
                # 最適化されたデータ構造では原形データがoriginal_lengthに保存される
                if "original_length" in self.dataset.sequence_data[seq_id]:
                    length = self.dataset.sequence_data[seq_id]["original_length"]
                else:
                    # [features, seq_len]形式なので2番目の次元を取得
                    length = self.dataset.sequence_data[seq_id]["imu"].shape[1]
            else:
                # フォールバック: 実際にデータを取得
                sample = self.dataset[i]
                if "original_length" in sample:
                    length = sample["original_length"]
                else:
                    length = sample["imu"].shape[-1]  # [features, seq_len]
            lengths.append(length)
        return lengths

    def __iter__(self):
        """イテレーター."""
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.mega_batch_mult, self.generator)
        return iter(indices)

    def __len__(self) -> int:
        """サンプル数."""
        # 完全なバッチのみを考慮
        return (len(self.dataset) // self.batch_size) * self.batch_size


def dynamic_collate_fn(
    batch: list[dict[str, Any]], percentile_max_length: float = 0.95, pad_token_id: float = 0.0
) -> dict[str, torch.Tensor]:
    """
    動的パディングCollate関数（欠損値マスク対応）.

    Args:
        batch: バッチデータのリスト
        percentile_max_length: パディング最大長のパーセンタイル
        pad_token_id: パディング値

    Returns:
        コレートされたバッチデータ
    """
    batch_size = len(batch)

    # 系列長を取得
    if "original_length" in batch[0]:
        lengths = [item["original_length"] for item in batch]
    else:
        lengths = [item["imu"].shape[-1] for item in batch]

    # パーセンタイルベースの最大長決定
    max_length = int(np.percentile(lengths, percentile_max_length * 100))
    max_length = max(max_length, max(lengths))  # 最低でも最大系列長は確保

    # テンソルの初期化
    imu_batch = torch.full((batch_size, 7, max_length), pad_token_id, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    multiclass_labels = torch.zeros(batch_size, dtype=torch.long)
    binary_labels = torch.zeros(batch_size, dtype=torch.float32)
    sequence_ids = []
    gestures = []

    for i, item in enumerate(batch):
        seq_len = lengths[i]

        # IMUデータをコピー（最大長まで）
        actual_len = min(seq_len, max_length)
        imu_batch[i, :, :actual_len] = item["imu"][:, :actual_len]

        # アテンションマスクを設定（True = 有効な位置）
        attention_mask[i, :actual_len] = True

        # 欠損値マスクがある場合は統合
        if "missing_mask" in item:
            missing_mask = item["missing_mask"]
            # missing_maskの長さを調整
            missing_len = min(len(missing_mask), actual_len)
            # 欠損部分（missing_mask=True）はattention_maskでFalseにする
            attention_mask[i, :missing_len] = attention_mask[i, :missing_len] & (~missing_mask[:missing_len])

        # ラベルとメタデータ
        multiclass_labels[i] = item["multiclass_label"]
        binary_labels[i] = item["binary_label"]
        sequence_ids.append(item["sequence_id"])
        gestures.append(item["gesture"])

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
            use_dynamic_padding: 動的パディングを使用するかどうか
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

        # ジェスチャーラベルのマッピング作成
        self._create_label_mappings()

        # データ前処理
        print("Preprocessing data...")
        self._preprocess_data()
        print("Data preprocessing completed")

    def _create_label_mappings(self):
        """ジェスチャーラベルのマッピングを作成."""
        # Target gestures (BFRB-like)
        target_gestures = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        ]

        # Non-target gestures
        non_target_gestures = [
            "Drink from bottle/cup",
            "Glasses on/off",
            "Pull air toward your face",
            "Pinch knee/leg skin",
            "Scratch knee/leg skin",
            "Write name on leg",
            "Text on phone",
            "Feel around in tray and pull out an object",
            "Write name in air",
            "Wave hello",
        ]

        # 全ジェスチャーの一覧取得
        all_gestures = self.df.get_column("gesture").unique().to_list()

        # マルチクラスラベルマッピング
        self.gesture_to_id = {gesture: idx for idx, gesture in enumerate(sorted(all_gestures))}
        self.id_to_gesture = {idx: gesture for gesture, idx in self.gesture_to_id.items()}

        # バイナリラベルマッピング
        self.is_target_gesture = {gesture: 1 if gesture in target_gestures else 0 for gesture in all_gestures}

        self.num_classes = len(all_gestures)

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
        n_features, current_length = data.shape

        if current_length == self.target_sequence_length:
            return data
        elif current_length < self.target_sequence_length:
            # パディング（最後の値で埋める）
            padding = np.repeat(data[:, -1:], self.target_sequence_length - current_length, axis=1)
            return np.concatenate([data, padding], axis=1)
        else:
            # scipy.interpolateを使った高速補間
            old_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, self.target_sequence_length)

            # 全特徴量を一度に補間
            interpolated_data = np.zeros((n_features, self.target_sequence_length))
            for i in range(n_features):
                f = interpolate.interp1d(old_indices, data[i], kind="linear", assume_sorted=True)
                interpolated_data[i] = f(new_indices)

            return interpolated_data

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
            # パディング（最後の値で埋める）
            padding = np.repeat(data[-1:], self.target_sequence_length - current_length, axis=0)
            return np.concatenate([data, padding], axis=0)
        else:
            # 線形補間によるダウンサンプリング
            indices = np.linspace(0, current_length - 1, self.target_sequence_length)
            interpolated_data = np.zeros((self.target_sequence_length, data.shape[1]))

            for i in range(data.shape[1]):
                interpolated_data[:, i] = np.interp(indices, np.arange(current_length), data[:, i])

            return interpolated_data

    def _apply_augmentation(self, imu_data: np.ndarray) -> np.ndarray:
        """データ拡張の適用."""
        if not self.augment:
            return imu_data

        # ガウシアンノイズ
        if np.random.random() < self.augmentation_config.get("gaussian_noise_prob", 0.0):
            noise_std = self.augmentation_config.get("gaussian_noise_std", 0.01)
            noise = np.random.normal(0, noise_std, imu_data.shape)
            imu_data = imu_data + noise

        # 時間スケーリング
        if np.random.random() < self.augmentation_config.get("time_scaling_prob", 0.0):
            scale_range = self.augmentation_config.get("time_scaling_range", [0.9, 1.1])
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])

            current_length = len(imu_data)
            new_length = int(current_length * scale_factor)

            # スケーリング後、元の長さに戻す
            indices = np.linspace(0, new_length - 1, current_length)
            scaled_data = np.zeros_like(imu_data)

            for i in range(imu_data.shape[1]):
                if scale_factor > 1.0:
                    # 時間軸を伸ばす（一部を切り出し）
                    extended_data = np.interp(np.arange(new_length), np.arange(current_length), imu_data[:, i])
                    scaled_data[:, i] = extended_data[:current_length]
                else:
                    # 時間軸を縮める（補間）
                    scaled_data[:, i] = np.interp(indices, np.arange(new_length), imu_data[: int(new_length), i])

            imu_data = scaled_data

        # 部分マスキング
        if np.random.random() < self.augmentation_config.get("partial_masking_prob", 0.0):
            mask_length_range = self.augmentation_config.get("partial_masking_length_range", [5, 20])
            mask_ratio = self.augmentation_config.get("partial_masking_ratio", 0.1)

            num_masks = int(len(imu_data) * mask_ratio / np.mean(mask_length_range))

            for _ in range(num_masks):
                mask_length = np.random.randint(mask_length_range[0], mask_length_range[1] + 1)
                start_idx = np.random.randint(0, max(1, len(imu_data) - mask_length))
                end_idx = min(start_idx + mask_length, len(imu_data))

                # マスク（ゼロ埋め）
                imu_data[start_idx:end_idx] = 0

        return imu_data

    def __len__(self) -> int:
        """データセットサイズ."""
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

        # 動的パディング使用時は元の系列長と欠損マスクも返す
        if self.use_dynamic_padding:
            result["original_length"] = data["original_length"]
            result["missing_mask"] = torch.tensor(data["missing_mask"], dtype=torch.bool)
        else:
            # 固定長の場合も欠損マスクを返す
            missing_mask = data["missing_mask"]
            if len(missing_mask) != self.target_sequence_length:
                # 長さ調整が行われた場合の欠損マスク処理
                if len(missing_mask) < self.target_sequence_length:
                    # パディング部分は正常（欠損なし）として扱う
                    padded_mask = np.zeros(self.target_sequence_length, dtype=bool)
                    padded_mask[: len(missing_mask)] = missing_mask
                    missing_mask = padded_mask
                else:
                    # ダウンサンプリング時は比例的に欠損マスクも調整
                    indices = np.linspace(0, len(missing_mask) - 1, self.target_sequence_length).astype(int)
                    missing_mask = missing_mask[indices]
            result["missing_mask"] = torch.tensor(missing_mask, dtype=torch.bool)

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

        # 長さグループ化設定
        self.use_length_grouping = config.length_grouping.enabled
        self.use_dynamic_padding = config.length_grouping.use_dynamic_padding
        self.mega_batch_mult = config.length_grouping.mega_batch_multiplier
        self.percentile_max_length = config.length_grouping.percentile_max_length

    @timer_decorator("IMUDataModule.setup")
    def setup(self, stage: str | None = None):
        """データセットのセットアップ."""
        print(f"\nSetting up data module for fold {self.fold}, stage: {stage}")
        if stage == "fit" or stage is None:
            # Train/Validationデータセットの作成
            print("Creating train/validation splits...")
            train_sequence_ids = self.train_df.filter(
                pl.col("sequence_id").is_in(
                    self.train_df.with_row_index()
                    .filter(pl.col("index").is_in(self.train_idx))
                    .get_column("sequence_id")
                    .unique()
                    .to_list()
                )
            )

            val_sequence_ids = self.train_df.filter(
                pl.col("sequence_id").is_in(
                    self.train_df.with_row_index()
                    .filter(pl.col("index").is_in(self.val_idx))
                    .get_column("sequence_id")
                    .unique()
                    .to_list()
                )
            )

            print(f"Creating train dataset ({len(train_sequence_ids)} rows)...")
            self.train_dataset = IMUDataset(
                train_sequence_ids,
                target_sequence_length=self.target_sequence_length,
                augment=True,
                augmentation_config=self.augmentation_config,
                use_dynamic_padding=self.use_dynamic_padding,
            )

            print(f"Creating validation dataset ({len(val_sequence_ids)} rows)...")
            self.val_dataset = IMUDataset(
                val_sequence_ids,
                target_sequence_length=self.target_sequence_length,
                augment=False,
                use_dynamic_padding=self.use_dynamic_padding,
            )
            print(f"Datasets created: train={len(self.train_dataset)}, val={len(self.val_dataset)} sequences")

    def train_dataloader(self) -> DataLoader:
        """訓練用データローダー."""
        # 長さグループ化使用時
        if self.use_length_grouping:
            sampler = LengthGroupedSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                mega_batch_mult=self.mega_batch_mult,
            )
            shuffle = False  # samplerを使用する場合はshuffleはFalse
        else:
            sampler = None
            shuffle = True

        # 動的パディング使用時
        if self.use_dynamic_padding:
            collate_fn = lambda batch: dynamic_collate_fn(batch, percentile_max_length=self.percentile_max_length)
        else:
            collate_fn = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """検証用データローダー."""
        # 動的パディング使用時
        if self.use_dynamic_padding:
            collate_fn = lambda batch: dynamic_collate_fn(batch, percentile_max_length=self.percentile_max_length)
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
