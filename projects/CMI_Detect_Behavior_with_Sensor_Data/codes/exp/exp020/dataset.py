"""IMU + Demographics統合データセットモジュール - exp013用（Demographics特徴量統合 + 物理ベースIMU特徴量 + 欠損値処理 + Polars最適化）."""

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


def remove_gravity_from_acc_pl(df: pl.LazyFrame, tol: float = 1e-6) -> pl.LazyFrame:
    """
    重力成分を除去して線形加速度を計算（Polars版）.

    Args:
        df: IMUデータを含むDataFrame（acc_x/y/z, rot_x/y/z/wが必要）
        tol: 四元数の有効判定閾値

    Returns:
        線形加速度のDataFrame
    """
    df = df.clone()

    x = pl.col("rot_x")
    y = pl.col("rot_y")
    z = pl.col("rot_z")
    w = pl.col("rot_w")

    # 有効判定：NaNなし & ノルムがしきい値超え
    norm = (x**2 + y**2 + z**2 + w**2).sqrt()
    valid = (~pl.any_horizontal([x.is_null(), y.is_null(), z.is_null(), w.is_null()])) & (norm > tol)

    # 正規化四元数
    xn = x / norm
    yn = y / norm
    zn = z / norm
    wn = w / norm

    # 逆回転後の重力（センサ座標）
    gx = 19.62 * (xn * zn - wn * yn)
    gy = 19.62 * (wn * xn + yn * zn)
    gz = 9.81 - 19.62 * (xn * xn + yn * yn)

    # 無効行は重力0扱いにして acc をそのまま通す
    gx = pl.when(valid).then(gx).otherwise(0.0)
    gy = pl.when(valid).then(gy).otherwise(0.0)
    gz = pl.when(valid).then(gz).otherwise(0.0)

    # 線形加速度 = 測定 - 重力（有効行） / 無効行は測定のまま
    lin_x = pl.when(valid).then(pl.col("acc_x") - gx).otherwise(pl.col("acc_x")).alias("linear_acc_x")
    lin_y = pl.when(valid).then(pl.col("acc_y") - gy).otherwise(pl.col("acc_y")).alias("linear_acc_y")
    lin_z = pl.when(valid).then(pl.col("acc_z") - gz).otherwise(pl.col("acc_z")).alias("linear_acc_z")

    return df.with_columns([lin_x, lin_y, lin_z]).select("linear_acc_x", "linear_acc_y", "linear_acc_z")


def calculate_angular_velocity_from_quat_pl(
    rot_df: pl.LazyFrame, time_delta: float = 1 / 200, tol: float = 1e-8
) -> pl.LazyFrame:
    """
    クォータニオンから角速度を計算（Polars版）.

    Args:
        rot_df: 四元数データを含むDataFrame（rot_x/y/z/wが必要）
        time_delta: タイムステップ（デフォルト1/200秒）
        tol: 数値安定化の閾値

    Returns:
        角速度のDataFrame
    """
    x1 = pl.col("rot_x")
    y1 = pl.col("rot_y")
    z1 = pl.col("rot_z")
    w1 = pl.col("rot_w")
    # シーケンス境界を尊重したshift操作
    if "sequence_id" in rot_df.collect_schema().names():
        x2 = x1.shift(-1).over("sequence_id")
        y2 = y1.shift(-1).over("sequence_id")
        z2 = z1.shift(-1).over("sequence_id")
        w2 = w1.shift(-1).over("sequence_id")
    else:
        x2 = x1.shift(-1)
        y2 = y1.shift(-1)
        z2 = z1.shift(-1)
        w2 = w1.shift(-1)

    # ノルムで正規化（SciPy 相当）
    n1 = (x1 * x1 + y1 * y1 + z1 * z1 + w1 * w1).sqrt()
    n2 = (x2 * x2 + y2 * y2 + z2 * z2 + w2 * w2).sqrt()
    xn1, yn1, zn1, wn1 = x1 / n1, y1 / n1, z1 / n1, w1 / n1
    xn2, yn2, zn2, wn2 = x2 / n2, y2 / n2, z2 / n2, w2 / n2

    # 連続性のための符号合わせ（dot<0なら後者を反転）
    dot = xn1 * xn2 + yn1 * yn2 + zn1 * zn2 + wn1 * wn2
    sgn = pl.when(dot.is_not_null() & (dot < 0)).then(-1.0).otherwise(1.0)
    xn2, yn2, zn2, wn2 = xn2 * sgn, yn2 * sgn, zn2 * sgn, wn2 * sgn

    # 単位四元数を前提に delta = q1^{-1} * q2 = conj(q1) * q2 を計算
    # conj(q1)=(-x1,-y1,-z1,w1)。四元数積の展開（スカラー末尾）
    dx = wn1 * xn2 - xn1 * wn2 - yn1 * zn2 + zn1 * yn2
    dy = wn1 * yn2 + xn1 * zn2 - yn1 * wn2 - zn1 * xn2
    dz = wn1 * zn2 - xn1 * yn2 + yn1 * xn2 - zn1 * wn2
    dw = wn1 * wn2 + xn1 * xn2 + yn1 * yn2 + zn1 * zn2

    # 数値安全のためクランプ
    dw = pl.min_horizontal(pl.lit(1.0), pl.max_horizontal(pl.lit(-1.0), dw))

    # 回転ベクトル = 角度 * 単位軸
    vnorm = (dx * dx + dy * dy + dz * dz).sqrt()
    angle = 2.0 * pl.arctan2(vnorm, dw)  # より安定（acos より端がマシ）
    scale = pl.when(vnorm > tol).then(angle / vnorm).otherwise(0.0)

    rvx = dx * scale
    rvy = dy * scale
    rvz = dz * scale

    # 角速度 [rad/s]
    wx = rvx / time_delta
    wy = rvy / time_delta
    wz = rvz / time_delta

    # 有効判定（NaN/Nullや最終行は無効 → 0）
    valid = (
        (~pl.any_horizontal([x1.is_null(), y1.is_null(), z1.is_null(), w1.is_null()]))
        & (~pl.any_horizontal([x2.is_null(), y2.is_null(), z2.is_null(), w2.is_null()]))
        & (n1 > tol)
        & (n2 > tol)
    )

    return rot_df.with_columns(
        [
            pl.when(valid).then(wx).otherwise(0.0).alias("angular_vel_x"),
            pl.when(valid).then(wy).otherwise(0.0).alias("angular_vel_y"),
            pl.when(valid).then(wz).otherwise(0.0).alias("angular_vel_z"),
        ]
    ).select("angular_vel_x", "angular_vel_y", "angular_vel_z")


def calculate_angular_distance_pl(rot_df: pl.LazyFrame, tol: float = 1e-8) -> pl.LazyFrame:
    """
    連続する四元数間の角距離を計算（Polars版）.

    Args:
        rot_df: 四元数データを含むDataFrame（rot_x/y/z/wが必要）
        tol: 数値安定化の閾値

    Returns:
        角距離のDataFrame
    """
    # 入力: rot_x, rot_y, rot_z, rot_w（スカラー末尾）
    x1 = pl.col("rot_x")
    y1 = pl.col("rot_y")
    z1 = pl.col("rot_z")
    w1 = pl.col("rot_w")
    # シーケンス境界を尊重したshift操作
    if "sequence_id" in rot_df.collect_schema().names():
        x2 = x1.shift(-1).over("sequence_id")
        y2 = y1.shift(-1).over("sequence_id")
        z2 = z1.shift(-1).over("sequence_id")
        w2 = w1.shift(-1).over("sequence_id")
    else:
        x2 = x1.shift(-1)
        y2 = y1.shift(-1)
        z2 = z1.shift(-1)
        w2 = w1.shift(-1)

    # 正規化（SciPy相当）
    n1 = (x1 * x1 + y1 * y1 + z1 * z1 + w1 * w1).sqrt()
    n2 = (x2 * x2 + y2 * y2 + z2 * z2 + w2 * w2).sqrt()
    xn1, yn1, zn1, wn1 = x1 / n1, y1 / n1, z1 / n1, w1 / n1
    xn2, yn2, zn2, wn2 = x2 / n2, y2 / n2, z2 / n2, w2 / n2

    # q と -q の同値対策（最短経路）
    dot = xn1 * xn2 + yn1 * yn2 + zn1 * zn2 + wn1 * wn2
    sgn = pl.when(dot.is_not_null() & (dot < 0)).then(-1.0).otherwise(1.0)
    xn2, yn2, zn2, wn2 = xn2 * sgn, yn2 * sgn, zn2 * sgn, wn2 * sgn

    # delta = q1^{-1} * q2 = conj(q1) * q2 （スカラー末尾の四元数積）
    dx = wn1 * xn2 - xn1 * wn2 - yn1 * zn2 + zn1 * yn2
    dy = wn1 * yn2 + xn1 * zn2 - yn1 * wn2 - zn1 * xn2
    dz = wn1 * zn2 - xn1 * yn2 + yn1 * xn2 - zn1 * wn2
    dw = wn1 * wn2 + xn1 * xn2 + yn1 * yn2 + zn1 * zn2

    # 数値安定化＆角度（0..π）
    vnorm = (dx * dx + dy * dy + dz * dz).sqrt()
    angle = 2.0 * pl.arctan2(vnorm, dw)  # = ||as_rotvec||

    # 有効判定（末尾行やNaN/ゼロ長は 0）
    valid = (
        (~pl.any_horizontal([x1.is_null(), y1.is_null(), z1.is_null(), w1.is_null()]))
        & (~pl.any_horizontal([x2.is_null(), y2.is_null(), z2.is_null(), w2.is_null()]))
        & (n1 > tol)
        & (n2 > tol)
    )

    return rot_df.with_columns(pl.when(valid).then(angle).otherwise(0.0).alias("angular_distance")).select(
        "angular_distance"
    )


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


class HandednessAugmentation:
    """利き手反転データオーグメンテーション."""

    def __init__(self, demographics_df: pl.DataFrame, flip_probability: float = 0.5):
        """
        初期化.

        Args:
            demographics_df: Demographics データフレーム
            flip_probability: 反転を適用する確率 (0.0-1.0)
        """
        self.flip_probability = flip_probability

        # subject -> handedness のマッピングを作成
        self.subject_to_handedness = {}
        if demographics_df is not None:
            for row in demographics_df.iter_rows(named=True):
                self.subject_to_handedness[row["subject"]] = row["handedness"]

        print(f"HandednessAugmentation initialized with {len(self.subject_to_handedness)} subjects")

    def should_flip(self) -> bool:
        """確率的に反転を適用するかどうかを決定."""
        return np.random.random() < self.flip_probability

    def flip_imu_data(self, imu_data: np.ndarray, imu_cols: list[str]) -> np.ndarray:
        """
        IMUデータの利き手反転.

        Args:
            imu_data: IMUデータ [features, seq_len] または [seq_len, features]
            imu_cols: IMU列名のリスト

        Returns:
            反転されたIMUデータ
        """
        flipped_data = imu_data.copy()

        # データの次元を確認
        if imu_data.shape[0] == len(imu_cols):  # [features, seq_len]
            is_features_first = True
        else:  # [seq_len, features]
            is_features_first = False
            flipped_data = flipped_data.T  # [features, seq_len]に変換

        # IMU列のインデックスを取得
        acc_y_idx = imu_cols.index("acc_y") if "acc_y" in imu_cols else -1
        rot_y_idx = imu_cols.index("rot_y") if "rot_y" in imu_cols else -1

        # 加速度Y軸の反転（左右の違いを表現）
        if acc_y_idx >= 0:
            flipped_data[acc_y_idx] = -flipped_data[acc_y_idx]

        # 回転Y軸の反転（四元数のY軸周りの回転を反転）
        if rot_y_idx >= 0:
            flipped_data[rot_y_idx] = -flipped_data[rot_y_idx]

        # 物理ベース特徴量の反転も処理
        linear_acc_y_idx = imu_cols.index("linear_acc_y") if "linear_acc_y" in imu_cols else -1
        angular_vel_y_idx = imu_cols.index("angular_vel_y") if "angular_vel_y" in imu_cols else -1

        if linear_acc_y_idx >= 0:
            flipped_data[linear_acc_y_idx] = -flipped_data[linear_acc_y_idx]

        if angular_vel_y_idx >= 0:
            flipped_data[angular_vel_y_idx] = -flipped_data[angular_vel_y_idx]

        # 元の次元に戻す
        if not is_features_first:
            flipped_data = flipped_data.T  # [seq_len, features]に戻す

        return flipped_data

    def get_handedness(self, subject_id: str) -> int:
        """
        被験者の利き手情報を取得.

        Args:
            subject_id: 被験者ID

        Returns:
            利き手 (0=左利き, 1=右利き, デフォルト=1)
        """
        return self.subject_to_handedness.get(subject_id, 1)  # デフォルトは右利き


class IMUDataset(Dataset):
    """IMU + Demographics統合データセット（acc_x/y/z + rot_w/x/y/z + demographics）."""

    def __init__(
        self,
        df: pl.DataFrame,
        target_sequence_length: int = 200,
        augment: bool = False,
        augmentation_config: dict | None = None,
        use_dynamic_padding: bool = False,
        demographics_data: pl.DataFrame | None = None,
        demographics_config: dict | None = None,
        enable_handedness_aug: bool = False,
        handedness_flip_prob: float = 0.5,
    ):
        """
        初期化.

        Args:
            df: Polarsデータフレーム
            target_sequence_length: 目標シーケンス長（固定長モード時）
            augment: データ拡張フラグ
            augmentation_config: データ拡張設定
            use_dynamic_padding: 動的パディングを使用するかどうか
            demographics_data: Demographics特徴量のデータフレーム（オプショナル）
            demographics_config: Demographics統合設定（オプショナル）
            enable_handedness_aug: 利き手反転オーグメンテーションを有効にするか
            handedness_flip_prob: 利き手反転の確率 (0.0-1.0)
        """
        self.df = df
        self.target_sequence_length = target_sequence_length
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.use_dynamic_padding = use_dynamic_padding

        # Demographics設定
        self.demographics_data = demographics_data
        self.demographics_config = demographics_config or {}
        self.use_demographics = (demographics_data is not None) and self.demographics_config.get("enabled", False)

        # 利き手反転オーグメンテーション設定
        self.enable_handedness_aug = enable_handedness_aug and augment  # augmentがTrueの時のみ有効
        if self.enable_handedness_aug and demographics_data is not None:
            self.handedness_aug = HandednessAugmentation(demographics_data, handedness_flip_prob)
            print(f"Handedness augmentation enabled with flip probability: {handedness_flip_prob}")
        else:
            self.handedness_aug = None
            if self.enable_handedness_aug:
                print("Handedness augmentation disabled: demographics_data is required")

        if self.use_demographics:
            print(f"Loading demographics data with {len(demographics_data)} subjects...")
            # subjectをキーとしてdictsに変換（高速アクセス用）
            self.demographics_dict = demographics_data.to_dict(as_series=False)
            self.subject_to_demographics = {}
            for i, subject in enumerate(self.demographics_dict["subject"]):
                self.subject_to_demographics[subject] = {
                    key: values[i] for key, values in self.demographics_dict.items() if key != "subject"
                }
            print(f"Demographics loaded for {len(self.subject_to_demographics)} subjects")

            # スケーリングパラメータの設定（設定値ベース）
            self._setup_scaling_params()

        # IMU列の定義（基本IMU + 物理ベース特徴量 + 高度な特徴量）
        self.imu_cols = [
            # 基本IMU
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            # 基本物理特徴量
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "linear_acc_mag",
            "linear_acc_mag_jerk",
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            "angular_distance",
            # 累積和
            "linear_acc_x_cumsum",
            "linear_acc_y_cumsum",
            "linear_acc_z_cumsum",
            "linear_acc_mag_cumsum",
            # 差分
            "linear_acc_x_diff",
            "linear_acc_y_diff",
            "linear_acc_z_diff",
            "linear_acc_mag_diff",
            "angular_vel_x_diff",
            "angular_vel_y_diff",
            "angular_vel_z_diff",
            # 長期差分 (5, 10, 20ステップ)
            "linear_acc_x_diff_5",
            "linear_acc_x_diff_10",
            "linear_acc_x_diff_20",
            "linear_acc_y_diff_5",
            "linear_acc_y_diff_10",
            "linear_acc_y_diff_20",
            "linear_acc_z_diff_5",
            "linear_acc_z_diff_10",
            "linear_acc_z_diff_20",
            "linear_acc_mag_diff_5",
            "linear_acc_mag_diff_10",
            "linear_acc_mag_diff_20",
            # シフト/ラグ
            "linear_acc_x_lag_1",
            "linear_acc_x_lag_3",
            "linear_acc_x_lag_5",
            "linear_acc_y_lag_1",
            "linear_acc_y_lag_3",
            "linear_acc_y_lag_5",
            "linear_acc_z_lag_1",
            "linear_acc_z_lag_3",
            "linear_acc_z_lag_5",
            "linear_acc_mag_lag_1",
            "linear_acc_mag_lag_3",
            "linear_acc_mag_lag_5",
            # 中央値からの差分
            "linear_acc_x_median_diff",
            "linear_acc_y_median_diff",
            "linear_acc_z_median_diff",
            "linear_acc_mag_median_diff",
            # 統計的特徴量
            "linear_acc_mag_rolling_mean",
            "linear_acc_mag_rolling_std",
            "linear_acc_energy",
            "linear_acc_x_zero_cross",
            "linear_acc_y_zero_cross",
            "linear_acc_z_zero_cross",
        ]

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

    def _setup_scaling_params(self):
        """設定値からスケーリングパラメータを設定."""
        if not self.use_demographics:
            return

        print("Setting up demographics scaling parameters from config...")

        # demographics_configから直接パラメータを取得
        self.scaling_params = {
            "age": (self.demographics_config.get("age_min", 8.0), self.demographics_config.get("age_max", 60.0)),
            "height_cm": (
                self.demographics_config.get("height_min", 130.0),
                self.demographics_config.get("height_max", 195.0),
            ),
            "shoulder_to_wrist_cm": (
                self.demographics_config.get("shoulder_to_wrist_min", 35.0),
                self.demographics_config.get("shoulder_to_wrist_max", 75.0),
            ),
            "elbow_to_wrist_cm": (
                self.demographics_config.get("elbow_to_wrist_min", 15.0),
                self.demographics_config.get("elbow_to_wrist_max", 50.0),
            ),
        }

        for feature, (min_val, max_val) in self.scaling_params.items():
            print(f"{feature}: using config range ({min_val:.2f}, {max_val:.2f})")

        print(f"Scaling parameters set for {len(self.scaling_params)} features from config")

    def _get_demographics_for_subject(
        self, subject: str, flip_handedness: bool = False
    ) -> dict[str, torch.Tensor] | None:
        """被験者のdemographics特徴量を取得してテンソル化."""
        if not self.use_demographics or subject not in self.subject_to_demographics:
            return None

        demographics_raw = self.subject_to_demographics[subject]
        demographics_tensors = {}

        # カテゴリカル特徴量の処理
        categorical_features = self.demographics_config.get(
            "categorical_features", ["adult_child", "sex", "handedness"]
        )
        for feature in categorical_features:
            if feature in demographics_raw:
                value = demographics_raw[feature]

                # 利き手反転処理
                if feature == "handedness" and flip_handedness:
                    value = 1 - value  # 0↔1反転

                # NaN値やNone値の処理
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0  # デフォルト値
                demographics_tensors[feature] = torch.tensor(int(value), dtype=torch.long)

        # 数値特徴量の処理（スケーリング済み）
        numerical_features = self.demographics_config.get(
            "numerical_features", ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
        )
        for feature in numerical_features:
            if feature in demographics_raw:
                value = demographics_raw[feature]
                # NaN値やNone値の処理
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    # デフォルト値を中央値とする
                    min_val, max_val = self.scaling_params.get(feature, (0.0, 1.0))
                    value = (min_val + max_val) / 2

                # スケーリングパラメータを適用（範囲外値をクリッピング）
                min_val, max_val = self.scaling_params.get(feature, (0.0, 1.0))
                clipped_value = np.clip(float(value), min_val, max_val)

                demographics_tensors[feature] = torch.tensor(clipped_value, dtype=torch.float32)

        return demographics_tensors

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
        # 基本IMU列の存在確認（物理特徴量は後で追加されるため除外）
        basic_imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
        missing_cols = [col for col in basic_imu_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing basic IMU columns: {missing_cols}")

        # Polarsベクトル化処理を使用（欠損値マスク付き）
        self.sequence_data = self._preprocess_data_vectorized_with_mask()

        # sequence_id → subject マッピングを作成（demographics用）
        if self.use_demographics:
            self.sequence_to_subject = {}
            unique_data = self.df.select(["sequence_id", "subject"]).unique()
            for row in unique_data.iter_rows(named=True):
                self.sequence_to_subject[row["sequence_id"]] = row["subject"]
            print(f"Created sequence-to-subject mapping for {len(self.sequence_to_subject)} sequences")

    @timer_decorator("IMUDataset._add_physics_features")
    def _add_physics_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        物理ベースIMU特徴量を追加（完全ベクトル化版）.

        Args:
            df: 元のIMUデータを含むDataFrame

        Returns:
            物理特徴量が追加されたDataFrame
        """
        print("Computing physics-based IMU features (vectorized)...")

        # データをsequence_idとsequence_counterでソート
        df_sorted = df.sort(["sequence_id", "sequence_counter"]).lazy()

        # 線形加速度（重力除去）- 全体に適用
        linear_acc_df = remove_gravity_from_acc_pl(df_sorted)

        # 角速度（ベクトル化、シーケンス境界尊重）
        angular_vel_df = calculate_angular_velocity_from_quat_pl(df_sorted)

        # 角距離（ベクトル化、シーケンス境界尊重）
        angular_dist_df = calculate_angular_distance_pl(df_sorted)

        # 全ての物理特徴量を一度に計算して結合
        df_with_physics = (
            pl.concat([df_sorted, linear_acc_df, angular_vel_df, angular_dist_df], how="horizontal")
            .with_columns(
                [
                    # 線形加速度の大きさ
                    (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                    .sqrt()
                    .alias("linear_acc_mag"),
                    # 線形加速度大きさのジャーク（シーケンス境界を尊重）
                    (
                        (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                        .sqrt()
                        .diff()
                        .over("sequence_id")
                        .fill_null(0.0)
                    ).alias("linear_acc_mag_jerk"),
                ]
            )
            .collect()
        )

        # 高度な物理ベース特徴量を追加
        df_with_advanced = self._add_advanced_physics_features(df_with_physics)

        print(f"Added physics features. DataFrame shape: {df_with_advanced.shape}")
        return df_with_advanced

    @timer_decorator("IMUDataset._add_advanced_physics_features")
    def _add_advanced_physics_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        高度な物理ベースIMU特徴量を追加.

        Args:
            df: 基本物理特徴量が追加されたDataFrame

        Returns:
            高度な特徴量が追加されたDataFrame
        """
        print("Computing advanced physics-based IMU features...")

        # ソートしてからグループごとに処理
        df_sorted = df.sort(["sequence_id", "sequence_counter"])
        
        def process_sequence_group(group_df):
            """各sequence_idグループに対して特徴量を計算."""
            return group_df.with_columns([
                # 1. 累積和 (Cumulative Sums)
                pl.col("linear_acc_x").cum_sum().alias("linear_acc_x_cumsum"),
                pl.col("linear_acc_y").cum_sum().alias("linear_acc_y_cumsum"),
                pl.col("linear_acc_z").cum_sum().alias("linear_acc_z_cumsum"),
                pl.col("linear_acc_mag").cum_sum().alias("linear_acc_mag_cumsum"),
                
                # 2. 差分 (Diffs)
                pl.col("linear_acc_x").diff().fill_null(0.0).alias("linear_acc_x_diff"),
                pl.col("linear_acc_y").diff().fill_null(0.0).alias("linear_acc_y_diff"),
                pl.col("linear_acc_z").diff().fill_null(0.0).alias("linear_acc_z_diff"),
                pl.col("linear_acc_mag").diff().fill_null(0.0).alias("linear_acc_mag_diff"),
                pl.col("angular_vel_x").diff().fill_null(0.0).alias("angular_vel_x_diff"),
                pl.col("angular_vel_y").diff().fill_null(0.0).alias("angular_vel_y_diff"),
                pl.col("angular_vel_z").diff().fill_null(0.0).alias("angular_vel_z_diff"),
                
                # 3. 長期差分 (Longer Ranged Diffs)
                # lag=5
                (pl.col("linear_acc_x") - pl.col("linear_acc_x").shift(5)).fill_null(0.0).alias("linear_acc_x_diff_5"),
                (pl.col("linear_acc_y") - pl.col("linear_acc_y").shift(5)).fill_null(0.0).alias("linear_acc_y_diff_5"),
                (pl.col("linear_acc_z") - pl.col("linear_acc_z").shift(5)).fill_null(0.0).alias("linear_acc_z_diff_5"),
                (pl.col("linear_acc_mag") - pl.col("linear_acc_mag").shift(5)).fill_null(0.0).alias("linear_acc_mag_diff_5"),
                # lag=10
                (pl.col("linear_acc_x") - pl.col("linear_acc_x").shift(10)).fill_null(0.0).alias("linear_acc_x_diff_10"),
                (pl.col("linear_acc_y") - pl.col("linear_acc_y").shift(10)).fill_null(0.0).alias("linear_acc_y_diff_10"),
                (pl.col("linear_acc_z") - pl.col("linear_acc_z").shift(10)).fill_null(0.0).alias("linear_acc_z_diff_10"),
                (pl.col("linear_acc_mag") - pl.col("linear_acc_mag").shift(10)).fill_null(0.0).alias("linear_acc_mag_diff_10"),
                # lag=20
                (pl.col("linear_acc_x") - pl.col("linear_acc_x").shift(20)).fill_null(0.0).alias("linear_acc_x_diff_20"),
                (pl.col("linear_acc_y") - pl.col("linear_acc_y").shift(20)).fill_null(0.0).alias("linear_acc_y_diff_20"),
                (pl.col("linear_acc_z") - pl.col("linear_acc_z").shift(20)).fill_null(0.0).alias("linear_acc_z_diff_20"),
                (pl.col("linear_acc_mag") - pl.col("linear_acc_mag").shift(20)).fill_null(0.0).alias("linear_acc_mag_diff_20"),
                
                # 4. シフト/ラグ (Shifts/Lags)
                # lag=1
                pl.col("linear_acc_x").shift(1).fill_null(0.0).alias("linear_acc_x_lag_1"),
                pl.col("linear_acc_y").shift(1).fill_null(0.0).alias("linear_acc_y_lag_1"),
                pl.col("linear_acc_z").shift(1).fill_null(0.0).alias("linear_acc_z_lag_1"),
                pl.col("linear_acc_mag").shift(1).fill_null(0.0).alias("linear_acc_mag_lag_1"),
                # lag=3
                pl.col("linear_acc_x").shift(3).fill_null(0.0).alias("linear_acc_x_lag_3"),
                pl.col("linear_acc_y").shift(3).fill_null(0.0).alias("linear_acc_y_lag_3"),
                pl.col("linear_acc_z").shift(3).fill_null(0.0).alias("linear_acc_z_lag_3"),
                pl.col("linear_acc_mag").shift(3).fill_null(0.0).alias("linear_acc_mag_lag_3"),
                # lag=5
                pl.col("linear_acc_x").shift(5).fill_null(0.0).alias("linear_acc_x_lag_5"),
                pl.col("linear_acc_y").shift(5).fill_null(0.0).alias("linear_acc_y_lag_5"),
                pl.col("linear_acc_z").shift(5).fill_null(0.0).alias("linear_acc_z_lag_5"),
                pl.col("linear_acc_mag").shift(5).fill_null(0.0).alias("linear_acc_mag_lag_5"),
                
                # 5. シーケンス中央値からの差分
                (pl.col("linear_acc_x") - pl.col("linear_acc_x").median()).alias("linear_acc_x_median_diff"),
                (pl.col("linear_acc_y") - pl.col("linear_acc_y").median()).alias("linear_acc_y_median_diff"),
                (pl.col("linear_acc_z") - pl.col("linear_acc_z").median()).alias("linear_acc_z_median_diff"),
                (pl.col("linear_acc_mag") - pl.col("linear_acc_mag").median()).alias("linear_acc_mag_median_diff"),
                
                # 6. ローリング統計（ウィンドウサイズ=10）
                pl.col("linear_acc_mag")
                .rolling_mean(window_size=10, min_samples=1)
                .alias("linear_acc_mag_rolling_mean"),
                pl.col("linear_acc_mag")
                .rolling_std(window_size=10, min_samples=1)
                .fill_null(0.0)
                .alias("linear_acc_mag_rolling_std"),
                
                # 7. エネルギー特徴（累積二乗和）
                (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                .cum_sum()
                .alias("linear_acc_energy"),
                
                # 8. ゼロクロス率（符号変化の累積カウント）
                ((pl.col("linear_acc_x") * pl.col("linear_acc_x").shift(1)) < 0)
                .cast(pl.Int32)
                .cum_sum()
                .alias("linear_acc_x_zero_cross"),
                ((pl.col("linear_acc_y") * pl.col("linear_acc_y").shift(1)) < 0)
                .cast(pl.Int32)
                .cum_sum()
                .alias("linear_acc_y_zero_cross"),
                ((pl.col("linear_acc_z") * pl.col("linear_acc_z").shift(1)) < 0)
                .cast(pl.Int32)
                .cum_sum()
                .alias("linear_acc_z_zero_cross"),
            ])
        
        # group_byとmap_groupsを使って各sequence_idごとに処理
        df_with_advanced = df_sorted.group_by("sequence_id", maintain_order=True).map_groups(process_sequence_group)

        print(f"Added advanced physics features. New shape: {df_with_advanced.shape}")
        return df_with_advanced

    @timer_decorator("IMUDataset._preprocess_data_vectorized_with_mask")
    def _preprocess_data_vectorized_with_mask(self) -> dict:
        """Polarsを使ったベクトル化前処理（欠損値マスク付き）."""
        print("Starting vectorized preprocessing with missing value mask...")

        # 物理ベース特徴量を計算
        print("Calculating physics-based IMU features...")
        df_with_physics = self._add_physics_features(self.df)

        # 基本IMU列のみ（物理特徴量は後で追加）
        basic_imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

        # 全ての物理特徴量（基本 + 高度）
        physics_cols = [col for col in self.imu_cols if col not in basic_imu_cols]

        # シーケンスごとの前処理済みデータを一括取得（欠損値処理前）
        processed_data = (
            df_with_physics.sort(["sequence_id", "sequence_counter"])
            .group_by("sequence_id")
            .agg(
                [
                    # 基本IMUデータをリストで集約（欠損値はそのまま）
                    *[pl.col(col) for col in basic_imu_cols],
                    # 物理特徴量をリストで集約
                    *[pl.col(col) for col in physics_cols],
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

    def _apply_augmentation(self, imu_data: np.ndarray, subject_id: str | None = None) -> tuple[np.ndarray, bool]:
        """データ拡張の適用."""
        was_handedness_flipped = False

        if not self.augment:
            return imu_data, was_handedness_flipped

        # 利き手反転オーグメンテーション（最初に適用）
        if self.handedness_aug is not None:
            if self.handedness_aug.should_flip():
                imu_data = self.handedness_aug.flip_imu_data(imu_data, self.imu_cols)
                was_handedness_flipped = True

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

        return imu_data, was_handedness_flipped

    def __len__(self) -> int:
        """データセットサイズ."""
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """データ取得."""
        seq_id = self.sequence_ids[idx]
        data = self.sequence_data[seq_id]

        # 被験者IDを取得（利き手反転オーグメンテーション用）
        subject_id = self.sequence_to_subject.get(seq_id) if self.use_demographics else None

        # IMUデータの取得とデータ拡張（反転フラグも取得）
        imu_data = data["imu"].copy()
        imu_data, was_handedness_flipped = self._apply_augmentation(imu_data, subject_id)

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

        # Demographics特徴量を追加（反転フラグを渡す）
        if self.use_demographics:
            subject = self.sequence_to_subject.get(seq_id)
            demographics = self._get_demographics_for_subject(subject, was_handedness_flipped)
            if demographics is not None:
                result["demographics"] = demographics

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

        # Demographics データの読み込み（設定で有効な場合）
        self.demographics_data = None
        if config.demographics.enabled:
            try:
                print(f"Loading demographics data from {config.data.demographics_train_path}...")
                self.demographics_data = pl.read_csv(config.data.demographics_train_path)
                print(f"Loaded demographics data for {len(self.demographics_data)} subjects")
            except Exception as e:
                print(f"Warning: Failed to load demographics data: {e}")
                print("Proceeding without demographics data")
                self.demographics_data = None

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
                demographics_data=self.demographics_data,
                demographics_config=self.config.demographics.model_dump(),
                enable_handedness_aug=self.config.augmentation.enable_handedness_flip,
                handedness_flip_prob=self.config.augmentation.handedness_flip_prob,
            )

            print(f"Creating validation dataset ({len(val_sequence_ids)} rows)...")
            self.val_dataset = IMUDataset(
                val_sequence_ids,
                target_sequence_length=self.target_sequence_length,
                augment=False,  # 検証時はオーグメンテーション無効
                use_dynamic_padding=self.use_dynamic_padding,
                demographics_data=self.demographics_data,
                demographics_config=self.config.demographics.model_dump(),
                enable_handedness_aug=False,  # 検証時は利き手反転も無効
                handedness_flip_prob=0.0,
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

            def collate_fn(batch):
                return dynamic_collate_fn(batch, percentile_max_length=self.percentile_max_length)
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

            def collate_fn(batch):
                return dynamic_collate_fn(batch, percentile_max_length=self.percentile_max_length)
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

    def __init__(
        self,
        sequence_df: pl.DataFrame,
        target_sequence_length: int = 200,
        use_demographics: bool = False,
        demographics_data: pl.DataFrame | None = None,
        subject: str | None = None,
        demographics_config: dict | None = None,
    ):
        """
        初期化.

        Args:
            sequence_df: 単一シーケンスのPolarsデータフレーム
            target_sequence_length: 目標シーケンス長
            use_demographics: Demographics統合を使用するかどうか
            demographics_data: Demographics特徴量のデータフレーム（オプショナル）
            subject: 対象のsubject ID（Demographics取得用、オプショナル）
            demographics_config: Demographics統合設定（オプショナル）
        """
        self.df = sequence_df
        self.target_sequence_length = target_sequence_length
        self.augment = False  # 推論時は拡張なし

        # Demographics設定
        self.use_demographics = use_demographics and demographics_data is not None
        self.demographics_data = demographics_data
        self.demographics_config = demographics_config or {}

        # Subject の取得（引数で指定されない場合はsequence_dfから取得）
        if subject is not None:
            self.subject = subject
        elif "subject" in sequence_df.columns:
            self.subject = sequence_df.get_column("subject")[0]
        else:
            self.subject = None

        # Demographics データ前処理
        if self.use_demographics:
            self._setup_demographics()

        # IMU列の定義（基本IMU + 物理ベース特徴量）
        self.imu_cols = [
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "linear_acc_mag",
            "linear_acc_mag_jerk",
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            "angular_distance",
        ]

        # シーケンスIDを取得(単一シーケンスなので最初の値)
        self.sequence_id = sequence_df.get_column("sequence_id")[0]

        # データ前処理
        self._preprocess_data()

    def _add_physics_features_single(self, df: pl.DataFrame) -> pl.DataFrame:
        """単一シーケンス用の物理特徴量計算."""
        # LazyFrameに変換
        df_lazy = df.lazy()

        # 線形加速度（重力除去）
        linear_acc_df = remove_gravity_from_acc_pl(df_lazy)

        # 角速度（単一シーケンスなのでsequence_idなしで計算）
        angular_vel_df = calculate_angular_velocity_from_quat_pl(df_lazy)

        # 角距離
        angular_dist_df = calculate_angular_distance_pl(df_lazy)

        # 全ての物理特徴量を結合
        df_with_physics = (
            pl.concat([df_lazy, linear_acc_df, angular_vel_df, angular_dist_df], how="horizontal")
            .with_columns(
                [
                    # 線形加速度の大きさ
                    (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                    .sqrt()
                    .alias("linear_acc_mag"),
                    # 線形加速度大きさのジャーク
                    (
                        (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                        .sqrt()
                        .diff()
                        .fill_null(0.0)
                    ).alias("linear_acc_mag_jerk"),
                ]
            )
            .collect()
        )

        return df_with_physics

    def _setup_demographics(self):
        """Demographics データの前処理."""
        if not self.use_demographics or self.demographics_data is None:
            return

        print(f"Setting up demographics for subject: {self.subject}")

        # Demographics データを dict 形式に変換（高速アクセス用）
        demographics_dict = self.demographics_data.to_dict(as_series=False)

        # subject-to-demographics マッピングを作成
        self.subject_to_demographics = {}
        for i, subj in enumerate(demographics_dict["subject"]):
            self.subject_to_demographics[subj] = {
                key: values[i] for key, values in demographics_dict.items() if key != "subject"
            }

        # スケーリングパラメータの設定（設定値ベース）
        self._setup_scaling_params()

        print(f"Demographics setup completed for {len(self.subject_to_demographics)} subjects")

    def _setup_scaling_params(self):
        """設定値からスケーリングパラメータを設定."""
        if not self.use_demographics:
            return

        # demographics_configから直接パラメータを取得
        self.scaling_params = {
            "age": (self.demographics_config.get("age_min", 8.0), self.demographics_config.get("age_max", 60.0)),
            "height_cm": (
                self.demographics_config.get("height_min", 130.0),
                self.demographics_config.get("height_max", 195.0),
            ),
            "shoulder_to_wrist_cm": (
                self.demographics_config.get("shoulder_to_wrist_min", 35.0),
                self.demographics_config.get("shoulder_to_wrist_max", 75.0),
            ),
            "elbow_to_wrist_cm": (
                self.demographics_config.get("elbow_to_wrist_min", 15.0),
                self.demographics_config.get("elbow_to_wrist_max", 50.0),
            ),
        }

    def _get_demographics_for_subject(self) -> dict[str, torch.Tensor] | None:
        """被験者のdemographics特徴量を取得してテンソル化."""
        if not self.use_demographics or self.subject is None or self.subject not in self.subject_to_demographics:
            return None

        demographics_raw = self.subject_to_demographics[self.subject]
        demographics_tensors = {}

        # カテゴリカル特徴量の処理
        categorical_features = self.demographics_config.get(
            "categorical_features", ["adult_child", "sex", "handedness"]
        )
        for feature in categorical_features:
            if feature in demographics_raw:
                value = demographics_raw[feature]
                # NaN値やNone値の処理
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0  # デフォルト値
                demographics_tensors[feature] = torch.tensor(int(value), dtype=torch.long)

        # 数値特徴量の処理（スケーリング済み）
        numerical_features = self.demographics_config.get(
            "numerical_features", ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
        )
        for feature in numerical_features:
            if feature in demographics_raw:
                value = demographics_raw[feature]
                # NaN値やNone値の処理
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    # デフォルト値を中央値とする
                    min_val, max_val = self.scaling_params.get(feature, (0.0, 1.0))
                    value = (min_val + max_val) / 2

                # スケーリングパラメータを適用（範囲外値をクリッピング）
                min_val, max_val = self.scaling_params.get(feature, (0.0, 1.0))
                clipped_value = np.clip(float(value), min_val, max_val)

                demographics_tensors[feature] = torch.tensor(clipped_value, dtype=torch.float32)

        return demographics_tensors

    def _preprocess_data(self):
        """データの前処理(単一シーケンス用)."""
        # 基本IMU列の存在確認（物理特徴量は後で計算）
        basic_imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
        missing_cols = [col for col in basic_imu_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing basic IMU columns: {missing_cols}")

        # シーケンスを時間順でソート
        seq_df = self.df.sort("sequence_counter")

        # 物理特徴量を計算（単一シーケンス用に簡略化）
        seq_df = self._add_physics_features_single(seq_df)

        # IMUデータの抽出（物理特徴量込み）
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
        """データ取得（単一シーケンス用）."""
        if idx != 0:
            raise IndexError("Single sequence dataset only has one item")

        # IMUデータの取得
        imu_data = self.imu_data.copy()

        # テンソル化 [seq_len, features] -> [features, seq_len]
        imu_tensor = torch.tensor(imu_data.T, dtype=torch.float32)

        # missing_maskからattention_maskを作成（欠損位置を無視)
        attention_mask = torch.tensor(~self.missing_mask, dtype=torch.bool)  # 欠損位置でFalse、正常位置でTrue

        result = {"imu": imu_tensor, "attention_mask": attention_mask, "sequence_id": self.sequence_id}

        # Demographicsデータが利用可能な場合は追加
        if self.use_demographics:
            demographics = self._get_demographics_for_subject()
            if demographics is not None:
                result["demographics"] = demographics

        return result


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
