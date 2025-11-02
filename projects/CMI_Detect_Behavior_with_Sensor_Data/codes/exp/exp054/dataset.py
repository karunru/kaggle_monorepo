"""exp036 dataset module for IMU data processing with tsai transforms and RandAugment framework."""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as lightning
import torch
from config import AugmentationConfig, Config, DemographicsConfig

# Human Normalization imports
from human_normalization import HNConfig, compute_hn_features, get_hn_feature_columns
from scipy import interpolate
from scipy.spatial.transform import Rotation, Rotation as R

# Imports from project root
from src.utils.timer import timer_decorator
from src.validation.factory import get_validation
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from tsai.data.transforms import TSMagAddNoise, TSMagScale, TSMaskOut, TSTimeWarp


# Jiazhuang notebook feature engineering functions
def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity from acceleration data using quaternion rotation."""
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)

    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]

    return linear_accel


def calculate_angular_velocity_from_quat(rot_data, time_delta=1 / 200):
    """Calculate angular velocity from quaternion data."""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]

        if (
            np.all(np.isnan(q_t))
            or np.all(np.isclose(q_t, 0))
            or np.all(np.isnan(q_t_plus_dt))
            or np.all(np.isclose(q_t_plus_dt, 0))
        ):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt

            # Convert delta rotation to angular velocity vector
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass

    return angular_vel


def calculate_angular_distance(rot_data):
    """Calculate angular distance between consecutive quaternions."""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i + 1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Calculate relative rotation and its angle
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass

    return angular_dist


def feature_engineering_jiazhuang(df):
    """
    Apply jiazhuang notebook feature engineering to create 19 IMU features.

    Features:
    1. acc_x, acc_y, acc_z (original)
    2. rot_w, rot_x, rot_y, rot_z (original)
    3. acc_mag (magnitude)
    4. rot_angle (rotation angle)
    5. acc_mag_jerk (jerk of acceleration magnitude)
    6. rot_angle_vel (angular velocity)
    7. linear_acc_x, linear_acc_y, linear_acc_z (gravity-removed)
    8. linear_acc_mag (magnitude of linear acceleration)
    9. linear_acc_mag_jerk (jerk of linear acceleration magnitude)
    10. angular_vel_x, angular_vel_y, angular_vel_z (angular velocity)
    11. angular_distance (angular distance)
    """
    # IMU magnitude
    df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)

    # IMU angle
    df["rot_angle"] = 2 * np.arccos(df["rot_w"].clip(-1, 1))

    # IMU jerk, angular velocity
    df["acc_mag_jerk"] = df.groupby("sequence_id")["acc_mag"].diff().fillna(0)
    df["rot_angle_vel"] = df.groupby("sequence_id")["rot_angle"].diff().fillna(0)

    # Remove gravity
    def get_linear_accel(sequence_df):
        res = remove_gravity_from_acc(
            sequence_df[["acc_x", "acc_y", "acc_z"]], sequence_df[["rot_x", "rot_y", "rot_z", "rot_w"]]
        )
        res = pd.DataFrame(res, columns=["linear_acc_x", "linear_acc_y", "linear_acc_z"], index=sequence_df.index)
        return res

    linear_accel_df = df.groupby("sequence_id").apply(get_linear_accel, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel("sequence_id")
    df = df.join(linear_accel_df)

    df["linear_acc_mag"] = np.sqrt(df["linear_acc_x"] ** 2 + df["linear_acc_y"] ** 2 + df["linear_acc_z"] ** 2)
    df["linear_acc_mag_jerk"] = df.groupby("sequence_id")["linear_acc_mag"].diff().fillna(0)

    # Calculate angular velocity
    def calc_angular_velocity(sequence_df):
        res = calculate_angular_velocity_from_quat(sequence_df[["rot_x", "rot_y", "rot_z", "rot_w"]])
        res = pd.DataFrame(res, columns=["angular_vel_x", "angular_vel_y", "angular_vel_z"], index=sequence_df.index)
        return res

    angular_velocity_df = df.groupby("sequence_id").apply(calc_angular_velocity, include_groups=False)
    angular_velocity_df = angular_velocity_df.droplevel("sequence_id")
    df = df.join(angular_velocity_df)

    # Calculate angular distance
    def calc_angular_distance(sequence_df):
        res = calculate_angular_distance(sequence_df[["rot_x", "rot_y", "rot_z", "rot_w"]])
        res = pd.DataFrame(res, columns=["angular_distance"], index=sequence_df.index)
        return res

    angular_distance_df = df.groupby("sequence_id").apply(calc_angular_distance, include_groups=False)
    angular_distance_df = angular_distance_df.droplevel("sequence_id")
    df = df.join(angular_distance_df)

    # Define the 19 IMU features
    imu_feature_names = [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        "acc_mag",
        "rot_angle",
        "acc_mag_jerk",
        "rot_angle_vel",
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

    # Fill missing values and convert to float32
    df[imu_feature_names] = df[imu_feature_names].ffill().bfill().fillna(0).astype("float32")

    return df, imu_feature_names


def calculate_sequence_statistics(df):
    """
    Calculate sequence-level statistical features for each sequence_id using Polars LazyFrame for high performance.

    Args:
        df: Polars DataFrame or LazyFrame with IMU features

    Returns:
        Polars DataFrame with statistical features for each sequence_id
    """
    base_feats = [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        "acc_mag",
        "rot_angle",
        "acc_mag_jerk",
        "rot_angle_vel",
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

    # Convert to LazyFrame if needed (from pandas or Polars DataFrame)
    if hasattr(df, "lazy"):
        # Polars DataFrame
        lf = df.lazy()
    elif hasattr(df, "to_pandas"):
        # Polars LazyFrame (already lazy)
        lf = df
    else:
        # pandas DataFrame - convert to Polars LazyFrame
        lf = pl.from_pandas(df).lazy()

    # Filter base_feats to only include columns that exist in the dataframe
    # Use collect_schema() to avoid performance warning
    schema_columns = lf.collect_schema().names()
    available_feats = [feat for feat in base_feats if feat in schema_columns]

    # Build aggregation expressions
    agg_exprs = []

    # Basic statistics for each feature
    for feat in available_feats:
        agg_exprs.extend(
            [
                pl.col(feat).min().alias(f"{feat}_min"),
                pl.col(feat).max().alias(f"{feat}_max"),
                pl.col(feat).mean().alias(f"{feat}_mean"),
                pl.col(feat).median().alias(f"{feat}_median"),
                pl.col(feat).quantile(0.25).alias(f"{feat}_q25"),
                pl.col(feat).quantile(0.75).alias(f"{feat}_q75"),
                pl.col(feat).std().alias(f"{feat}_std"),
                pl.col(feat).skew().alias(f"{feat}_skew"),
                # Zero crossing count: count sign changes
                (pl.col(feat).sign().diff().abs() == 2).sum().alias(f"{feat}_zero_crossings"),
            ]
        )

    # Sequence length
    agg_exprs.append(pl.len().alias("sequence_length"))

    # Sequence counter range (if column exists)
    if "sequence_counter" in schema_columns:
        agg_exprs.append(
            (pl.col("sequence_counter").max() - pl.col("sequence_counter").min()).alias("sequence_counter_range")
        )
    else:
        agg_exprs.append(pl.lit(0).alias("sequence_counter_range"))

    # Execute the group_by aggregation
    stats_df = lf.group_by("sequence_id").agg(agg_exprs).collect()

    return stats_df


def normalize_statistical_features(stats_df, scaling_params=None):
    """
    Normalize statistical features using StandardScaler-like normalization with Polars for high performance.

    Args:
        stats_df: Polars DataFrame with statistical features
        scaling_params: Dictionary with 'mean' and 'std' for normalization

    Returns:
        Normalized Polars DataFrame and scaling parameters
    """
    # Exclude sequence_id from normalization
    feature_cols = [col for col in stats_df.columns if col != "sequence_id"]

    if scaling_params is None:
        # Calculate scaling parameters using Polars efficient operations
        scaling_params = {}

        # Calculate all means and stds in one pass for efficiency
        stats = stats_df.select(
            [pl.col(col).mean().alias(f"{col}_mean") for col in feature_cols]
            + [pl.col(col).std().alias(f"{col}_std") for col in feature_cols]
        ).row(0)

        # Create scaling_params dictionary
        for i, col in enumerate(feature_cols):
            mean_val = stats[i]  # means come first
            std_val = stats[i + len(feature_cols)]  # stds come second
            scaling_params[col] = {"mean": mean_val, "std": std_val}

    # Apply normalization using Polars expressions
    normalized_exprs = [pl.col("sequence_id")]  # Keep sequence_id as is

    for col in feature_cols:
        mean_val = scaling_params[col]["mean"]
        std_val = scaling_params[col]["std"]

        if std_val > 0:
            # Standard normalization: (x - mean) / std
            normalized_exprs.append(((pl.col(col) - mean_val) / std_val).alias(col))
        else:
            # Handle zero std case
            normalized_exprs.append(pl.lit(0.0).alias(col))

    # Apply all normalizations in one select operation
    normalized_df = stats_df.select(normalized_exprs)

    return normalized_df, scaling_params


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
    nine_class_labels = torch.zeros(batch_size, dtype=torch.long)
    sequence_ids = []
    gestures = []

    # 統計量特徴量のバッチ処理
    statistical_features_batch = None
    if "statistical_features" in batch[0]:
        stat_feat_dim = batch[0]["statistical_features"].shape[0]
        statistical_features_batch = torch.zeros(batch_size, stat_feat_dim, dtype=torch.float32)

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
        nine_class_labels[i] = item["nine_class_label"]
        sequence_ids.append(item["sequence_id"])
        gestures.append(item["gesture"])

        # 統計量特徴量をバッチに追加
        if statistical_features_batch is not None and "statistical_features" in item:
            statistical_features_batch[i] = item["statistical_features"]

    result = {
        "imu": imu_batch,
        "attention_mask": attention_mask,
        "multiclass_label": multiclass_labels,
        "binary_label": binary_labels,
        "nine_class_label": nine_class_labels,
        "sequence_id": sequence_ids,
        "gesture": gestures,
        "max_length": max_length,  # デバッグ用
        "original_lengths": lengths,  # デバッグ用
    }

    # 統計量特徴量を結果に追加
    if statistical_features_batch is not None:
        result["statistical_features"] = statistical_features_batch

    return result


# IMU回転変換 (RandTransform継承)
class IMURotationTransform:
    """IMU 3軸データの回転変換（物理的に妥当な回転行列を適用）."""

    def __init__(self, angle_range: float = 0.2, imu_cols: list[str] = None):
        """
        Initialize IMU rotation transform.

        Args:
            angle_range: 回転角度の範囲（ラジアン）
            imu_cols: IMU列名のリスト
        """
        self.angle_range = angle_range
        self.imu_cols = imu_cols or []

    def __call__(self, x):
        """
        Apply random 3D rotation to IMU data.

        Args:
            x: IMU data tensor/array [seq_len, features] or [features, seq_len]

        Returns:
            Rotated IMU data
        """

        # Convert to numpy if tensor
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
            was_tensor = True
        else:
            x_np = x
            was_tensor = False

        # Check data format
        if x_np.shape[0] == len(self.imu_cols):  # [features, seq_len]
            is_features_first = True
            data = x_np.T  # Convert to [seq_len, features]
        else:  # [seq_len, features]
            is_features_first = False
            data = x_np

        # Find acceleration and rotation indices
        acc_indices = []
        rot_indices = []

        for i, col in enumerate(self.imu_cols):
            if col in ["acc_x", "acc_y", "acc_z", "linear_acc_x", "linear_acc_y", "linear_acc_z"]:
                acc_indices.append(i)
            elif col in ["rot_x", "rot_y", "rot_z"]:  # Quaternion rotation components
                rot_indices.append(i)

        if len(acc_indices) >= 3:
            # Generate random rotation
            angles = np.random.uniform(-self.angle_range, self.angle_range, 3)
            rotation = Rotation.from_euler("xyz", angles)
            rotation_matrix = rotation.as_matrix()

            # Apply rotation to acceleration data
            acc_data = data[:, acc_indices[:3]]  # Take first 3 acceleration components
            rotated_acc = acc_data @ rotation_matrix.T
            data[:, acc_indices[:3]] = rotated_acc

            # Apply rotation to linear acceleration if present
            linear_acc_start = 3
            if len(acc_indices) >= 6:
                linear_acc_data = data[:, acc_indices[linear_acc_start : linear_acc_start + 3]]
                rotated_linear_acc = linear_acc_data @ rotation_matrix.T
                data[:, acc_indices[linear_acc_start : linear_acc_start + 3]] = rotated_linear_acc

        # Convert back to original format
        if is_features_first:
            result = data.T
        else:
            result = data

        # Convert back to tensor if needed
        if was_tensor:
            return torch.from_numpy(result).float()
        return result


class HandednessAugmentation:
    """利き手反転データオーグメンテーション（RandTransform風に再実装）."""

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
        IMUデータの利き手反転（Y軸方向の反転）.

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

        # Y軸関連の列を反転
        y_axis_cols = ["acc_y", "rot_y", "linear_acc_y", "angular_vel_y"]

        for col in y_axis_cols:
            if col in imu_cols:
                col_idx = imu_cols.index(col)
                flipped_data[col_idx] = -flipped_data[col_idx]

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
        augmentation_config: AugmentationConfig | None = None,
        use_dynamic_padding: bool = False,
        demographics_data: pl.DataFrame | None = None,
        demographics_config: DemographicsConfig | None = None,
        enable_handedness_aug: bool = False,
        handedness_flip_prob: float = 0.5,
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
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
        self.augmentation_config = augmentation_config

        # ジェスチャーリストの設定（デフォルト値を設定）
        self.target_gestures = target_gestures or [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        ]
        self.non_target_gestures = non_target_gestures or [
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
        self.use_dynamic_padding = use_dynamic_padding

        # Demographics設定
        self.demographics_data = demographics_data
        self.demographics_config = demographics_config or {}
        self.use_demographics = (demographics_data is not None) and self.demographics_config.enabled

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

        # IMU列の定義（jiazhuang notebook compatible: 19 physical features）
        self.imu_cols = [
            # Original IMU features (7)
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            # Basic engineered features (4)
            "acc_mag",
            "rot_angle",
            "acc_mag_jerk",
            "rot_angle_vel",
            # Linear acceleration features (5)
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "linear_acc_mag",
            "linear_acc_mag_jerk",
            # Angular velocity features (3)
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            # Angular distance (1)
            "angular_distance",
        ]

        # Human Normalization特徴量の追加
        if self.use_demographics and self.demographics_config.hn_enabled:
            hn_config = HNConfig(
                hn_enabled=self.demographics_config.hn_enabled,
                hn_eps=self.demographics_config.hn_eps,
                hn_radius_min_max=self.demographics_config.hn_radius_min_max,
                hn_features=self.demographics_config.hn_features,
            )
            hn_cols = get_hn_feature_columns(hn_config)
            self.imu_cols.extend(hn_cols)
            print(f"Human Normalization enabled: added {len(hn_cols)} HN features")

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

        # 統計量特徴量の計算と正規化
        print("Calculating statistical features...")
        self._setup_statistical_features()
        print("Statistical features setup completed")

        # tsai transformsのセットアップ
        if self.augment:
            self.setup_tsai_augmentation()

    def _setup_scaling_params(self):
        """設定値からスケーリングパラメータを設定."""
        if not self.use_demographics:
            return

        print("Setting up demographics scaling parameters from config...")

        # demographics_configから直接パラメータを取得
        self.scaling_params = {
            "age": (self.demographics_config.age_min, self.demographics_config.age_max),
            "height_cm": (
                self.demographics_config.height_min,
                self.demographics_config.height_max,
            ),
            "shoulder_to_wrist_cm": (
                self.demographics_config.shoulder_to_wrist_min,
                self.demographics_config.shoulder_to_wrist_max,
            ),
            "elbow_to_wrist_cm": (
                self.demographics_config.elbow_to_wrist_min,
                self.demographics_config.elbow_to_wrist_max,
            ),
        }

        for feature, (min_val, max_val) in self.scaling_params.items():
            print(f"{feature}: using config range ({min_val:.2f}, {max_val:.2f})")

        print(f"Scaling parameters set for {len(self.scaling_params)} features from config")

    def _setup_statistical_features(self):
        """統計量特徴量を計算し、正規化パラメータを設定（Polars高速化版）."""
        # 物理特徴量を含むデータフレームを作成（_preprocess_dataと同じ処理）
        df_with_physics = self._add_physics_features(self.df)

        # 統計量特徴量を計算（物理特徴量を含むデータフレームを使用）
        self.stats_df = calculate_sequence_statistics(df_with_physics)

        # 統計量特徴量の正規化（Polars版）
        self.stats_df_normalized, self.stats_scaling_params = normalize_statistical_features(self.stats_df)

        # sequence_idをキーとした辞書に変換（Polars to_dict使用で高速化）
        stats_dict = self.stats_df_normalized.to_dict(as_series=False)

        self.sequence_to_stats = {}
        for i, seq_id in enumerate(stats_dict["sequence_id"]):
            # sequence_id以外の列を特徴量として取得
            stats_features = np.array(
                [stats_dict[col][i] for col in stats_dict.keys() if col != "sequence_id"], dtype=np.float32
            )
            self.sequence_to_stats[seq_id] = stats_features

        # 統計量特徴量の次元数を記録
        self.stats_feature_dim = len(stats_features)
        print(f"Statistical features dimension: {self.stats_feature_dim}")
        print(f"Statistical features calculated for {len(self.sequence_to_stats)} sequences")

    def _get_demographics_for_subject(
        self, subject: str, flip_handedness: bool = False
    ) -> dict[str, torch.Tensor] | None:
        """被験者のdemographics特徴量を取得してテンソル化."""
        if not self.use_demographics or subject not in self.subject_to_demographics:
            return None

        demographics_raw = self.subject_to_demographics[subject]
        demographics_tensors = {}

        # カテゴリカル特徴量の処理
        categorical_features = self.demographics_config.categorical_features
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
        numerical_features = self.demographics_config.numerical_features
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

        # 9クラス用のマッピングを作成（8 target + 1 non-target）
        self.gesture_to_nine_class_id = {}
        for idx, gesture in enumerate(self.target_gestures):
            self.gesture_to_nine_class_id[gesture] = idx  # 0-7: target gestures
        for gesture in self.non_target_gestures:
            self.gesture_to_nine_class_id[gesture] = 8  # 8: all non-target gestures

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
        jiazhuang notebook compatible: 19 IMU physical features.

        Args:
            df: 元のIMUデータを含むDataFrame

        Returns:
            物理特徴量が追加されたDataFrame (19 features total)
        """
        print("Computing jiazhuang notebook compatible IMU features (19 features)...")

        # データをsequence_idとsequence_counterでソート
        df_sorted = df.sort(["sequence_id", "sequence_counter"]).lazy()

        # 線形加速度（重力除去）- jiazhuang方式
        linear_acc_df = remove_gravity_from_acc_pl(df_sorted)

        # 角速度（ベクトル化、シーケンス境界尊重）
        angular_vel_df = calculate_angular_velocity_from_quat_pl(df_sorted)

        # 角距離（ベクトル化、シーケンス境界尊重）
        angular_dist_df = calculate_angular_distance_pl(df_sorted)

        # 全ての物理特徴量を一度に計算して結合 (jiazhuang notebook compatible)
        df_with_physics = (
            pl.concat([df_sorted, linear_acc_df, angular_vel_df, angular_dist_df], how="horizontal")
            .with_columns(
                [
                    # Basic engineered features (jiazhuang notebook)
                    # 1. acc_mag - acceleration magnitude
                    (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2).sqrt().alias("acc_mag"),
                    # 2. rot_angle - rotation angle from quaternion
                    (2 * pl.col("rot_w").clip(-1, 1).arccos()).alias("rot_angle"),
                    # 3. acc_mag_jerk - jerk of acceleration magnitude
                    (
                        (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2)
                        .sqrt()
                        .diff()
                        .over("sequence_id")
                        .fill_null(0.0)
                    ).alias("acc_mag_jerk"),
                    # 4. rot_angle_vel - angular velocity from rotation angle
                    ((2 * pl.col("rot_w").clip(-1, 1).arccos()).diff().over("sequence_id").fill_null(0.0)).alias(
                        "rot_angle_vel"
                    ),
                    # Linear acceleration features
                    # 5. linear_acc_mag - magnitude of linear acceleration
                    (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                    .sqrt()
                    .alias("linear_acc_mag"),
                    # 6. linear_acc_mag_jerk - jerk of linear acceleration magnitude
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

        print(f"Added physics features. DataFrame shape: {df_with_physics.shape}")

        # Verify 19 IMU features are available
        available_imu_features = [col for col in self.imu_cols if col in df_with_physics.columns]
        print(f"Available IMU features ({len(available_imu_features)}/19): {available_imu_features}")

        if len(available_imu_features) != 19:
            missing_features = [col for col in self.imu_cols if col not in df_with_physics.columns]
            print(f"Warning: Missing IMU features: {missing_features}")

        # Human Normalization特徴量の計算（exp028では無効）
        if self.use_demographics and self.demographics_config.hn_enabled:
            print("Computing Human Normalization features...")
            hn_config = HNConfig(
                hn_enabled=self.demographics_config.hn_enabled,
                hn_eps=self.demographics_config.hn_eps,
                hn_radius_min_max=self.demographics_config.hn_radius_min_max,
                hn_features=self.demographics_config.hn_features,
            )
            df_with_hn = compute_hn_features(df_with_physics.lazy(), self.demographics_data, hn_config).collect()
            print(f"Added Human Normalization features. DataFrame shape: {df_with_hn.shape}")
            return df_with_hn

        return df_with_physics

    @timer_decorator("IMUDataset._preprocess_data_vectorized_with_mask")
    def _preprocess_data_vectorized_with_mask(self) -> dict:
        """Polarsを使ったベクトル化前処理（欠損値マスク付き）."""
        print("Starting vectorized preprocessing with missing value mask...")

        # 物理ベース特徴量を計算
        print("Calculating physics-based IMU features...")
        df_with_physics = self._add_physics_features(self.df)

        # 実際に使用するIMU列を取得（HN特徴量を含む）
        # DataFrameに存在し、かつself.imu_colsに含まれるカラムのみを使用
        available_cols = [col for col in self.imu_cols if col in df_with_physics.columns]

        print(f"Using {len(available_cols)} IMU columns (including HN features if enabled)")

        # シーケンスごとの前処理済みデータを一括取得（欠損値処理前）
        processed_data = (
            df_with_physics.sort(["sequence_id", "sequence_counter"])
            .group_by("sequence_id")
            .agg(
                [
                    # 利用可能なIMUデータをリストで集約（HN特徴量を含む）
                    *[pl.col(col) for col in available_cols],
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
                # IMUデータを追加（実際に存在するカラムのみ）
                available_cols = [col for col in self.imu_cols if col in batch_dict.keys()]
                for col in available_cols:
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
                    "nine_class_label": self.gesture_to_nine_class_id[gesture],
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

    def _apply_augmentation(self, imu_data: np.ndarray, subject_id: str = None) -> tuple[np.ndarray, bool]:
        """データ拡張の適用（tsai transforms + handedness flip）."""
        was_handedness_flipped = False

        if not self.augment:
            return imu_data, was_handedness_flipped

        # 利き手反転オーグメンテーション（最初に適用）
        if self.handedness_aug is not None:
            if self.handedness_aug.should_flip():
                imu_data = self.handedness_aug.flip_imu_data(imu_data, self.imu_cols)
                was_handedness_flipped = True

        # tsai transforms（RandAugment無効の場合は従来の確率的適用）
        if self.augmentation_config:
            if hasattr(self, "_time_series_aug") and self._time_series_aug is not None:
                # tsai transformsを使用
                if random.random() < self.augmentation_config.randaugment_prob:
                    imu_data = self._time_series_aug(imu_data)
            else:
                # 従来のオーグメンテーション（後方互換性）
                self._apply_legacy_augmentation(imu_data)

        return imu_data, was_handedness_flipped

    def _apply_legacy_augmentation(self, imu_data: np.ndarray) -> np.ndarray:
        """従来のオーグメンテーション（後方互換性用）."""
        # ガウシアンノイズ
        if np.random.random() < self.augmentation_config.gaussian_noise_prob:
            noise_std = self.augmentation_config.gaussian_noise_std
            noise = np.random.normal(0, noise_std, imu_data.shape)
            imu_data = imu_data + noise

        # 時間スケーリング
        if np.random.random() < self.augmentation_config.time_scaling_prob:
            scale_range = self.augmentation_config.time_scaling_range
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
        if np.random.random() < self.augmentation_config.partial_masking_prob:
            mask_length_range = self.augmentation_config.partial_masking_length_range
            mask_ratio = self.augmentation_config.partial_masking_ratio

            num_masks = int(len(imu_data) * mask_ratio / np.mean(mask_length_range))

            for _ in range(num_masks):
                mask_length = np.random.randint(mask_length_range[0], mask_length_range[1] + 1)
                start_idx = np.random.randint(0, max(1, len(imu_data) - mask_length))
                end_idx = min(start_idx + mask_length, len(imu_data))

                # マスク（ゼロ埋め）
                imu_data[start_idx:end_idx] = 0

        return imu_data

    def setup_tsai_augmentation(self):
        """tsai transformsのセットアップ."""
        if self.augmentation_config and self.augmentation_config.enable_randaugment:
            self._time_series_aug = TimeSeriesAugmentation(self.augmentation_config, self.imu_cols)
            # Crop sizeを設定
            if hasattr(self, "target_sequence_length"):
                crop_size = int(self.target_sequence_length * self.augmentation_config.crop_ratio)
                self._time_series_aug.set_crop_size(crop_size)
            print(
                f"tsai transforms enabled with RandAugment "
                f"(N={self.augmentation_config.randaugment_n}, M={self.augmentation_config.randaugment_m})"
            )
        else:
            self._time_series_aug = None
            print("tsai transforms disabled, using legacy augmentation")

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
            "nine_class_label": torch.tensor(data["nine_class_label"], dtype=torch.long),
            "sequence_id": seq_id,
            "gesture": data["gesture"],
        }

        # Demographics特徴量を追加（反転フラグを渡す）
        if self.use_demographics:
            subject = self.sequence_to_subject.get(seq_id)
            demographics = self._get_demographics_for_subject(subject, was_handedness_flipped)
            if demographics is not None:
                result["demographics"] = demographics

        # 統計量特徴量を追加
        if hasattr(self, "sequence_to_stats") and seq_id in self.sequence_to_stats:
            result["statistical_features"] = torch.tensor(self.sequence_to_stats[seq_id], dtype=torch.float32)

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
        self.augmentation_config = config.augmentation

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
                demographics_config=self.config.demographics,
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
                demographics_config=self.config.demographics,
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

        # IMU列の定義（jiazhuang notebook compatible: 20 physical features）
        self.imu_cols = [
            # Original IMU features (7)
            "acc_x",
            "acc_y",
            "acc_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            # Basic engineered features (4)
            "acc_mag",
            "rot_angle",
            "acc_mag_jerk",
            "rot_angle_vel",
            # Linear acceleration features (5)
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "linear_acc_mag",
            "linear_acc_mag_jerk",
            # Angular velocity features (3)
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            # Angular distance (1)
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

        # 全ての物理特徴量を結合（jiazhuang notebook compatible: 20 features）
        df_with_physics = (
            pl.concat([df_lazy, linear_acc_df, angular_vel_df, angular_dist_df], how="horizontal")
            .with_columns(
                [
                    # Basic engineered features (jiazhuang notebook)
                    # 1. acc_mag - acceleration magnitude
                    (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2).sqrt().alias("acc_mag"),
                    # 2. rot_angle - rotation angle from quaternion
                    (2 * pl.col("rot_w").clip(-1, 1).arccos()).alias("rot_angle"),
                    # 3. acc_mag_jerk - jerk of acceleration magnitude
                    (
                        (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2)
                        .sqrt()
                        .diff()
                        .fill_null(0.0)
                    ).alias("acc_mag_jerk"),
                    # 4. rot_angle_vel - angular velocity from rotation angle
                    ((2 * pl.col("rot_w").clip(-1, 1).arccos()).diff().fill_null(0.0)).alias("rot_angle_vel"),
                    # Linear acceleration features
                    # 5. linear_acc_mag - magnitude of linear acceleration
                    (pl.col("linear_acc_x") ** 2 + pl.col("linear_acc_y") ** 2 + pl.col("linear_acc_z") ** 2)
                    .sqrt()
                    .alias("linear_acc_mag"),
                    # 6. linear_acc_mag_jerk - jerk of linear acceleration magnitude
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
            "age": (self.demographics_config.age_min, self.demographics_config.age_max),
            "height_cm": (
                self.demographics_config.height_min,
                self.demographics_config.height_max,
            ),
            "shoulder_to_wrist_cm": (
                self.demographics_config.shoulder_to_wrist_min,
                self.demographics_config.shoulder_to_wrist_max,
            ),
            "elbow_to_wrist_cm": (
                self.demographics_config.elbow_to_wrist_min,
                self.demographics_config.elbow_to_wrist_max,
            ),
        }

    def _get_demographics_for_subject(self) -> dict[str, torch.Tensor] | None:
        """被験者のdemographics特徴量を取得してテンソル化."""
        if not self.use_demographics or self.subject is None or self.subject not in self.subject_to_demographics:
            return None

        demographics_raw = self.subject_to_demographics[self.subject]
        demographics_tensors = {}

        # カテゴリカル特徴量の処理
        categorical_features = self.demographics_config.categorical_features
        for feature in categorical_features:
            if feature in demographics_raw:
                value = demographics_raw[feature]
                # NaN値やNone値の処理
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0  # デフォルト値
                demographics_tensors[feature] = torch.tensor(int(value), dtype=torch.long)

        # 数値特徴量の処理（スケーリング済み）
        numerical_features = self.demographics_config.numerical_features
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


# Jiazhuang notebook data augmentation classes
class RandAugmentTimeSeries:
    """RandAugment for time series data using tsai transforms + custom transforms."""

    def __init__(self, config, imu_cols: list[str]):
        """
        Initialize RandAugment for time series.

        Args:
            config: AugmentationConfig containing RandAugment parameters
            imu_cols: IMU column names for custom transforms
        """

        self.config = config
        self.imu_cols = imu_cols
        self.N = config.randaugment_n  # Number of transforms to apply
        self.M = config.randaugment_m  # Magnitude of transforms

        # tsai transforms
        self.tsai_transforms = [
            ("AddNoise", TSMagAddNoise(magnitude=config.add_noise_scale)),
            (
                "MagScale",
                TSMagScale(magnitude=max(abs(config.mag_scale_range[0] - 1.0), abs(config.mag_scale_range[1] - 1.0))),
            ),
            ("TimeWarp", TSTimeWarp(magnitude=config.time_warp_max)),
            ("Mask", TSMaskOut(magnitude=config.mask_ratio)),
        ]

        # Custom transforms
        self.custom_transforms = [
            ("IMURotation", IMURotationTransform(angle_range=config.imu_rotation_angle_range, imu_cols=imu_cols)),
        ]

        # Combine all transforms
        self.all_transforms = self.tsai_transforms + self.custom_transforms

        # Add crop transform if size is set
        self.crop_size = None

    def set_crop_size(self, size):
        """Set crop size for crop transform."""
        from tsai.data.transforms import TSResize

        self.crop_size = size
        crop_transform = ("Crop", TSResize(size))

        # Update transforms list
        self.all_transforms = self.tsai_transforms + [crop_transform] + self.custom_transforms

    def __call__(self, sequence):
        """Apply RandAugment to sequence."""

        # Convert to torch tensor if numpy
        if isinstance(sequence, np.ndarray):
            x = torch.from_numpy(sequence).float()
            was_numpy = True
        else:
            x = sequence
            was_numpy = False

        # Apply RandAugment: randomly select N transforms
        if len(self.all_transforms) > 0:
            selected_transforms = random.sample(self.all_transforms, min(self.N, len(self.all_transforms)))

            for name, transform in selected_transforms:
                # Apply magnitude scaling based on RandAugment paper
                magnitude = random.uniform(0, self.M) / 10.0  # Normalize to [0, M/10]

                try:
                    if name == "AddNoise":
                        # Scale noise by magnitude
                        original_scale = transform.scale
                        transform.scale = original_scale * (1 + magnitude)
                        x = transform(x)
                        transform.scale = original_scale  # Reset
                    elif name == "MagScale":
                        # Scale magnitude by RandAugment magnitude
                        original_gain = transform.max_gain
                        transform.max_gain = original_gain * (1 + magnitude)
                        x = transform(x)
                        transform.max_gain = original_gain  # Reset
                    elif name == "TimeWarp":
                        # Scale warp by magnitude
                        original_warp = transform.max_warp
                        transform.max_warp = original_warp * (1 + magnitude)
                        x = transform(x)
                        transform.max_warp = original_warp  # Reset
                    elif name == "IMURotation":
                        # Scale rotation angle by magnitude
                        original_angle = transform.angle_range
                        transform.angle_range = original_angle * (1 + magnitude)
                        x = transform(x)
                        transform.angle_range = original_angle  # Reset
                    else:
                        # Apply transform as is for others
                        x = transform(x)
                except Exception:
                    # Skip transform if it fails
                    continue

        # Convert back to numpy if original was numpy
        if was_numpy:
            if isinstance(x, torch.Tensor):
                return x.numpy()
            return x
        return x


class TimeSeriesAugmentation:
    """Time series data augmentation using RandAugment framework."""

    def __init__(self, config, imu_cols: list[str] = None):
        """
        Initialize time series augmentation.

        Args:
            config: AugmentationConfig containing augmentation parameters
            imu_cols: IMU column names for custom transforms
        """
        self.config = config
        self.imu_cols = imu_cols or []

        if config.enable_randaugment:
            self.augmenter = RandAugmentTimeSeries(config, imu_cols)
        else:
            # Fallback to simple tsai transforms

            self.basic_transforms = [
                TSMagAddNoise(magnitude=config.add_noise_scale),
                TSMagScale(magnitude=max(abs(config.mag_scale_range[0] - 1.0), abs(config.mag_scale_range[1] - 1.0))),
                TSTimeWarp(magnitude=config.time_warp_max),
                TSMaskOut(magnitude=config.mask_ratio),
            ]
            self.augmenter = None

    def __call__(self, sequence):
        """Apply augmentation to sequence."""
        if self.augmenter is not None:
            return self.augmenter(sequence)
        else:
            # Fallback: randomly apply basic transforms
            import torch

            if isinstance(sequence, np.ndarray):
                x = torch.from_numpy(sequence).float()
                was_numpy = True
            else:
                x = sequence
                was_numpy = False

            # Apply basic transforms with probability
            if len(self.basic_transforms) > 0:
                num_transforms = min(len(self.basic_transforms), random.randint(1, 2))
                selected_transforms = random.sample(self.basic_transforms, num_transforms)

                for transform in selected_transforms:
                    if random.random() < 0.5:
                        try:
                            x = transform(x)
                        except Exception:
                            continue

            if was_numpy:
                if isinstance(x, torch.Tensor):
                    return x.numpy()
                return x
            return x

    def set_crop_size(self, size):
        """Set crop size for crop transforms."""
        if self.augmenter is not None:
            self.augmenter.set_crop_size(size)


class MixupAugmentation:
    """Mixup data augmentation - mix two different sequences."""

    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Application probability for Mixup
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch_sequences, batch_labels):
        """
        Apply Mixup to batch data.

        Args:
            batch_sequences: (batch_size, seq_len, num_features)
            batch_labels: (batch_size,)

        Returns:
            mixed_sequences: Mixed sequences
            labels_a: Original labels
            labels_b: Mixed labels
            lambda_: Mixing ratio
        """
        if random.random() > self.prob:
            return batch_sequences, batch_labels, batch_labels, 1.0

        batch_size = batch_sequences.size(0)
        lambda_ = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # Random shuffle order
        index = torch.randperm(batch_size)

        # Mix sequences
        mixed_sequences = lambda_ * batch_sequences + (1 - lambda_) * batch_sequences[index]

        labels_a = batch_labels
        labels_b = batch_labels[index]

        return mixed_sequences, labels_a, labels_b, lambda_


def mixup_criterion(criterion, outputs, labels_a, labels_b, lambda_):
    """Mixup loss function."""
    return lambda_ * criterion(outputs, labels_a) + (1 - lambda_) * criterion(outputs, labels_b)


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
