"""Human Normalization機能の実装.

人体測定値（身長、肘から手首の距離、肩から手首の距離）を使用して
IMUデータを正規化し、体格に依存しない特徴量を生成する。
"""

from dataclasses import dataclass

import polars as pl


@dataclass
class HNConfig:
    """Human Normalization設定クラス."""

    hn_enabled: bool = True
    hn_eps: float = 1e-3  # より大きなepsilonで数値安定性向上
    hn_radius_min_max: tuple[float, float] = (0.15, 0.9)
    hn_features: list[str] = None

    def __post_init__(self):
        """デフォルト特徴量リストを設定."""
        if self.hn_features is None:
            self.hn_features = [
                "linear_acc_mag_per_h",
                "linear_acc_mag_per_rS",
                "linear_acc_mag_per_rE",
                "acc_over_centripetal_rS",
                "acc_over_centripetal_rE",
                "alpha_like_rS",
                "alpha_like_rE",
                "v_over_h",
                "v_over_rS",
                "v_over_rE",
            ]


def join_subject_anthro(frame: pl.LazyFrame, demo_df: pl.DataFrame) -> pl.LazyFrame:
    """subjectに人体測定値をjoin.

    Args:
        frame: IMUデータのLazyFrame（subjectカラムが必要）
        demo_df: Demographics DataFrame

    Returns:
        人体測定値がjoinされたLazyFrame
    """
    if demo_df is None or demo_df.is_empty():
        raise ValueError("Demographics data is required but not provided")

    # 人体測定値カラムを選択し、メートル単位に変換
    anthro_cols = ["subject", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
    missing_cols = [col for col in anthro_cols if col not in demo_df.columns]
    if missing_cols:
        raise ValueError(f"Missing demographics columns: {missing_cols}")

    # 人体測定値を前処理（欠損値処理とメートル変換）
    demo_processed = demo_df.select(
        [
            pl.col("subject"),
            # 身長（cm → m）
            (pl.col("height_cm") / 100.0).alias("h"),
            # 前腕レバー長（cm → m）
            (pl.col("elbow_to_wrist_cm") / 100.0).alias("r_elbow"),
            # 肩-手首レバー長（cm → m）
            (pl.col("shoulder_to_wrist_cm") / 100.0).alias("r_shoulder"),
        ]
    )

    # 欠損値を中央値で埋める
    medians = demo_processed.select(
        [
            pl.col("h").median().alias("h_median"),
            pl.col("r_elbow").median().alias("r_elbow_median"),
            pl.col("r_shoulder").median().alias("r_shoulder_median"),
        ]
    )

    h_median = medians["h_median"][0]
    r_elbow_median = medians["r_elbow_median"][0]
    r_shoulder_median = medians["r_shoulder_median"][0]

    # 欠損値フラグを作成
    demo_with_fallback = demo_processed.with_columns(
        [
            # 欠損値または非正値をチェック
            (
                pl.col("h").is_null()
                | pl.col("r_elbow").is_null()
                | pl.col("r_shoulder").is_null()
                | (pl.col("h") <= 0)
                | (pl.col("r_elbow") <= 0)
                | (pl.col("r_shoulder") <= 0)
            ).alias("hn_used_fallback"),
            # 欠損値を中央値で埋める
            pl.col("h").fill_null(h_median).clip(0.01, None).alias("h"),
            pl.col("r_elbow").fill_null(r_elbow_median).clip(0.01, None).alias("r_elbow"),
            pl.col("r_shoulder").fill_null(r_shoulder_median).clip(0.01, None).alias("r_shoulder"),
        ]
    )

    # LazyFrameでjoin
    return frame.join(demo_with_fallback.lazy(), on="subject", how="left")


def derive_hn_channels(frame: pl.LazyFrame, eps: float, bounds: tuple[float, float]) -> pl.LazyFrame:
    """正規化チャンネルを導出.

    Args:
        frame: 人体測定値がjoinされたLazyFrame
        eps: 数値安定性のためのepsilon
        bounds: r_shoulderのクリッピング範囲

    Returns:
        正規化特徴量が追加されたLazyFrame
    """
    min_radius, max_radius = bounds

    # 必要な物理特徴量の存在確認
    required_cols = [
        "linear_acc_x",
        "linear_acc_y",
        "linear_acc_z",
        "linear_acc_mag",
        "angular_vel_x",
        "angular_vel_y",
        "angular_vel_z",
        "h",
        "r_elbow",
        "r_shoulder",
    ]

    # 有効半径を計算
    frame_with_r_eff = frame.with_columns([pl.col("r_shoulder").clip(min_radius, max_radius).alias("r_eff")])

    # 角速度の大きさを計算
    frame_with_omega = frame_with_r_eff.with_columns(
        [
            (pl.col("angular_vel_x").pow(2) + pl.col("angular_vel_y").pow(2) + pl.col("angular_vel_z").pow(2))
            .sqrt()
            .alias("omega")
        ]
    )

    # 速度と遠心加速度を計算
    frame_with_kinematics = frame_with_omega.with_columns(
        [
            # 肘での接線速度
            (pl.col("omega") * pl.col("r_elbow")).alias("v_elbow"),
            # 肩での接線速度
            (pl.col("omega") * pl.col("r_shoulder")).alias("v_shoulder"),
            # 肘での遠心加速度
            (pl.col("omega").pow(2) * pl.col("r_elbow")).alias("a_c_elbow"),
            # 肩での遠心加速度
            (pl.col("omega").pow(2) * pl.col("r_shoulder")).alias("a_c_shoulder"),
        ]
    )

    # 正規化特徴量を計算（数値安定性と極値処理を強化）
    return frame_with_kinematics.with_columns(
        [
            # 身長正規化
            (pl.col("linear_acc_mag") / pl.col("h").clip(eps, None)).clip(0, 100).alias("linear_acc_mag_per_h"),
            # レバー長正規化
            (pl.col("linear_acc_mag") / pl.col("r_shoulder").clip(eps, None))
            .clip(0, 100)
            .alias("linear_acc_mag_per_rS"),
            (pl.col("linear_acc_mag") / pl.col("r_elbow").clip(eps, None)).clip(0, 100).alias("linear_acc_mag_per_rE"),
            # 無次元化比率（遠心加速度との比）- 極値クリッピングを強化
            (pl.col("linear_acc_mag") / pl.col("a_c_shoulder").clip(eps, None))
            .clip(0, 1000)
            .alias("acc_over_centripetal_rS"),
            (pl.col("linear_acc_mag") / pl.col("a_c_elbow").clip(eps, None))
            .clip(0, 1000)
            .alias("acc_over_centripetal_rE"),
            # 角加速度近似（角速度ベース） - 重複を修正
            (pl.col("linear_acc_mag") / (pl.col("omega") * pl.col("r_shoulder")).clip(eps, None))
            .clip(0, 1000)
            .alias("alpha_like_rS"),
            (pl.col("linear_acc_mag") / (pl.col("omega") * pl.col("r_elbow")).clip(eps, None))
            .clip(0, 1000)
            .alias("alpha_like_rE"),
            # 速度複合特徴量
            (pl.col("v_shoulder") / pl.col("h").clip(eps, None)).clip(0, 100).alias("v_over_h"),
            (pl.col("v_shoulder") / pl.col("r_shoulder").clip(eps, None)).clip(0, 100).alias("v_over_rS"),
            (pl.col("v_elbow") / pl.col("r_elbow").clip(eps, None)).clip(0, 100).alias("v_over_rE"),
        ]
    )


def compute_hn_features(frame: pl.LazyFrame, demo_df: pl.DataFrame, cfg: HNConfig) -> pl.LazyFrame:
    """Human Normalization特徴量を計算.

    Args:
        frame: IMUデータのLazyFrame
        demo_df: Demographics DataFrame
        cfg: HN設定

    Returns:
        HN特徴量が追加されたLazyFrame
    """
    if not cfg.hn_enabled:
        return frame

    # 人体測定値をjoin
    frame_with_anthro = join_subject_anthro(frame, demo_df)

    # 正規化特徴量を計算
    frame_with_hn = derive_hn_channels(frame_with_anthro, eps=cfg.hn_eps, bounds=cfg.hn_radius_min_max)

    return frame_with_hn


def get_hn_feature_columns(cfg: HNConfig) -> list[str]:
    """有効化されたHN特徴量のカラム名リストを取得.

    Args:
        cfg: HN設定

    Returns:
        有効化されたHN特徴量のカラム名リスト
    """
    if not cfg.hn_enabled:
        return []

    return cfg.hn_features.copy()
