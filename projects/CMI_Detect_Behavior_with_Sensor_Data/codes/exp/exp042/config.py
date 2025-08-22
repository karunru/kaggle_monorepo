"""pydantic-settings based configuration for exp018（BERT統合版）."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# 実験番号の一元化
EXP_NUM = "exp042"


class ExperimentConfig(BaseModel):
    """実験メタデータ設定."""

    exp_num: str = EXP_NUM
    name: str = Field(default=f"{EXP_NUM}_configurable_schedule_free", description="実験名")
    description: str = Field(
        default="ConvNeXt1D with configurable parameters and Schedule-Free optimizer",
        description="実験説明",
    )
    tags: list[str] = Field(
        default=[
            "imu_only",
            "lstm",
            "bigru",
            "attention",
            "residual_se_convnext",
            "convnext",
            "human_normalization",
            "demographics",
            "soft_f1_acls",
            "bert",
            "tsai",
            "randaugment",
            "imu_rotation",
            "handedness_flip",
            "orient_head",
            "auxiliary_task",
            "score_fusion",
            "orientation_prior",
            "configurable_model",
            "schedule_free",
        ],
        description="実験タグ",
    )


class PathsConfig(BaseModel):
    """パス設定."""

    output_dir: str = Field(default=f"../../../outputs/{EXP_NUM}", description="出力ディレクトリ")


class DataConfig(BaseModel):
    """データ設定."""

    root: str = Field(default="../../../data", description="データルートディレクトリ")
    train_path: str = Field(default="../../../data/train.csv", description="訓練データパス")
    test_path: str = Field(default="../../../data/test.csv", description="テストデータパス")
    demographics_train_path: str = Field(
        default="../../../data/train_demographics.csv", description="訓練人口統計データパス"
    )
    demographics_test_path: str = Field(
        default="../../../data/test_demographics.csv", description="テスト人口統計データパス"
    )


class ModelConfig(BaseModel):
    """モデル設定."""

    name: str = Field(default="cmi_squeezeformer_bert", description="モデル名")
    input_dim: int = Field(
        default=20, description="入力次元数（基本IMU 7 + 物理特徴量 13 = 20、jiazhuang notebook compatible）"
    )
    d_model: int = Field(default=256, description="モデル次元")
    n_layers: int = Field(default=8, description="レイヤー数")
    n_heads: int = Field(default=8, description="アテンションヘッド数")
    d_ff: int = Field(default=1024, description="フィードフォワード次元")
    num_classes: int = Field(default=18, description="クラス数")
    kernel_size: int = Field(default=31, description="Convolutionカーネルサイズ")
    dropout: float = Field(default=0.1, description="ドロップアウト率")
    
    # IMUOnlyLSTM specific configuration
    imu_block1_out_channels: int = Field(default=64, description="IMU Block1出力チャネル数")
    imu_block1_kernel_size: int = Field(default=3, description="IMU Block1カーネルサイズ")
    imu_block1_num_blocks: int = Field(default=2, description="IMU Block1ブロック数")
    imu_block1_drop_path: float = Field(default=0.1, description="IMU Block1ドロップパス率")
    imu_block2_out_channels: int = Field(default=128, description="IMU Block2出力チャネル数")
    imu_block2_kernel_size: int = Field(default=5, description="IMU Block2カーネルサイズ")
    imu_block2_num_blocks: int = Field(default=2, description="IMU Block2ブロック数")
    imu_block2_drop_path: float = Field(default=0.15, description="IMU Block2ドロップパス率")
    bigru_hidden_size: int = Field(default=128, description="BiGRU隠れ層サイズ")
    gru_dropout: float = Field(default=0.4, description="GRUドロップアウト率")
    dense1_dropout: float = Field(default=0.5, description="Dense1ドロップアウト率")
    dense2_dropout: float = Field(default=0.3, description="Dense2ドロップアウト率")


class SchedulerConfig(BaseModel):
    """スケジューラ設定."""

    type: Literal["cosine", "plateau"] | None = Field(default="cosine", description="スケジューラタイプ")
    min_lr: float = Field(default=1e-6, description="最小学習率")
    factor: float = Field(default=0.5, description="学習率減衰係数")
    patience: int = Field(default=5, description="プラトー待機エポック数")


class TrainingConfig(BaseModel):
    """訓練設定"""

    # 基本設定
    seed: int = Field(default=42, description="ランダムシード")
    batch_size: int = Field(default=128, description="バッチサイズ")
    num_workers: int = Field(default=8, description="データローダーワーカー数")
    epochs: int = Field(default=200, description="エポック数")

    # 最適化設定（NaN防止のため学習率を下げて安定化）
    learning_rate: float = Field(default=3e-4, description="学習率")
    weight_decay: float = Field(default=1e-5, description="重み減衰")
    gradient_clip_val: float = Field(default=0.5, description="勾配クリッピング値（NaN防止のため強化）")

    # スケジューラ設定
    scheduler_type: Literal["cosine", "plateau"] | None = Field(default="cosine", description="スケジューラタイプ")
    scheduler_min_lr: float = Field(default=1e-6, description="最小学習率")
    scheduler_factor: float = Field(default=0.5, description="学習率減衰係数")
    scheduler_patience: int = Field(default=5, description="プラトー待機エポック数")

    # ハードウェア設定
    use_mixed_precision: bool = Field(default=True, description="混合精度使用フラグ")
    accumulate_grad_batches: int = Field(default=1, description="勾配累積バッチ数")


class DemographicsConfig(BaseModel):
    """Demographics統合設定."""

    # 基本設定
    enabled: bool = Field(default=True, description="Demographics統合を有効にするかどうか")

    # 埋め込み設定
    embedding_dim: int = Field(default=16, description="Demographics埋め込み次元")
    categorical_embedding_dims: dict[str, int] = Field(
        default={"adult_child": 2, "sex": 2, "handedness": 2}, description="カテゴリカル特徴量の埋め込み次元"
    )

    # スケーリング設定
    numerical_scaling_method: Literal["minmax", "standard"] = Field(
        default="minmax", description="数値特徴量のスケーリング手法"
    )
    clip_outliers: bool = Field(default=True, description="範囲外値をクリップするかどうか")

    # スケーリングパラメータ（実データ分析に基づく固定値）
    age_min: float = Field(default=8.0, description="年齢の最小値（実データ: 10-53 + マージン）")
    age_max: float = Field(default=60.0, description="年齢の最大値（実データ: 10-53 + マージン）")
    height_min: float = Field(default=130.0, description="身長の最小値（cm）（実データ: 135-190.5 + マージン）")
    height_max: float = Field(default=195.0, description="身長の最大値（cm）（実データ: 135-190.5 + マージン）")
    shoulder_to_wrist_min: float = Field(
        default=35.0, description="肩-手首距離の最小値（cm）（実データ: 41-71 + マージン）"
    )
    shoulder_to_wrist_max: float = Field(
        default=75.0, description="肩-手首距離の最大値（cm）（実データ: 41-71 + マージン）"
    )
    elbow_to_wrist_min: float = Field(
        default=15.0, description="肘-手首距離の最小値（cm）（実データ: 18-44 + マージン）"
    )
    elbow_to_wrist_max: float = Field(
        default=50.0, description="肘-手首距離の最大値（cm）（実データ: 18-44 + マージン）"
    )

    # 特徴量リスト
    categorical_features: list[str] = Field(
        default=["adult_child", "sex", "handedness"], description="カテゴリカル特徴量リスト"
    )
    numerical_features: list[str] = Field(
        default=["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"], description="数値特徴量リスト"
    )

    # Human Normalization設定
    hn_enabled: bool = Field(default=False, description="Human Normalization機能を有効にするかどうか")
    hn_eps: float = Field(default=1e-3, description="数値安定性のためのepsilon値（大きくしてNaN防止）")
    hn_radius_min_max: tuple[float, float] = Field(default=(0.15, 0.9), description="有効半径の最小値・最大値")
    hn_features: list[str] = Field(
        default=[
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
        ],
        description="有効にするHuman Normalization特徴量のリスト",
    )


class ACLSConfig(BaseModel):
    """ACLS損失関数設定."""

    # Label Smoothing
    label_smoothing_alpha: float = Field(default=0.1, description="Label smoothing parameter")

    # Margin-based Label Smoothing
    mbls_margin: float = Field(default=10.0, description="MbLS margin parameter")
    mbls_alpha: float = Field(default=0.1, description="MbLS weight parameter")
    mbls_schedule: str | None = Field(
        default=None, description="MbLS alpha scheduling ('add', 'multiply', 'step', None)"
    )

    # ACLS
    acls_pos_lambda: float = Field(default=1.0, description="ACLS positive sample regularization weight")
    acls_neg_lambda: float = Field(default=0.1, description="ACLS negative sample regularization weight")
    acls_alpha: float = Field(default=0.1, description="ACLS regularization term weight")
    acls_margin: float = Field(default=10.0, description="ACLS margin for distance calculation")


class LossConfig(BaseModel):
    """損失関数設定."""

    type: Literal["cmi", "cmi_focal", "soft_f1", "soft_f1_acls", "acls", "label_smoothing", "mbls"] = Field(
        default="acls", description="損失関数タイプ"
    )
    alpha: float = Field(default=0.5, description="バイナリ vs マルチクラス損失の重み")
    focal_gamma: float = Field(default=2.0, description="Focal Loss gamma パラメータ")
    focal_alpha: float = Field(default=1.0, description="Focal Loss alpha パラメータ")
    soft_f1_beta: float = Field(default=1.0, description="SoftF1Loss beta パラメータ (F-beta score)")
    soft_f1_eps: float = Field(default=1e-6, description="SoftF1Loss epsilon パラメータ (数値安定性)")
    label_smoothing: float = Field(default=0.1, description="Label smoothing パラメータ（0.0で無効, 0.0-1.0の範囲）")

    # 9クラスヘッド設定
    nine_class_head_enabled: bool = Field(
        default=True, description="9クラスヘッド（8 target + 1 non-target）を有効にするかどうか"
    )
    nine_class_loss_weight: float = Field(default=0.2, description="9クラスヘッドの損失重み")

    # KL divergence loss設定
    kl_weight: float = Field(default=0.1, description="KL divergence損失の重み")
    kl_temperature: float = Field(default=1.0, description="KL divergence計算時の温度パラメータ")

    # 学習可能重み付け設定
    auto_weighting: Literal["none", "uncertainty", "direct"] = Field(default="none", description="自動重み付け方式")
    auto_weight_clamp: tuple[float, float] = Field(
        default=(-10.0, 10.0), description="学習可能パラメータのクランプ範囲"
    )
    uncertainty_init_value: float = Field(default=0.0, description="不確かさパラメータの初期値")

    use_mixup: bool = Field(default=True, description="mixupを有効にするかどうか")

    # Orient head設定
    orient_loss_weight: float = Field(default=0.3, description="Orient頭部の損失重み")


class LengthGroupingConfig(BaseModel):
    """系列長グループ化設定"""

    enabled: bool = Field(default=False, description="長さグループ化を有効にするかどうか")
    use_dynamic_padding: bool = Field(default=False, description="動的パディングを使用するかどうか")
    mega_batch_multiplier: int = Field(default=8, description="メガバッチサイズの倍率")
    percentile_max_length: float = Field(default=0.95, description="パディング最大長のパーセンタイル")
    min_sequence_length: int = Field(default=50, description="最小系列長")
    max_sequence_length: int = Field(default=500, description="最大系列長")
    put_longest_first: bool = Field(default=True, description="最長バッチを先頭に配置するかどうか")
    fallback_to_fixed_length: bool = Field(default=True, description="エラー時に固定長にフォールバックするかどうか")


class ScheduleFreeConfig(BaseModel):
    """Schedule Free Optimizer設定"""

    enabled: bool = Field(default=True, description="Schedule Free使用フラグ")
    optimizer_type: Literal["RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree"] = Field(
        default="RAdamScheduleFree", description="オプティマイザータイプ"
    )
    learning_rate_multiplier: float = Field(default=5.0, description="学習率倍率")
    warmup_steps: int = Field(default=1000, description="ウォームアップステップ数")
    batch_norm_calibration_steps: int = Field(default=50, description="BatchNorm較正ステップ数")


class EMAConfig(BaseModel):
    """EMA（Exponential Moving Average）設定."""

    enabled: bool = Field(default=True, description="EMA使用フラグ")
    beta: float = Field(default=0.9999, description="EMA減衰係数")
    update_after_step: int = Field(default=1000, description="EMA更新開始ステップ")
    update_every: int = Field(default=10, description="EMA更新頻度")
    update_model_with_ema_every: int = Field(default=1000, description="Switch EMA更新頻度")
    use_ema_for_validation: bool = Field(default=True, description="検証時にEMAモデルを使用")


class ValidationConfigParams(BaseModel):
    """クロスバリデーション詳細パラメータ."""

    n_splits: int = Field(default=5, description="クロスバリデーション分割数")
    random_state: int = Field(default=42, description="ランダムステート")
    target: str = Field(default="gesture", description="ターゲット列名")
    group: str = Field(default="subject", description="グループ列名")
    id: str = Field(default="sequence_id", description="ID列名")
    force_recreate: bool = Field(default=False, description="強制再作成フラグ")


class ValidationConfig(BaseModel):
    """クロスバリデーション設定."""

    name: str = Field(default="stratified_group_kfold", description="バリデーション手法名")
    params: ValidationConfigParams = Field(
        default_factory=ValidationConfigParams, description="バリデーション詳細パラメータ"
    )


class PreprocessingConfig(BaseModel):
    """前処理設定."""

    target_sequence_length: int = Field(default=200, description="目標シーケンス長")


class AugmentationConfig(BaseModel):
    """データ拡張設定（tsai + RandAugment）."""

    # RandAugment設定
    enable_randaugment: bool = Field(default=True, description="RandAugmentを有効にする")
    randaugment_n: int = Field(default=4, description="RandAugment: 適用する変換数")
    randaugment_m: float = Field(default=4.0, description="RandAugment: 変換強度")
    randaugment_prob: float = Field(default=0.9, description="RandAugment適用確率")

    # tsai transforms設定
    add_noise_scale: float = Field(default=0.02, description="AddNoise: ノイズ標準偏差")
    mag_scale_range: list[float] = Field(default=[0.9, 1.1], description="MagScale: スケーリング範囲")
    time_warp_max: float = Field(default=0.2, description="TimeWarp: 最大ワープ率")
    crop_ratio: float = Field(default=0.9, description="Crop: 切り取り比率")
    mask_ratio: float = Field(default=0.1, description="Mask: マスク比率")

    # カスタム変換設定
    imu_rotation_angle_range: float = Field(default=0.2, description="IMU回転: 角度範囲（ラジアン）")
    handedness_flip_prob: float = Field(default=0.0, description="利き手反転: 適用確率")

    # 従来設定（後方互換性）
    enable_handedness_flip: bool = Field(default=False, description="利き手反転オーグメンテーションを有効にする")
    gaussian_noise_prob: float = Field(default=0.0, description="ガウシアンノイズ適用確率（非推奨）")
    gaussian_noise_std: float = Field(default=0.01, description="ガウシアンノイズ標準偏差（非推奨）")
    time_scaling_prob: float = Field(default=0.0, description="時間スケーリング適用確率（非推奨）")
    time_scaling_range: list[float] = Field(default=[0.9, 1.1], description="時間スケーリング範囲（非推奨）")
    partial_masking_prob: float = Field(default=0.0, description="部分マスキング適用確率（非推奨）")
    partial_masking_length_range: list[int] = Field(default=[5, 20], description="マスク長範囲（非推奨）")
    partial_masking_ratio: float = Field(default=0.1, description="マスク比率（非推奨）")


class LoggingConfig(BaseModel):
    """WandBログ設定."""

    wandb_enabled: bool = Field(default=True, description="WandB有効フラグ")
    wandb_project: str = Field(default="CMI-Detect-Behavior-with-Sensor-Data", description="WandBプロジェクト名")
    wandb_name: str = Field(default=f"{EXP_NUM}", description="WandB実行名")
    wandb_tags: list[str] = Field(
        default=[
            EXP_NUM,
            "imu_only",
            "lstm",
            "bigru",
            "attention",
            "residual_se_convnext",
            "convnext",
            "human_normalization",
            "demographics",
            "soft_f1_acls",
            "bert",
            "tsai",
            "randaugment",
            "imu_rotation",
            "handedness_flip",
            "orient_head",
            "auxiliary_task",
            "score_fusion",
            "orientation_prior",
            "configurable_model",
            "schedule_free",
        ],
        description="WandBタグ",
    )


class TrainerConfig(BaseModel):
    """PyTorch Lightning Trainer設定."""

    accelerator: str = Field(default="auto", description="使用デバイス")
    devices: str = Field(default="auto", description="デバイス数")
    precision: Literal["16-mixed", "32-true", "bf16-mixed"] = Field(default="16-mixed", description="精度設定")
    deterministic: bool = Field(default=False, description="決定的実行")
    benchmark: bool = Field(default=True, description="CUDA最適化")
    enable_checkpointing: bool = Field(default=True, description="チェックポイント有効化")
    log_every_n_steps: int = Field(default=10, description="ログ出力間隔")
    check_val_every_n_epoch: int = Field(default=1, description="検証実行間隔")
    val_check_interval: float = Field(default=1.0, description="検証実行タイミング")


class ModelCheckpointConfig(BaseModel):
    """ModelCheckpoint設定."""

    monitor: str = Field(default="val_cmi_score", description="監視メトリクス")
    mode: str = Field(default="max", description="最適化方向")
    save_top_k: int = Field(default=1, description="保存する上位モデル数")
    save_last: bool = Field(default=True, description="最新モデルの保存")
    filename: str = Field(default="epoch-{epoch:02d}-val_cmi_score-{val_cmi_score:.4f}", description="ファイル名形式")
    auto_insert_metric_name: bool = Field(default=False, description="メトリクス名自動挿入")


class EarlyStoppingConfig(BaseModel):
    """EarlyStopping設定."""

    monitor: str = Field(default="val_cmi_score", description="監視メトリクス")
    mode: str = Field(default="max", description="最適化方向")
    patience: int = Field(default=30, description="待機エポック数")
    min_delta: float = Field(default=0.0001, description="改善の最小閾値")
    verbose: bool = Field(default=True, description="詳細ログ出力")


class LRMonitorConfig(BaseModel):
    """LearningRateMonitor設定."""

    logging_interval: str = Field(default="epoch", description="ログ記録間隔")


class RichProgressBarConfig(BaseModel):
    """RichProgressBar設定."""

    enable: bool = Field(default=False, description="RichProgressBar有効化")


class CallbacksConfig(BaseModel):
    """コールバック設定."""

    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    lr_monitor: LRMonitorConfig = Field(default_factory=LRMonitorConfig)
    rich_progress_bar: RichProgressBarConfig = Field(default_factory=RichProgressBarConfig)


class LightningConfig(BaseModel):
    """PyTorch Lightning設定."""

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)


class Config(BaseSettings):
    """メイン設定クラス."""

    # 各セクションの設定
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig, description="実験設定")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="パス設定")
    data: DataConfig = Field(default_factory=DataConfig, description="データ設定")
    demographics: DemographicsConfig = Field(default_factory=DemographicsConfig, description="Demographics設定")
    model: ModelConfig = Field(default_factory=ModelConfig, description="モデル設定")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="訓練設定")
    loss: LossConfig = Field(default_factory=LossConfig, description="損失関数設定")
    acls: ACLSConfig = Field(default_factory=ACLSConfig, description="ACLS損失関数設定")
    length_grouping: LengthGroupingConfig = Field(
        default_factory=LengthGroupingConfig, description="系列長グループ化設定"
    )
    schedule_free: ScheduleFreeConfig = Field(default_factory=ScheduleFreeConfig, description="Schedule Free設定")
    ema: EMAConfig = Field(default_factory=EMAConfig, description="EMA設定")
    val: ValidationConfig = Field(default_factory=ValidationConfig, description="バリデーション設定")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig, description="前処理設定")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig, description="データ拡張設定")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="WandBログ設定")
    lightning: LightningConfig = Field(default_factory=LightningConfig, description="PyTorch Lightning設定")

    # get_validation関数が期待するrootフィールド
    root: str = Field(default=f"../../../outputs/{EXP_NUM}", description="出力ルートディレクトリ（get_validation用）")

    # ジェスチャーリスト
    target_gestures: list[str] = Field(
        default=[
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        ],
        description="ターゲットジェスチャー（BFRB様）",
    )

    non_target_gestures: list[str] = Field(
        default=[
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
        ],
        description="ノンターゲットジェスチャー",
    )

    imu_features: list[str] = Field(
        default=[
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
        ],
        description="IMU特徴量列名（基本IMU 7 + 物理ベース特徴量 9）",
    )

    class Config:
        """pydantic設定."""

        env_prefix = f"{EXP_NUM.upper()}_"  # 環境変数プレフィックス
        case_sensitive = False
        extra = "forbid"  # 未定義フィールドを禁止

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return self.model_dump()

    def validate_paths(self) -> bool:
        """パスの存在確認."""
        paths_to_check = [
            self.data.train_path,
            self.data.test_path,
            self.data.demographics_train_path,
            self.data.demographics_test_path,
        ]

        missing_paths = []
        for path_str in paths_to_check:
            path = Path(path_str)
            if not path.exists():
                missing_paths.append(path_str)

        if missing_paths:
            print(f"Warning: Missing paths: {missing_paths}")
            return False

        return True

    def create_output_dirs(self) -> None:
        """出力ディレクトリの作成."""
        output_dirs = [
            Path(self.paths.output_dir),
            Path(self.paths.output_dir) / "logs",
        ]

        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


if __name__ == "__main__":
    """設定のテスト実行."""
    # デフォルト設定でのテスト
    config = Config()
    print("✓ Default config created successfully")

    # 設定値の確認
    print(f"Model input_dim: {config.model.input_dim}")
    print(f"Training batch_size: {config.training.batch_size}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Validation n_splits: {config.val.params.n_splits}")
    print(f"WandB enabled: {config.logging.wandb_enabled}")
    print(f"Number of target gestures: {len(config.target_gestures)}")
    print(f"Number of IMU features: {len(config.imu_features)}")

    # Lightning設定の確認
    print(f"Lightning trainer max_epochs: {config.training.epochs}")
    print(f"Lightning callbacks checkpoint monitor: {config.lightning.callbacks.model_checkpoint.monitor}")

    # バリデーションテスト
    try:
        config.validate_paths()
        print("✓ Path validation completed")
    except Exception as e:
        print(f"! Path validation warning: {e}")

    print("Config test completed successfully!")
