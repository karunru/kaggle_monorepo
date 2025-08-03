"""pydantic-settings based configuration for exp004."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ExperimentConfig(BaseModel):
    """実験メタデータ設定."""

    name: str = Field(default="exp004_imu_only_squeezeformer", description="実験名")
    description: str = Field(default="IMU-only Squeezeformer model using PyTorch Lightning", description="実験説明")
    tags: list[str] = Field(
        default=["imu_only", "squeezeformer", "pytorch_lightning", "stratified_group_kfold"], description="実験タグ"
    )


class PathsConfig(BaseModel):
    """パス設定."""

    experiment_dir: str = Field(default=".", description="実験ディレクトリ")
    output_dir: str = Field(default="../../outputs/exp004", description="出力ディレクトリ")


class DataConfig(BaseModel):
    """データ設定."""

    root: str = Field(default="../../data", description="データルートディレクトリ")
    train_path: str = Field(default="../../data/train.csv", description="訓練データパス")
    test_path: str = Field(default="../../data/test.csv", description="テストデータパス")
    demographics_train_path: str = Field(
        default="../../data/train_demographics.csv", description="訓練人口統計データパス"
    )
    demographics_test_path: str = Field(
        default="../../data/test_demographics.csv", description="テスト人口統計データパス"
    )


class ModelConfig(BaseModel):
    """モデル設定."""

    name: str = Field(default="cmi_squeezeformer", description="モデル名")
    input_dim: int = Field(default=7, description="入力次元数（IMU: acc_x/y/z + rot_w/x/y/z）")
    d_model: int = Field(default=256, description="モデル次元")
    n_layers: int = Field(default=8, description="レイヤー数")
    n_heads: int = Field(default=8, description="アテンションヘッド数")
    d_ff: int = Field(default=1024, description="フィードフォワード次元")
    num_classes: int = Field(default=18, description="クラス数")
    kernel_size: int = Field(default=31, description="Convolutionカーネルサイズ")
    dropout: float = Field(default=0.1, description="ドロップアウト率")

    @validator("input_dim")
    def validate_input_dim(cls, v):
        if v <= 0:
            raise ValueError("input_dim must be positive")
        return v

    @validator("dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("dropout must be between 0 and 1")
        return v


class SchedulerConfig(BaseModel):
    """スケジューラ設定."""

    type: Literal["cosine", "plateau"] | None = Field(default="cosine", description="スケジューラタイプ")
    min_lr: float = Field(default=1e-6, description="最小学習率")
    factor: float = Field(default=0.5, description="学習率減衰係数")
    patience: int = Field(default=5, description="プラトー待機エポック数")


class TrainingConfig(BaseModel):
    """訓練設定."""

    seed: int = Field(default=42, description="ランダムシード")
    batch_size: int = Field(default=128, description="バッチサイズ")
    num_workers: int = Field(default=4, description="データローダーワーカー数")
    epochs: int = Field(default=100, description="エポック数")
    learning_rate: float = Field(default=3e-4, description="学習率")
    weight_decay: float = Field(default=1e-5, description="重み減衰")
    early_stopping_patience: int = Field(default=15, description="早期停止待機エポック数")
    n_folds: int = Field(default=5, description="クロスバリデーションフォールド数")
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig, description="スケジューラ設定")

    @validator("batch_size", "epochs", "n_folds")
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @validator("learning_rate", "weight_decay")
    def validate_positive_float(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class LossConfig(BaseModel):
    """損失関数設定."""

    type: Literal["cmi", "cmi_focal"] = Field(default="cmi", description="損失関数タイプ")
    alpha: float = Field(default=0.5, description="バイナリ vs マルチクラス損失の重み")
    focal_gamma: float = Field(default=2.0, description="Focal Loss gamma パラメータ")
    focal_alpha: float = Field(default=1.0, description="Focal Loss alpha パラメータ")

    @validator("alpha")
    def validate_alpha(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("alpha must be between 0 and 1")
        return v


class ScheduleFreeConfig(BaseModel):
    """Schedule Free Optimizer設定."""

    enabled: bool = Field(default=False, description="Schedule Free使用フラグ")
    optimizer_type: Literal["RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree"] = Field(
        default="RAdamScheduleFree", description="オプティマイザータイプ"
    )
    learning_rate_multiplier: float = Field(default=5.0, description="学習率倍率")
    warmup_steps: int = Field(default=1000, description="ウォームアップステップ数")
    batch_norm_calibration_steps: int = Field(default=50, description="BatchNorm較正ステップ数")

    @validator("learning_rate_multiplier")
    def validate_lr_multiplier(cls, v):
        if v <= 0:
            raise ValueError("learning_rate_multiplier must be positive")
        return v

    @validator("warmup_steps")
    def validate_warmup_steps(cls, v):
        if v < 0:
            raise ValueError("warmup_steps must be non-negative")
        return v

    @validator("batch_norm_calibration_steps")
    def validate_calibration_steps(cls, v):
        if v < 1:
            raise ValueError("batch_norm_calibration_steps must be at least 1")
        return v


class ValidationParamsConfig(BaseModel):
    """バリデーションパラメータ設定."""

    n_splits: int = Field(default=5, description="分割数")
    random_state: int = Field(default=42, description="ランダムステート")
    target: str = Field(default="gesture", description="ターゲット列名")
    group: str = Field(default="subject", description="グループ列名")
    id: str = Field(default="sequence_id", description="ID列名")
    force_recreate: bool = Field(default=False, description="強制再作成フラグ")


class ValidationConfig(BaseModel):
    """クロスバリデーション設定."""

    name: str = Field(default="stratified_group_kfold", description="バリデーション手法名")
    params: ValidationParamsConfig = Field(
        default_factory=ValidationParamsConfig, description="バリデーションパラメータ"
    )


class PreprocessingConfig(BaseModel):
    """前処理設定."""

    target_sequence_length: int = Field(default=200, description="目標シーケンス長")
    imu_normalization: Literal["zscore", "minmax"] = Field(default="zscore", description="IMU正規化手法")
    apply_lowpass_filter: bool = Field(default=False, description="ローパスフィルタ適用フラグ")
    lowpass_cutoff: int = Field(default=10, description="ローパスフィルタカットオフ周波数")
    lowpass_fs: int = Field(default=50, description="ローパスフィルタサンプリング周波数")
    lowpass_order: int = Field(default=4, description="ローパスフィルタ次数")


class GaussianNoiseConfig(BaseModel):
    """ガウシアンノイズ設定."""

    probability: float = Field(default=0.3, description="適用確率")
    std: float = Field(default=0.01, description="標準偏差")


class TimeScalingConfig(BaseModel):
    """時間スケーリング設定."""

    probability: float = Field(default=0.3, description="適用確率")
    scale_range: list[float] = Field(default=[0.9, 1.1], description="スケール範囲")


class PartialMaskingConfig(BaseModel):
    """部分マスキング設定."""

    probability: float = Field(default=0.2, description="適用確率")
    mask_length_range: list[int] = Field(default=[5, 20], description="マスク長範囲")
    mask_ratio: float = Field(default=0.1, description="マスク比率")


class AugmentationConfig(BaseModel):
    """データ拡張設定."""

    gaussian_noise: GaussianNoiseConfig = Field(default_factory=GaussianNoiseConfig, description="ガウシアンノイズ")
    time_scaling: TimeScalingConfig = Field(default_factory=TimeScalingConfig, description="時間スケーリング")
    partial_masking: PartialMaskingConfig = Field(default_factory=PartialMaskingConfig, description="部分マスキング")


class TrainerConfig(BaseModel):
    """PyTorch Lightning Trainer設定."""

    accelerator: str = Field(default="auto", description="アクセラレータ")
    devices: str = Field(default="auto", description="デバイス")
    precision: str = Field(default="16-mixed", description="精度")
    max_epochs: int = Field(default=100, description="最大エポック数")
    gradient_clip_val: float = Field(default=1.0, description="勾配クリッピング値")
    accumulate_grad_batches: int = Field(default=1, description="勾配累積バッチ数")
    deterministic: bool = Field(default=False, description="決定論的フラグ")
    benchmark: bool = Field(default=True, description="ベンチマークフラグ")
    enable_checkpointing: bool = Field(default=True, description="チェックポイント有効フラグ")
    default_root_dir: str = Field(default="../../outputs/exp004", description="デフォルトルートディレクトリ")
    enable_progress_bar: bool = Field(default=True, description="プログレスバー有効フラグ")
    log_every_n_steps: int = Field(default=50, description="ログ間隔ステップ数")
    check_val_every_n_epoch: int = Field(default=1, description="検証実行間隔エポック数")
    val_check_interval: float = Field(default=1.0, description="検証チェック間隔")
    enable_early_stopping: bool = Field(default=True, description="早期停止有効フラグ")


class ModelCheckpointConfig(BaseModel):
    """モデルチェックポイント設定."""

    monitor: str = Field(default="val_cmi_score", description="監視メトリクス")
    mode: str = Field(default="max", description="監視モード")
    save_top_k: int = Field(default=3, description="保存する上位K個")
    save_last: bool = Field(default=True, description="最後のモデル保存フラグ")
    filename: str = Field(
        default="epoch-{epoch:02d}-val_cmi_score-{val_cmi_score:.4f}", description="ファイル名フォーマット"
    )
    auto_insert_metric_name: bool = Field(default=False, description="メトリクス名自動挿入フラグ")


class EarlyStoppingConfig(BaseModel):
    """早期停止設定."""

    monitor: str = Field(default="val_cmi_score", description="監視メトリクス")
    mode: str = Field(default="max", description="監視モード")
    patience: int = Field(default=15, description="待機エポック数")
    min_delta: float = Field(default=0.001, description="最小変化量")
    verbose: bool = Field(default=True, description="詳細出力フラグ")


class LearningRateMonitorConfig(BaseModel):
    """学習率監視設定."""

    logging_interval: str = Field(default="epoch", description="ログ間隔")


class RichProgressBarConfig(BaseModel):
    """リッチプログレスバー設定."""

    enable: bool = Field(default=True, description="有効フラグ")


class CallbacksConfig(BaseModel):
    """コールバック設定."""

    model_checkpoint: ModelCheckpointConfig = Field(
        default_factory=ModelCheckpointConfig, description="モデルチェックポイント"
    )
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig, description="早期停止")
    lr_monitor: LearningRateMonitorConfig = Field(default_factory=LearningRateMonitorConfig, description="学習率監視")
    rich_progress_bar: RichProgressBarConfig = Field(
        default_factory=RichProgressBarConfig, description="リッチプログレスバー"
    )


class LightningConfig(BaseModel):
    """PyTorch Lightning設定."""

    trainer: TrainerConfig = Field(default_factory=TrainerConfig, description="Trainer設定")
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig, description="コールバック設定")


class WandbConfig(BaseModel):
    """WandB設定."""

    enabled: bool = Field(default=True, description="有効フラグ")
    project: str = Field(default="CMI-Detect-Behavior-with-Sensor-Data", description="プロジェクト名")
    name: str = Field(default="exp004", description="実行名")
    tags: list[str] = Field(default=["exp004", "imu_only", "squeezeformer"], description="タグ")


class LoggingConfig(BaseModel):
    """ログ設定."""

    level: str = Field(default="INFO", description="ログレベル")
    log_dir: str = Field(default="logs", description="ログディレクトリ")
    log_file: str = Field(default="exp004.log", description="ログファイル名")
    wandb: WandbConfig = Field(default_factory=WandbConfig, description="WandB設定")


class HardwareConfig(BaseModel):
    """ハードウェア設定."""

    use_mixed_precision: bool = Field(default=True, description="混合精度使用フラグ")
    gradient_clip_norm: float = Field(default=1.0, description="勾配クリッピングノルム")


class InferenceConfig(BaseModel):
    """推論設定."""

    batch_size: int = Field(default=64, description="バッチサイズ")
    use_tta: bool = Field(default=False, description="TTA使用フラグ")
    tta_iterations: int = Field(default=5, description="TTA反復回数")
    ensemble_strategy: Literal["average", "voting", "weighted"] = Field(
        default="average", description="アンサンブル戦略"
    )


class Config(BaseSettings):
    """メイン設定クラス."""

    # Root directory
    root: str = Field(default="../../outputs/exp004", description="ルートディレクトリ")

    # 各セクションの設定
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig, description="実験設定")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="パス設定")
    data: DataConfig = Field(default_factory=DataConfig, description="データ設定")
    model: ModelConfig = Field(default_factory=ModelConfig, description="モデル設定")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="訓練設定")
    loss: LossConfig = Field(default_factory=LossConfig, description="損失関数設定")
    schedule_free: ScheduleFreeConfig = Field(default_factory=ScheduleFreeConfig, description="Schedule Free設定")
    val: ValidationConfig = Field(default_factory=ValidationConfig, description="バリデーション設定")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig, description="前処理設定")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig, description="データ拡張設定")
    lightning: LightningConfig = Field(default_factory=LightningConfig, description="PyTorch Lightning設定")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="ログ設定")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="ハードウェア設定")
    inference: InferenceConfig = Field(default_factory=InferenceConfig, description="推論設定")

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
        default=["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"], description="IMU特徴量列名"
    )

    class Config:
        """pydantic設定."""

        env_prefix = "EXP004_"  # 環境変数プレフィックス
        case_sensitive = False
        extra = "forbid"  # 未定義フィールドを禁止

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return self.dict()

    def validate_paths(self) -> bool:
        """パスの存在確認."""
        paths_to_check = [
            self.data.train_path,
            self.data.test_path,
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
            Path(self.lightning.trainer.default_root_dir),
        ]

        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


def load_config() -> Config:
    """設定の読み込み."""
    # デフォルト設定を使用
    return Config()


if __name__ == "__main__":
    """設定のテスト実行."""
    # デフォルト設定でのテスト
    config = Config()
    print("✓ Default config created successfully")

    # 設定値の確認
    print(f"Model input_dim: {config.model.input_dim}")
    print(f"Training batch_size: {config.training.batch_size}")
    print(f"Number of target gestures: {len(config.target_gestures)}")
    print(f"Number of IMU features: {len(config.imu_features)}")

    # バリデーションテスト
    try:
        config.validate_paths()
        print("✓ Path validation completed")
    except Exception as e:
        print(f"! Path validation warning: {e}")

    print("Config test completed successfully!")
