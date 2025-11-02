"""CMI 2025 Submission - exp013用推論パイプライン(Demographics統合版)."""

import sys
from pathlib import Path

# Add codes directory to path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from config import Config
from dataset import SingleSequenceIMUDataset
from model import CMISqueezeformer

# Imports from project root
from src.utils.logger import create_logger
from src.utils.seed_everything import seed_everything
from torch.utils.data import DataLoader

# Warnings suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


if os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") != "local":
    sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")
    import kaggle_evaluation.cmi_inference_server

    kaggle_evaluation_module = kaggle_evaluation.cmi_inference_server
else:
    from src.kaggle_evaluation import cmi_inference_server

    kaggle_evaluation_module = cmi_inference_server

# Global configuration and models
config = None
models = None
device = None
logger = None

# Gesture name mapping
GESTURE_NAMES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Drink from bottle/cup",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Feel around in tray and pull out an object",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Glasses on/off",
    "Neck - pinch skin",
    "Neck - scratch",
    "Pinch knee/leg skin",
    "Pull air toward your face",
    "Scratch knee/leg skin",
    "Text on phone",
    "Wave hello",
    "Write name in air",
    "Write name on leg",
]


def get_latest_version_dir(fold_dir: Path) -> Path | None:
    """最新のversionディレクトリを取得."""
    version_dirs = [d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
    if not version_dirs:
        return None

    # version_X の X を数値として取得してソート
    latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[1]))
    return latest_version


def get_best_checkpoint_by_config(checkpoint_dir: Path, config: Config) -> Path | None:
    """configに基づいて最適なcheckpointを取得."""
    checkpoint_config = config.lightning.callbacks.model_checkpoint
    monitor = checkpoint_config.monitor
    mode = checkpoint_config.mode
    filename_pattern = checkpoint_config.filename

    # filename patternからglob patternを生成
    # "epoch-{epoch:02d}-val_cmi_score-{val_cmi_score:.4f}" → "epoch-*-val_cmi_score-*.ckpt"
    glob_pattern = filename_pattern.replace("{epoch:02d}", "*")
    # メトリクス名部分を動的に置換
    metric_placeholder = f"{{{monitor}:.4f}}"
    glob_pattern = glob_pattern.replace(metric_placeholder, "*") + ".ckpt"

    ckpt_files = list(checkpoint_dir.glob(glob_pattern))
    if not ckpt_files:
        return None

    # メトリクス値を抽出して最適化
    def extract_metric(ckpt_path):
        filename = ckpt_path.name
        parts = filename.split("-")
        # monitor名から"val_"プレフィックスを除去(例: val_cmi_score -> cmi_score)
        metric_name = monitor.replace("val_", "") if monitor.startswith("val_") else monitor
        for i, part in enumerate(parts):
            if part == metric_name and i + 1 < len(parts):
                metric_part = parts[i + 1].replace(".ckpt", "")
                return float(metric_part)
        return float("-inf") if mode == "max" else float("inf")

    # modeに応じて最適化
    if mode == "max":
        best_ckpt = max(ckpt_files, key=extract_metric)
    else:
        best_ckpt = min(ckpt_files, key=extract_metric)

    return best_ckpt


def load_models():
    """モデルの読み込み(グローバル初期化)."""
    global config, models, device, logger

    # 設定の読み込み
    config = Config()
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") != "local":
        config.paths.output_dir = f"/kaggle/input/cmi-{config.experiment.exp_num}/"

    # シード設定
    seed_everything(config.training.seed)

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ロガー設定
    logger = create_logger(__name__)

    # モデルパスの取得(CV結果から)
    model_paths = []
    output_dir = Path(config.paths.output_dir)

    for fold in range(config.val.params.n_splits):
        fold_dir = output_dir / f"fold_{fold}"
        if fold_dir.exists():
            # 最新versionから最高cmi_scoreのモデルを探す
            latest_version_dir = get_latest_version_dir(fold_dir)
            if latest_version_dir:
                checkpoint_dir = latest_version_dir / "checkpoints"
                if checkpoint_dir.exists():
                    best_checkpoint = get_best_checkpoint_by_config(checkpoint_dir, config)
                    if best_checkpoint:
                        model_paths.append(str(best_checkpoint))
                        print(f"Found model for fold {fold}: {best_checkpoint} (from {latest_version_dir.name})")

    if not model_paths:
        print("No trained models found. Using dummy model.")
        models = []
        return

    # モデルの読み込み
    models = []
    for model_path in model_paths:
        try:
            model = CMISqueezeformer.load_from_checkpoint(model_path)
            model.eval()
            model.to(device)
            models.append(model)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            # 1つでもモデルが読み込めれば続行
            continue

    if not models:
        raise RuntimeError("Failed to load any models")

    print(f"Successfully loaded {len(models)} models")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    IMU-only LSTM予測 (exp028 - jiazhuang notebook compatible).

    Args:
        sequence: シーケンスデータ(Polarsデータフレーム)
        demographics: 人口統計データ(Polarsデータフレーム) - IMU-onlyモードでは無視

    Returns:
        予測されたジェスチャー名(文字列)
    """
    print("in predict (IMU-only mode)")
    global config, models, device
    print(f"{config=}")
    print(f"{models=}")
    print(f"{device=}")

    # IMU-only: Demographics常に無効化
    use_demographics = False  # Force disabled for IMU-only experiment

    # subjectを取得（シーケンスデータから）
    subject = None
    if "subject" in sequence.columns:
        subject = sequence.get_column("subject")[0]

    # データセットの作成
    dataset = SingleSequenceIMUDataset(
        sequence,
        target_sequence_length=config.preprocessing.target_sequence_length,
        use_demographics=use_demographics,
        demographics_data=demographics if use_demographics else None,
        subject=subject,
        demographics_config=config.demographics.model_dump(),
    )
    print(f"{dataset=}")
    print(f"Using demographics: {use_demographics}")

    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 推論時は並列処理を無効化
        pin_memory=True,
    )
    print(f"{dataloader=}")

    # 予測実行
    all_multiclass_probs = []
    all_binary_probs = []

    with torch.no_grad():
        for batch in dataloader:
            print(f"{batch=}")
            imu = batch["imu"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Demographics データの処理
            demographics_batch = None
            if use_demographics and "demographics" in batch:
                demographics_batch = {k: v.to(device) for k, v in batch["demographics"].items()}
                print(f"Demographics batch: {demographics_batch}")

            # アンサンブル予測
            batch_multiclass_probs = []
            batch_binary_probs = []

            for model in models:
                # IMU-only LSTM推論 (single logits output)
                logits = model(imu, attention_mask)

                # Convert to probabilities
                multiclass_probs = torch.softmax(logits, dim=-1).cpu().numpy()
                print(f"{multiclass_probs=}")

                # For binary classification, compute target vs non-target
                # Target gestures: indices 0-7, Non-target: indices 8-17 (based on GESTURE_NAMES)
                target_prob = multiclass_probs[:, :8].sum(axis=1)  # Sum of target gesture probabilities
                binary_probs = target_prob  # Binary probability (target vs non-target)
                print(f"{binary_probs=}")

                batch_multiclass_probs.append(multiclass_probs)
                batch_binary_probs.append(binary_probs)

            # アンサンブル(平均)
            print(f"{batch_multiclass_probs=}")
            print(f"{batch_binary_probs=}")
            ensemble_multiclass_probs = np.mean(batch_multiclass_probs, axis=0)
            print(f"{ensemble_multiclass_probs=}")
            ensemble_binary_probs = np.mean(batch_binary_probs, axis=0)
            print(f"{ensemble_binary_probs=}")

            all_multiclass_probs.append(ensemble_multiclass_probs)
            all_binary_probs.append(ensemble_binary_probs)

    # 結果の処理
    print(f"{all_multiclass_probs=}")
    multiclass_probs = all_multiclass_probs[0][0]  # [batch_size=1, num_classes] -> [num_classes]

    # マルチクラス予測
    predicted_class = np.argmax(multiclass_probs)
    print(f"{predicted_class=}")

    # ジェスチャー名の取得
    predicted_gesture = GESTURE_NAMES[predicted_class] if predicted_class < len(GESTURE_NAMES) else "Text on phone"
    print(f"{predicted_gesture=}")

    return predicted_gesture


# メイン実行部分
if __name__ == "__main__":
    # モデルの初期化
    print("Loading models...")
    load_models()
    print("Models loaded successfully")

    # kaggle_evaluationが利用可能な場合のみ推論サーバーを実行
    inference_server = kaggle_evaluation_module.CMIInferenceServer(predict)
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    elif os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") == "local":
        print("Running inference on local machine...")
        inference_server.run_local_gateway(
            data_paths=(
                "../../../data/test.csv",
                "../../../data/test_demographics.csv",
            )
        )
    else:
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
            )
        )
