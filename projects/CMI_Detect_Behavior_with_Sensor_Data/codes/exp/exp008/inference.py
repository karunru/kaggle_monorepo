"""CMI 2025 Submission - exp007用推論パイプライン(デモサブミッション形式)."""

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


def get_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """val_lossが最小のcheckpointを取得."""
    ckpt_files = list(checkpoint_dir.glob("epoch-*-val_loss-*.ckpt"))
    if not ckpt_files:
        return None

    # ファイル名から val_loss を抽出して最小値を選択
    def extract_loss(ckpt_path):
        filename = ckpt_path.name
        # epoch-XX-val_loss-Y.YYYY.ckpt から Y.YYYY を抽出
        parts = filename.split("-")
        for i, part in enumerate(parts):
            if part == "val_loss" and i + 1 < len(parts):
                loss_part = parts[i + 1].replace(".ckpt", "")
                return float(loss_part)
        return float("inf")  # lossの場合，デフォルトは無限大

    best_ckpt = min(ckpt_files, key=extract_loss)
    return best_ckpt


def load_models():
    """モデルの読み込み(グローバル初期化)."""
    global config, models, device, logger

    # 設定の読み込み
    config = Config()
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") != "local":
        config.paths.output_dir = "/kaggle/input/cmi-exp007/"

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
                    best_checkpoint = get_best_checkpoint(checkpoint_dir)
                    if best_checkpoint:
                        model_paths.append(str(best_checkpoint))

    if not model_paths:
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
        except Exception:
            # 1つでもモデルが読み込めれば続行
            continue

    if not models:
        raise RuntimeError("Failed to load any models")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    単一シーケンスに対する予測.

    Args:
        sequence: シーケンスデータ(Polarsデータフレーム)
        demographics: 人口統計データ(Polarsデータフレーム)

    Returns:
        予測されたジェスチャー名(文字列)
    """
    global config, models, device
    print(f"{config=}")
    print(f"{device=}")

    # データセットの作成
    dataset = SingleSequenceIMUDataset(sequence, target_sequence_length=config.preprocessing.target_sequence_length)

    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 推論時は並列処理を無効化
        pin_memory=True,
    )

    # 予測実行
    all_multiclass_probs = []
    all_binary_probs = []

    with torch.no_grad():
        for batch in dataloader:
            imu = batch["imu"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # アンサンブル予測
            batch_multiclass_probs = []
            batch_binary_probs = []

            for model in models:
                # 通常推論
                multiclass_logits, binary_logits = model(imu, attention_mask)
                multiclass_probs = torch.softmax(multiclass_logits, dim=-1).cpu().numpy()
                print(f"{multiclass_probs=}")
                binary_probs = torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy()
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
