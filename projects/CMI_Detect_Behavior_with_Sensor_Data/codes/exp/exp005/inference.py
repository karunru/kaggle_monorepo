"""CMI 2025 Submission - exp005用推論パイプライン(デモサブミッション形式)."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))  # For src.kaggle_evaluation
# Local imports (from exp005 directory)
from config import Config
from dataset import SingleSequenceIMUDataset
from model import CMISqueezeformer
from src.utils.logger import create_logger
from src.utils.seed_everything import seed_everything

# Warnings suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Kaggle evaluation API
try:
    # Try to import from src directory first
    import src.kaggle_evaluation.cmi_inference_server

    kaggle_eval_module = src.kaggle_evaluation
except ImportError:
    # Fallback to data directory if src import fails
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "data"))
        import kaggle_evaluation.cmi_inference_server

        kaggle_eval_module = kaggle_evaluation
    except ImportError:
        kaggle_eval_module = None
        print("Warning: Could not import kaggle_evaluation module")

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


def load_models():
    """モデルの読み込み(グローバル初期化)."""
    global config, models, device, logger

    # 設定の読み込み
    config = Config()

    # シード設定
    seed_everything(config.training.seed)

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ロガー設定
    logger = create_logger(__name__)

    # モデルパスの取得(CV結果から)
    model_paths = []
    output_dir = Path(config.paths.output_dir)

    for fold in range(config.training.n_folds):
        fold_dir = output_dir / f"fold_{fold}"
        if fold_dir.exists():
            # checkpointsディレクトリから最良モデルを探す
            checkpoint_dir = fold_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
                if checkpoint_files:
                    # 最新のチェックポイントを使用
                    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                    model_paths.append(str(latest_checkpoint))
                    print(f"Found model for fold {fold}: {latest_checkpoint}")

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
    単一シーケンスに対する予測.

    Args:
        sequence: シーケンスデータ(Polarsデータフレーム)
        demographics: 人口統計データ(Polarsデータフレーム)

    Returns:
        予測されたジェスチャー名(文字列)
    """
    global config, models, device

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
                binary_probs = torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy()

                batch_multiclass_probs.append(multiclass_probs)
                batch_binary_probs.append(binary_probs)

            # アンサンブル(平均)
            ensemble_multiclass_probs = np.mean(batch_multiclass_probs, axis=0)
            ensemble_binary_probs = np.mean(batch_binary_probs, axis=0)

            all_multiclass_probs.append(ensemble_multiclass_probs)
            all_binary_probs.append(ensemble_binary_probs)

    # 結果の処理
    multiclass_probs = all_multiclass_probs[0][0]  # [batch_size=1, num_classes] -> [num_classes]
    all_binary_probs[0][0]  # [batch_size=1] -> scalar

    # マルチクラス予測
    predicted_class = np.argmax(multiclass_probs)

    # ジェスチャー名の取得
    predicted_gesture = GESTURE_NAMES[predicted_class] if predicted_class < len(GESTURE_NAMES) else "Text on phone"

    return predicted_gesture


# メイン実行部分
if __name__ == "__main__":
    # モデルの初期化
    print("Loading models...")
    load_models()
    print("Models loaded successfully")

    # kaggle_evaluationが利用可能な場合のみ推論サーバーを実行
    if kaggle_eval_module is not None:
        # モジュール名に応じてインスタンスを作成
        if "src.kaggle_evaluation" in sys.modules:
            inference_server = src.kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
        else:
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            inference_server.serve()
        elif os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") == "local":
            print("Running inference on local machine...")
            inference_server.run_local_gateway(
                data_paths=(
                    "../../data/test.csv",
                    "../../data/test_demographics.csv",
                )
            )
        else:
            inference_server.run_local_gateway(
                data_paths=(
                    "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                    "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
                )
            )
    else:
        # kaggle_evaluationが利用できない場合のローカルテスト
        print("\nRunning in local test mode without kaggle_evaluation...")
        print("To test inference, call predict() function with sequence and demographics data.")
        print("Example: predicted_gesture = predict(sequence_df, demographics_df)")
