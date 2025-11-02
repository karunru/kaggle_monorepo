"""Inference script for CMI Detect Behavior with Sensor Data - Experiment 001."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import hydra
import pandas as pd
from omegaconf import DictConfig
from src.inference import create_inference_pipeline


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """Run inference on test data."""
    print("🚀 Starting CMI Inference Pipeline")
    print("=" * 50)

    # Create experiment directory
    exp_dir = Path(config.paths.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("📊 Loading test data...")
    test_df = pd.read_csv(config.data.test_path)
    print(f"   Loaded {len(test_df)} test samples")
    print(f"   Unique sequences: {test_df['sequence_id'].nunique()}")

    # Create inference pipeline for each fold
    fold_predictions = []

    for fold in range(config.val.params.n_splits):
        print(f"\n🔮 Running inference for fold {fold + 1}/{config.val.params.n_splits}")

        try:
            # Create inference pipeline
            inference_pipeline = create_inference_pipeline(
                experiment_dir=str(exp_dir),
                fold=fold,
                device=config.training.device,
            )

            # Make predictions
            predictions_df = inference_pipeline.predict_batch(
                test_df,
                sequence_id_col="sequence_id",
                use_tta=config.inference.get("use_tta", False),
                imu_only=config.inference.get("imu_only", False),
            )

            # Save fold predictions
            fold_output_path = exp_dir / f"predictions_fold_{fold}.csv"
            predictions_df[["sequence_id", "gesture"]].rename(columns={"gesture": "prediction"}).to_csv(
                fold_output_path, index=False
            )

            fold_predictions.append(predictions_df)
            print(f"   ✅ Fold {fold} predictions saved to {fold_output_path}")

        except FileNotFoundError as e:
            print(f"   ⚠️  Skipping fold {fold}: {e}")
            continue

    if not fold_predictions:
        print("❌ No valid model checkpoints found for inference")
        return

    # Create ensemble predictions
    print(f"\n🎯 Creating ensemble from {len(fold_predictions)} folds")

    # Simple majority voting ensemble
    ensemble_predictions = []
    sequence_ids = fold_predictions[0]["sequence_id"].unique()

    for seq_id in sequence_ids:
        # Get predictions from all folds for this sequence
        seq_predictions = []
        for fold_df in fold_predictions:
            if seq_id in fold_df["sequence_id"].values:
                pred = fold_df[fold_df["sequence_id"] == seq_id]["gesture"].iloc[0]
                seq_predictions.append(pred)

        if seq_predictions:
            # Majority vote
            from collections import Counter

            most_common = Counter(seq_predictions).most_common(1)[0][0]
            ensemble_predictions.append(
                {
                    "sequence_id": seq_id,
                    "prediction": most_common,
                }
            )

    # Save ensemble submission
    ensemble_df = pd.DataFrame(ensemble_predictions)
    submission_path = exp_dir / "submission.csv"
    ensemble_df.to_csv(submission_path, index=False)

    print(f"🏆 Final submission saved to {submission_path}")
    print(f"   Total predictions: {len(ensemble_df)}")

    # Print prediction distribution
    print("\n📈 Prediction Distribution:")
    print("-" * 40)
    prediction_counts = ensemble_df["prediction"].value_counts()
    for gesture, count in prediction_counts.items():
        percentage = (count / len(ensemble_df)) * 100
        print(f"   {gesture}: {count} ({percentage:.1f}%)")

    # Check for target gestures
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

    target_count = sum(prediction_counts.get(gesture, 0) for gesture in target_gestures)
    target_percentage = (target_count / len(ensemble_df)) * 100

    print(f"\n🎯 Target BFRB Gestures: {target_count} ({target_percentage:.1f}%)")
    print("=" * 50)
    print("✅ Inference complete!")


if __name__ == "__main__":
    main()
