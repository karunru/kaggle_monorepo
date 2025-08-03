"""Simple inference runner for CMI Detect Behavior with Sensor Data."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
from omegaconf import OmegaConf
from src.inference import create_inference_pipeline


def main():
    """Run inference with simplified setup."""
    print("🚀 CMI Simple Inference Runner")
    print("=" * 40)

    # Configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / "config.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    config = OmegaConf.load(config_path)

    # Check for test data
    test_path = Path(config.data.test_path)
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("   Please ensure test data is available at the specified path")
        return

    # Load test data
    print(f"📊 Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"   Loaded {len(test_df)} test samples")
    print(f"   Unique sequences: {test_df['sequence_id'].nunique()}")

    # Find available model checkpoints
    available_folds = []
    for fold in range(config.val.params.n_splits):
        model_path = exp_dir / f"best_model_fold_{fold}.pt"
        if model_path.exists():
            available_folds.append(fold)

    if not available_folds:
        print("❌ No trained model checkpoints found")
        print("   Please run training first to generate model checkpoints")
        return

    print(f"🔮 Found models for folds: {available_folds}")

    # Run inference for available folds
    fold_predictions = []

    for fold in available_folds:
        print(f"\n📈 Running inference for fold {fold}")

        try:
            # Create inference pipeline
            inference_pipeline = create_inference_pipeline(
                experiment_dir=str(exp_dir),
                fold=fold,
                device="auto",
            )

            # Make predictions
            predictions_df = inference_pipeline.predict_batch(
                test_df,
                sequence_id_col="sequence_id",
                use_tta=False,  # Disable TTA for faster inference
                imu_only=False,
            )

            fold_predictions.append(predictions_df)
            print(f"   ✅ Fold {fold} completed")

        except Exception as e:
            print(f"   ⚠️  Error in fold {fold}: {e}")
            continue

    if not fold_predictions:
        print("❌ No successful predictions generated")
        return

    # Create ensemble if multiple folds available
    if len(fold_predictions) > 1:
        print(f"\n🎯 Creating ensemble from {len(fold_predictions)} folds")

        ensemble_predictions = []
        sequence_ids = fold_predictions[0]["sequence_id"].unique()

        for seq_id in sequence_ids:
            # Collect predictions from all folds
            seq_predictions = []
            for fold_df in fold_predictions:
                if seq_id in fold_df["sequence_id"].values:
                    pred = fold_df[fold_df["sequence_id"] == seq_id]["gesture"].iloc[0]
                    seq_predictions.append(pred)

            if seq_predictions:
                # Majority voting
                from collections import Counter

                most_common = Counter(seq_predictions).most_common(1)[0][0]
                ensemble_predictions.append(
                    {
                        "sequence_id": seq_id,
                        "prediction": most_common,
                    }
                )

        final_predictions = pd.DataFrame(ensemble_predictions)
    else:
        # Single fold
        print("\n📊 Using single fold predictions")
        final_predictions = fold_predictions[0][["sequence_id", "gesture"]].rename(columns={"gesture": "prediction"})

    # Save submission
    submission_path = exp_dir / "submission.csv"
    final_predictions.to_csv(submission_path, index=False)

    print(f"\n🏆 Submission saved to {submission_path}")
    print(f"   Total predictions: {len(final_predictions)}")

    # Show prediction distribution
    print("\n📈 Prediction Distribution:")
    print("-" * 30)
    prediction_counts = final_predictions["prediction"].value_counts()
    for gesture, count in prediction_counts.head(10).items():
        percentage = (count / len(final_predictions)) * 100
        print(f"   {gesture}: {count} ({percentage:.1f}%)")

    if len(prediction_counts) > 10:
        print(f"   ... and {len(prediction_counts) - 10} more")

    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
