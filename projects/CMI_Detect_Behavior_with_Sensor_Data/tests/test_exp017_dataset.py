"""Test dataset module for exp017."""

from codes.exp.exp017.dataset import CMIDataModule


def test_datamodule_creation():
    """Test that CMIDataModule can be created."""
    # Simple configuration for testing
    config = {
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "train_data_path": "data/train.csv",
            "test_data_path": "data/test.csv",
            "gesture_labels": [],
            "max_sequence_length": 1000,
            "sequence_stride": 100,
            "sequence_overlap": 0.5,
            "demographics_path": None,
        },
        "preprocessing": {
            "use_physics_features": True,
            "physics_features": ["jerk", "angular_velocity_magnitude"],
            "stats_window_size": 100,
            "normalize_method": "standard",
            "handle_missing": "mask",
        },
        "augmentation": {
            "enabled": False,
        },
        "validation": {
            "strategy": "kfold",
            "params": {"n_splits": 5, "shuffle": True, "random_state": 42},
        },
        "length_grouping": {
            "enabled": False,
        },
    }

    # DataModuleを作成（実際のデータは読み込まない）
    datamodule = CMIDataModule(config)
    assert datamodule is not None
    assert datamodule.batch_size == 32
    assert datamodule.num_workers == 4
