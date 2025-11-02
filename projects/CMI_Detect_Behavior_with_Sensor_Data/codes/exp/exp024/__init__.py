"""exp019: 利き手反転データオーグメンテーション実験.

This experiment extends exp018 by adding handedness flip data augmentation
to improve model generalization and handle handedness imbalance in the dataset.

Based on:
- docs/handedness_augmentation_plan.md: 利き手反転オーグメンテーションの技術詳細

Key improvements:
- Handedness flip data augmentation for IMU data
- Y-axis inversion for acceleration and rotation data
- Probabilistic augmentation application (50% chance by default)
- Better generalization across left-handed and right-handed subjects
- Enhanced robustness through balanced handedness representation
"""
