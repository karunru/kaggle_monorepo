"""exp028: IMU-only LSTM with ResidualSE-CNN and BiGRU attention baseline.

This experiment implements an IMU-only model based on jiazhuang's Kaggle notebook
(cmi-imu-only-lstm), featuring ResidualSE-CNN blocks and BiGRU with attention mechanism.

Based on:
- https://www.kaggle.com/code/jiazhuang/cmi-imu-only-lstm: Kaggleノートブックベースライン

Key features:
- IMU-only architecture (19 physical features)
- ResidualSE-CNN blocks with Squeeze-and-Excitation attention
- Bidirectional GRU with attention mechanism
- Physical feature engineering (gravity removal, angular velocity, etc.)
- K-fold cross-validation with ensemble prediction
- TimeSeriesAugmentation and Mixup data augmentation
- Demographics disabled for fair comparison with IMU-only baseline
"""
