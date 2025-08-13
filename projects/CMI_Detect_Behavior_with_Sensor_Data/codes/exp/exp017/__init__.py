"""exp017: Soft F1 Loss optimization.

This experiment extends exp013 by changing the default loss function to Soft F1 Loss
for better F1 score optimization.

Key improvements:
- Default loss function changed from ACLS to Soft F1 Loss
- Direct optimization for F1 metric which is closer to the competition metric
- Maintains all other improvements from exp013:
  - Physics-based IMU features
  - Attention mask for missing values handling
  - Multiple loss function support (cmi, cmi_focal, soft_f1, acls, label_smoothing, mbls)
"""
