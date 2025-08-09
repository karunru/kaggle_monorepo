"""exp012: ACLS integration for calibration improvement.

This experiment extends exp011 by integrating Adaptive and Conditional Label Smoothing (ACLS)
for both binary and multiclass losses to improve model calibration.

Based on:
- ACLS: Adaptive and Conditional Label Smoothing for Network Calibration (ICCV 2023)
- GitHub: https://github.com/cvlab-yonsei/ACLS

Key improvements:
- ACLS loss for improved model calibration
- Margin-based Label Smoothing (MbLS) support
- Label Smoothing Cross-Entropy for both binary and multiclass
- Physics-based IMU features from exp011
- Attention mask for missing values handling
"""