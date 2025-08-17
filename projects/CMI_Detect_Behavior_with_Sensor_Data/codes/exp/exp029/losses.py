"""ACLS損失関数群実装.

Based on cvlab-yonsei/ACLS repository:
https://github.com/cvlab-yonsei/ACLS

Implements:
- ACLS (Adaptive and Conditional Label Smoothing)
- MbLS (Margin-based Label Smoothing)
- LabelSmoothingCrossEntropy
- Binary versions of above losses
"""

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross-Entropy Loss.

    Based on: https://github.com/cvlab-yonsei/ACLS/blob/main/calibrate/losses/label_smoothing.py
    """

    def __init__(self, alpha: float = 0.1, ignore_index: int = -100, reduction: str = "mean"):
        """
        Initialize Label Smoothing Cross-Entropy Loss.

        Args:
            alpha: Label smoothing parameter (0.0 = no smoothing)
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Label smoothed cross-entropy loss
        """
        # Handle multi-dimensional inputs
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        if targets.dim() > 1:
            targets = targets.view(-1)

        num_classes = inputs.size(-1)

        # Create one-hot targets
        one_hot = F.one_hot(targets, num_classes).float()

        # Apply label smoothing
        # smooth_targets = (1 - alpha) * one_hot + alpha / num_classes
        smooth_targets = one_hot * (1 - self.alpha) + self.alpha / num_classes

        # Calculate log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Handle ignore_index
        if self.ignore_index != -100:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask

        # Apply reduction
        if self.reduction == "mean":
            if self.ignore_index != -100:
                return loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE(nn.Module):
    """Label Smoothing for Binary Classification.

    Applies label smoothing to binary cross-entropy loss.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize Label Smoothing BCE.

        Args:
            alpha: Label smoothing parameter
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size] or [batch_size, 1]
            targets: Target labels [batch_size]

        Returns:
            Label smoothed BCE loss
        """
        # Ensure targets are float
        targets = targets.float()

        # Apply label smoothing
        # positive labels: 1 -> (1 - alpha)
        # negative labels: 0 -> alpha
        smooth_targets = targets * (1 - self.alpha) + (1 - targets) * self.alpha

        return F.binary_cross_entropy_with_logits(inputs, smooth_targets)


class MbLS(nn.Module):
    """Margin-based Label Smoothing.

    Based on: https://github.com/cvlab-yonsei/ACLS/blob/main/calibrate/losses/mbls.py
    """

    def __init__(
        self, margin: float = 10.0, alpha: float = 0.1, ignore_index: int = -100, alpha_schedule: str | None = None
    ):
        """
        Initialize Margin-based Label Smoothing.

        Args:
            margin: Margin parameter
            alpha: Weight for margin penalty
            ignore_index: Index to ignore
            alpha_schedule: Alpha scheduling method ('add', 'multiply', 'step', None)
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.alpha_schedule = alpha_schedule

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            MbLS loss: CE + alpha * max(0, max(l^n) - l^n - margin)
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)

        # Margin-based penalty
        if self.alpha > 0:
            # Get max logit for each sample
            max_logits = inputs.max(dim=-1, keepdim=True)[0]  # [batch_size, 1]

            # Compute margin penalty: max(0, max(l^n) - l^n - margin)
            logit_diff = max_logits - inputs  # [batch_size, num_classes]
            margin_penalty = F.relu(logit_diff - self.margin)  # [batch_size, num_classes]

            # Average over classes and batch
            margin_penalty = margin_penalty.mean()

            total_loss = ce_loss + self.alpha * margin_penalty
        else:
            total_loss = ce_loss

        return total_loss

    def schedule_alpha(self, epoch: int, schedule_step: float = 0.01, schedule_factor: float = 1.1):
        """
        Schedule alpha parameter.

        Args:
            epoch: Current epoch
            schedule_step: Step size for 'add' schedule
            schedule_factor: Factor for 'multiply' schedule
        """
        if self.alpha_schedule == "add":
            self.alpha += schedule_step
        elif self.alpha_schedule == "multiply":
            self.alpha *= schedule_factor
        elif self.alpha_schedule == "step":
            if epoch > 0 and epoch % 10 == 0:  # Every 10 epochs
                self.alpha *= schedule_factor


class MbLSBinary(nn.Module):
    """Margin-based Label Smoothing for Binary Classification."""

    def __init__(self, margin: float = 10.0, alpha: float = 0.1):
        """
        Initialize Binary MbLS.

        Args:
            margin: Margin parameter
            alpha: Weight for margin penalty
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size] or [batch_size, 1]
            targets: Target labels [batch_size]

        Returns:
            Binary MbLS loss
        """
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())

        # Margin-based penalty for binary case
        if self.alpha > 0:
            # For binary case, we consider positive and negative logits
            if inputs.dim() > 1:
                inputs = inputs.squeeze(-1)

            # Create binary logits [batch_size, 2]
            binary_logits = torch.stack([-inputs, inputs], dim=-1)

            # Get max logit for each sample
            max_logits = binary_logits.max(dim=-1, keepdim=True)[0]

            # Compute margin penalty
            logit_diff = max_logits - binary_logits
            margin_penalty = F.relu(logit_diff - self.margin).mean()

            total_loss = bce_loss + self.alpha * margin_penalty
        else:
            total_loss = bce_loss

        return total_loss


class ACLS(nn.Module):
    """Adaptive and Conditional Label Smoothing.

    Based on: https://github.com/cvlab-yonsei/ACLS/blob/main/calibrate/losses/acls.py
    """

    def __init__(
        self,
        pos_lambda: float = 1.0,
        neg_lambda: float = 0.1,
        alpha: float = 0.1,
        margin: float = 10.0,
        num_classes: int = 200,
    ):
        """
        Initialize ACLS loss.

        Args:
            pos_lambda: Positive sample regularization weight
            neg_lambda: Negative sample regularization weight
            alpha: Regularization term weight
            margin: Margin for distance calculation
            num_classes: Number of classes
        """
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            ACLS loss: CE + regularization term
        """
        # Handle multi-dimensional inputs
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        if targets.dim() > 1:
            targets = targets.view(-1)

        batch_size = inputs.size(0)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets)

        # Regularization term
        reg_loss = self.get_reg(inputs, targets)

        total_loss = ce_loss + self.alpha * reg_loss

        return total_loss

    def get_reg(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization term.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Regularization loss
        """
        batch_size, num_classes = inputs.shape

        # Get predicted class (max confidence)
        pred_class = inputs.argmax(dim=-1)  # [batch_size]

        # Create masks for positive and negative samples
        pos_mask = pred_class == targets  # [batch_size]
        neg_mask = ~pos_mask  # [batch_size]

        reg_loss = 0.0

        if pos_mask.any():
            # Positive samples: correctly predicted
            pos_inputs = inputs[pos_mask]  # [num_pos, num_classes]
            pos_targets = targets[pos_mask]  # [num_pos]

            # Distance calculation for positive samples
            pos_logits = pos_inputs.gather(1, pos_targets.unsqueeze(1)).squeeze(1)  # [num_pos]
            max_logits = pos_inputs.max(dim=-1)[0]  # [num_pos]

            pos_distances = F.relu(max_logits - pos_logits - self.margin)
            pos_reg = pos_distances.mean()

            reg_loss += self.pos_lambda * pos_reg

        if neg_mask.any():
            # Negative samples: incorrectly predicted
            neg_inputs = inputs[neg_mask]  # [num_neg, num_classes]
            neg_targets = targets[neg_mask]  # [num_neg]

            # Distance calculation for negative samples
            neg_logits = neg_inputs.gather(1, neg_targets.unsqueeze(1)).squeeze(1)  # [num_neg]
            max_logits = neg_inputs.max(dim=-1)[0]  # [num_neg]

            neg_distances = F.relu(max_logits - neg_logits - self.margin)
            neg_reg = neg_distances.mean()

            reg_loss += self.neg_lambda * neg_reg

        return reg_loss


class ACLSBinary(nn.Module):
    """ACLS for Binary Classification."""

    def __init__(
        self,
        pos_lambda: float = 1.0,
        neg_lambda: float = 0.1,
        alpha: float = 0.1,
        margin: float = 10.0,
        num_classes: int = 2,
    ):
        """
        Initialize Binary ACLS loss.

        Args:
            pos_lambda: Positive sample regularization weight
            neg_lambda: Negative sample regularization weight
            alpha: Regularization term weight
            margin: Margin for distance calculation
            num_classes: Number of classes (always 2 for binary)
        """
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = 2

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [batch_size] or [batch_size, 1]
            targets: Target labels [batch_size]

        Returns:
            Binary ACLS loss
        """
        # Ensure inputs are 1D
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)

        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())

        # Convert to binary logits for regularization
        # Create [batch_size, 2] logits: [neg_logit, pos_logit]
        binary_logits = torch.stack([-inputs, inputs], dim=-1)

        # Regularization term using ACLS approach
        reg_loss = self.get_reg_binary(binary_logits, targets.long())

        total_loss = bce_loss + self.alpha * reg_loss

        return total_loss

    def get_reg_binary(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute binary regularization term.

        Args:
            inputs: Binary logits [batch_size, 2]
            targets: Target labels [batch_size] (0 or 1)

        Returns:
            Binary regularization loss
        """
        batch_size = inputs.size(0)

        # Get predicted class
        pred_class = inputs.argmax(dim=-1)  # [batch_size]

        # Create masks
        pos_mask = pred_class == targets
        neg_mask = ~pos_mask

        reg_loss = 0.0

        if pos_mask.any():
            # Positive samples
            pos_inputs = inputs[pos_mask]  # [num_pos, 2]
            pos_targets = targets[pos_mask]  # [num_pos]

            pos_logits = pos_inputs.gather(1, pos_targets.unsqueeze(1)).squeeze(1)
            max_logits = pos_inputs.max(dim=-1)[0]

            pos_distances = F.relu(max_logits - pos_logits - self.margin)
            pos_reg = pos_distances.mean()

            reg_loss += self.pos_lambda * pos_reg

        if neg_mask.any():
            # Negative samples
            neg_inputs = inputs[neg_mask]  # [num_neg, 2]
            neg_targets = targets[neg_mask]  # [num_neg]

            neg_logits = neg_inputs.gather(1, neg_targets.unsqueeze(1)).squeeze(1)
            max_logits = neg_inputs.max(dim=-1)[0]

            neg_distances = F.relu(max_logits - neg_logits - self.margin)
            neg_reg = neg_distances.mean()

            reg_loss += self.neg_lambda * neg_reg

        return reg_loss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup criterion for combining two targets with lambda weighting.

    Compatible with jiazhuang notebook implementation.

    Args:
        criterion: Loss function to apply
        pred: Model predictions [batch_size, ...]
        y_a: First set of targets [batch_size]
        y_b: Second set of targets [batch_size]
        lam: Mixing parameter [0, 1]

    Returns:
        Mixed loss: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MulticlassSoftF1Loss(nn.Module):
    """マルチクラス分類用のSoftF1Loss実装（Macro F1ベース）."""

    def __init__(self, num_classes: int, beta: float = 1.0, eps: float = 1e-6):
        """
        初期化.

        Args:
            num_classes: クラス数
            beta: F-beta scoreのbetaパラメータ
            eps: 数値安定性のためのepsilon
        """
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            inputs: 予測ロジット [batch, num_classes]
            targets: ターゲットラベル [batch] (クラスID)

        Returns:
            SoftF1Loss (1 - Macro F1)
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=-1)

        # one-hotエンコーディング
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        f1_scores = []
        for class_idx in range(self.num_classes):
            # クラスごとの予測確率とターゲット
            class_probs = probs[:, class_idx]
            class_targets = targets_onehot[:, class_idx]

            # True Positives, False Positives, False Negatives
            tp = (class_probs * class_targets).sum()
            fp = (class_probs * (1 - class_targets)).sum()
            fn = ((1 - class_probs) * class_targets).sum()

            # Precision, Recall
            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)

            # F-beta score
            f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.eps)
            f1_scores.append(f_beta)

        # Macro F1 (全クラスの平均)
        macro_f1 = torch.stack(f1_scores).mean()

        return 1.0 - macro_f1


class MixupLoss(nn.Module):
    """
    Mixup-compatible loss wrapper for jiazhuang notebook compatibility.

    Automatically handles both regular and mixup training.
    """

    def __init__(self, base_criterion):
        """
        Initialize mixup loss wrapper.

        Args:
            base_criterion: Base loss function (e.g., CrossEntropyLoss, FocalLoss)
        """
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, pred, target, mixup_target=None, mixup_lam=None):
        """
        Forward pass supporting both regular and mixup training.

        Args:
            pred: Model predictions [batch_size, num_classes]
            target: Primary targets [batch_size]
            mixup_target: Secondary targets for mixup (optional) [batch_size]
            mixup_lam: Mixup lambda parameter (optional)

        Returns:
            Loss value
        """
        if mixup_target is not None and mixup_lam is not None:
            # Mixup mode
            return mixup_criterion(self.base_criterion, pred, target, mixup_target, mixup_lam)
        else:
            # Regular mode
            return self.base_criterion(pred, target)


if __name__ == "__main__":
    """Test the loss functions."""
    # Test parameters
    batch_size = 4
    num_classes = 5

    # Create test data
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    binary_inputs = torch.randn(batch_size)
    binary_targets = torch.randint(0, 2, (batch_size,))

    print("Testing loss functions...")

    # Test LabelSmoothingCrossEntropy
    ls_loss = LabelSmoothingCrossEntropy(alpha=0.1)
    ls_output = ls_loss(inputs, targets)
    print(f"LabelSmoothingCrossEntropy: {ls_output.item():.4f}")

    # Test LabelSmoothingBCE
    ls_bce = LabelSmoothingBCE(alpha=0.1)
    ls_bce_output = ls_bce(binary_inputs, binary_targets.float())
    print(f"LabelSmoothingBCE: {ls_bce_output.item():.4f}")

    # Test MbLS
    mbls = MbLS(margin=10.0, alpha=0.1)
    mbls_output = mbls(inputs, targets)
    print(f"MbLS: {mbls_output.item():.4f}")

    # Test MbLSBinary
    mbls_bin = MbLSBinary(margin=10.0, alpha=0.1)
    mbls_bin_output = mbls_bin(binary_inputs, binary_targets.float())
    print(f"MbLSBinary: {mbls_bin_output.item():.4f}")

    # Test ACLS
    acls = ACLS(pos_lambda=1.0, neg_lambda=0.1, alpha=0.1, margin=10.0, num_classes=num_classes)
    acls_output = acls(inputs, targets)
    print(f"ACLS: {acls_output.item():.4f}")

    # Test ACLSBinary
    acls_bin = ACLSBinary(pos_lambda=1.0, neg_lambda=0.1, alpha=0.1, margin=10.0)
    acls_bin_output = acls_bin(binary_inputs, binary_targets.float())
    print(f"ACLSBinary: {acls_bin_output.item():.4f}")

    # Test Mixup functions
    print("\nTesting Mixup functions...")
    base_criterion = torch.nn.CrossEntropyLoss()

    # Test mixup_criterion function
    targets_a = torch.randint(0, num_classes, (batch_size,))
    targets_b = torch.randint(0, num_classes, (batch_size,))
    lam = 0.7

    mixup_loss_val = mixup_criterion(base_criterion, inputs, targets_a, targets_b, lam)
    print(f"mixup_criterion: {mixup_loss_val.item():.4f}")

    # Test MixupLoss wrapper
    mixup_wrapper = MixupLoss(base_criterion)

    # Regular mode
    regular_loss = mixup_wrapper(inputs, targets)
    print(f"MixupLoss (regular): {regular_loss.item():.4f}")

    # Mixup mode
    mixup_loss = mixup_wrapper(inputs, targets_a, targets_b, lam)
    print(f"MixupLoss (mixup): {mixup_loss.item():.4f}")

    print("All tests completed successfully!")
