"""IMU-only LSTM with ResidualSE-CNN and BiGRU attention - exp028用."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# ACLS losses import
from losses import ACLS, LabelSmoothingCrossEntropy, MixupLoss
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures):
    """
    CMI評価指標の計算（公式実装版）.
    Binary F1 (Target vs Non-Target) と Macro F1 (個別ジェスチャー vs 'non_target') の平均.

    Args:
        gesture_true: ジェスチャー名の真値リスト
        gesture_pred: ジェスチャー名の予測リスト
        target_gestures: ターゲットジェスチャーのリスト
        non_target_gestures: ノンターゲットジェスチャーのリスト

    Returns:
        CMIスコア
    """

    # Binary F1 (Target vs Non-Target)
    y_true_bin = np.array([1 if g in target_gestures else 0 for g in gesture_true])
    y_pred_bin = np.array([1 if g in target_gestures else 0 for g in gesture_pred])
    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0, average="binary")

    # Multiclass F1 (個別ジェスチャー vs 'non_target')
    y_true_mc = [g if g in target_gestures else "non_target" for g in gesture_true]
    y_pred_mc = [g if g in target_gestures else "non_target" for g in gesture_pred]
    multiclass_f1 = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)

    # CMIスコア（平均）
    cmi_score = 0.5 * binary_f1 + 0.5 * multiclass_f1

    return cmi_score


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    """Residual CNN block with Squeeze-and-Excitation attention."""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = self.shortcut(x)

        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))

        # SE block
        out = self.se(out)

        # Add shortcut
        out += shortcut
        out = F.relu(out)

        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)

        return out


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence modeling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context


class IMUOnlyLSTM(nn.Module):
    """IMU-only LSTM model with ResidualSE-CNN and BiGRU attention (based on jiazhuang notebook)."""

    def __init__(self, imu_dim=20, n_classes=18, weight_decay=1e-4):
        super().__init__()
        self.imu_dim = imu_dim
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch with ResidualSE-CNN blocks
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=0.3, weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=0.3, weight_decay=weight_decay)

        # BiGRU
        self.bigru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)

        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional

        # Dense layers
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, imu_dim) -> (batch, imu_dim, seq_len)
        imu = x.transpose(1, 2)

        # IMU branch with ResidualSE-CNN
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)

        # Transpose back for BiGRU: (batch, seq_len, hidden_dim)
        merged = x1.transpose(1, 2)

        # BiGRU
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)

        # Attention
        attended = self.attention(gru_out)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        logits = self.classifier(x)
        return logits


class CMISqueezeformer(pl.LightningModule):
    """IMU-only LSTM model for CMI competition (exp028 - jiazhuang baseline)."""

    def __init__(
        self,
        input_dim: int = 20,  # IMU: 20次元（物理特徴量含む、jiazhuang compatible）
        num_classes: int = 18,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        scheduler_config: dict | None = None,
        loss_config: dict | None = None,
        acls_config: dict | None = None,
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
        id_to_gesture: dict[int, str] | None = None,
        **kwargs,  # 他の引数は無視（後方互換性のため）
    ):
        """
        初期化.

        Args:
            input_dim: 入力次元数（IMU物理特徴量: 19次元）
            num_classes: クラス数
            learning_rate: 学習率
            weight_decay: 重み減衰
            scheduler_config: スケジューラ設定
            loss_config: 損失関数設定
            target_gestures: ターゲットジェスチャーのリスト
            non_target_gestures: ノンターゲットジェスチャーのリスト
            id_to_gesture: ID→ジェスチャー名のマッピング
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.loss_config = loss_config or {}
        self.acls_config = acls_config or {}

        # ジェスチャー関連の設定
        self.target_gestures = target_gestures or [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        ]
        self.non_target_gestures = non_target_gestures or [
            "Drink from bottle/cup",
            "Glasses on/off",
            "Pull air toward your face",
            "Pinch knee/leg skin",
            "Scratch knee/leg skin",
            "Write name on leg",
            "Text on phone",
            "Feel around in tray and pull out an object",
            "Write name in air",
            "Wave hello",
        ]
        self.id_to_gesture = id_to_gesture

        # IMU-only LSTMモデル
        self.model = IMUOnlyLSTM(imu_dim=input_dim, n_classes=num_classes, weight_decay=weight_decay)

        # 損失関数の設定
        self._setup_loss_functions()

        # メトリクス保存用
        self.validation_outputs = []

    def _setup_loss_functions(self):
        """
        損失関数の設定（exp025参考、IMU-only LSTM対応版）.

        支援する損失関数タイプ:
        - "focal": Focal Loss（デフォルト、jiazhuangノートブック推奨）
        - "cross_entropy": 基本クロスエントロピー
        - "label_smoothing": Label Smoothing Cross-Entropy
        - "soft_f1": SoftF1Loss（マクロF1最適化）
        - "mixup": Mixup対応損失（任意のベース損失にラップ）
        """
        loss_type = self.loss_config.get("type", "focal")

        # ベース損失関数の作成
        if loss_type == "focal":
            # Focal Loss（jiazhuangノートブック推奨、不均衡データ対応）
            base_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )

        elif loss_type == "cross_entropy":
            # 基本クロスエントロピー
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        elif loss_type == "label_smoothing":
            # Label Smoothing Cross-Entropy（カスタム実装）
            alpha = self.loss_config.get("label_smoothing", 0.1)
            base_criterion = LabelSmoothingCrossEntropy(alpha=alpha)

        elif loss_type == "soft_f1":
            # SoftF1Loss（マクロF1最適化、CMI評価指標に適した損失）
            beta = self.loss_config.get("soft_f1_beta", 1.0)
            eps = self.loss_config.get("soft_f1_eps", 1e-6)
            base_criterion = MulticlassSoftF1Loss(num_classes=self.num_classes, beta=beta, eps=eps)
        elif loss_type == "acls":
            base_criterion = ACLS(
                pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
                neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
                alpha=self.acls_config.get("acls_alpha", 0.1),
                margin=self.acls_config.get("acls_margin", 10.0),
                num_classes=self.num_classes,
            )

        else:
            # デフォルト：基本的なクロスエントロピー
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Mixup対応の確認
        use_mixup = self.loss_config.get("use_mixup", False)
        if use_mixup or loss_type == "mixup":
            # Mixup対応損失ラッパーを適用

            self.criterion = MixupLoss(base_criterion)
            self.supports_mixup = True
        else:
            # 通常損失
            self.criterion = base_criterion
            self.supports_mixup = False

        # 損失関数情報をログ出力
        print(f"Loss function setup: {loss_type}, mixup_support: {self.supports_mixup}")

        # 追加設定（将来の拡張用）
        self.loss_weight = self.loss_config.get("loss_weight", 1.0)
        self.grad_clip_enabled = self.loss_config.get("gradient_clipping", True)

    def forward(
        self,
        imu: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        demographics: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        前向き計算.

        Args:
            imu: IMUデータ [batch, input_dim, seq_len] or [batch, seq_len, input_dim]
            attention_mask: アテンションマスク（無視される）
            demographics: Demographics特徴量（無視される）

        Returns:
            logits: 予測 [batch, num_classes]
        """
        # 入力形状の調整
        if imu.dim() == 3:
            if imu.size(1) == self.input_dim:
                # [batch, input_dim, seq_len] -> [batch, seq_len, input_dim]
                imu = imu.transpose(1, 2)
            # else: already [batch, seq_len, input_dim]

        return self.model(imu)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        訓練ステップ（Mixup対応）.

        Args:
            batch: バッチデータ（Mixup使用時は追加フィールドあり）
                - "imu": IMUデータ [batch, seq_len, features]
                - "multiclass_label": 主要ラベル [batch]
                - "mixup_target": Mixup用ラベル [batch] (オプショナル)
                - "mixup_lam": Mixupパラメータ [batch] (オプショナル)
            batch_idx: バッチインデックス

        Returns:
            損失値
        """
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]

        # 前向き計算
        logits = self(imu)

        # Mixup対応損失計算
        if self.supports_mixup and "mixup_target" in batch and "mixup_lam" in batch:
            # Mixupモード: 追加パラメータを使用
            mixup_target = batch["mixup_target"]
            mixup_lam = batch["mixup_lam"]

            # MixupLossのforward呼び出し
            loss = self.criterion(pred=logits, target=multiclass_labels, mixup_target=mixup_target, mixup_lam=mixup_lam)

            # Mixup使用をログ
            self.log("train_mixup_used", 1.0, prog_bar=False)

        else:
            # 通常モード: 基本損失計算
            loss = self.criterion(logits, multiclass_labels)
            self.log("train_mixup_used", 0.0, prog_bar=False)

        # 重み付き損失（設定で重みが指定されている場合）
        if hasattr(self, "loss_weight") and self.loss_weight != 1.0:
            loss = loss * self.loss_weight

        # ログ
        self.log("train_loss", loss, prog_bar=True)

        # デバッグ用：損失値の確認
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """検証ステップ."""
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]
        sequence_ids = batch["sequence_id"]
        gestures = batch["gesture"]

        # 前向き計算
        logits = self(imu)

        # 損失計算
        loss = self.criterion(logits, multiclass_labels)

        # 予測値の計算（メトリクス計算のため）
        probs = F.softmax(logits, dim=-1)

        # 結果を保存
        output = {
            "val_loss": loss,
            "probs": probs.cpu(),
            "labels": multiclass_labels.cpu(),
            "sequence_ids": sequence_ids,
            "gestures": gestures,
        }

        self.validation_outputs.append(output)

        # ログ
        self.log("val_loss", loss, prog_bar=True)

        return output

    def on_validation_epoch_end(self):
        """検証エポック終了時の処理."""
        if not self.validation_outputs:
            return

        # すべての出力を結合
        all_probs = torch.cat([x["probs"] for x in self.validation_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_outputs])

        # ジェスチャー名のリストを結合
        all_gestures = []
        for x in self.validation_outputs:
            all_gestures.extend(x["gestures"])

        # CMIスコアの計算
        try:
            # マルチクラス予測
            preds = torch.argmax(all_probs, dim=-1)

            # ID→ジェスチャー名の変換が利用可能な場合
            if self.id_to_gesture:
                # 予測IDをジェスチャー名に変換
                gesture_preds = [self.id_to_gesture[idx.item()] for idx in preds]

                # CMIスコア計算（ジェスチャー名を使用）
                cmi_score = compute_cmi_score(
                    all_gestures, gesture_preds, self.target_gestures, self.non_target_gestures
                )
            else:
                # 従来の方法（後方互換性のため）
                # 簡易的にaccuracyを使用
                correct = (preds == all_labels).sum().item()
                total = len(all_labels)
                cmi_score = correct / total

            self.log("val_cmi_score", cmi_score, prog_bar=True)

        except Exception as e:
            print(f"CMI score calculation failed: {e}")

        # 出力をクリア
        self.validation_outputs.clear()

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler_type = self.scheduler_config.get("type")

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.scheduler_config.get("min_lr", 1e-6)
            )
            return [optimizer], [scheduler]

        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 5),
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_cmi_score",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            return optimizer


class FocalLoss(nn.Module):
    """Focal Loss実装."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, label_smoothing: float = 0.0):
        """
        初期化.

        Args:
            gamma: Focusing parameter
            alpha: Class balancing parameter
            label_smoothing: Label smoothing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            inputs: 予測ロジット [batch, num_classes]
            targets: ターゲットラベル [batch]

        Returns:
            Focal Loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class BinarySoftF1Loss(nn.Module):
    """バイナリ分類用のSoftF1Loss実装."""

    def __init__(self, beta: float = 1.0, eps: float = 1e-6):
        """
        初期化.

        Args:
            beta: F-beta scoreのbetaパラメータ
            eps: 数値安定性のためのepsilon
        """
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            probs: 予測確率 [batch] (sigmoid後の値)
            targets: ターゲットラベル [batch] (0 or 1)

        Returns:
            SoftF1Loss (1 - F1)
        """
        # True Positives, False Positives, False Negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        # Precision, Recall
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        # F-beta score
        f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.eps)

        return 1.0 - f_beta


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

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            probs: 予測確率 [batch, num_classes] (softmax後の値)
            targets: ターゲットラベル [batch] (クラスID)

        Returns:
            SoftF1Loss (1 - Macro F1)
        """
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


if __name__ == "__main__":
    """テスト実行."""
    # IMU-only LSTMモデルのテスト
    model = CMISqueezeformer(input_dim=19, num_classes=18)

    # テストデータ（IMU物理特徴量19次元）
    batch_size = 2
    seq_len = 200
    imu_data = torch.randn(batch_size, seq_len, 19)  # [batch, seq_len, imu_dim]

    # 前向き計算
    logits = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output shape: {logits.shape}")

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # IMUOnlyLSTMクラス単体のテスト
    print("\n--- IMUOnlyLSTM Model Test ---")
    imu_model = IMUOnlyLSTM(imu_dim=19, n_classes=18)
    logits_direct = imu_model(imu_data)
    print(f"IMUOnlyLSTM output shape: {logits_direct.shape}")

    imu_params = sum(p.numel() for p in imu_model.parameters())
    print(f"IMUOnlyLSTM parameters: {imu_params:,}")
