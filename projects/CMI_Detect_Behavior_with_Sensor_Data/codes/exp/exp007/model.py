"""IMU特化Squeezeformerモデル - exp007用（欠損値をattention_maskで処理）."""


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Manual EMA implementation (avoiding recursion issues with ema-pytorch)
EMA_AVAILABLE = True  # Manual implementation is always available

# Schedule Free optimizers
try:
    from schedulefree import AdamWScheduleFree, RAdamScheduleFree, SGDScheduleFree

    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False


def compute_cmi_score(binary_true, binary_pred, multiclass_true, multiclass_pred):
    """
    CMI評価指標の計算（Binary F1 + Macro F1）/ 2.

    Args:
        binary_true: バイナリ真値
        binary_pred: バイナリ予測
        multiclass_true: マルチクラス真値
        multiclass_pred: マルチクラス予測

    Returns:
        CMIスコア
    """
    # バイナリF1スコア
    binary_f1 = f1_score(binary_true, binary_pred, average="binary")

    # マルチクラスMacro F1スコア
    multiclass_f1 = f1_score(multiclass_true, multiclass_pred, average="macro")

    # CMIスコア（平均）
    cmi_score = (binary_f1 + multiclass_f1) / 2.0

    return cmi_score


class PositionalEncoding(nn.Module):
    """位置エンコーディング."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初期化.

        Args:
            d_model: モデル次元
            dropout: ドロップアウト率
            max_len: 最大シーケンス長
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            x: 入力テンソル [seq_len, batch, d_model]

        Returns:
            位置エンコーディング済みテンソル
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConvolutionModule(nn.Module):
    """Squeezeformer用のConvolutionモジュール."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        """
        初期化.

        Args:
            d_model: モデル次元
            kernel_size: カーネルサイズ
            dropout: ドロップアウト率
        """
        super().__init__()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)

        # GLU activation
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=d_model
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Swish activation
        self.swish = nn.SiLU()

        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            x: 入力テンソル [batch, seq_len, d_model]

        Returns:
            出力テンソル [batch, seq_len, d_model]
        """
        residual = x

        # Layer norm
        x = self.layer_norm(x)

        # [batch, seq_len, d_model] -> [batch, d_model, seq_len]
        x = x.transpose(1, 2)

        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Depthwise conv + batch norm + swish
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)

        # Pointwise conv + dropout
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # [batch, d_model, seq_len] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        # Residual connection
        return x + residual


class FeedForwardModule(nn.Module):
    """Feed-Forwardモジュール."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初期化.

        Args:
            d_model: モデル次元
            d_ff: フィードフォワード次元
            dropout: ドロップアウト率
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            x: 入力テンソル [batch, seq_len, d_model]

        Returns:
            出力テンソル [batch, seq_len, d_model]
        """
        residual = x

        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x + residual


class SqueezeformerBlock(nn.Module):
    """Squeezeformerブロック."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int = 31, dropout: float = 0.1):
        """
        初期化.

        Args:
            d_model: モデル次元
            n_heads: アテンションヘッド数
            d_ff: フィードフォワード次元
            kernel_size: Convolutionカーネルサイズ
            dropout: ドロップアウト率
        """
        super().__init__()

        # Feed-forward module 1
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)

        # Feed-forward module 2
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向き計算.

        Args:
            x: 入力テンソル [batch, seq_len, d_model]
            attention_mask: アテンションマスク

        Returns:
            出力テンソル [batch, seq_len, d_model]
        """
        # Feed-forward 1
        x = self.ff1(x)

        # Multi-head self-attention
        residual = x
        x = self.self_attn_norm(x)

        # attention_maskをkey_padding_maskに変換
        # attention_mask: [batch, seq_len], True=有効, False=無効
        # key_padding_mask: [batch, seq_len], True=無効, False=有効 (逆)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # 論理反転

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.self_attn_dropout(attn_output) + residual

        # Convolution module
        x = self.conv_module(x)

        # Feed-forward 2
        x = self.ff2(x)

        return x


class CMISqueezeformer(pl.LightningModule):
    """CMI用のSqueezeformerモデル（IMU特化版）."""

    def __init__(
        self,
        input_dim: int = 7,  # IMU: 7次元
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_classes: int = 18,
        kernel_size: int = 31,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        scheduler_config: dict | None = None,
        loss_config: dict | None = None,
        schedule_free_config: dict | None = None,
        ema_config: dict | None = None,
    ):
        """
        初期化.

        Args:
            input_dim: 入力次元数（IMU: 7次元）
            d_model: モデル次元
            n_layers: レイヤー数
            n_heads: アテンションヘッド数
            d_ff: フィードフォワード次元
            num_classes: クラス数
            kernel_size: Convolutionカーネルサイズ
            dropout: ドロップアウト率
            learning_rate: 学習率
            weight_decay: 重み減衰
            scheduler_config: スケジューラ設定
            loss_config: 損失関数設定
            schedule_free_config: Schedule Free オプティマイザ設定
            ema_config: EMA（Exponential Moving Average）設定
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.loss_config = loss_config or {}
        self.schedule_free_config = schedule_free_config or {}
        self.ema_config = ema_config or {}

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Squeezeformer blocks
        self.blocks = nn.ModuleList(
            [SqueezeformerBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)]
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification heads
        self.multiclass_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # 損失関数の設定
        self._setup_loss_functions()

        # 手動EMAの初期化フラグ（自己参照問題を回避）
        self._ema_initialized = False

        # メトリクス保存用
        self.validation_outputs = []

    def _setup_loss_functions(self):
        """損失関数の設定."""
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "cmi":
            # 基本的なクロスエントロピー損失
            self.multiclass_criterion = nn.CrossEntropyLoss()
            self.binary_criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "cmi_focal":
            # Focal Loss
            self.multiclass_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0), alpha=self.loss_config.get("focal_alpha", 1.0)
            )
            self.binary_criterion = nn.BCEWithLogitsLoss()
        else:
            # デフォルト
            self.multiclass_criterion = nn.CrossEntropyLoss()
            self.binary_criterion = nn.BCEWithLogitsLoss()

        self.loss_alpha = self.loss_config.get("alpha", 0.5)

    def setup(self, stage: str) -> None:
        """PyTorch Lightning setup hook - EMAの初期化はここで行う."""
        if not self._ema_initialized and self.ema_config.get("enabled", False):
            # 手動EMA実装を使用（自己参照問題を回避）
            self._setup_manual_ema()
            self._ema_initialized = True

    def _setup_manual_ema(self):
        """手動EMA実装の初期化."""
        self.ema_decay = self.ema_config.get("beta", 0.9999)
        self.ema_update_after_step = self.ema_config.get("update_after_step", 1000)
        self.ema_update_every = self.ema_config.get("update_every", 10)
        self.ema_switch_every = self.ema_config.get("update_model_with_ema_every", 1000)

        # EMAパラメータの保存用辞書（デバイスを考慮）
        self.ema_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                # モデルと同じデバイスにEMAパラメータを配置
                self.ema_params[name] = param.data.clone().to(param.device)

        self.ema_step_count = 0
        print(f"Manual EMA initialized with decay={self.ema_decay}, device={next(self.parameters()).device}")

    def _update_ema_params(self):
        """EMAパラメータの更新."""
        if not hasattr(self, "ema_params") or not self.ema_config.get("enabled", False):
            return

        self.ema_step_count += 1

        # update_after_stepより後で更新開始
        if self.ema_step_count <= self.ema_update_after_step:
            return

        # update_everyステップごとに更新
        if self.ema_step_count % self.ema_update_every != 0:
            return

        # EMAパラメータの更新（デバイス不一致を回避）
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    # EMAパラメータとモデルパラメータのデバイスが異なる場合は移動
                    if self.ema_params[name].device != param.device:
                        self.ema_params[name] = self.ema_params[name].to(param.device)

                    self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _apply_ema_to_model(self):
        """EMAパラメータをモデルに適用（Switch EMA）."""
        if not hasattr(self, "ema_params") or not self.ema_config.get("enabled", False):
            return

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    # EMAパラメータとモデルパラメータのデバイスが異なる場合は移動
                    if self.ema_params[name].device != param.device:
                        self.ema_params[name] = self.ema_params[name].to(param.device)

                    param.data.copy_(self.ema_params[name])

    def forward(
        self, imu: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向き計算.

        Args:
            imu: IMUデータ [batch, input_dim, seq_len]
            attention_mask: アテンションマスク

        Returns:
            multiclass_logits: マルチクラス予測 [batch, num_classes]
            binary_logits: バイナリ予測 [batch, 1]
        """
        # [batch, input_dim, seq_len] -> [batch, seq_len, input_dim]
        x = imu.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]

        # Squeezeformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Masked Global pooling
        if attention_mask is not None:
            # マスクを考慮したpooling
            # attention_mask: [batch, seq_len], True=有効, False=無効
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)  # [batch, seq_len, d_model]
            masked_x = x * mask_expanded.float()  # 無効部分を0にマスク

            # 各サンプルの有効長を計算
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            seq_lengths = torch.clamp(seq_lengths, min=1.0)  # ゼロ除算防止

            # マスクされた平均プーリング
            x = masked_x.sum(dim=1) / seq_lengths  # [batch, d_model]
        else:
            # 従来の平均プーリング
            x = x.mean(dim=1)  # [batch, d_model]

        # Classification heads
        multiclass_logits = self.multiclass_head(x)
        binary_logits = self.binary_head(x)

        return multiclass_logits, binary_logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """訓練ステップ."""
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]
        binary_labels = batch["binary_label"]
        attention_mask = batch.get("attention_mask", None)  # 動的パディング時のみ存在

        # 前向き計算
        multiclass_logits, binary_logits = self(imu, attention_mask)

        # 損失計算
        multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
        binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)

        total_loss = self.loss_alpha * multiclass_loss + (1 - self.loss_alpha) * binary_loss

        # ログ
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_multiclass_loss", multiclass_loss)
        self.log("train_binary_loss", binary_loss)

        # 手動EMA更新
        self._update_ema_params()

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """検証ステップ."""
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]
        binary_labels = batch["binary_label"]
        sequence_ids = batch["sequence_id"]
        gestures = batch["gesture"]
        attention_mask = batch.get("attention_mask", None)  # 動的パディング時のみ存在

        # 前向き計算
        multiclass_logits, binary_logits = self(imu, attention_mask)

        # 損失計算
        multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
        binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)

        total_loss = self.loss_alpha * multiclass_loss + (1 - self.loss_alpha) * binary_loss

        # 予測値の計算
        multiclass_probs = F.softmax(multiclass_logits, dim=-1)
        binary_probs = torch.sigmoid(binary_logits.squeeze(-1))

        # 結果を保存
        output = {
            "val_loss": total_loss,
            "val_multiclass_loss": multiclass_loss,
            "val_binary_loss": binary_loss,
            "multiclass_probs": multiclass_probs.cpu(),
            "binary_probs": binary_probs.cpu(),
            "multiclass_labels": multiclass_labels.cpu(),
            "binary_labels": binary_labels.cpu(),
            "sequence_ids": sequence_ids,
            "gestures": gestures,
        }

        self.validation_outputs.append(output)

        # ログ
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_multiclass_loss", multiclass_loss)
        self.log("val_binary_loss", binary_loss)

        return output

    def on_train_epoch_end(self):
        """訓練エポック終了時の処理（Switch EMA）."""
        if hasattr(self, "ema_params") and self.ema_config.get("enabled", False):
            update_freq = self.ema_config.get("update_model_with_ema_every", 1000)
            if self.global_step > 0 and self.global_step % update_freq == 0:
                print(f"Applying Switch EMA at step {self.global_step}")
                self._apply_ema_to_model()

    def on_validation_epoch_end(self):
        """検証エポック終了時の処理."""
        if not self.validation_outputs:
            return

        # すべての出力を結合
        all_multiclass_probs = torch.cat([x["multiclass_probs"] for x in self.validation_outputs])
        all_binary_probs = torch.cat([x["binary_probs"] for x in self.validation_outputs])
        all_multiclass_labels = torch.cat([x["multiclass_labels"] for x in self.validation_outputs])
        all_binary_labels = torch.cat([x["binary_labels"] for x in self.validation_outputs])

        # CMIスコアの計算
        try:
            # バイナリ予測
            binary_preds = (all_binary_probs > 0.5).int()

            # マルチクラス予測
            multiclass_preds = torch.argmax(all_multiclass_probs, dim=-1)

            # CMIスコア計算（NumPy配列に変換）
            cmi_score = compute_cmi_score(
                all_binary_labels.numpy(), binary_preds.numpy(), all_multiclass_labels.numpy(), multiclass_preds.numpy()
            )

            self.log("val_cmi_score", cmi_score, prog_bar=True)

        except Exception as e:
            print(f"CMI score calculation failed: {e}")

        # 出力をクリア
        self.validation_outputs.clear()

    def on_validation_epoch_start(self):
        """検証エポック開始時の処理（Schedule Free BatchNorm較正）."""
        if self.schedule_free_config.get("enabled", False):
            self._calibrate_batch_norm()

    def _calibrate_batch_norm(self):
        """BatchNorm統計量の較正（Schedule Free用）."""
        if not SCHEDULEFREE_AVAILABLE:
            return

        calibration_steps = self.schedule_free_config.get("batch_norm_calibration_steps", 50)

        # オプティマイザがSchedule Freeかチェック
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedule_free_optimizers = []
        for opt in optimizers:
            if hasattr(opt, "eval") and hasattr(opt, "train"):
                schedule_free_optimizers.append(opt)

        if not schedule_free_optimizers:
            return

        # モデルを一時的に訓練モードに
        original_training_mode = self.training
        self.train()

        try:
            # Schedule Free optimizerをevalモードに
            for opt in schedule_free_optimizers:
                opt.eval()

            # BatchNorm統計量をリセット
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.reset_running_stats()

            # 較正用の前向きパス実行
            with torch.no_grad():
                for step in range(calibration_steps):
                    # ダミーデータで前向きパス
                    dummy_input = torch.randn(
                        self.schedule_free_config.get("calibration_batch_size", 32),
                        self.input_dim,
                        self.schedule_free_config.get("calibration_seq_len", 200),
                        device=self.device,
                    )
                    _ = self(dummy_input)

                    if step % 10 == 0:
                        print(f"BatchNorm calibration step: {step}/{calibration_steps}")

            # Schedule Free optimizerを元に戻す
            for opt in schedule_free_optimizers:
                opt.train()

        except Exception as e:
            print(f"BatchNorm calibration failed: {e}")
        finally:
            # モデルのモードを元に戻す
            self.train(original_training_mode)

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定."""
        # Schedule Free optimizer使用時
        if self.schedule_free_config.get("enabled", False) and SCHEDULEFREE_AVAILABLE:
            optimizer_type = self.schedule_free_config.get("optimizer_type", "RAdamScheduleFree")
            lr_multiplier = self.schedule_free_config.get("learning_rate_multiplier", 5.0)
            effective_lr = self.learning_rate * lr_multiplier

            # Schedule Free optimizerの選択
            if optimizer_type == "RAdamScheduleFree":
                optimizer = RAdamScheduleFree(
                    self.parameters(),
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            elif optimizer_type == "AdamWScheduleFree":
                optimizer = AdamWScheduleFree(
                    self.parameters(),
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            elif optimizer_type == "SGDScheduleFree":
                optimizer = SGDScheduleFree(
                    self.parameters(),
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            else:
                print(f"Unknown Schedule Free optimizer: {optimizer_type}, fallback to RAdamScheduleFree")
                optimizer = RAdamScheduleFree(
                    self.parameters(),
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )

            print(f"Using {optimizer_type} with learning rate: {effective_lr:.2e}")
            return optimizer

        # 通常のオプティマイザ
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

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        """
        初期化.

        Args:
            gamma: Focusing parameter
            alpha: Class balancing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向き計算.

        Args:
            inputs: 予測ロジット [batch, num_classes]
            targets: ターゲットラベル [batch]

        Returns:
            Focal Loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


if __name__ == "__main__":
    """テスト実行."""
    # モデルのテスト
    model = CMISqueezeformer(input_dim=7, d_model=256, n_layers=4, n_heads=8, d_ff=1024, num_classes=18)

    # テストデータ
    batch_size = 2
    seq_len = 200
    imu_data = torch.randn(batch_size, 7, seq_len)

    # 前向き計算
    multiclass_logits, binary_logits = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Multiclass output shape: {multiclass_logits.shape}")
    print(f"Binary output shape: {binary_logits.shape}")

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
