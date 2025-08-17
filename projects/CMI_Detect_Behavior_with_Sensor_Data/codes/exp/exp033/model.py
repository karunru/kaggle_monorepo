"""IMU-only LSTM with ResidualSE-CNN and BiGRU attention - exp029用 (ヒューマンノーマライゼーション、デモグラフィック、マルチヘッドKL divergence付き)."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# ACLS losses import
from losses import (
    ACLS,
    ACLSBinary,
    BinarySoftF1ACLS,
    LabelSmoothingBCE,
    LabelSmoothingCrossEntropy,
    MbLS,
    MbLSBinary,
    MulticlassSoftF1ACLS,
)
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Schedule Free optimizer imports (optional)
try:
    from schedulefree import AdamWScheduleFree, RAdamScheduleFree, SGDScheduleFree

    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False


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


class DemographicsEmbedding(nn.Module):
    """Demographics特徴量の埋め込み層."""

    def __init__(
        self,
        categorical_features: list[str],
        numerical_features: list[str],
        categorical_embedding_dims: dict[str, int],
        embedding_dim: int = 16,
        dropout: float = 0.1,
        # 設定値ベースのスケーリングパラメータ
        age_min: float = 8.0,
        age_max: float = 60.0,
        height_min: float = 130.0,
        height_max: float = 195.0,
        shoulder_to_wrist_min: float = 35.0,
        shoulder_to_wrist_max: float = 75.0,
        elbow_to_wrist_min: float = 15.0,
        elbow_to_wrist_max: float = 50.0,
    ):
        """
        初期化.

        Args:
            categorical_features: カテゴリカル特徴量のリスト
            numerical_features: 数値特徴量のリスト
            categorical_embedding_dims: カテゴリカル特徴量の埋め込み次元
            embedding_dim: 最終埋め込み次元
            dropout: ドロップアウト率
            age_min: 年齢の最小値（設定値ベース）
            age_max: 年齢の最大値（設定値ベース）
            height_min: 身長の最小値（設定値ベース）
            height_max: 身長の最大値（設定値ベース）
            shoulder_to_wrist_min: 肩-手首距離の最小値（設定値ベース）
            shoulder_to_wrist_max: 肩-手首距離の最大値（設定値ベース）
            elbow_to_wrist_min: 肘-手首距離の最小値（設定値ベース）
            elbow_to_wrist_max: 肘-手首距離の最大値（設定値ベース）
        """
        super().__init__()

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        # カテゴリカル特徴量用の埋め込み層
        self.categorical_embeddings = nn.ModuleDict()
        total_categorical_dim = 0

        for feature in categorical_features:
            # カテゴリ数は2（0/1の二値）と仮定
            num_categories = 2
            embed_dim = categorical_embedding_dims.get(feature, 2)
            self.categorical_embeddings[feature] = nn.Embedding(num_categories, embed_dim)
            total_categorical_dim += embed_dim

        # 数値特徴量の次元
        numerical_dim = len(numerical_features)

        # 結合後の特徴量次元
        combined_dim = total_categorical_dim + numerical_dim

        # 最終埋め込み層（結合された特徴量を所定の次元に変換）
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        # スケーリング用のパラメータ（設定値ベース、推論時の一貫性確保）
        self.register_buffer("age_min", torch.tensor(age_min))
        self.register_buffer("age_max", torch.tensor(age_max))
        self.register_buffer("height_min", torch.tensor(height_min))
        self.register_buffer("height_max", torch.tensor(height_max))
        self.register_buffer("shoulder_to_wrist_min", torch.tensor(shoulder_to_wrist_min))
        self.register_buffer("shoulder_to_wrist_max", torch.tensor(shoulder_to_wrist_max))
        self.register_buffer("elbow_to_wrist_min", torch.tensor(elbow_to_wrist_min))
        self.register_buffer("elbow_to_wrist_max", torch.tensor(elbow_to_wrist_max))

    def _normalize_numerical(self, demographics: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """数値特徴量の正規化とクリッピング."""
        normalized_features = []

        for feature in self.numerical_features:
            values = demographics[feature].float()

            if feature == "age":
                # 範囲外値をクリッピング
                values = torch.clamp(values, self.age_min.item(), self.age_max.item())
                # MinMaxスケーリング
                normalized = (values - self.age_min) / (self.age_max - self.age_min + 1e-8)
            elif feature == "height_cm":
                values = torch.clamp(values, self.height_min.item(), self.height_max.item())
                normalized = (values - self.height_min) / (self.height_max - self.height_min + 1e-8)
            elif feature == "shoulder_to_wrist_cm":
                values = torch.clamp(values, self.shoulder_to_wrist_min.item(), self.shoulder_to_wrist_max.item())
                normalized = (values - self.shoulder_to_wrist_min) / (
                    self.shoulder_to_wrist_max - self.shoulder_to_wrist_min + 1e-8
                )
            elif feature == "elbow_to_wrist_cm":
                values = torch.clamp(values, self.elbow_to_wrist_min.item(), self.elbow_to_wrist_max.item())
                normalized = (values - self.elbow_to_wrist_min) / (
                    self.elbow_to_wrist_max - self.elbow_to_wrist_min + 1e-8
                )
            else:
                # デフォルトは標準化（平均0、標準偏差1）
                normalized = (values - values.mean()) / (values.std() + 1e-8)

            normalized_features.append(normalized.unsqueeze(-1))  # [batch, 1]

        return normalized_features

    def forward(self, demographics: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向き計算.

        Args:
            demographics: Demographics特徴量の辞書
                - カテゴリカル特徴量: [batch] (long型)
                - 数値特徴量: [batch] (float型)

        Returns:
            埋め込みベクトル [batch, embedding_dim]
        """
        batch_size = next(iter(demographics.values())).size(0)

        # カテゴリカル特徴量の埋め込み
        categorical_embeddings = []
        for feature in self.categorical_features:
            if feature in demographics:
                # カテゴリ値を0/1に制限
                cat_values = torch.clamp(demographics[feature].long(), 0, 1)
                embed = self.categorical_embeddings[feature](cat_values)  # [batch, embed_dim]
                categorical_embeddings.append(embed)

        # 数値特徴量の正規化
        numerical_embeddings = self._normalize_numerical(demographics)

        # 全ての特徴量を結合
        all_features = []

        # カテゴリカル埋め込みを結合
        if categorical_embeddings:
            cat_concat = torch.cat(categorical_embeddings, dim=-1)  # [batch, total_cat_dim]
            all_features.append(cat_concat)

        # 数値特徴量を結合
        if numerical_embeddings:
            num_concat = torch.cat(numerical_embeddings, dim=-1)  # [batch, numerical_dim]
            all_features.append(num_concat)

        # 全特徴量を結合
        if all_features:
            combined_features = torch.cat(all_features, dim=-1)  # [batch, combined_dim]
        else:
            # フォールバック: ゼロベクトル
            combined_features = torch.zeros(batch_size, 1, device=next(iter(demographics.values())).device)

        # 最終埋め込みベクトルを生成
        embedding = self.projection(combined_features)  # [batch, embedding_dim]

        return embedding


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class SpectralConv1d(nn.Module):
    """FNO-1D: 時間次元でrFFT→上位modesのみ学習→irFFT"""

    def __init__(self, in_channels, out_channels, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # 複素数重み（実部・虚部）をパラメータ化
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, modes, 2) * 0.02)

    def compl_mul1d(self, x_ft, w):
        # x_ft: [B, C_in, F] 複素, w: [C_in, C_out, modes, 2]
        real = torch.einsum("bif,iof->bof", x_ft.real[:, :, :self.modes], w[..., 0]) \
             - torch.einsum("bif,iof->bof", x_ft.imag[:, :, :self.modes], w[..., 1])
        imag = torch.einsum("bif,iof->bof", x_ft.real[:, :, :self.modes], w[..., 1]) \
             + torch.einsum("bif,iof->bof", x_ft.imag[:, :, :self.modes], w[..., 0])
        out = torch.complex(real, imag)
        return out

    def forward(self, x):  # x: [B, C_in, T]
        batch, channels, length = x.shape
        original_length = length

        # 2の累乗へのパディング（半精度対応）
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            # 次の2の累乗を計算
            padded_length = 2 ** (length - 1).bit_length()
            if padded_length != length:
                # ゼロパディング
                padding = padded_length - length
                x = F.pad(x, (0, padding), mode='constant', value=0)

        # FFT実行
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, C_in, F_r]
        out_ft = torch.zeros(x.size(0), self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft, self.weight)
        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)  # [B, C_out, T]

        # 元のサイズに戻す
        if x_out.size(-1) != original_length:
            x_out = x_out[:, :, :original_length]

        return x_out


class FNOBlock1D(nn.Module):
    """FNOの基本ブロック（残差接続付き）"""

    def __init__(self, channels, modes=16, dropout=0.1):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.w = nn.Conv1d(channels, channels, 1)  # 位置空間の補助線形
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        y = self.spectral(x) + self.w(x)
        y = F.mish(self.norm(y))
        return self.drop(x + y)


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
        out = F.mish(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))

        # SE block
        out = self.se(out)

        # Add shortcut
        out += shortcut
        out = F.mish(out)

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
    """IMU-only LSTM model with ResidualSE-CNN and BiGRU attention (exp029: demographics + multi-head support)."""

    def __init__(
        self,
        imu_dim=20,
        n_classes=18,
        weight_decay=1e-4,
        demographics_dim=0,
        dropout=0.1,
    ):
        super().__init__()
        self.imu_dim = imu_dim
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.demographics_dim = demographics_dim

        # ResidualSECNN ブランチ（exp031ベース）
        self.residual_branch = nn.Sequential(
            ResidualSECNNBlock(imu_dim, 64, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 5, dropout=0.3, weight_decay=weight_decay)
        )

        # FNO ブランチ（exp032ベース）
        self.fno_proj = nn.Sequential(
            nn.Conv1d(imu_dim, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish()
        )
        self.fno_branch = nn.Sequential(
            FNOBlock1D(128, modes=32, dropout=0.2),
            FNOBlock1D(128, modes=32, dropout=0.2),
            FNOBlock1D(128, modes=32, dropout=0.2),
            FNOBlock1D(128, modes=32, dropout=0.2),
            SEBlock(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # 特徴量融合層（128+128=256次元を256次元に統合）
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.2)
        )

        # BiGRU（入力次元を256に変更）
        self.bigru = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)

        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional

        # Dense layers (基本特徴量抽出)
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        # Classification heads (IMU特徴量 + Demographics特徴量を結合)
        classification_input_dim = 128 + self.demographics_dim

        # 18クラス分類ヘッド
        self.multiclass_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # バイナリ分類ヘッド（Target vs Non-Target）
        self.binary_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 9クラス分類ヘッド（8 target + 1 non-target）
        self.nine_class_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 9),
        )

    def forward(self, x, demographics_embedding=None):
        """
        前向き計算.

        Args:
            x: IMU データ [batch, seq_len, imu_dim]
            demographics_embedding: Demographics埋め込み [batch, demographics_dim] (オプション)

        Returns:
            tuple: (multiclass_logits, binary_logits, nine_class_logits)
        """
        # x shape: (batch, seq_len, imu_dim) -> (batch, imu_dim, seq_len)
        imu = x.transpose(1, 2)

        # ResidualSECNN ブランチ処理
        residual_features = self.residual_branch(imu)  # [batch, 128, seq_len//4]

        # FNO ブランチ処理
        fno_input = self.fno_proj(imu)  # [batch, 128, seq_len]
        fno_features = self.fno_branch(fno_input)  # [batch, 128, seq_len//2]

        # 系列長を合わせるため、短い方に合わせる（residual_featuresのseq_len//4）
        target_length = residual_features.size(-1)
        if fno_features.size(-1) != target_length:
            # FNO特徴量をresidual特徴量の長さに合わせてpooling
            fno_features = F.adaptive_avg_pool1d(fno_features, target_length)

        # チャネル次元で結合 [batch, 256, target_length]
        concatenated = torch.cat([residual_features, fno_features], dim=1)

        # 融合層で統合 [batch, 256, target_length]
        merged_features = self.fusion_conv(concatenated)

        # Transpose back for BiGRU: (batch, seq_len, hidden_dim)
        merged = merged_features.transpose(1, 2)

        # BiGRU
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)

        # Attention
        attended = self.attention(gru_out)

        # Dense layers (基本特徴量抽出)
        x = F.mish(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        imu_features = F.mish(self.bn_dense2(self.dense2(x)))
        imu_features = self.drop2(imu_features)

        # Demographics特徴量との結合
        if demographics_embedding is not None and self.demographics_dim > 0:
            # IMU特徴量とDemographics特徴量を結合
            combined_features = torch.cat([imu_features, demographics_embedding], dim=-1)
        else:
            combined_features = imu_features

        # 3つの分類ヘッド
        multiclass_logits = self.multiclass_head(combined_features)
        binary_logits = self.binary_head(combined_features)
        nine_class_logits = self.nine_class_head(combined_features)

        return multiclass_logits, binary_logits, nine_class_logits


class CMISqueezeformer(pl.LightningModule):
    """IMU-only LSTM model for CMI competition (exp029: demographics + multi-head KL divergence)."""

    def __init__(
        self,
        input_dim: int = 20,  # IMU: 20次元（物理特徴量含む、jiazhuang compatible）
        num_classes: int = 18,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        scheduler_config: dict | None = None,
        loss_config: dict | None = None,
        acls_config: dict | None = None,
        schedule_free_config: dict | None = None,
        ema_config: dict | None = None,
        demographics_config: dict | None = None,
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
        id_to_gesture: dict[int, str] | None = None,
        **kwargs,  # 他の引数は無視（後方互換性のため）
    ):
        """
        初期化.

        Args:
            input_dim: 入力次元数（IMU物理特徴量: 20次元）
            num_classes: クラス数
            learning_rate: 学習率
            weight_decay: 重み減衰
            dropout: ドロップアウト率
            scheduler_config: スケジューラ設定
            loss_config: 損失関数設定
            acls_config: ACLS損失関数設定
            schedule_free_config: Schedule Free オプティマイザ設定
            ema_config: EMA設定
            demographics_config: Demographics統合設定
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
        self.schedule_free_config = schedule_free_config or {}
        self.ema_config = ema_config or {}
        self.demographics_config = demographics_config or {}

        # Demographics統合の設定
        self.use_demographics = self.demographics_config.get("enabled", False)
        if self.use_demographics:
            self.demographics_embedding = DemographicsEmbedding(
                categorical_features=self.demographics_config.get(
                    "categorical_features", ["adult_child", "sex", "handedness"]
                ),
                numerical_features=self.demographics_config.get(
                    "numerical_features", ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
                ),
                categorical_embedding_dims=self.demographics_config.get(
                    "categorical_embedding_dims", {"adult_child": 2, "sex": 2, "handedness": 2}
                ),
                embedding_dim=self.demographics_config.get("embedding_dim", 16),
                dropout=dropout,
                # 設定値ベースのスケーリングパラメータを追加
                age_min=self.demographics_config.get("age_min", 8.0),
                age_max=self.demographics_config.get("age_max", 60.0),
                height_min=self.demographics_config.get("height_min", 130.0),
                height_max=self.demographics_config.get("height_max", 195.0),
                shoulder_to_wrist_min=self.demographics_config.get("shoulder_to_wrist_min", 35.0),
                shoulder_to_wrist_max=self.demographics_config.get("shoulder_to_wrist_max", 75.0),
                elbow_to_wrist_min=self.demographics_config.get("elbow_to_wrist_min", 15.0),
                elbow_to_wrist_max=self.demographics_config.get("elbow_to_wrist_max", 50.0),
            )
            self.demographics_dim = self.demographics_config.get("embedding_dim", 16)
        else:
            self.demographics_embedding = None
            self.demographics_dim = 0

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

        # IMU-only LSTMモデル（Demographics統合対応）
        self.model = IMUOnlyLSTM(
            imu_dim=input_dim,
            n_classes=num_classes,
            weight_decay=weight_decay,
            demographics_dim=self.demographics_dim,
            dropout=dropout,
        )

        # KL divergence loss関連の設定
        self.kl_weight = self.loss_config.get("kl_weight", 0.1)
        self.kl_temperature = self.loss_config.get("kl_temperature", 1.0)
        self.nine_class_loss_weight = self.loss_config.get("nine_class_loss_weight", 0.2)
        self.loss_alpha = self.loss_config.get("alpha", 0.5)

        # 損失関数の設定
        self._setup_loss_functions()

        # メトリクス保存用
        self.validation_outputs = []

        # 手動EMAの初期化フラグ（自己参照問題を回避）
        self._ema_initialized = False

    def _setup_loss_functions(self):
        """損失関数の設定."""
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "cmi":
            # 基本的なクロスエントロピー損失
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.multiclass_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.binary_criterion = nn.BCEWithLogitsLoss()
            self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        elif loss_type == "cmi_focal":
            # Focal Loss
            self.multiclass_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )
            self.binary_criterion = nn.BCEWithLogitsLoss()
            # 9クラス用もFocalLoss
            self.nine_class_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )

        elif loss_type == "soft_f1":
            # SoftF1Loss
            self.multiclass_criterion = self._create_soft_f1_criterion(self.num_classes)
            self.binary_criterion = self._create_binary_soft_f1_criterion()
            self.nine_class_criterion = self._create_soft_f1_criterion(9)

        elif loss_type == "soft_f1_acls":
            # SoftF1Loss with ACLS regularization
            self.multiclass_criterion = self._create_soft_f1_acls_criterion(self.num_classes)
            self.binary_criterion = self._create_binary_soft_f1_acls_criterion()
            self.nine_class_criterion = self._create_soft_f1_acls_criterion(9)

        elif loss_type == "acls":
            # ACLS (Adaptive and Conditional Label Smoothing)
            self.multiclass_criterion = self._create_acls_criterion(self.num_classes)
            self.binary_criterion = self._create_acls_binary_criterion()
            self.nine_class_criterion = self._create_acls_criterion(9)

        elif loss_type == "label_smoothing":
            # Label Smoothing Cross-Entropy
            alpha = self.acls_config.get("label_smoothing_alpha", 0.1)
            self.multiclass_criterion = LabelSmoothingCrossEntropy(alpha=alpha)
            self.binary_criterion = LabelSmoothingBCE(alpha=alpha)
            self.nine_class_criterion = LabelSmoothingCrossEntropy(alpha=alpha)

        elif loss_type == "mbls":
            # Margin-based Label Smoothing
            margin = self.acls_config.get("mbls_margin", 10.0)
            alpha = self.acls_config.get("mbls_alpha", 0.1)
            alpha_schedule = self.acls_config.get("mbls_schedule", None)
            self.multiclass_criterion = MbLS(margin=margin, alpha=alpha, alpha_schedule=alpha_schedule)
            self.binary_criterion = MbLSBinary(margin=margin, alpha=alpha)
            self.nine_class_criterion = MbLS(margin=margin, alpha=alpha, alpha_schedule=alpha_schedule)

        else:
            # デフォルト：基本的なクロスエントロピー
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.multiclass_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.binary_criterion = nn.BCEWithLogitsLoss()
            self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 共通設定
        self.loss_alpha = self.loss_config.get("alpha", 0.5)
        self.nine_class_loss_weight = self.loss_config.get("nine_class_loss_weight", 0.2)

        # KL divergence loss設定
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.kl_weight = self.loss_config.get("kl_weight", 0.1)
        self.kl_temperature = self.loss_config.get("kl_temperature", 1.0)

        # 学習可能重み付けの設定
        self._setup_learnable_weights()

    def _create_soft_f1_criterion(self, num_classes: int) -> "MulticlassSoftF1Loss":
        """SoftF1Loss criterion作成の共通化."""
        beta = self.loss_config.get("soft_f1_beta", 1.0)
        eps = self.loss_config.get("soft_f1_eps", 1e-6)
        return MulticlassSoftF1Loss(num_classes=num_classes, beta=beta, eps=eps)

    def _create_binary_soft_f1_criterion(self) -> "BinarySoftF1Loss":
        """Binary SoftF1Loss criterion作成の共通化."""
        beta = self.loss_config.get("soft_f1_beta", 1.0)
        eps = self.loss_config.get("soft_f1_eps", 1e-6)
        return BinarySoftF1Loss(beta=beta, eps=eps)

    def _create_acls_criterion(self, num_classes: int) -> "ACLS":
        """ACLS criterion作成の共通化."""
        return ACLS(
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
            num_classes=num_classes,
        )

    def _create_acls_binary_criterion(self) -> "ACLSBinary":
        """ACLS Binary criterion作成の共通化."""
        return ACLSBinary(
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
            num_classes=2,
        )

    def _create_soft_f1_acls_criterion(self, num_classes: int) -> "MulticlassSoftF1ACLS":
        """SoftF1ACLS criterion作成の共通化."""
        return MulticlassSoftF1ACLS(
            num_classes=num_classes,
            beta=self.loss_config.get("soft_f1_beta", 1.0),
            eps=self.loss_config.get("soft_f1_eps", 1e-6),
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
        )

    def _create_binary_soft_f1_acls_criterion(self) -> "BinarySoftF1ACLS":
        """Binary SoftF1ACLS criterion作成の共通化."""
        return BinarySoftF1ACLS(
            beta=self.loss_config.get("soft_f1_beta", 1.0),
            eps=self.loss_config.get("soft_f1_eps", 1e-6),
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
        )

    def forward(
        self,
        imu: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        demographics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き計算.

        Args:
            imu: IMUデータ [batch, input_dim, seq_len] or [batch, seq_len, input_dim]
            attention_mask: アテンションマスク（無視される）
            demographics: Demographics特徴量（オプショナル）

        Returns:
            tuple: (multiclass_logits, binary_logits, nine_class_logits)
        """
        # 入力形状の調整
        if imu.dim() == 3:
            if imu.size(1) == self.input_dim:
                # [batch, input_dim, seq_len] -> [batch, seq_len, input_dim]
                imu = imu.transpose(1, 2)
            # else: already [batch, seq_len, input_dim]

        # Demographics埋め込み
        demographics_embedding = None
        if self.use_demographics and demographics is not None:
            demographics_embedding = self.demographics_embedding(demographics)

        return self.model(imu, demographics_embedding)

    def compute_kl_loss(self, multiclass_logits: torch.Tensor, nine_class_logits: torch.Tensor) -> torch.Tensor:
        """
        18クラスを9クラスに集約してKL divergenceを計算.

        Args:
            multiclass_logits: [batch, 18] 18クラスのlogits
            nine_class_logits: [batch, 9] 9クラスのlogits

        Returns:
            KL divergence loss
        """
        temperature = self.kl_temperature

        # 18クラスの確率分布
        p_18 = F.softmax(multiclass_logits / temperature, dim=-1)

        # 18→9クラスへの集約
        # Target 8クラス（0-7）はそのまま、Non-target 10クラス（8-17）はindex 8に集約
        p_9_from_18 = torch.zeros(p_18.size(0), 9, device=p_18.device, dtype=p_18.dtype)
        p_9_from_18[:, :8] = p_18[:, :8]  # Target classes
        p_9_from_18[:, 8] = p_18[:, 8:].sum(dim=-1)  # Non-target classes

        # 9クラスのlog確率分布
        log_p_9 = F.log_softmax(nine_class_logits / temperature, dim=-1)

        # KLDivLoss: KL(p_9_from_18 || p_9)
        # nn.KLDivLossはinputがlog_p、targetがpを期待
        kl_loss = self.kl_criterion(log_p_9, p_9_from_18)

        return kl_loss

    def _setup_learnable_weights(self):
        """学習可能重み付けの設定."""
        aw_type = self.loss_config.get("auto_weighting", "none")

        if aw_type == "uncertainty":
            # 不確かさベース自動重み付け
            self.loss_s_params = nn.ParameterDict()
            init_val = self.loss_config.get("uncertainty_init_value", 0.0)

            # 基本的な損失項
            self.loss_s_params["multiclass"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            self.loss_s_params["binary"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

            # 9クラス損失項（有効な場合のみ）
            if self.loss_config.get("nine_class_head_enabled", True):
                self.loss_s_params["nine_class"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

            # KL損失項（有効な場合のみ）
            if self.kl_weight > 0:
                self.loss_s_params["kl"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

        elif aw_type == "direct":
            # 直接的な学習可能重み
            init_alpha = self.loss_config.get("alpha", 0.5)
            init_w9 = self.loss_config.get("nine_class_loss_weight", 0.2)
            init_wkl = self.loss_config.get("kl_weight", 0.1)

            # alpha用（logit parameterization: alpha = sigmoid(alpha_raw)）
            alpha_logit = np.log(init_alpha / (1 - init_alpha + 1e-8))
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

            # その他重み用（log parameterization: weight = exp(weight_raw)）
            self.w9_raw = nn.Parameter(torch.tensor(np.log(max(init_w9, 1e-8)), dtype=torch.float32))
            self.wkl_raw = nn.Parameter(torch.tensor(np.log(max(init_wkl, 1e-8)), dtype=torch.float32))

        else:
            # "none": 従来の固定重み（学習可能パラメータなし）
            pass

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """訓練ステップ."""
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]
        binary_labels = batch["binary_label"]
        nine_class_labels = batch["nine_class_label"]
        attention_mask = batch.get("attention_mask", None)  # 動的パディング時のみ存在
        demographics = batch.get("demographics", None)  # Demographics特徴量（オプショナル）

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = self(imu, attention_mask, demographics)

        # 損失計算
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "soft_f1":
            # SoftF1Lossは確率値を期待するため、sigmoid/softmaxを適用
            multiclass_probs = F.softmax(multiclass_logits, dim=-1)
            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
            nine_class_probs = F.softmax(nine_class_logits, dim=-1)

            multiclass_loss = self.multiclass_criterion(multiclass_probs, multiclass_labels)
            binary_loss = self.binary_criterion(binary_probs, binary_labels.float())
            nine_class_loss = self.nine_class_criterion(nine_class_probs, nine_class_labels)
        elif loss_type == "soft_f1_acls":
            # SoftF1ACLS: ロジットを直接渡す（内部で確率変換とACLS正則化を処理）
            multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
            binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)
            nine_class_loss = self.nine_class_criterion(nine_class_logits, nine_class_labels)
        else:
            # 通常の損失関数（ロジットを直接使用）
            multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
            binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)
            nine_class_loss = self.nine_class_criterion(nine_class_logits, nine_class_labels)

        # KL divergence loss
        if self.kl_weight > 0:
            kl_loss = self.compute_kl_loss(multiclass_logits, nine_class_logits)
        else:
            kl_loss = torch.tensor(0.0, device=multiclass_logits.device)

        # 学習可能重み付き損失の計算
        aw_type = self.loss_config.get("auto_weighting", "none")
        if aw_type == "direct":
            # 直接学習可能重み
            alpha_clamped = torch.clamp(self.alpha_raw, self.clamp_min, self.clamp_max)
            w9_clamped = torch.clamp(self.w9_raw, self.clamp_min, self.clamp_max)
            wkl_clamped = torch.clamp(self.wkl_raw, self.clamp_min, self.clamp_max)

            loss_alpha = torch.sigmoid(alpha_clamped)
            nine_class_weight = torch.sigmoid(w9_clamped)
            kl_weight = torch.sigmoid(wkl_clamped)

            total_loss = (
                loss_alpha * multiclass_loss
                + (1 - loss_alpha) * binary_loss
                + nine_class_weight * nine_class_loss
                + kl_weight * kl_loss
            )
        else:
            # 従来の固定重み損失
            total_loss = (
                self.loss_alpha * multiclass_loss
                + (1 - self.loss_alpha) * binary_loss
                + self.nine_class_loss_weight * nine_class_loss
                + self.kl_weight * kl_loss
            )

        # ログ
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_multiclass_loss", multiclass_loss)
        self.log("train_binary_loss", binary_loss)
        self.log("train_nine_class_loss", nine_class_loss)
        self.log("train_kl_loss", kl_loss)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """検証ステップ."""
        imu = batch["imu"]
        multiclass_labels = batch["multiclass_label"]
        binary_labels = batch["binary_label"]
        nine_class_labels = batch["nine_class_label"]
        sequence_ids = batch["sequence_id"]
        gestures = batch["gesture"]
        attention_mask = batch.get("attention_mask", None)  # 動的パディング時のみ存在
        demographics = batch.get("demographics", None)  # Demographics特徴量（オプショナル）

        # 前向き計算
        multiclass_logits, binary_logits, nine_class_logits = self(imu, attention_mask, demographics)

        # 損失計算
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "soft_f1":
            # SoftF1Lossは確率値を期待するため、sigmoid/softmaxを適用
            multiclass_probs = F.softmax(multiclass_logits, dim=-1)
            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
            nine_class_probs = F.softmax(nine_class_logits, dim=-1)

            multiclass_loss = self.multiclass_criterion(multiclass_probs, multiclass_labels)
            binary_loss = self.binary_criterion(binary_probs, binary_labels.float())
            nine_class_loss = self.nine_class_criterion(nine_class_probs, nine_class_labels)
        else:
            # 通常の損失関数（ロジットを直接使用）
            multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
            binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)
            nine_class_loss = self.nine_class_criterion(nine_class_logits, nine_class_labels)

        # KL divergence loss
        if self.kl_weight > 0:
            kl_loss = self.compute_kl_loss(multiclass_logits, nine_class_logits)
        else:
            kl_loss = torch.tensor(0.0, device=multiclass_logits.device)

        # 学習可能重み付き損失の計算
        aw_type = self.loss_config.get("auto_weighting", "none")
        if aw_type == "direct":
            # 直接学習可能重み
            alpha_clamped = torch.clamp(self.alpha_raw, self.clamp_min, self.clamp_max)
            w9_clamped = torch.clamp(self.w9_raw, self.clamp_min, self.clamp_max)
            wkl_clamped = torch.clamp(self.wkl_raw, self.clamp_min, self.clamp_max)

            loss_alpha = torch.sigmoid(alpha_clamped)
            nine_class_weight = torch.sigmoid(w9_clamped)
            kl_weight = torch.sigmoid(wkl_clamped)

            total_loss = (
                loss_alpha * multiclass_loss
                + (1 - loss_alpha) * binary_loss
                + nine_class_weight * nine_class_loss
                + kl_weight * kl_loss
            )
        else:
            # 従来の固定重み損失
            total_loss = (
                self.loss_alpha * multiclass_loss
                + (1 - self.loss_alpha) * binary_loss
                + self.nine_class_loss_weight * nine_class_loss
                + self.kl_weight * kl_loss
            )

        # 予測値の計算（メトリクス計算のため）
        multiclass_probs = F.softmax(multiclass_logits, dim=-1)
        binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
        nine_class_probs = F.softmax(nine_class_logits, dim=-1)

        # 結果を保存
        output = {
            "val_loss": total_loss,
            "val_multiclass_loss": multiclass_loss,
            "val_binary_loss": binary_loss,
            "val_nine_class_loss": nine_class_loss,
            "val_kl_loss": kl_loss,
            "multiclass_probs": multiclass_probs.cpu(),
            "binary_probs": binary_probs.cpu(),
            "nine_class_probs": nine_class_probs.cpu(),
            "multiclass_labels": multiclass_labels.cpu(),
            "binary_labels": binary_labels.cpu(),
            "nine_class_labels": nine_class_labels.cpu(),
            "sequence_ids": sequence_ids,
            "gestures": gestures,
        }

        self.validation_outputs.append(output)

        # ログ
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_multiclass_loss", multiclass_loss)
        self.log("val_binary_loss", binary_loss)
        self.log("val_nine_class_loss", nine_class_loss)
        self.log("val_kl_loss", kl_loss)

        return output

    def on_validation_epoch_end(self):
        """検証エポック終了時の処理."""
        if not self.validation_outputs:
            return

        # すべての出力を結合
        all_multiclass_probs = torch.cat([x["multiclass_probs"] for x in self.validation_outputs])
        all_binary_probs = torch.cat([x["binary_probs"] for x in self.validation_outputs])
        all_multiclass_labels = torch.cat([x["multiclass_labels"] for x in self.validation_outputs])
        all_binary_labels = torch.cat([x["binary_labels"] for x in self.validation_outputs])

        # ジェスチャー名のリストを結合
        all_gestures = []
        for x in self.validation_outputs:
            all_gestures.extend(x["gestures"])

        # CMIスコアの計算
        try:
            # マルチクラス予測
            multiclass_preds = torch.argmax(all_multiclass_probs, dim=-1)

            # ID→ジェスチャー名の変換が利用可能な場合
            if self.id_to_gesture:
                # 予測IDをジェスチャー名に変換
                gesture_preds = [self.id_to_gesture[idx.item()] for idx in multiclass_preds]

                # CMIスコア計算（ジェスチャー名を使用）
                cmi_score = compute_cmi_score(
                    all_gestures, gesture_preds, self.target_gestures, self.non_target_gestures
                )
            else:
                # 従来の方法（後方互換性のため）
                binary_preds = (all_binary_probs > 0.5).int()
                cmi_score = compute_cmi_score(
                    all_binary_labels.numpy(),
                    binary_preds.numpy(),
                    all_multiclass_labels.numpy(),
                    multiclass_preds.numpy(),
                )

            self.log("val_cmi_score", cmi_score, prog_bar=True)

        except Exception as e:
            print(f"CMI score calculation failed: {e}")

        # 出力をクリア
        self.validation_outputs.clear()

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定."""
        # Schedule Free optimizer使用時
        if self.schedule_free_config.get("enabled", False) and SCHEDULEFREE_AVAILABLE:
            # 学習可能重みパラメータの分離
            aw_params = []
            aw_type = self.loss_config.get("auto_weighting", "none")

            if aw_type == "direct" and hasattr(self, "alpha_raw"):
                aw_params = [self.alpha_raw, self.w9_raw, self.wkl_raw]

            optimizer_type = self.schedule_free_config.get("optimizer_type", "RAdamScheduleFree")
            lr_multiplier = self.schedule_free_config.get("learning_rate_multiplier", 5.0)
            effective_lr = self.learning_rate * lr_multiplier

            # パラメータの準備
            if aw_params:
                base_params = [p for n, p in self.named_parameters() if p not in set(aw_params)]
                params = [
                    {"params": base_params},
                    {"params": aw_params, "lr": effective_lr * 0.1, "weight_decay": 0.0},
                ]
            else:
                params = self.parameters()

            # Schedule Free optimizerの選択
            if optimizer_type == "RAdamScheduleFree":
                optimizer = RAdamScheduleFree(
                    params,
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            elif optimizer_type == "AdamWScheduleFree":
                optimizer = AdamWScheduleFree(
                    params,
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            elif optimizer_type == "SGDScheduleFree":
                optimizer = SGDScheduleFree(
                    params,
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )
            else:
                print(f"Unknown Schedule Free optimizer: {optimizer_type}, fallback to RAdamScheduleFree")
                optimizer = RAdamScheduleFree(
                    params,
                    lr=effective_lr,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.schedule_free_config.get("warmup_steps", 1000),
                )

            print(f"Using {optimizer_type} with learning rate: {effective_lr:.2e}")
            if aw_params:
                print(f"Using separate parameter group for learnable weights with lr: {effective_lr * 0.1:.2e}")
            return optimizer

        # 学習可能重みパラメータの分離
        aw_params = []
        aw_type = self.loss_config.get("auto_weighting", "none")

        if aw_type == "direct" and hasattr(self, "alpha_raw"):
            aw_params = [self.alpha_raw, self.w9_raw, self.wkl_raw]

        # 通常のオプティマイザ
        if aw_params:
            # 基本パラメータと学習可能重みパラメータを分離
            base_params = [p for n, p in self.named_parameters() if p not in set(aw_params)]

            # パラメータグループ別最適化（学習可能重みは小さな学習率、weight decay無し）
            optimizer = AdamW(
                [
                    {"params": base_params},
                    {"params": aw_params, "lr": self.learning_rate * 0.1, "weight_decay": 0.0},
                ],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
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
    # exp029用IMU-only LSTMモデルのテスト（demographics + multi-head）
    model = CMISqueezeformer(
        input_dim=20,
        num_classes=18,
        demographics_config={
            "enabled": True,
            "embedding_dim": 16,
            "categorical_features": ["adult_child", "sex", "handedness"],
            "numerical_features": ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"],
            "categorical_embedding_dims": {"adult_child": 2, "sex": 2, "handedness": 2},
        },
        loss_config={
            "type": "focal",
            "kl_weight": 0.1,
            "nine_class_loss_weight": 0.2,
            "alpha": 0.5,
            "auto_weighting": "direct",
        },
    )

    # テストデータ（IMU物理特徴量20次元）
    batch_size = 2
    seq_len = 200
    imu_data = torch.randn(batch_size, seq_len, 20)  # [batch, seq_len, imu_dim]

    # Demographics テストデータ
    demographics_data = {
        "adult_child": torch.randint(0, 2, (batch_size,)),
        "sex": torch.randint(0, 2, (batch_size,)),
        "handedness": torch.randint(0, 2, (batch_size,)),
        "age": torch.randn(batch_size) * 10 + 25,  # 年齢
        "height_cm": torch.randn(batch_size) * 20 + 170,  # 身長
        "shoulder_to_wrist_cm": torch.randn(batch_size) * 10 + 55,  # 肩-手首
        "elbow_to_wrist_cm": torch.randn(batch_size) * 5 + 30,  # 肘-手首
    }

    # 前向き計算（multi-head）
    multiclass_logits, binary_logits, nine_class_logits = model(imu_data, demographics=demographics_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Multiclass output shape: {multiclass_logits.shape}")
    print(f"Binary output shape: {binary_logits.shape}")
    print(f"Nine-class output shape: {nine_class_logits.shape}")

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # IMUOnlyLSTMクラス単体のテスト
    print("\n--- IMUOnlyLSTM Model Test ---")
    imu_model = IMUOnlyLSTM(imu_dim=20, n_classes=18, demographics_dim=16)
    demographics_embedding = torch.randn(batch_size, 16)  # テスト用埋め込み
    multiclass_logits_direct, binary_logits_direct, nine_class_logits_direct = imu_model(
        imu_data, demographics_embedding
    )
    print(f"IMUOnlyLSTM multiclass output shape: {multiclass_logits_direct.shape}")
    print(f"IMUOnlyLSTM binary output shape: {binary_logits_direct.shape}")
    print(f"IMUOnlyLSTM nine-class output shape: {nine_class_logits_direct.shape}")

    imu_params = sum(p.numel() for p in imu_model.parameters())
    print(f"IMUOnlyLSTM parameters: {imu_params:,}")

    # KL divergence lossのテスト
    print("\n--- KL Loss Test ---")
    kl_loss = model.compute_kl_loss(multiclass_logits, nine_class_logits)
    print(f"KL divergence loss: {kl_loss.item():.6f}")
