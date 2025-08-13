"""IMU特化Squeezeformerモデル - exp013用（Demographics統合版）."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# ACLS losses import
from losses import ACLS, ACLSBinary, LabelSmoothingBCE, LabelSmoothingCrossEntropy, MbLS, MbLSBinary
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Schedule Free optimizers
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


class IMUFeatureExtractor(nn.Module):
    """
    IMU特徴量抽出モジュール（exp013準拠版）.

    生のIMUデータ（7次元）から物理ベース特徴量（16次元）を生成。
    exp013のPolars実装をPyTorchで再現し、数値的一致を目指す。
    """

    def __init__(self, sampling_rate: float = 100.0, time_delta: float = 1.0 / 200.0, tol: float = 1e-8):
        """
        初期化.

        Args:
            sampling_rate: サンプリングレート [Hz] (フィルタ用、使用しない)
            time_delta: タイムステップ [s] (デフォルト1/200秒、exp013準拠)
            tol: 数値安定化の閾値 (exp013準拠の1e-8)
        """
        super().__init__()

        self.sampling_rate = sampling_rate  # 使用しないが互換性のため保持
        self.time_delta = time_delta  # exp013と同じ値
        self.tol = tol  # exp013と同じtolerance

    def remove_gravity(self, acc: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """
        重力成分を除去して線形加速度を計算（exp013準拠）.

        Args:
            acc: 加速度 [B, 3, T] (x, y, z)
            quat: 四元数 [B, 4, T] (w, x, y, z)

        Returns:
            linear_acc: 線形加速度 [B, 3, T]
        """
        # exp013のPolars実装に準拠した四元数変数名
        x = quat[:, 1:2, :]  # rot_x
        y = quat[:, 2:3, :]  # rot_y
        z = quat[:, 3:4, :]  # rot_z
        w = quat[:, 0:1, :]  # rot_w

        # 有効判定：ノルムがしきい値超え
        norm = torch.sqrt(x**2 + y**2 + z**2 + w**2)
        valid = norm > self.tol

        # 正規化四元数
        xn = x / norm
        yn = y / norm
        zn = z / norm
        wn = w / norm

        # exp013の実装と完全一致：重力ベクトル（センサ座標）
        gx = 19.62 * (xn * zn - wn * yn)
        gy = 19.62 * (wn * xn + yn * zn)
        gz = 9.81 - 19.62 * (xn * xn + yn * yn)

        # 無効行は重力0扱いにしてaccをそのまま通す
        gx = torch.where(valid, gx, 0.0)
        gy = torch.where(valid, gy, 0.0)
        gz = torch.where(valid, gz, 0.0)

        gravity = torch.cat([gx, gy, gz], dim=1)  # [B, 3, T]

        # 線形加速度 = 測定 - 重力（有効行）/ 無効行は測定のまま
        linear_acc = torch.where(valid.expand(-1, 3, -1), acc - gravity, acc)

        return linear_acc

    def quaternion_to_angular_velocity(self, quat: torch.Tensor) -> torch.Tensor:
        """
        四元数から角速度を計算（exp013準拠のdelta quaternion approach）.

        Args:
            quat: 四元数 [B, 4, T] (w, x, y, z)

        Returns:
            angular_vel: 角速度 [B, 3, T] (x, y, z)
        """
        # exp013と同じ変数名
        x1 = quat[:, 1:2, :]  # rot_x
        y1 = quat[:, 2:3, :]  # rot_y
        z1 = quat[:, 3:4, :]  # rot_z
        w1 = quat[:, 0:1, :]  # rot_w

        # シフト操作（次のタイムステップ）
        # 最後の行は前の行を複製（シーケンス境界を考慮）
        x2 = F.pad(x1[:, :, 1:], (0, 1), mode='replicate')
        y2 = F.pad(y1[:, :, 1:], (0, 1), mode='replicate')
        z2 = F.pad(z1[:, :, 1:], (0, 1), mode='replicate')
        w2 = F.pad(w1[:, :, 1:], (0, 1), mode='replicate')

        # ノルムで正規化（exp013準拠）
        n1 = torch.sqrt(x1**2 + y1**2 + z1**2 + w1**2)
        n2 = torch.sqrt(x2**2 + y2**2 + z2**2 + w2**2)
        xn1, yn1, zn1, wn1 = x1 / n1, y1 / n1, z1 / n1, w1 / n1
        xn2, yn2, zn2, wn2 = x2 / n2, y2 / n2, z2 / n2, w2 / n2

        # 連続性のための符号合わせ（dot<0なら後者を反転）
        dot = xn1 * xn2 + yn1 * yn2 + zn1 * zn2 + wn1 * wn2
        sgn = torch.where(dot < 0, -1.0, 1.0)
        xn2, yn2, zn2, wn2 = xn2 * sgn, yn2 * sgn, zn2 * sgn, wn2 * sgn

        # delta = q1^{-1} * q2 = conj(q1) * q2を計算
        # conj(q1)=(-x1,-y1,-z1,w1)。四元数積の展開（スカラー末尾）
        dx = wn1 * xn2 - xn1 * wn2 - yn1 * zn2 + zn1 * yn2
        dy = wn1 * yn2 + xn1 * zn2 - yn1 * wn2 - zn1 * xn2
        dz = wn1 * zn2 - xn1 * yn2 + yn1 * xn2 - zn1 * wn2
        dw = wn1 * wn2 + xn1 * xn2 + yn1 * yn2 + zn1 * zn2

        # 数値安全のためクランプ
        dw = torch.clamp(dw, -1.0, 1.0)

        # 回転ベクトル = 角度 * 単位軸
        vnorm = torch.sqrt(dx**2 + dy**2 + dz**2)
        angle = 2.0 * torch.atan2(vnorm, dw)  # より安定（acos より端がマシ）
        scale = torch.where(vnorm > self.tol, angle / vnorm, 0.0)

        rvx = dx * scale
        rvy = dy * scale
        rvz = dz * scale

        # 角速度 [rad/s]
        wx = rvx / self.time_delta
        wy = rvy / self.time_delta
        wz = rvz / self.time_delta

        # 有効判定（NaN/Nullや最終行は無効 → 0）
        valid = (
            (n1 > self.tol) &
            (n2 > self.tol) &
            torch.isfinite(wx) &
            torch.isfinite(wy) &
            torch.isfinite(wz)
        )

        wx = torch.where(valid, wx, 0.0)
        wy = torch.where(valid, wy, 0.0)
        wz = torch.where(valid, wz, 0.0)

        angular_vel = torch.cat([wx, wy, wz], dim=1)

        return angular_vel

    def calculate_angular_distance(self, quat: torch.Tensor) -> torch.Tensor:
        """
        連続する四元数間の角距離を計算（exp013準拠）.

        Args:
            quat: 四元数 [B, 4, T] (w, x, y, z)

        Returns:
            angular_distance: 角距離 [B, 1, T]
        """
        # 入力: rot_x, rot_y, rot_z, rot_w（スカラー末尾）
        x1 = quat[:, 1:2, :]  # rot_x
        y1 = quat[:, 2:3, :]  # rot_y
        z1 = quat[:, 3:4, :]  # rot_z
        w1 = quat[:, 0:1, :]  # rot_w

        # シフト操作
        x2 = F.pad(x1[:, :, 1:], (0, 1), mode='replicate')
        y2 = F.pad(y1[:, :, 1:], (0, 1), mode='replicate')
        z2 = F.pad(z1[:, :, 1:], (0, 1), mode='replicate')
        w2 = F.pad(w1[:, :, 1:], (0, 1), mode='replicate')

        # 正規化（exp013準拠）
        n1 = torch.sqrt(x1**2 + y1**2 + z1**2 + w1**2)
        n2 = torch.sqrt(x2**2 + y2**2 + z2**2 + w2**2)
        xn1, yn1, zn1, wn1 = x1 / n1, y1 / n1, z1 / n1, w1 / n1
        xn2, yn2, zn2, wn2 = x2 / n2, y2 / n2, z2 / n2, w2 / n2

        # q と -q の同値対策（最短経路）
        dot = xn1 * xn2 + yn1 * yn2 + zn1 * zn2 + wn1 * wn2
        sgn = torch.where(dot < 0, -1.0, 1.0)
        xn2, yn2, zn2, wn2 = xn2 * sgn, yn2 * sgn, zn2 * sgn, wn2 * sgn

        # delta = q1^{-1} * q2 = conj(q1) * q2 （スカラー末尾の四元数積）
        dx = wn1 * xn2 - xn1 * wn2 - yn1 * zn2 + zn1 * yn2
        dy = wn1 * yn2 + xn1 * zn2 - yn1 * wn2 - zn1 * xn2
        dz = wn1 * zn2 - xn1 * yn2 + yn1 * xn2 - zn1 * wn2
        dw = wn1 * wn2 + xn1 * xn2 + yn1 * yn2 + zn1 * zn2

        # 数値安定化＆角度（0..π）
        vnorm = torch.sqrt(dx**2 + dy**2 + dz**2)
        angle = 2.0 * torch.atan2(vnorm, dw)  # = ||as_rotvec||

        # 有効判定（末尾行やNaN/ゼロ長は 0）
        valid = (n1 > self.tol) & (n2 > self.tol) & torch.isfinite(angle)

        angular_distance = torch.where(valid, angle, 0.0)

        return angular_distance

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        前向き伝搬（exp013準拠の特徴量のみ生成）.

        Args:
            imu_data: IMUデータ [B, 7, T] (acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z)

        Returns:
            features: 拡張特徴量 [B, 16, T] (exp013準拠)
        """
        # 入力データの分離
        acc = imu_data[:, :3, :]  # [B, 3, T] - 加速度
        quat = imu_data[:, 3:, :]  # [B, 4, T] - 四元数 (w, x, y, z)

        # 1. 基本IMU特徴量 (7次元)
        features_list = [imu_data]  # 元の7次元データを含める

        # 2. 物理ベース特徴量（exp013から）(9次元)
        # 線形加速度（重力除去）
        linear_acc = self.remove_gravity(acc, quat)
        features_list.append(linear_acc)  # [B, 3, T]

        # 線形加速度の大きさ
        linear_acc_mag = torch.norm(linear_acc, dim=1, keepdim=True)
        features_list.append(linear_acc_mag)  # [B, 1, T]

        # 線形加速度大きさのジャーク
        linear_acc_mag_jerk = F.pad(linear_acc_mag[:, :, 1:] - linear_acc_mag[:, :, :-1], (1, 0))
        features_list.append(linear_acc_mag_jerk)  # [B, 1, T]

        # 角速度（四元数から計算）
        angular_vel = self.quaternion_to_angular_velocity(quat)
        features_list.append(angular_vel)  # [B, 3, T]

        # 角距離
        angular_distance = self.calculate_angular_distance(quat)
        features_list.append(angular_distance)  # [B, 1, T]

        # 全特徴量を結合
        features = torch.cat(features_list, dim=1)  # [B, 16, T]

        return features


class CMISqueezeformer(pl.LightningModule):
    """CMI用のSqueezeformerモデル（IMU特化版）."""

    def __init__(
        self,
        input_dim: int = 7,  # IMU: 7次元（生データ）
        feature_extractor_config: dict | None = None,
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
        acls_config: dict | None = None,
        schedule_free_config: dict | None = None,
        ema_config: dict | None = None,
        demographics_config: dict | None = None,
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
        id_to_gesture: dict[int, str] | None = None,
    ):
        """
        初期化.

        Args:
            input_dim: 入力次元数（IMU: 7次元、生データ）
            feature_extractor_config: IMU特徴量抽出器の設定
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
            acls_config: ACLS損失関数設定
            schedule_free_config: Schedule Free オプティマイザ設定
            ema_config: EMA（Exponential Moving Average）設定
            demographics_config: Demographics統合設定
            target_gestures: ターゲットジェスチャーのリスト
            non_target_gestures: ノンターゲットジェスチャーのリスト
            id_to_gesture: ID→ジェスチャー名のマッピング
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
        self.acls_config = acls_config or {}
        self.schedule_free_config = schedule_free_config or {}
        self.ema_config = ema_config or {}
        self.demographics_config = demographics_config or {}
        self.feature_extractor_config = feature_extractor_config or {}

        # IMU特徴量抽出器（exp013準拠）
        self.imu_feature_extractor = IMUFeatureExtractor(
            sampling_rate=self.feature_extractor_config.get("sampling_rate", 100.0),
            time_delta=self.feature_extractor_config.get("time_delta", 1.0 / 200.0),
            tol=self.feature_extractor_config.get("tol", 1e-8),
        )

        # 特徴量抽出後の次元数（16次元、exp013準拠）
        self.extracted_feature_dim = 16

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

        # Input projection（特徴量抽出後の16次元 → d_model）
        self.input_projection = nn.Linear(self.extracted_feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Squeezeformer blocks
        self.blocks = nn.ModuleList(
            [SqueezeformerBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)]
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification heads (IMU + Demographics特徴量を結合)
        classification_input_dim = d_model + self.demographics_dim

        self.multiclass_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.binary_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, d_model // 4),
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
        elif loss_type == "soft_f1":
            # SoftF1Loss
            beta = self.loss_config.get("soft_f1_beta", 1.0)
            eps = self.loss_config.get("soft_f1_eps", 1e-6)
            self.multiclass_criterion = MulticlassSoftF1Loss(num_classes=self.num_classes, beta=beta, eps=eps)
            self.binary_criterion = BinarySoftF1Loss(beta=beta, eps=eps)
        elif loss_type == "acls":
            # ACLS (Adaptive and Conditional Label Smoothing)
            self.multiclass_criterion = ACLS(
                pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
                neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
                alpha=self.acls_config.get("acls_alpha", 0.1),
                margin=self.acls_config.get("acls_margin", 10.0),
                num_classes=self.num_classes,
            )
            self.binary_criterion = ACLSBinary(
                pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
                neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
                alpha=self.acls_config.get("acls_alpha", 0.1),
                margin=self.acls_config.get("acls_margin", 10.0),
                num_classes=2,
            )
        elif loss_type == "label_smoothing":
            # Label Smoothing Cross-Entropy
            self.multiclass_criterion = LabelSmoothingCrossEntropy(
                alpha=self.acls_config.get("label_smoothing_alpha", 0.1)
            )
            self.binary_criterion = LabelSmoothingBCE(alpha=self.acls_config.get("label_smoothing_alpha", 0.1))
        elif loss_type == "mbls":
            # Margin-based Label Smoothing
            self.multiclass_criterion = MbLS(
                margin=self.acls_config.get("mbls_margin", 10.0),
                alpha=self.acls_config.get("mbls_alpha", 0.1),
                alpha_schedule=self.acls_config.get("mbls_schedule", None),
            )
            self.binary_criterion = MbLSBinary(
                margin=self.acls_config.get("mbls_margin", 10.0), alpha=self.acls_config.get("mbls_alpha", 0.1)
            )
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
        self,
        imu: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        demographics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向き計算.

        Args:
            imu: 生のIMUデータ [batch, 7, seq_len] (acc_x/y/z, rot_w/x/y/z)
            attention_mask: アテンションマスク
            demographics: Demographics特徴量の辞書（オプショナル）

        Returns:
            multiclass_logits: マルチクラス予測 [batch, num_classes]
            binary_logits: バイナリ予測 [batch, 1]
        """
        # 特徴量抽出: [batch, 7, seq_len] -> [batch, 39, seq_len]
        x = self.imu_feature_extractor(imu)

        # [batch, extracted_feature_dim, seq_len] -> [batch, seq_len, extracted_feature_dim]
        x = x.transpose(1, 2)

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

        # Demographics特徴量の統合
        if self.use_demographics and demographics is not None:
            demographics_embedding = self.demographics_embedding(demographics)  # [batch, demographics_dim]
            # IMU特徴量とdemographics特徴量を結合
            x = torch.cat([x, demographics_embedding], dim=-1)  # [batch, d_model + demographics_dim]
        elif self.use_demographics and demographics is None:
            # Demographics使用設定だが、データがない場合はゼロパディング
            batch_size = x.size(0)
            demographics_padding = torch.zeros(batch_size, self.demographics_dim, device=x.device)
            x = torch.cat([x, demographics_padding], dim=-1)

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
        demographics = batch.get("demographics", None)  # Demographics特徴量（オプショナル）

        # 前向き計算
        multiclass_logits, binary_logits = self(imu, attention_mask, demographics)

        # 損失計算
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "soft_f1":
            # SoftF1Lossは確率値を期待するため、sigmoid/softmaxを適用
            multiclass_probs = F.softmax(multiclass_logits, dim=-1)
            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))

            multiclass_loss = self.multiclass_criterion(multiclass_probs, multiclass_labels)
            binary_loss = self.binary_criterion(binary_probs, binary_labels.float())
        else:
            # 通常の損失関数（ロジットを直接使用）
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
        demographics = batch.get("demographics", None)  # Demographics特徴量（オプショナル）

        # 前向き計算
        multiclass_logits, binary_logits = self(imu, attention_mask, demographics)

        # 損失計算
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "soft_f1":
            # SoftF1Lossは確率値を期待するため、sigmoid/softmaxを適用
            multiclass_probs = F.softmax(multiclass_logits, dim=-1)
            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))

            multiclass_loss = self.multiclass_criterion(multiclass_probs, multiclass_labels)
            binary_loss = self.binary_criterion(binary_probs, binary_labels.float())
        else:
            # 通常の損失関数（ロジットを直接使用）
            multiclass_loss = self.multiclass_criterion(multiclass_logits, multiclass_labels)
            binary_loss = self.binary_criterion(binary_logits.squeeze(-1), binary_labels)

        total_loss = self.loss_alpha * multiclass_loss + (1 - self.loss_alpha) * binary_loss

        # 予測値の計算（メトリクス計算のため）
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
