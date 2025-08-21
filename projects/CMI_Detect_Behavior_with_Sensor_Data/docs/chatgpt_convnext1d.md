**You:**
docs/competition_overview.md
docs/data_description.md
このコンペで、imuのデータだけでシングルモデルで0.8以上のスコアを出すための前処理、特徴量エンジニアリング、モデルアーキテクチャ、augmentation, lossを教えて

* * *

**ChatGPT:**

狙い：IMU（加速度＋姿勢/回転）だけ・単一モデルでLB≥0.80。  
このコンペは「ターゲット行動か否かのBinary F1」と「各ジェスチャーのMacro F1」を等重みで平均した独自Macro F1が評価です。データは手首装着Heliosのセンサ（IMU等）で取得された時系列です。[Kaggle+1](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data?utm_source=chatgpt.com)

* * *

# 推奨レシピ（前処理→特徴量→モデル→Aug→Loss）

## 1) 前処理（IMU専用）

* **重力除去**：四元数（`rot_w,x,y,z`）でワールド座標へ回転→加速度から重力ベクトルを引いて**線形加速度**を得る。  
    併せて**角速度**は四元数差分から近似（短窓差分＋小角近似）。
    
* **姿勢基準化**：開始数百msの平均姿勢で基準化し，各サンプルを**身体座標に再マップ**（利き手/装着向き差を低減）。
    
* **ドリフト&ノイズ整形**：ローパス（例：3–5 Hz）で重力・高周波を分離し，ジャーク（一次差分）用には5–10 Hzハイパスも。必要なら**VQF**等の高精度姿勢推定で四元数の安定化。[arXiv](https://arxiv.org/abs/2203.17024?utm_source=chatgpt.com)
    
* **系列整形**：サンプリング不均一ならリサンプリング（例：100 Hz固定）。学習は**可変長対応**（padding＋mask）にして切り捨てを避ける。
    
* **標準化**：被験者/シーケンス単位でチャンネル毎にz-score（リーク防止のためfold内fit）。
    

## 2) 特徴量エンジニアリング（“生+派生”の両輪）

* **生波形**：`acc_x,y,z` と（四元数由来の）**角速度**3軸。
    
* **派生**
    
    * スカラー：‖acc‖, ‖gyro‖, **ジャーク**（Δacc）, 姿勢角（roll/pitch/yaw）
        
    * 統計：移動窓のmean/std/RMS/energy, max−min, IQR
        
    * 相関：軸間相関・自己相関ピーク・周期推定
        
    * 周波数：STFT/FFTの帯域パワー（0.3–3 Hz, 3–8 Hz, 8–15 Hz…）と**スペクトル重心**・**帯域比**
        
* **二重表現**：**生時系列ブランチ**＋**周波数ブランチ**（STFT特徴）を並行学習し，後段でAttention融合（HARのAttention融合は有効）。[arXiv](https://arxiv.org/abs/2112.11224?utm_source=chatgpt.com)
    

## 3) 単一モデルのアーキテクチャ（推し構成）

**“セグメント注意付き”1D-CNN→Transformerの多タスク単体モデル**

* **バックボーン**：ResNet1D/ConvNeXt1D（カーネル 5–9，埋め256）→Transformer Encoder（4–6層，ヘッド4–8）。
    
* **セグメント頭**（frame-wise _gesture_確率）：1D U-Net風デコーダで2値（ジェスチャー vs 非）。
    
* **分類頭**（18クラス）：セグメント確率を**重み**に使う**Weighted Global Average Pooling**で文脈プーリング→MLP。
    
* **“単一モデル”要件**：上記2ヘッドは**同一ネット内**。推論はセグメント頭の出力で重み付けプーリングするだけ（後段モデルは無し）。
    
* **可変長対応**：mask注意＋Global poolingで対応。
    
* **実装Tips**：DropPath 0.1–0.2，LayerScale，pre-norm，EMA，有効。
    

> この設計は「どこがジェスチャーか」を**モデル自身に学習させ**，その**注意マップ**で分類特徴を集約するので，trainにあるphase情報の恩恵を受けつつtestでは**単体モデル**で完結します。

## 4) Augmentation（物理一貫性を担保）

* **SO(3)回転**：加速度ベクトルをランダム回転（±15–25°）。四元数/角速度にも**同じ回転**を適用して一貫性を保つ。
    
* **タイムワープ/スケーリング**：ランダム時間伸縮（0.9–1.1×），微小速度スケール（gyro×[0.9,1.1]）。
    
* **Jitter/Drift**：加速度に微小ガウスノイズ（σ=0.01–0.03g），gyroにランダムドリフト（線形±小）。
    
* **TimeCutMix/Concat**：同クラスの別シーケンスから時間区間を貼り付け（5–30%）／前後接続。
    
* **ChannelDrop**：まれに1軸をマスク（p=0.05）→ロバスト化。
    
* **SpecAugment的**：STFTブランチに時間/周波数マスク（小さめ）。
    

## 5) Loss設計（評価指標に直結）

* **分類ヘッド**：`FocalLoss(γ=2) + SoftF1Loss` の合成（λで和）。クラス不均衡には**class-balanced focal**や**logit adjustment**。
    
* **セグメントヘッド**：`BCEWithLogits + Dice`（薄い領域の検出安定化）。
    
* **階層一貫性**：
    
    * 一貫性正則化：Binary(=ターゲット/非) ≈ Multi(=18クラス)の**総和**になるようKL/CEで整合。
        
    * 推論時も **P(クラス)=P(クラス|ターゲット)×P(ターゲット)** のゲーティングで最終確率を出力（安定化に効く）。
        
* **学習スケジュール**：AdamW, lr 1e-3→Cosine, warmup 5 % , batch 64–128, 30–80epoch, gradient clipping。
    

* * *

# 検証・分割と実運用Tips

* **CV設計**：**被験者や姿勢条件（例：立位/座位/臥位）でリークを断つGroupKFold**。公開議論でも「未知の姿勢に対する汎化」を意識する声あり。[Kaggle](https://www.kaggle.com/code/stemosamaghandour/cmi-detect-behavior-with-sensor-data-by-stemer?utm_source=chatgpt.com)
    
* **TTA**：小回転×数回＋時間スケール±5% で確率平均。
    
* **閾値最適化**：検証FoldごとにBinary F1とMacro F1の**平均スコア**を直接最大化する閾値探索（Grid or Optuna）。評価が**Binary＋Macroの等重み**であることを損なわない。[Kaggle](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data?utm_source=chatgpt.com)
    
* **上振れ余地**：公開情報ベースでトップは0.86付近（全モダリティ）。IMU単体でも設計とCVを詰めれば0.8台は現実的。[clist.by](https://clist.by/standings/cmi-detect-behavior-with-sensor-data-health-time-series-analysis-custom-metric-59965894/?utm_source=chatgpt.com)
    

* * *

## まずの実装順（短期でLB0.8狙いの最短手順）

1. **重力除去＋角速度導出＋z-score**のみでResNet1D→Transformer（分類単頭）を作り，ベースライン確立。
    
2. **セグメント頭追加**＋Weighted pooling化（単一モデル完成）。
    
3. **SO(3)回転＋TimeWarp＋Jitter**のAug 3点セット導入。
    
4. **Focal+SoftF1**に切替え，**P(クラス|ターゲット)×P(ターゲット)**のゲート適用。
    
5. **二重表現**（生＋STFT）を追加しAttention融合。
    
6. **TTA＋閾値最適化**で詰める。
    

必要なら、この設計で**学習コード雛形（PyTorch）**と**前処理ユーティリティ**（重力除去・四元数→角速度）をすぐ書きます。コンペ仕様要点は上記ページをご参照ください（評価とデータ記述）。[Kaggle+1](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data?utm_source=chatgpt.com)

* * *

**You:**

モデルのpytorch lightning実装例を教えて

* * *

**ChatGPT:**

了解。下記は「単一モデル（1D-CNN→Transformer）＋セグメント頭（frame-wise gesture確率）＋分類頭（Weighted GAP）＋Focal+SoftF1 / BCE+Dice」の PyTorch Lightning 実装例です。  
前提：バッチは `{"x": (B,C,T), "mask": (B,T[bool]), "y_class": (B,), "y_seg": (B,T)} を返す DataLoader` を想定。

```python
# pip install torch pytorch-lightning
from typing import Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# -----------------------
# Loss utilities
# -----------------------
class FocalLoss(nn.Module):
    """
    Multiclass focal loss on logits.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,)
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)  # = softmax(logits)[range(B), target]
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftF1Loss(nn.Module):
    """
    Macro Soft-F1 for multiclass (probabilities via softmax).
    Works on batch; returns 1 - mean(F1 per class).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,)
        probs = F.softmax(logits, dim=-1)  # (B, C)
        y_true = F.one_hot(target, num_classes=probs.size(-1)).float()  # (B, C)

        tp = (probs * y_true).sum(dim=0)
        fp = (probs * (1.0 - y_true)).sum(dim=0)
        fn = ((1.0 - probs) * y_true).sum(dim=0)

        precision = tp / (tp + fp + self.eps)
        recall    = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1.0 - f1.mean()


class DiceLoss(nn.Module):
    """Binary Dice on logits with optional mask (B, T)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # logits/target: (B, T); mask: (B, T) bool
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask
            target = target * mask
        inter = (probs * target).sum(dim=1)
        denom = (probs * probs).sum(dim=1) + (target * target).sum(dim=1) + self.eps
        dice = 2 * inter / denom
        return 1.0 - dice.mean()


# -----------------------
# Model blocks
# -----------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=None, groups=1):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
    def forward(self, x):  # (B, C, T)
        return self.act(self.bn(self.conv(x)))


class ResBlock1D(nn.Module):
    def __init__(self, ch, k=7, s=1):
        super().__init__()
        p = k // 2
        self.conv1 = ConvBNAct(ch, ch, k=k, s=s, p=p)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, stride=1, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(ch)
        self.act   = nn.GELU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE for (B, T, D)."""
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class WeightedGAP(nn.Module):
    """
    Weighted Global Average Pooling over time with mask.
    x: (B, T, D), w: (B, T) in [0,1], mask: (B, T) bool
    returns: (B, D)
    """
    def forward(self, x: torch.Tensor, w: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            w = w * mask.float()
        # avoid all-zero
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (x * w.unsqueeze(-1)).sum(dim=1) / denom


# -----------------------
# LightningModule
# -----------------------
class IMUSegClassModel(pl.LightningModule):
    """
    Single model:
      - CNN stem + ResBlocks
      - Transformer encoder (mask-aware)
      - Segmentation head: frame-wise gesture logit (B, T)
      - Classification head: WeightedGAP by seg prob -> MLP -> class logits (B, C)
      - Inference: P(class) = softmax(logits_cond) * mean_t(p_seg)
    """
    def __init__(
        self,
        in_chans: int = 6,              # e.g., acc(3) + gyro(3) or acc(3)+quat(4)なら変更
        num_classes: int = 18,
        emb_dim: int = 256,
        n_resblocks: int = 3,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        lambda_focal: float = 0.5,
        lambda_softf1: float = 0.5,
        lambda_dice: float = 0.5,
        use_class_weights: Optional[torch.Tensor] = None,  # (num_classes,)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["use_class_weights"])  # nice to have in Lightning logs
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        # --- CNN stem ---
        ch = emb_dim // 2
        self.stem = nn.Sequential(
            ConvBNAct(in_chans, ch, k=7),
            ResBlock1D(ch, k=7),
            ConvBNAct(ch, emb_dim, k=5),
        )
        self.resblocks = nn.Sequential(*[ResBlock1D(emb_dim, k=5) for _ in range(n_resblocks)])

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.posenc = PositionalEncoding(emb_dim)

        # --- Heads ---
        # Segmentation (frame-wise binary logit for "gesture"):
        self.seg_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1)   # -> (B, T, 1)
        )
        # Classification (sequence-level, conditional on gesture):
        self.pool = WeightedGAP()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

        # --- Losses ---
        class_weight = use_class_weights if use_class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=class_weight)
        self.focal = FocalLoss(gamma=2.0, alpha=class_weight)
        self.softf1 = SoftF1Loss()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # we will mask then mean
        self.dice = DiceLoss()

    def forward_features(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T) -> features: (B, T, D)
        h = self.stem(x)            # (B, D, T)
        h = self.resblocks(h)       # (B, D, T)
        h = h.transpose(1, 2)       # (B, T, D)
        h = self.posenc(h)          # (B, T, D)
        # Transformer requires key_padding_mask: (B, T) where True means PAD (to be ignored)
        key_padding_mask = (~mask) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, T, D)
        return h

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            seg_logits: (B, T)  frame-wise gesture logits
            cls_logits_cond: (B, C)  class logits P(class | target)
            p_target_seq: (B,) mean_t sigmoid(seg_logits) over valid frames
            p_class: (B, C) final P(class) = softmax(cls_logits_cond) * p_target_seq
        """
        feats = self.forward_features(x, mask)     # (B, T, D)
        seg_logits = self.seg_head(feats).squeeze(-1)  # (B, T)
        p_seg = torch.sigmoid(seg_logits)
        if mask is not None:
            p_target_seq = (p_seg * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1e-6)
        else:
            p_target_seq = p_seg.mean(dim=1)

        pooled = self.pool(feats, w=p_seg, mask=mask)  # (B, D)
        cls_logits_cond = self.cls_head(pooled)        # (B, C)
        p_class = F.softmax(cls_logits_cond, dim=-1) * p_target_seq.unsqueeze(-1)
        return {
            "seg_logits": seg_logits,
            "cls_logits_cond": cls_logits_cond,
            "p_target_seq": p_target_seq,
            "p_class": p_class,
        }

    # -----------------------
    # Lightning training/val
    # -----------------------
    def _seg_losses(self, seg_logits: torch.Tensor, y_seg: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # BCE with masking
        bce = self.bce(seg_logits, y_seg)  # (B, T)
        if mask is not None:
            bce = (bce * mask.float()).sum() / mask.float().sum().clamp_min(1e-6)
        else:
            bce = bce.mean()
        dice = self.dice(seg_logits, y_seg, mask=mask)
        return {"bce": bce, "dice": dice}

    def _cls_losses(self, cls_logits_cond: torch.Tensor, y_class: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = self.ce(cls_logits_cond, y_class)
        focal = self.focal(cls_logits_cond, y_class)
        softf1 = self.softf1(cls_logits_cond, y_class)
        return {"ce": ce, "focal": focal, "softf1": softf1}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x, mask = batch["x"], batch.get("mask", None)
        y_class = batch["y_class"]
        y_seg = batch.get("y_seg", None)  # (B, T) or None

        out = self(x, mask=mask)
        # classification losses (on conditional logits)
        cls_ls = self._cls_losses(out["cls_logits_cond"], y_class)
        loss_cls = cls_ls["ce"] + self.hparams.lambda_focal * cls_ls["focal"] + self.hparams.lambda_softf1 * cls_ls["softf1"]

        # segmentation losses (if provided)
        if y_seg is not None:
            seg_ls = self._seg_losses(out["seg_logits"], y_seg, mask=mask)
            loss_seg = seg_ls["bce"] + self.hparams.lambda_dice * seg_ls["dice"]
        else:
            seg_ls = {"bce": torch.tensor(0.0, device=self.device), "dice": torch.tensor(0.0, device=self.device)}
            loss_seg = torch.tensor(0.0, device=self.device)

        loss = loss_cls + loss_seg

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_cls": loss_cls,
                "train/loss_seg": loss_seg,
                "train/ce": cls_ls["ce"],
                "train/focal": cls_ls["focal"],
                "train/softf1": cls_ls["softf1"],
                "train/bce": seg_ls["bce"],
                "train/dice": seg_ls["dice"],
            },
            prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        x, mask = batch["x"], batch.get("mask", None)
        y_class = batch["y_class"]
        y_seg = batch.get("y_seg", None)

        out = self(x, mask=mask)
        cls_ls = self._cls_losses(out["cls_logits_cond"], y_class)
        loss_cls = cls_ls["ce"] + self.hparams.lambda_focal * cls_ls["focal"] + self.hparams.lambda_softf1 * cls_ls["softf1"]

        if y_seg is not None:
            seg_ls = self._seg_losses(out["seg_logits"], y_seg, mask=mask)
            loss_seg = seg_ls["bce"] + self.hparams.lambda_dice * seg_ls["dice"]
        else:
            seg_ls = {"bce": torch.tensor(0.0, device=self.device), "dice": torch.tensor(0.0, device=self.device)}
            loss_seg = torch.tensor(0.0, device=self.device)

        loss = loss_cls + loss_seg

        # simple metrics
        preds = out["p_class"].argmax(dim=-1)  # argmax over final P(class)
        acc = (preds == y_class).float().mean()

        self.log_dict(
            {
                "val/loss": loss,
                "val/loss_cls": loss_cls,
                "val/loss_seg": loss_seg,
                "val/acc": acc,
            },
            prog_bar=True, on_epoch=True
        )
        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        x, mask = batch["x"], batch.get("mask", None)
        out = self(x, mask=mask)
        # Final class probabilities already gated by P(target)
        # If you also need P(target | sequence) separately, it's out["p_target_seq"]
        return {"p_class": out["p_class"], "p_target": out["p_target_seq"]}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 50)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
```

### 使い方（最小例）

```python
import torch
from torch.utils.data import DataLoader

# ダミーデータ
B, C, T = 8, 6, 1024
x = torch.randn(B, C, T)
mask = torch.ones(B, T, dtype=torch.bool)         # 可変長なら適宜 False を混ぜる
y_class = torch.randint(0, 18, (B,))
y_seg = (torch.rand(B, T) > 0.8).float()          # frame-wise gestureラベル例

batch = {"x": x, "mask": mask, "y_class": y_class, "y_seg": y_seg}
loader = DataLoader([batch]*50, batch_size=None)  # デモ用

model = IMUSegClassModel(in_chans=C, num_classes=18)
trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1, accelerator="auto")
trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
```

### 実運用メモ

* 入力は `(B,C,T)` 固定。可変長は `mask` で対応（`True=有効`）。
    
* 推論では `predict_step` が `{"p_class": (B,C), "p_target": (B,)}` を返します。`p_class` はすでに `P(class|target)×P(target)` のゲート後。
    
* 学習で frame-wise ラベルが無い場合は `y_seg=None` にし、`lambda_dice` を 0 に。
    
* クラス不均衡が強い場合は `use_class_weights`（`torch.tensor([...])`）や `FocalLoss` の `alpha` を指定。
    
* STFTブランチ等を併設する場合は `forward_features` 前後で特徴を結合してください（例：`feats = torch.cat([feats, feats_stft], dim=-1)` など）。
    

この骨格をベースに、前処理（重力除去・角速度導出）、SO(3)回転Aug、TTA、閾値最適化を足せばLB 0.8台を現実的に狙えます。必要なら DataModule＋前処理ユーティリティも続けて用意します。

* * *

**You:**

conv1dをconvnext1dにして

* * *

**ChatGPT:**

了解。ConvNeXtの設計（DepthwiseConv→LayerNorm→Pointwise MLP→GELU＋DropPath 残差）を1Dへ移植して、元のLightningモデルのConv1d/ResBlock部を「ConvNeXt1Dブロック」に差し替えた実装例です（Stochastic Depthは`torchvision.ops.stochastic_depth`を使用）。ブロック構成は公式実装/解説に準拠しています。[PyTorch](https://pytorch.org/vision/main/models/convnext.html?utm_source=chatgpt.com)[PyTorch Documentation](https://docs.pytorch.org/vision/main/generated/torchvision.ops.stochastic_depth.html?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/convnext?utm_source=chatgpt.com)

```python
# pip install torch pytorch-lightning torchvision
from typing import Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from torchvision.ops import stochastic_depth  # Stochastic Depth (DropPath)
except Exception:
    # Fallback (scale_by_keep=True 相当)
    def stochastic_depth(x, p: float, mode: str, training: bool = True):
        if (not training) or p == 0.0:
            return x
        keep = 1.0 - p
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device).bernoulli_(keep)
        return x / keep * mask

# -----------------------
# Loss utilities (そのまま)
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

class SoftF1Loss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        y = F.one_hot(target, num_classes=p.size(-1)).float()
        tp = (p * y).sum(0); fp = (p * (1 - y)).sum(0); fn = ((1 - p) * y).sum(0)
        prec = tp / (tp + fp + self.eps); rec = tp / (tp + fn + self.eps)
        f1 = 2 * prec * rec / (prec + rec + self.eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1.0 - f1.mean()

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask; target = target * mask
        inter = (probs * target).sum(1)
        denom = (probs.square().sum(1) + target.square().sum(1) + self.eps)
        dice = 2 * inter / denom
        return 1.0 - dice.mean()

# -----------------------
# ConvNeXt1D Blocks
# -----------------------
class LayerNorm1d(nn.Module):
    """LayerNorm for (B, C, T) by applying LN over channel dim in channels-last."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x_perm = x.transpose(1, 2)          # (B, T, C)
        x_norm = self.ln(x_perm)
        return x_norm.transpose(1, 2)       # (B, C, T)

class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt block (1D):
      - Depthwise Conv1d (groups=C, large kernel)
      - LayerNorm (channels-last)
      - Pointwise MLP: Linear( C -> 4C ) + GELU + Linear( 4C -> C )
      - Optional LayerScale (gamma) and StochasticDepth
    """
    def __init__(self, channels: int, kernel_size: int = 7, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        pad = kernel_size // 2
        self.dwconv = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.norm = LayerNorm1d(channels)
        self.pw1 = nn.Conv1d(channels, 4 * channels, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(4 * channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)                # (B, C, T)
        x = self.pw1(x); x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            # apply per-channel scale: (B,C,T) * (C,)
            x = x * self.gamma.view(1, -1, 1)
        x = stochastic_depth(x, p=self.drop_path, mode="row", training=self.training) + shortcut
        return x

class ConvNeXt1DStage(nn.Module):
    """Stack of ConvNeXt1DBlock with drop-path schedule."""
    def __init__(self, channels: int, depth: int, kernel_size: int = 7, drop_path_rate: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.Sequential(*[
            ConvNeXt1DBlock(channels, kernel_size=kernel_size, drop_path=dpr[i], layer_scale_init=layer_scale_init)
            for i in range(depth)
        ])
    def forward(self, x): return self.blocks(x)

# -----------------------
# Pooling
# -----------------------
class WeightedGAP(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None: w = w * mask.float()
        denom = w.sum(1, keepdim=True).clamp_min(1e-6)
        return (x * w.unsqueeze(1)).sum(2) / denom  # (B, C, T)->(B,C)

# -----------------------
# LightningModule with ConvNeXt1D backbone
# -----------------------
class IMUSegClassModel(pl.LightningModule):
    """
    Single model:
      - ConvNeXt1D stem + stage  (保持: 時系列長は変えない)
      - Transformer encoder (mask-aware)
      - Segmentation head (frame-wise gesture)
      - Classification head with WeightedGAP (seg prob as weights)
    """
    def __init__(
        self,
        in_chans: int = 6,
        num_classes: int = 18,
        emb_dim: int = 256,
        convnext_depth: int = 6,           # ConvNeXt1D blocks数
        convnext_kernel: int = 7,
        convnext_drop_path: float = 0.2,   # 線形スケジュールの最大値
        layer_scale_init: float = 1e-6,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        lambda_focal: float = 0.5,
        lambda_softf1: float = 0.5,
        lambda_dice: float = 0.5,
        use_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["use_class_weights"])
        self.num_classes = num_classes
        self.lr = lr; self.weight_decay = weight_decay

        # --- Stem: project to emb_dim (stride=1 to keep T for segmentation) ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, emb_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )

        # --- ConvNeXt1D Stage ---
        self.cnx = ConvNeXt1DStage(
            channels=emb_dim, depth=convnext_depth,
            kernel_size=convnext_kernel,
            drop_path_rate=convnext_drop_path,
            layer_scale_init=layer_scale_init
        )

        # --- Transformer (sequence modeling, mask-aware) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.posenc = self._build_posenc(emb_dim)

        # --- Heads ---
        self.seg_head = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(emb_dim // 2, 1, kernel_size=1)     # -> (B,1,T)
        )
        self.pool = WeightedGAP()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

        # --- Losses ---
        class_weight = use_class_weights if use_class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=class_weight)
        self.focal = FocalLoss(gamma=2.0, alpha=class_weight)
        self.softf1 = SoftF1Loss()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()

    # Positional encoding (sin/cos)
    def _build_posenc(self, d_model: int, max_len: int = 20000) -> nn.Parameter:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, D)

    def forward_features(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T) -> feats: (B, T, D)
        h = self.stem(x)              # (B, D, T)
        h = self.cnx(h)               # (B, D, T)
        h = h.transpose(1, 2)         # (B, T, D)
        h = h + self.posenc[:, :h.size(1), :]
        key_padding_mask = (~mask) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, T, D)
        return h

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feats = self.forward_features(x, mask)              # (B, T, D)
        seg_logits = self.seg_head(feats.transpose(1, 2)).squeeze(1)  # (B, T)
        p_seg = torch.sigmoid(seg_logits)
        if mask is not None:
            p_target_seq = (p_seg * mask.float()).sum(1) / mask.float().sum(1).clamp_min(1e-6)
        else:
            p_target_seq = p_seg.mean(1)
        pooled = self.pool(feats.transpose(1, 2), w=p_seg, mask=mask)  # (B, D)
        cls_logits_cond = self.cls_head(pooled)
        p_class = F.softmax(cls_logits_cond, dim=-1) * p_target_seq.unsqueeze(-1)
        return {"seg_logits": seg_logits, "cls_logits_cond": cls_logits_cond, "p_target_seq": p_target_seq, "p_class": p_class}

    # ---- train/val loop (同じ) ----
    def _seg_losses(self, seg_logits, y_seg, mask):
        bce = self.bce(seg_logits, y_seg); bce = (bce * (mask.float() if mask is not None else 1.)).sum() / (mask.float().sum().clamp_min(1e-6) if mask is not None else bce.numel())
        dice = self.dice(seg_logits, y_seg, mask=mask)
        return {"bce": bce, "dice": dice}

    def _cls_losses(self, cls_logits_cond, y_class):
        ce = self.ce(cls_logits_cond, y_class); focal = self.focal(cls_logits_cond, y_class); softf1 = self.softf1(cls_logits_cond, y_class)
        return {"ce": ce, "focal": focal, "softf1": softf1}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x, mask = batch["x"], batch.get("mask", None)
        y_class, y_seg = batch["y_class"], batch.get("y_seg", None)
        out = self(x, mask=mask)
        cls_ls = self._cls_losses(out["cls_logits_cond"], y_class)
        loss_cls = cls_ls["ce"] + self.hparams.lambda_focal * cls_ls["focal"] + self.hparams.lambda_softf1 * cls_ls["softf1"]
        if y_seg is not None:
            seg_ls = self._seg_losses(out["seg_logits"], y_seg, mask=mask)
            loss_seg = seg_ls["bce"] + self.hparams.lambda_dice * seg_ls["dice"]
        else:
            seg_ls = {"bce": torch.tensor(0.0, device=self.device), "dice": torch.tensor(0.0, device=self.device)}
            loss_seg = torch.tensor(0.0, device=self.device)
        loss = loss_cls + loss_seg
        self.log_dict({"train/loss": loss, "train/ce": cls_ls["ce"], "train/focal": cls_ls["focal"], "train/softf1": cls_ls["softf1"], "train/bce": seg_ls["bce"], "train/dice": seg_ls["dice"]}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        x, mask = batch["x"], batch.get("mask", None)
        y_class, y_seg = batch["y_class"], batch.get("y_seg", None)
        out = self(x, mask=mask)
        cls_ls = self._cls_losses(out["cls_logits_cond"], y_class)
        loss_cls = cls_ls["ce"] + self.hparams.lambda_focal * cls_ls["focal"] + self.hparams.lambda_softf1 * cls_ls["softf1"]
        if y_seg is not None:
            seg_ls = self._seg_losses(out["seg_logits"], y_seg, mask=mask)
            loss_seg = seg_ls["bce"] + self.hparams.lambda_dice * seg_ls["dice"]
        else:
            seg_ls = {"bce": torch.tensor(0.0, device=self.device), "dice": torch.tensor(0.0, device=self.device)}
            loss_seg = torch.tensor(0.0, device=self.device)
        loss = loss_cls + loss_seg
        preds = out["p_class"].argmax(-1); acc = (preds == y_class).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        x, mask = batch["x"], batch.get("mask", None)
        out = self(x, mask=mask)
        return {"p_class": out["p_class"], "p_target": out["p_target_seq"]}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 50)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
```

### 変更点の要旨

* **ConvNeXt1Dブロック**を追加（DepthwiseConv1d→LN→1×1 Conv(MLP)→GELU→LayerScale→DropPath残差）。ConvNeXtの基本設計に基づく置換です。[PyTorch](https://pytorch.org/vision/main/models/convnext.html?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/convnext?utm_source=chatgpt.com)
    
* **Stochastic Depth**は`torchvision.ops.stochastic_depth`を使用し、ブロック深さに沿って線形に`drop_path`を増やしています。[PyTorch Documentation](https://docs.pytorch.org/vision/main/generated/torchvision.ops.stochastic_depth.html?utm_source=chatgpt.com)
    

必要なら、**パッチ化ダウンサンプリング**（stride>1）や**多段ステージ化**（Tiny/Small相当の深さ配分）にも拡張できますが、今回は**セグメンテーションの時間解像度を保持**するため、すべてstride=1のままにしています（ConvNeXtの一般的なブロック構成はそのまま）。[PyTorch](https://pytorch.org/vision/main/models/convnext.html?utm_source=chatgpt.com)[Medium](https://medium.com/augmented-startups/convnext-the-return-of-convolution-networks-e70cbe8dabcc?utm_source=chatgpt.com)

このまま差し替えで動きます。カーネル（`convnext_kernel`）、深さ（`convnext_depth`）、DropPath（`convnext_drop_path`）はお好みで。