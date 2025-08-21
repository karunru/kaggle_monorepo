**You:**

data_description.md
ファイル
competition_overview.md
ファイル
codes/exp/exp039/model.py
Python
on_validation_epoch_end とか最終的なジェスチャーを出力するときに、現状はmulticlass_preds = torch.argmax(all_multiclass_probs, dim=-1)を使っているけど、all_multiclass_probs, all_binary_probs, all_orientation_probs, nine_class_probsを組み合わせて精度を上げる方法を考えて

* * *

**ChatGPT:**

いいね、その4ヘッドを“素で argmax”せずに噛み合わせると、CMI指標を直接押し上げやすい。以下の順で入れ替えてみて。

# 1) まず全ヘッドを確率校正（軽量）

* multiclass・nine-class・binary・orientation の各確率に **温度スケーリング**（クラス共有TでOK）。検証foldでNLL最小化でTを1変数最適化。これで後段のしきい値探索や重み付けが安定する。温度スケーリングは“予測ラベルは変えずに確率だけ直す”のが利点。[arXiv](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com)
    
* scikit-learnの `CalibrationDisplay` で信頼度曲線チェック（sigmoid/等幅binningも比較）。[scikit-learn.org](https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com)
    

# 2) 評価指標に合う推論フロー（two-stage + しきい値最適化）

CMIの最終スコアは **Binary F1 と 9クラスMacro F1 の平均** なので、推論は二段構えが定石。

**Stage A: TargetかNon-targetか**

* binaryヘッドの校正確率 $p_b$ に対して、検証で **F1最大化しきい値** $t_b$ を決める（単一閾値）。理論的にもF1最適化は一般の0.5閾値と一致しない。[arXiv](https://arxiv.org/abs/1402.1892?utm_source=chatgpt.com)[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/?utm_source=chatgpt.com)
    
* $p_b < t_b$ なら **即 `non_target` を出す**（ここでMacro F1側の`non_target`が効く）。
    

**Stage B: ターゲット8クラスの中で選ぶ**

* nine-classヘッドのターゲット8クラス確率 $p_{9,c}$ と、multiclassヘッドの18クラス確率 $p_{m,c}$（ターゲット8クラスに絞る）を融合して1つの posterior を作る（やり方は #3）。
    
* さらに **各ターゲットに個別しきい値 $t_c$** を用意（検証でMacro F1最大化）。上位クラス $c^*$ が $p(c^*) \ge t_{c^*}$ を満たさないなら **`non_target`** にフォールバック（外しを減らしBinary F1も守る）。F1最適化のしきい値設計は理論・実務の両面で裏付けあり。[arXiv](https://arxiv.org/abs/1402.1892?utm_source=chatgpt.com)
    

# 3) スコア融合（logit和 + orientationのソフト前処理）

複数分類器の確率は **“和ルール/積ルール”** が古典的に強い。logitにして重み付き和→softmax が扱いやすい。[SpringerLink](https://link.springer.com/article/10.1007/BF01238023?utm_source=chatgpt.com)[decom.ufop.br](https://www.decom.ufop.br/menotti/rp102/slides/14-1998-CombiningClassifiers-PAAv1n1pp1827.pdf?utm_source=chatgpt.com)

* 二段目で使う **ターゲット8クラス** について
    
    $$\textstyle s_c \;=\; w_m \,\log p_{m,c} \;+\; w_9 \,\log p_{9,c} \;+\; w_b \,\log p_b \;+\; w_o \,\log \pi(c\,|\,\hat{o})$$
    * $p_b$: binaryの“target”確率（全ターゲット共通のゲート）
        
    * $\pi(c\,|\,\hat{o})$: **orientation依存の事前**。orientationヘッドの予測 $\hat{o}$ を使い、学習データの共起から $p(o|c)$ を数え上げて “あり得るクラスに+、起きにくいクラスに−” のソフト重みを掛ける（ゼロは危険なのでεで下限）。本データは **4姿勢で実施** されていて、姿勢限定のジェスチャーもあるため、ここが効く。
        
    * non-target については
        
        $$s_{\text{non}} = w_9 \,\log p_{9,\text{non}} \;+\; w_b \,\log(1-p_b)$$
* 重み $w_m,w_9,w_b,w_o$ は検証で **CMIスコア** 最大化になるように（シンプルにグリッド/ベイズ最適化で良い）。CMI計算はあなたの `compute_cmi_score` をそのまま使える。
    
* orientationヘッドは既に4クラスで実装済み（model.pyのヘッド定義）。
    

# 4) もう一段攻める：**スタッキング**（軽いメタ分類器）

上のルールベースでも伸びるが、さらに検証OOFで

$$[p_{m,c},\,p_{9,c},\,p_b,\,p_{o,1..4}]$$

を特徴に **9クラスのロジスティック回帰**（L2弱正則化）を学習＝**Stacked Generalization**。確率校正にもなり、重み学習を自動化できる。理論と実証の古典。[サイエンスダイレクト](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231?utm_source=chatgpt.com)[ijcai.org](https://www.ijcai.org/Proceedings/97-2/Papers/011.pdf?utm_source=chatgpt.com)

# 5) 実装の“置き換え”例（Lightningの `on_validation_epoch_end` 想定）

```python
# 事前に：各ヘッドを温度スケーリングで校正済み (T_m, T_9, T_b, T_o)
# 事前に：p(o|c) の表 (train共起から) と、しきい値 {t_b, t_c}、重み {w_m,w_9,w_b,w_o} を検証で確定

def fuse_and_predict(all_multiclass_probs, all_binary_probs, all_orientation_probs, nine_class_probs):
    eps = 1e-8
    # logits
    log_m = (all_multiclass_probs.clamp_min(eps)).log()              # [N,18]
    log_9 = (nine_class_probs.clamp_min(eps)).log()                  # [N,9]
    log_pb = (all_binary_probs.clamp_min(eps)).log()                 # [N,1]
    log_pnb = (1 - all_binary_probs).clamp_min(eps).log()            # [N,1]

    # orientation prior π(c|o_hat) をsoft適用
    o_hat = all_orientation_probs.argmax(dim=-1)                     # [N]
    pi = prior_pi_c_given_o[o_hat]                                   # [N, 9]  # 9列: 8 target + non
    log_pi = (pi.clamp_min(1e-3)).log()                              # ゼロ殺し

    # ターゲット8クラスのスコア
    s_target = w_m*log_m[:, target_idx18] + w_9*log_9[:, target_idx9] + w_b*log_pb + w_o*log_pi[:, target_idx9]
    # non_target のスコア
    s_non = w_9*log_9[:, non_idx9] + w_b*log_pnb
    s9 = torch.cat([s_target, s_non], dim=1)                         # [N,9]

    # softmax後、しきい値適用
    p9 = s9.log_softmax(dim=1).exp()
    top_p, top_i = p9.max(dim=1)
    # 二段判定
    pred = torch.full((p9.size(0),), fill_value=non_idx9, dtype=torch.long, device=p9.device)
    # Stage A
    pred[all_binary_probs.squeeze(-1) >= t_b] = top_i[all_binary_probs.squeeze(-1) >= t_b]
    # Stage B: ターゲットしきい値
    mask_t = pred != non_idx9
    ok = top_p[mask_t] >= per_class_thresholds[top_i[mask_t]]
    pred[mask_t.nonzero()[~ok]] = non_idx9

    return pred  # 9クラスID（nonは1つ）。最終的に18ラベルへマッピングして出力
```

> 注1: 検証で `w_*` と `t_b, t_c` を **CMIスコア** 最大化で同時/交互に最適化。  
> 注2: orientationは **ハードなマスク**（不可能クラス=確率0）にせず **ソフト** で掛ける（誤検出のリスク緩和）。本データは4つの姿勢で実施され、姿勢はメタ情報として与えられている。

# 6) 代替：メタ学習（スタッキング）に丸投げ

上の手続きを **特徴→ロジスティック回帰** に置換して同等の挙動を学習させても良い（OOFで学習→本番foldで推論）。理屈はスタッキング。[サイエンスダイレクト](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231?utm_source=chatgpt.com)

# 7) 仕上げの小技

* **クラス別温度**（vector scaling）は過学習しやすいので、データ量次第で基本は温度一つ。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com)
    
* 9クラスヘッドは **評価に直結**（8 target + `non_target`）。multiclassヘッド（18クラス）は **ターゲット内での識別器** として寄与させる。コンペ設計的にも9クラス軸が自然。
    
* 提示の4ヘッド（multiclass/binary/nine/orientation）は model.py で同じ特徴から出ているので、logit和は安定して効く。
    

* * *

この方針は、(1) 確率を直して（校正）、(2) 指標に合わせて**ゲート＋閾値**で意思決定し、(3) スコアは**理にかなったルール（logit和）**または**軽いメタ学習（スタッキング）**で融合、という王道。  
校正→しきい値最適化→融合の各段に理論的裏付けがあり、実装も軽いのがメリットです。[arXiv+1](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)[SpringerLink](https://link.springer.com/article/10.1007/BF01238023?utm_source=chatgpt.com)

（補足：データは手首装着デバイスHelios、IMU/ToF/サーモと4姿勢・3フェーズの設計。テストはIMUのみのシーケンスも半分ある前提。推論設計の意図付けに。 ）

* * *

**You:**

1234それぞれ個別に実装も交えて説明して

* * *

**ChatGPT:**

了解。前回の「1–4」を、それぞれ“狙い→やり方→実装（コピペ可）”で分解します。形は PyTorch/Lightning 前提、保存・再利用しやすい関数にしてあります。

* * *

# 1) 確率校正（温度スケーリング）

**狙い**  
logit を 1/T で割って softmax/sigmoid するだけの軽量校正。学習済みモデルは固定、T だけ検証で最適化（NLL最小）。ヘッドごとに T を 1 つずつ持つのが基本。

**実装（マルチクラス/バイナリ両対応）**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    """logits -> logits / T"""
    def __init__(self, init_T: float = 1.0, is_binary: bool = False):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor([init_T]).log())
        self.is_binary = is_binary

    @property
    def T(self):
        return self.log_T.exp().clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    @torch.no_grad()
    def transform_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """既に確率しかない場合の救済: probs -> logits -> 温度 -> probs"""
        eps = 1e-8
        if self.is_binary:
            logit = torch.log(probs.clamp(eps, 1-eps) / (1 - probs.clamp(eps, 1-eps)))
            return torch.sigmoid((logit / self.T))
        else:
            logit = torch.log(probs.clamp_min(eps))
            logit = logit - logit.logsumexp(dim=-1, keepdim=True)
            return F.softmax((logit / self.T), dim=-1)

def fit_temperature_scaler(logits: torch.Tensor, y: torch.Tensor, is_binary: bool=False, max_iter: int=2000) -> TemperatureScaler:
    """
    logits: [N, C] (multiclass) or [N] / [N,1] (binary, raw logits)
    y:      LongTensor (multiclass: [N]), FloatTensor 0/1 (binary: [N])
    """
    scaler = TemperatureScaler(is_binary=is_binary).to(logits.device)
    opt = torch.optim.LBFGS([scaler.log_T], lr=0.1, max_iter=256, line_search_fn='strong_wolfe')
    logits = logits.detach()

    def closure():
        opt.zero_grad(set_to_none=True)
        z = scaler(logits)
        if is_binary:
            z = z.view(-1)
            loss = F.binary_cross_entropy_with_logits(z, y.float())
        else:
            loss = F.cross_entropy(z, y.long())
        loss.backward()
        return loss

    prev = 1e9
    for _ in range(max_iter//10):
        loss = opt.step(closure)
        if abs(prev - loss.item()) < 1e-7:
            break
        prev = loss.item()
    return scaler
```

**使い方（例）**

```python
# 検証OOFで1回だけフィットして保存 → 推論時は scaler(logits) 経由で確率化
scaler_multi = fit_temperature_scaler(val_logits_multi, val_y_multi, is_binary=False)
scaler_bin   = fit_temperature_scaler(val_logits_bin.squeeze(-1), val_y_bin, is_binary=True)

# 推論
calib_multi_probs = F.softmax(scaler_multi(test_logits_multi), dim=-1)
calib_bin_probs   = torch.sigmoid(scaler_bin(test_logits_bin).squeeze(-1))
```

* * *

# 2) 指標に合わせた二段推論＋しきい値最適化

**狙い**  
CMIは「Binary F1（target vs non）＋ 9クラスMacro F1」の平均。  
A段（バイナリ）で target 候補を絞り、B段（ターゲット内）でクラス別しきい値をかけて外したら non_target に落とす。

**実装（最適なしきい値探索 & 二段推論）**

```python
import numpy as np
import torch

def best_binary_threshold(y_true_bin: np.ndarray, p_bin: np.ndarray, grid: int=1001) -> float:
    # F1最大化閾値（0~1の等間隔で探索）
    ths = np.linspace(0.0, 1.0, grid)
    best_f1, best_th = -1.0, 0.5
    for t in ths:
        y_pred = (p_bin >= t).astype(int)
        tp = ((y_true_bin==1)&(y_pred==1)).sum()
        fp = ((y_true_bin==0)&(y_pred==1)).sum()
        fn = ((y_true_bin==1)&(y_pred==0)).sum()
        p = tp/(tp+fp+1e-12); r = tp/(tp+fn+1e-12)
        f1 = 2*p*r/(p+r+1e-12)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    return float(best_th)

def best_per_class_thresholds(y_true_9: np.ndarray, top_idx: np.ndarray, top_p: np.ndarray, n_classes: int=9, grid: int=501):
    """
    y_true_9: [N] int in [0..8]
    top_idx, top_p: 9クラス確率の最大クラスとその確率
    クラスごとに "top_idx==c かつ top_p >= t なら c、それ以外 non" の一対他F1で閾値を最適化
    """
    ths = np.linspace(0.0,1.0,grid)
    per_th = np.zeros(n_classes) + 0.5
    for c in range(n_classes-1):  # ここでは [0..7] をターゲット、8を non_target と想定
        mask = (top_idx==c)
        best_f1, best_t = -1.0, 0.5
        for t in ths:
            y_pred = np.full_like(y_true_9, fill_value=(n_classes-1))  # non
            y_pred[mask & (top_p>=t)] = c
            # c vs rest のF1
            tp = ((y_true_9==c)&(y_pred==c)).sum()
            fp = ((y_true_9!=c)&(y_pred==c)).sum()
            fn = ((y_true_9==c)&(y_pred!=c)).sum()
            p = tp/(tp+fp+1e-12); r = tp/(tp+fn+1e-12)
            f1 = 2*p*r/(p+r+1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        per_th[c] = best_t
    return per_th  # non側(8)は未使用

def two_stage_decision(p_bin: torch.Tensor, p9: torch.Tensor, t_b: float, per_class_t: np.ndarray, non_idx9: int):
    """
    p_bin: [N] target確率（校正済）
    p9:    [N,9] 9クラス確率（融合後; 後述 #3）
    """
    top_p, top_i = p9.max(dim=1)
    pred = torch.full((p9.size(0),), fill_value=non_idx9, dtype=torch.long, device=p9.device)

    # Stage A: binaryゲート
    gate = (p_bin >= t_b)
    pred[gate] = top_i[gate]

    # Stage B: ターゲットクラスの個別しきい値
    if isinstance(per_class_t, torch.Tensor):
        per_class_t = per_class_t.cpu().numpy()
    top_p_np = top_p.detach().cpu().numpy()
    top_i_np = top_i.detach().cpu().numpy()
    per_class_t = per_class_t.astype(np.float64)

    mask = gate.cpu().numpy()
    for i in np.where(mask)[0]:
        c = top_i_np[i]
        if c != non_idx9 and top_p_np[i] < per_class_t[c]:
            pred[i] = non_idx9
    return pred
```

**手順**

1. 検証OOFで `t_b = best_binary_threshold(y_true_bin, p_bin)` を求める。
    
2. （#3 の融合後の）`p9` から `top_idx, top_p` を出し、`per_class_t = best_per_class_thresholds(...)` を求める。
    
3. 本番推論は `two_stage_decision(p_bin, p9, t_b, per_class_t, non_idx9=8)`。
    

* * *

# 3) スコア融合（logit和＋orientation事前のソフト掛け）

**狙い**  
複数ヘッド（multiclass 18、nine 9、binary 2、orientation 4）を **logit の加重和** で一つの 9クラス分布にまとめる。orientation は「その姿勢で起きやすいジェスチャー」を **事前確率 π(c|o)** として緩く反映（ゼロマスクは危険なのでε下限）。

**実装（事前の構築＋融合関数）**

```python
import torch

def build_orientation_prior(counts_c_given_o: torch.Tensor, smoothing: float=1.0) -> torch.Tensor:
    """
    counts_c_given_o: [O, 9]  例: O=4 (姿勢), 9クラス(8 target + non)
    出力: π(c|o) = normalized counts
    """
    prior = counts_c_given_o + smoothing
    prior = prior / prior.sum(dim=1, keepdim=True)
    return prior  # [O,9]

def fuse_to_9class(p_multi_18: torch.Tensor,
                   p_nine_9: torch.Tensor,
                   p_bin: torch.Tensor,                      # target 確率 [N] or [N,1]
                   p_orient_4: torch.Tensor,                 # [N,4]
                   target_idx18: torch.Tensor,               # 長さ8: 18内のターゲット列index
                   target_idx9: torch.Tensor,                # 長さ8: 9内のターゲット列index (通常 [0..7])
                   non_idx9: int,                            # 9内の non列index (通常 8)
                   orient_prior: torch.Tensor,               # [4,9] π(c|o)
                   weights=(1.0, 1.0, 0.5, 0.25)):           # (w_m, w_9, w_b, w_o)
    """
    返り値: p9 [N,9]（softmaxで正規化済）
    """
    eps = 1e-8
    w_m, w_9, w_b, w_o = weights

    # log
    log_m = torch.log(p_multi_18.clamp_min(eps))
    log_9 = torch.log(p_nine_9.clamp_min(eps))
    log_pb = torch.log(p_bin.view(-1,1).clamp_min(eps))
    log_pnb = torch.log((1 - p_bin.view(-1,1)).clamp_min(eps))

    # orientation: o* = argmax p(o)
    o_hat = p_orient_4.argmax(dim=1)                                # [N]
    log_pi = torch.log(orient_prior.clamp_min(1e-3))                 # [4,9]
    log_pi_n = log_pi[o_hat]                                         # [N,9]

    # ターゲット8: 18の該当列 + 9の該当列 + binary(target) + orientation
    s_target = w_m*log_m[:, target_idx18] + w_9*log_9[:, target_idx9] + w_b*log_pb + w_o*log_pi_n[:, target_idx9]
    # non
    s_non = w_9*log_9[:, non_idx9:non_idx9+1] + w_b*log_pnb

    s = torch.cat([s_target, s_non], dim=1)                          # [N,9]
    p9 = s.log_softmax(dim=1).exp()
    return p9
```

**使い方（例）**

```python
# 事前: 学習データ or OOF から counts_c_given_o をカウント（姿勢×9クラスの共起）
# counts[o, c] = 該当件数
orient_prior = build_orientation_prior(counts_c_given_o)   # [4,9]

# 推論時
p9 = fuse_to_9class(
    p_multi_18=calib_multi_probs,          # #1で校正したもの
    p_nine_9=calib_nine_probs,
    p_bin=calib_bin_probs,
    p_orient_4=calib_orient_probs,         # orientation も温度校正してOK
    target_idx18=torch.tensor(target_idx_18, device=device),
    target_idx9=torch.arange(8, device=device),
    non_idx9=8,
    orient_prior=orient_prior.to(device),
    weights=(w_m, w_9, w_b, w_o)           # 検証で決める。最初は (1,1,0.5,0.25) から
)
```

**補足（重み最適化）**  
`weights` は検証OOFで CMI スコア最大化になるようグリッド/ベイズ探索。最初は粗く `(w_m,w_9)∈{0.5,1,2}`, `(w_b)∈{0.25,0.5,1.0}`, `(w_o)∈{0,0.25,0.5}` などで充分。

* * *

# 4) スタッキング（軽いメタ分類器）

**狙い**  
ルールでなく学習に任せたい場合、OOFの各ヘッド確率を特徴に **9クラスのロジスティック回帰** を学習。確率校正も兼ねる。リーク防止のため **OOFで学習→本番foldに適用** を厳守。

**実装（学習 & 推論）**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load

def stacker_train(oof_p_multi18: np.ndarray,    # [N,18]
                  oof_p_nine9: np.ndarray,      # [N,9]
                  oof_p_bin: np.ndarray,        # [N] (target確率)
                  oof_p_orient4: np.ndarray,    # [N,4]
                  y_true_9: np.ndarray,         # [N] in [0..8]
                  target_idx18: list[int],
                  non_idx9: int=8,
                  C: float=2.0,
                  max_iter: int=2000,
                  seed: int=42,
                 ) -> Pipeline:
    """
    特徴: [p_m[target8], p_9[9], p_b[1], p_o[4]] -> 8+9+1+4 = 22次元
    """
    X = np.concatenate([
        oof_p_multi18[:, target_idx18],  # 8
        oof_p_nine9,                     # 9
        oof_p_bin[:, None],              # 1
        oof_p_orient4,                   # 4
    ], axis=1).astype(np.float32)

    # L2ロジスティック (multinomial, saga) が扱いやすい
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            penalty="l2", C=C, max_iter=max_iter, random_state=seed,
            multi_class="multinomial", solver="saga", n_jobs=-1
        ))
    ])
    clf.fit(X, y_true_9)
    return clf

def stacker_predict(clf: Pipeline,
                    p_multi18: np.ndarray,
                    p_nine9: np.ndarray,
                    p_bin: np.ndarray,
                    p_orient4: np.ndarray,
                    target_idx18: list[int]) -> np.ndarray:
    X = np.concatenate([
        p_multi18[:, target_idx18],
        p_nine9,
        p_bin[:, None],
        p_orient4,
    ], axis=1).astype(np.float32)
    p9 = clf.predict_proba(X)  # [N,9]
    return p9
```

**使い方**

```python
# 学習（検証OOF）:
clf = stacker_train(oof_p_m18, oof_p_9, oof_p_b, oof_p_o4, y_true_9, target_idx18)
dump(clf, "stacker.joblib")

# 本番fold / テストの推論:
clf = load("stacker.joblib")
p9 = stacker_predict(clf, p_m18, p_9, p_b, p_o4, target_idx18)
```

* * *

## まとめ：`on_validation_epoch_end` の置き換え骨子

1. ログ収集：各ヘッドの **logits** と **labels** を回収
    
2. **#1** で各ヘッドに温度 T をフィット（OOFで1回だけ）。以後は logits→scaler→確率化
    
3. **#3** で `p9` を生成（ルール融合 or **#4** のスタッカー出力を使う）
    
4. **#2** で `t_b` と `per_class_t` を確定 → `two_stage_decision` で最終ラベル
    
5. CMI を計算 → 保存
    

※ しきい値・重み・スタッカーは **検証OOFで確定** し、本番推論では固定値を使う（リーク防止）。

必要なら、この骨子をあなたの `LightningModule` の実装に直接埋め込む形で書き換え例も出します。

* * *

**You:**

3) スコア融合 はpytorch-lightningではどのstepで実装したらいい？
元のmodel.pyに追加する形で教えて


* * *

**ChatGPT:**

いい質問。**3) スコア融合**は、検証時は**`on_validation_epoch_end`**（全バッチ結合後）でやるのがベスト。単発推論の導線があるなら**`predict_step`/専用推論関数**でも同じ関数を呼ぶ形にします。  
以下、あなたの `model.py`（`CMISqueezeformer`）に**追記/置換**する最小差分です。

* * *

# 追加①：クラス内に“融合”ユーティリティを生やす

`CMISqueezeformer` クラスのメソッドとして下記を**追記**（`__init__`の後/好きな場所でOK）。

```python
# === スコア融合のための補助 ===
def set_fusion_indices(self, target_idx18: list[int], non_idx9: int = 8):
    """
    18クラス中のターゲット8クラスの列indexと、9クラス側のnon列indexを登録。
    例) target_idx18=[0,1,2,3,4,5,6,7], non_idx9=8 など
    """
    import torch
    self.register_buffer("target_idx18", torch.tensor(target_idx18, dtype=torch.long))
    self.register_buffer("target_idx9", torch.arange(8, dtype=torch.long))
    self.non_idx9 = int(non_idx9)

def set_orientation_prior_from_counts(self, counts_c_given_o: torch.Tensor, smoothing: float = 1.0):
    """
    姿勢oごとのクラスc出現回数から π(c|o) を作って登録。
    counts_c_given_o: [O, 9] (O=姿勢数=4想定, 9=8ターゲット+non)
    """
    prior = counts_c_given_o.float() + smoothing
    prior = prior / prior.sum(dim=1, keepdim=True)
    self.register_buffer("orientation_prior", prior)  # [O,9]

@torch.no_grad()
def fuse_to_9class(
    self,
    p_multi_18: torch.Tensor,    # [N,18]
    p_nine_9: torch.Tensor,      # [N,9]
    p_bin: torch.Tensor,         # [N] or [N,1]  (target確率)
    p_orient_4: torch.Tensor,    # [N,4]
    weights = (1.0, 1.0, 0.5, 0.25),  # (w_m, w_9, w_b, w_o)
) -> torch.Tensor:
    """
    4ヘッド確率を logit加重和 + orientation事前で融合して 9クラス確率を返す。
    事前に set_fusion_indices(), set_orientation_prior_from_counts() を設定しておくこと。
    """
    eps = 1e-8
    w_m, w_9, w_b, w_o = weights

    if p_bin.ndim == 2 and p_bin.size(-1) == 1:
        p_bin = p_bin.squeeze(-1)

    # log-prob
    log_m  = torch.log(p_multi_18.clamp_min(eps))                  # [N,18]
    log_9  = torch.log(p_nine_9.clamp_min(eps))                    # [N,9]
    log_pb = torch.log(p_bin.clamp_min(eps)).unsqueeze(-1)         # [N,1]
    log_pn = torch.log((1 - p_bin).clamp_min(eps)).unsqueeze(-1)   # [N,1]

    # orientation prior π(c|o_hat)
    o_hat  = p_orient_4.argmax(dim=1)                              # [N]
    if not hasattr(self, "orientation_prior"):
        # 事前未設定なら一様で代用
        prior = torch.full((p_orient_4.size(1), 9), 1.0/9, device=p_orient_4.device)
        self.register_buffer("orientation_prior", prior)
    log_pi = torch.log(self.orientation_prior.clamp_min(1e-3))     # [4,9]
    log_pi_n = log_pi[o_hat]                                       # [N,9]

    # 必須インデックス
    assert hasattr(self, "target_idx18") and hasattr(self, "target_idx9"), \
        "set_fusion_indices() で target/non のインデックスを登録してください。"
    non_idx9 = getattr(self, "non_idx9", 8)

    # ターゲット8
    s_target = (
        w_m * log_m[:, self.target_idx18] +
        w_9 * log_9[:, self.target_idx9] +
        w_b * log_pb +                                  # target共通ゲート
        w_o * log_pi_n[:, self.target_idx9]
    )
    # non_target
    s_non = w_9 * log_9[:, non_idx9:non_idx9+1] + w_b * log_pn

    s = torch.cat([s_target, s_non], dim=1)            # [N,9]
    return s.log_softmax(dim=1).exp()
```

> 最低限、学習前/検証前のどこかで  
> `self.set_fusion_indices(target_idx18=[…], non_idx9=8)` を一度呼んでおいてください。  
> 姿勢の事前 `π(c|o)` を使うなら、OOFか学習データからカウントして  
> `self.set_orientation_prior_from_counts(counts)` を1回呼べばOK（未設定なら一様）。

* * *

# 追加②：`validation_step` は変更不要（確率を溜めているのでOK）

あなたの `validation_step` はすでに  
`"multiclass_probs" / "nine_class_probs" / "binary_probs" / "orientation_probs"` を `self.validation_outputs` に詰めています。  
ここはそのままでOKです。

* * *

# 置換：`on_validation_epoch_end` の最終予測を“融合版”に差し替え

`on_validation_epoch_end` 内の

```python
# (元) マルチクラス argmax による予測
multiclass_preds = torch.argmax(all_multiclass_probs, dim=-1)
```

を **以下に置換**。バッチ結合後に一気に融合→9クラス確率→予測ラベルを出します。

```python
# すべての出力を結合（nine_class を取り出していなければ追記）
all_nine_class_probs = torch.cat([x["nine_class_probs"] for x in self.validation_outputs])

# 融合 (logit和 + orientation 事前)
fused_p9 = self.fuse_to_9class(
    p_multi_18      = all_multiclass_probs.to(self.device),
    p_nine_9        = all_nine_class_probs.to(self.device),
    p_bin           = all_binary_probs.to(self.device),
    p_orient_4      = all_orientation_probs.to(self.device),
    # weights=(1.0, 1.0, 0.5, 0.25),  # 必要ならここで上書き
)

# 9クラスの最終予測（8ターゲット + non）
fused_preds_9 = fused_p9.argmax(dim=1).cpu()
```

以降の**ジェスチャー名への変換/CMI集計**は、いままで `multiclass_preds` を使っていた箇所を `fused_preds_9` に置き換えてください（`id_to_gesture` の仕様に合わせて必要なら 9→18 のマッピングを噛ませるだけです）。

* * *

# （任意）単発推論でも使いたい場合

推論用の小関数をクラスに**追記**し、エクスポート・提出コードでも再利用できます。

```python
@torch.no_grad()
def infer_fused_p9(self, imu, attention_mask=None, demographics=None, weights=(1.0,1.0,0.5,0.25)):
    """バッチ入力に対して融合済み9クラス確率を返す"""
    multiclass_logits, binary_logits, nine_class_logits, orientation_logits = self(
        imu, attention_mask, demographics
    )
    p_m  = F.softmax(multiclass_logits, dim=-1)
    p_b  = torch.sigmoid(binary_logits.squeeze(-1))
    p_9  = F.softmax(nine_class_logits, dim=-1)
    p_o  = F.softmax(orientation_logits, dim=-1)
    return self.fuse_to_9class(p_m, p_9, p_b, p_o, weights=weights)
```

* * *

## 最後に（使い方の流れ）

1. どこか1回だけ
    

```python
self.set_fusion_indices(target_idx18=[…], non_idx9=8)
# 使うなら
self.set_orientation_prior_from_counts(counts_c_given_o)  # [4,9]
```

2. 検証ループでは何も変えず `validation_step` に確率を貯める
    
3. `on_validation_epoch_end` で上記の**融合→予測**に置換（CMI集計は既存コード流用）
    

この差分だけで、「各ヘッドの確率を活かした**融合推論**」に切り替わります。

* * *

**You:**

counts_c_given_oって何？ どこで計算するの

* * *

**ChatGPT:**

`counts_c_given_o` は、**「姿勢 o ごとに 9クラス c が何回出たか」**の集計表です。行=姿勢（O=4）、列=9クラス（8 target + non）で、形状は **[O, 9]**。これを確率に正規化して $\pi(c|o)$（orientation 事前）として使います。あなたのモデルは orientation 4クラスヘッドを持っているので（4 orientations）、ここで作る $\pi(c|o)$ が融合(#3)で効きます。また 18→9 への集約は「0–7=targetのまま、8–17をnonに和」なので、この規則で9クラスに落とします。  
データは4つの体位（姿勢）で実験されている前提です。

* * *

## どこで計算する？

**リークを避けるため各foldの“学習データ部分”から作る**のが推奨です。Lightningなら次のいずれか。

* `LightningModule.on_fit_start`（各fold開始時に1回）
    
* `LightningDataModule.setup("fit")`（学習DataFrameがあるならここで前計算）
    

ゼロ出現の列がある（体位限定ジェスチャーがある）ので、**+1の平滑化**は忘れずに。

* * *

## 実装例（pandasの学習DataFrameがある場合）

```python
# 例: train_df には 18クラスID列 'gesture_id_18' と、
#     学習時に使っている orientation の数値ID列 'orientation_id' (0..3) があるとする
import numpy as np
import pandas as pd
import torch

def gesture18_to9(arr18: np.ndarray) -> np.ndarray:
    # 0-7: そのまま, 8-17: non(=8)
    y9 = np.where(arr18 < 8, arr18, 8).astype(int)
    return y9

def make_counts_c_given_o(train_df: pd.DataFrame,
                          y18_col="gesture_id_18",
                          o_col="orientation_id",
                          n_orientations: int = 4) -> torch.Tensor:
    y9 = gesture18_to9(train_df[y18_col].to_numpy())
    o  = train_df[o_col].to_numpy().astype(int)
    # 集計
    counts = np.zeros((n_orientations, 9), dtype=np.int64)
    for oo, cc in zip(o, y9):
        counts[oo, cc] += 1
    return torch.tensor(counts, dtype=torch.float32)  # [O,9]

# --- LightningModule で一度だけ登録 ---
def on_fit_start(self):
    dm = self.trainer.datamodule
    # 各foldの train 部分のDFを想定（自前のデータ構造に合わせて差し替え）
    train_df = dm.train_df  # 例
    counts = make_counts_c_given_o(train_df, y18_col="gesture_id_18", o_col="orientation_id")
    # +1平滑化と正規化は set_orientation_prior_from_counts 側でやる設計にしていた
    self.set_orientation_prior_from_counts(counts, smoothing=1.0)
```

> DataFrameがなく、Datasetだけなら `train_dataloader().dataset` から `gesture_id_18` と `orientation_id` を取り出して同様に数えるだけです。

* * *

## Dataset（配列）から作る場合

```python
import numpy as np
import torch

def counts_from_arrays(y18: np.ndarray, o: np.ndarray, n_orient=4) -> torch.Tensor:
    y9 = np.where(y18 < 8, y18, 8).astype(int)
    counts = np.zeros((n_orient, 9), dtype=np.int64)
    for oo, cc in zip(o.astype(int), y9):
        counts[oo, cc] += 1
    return torch.tensor(counts, dtype=torch.float32)

def on_fit_start(self):
    ds = self.trainer.datamodule.train_dataloader().dataset
    y18 = np.array([ex["gesture_id_18"] for ex in ds])
    o   = np.array([ex["orientation_id"] for ex in ds])
    counts = counts_from_arrays(y18, o, n_orient=4)
    self.set_orientation_prior_from_counts(counts, smoothing=1.0)
```

* * *

## 使い方（復習）

1. 学習開始時に上のいずれかで
    

```python
self.set_fusion_indices(target_idx18=[0,1,2,3,4,5,6,7], non_idx9=8)
self.set_orientation_prior_from_counts(counts)  # [4,9]
```

2. 検証・推論では `fuse_to_9class(...)` が自動で $\log \pi(c|o)$ を参照し、9クラス確率を返します。
    

* * *

### 補足

* orientationラベルの**符号化順序（0..3）**は、学習時に orientation ヘッドへ食わせた順と一致させてください（「座位・前傾座位・仰向け・横向き」などの並び）。
    
* 集計は**元データの素の出現回数**でOK（クラス重み付けやサンプリングの影響を混ぜない）。
    
* 体位限定ジェスチャーで 0 カウントが出やすいので、平滑化（`smoothing=1.0`）は必須。
    

これで `counts_c_given_o` の中身と、**どこでどう作るか**が繋がるはず。