**You:**

codes/exp/exp036/model.py
Python
competition_overview.md
ファイル
data_description.md
ファイル
on_validation_epoch_end とか最終的なジェスチャーを出力するときに、現状はmulticlass_preds = torch.argmax(all_multiclass_probs, dim=-1)を使っているけど、all_multiclass_probs, all_binary_probs, nine_class_probsを組み合わせて精度を上げる方法を考えて

* * *

**ChatGPT:**

いいね、その3ヘッドを“階層 + 校正 + 融合”で活かすと、素朴な `argmax(all_multiclass_probs)` よりだいぶ伸びます。要点だけ先に：

* **階層ベイズ融合**：`P(T)`（binary）でターゲット/非ターゲットをゲートし、  
    8ターゲットは `P(i)=P(T)·P(i|T)`（nine-classの8クラスを正規化）、  
    10非ターゲットは `P(j)=(1−P(T))·P(j|¬T)`（18クラス出力から非ターゲット10クラスを正規化）。
    
* **確率の校正**：各ヘッドのロジット/確率を温度スケーリング（簡単・強力）やDirichlet calibrationで補正してから融合。 [arXiv+1](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com)[NeurIPS Papers](https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf?utm_source=chatgpt.com)[University of Bristol](https://research-information.bris.ac.uk/files/219705598/NeurIPS2019_Dirichlet_Calibration_PF.pdf?utm_source=chatgpt.com)
    
* **しきい値最適化**：`P(T)`のしきい値τはF1最大化でバリデーション最適化（Binary F1に直結）。 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1402.1892?utm_source=chatgpt.com)
    
* **一貫性の強制**：和がズレないよう、`∑_i P(i|T)=1`, `∑_j P(j|¬T)=1` を必ず満たす。
    
* **（任意）メタ・スタッキング**：3ヘッドのロジットを結合→L2付き多項ロジット回帰で18クラス最終器を学習（Zadrozny & Elkan系の“確率結合”の現代版）。 [NeurIPS Papers](https://papers.neurips.cc/paper/2122-reducing-multiclass-to-binary-by-coupling-probability-estimates.pdf?utm_source=chatgpt.com)[ACM Digital Library](https://dl.acm.org/doi/10.1145/775047.775151?utm_source=chatgpt.com)
    

この設計は、評価指標（Binary F1 と “8ターゲット + non_target” のMacro F1 の平均）に素直に合います。non_target側に入った時点で9クラス評価では“non_target”扱いになるので、**まず正しく T / ¬T を当てる**ことが特に重要です。

* * *

# 実装レシピ（そのまま差し替えOK）

```python
import torch
import torch.nn.functional as F

EPS = 1e-12

def _safe_softmax(logits, dim=-1, T=1.0):
    # logitsでも確率でもOK（確率ならlogで安定化）
    if logits.min() >= 0.0 and logits.max() <= 1.0 and torch.allclose(logits.sum(dim=dim), torch.ones_like(logits.sum(dim=dim)), atol=1e-3):
        l = torch.log(torch.clamp(logits, EPS, 1.0))  # prob→logit（対数）
    else:
        l = logits
    return F.softmax(l / T, dim=dim)

def fuse_heads(
    all_multiclass_out,  # [N,18]  ロジット or 確率（自動判定）
    binary_out,          # [N,2]   ロジット or 確率（列1がTarget想定）
    nine_class_out,      # [N,9]   ロジット or 確率（8ターゲット + non_target）
    target_idx18,        # 18クラス中のターゲット8クラスのインデックス（list[int]）
    nontarget_idx18,     # 18クラス中の非ターゲット10クラスのインデックス（list[int]）
    nine_non_target_id,  # nine-class の 'non_target' のインデックス（int）
    T18_all=1.0, Tbin=1.0, T9=1.0,  # 温度（バリデーションで推定）
    w_hier=0.6,          # 階層分布と18クラス分布の幾何平均の重み（0.3〜0.8をCVで）
    tau=0.5              # P(T)しきい値（F1最適化）
):
    # 1) 各ヘッドを温度スケーリングで校正
    p18  = _safe_softmax(all_multiclass_out, dim=-1, T=T18_all)         # [N,18]
    pbin = _safe_softmax(binary_out,         dim=-1, T=Tbin)            # [N,2]
    p9   = _safe_softmax(nine_class_out,     dim=-1, T=T9)              # [N,9]

    pT = pbin[:, 1].clamp(EPS, 1 - EPS)  # Target確率
    # 2) 条件付き分布を構成
    p9_targets = p9[:, :].clone()
    p9_targets[:, nine_non_target_id] = 0.0
    p9_targets = p9_targets / (p9_targets.sum(dim=-1, keepdim=True) + EPS)  # P(i|T) for 8 classes（順番合わせ注意）

    p18_nonT = p18[:, nontarget_idx18]
    p18_nonT = p18_nonT / (p18_nonT.sum(dim=-1, keepdim=True) + EPS)       # P(j|¬T) for 10 classes

    # 3) 階層ベイズ分布 P_hier
    N = p18.size(0)
    P_hier = torch.zeros_like(p18)
    # ターゲット8
    for k, cls_id in enumerate(target_idx18):
        # nine_class 側の8クラス順と cls_id の対応だけ合わせればOK
        P_hier[:, cls_id] = pT * p9_targets[:, k]
    # 非ターゲット10
    P_hier[:, nontarget_idx18] = (1.0 - pT[:, None]) * p18_nonT

    # 4) 幾何平均（product-of-experts風）で18クラス分布と融合 → 正規化
    log_p = w_hier * torch.log(P_hier.clamp_min(EPS)) + (1 - w_hier) * torch.log(p18.clamp_min(EPS))
    P_final = torch.softmax(log_p, dim=-1)

    # 5) 最終ラベル（オプションのゲート規則）
    #   まず T/¬T を τ で決め、その側の最大クラスだけに限定して argmax する“強ゲート”も試すと良い
    is_target = pT >= tau
    preds = torch.empty(N, dtype=torch.long)
    # ターゲット側 argmax
    tgt_scores = P_final[:, target_idx18]
    tgt_best = target_idx18[torch.argmax(tgt_scores, dim=-1)]
    # 非ターゲット側 argmax
    nont_scores = P_final[:, nontarget_idx18]
    nont_best = torch.tensor(nontarget_idx18, device=P_final.device)[torch.argmax(nont_scores, dim=-1)]

    preds[is_target] = tgt_best[is_target]
    preds[~is_target] = nont_best[~is_target]
    return P_final, preds
```

> しきい値τは**Binary F1最大化**でホールドアウト/OOFから決めるのが定石。F1は0.5固定では最適になりません。 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1402.1892?utm_source=chatgpt.com)

* * *

## 校正（Temperature / Dirichlet）

* **Temperature scaling**：最後のロジットに1スカラーTを掛けるだけ。NLL最小化でTを学習、クラスは変えず信頼度だけ整うので安全。まずはこれ。 [arXiv](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com)
    
* **Dirichlet calibration**：多クラス用の高性能な後処理。余裕があれば nine-class と 18-class に適用してみる。 [arXiv](https://arxiv.org/abs/1910.12656?utm_source=chatgpt.com)[NeurIPS Papers](https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf?utm_source=chatgpt.com)
    

> 実務ポイント：reliability diagram / ECE を見てから導入可否を判断（過信なら高確率域を冷やすT>1が効きやすい）。 [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com)

* * *

## なぜこの形が効くか

* **評価に整合**：まず T/¬T を当てれば、9クラス側のMacro F1は“non_target”として救われる設計（最終提出は18クラスだが、スコア集計ではnon_targetに丸められる）。したがって `P(T)` を軸にゲートするのは理にかなう。
    
* **確率の一貫性**：Zadrozny & Elkan系の“**確率結合（coupling）**”は、複数の出力を**条件付き確率**として結び、全体の整合性を保つ古典的手法。ここでも `P(i)=P(T)·P(i|T)` などで自然に達成できる。 [NeurIPS Papers](https://papers.neurips.cc/paper/2122-reducing-multiclass-to-binary-by-coupling-probability-estimates.pdf?utm_source=chatgpt.com)[ACM Digital Library](https://dl.acm.org/doi/10.1145/775047.775151?utm_source=chatgpt.com)
    
* **計算も軽い**：モデルを作り直さず**推論後処理**で入れ替え可能（hierarchical softmax的な発想だが学習をやり直さない）。 [arXiv](https://arxiv.org/pdf/2308.01210?utm_source=chatgpt.com)[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/727855c31df8821fd18d41c23daebf10-Paper-Conference.pdf?utm_source=chatgpt.com)
    

* * *

## チューニング手順（最短ルート）

1. OOF/val から3ヘッドの出力を保存（フレーム→シーケンスはlog-sum-expや平均で集約）。
    
2. 各ヘッドに **T** を学習（NLL最小化）。まずは `T18_all, Tbin, T9` の3つ。 [arXiv](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)
    
3. `τ ∈ [0.2, 0.8]` を粗くグリッド → Binary F1最大点を採用。 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/?utm_source=chatgpt.com)
    
4. `w_hier ∈ [0.3, 0.8]` を粗くグリッド → **最終スコア（Binary F1 + 9クラスMacro F1の平均）**最大のもの。
    
5. 余裕があれば Dirichlet calibration を 18/9 クラスに入れて比較。 [arXiv](https://arxiv.org/abs/1910.12656?utm_source=chatgpt.com)
    

* * *

## 追加のコツ

* **強ゲート規則**（簡単で効きやすい）：  
    `P(T)≥τ_high` ならターゲット側だけでargmax、`P(T)≤τ_low` なら非ターゲット側だけでargmax、真ん中は通常融合に落とす（τ_high=0.6, τ_low=0.4などをCV）。
    
* **non_target整合**：nine-class の `P(non_target)` と `1−P(T)` が乖離するなら、Tbin or T9 を軽く再学習して差を詰める（ECEが下がる方向）。 [arXiv](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com)
    
* **スタッキング**：`[logits18, logits9(8+1), logits2]` を連結→多項ロジット回帰/小さなMLPで18クラスに写像（過学習防止にL2 & 早期打ち切り）。 [NeurIPS Papers](https://papers.neurips.cc/paper/2122-reducing-multiclass-to-binary-by-coupling-probability-estimates.pdf?utm_source=chatgpt.com)
    

* * *

この設計は**IMUのみのテスト列**が混じる本戦でも堅い（ヘッドの確信度の差が自然にゲートへ反映される）ので、そのまま最終推論にも持ち込めます。

必要なら、あなたの `on_validation_epoch_end` / 推論スクリプトに合わせて上記関数を埋め込み用に書き換えて渡すよ。
