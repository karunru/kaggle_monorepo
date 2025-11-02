# CMI モデルアーキテクチャ調査メモ (exp021)

## 目的
- 競技の cmi_score (Binary F1 と Macro F1 の平均) を最大化するために、現行実装の設計妥当性を検討し、改善案を整理する。

## 参考資料
- コンペ概要: docs/competition_overview.md
- データ詳細: docs/data_description.md
- 実装参照: codes/exp/exp021/ (dataset.py, model.py, config.py, train.py, inference.py)

## 現行実装の要点 (exp021)
- 入力特徴: IMU 基本 7 次元 + 物理ベース特徴 9 次元 = 16 次元 (config.model.input_dim=16)。
  - 物理特徴は Polars でベクトル化計算 (dataset._add_physics_features)。
- モデル: Squeezeformer ブロック + BertModel (自己注意スタック) を通し、最終に 2 ヘッド。
  - マルチクラスヘッド: 18 クラス (全ジェスチャー)。
  - バイナリヘッド: target vs non-target。
- 損失: 2 つの損失を重み付き (loss.alpha) で合算。CrossEntropy + BCE が既定。Focal / SoftF1 / ACLS / MbLS も選択可。
- デモグラフィクス: 省略可能な埋め込みを BERT 出力に連結 (DemographicsEmbedding)。
- 推論: inference.py は「18 クラスの softmax の argmax のみ」を用いて gesture を決定。バイナリヘッドは推論で未使用。

## 評価指標との整合性の観点
- cmi_score は (1) target vs non-target の Binary F1 と、(2) target 8 クラス + non_target の Macro F1 の平均。
- 現行の 18 クラス学習は non-target 同士の識別も学習しており、評価上は不要な難題を同時に解いている可能性。
- さらに推論でバイナリヘッドを使っていないため、Binary F1 最適化への寄与が限定的。

## リスク/課題の整理
- 推論時にバイナリヘッド未使用: binary loss で学んだ分離が最終予測に活きない。
- 18 クラス argmax 方針: 「target/非 target の境目」を跨ぐ誤りが Binary F1 を強く悪化させうる。
- 学習目標と評価整合性: Macro F1 は non_target を単一クラスに潰す計算。18 クラス学習はややミスマッチ。

## 改善提案
1) 推論での階層ゲーティング (2 ヘッドの統合利用)
   - バイナリヘッド出力 p_t = P(target) を用い、
     - target クラス c に対し: score(c) = p_t * softmax_c / sum_softmax(target集合)
     - non-target クラス c に対し: score(c) = (1-p_t) * softmax_c / sum_softmax(non-target集合)
   - 上記 score の最大で最終ラベル決定。Binary F1 を保ちつつ、ターゲット側/非ターゲット側の内部順位付けは 18 クラス頭で維持。
   - 変更影響: inference.py のみ (学習コード変更不要)。

2) 9 クラス補助ヘッドの追加 (8 target + non_target)
   - 18 クラス + バイナリに加え、non_target を単一クラスに潰した 9 クラス補助ヘッドを導入。
   - 損失は CE/SoftF1/ACLS/MbLS から選択し、cmi_score との整合性を強化。
   - 期待効果: Macro F1 直結の表現学習を促しつつ、18 クラスヘッドで非 target 内のスコア分配を保持。

3) バイナリの「暗黙推定」も併用して較正
   - 18 クラス softmax から p_t_soft = sum_{c in target} softmax_c を算出し、バイナリヘッドの sigmoid 出力 p_t と混合:
     p_t_blend = w * p_t + (1-w) * p_t_soft (w は学習/検証で較正)。
   - テスト時のドメイン差 (IMU のみ/全センサー) に対するロバスト性を狙った確率融合。

4) 損失設計の見直し (cmi_score 志向)
   - 9 クラスヘッドに SoftF1Loss を適用 (Macro F1 の微分可能近似)。
   - バイナリに BinarySoftF1Loss を適用し、(MacroF1+BinaryF1) 方向に学習信号を寄せる。
   - 18 クラスヘッドはラベルスムージング or ACLS/MbLS で安定化。

5) メタ情報活用の拡張
   - 既存の DemographicsEmbedding は有効。加えて以下の軽量拡張を検討:
     - orientation/behavior への埋め込み (token type 的に時系列へ付与)。
     - subject 埋め込み (CV 分割とリークに注意)。

## 実装インパクト (小→中)
- 最小変更: 提案1 (推論のスコア統合) は inference.py のみで完結。
- 中規模: 提案2/4 は model.py に 9 クラス補助ヘッドと loss 和の追加、dataset 側で 9 クラス用ラベル (non_target へ潰す) の提供が必要。

## 簡易実装プラン
- P1: inference.py に階層ゲーティングを追加し、CV で cmi_score を検証。
- P2: 9 クラス補助ヘッド + SoftF1/BinarySoftF1 を有効化し、loss 重み (multi:bin:9cls) を探索。
- P3: p_t 融合 (p_t_blend) の w を CV で較正。
- P4: behavior/orientation 埋め込みの有効性を確認。

## 期待される効果
- Binary F1 の明確な改善 (推論でのゲート利用)。
- Macro F1 の改善 (9 クラス補助学習 + SoftF1)。
- 18 クラス学習の表現力を維持しつつ、評価指標に直結した学習へバランス。

## 追加メモ
- 現状 inference.py は 18 クラス argmax 固定。提案1 のみでも「二段階の意思決定 (target/非 target → 細分類)」になり cmi_score 向上が見込める。
- 学習済み重みの互換性: 提案1 は再学習不要。提案2 以降は再学習が必要。

