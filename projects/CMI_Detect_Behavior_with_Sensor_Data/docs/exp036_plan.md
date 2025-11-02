# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp035` を `codes/exp/exp036` にコピーしてからexp036を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp035` からのインポートは禁止
- `codes/exp/exp036` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- model.pyのCMISqueezeformerのforwordは以下にして
```python
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
```
  - この変更により使わなくなった変数とかは消して
- docs/chatgpt_augmentation.md を参照して、 tsai https://timeseriesai.github.io/tsai/data.transforms.html を使ってaugmentationを実装して
  - 既に実装済みのものがあれば tsai.data.transform のクラスに置き換えて
  - imu の回転 は実装して
    - https://github.com/timeseriesAI/tsai/blob/main/tsai/data/transforms.py を参照して、 `from fastai.vision.augment import RandTransform` を継承して
  - 利き手での反転も `from fastai.vision.augment import RandTransform` を継承して実装し直して
- RandAugmentも実装して
- Augmentationに関するpydantic config classを作成してこのクラスで設定できるようにして 