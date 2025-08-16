# exp025実装計画：学習可能重み付き損失関数

## 1. プロジェクト概要

### 1.1 コンペティション背景
- **コンペ名**: CMI - Detect Behavior with Sensor Data
- **目標**: 手首装着デバイスのセンサーデータからBFRB様とnon-BFRB様活動を区別
- **評価指標**: 2値F1スコア（target vs non-target）とマクロF1スコア（8 target + その他）の平均

### 1.2 データ構造
- **18クラス**: 8つのtargetクラス + 10つのnon-targetクラス
- **9クラス**: 8つのtargetクラス + 1つの統合non-targetクラス  
- **バイナリ**: target vs non-target分類

### 1.3 exp025の目標
exp024の固定重み損失関数を学習可能パラメータに変更し、訓練中に自動的に最適な重み配分を学習する。

## 2. 現状分析（exp024）

### 2.1 現在の損失関数構成
```python
total_loss = (
    self.loss_alpha * multiclass_loss           # 18クラス分類損失
    + (1 - self.loss_alpha) * binary_loss       # バイナリ分類損失  
    + self.nine_class_loss_weight * nine_class_loss  # 9クラス分類損失
    + self.kl_weight * kl_loss                  # 階層一貫性損失
)
```

### 2.2 固定パラメータ（LossConfig）
- `alpha`: 0.5（バイナリvsマルチクラス重み）
- `nine_class_loss_weight`: 0.2（9クラス損失重み）
- `kl_weight`: 0.1（KL divergence損失重み）

### 2.3 課題
- 重みが固定のため、タスクの難易度や学習進行に応じた自動調整ができない
- 手動チューニングが必要で最適化が困難

## 3. 実装方針

### 3.1 アプローチ選択
**不確かさベース自動重み付け（Kendall & Gal 2018）を採用**

**理由**:
- 理論的根拠が明確（homoscedastic uncertainty）
- スケール不変で安定
- 自動的に難しいタスクの重みを下げ、易しいタスクの重みを上げる
- 既存の実装との互換性が高い

### 3.2 数式
```
s_i = log(σ_i²)  # 学習可能パラメータ
L_total = Σ [exp(-s_i) * L_i + s_i]
```

ここで：
- `s_i`: 各損失項の学習可能パラメータ
- `L_i`: 個別損失項
- `exp(-s_i)`: 自動調整される重み
- `s_i`: 正則化項

## 4. 実装タスク

### 4.1 タスク1: LossConfig拡張
**ファイル**: `codes/exp/exp025/config.py`

**追加項目**:
```python
class LossConfig(BaseModel):
    # 既存の設定項目は維持
    
    # 新規追加
    auto_weighting: Literal["none", "uncertainty", "direct"] = Field(
        default="none", 
        description="自動重み付け方式"
    )
    auto_weight_clamp: tuple[float, float] = Field(
        default=(-10.0, 10.0), 
        description="学習可能パラメータのクランプ範囲"
    )
    uncertainty_init_value: float = Field(
        default=0.0, 
        description="不確かさパラメータの初期値"
    )
```

### 4.2 タスク2: CMISqueezeformerモデル拡張
**ファイル**: `codes/exp/exp025/model.py`

**変更箇所**:

1. **`__init__`メソッド**: 学習可能パラメータの初期化
```python
def __init__(self, ...):
    # 既存の初期化処理
    
    # 自動重み付けパラメータの設定
    aw_type = self.loss_config.get("auto_weighting", "none")
    if aw_type == "uncertainty":
        self.loss_s_params = nn.ParameterDict()
        init_val = self.loss_config.get("uncertainty_init_value", 0.0)
        self.loss_s_params["multiclass"] = nn.Parameter(torch.tensor(init_val))
        self.loss_s_params["binary"] = nn.Parameter(torch.tensor(init_val))
        if self.loss_config.get("nine_class_head_enabled", True):
            self.loss_s_params["nine_class"] = nn.Parameter(torch.tensor(init_val))
        if self.kl_weight > 0:
            self.loss_s_params["kl"] = nn.Parameter(torch.tensor(init_val))
```

2. **`training_step`メソッド**: 自動重み付き損失計算
```python
def training_step(self, batch, batch_idx):
    # 既存の損失計算
    
    # 自動重み付け損失の計算
    aw_type = self.loss_config.get("auto_weighting", "none")
    if aw_type == "uncertainty":
        total_loss = self._compute_uncertainty_weighted_loss(
            multiclass_loss, binary_loss, nine_class_loss, kl_loss
        )
        # 重みをログ出力
        self._log_uncertainty_weights()
    else:
        # 従来の固定重み損失
        total_loss = (
            self.loss_alpha * multiclass_loss
            + (1 - self.loss_alpha) * binary_loss
            + self.nine_class_loss_weight * nine_class_loss
            + self.kl_weight * kl_loss
        )
```

3. **新規メソッド**: 不確かさ重み付き損失計算
```python
def _compute_uncertainty_weighted_loss(self, mc_loss, bin_loss, nine_loss, kl_loss):
    """不確かさベース自動重み付き損失の計算"""
    clamp_range = self.loss_config.get("auto_weight_clamp", (-10.0, 10.0))
    
    terms = []
    
    # マルチクラス損失
    s_mc = self.loss_s_params["multiclass"].clamp(*clamp_range)
    terms.append(torch.exp(-s_mc) * mc_loss + s_mc)
    
    # バイナリ損失
    s_bin = self.loss_s_params["binary"].clamp(*clamp_range)
    terms.append(torch.exp(-s_bin) * bin_loss + s_bin)
    
    # 9クラス損失（有効な場合のみ）
    if "nine_class" in self.loss_s_params:
        s_nine = self.loss_s_params["nine_class"].clamp(*clamp_range)
        terms.append(torch.exp(-s_nine) * nine_loss + s_nine)
    
    # KL損失（有効な場合のみ）
    if "kl" in self.loss_s_params and self.kl_weight > 0:
        s_kl = self.loss_s_params["kl"].clamp(*clamp_range)
        terms.append(torch.exp(-s_kl) * kl_loss + s_kl)
    
    return sum(terms)
```

4. **新規メソッド**: 重みログ出力
```python
def _log_uncertainty_weights(self):
    """学習された重みをログ出力"""
    if hasattr(self, "loss_s_params"):
        for name, param in self.loss_s_params.items():
            s_val = param.clamp(*self.loss_config.get("auto_weight_clamp", (-10.0, 10.0)))
            weight = torch.exp(-s_val)
            self.log(f"weight_{name}", weight.detach())
            self.log(f"s_{name}", s_val.detach())
```

5. **`configure_optimizers`メソッド**: 学習可能パラメータの最適化設定
```python
def configure_optimizers(self):
    # 自動重み付けパラメータの分離
    aw_params = []
    if hasattr(self, "loss_s_params"):
        aw_params = list(self.loss_s_params.parameters())
    
    if aw_params:
        # 基本パラメータと自動重み付けパラメータを分離
        base_params = [p for n, p in self.named_parameters() if p not in set(aw_params)]
        
        # パラメータグループ別最適化
        optimizer = AdamW([
            {"params": base_params},
            {"params": aw_params, "lr": self.learning_rate * 0.1, "weight_decay": 0.0},
        ], lr=self.learning_rate, weight_decay=self.weight_decay)
        
        return optimizer
    else:
        # 従来通り
        return super().configure_optimizers()
```

### 4.3 タスク3: 下位互換性確保
- `auto_weighting="none"`の場合、既存のexp024と完全に同じ動作
- 既存のLossConfigパラメータはすべて保持
- デフォルト設定では新機能は無効

### 4.4 タスク4: テストコード作成
**ファイル**: `tests/test_exp025_learnable_weights.py`

**テスト項目**:
1. 学習可能パラメータの初期化テスト
2. 不確かさ重み付き損失の計算テスト
3. 勾配計算とパラメータ更新テスト
4. ログ出力テスト
5. 下位互換性テスト（`auto_weighting="none"`）

## 5. 実装順序

### 5.1 段階1: 基本構造の構築
1. exp024をexp025にコピー
2. LossConfigの拡張
3. CMISqueezeformerの基本的な学習可能パラメータ追加

### 5.2 段階2: 損失計算の実装
1. 不確かさ重み付き損失計算メソッドの実装
2. training_step/validation_stepの修正
3. ログ機能の追加

### 5.3 段階3: 最適化の調整
1. パラメータグループ別最適化の実装
2. configure_optimizersの修正

### 5.4 段階4: テストとデバッグ
1. テストコードの作成と実行
2. 動作確認
3. 静的解析とコード品質チェック

## 6. 期待される効果

### 6.1 性能向上
- タスクの相対的難易度に応じた自動重み調整
- 学習進行に伴う動的な重み最適化
- 手動チューニング不要による効率化

### 6.2 実験的価値
- 各損失項の重要度の自動発見
- 学習過程での重み変化の観察
- 異なるデータセットでの汎用性評価

## 7. リスク管理

### 7.1 互換性リスク
- **軽減策**: デフォルトで新機能無効、既存設定での完全後方互換性

### 7.2 数値安定性リスク
- **軽減策**: パラメータクランプ、勾配クリッピング継続

### 7.3 収束リスク
- **軽減策**: 小さな学習率、保守的な初期化

## 8. 成果物

### 8.1 実装ファイル
- `codes/exp/exp025/config.py` - 拡張されたLossConfig
- `codes/exp/exp025/model.py` - 学習可能重み付きCMISqueezeformer
- `codes/exp/exp025/[その他]` - exp024からの継承ファイル

### 8.2 テストファイル
- `tests/test_exp025_learnable_weights.py` - 包括的テストスイート

### 8.3 ドキュメント
- `outputs/claude/exp025_implementation_completion.md` - 実装完了報告書

---

**本計画書は、exp024の既存機能を保持しつつ、学習可能な損失重み付け機能を追加することで、モデルの自動最適化能力を向上させることを目指しています。**