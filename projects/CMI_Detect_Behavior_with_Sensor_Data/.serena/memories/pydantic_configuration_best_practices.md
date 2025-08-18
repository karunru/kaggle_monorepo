# Pydantic設定管理のベストプラクティス

## 概要
このプロジェクトではPydantic Settingsを使用した型安全な設定管理を採用している。
以下は、実際の経験から得られた重要な禁止事項とベストプラクティスである。

## 重要な禁止事項（必読！）

### 1. 辞書的アクセスの禁止
```python
# ❌ 絶対に使用禁止
config.get("learning_rate", 0.001)
augmentation_config.get("gaussian_noise_prob", 0.0)

# ✅ 正しい使用方法
config.learning_rate
augmentation_config.gaussian_noise_prob
```

**理由**: 
- 型安全性の喪失
- IDEの補完・エラー検出が働かない
- typoによるバグの温床

### 2. 両対応関数の禁止
```python
# ❌ 絶対に作成・使用禁止
def _safe_get_attr(obj, attr_name, default=None):
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    elif isinstance(obj, dict):
        return obj.get(attr_name, default)
    else:
        return default

def _convert_dict_to_config(config_dict, config_class):
    if isinstance(config_dict, dict):
        return config_class(**config_dict)
    return config_dict
```

**理由**:
- 型安全性を完全に破壊
- どの型が渡されるか不明確
- デバッグが困難

### 3. .model_dump()による設定渡しの禁止
```python
# ❌ 絶対に禁止
model = CMISqueezeformer(
    loss_config=config.loss.model_dump(),  # 辞書化は禁止
    acls_config=config.acls.model_dump()
)

# ✅ 正しい方法
model = CMISqueezeformer(
    loss_config=config.loss,  # Pydanticオブジェクトを直接渡す
    acls_config=config.acls
)
```

**理由**:
- 型情報の損失
- バリデーションの無効化
- 設定ミスの検出不可能

### 4. オプション設定の濫用禁止
```python
# ❌ 必須設定をオプション扱いするのは禁止
def __init__(
    self,
    loss_config: LossConfig | None = None,  # 設定ミスを隠蔽
    acls_config: ACLSConfig | None = None
):
    self.loss_config = loss_config or LossConfig()

# ✅ 必須設定は明示的に要求
def __init__(
    self,
    loss_config: LossConfig,  # 必須引数として明示
    acls_config: ACLSConfig
):
    self.loss_config = loss_config
```

**理由**:
- 設定ミスの早期発見
- 意図しないデフォルト値使用の防止
- API契約の明確化

## 推奨パターン

### 1. 設定クラスの定義
```python
from pydantic import BaseModel, Field

class LossConfig(BaseModel):
    type: Literal["cmi", "focal", "soft_f1"] = Field(default="cmi")
    focal_gamma: float = Field(default=2.0, description="Focal loss gamma")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
```

### 2. 設定クラスの使用
```python
class Model:
    def __init__(self, loss_config: LossConfig):
        self.loss_config = loss_config
        
        # ✅ 直接属性アクセス
        if self.loss_config.type == "focal":
            self.criterion = FocalLoss(
                gamma=self.loss_config.focal_gamma,
                alpha=self.loss_config.focal_alpha
            )
```

### 3. 設定の渡し方
```python
# config.pyでの定義
class Config(BaseSettings):
    loss: LossConfig = Field(default_factory=LossConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

# 使用時
config = Config()
model = Model(loss_config=config.loss)  # Pydanticオブジェクトを直接渡す
```

## 例外的に.model_dump()が許可される場面

以下の場面でのみ.model_dump()の使用が許可される：

1. **ログ出力**: `logger.info(f"Config: {config.model_dump()}")`
2. **外部API**: WandBなどの外部サービスへの送信
3. **シリアライゼーション**: JSON保存など
4. **レガシー関数**: 辞書を期待する外部ライブラリとの連携

```python
# ✅ 許可される使用例
logger.info(f"Configuration: {config.model_dump()}")
wandb.init(config=config.model_dump())
with open("config.json", "w") as f:
    json.dump(config.model_dump(), f)
```

## 型安全性の確保

### 1. 型ヒントの活用
```python
# ✅ 明確な型指定
def create_model(
    loss_config: LossConfig,
    training_config: TrainingConfig
) -> torch.nn.Module:
    pass
```

### 2. IDEサポートの活用
- 自動補完による typo 防止
- 未使用属性の検出
- リファクタリング支援

### 3. 実行時バリデーション
```python
class LossConfig(BaseModel):
    learning_rate: float = Field(gt=0.0, le=1.0)  # 範囲制限
    
    @model_validator(mode='after')
    def validate_consistency(self):
        # カスタムバリデーション
        if self.type == "focal" and self.focal_gamma <= 0:
            raise ValueError("focal_gamma must be positive for focal loss")
        return self
```

## 移行時の注意点

既存コードを修正する際の手順：

1. `config.get()` を `config.attribute` に置換
2. `_safe_get_attr()` 呼び出しを直接属性アクセスに変更
3. `.model_dump()` による設定渡しを直接渡しに変更
4. 型ヒントを追加
5. テストで動作確認

## まとめ

これらのベストプラクティスにより：
- **型安全性の向上**: コンパイル時エラー検出
- **バグの早期発見**: 設定ミスの即座な検出
- **保守性の向上**: IDEサポートによる安全なリファクタリング
- **可読性の向上**: 設定の意図が明確

**重要**: これらの禁止事項は過去の実装ミスから学んだ教訓である。違反すると型安全性が損なわれ、デバッグが困難になる。