# exp035 - try-except import削除作業から学んだ禁止事項

## 概要
exp035での`try-except ImportError`パターン削除作業で判明した、今後避けるべきコードパターンと推奨プラクティス。

## 禁止事項

### 1. try-except ImportErrorパターンの使用禁止
```python
# ❌ 禁止パターン
try:
    from config import ACLSConfig, BertConfig
except ImportError:
    ACLSConfig = None
    BertConfig = None

# ✅ 推奨パターン
from config import (
    ACLSConfig,
    BertConfig,
    DemographicsConfig,
    EMAConfig,
    LossConfig,
    ScheduleFreeConfig,
    SchedulerConfig,
)
```

### 2. Optional configパターンの禁止
```python
# ❌ 禁止パターン
def __init__(self, config: Config | None = None):
    self.config = config or {}

# ✅ 推奨パターン  
def __init__(self, config: Config):
    self.config = config
```

### 3. config.get()パターンの使用禁止
```python
# ❌ 禁止パターン
enabled = config.get("enabled", False)
value = config.get("some_key", default_value)

# ✅ 推奨パターン
enabled = config.enabled
value = config.some_key
```

### 4. _safe_get_attr等のヘルパー関数禁止
```python
# ❌ 禁止パターン
def _safe_get_attr(obj, attr, default=None):
    return getattr(obj, attr, default) if obj else default

# ✅ 推奨パターン
# 必要な属性は直接アクセス、オブジェクトは必須パラメータとして渡す
```

## 推奨プラクティス

### 1. Pydantic設定クラスの使用
- すべての設定は厳密に型付けされたPydanticクラスで管理
- 辞書的アクセスではなくオブジェクト属性としてアクセス
- 必須パラメータは明示的に定義し、Noneを許可しない

### 2. Fail-fast原則の実装
- 設定が不正または不足している場合は起動時に即座にエラー
- 実行時ではなく設定読み込み時にバリデーション実行
- try-catch による安全回避を行わない

### 3. 明示的依存関係管理
- 必要な依存関係は明示的にimport
- ImportErrorによる条件付きimportを避ける
- 依存関係が存在しない場合は明確にエラーとする

### 4. 設定オブジェクトの責務分離
- 各設定クラスは独立したPydantic Modelとして定義
- 辞書から設定クラスへの変換関数は削除
- 設定の更新は型安全な方法で実行

## 実装例

### 良い設定クラスの定義
```python
class SchedulerConfig(BaseModel):
    """スケジューラ設定."""
    type: Literal["cosine", "plateau"] | None = Field(default="cosine")
    min_lr: float = Field(default=1e-6)
    factor: float = Field(default=0.5)
    patience: int = Field(default=5)
```

### 良いコンストラクタの定義
```python
def __init__(
    self,
    loss_config: LossConfig,
    acls_config: ACLSConfig,
    demographics_config: DemographicsConfig,
    bert_config: BertConfig,
    schedule_free_config: ScheduleFreeConfig,
    ema_config: EMAConfig,
    scheduler_config: SchedulerConfig,
    # すべて必須パラメータとして定義
):
```

## 今後の開発指針

1. **設定管理**: Pydantic + 厳密型付けを必須とする
2. **エラーハンドリング**: Fail-fastを基本とし、安全回避を行わない
3. **依存関係**: 明示的importを基本とし、条件付きimportを避ける
4. **コードスタイル**: 辞書アクセスではなくオブジェクト属性アクセスを使用

この方針により、設定ミスや依存関係の問題を開発段階で早期発見でき、本番環境での予期しないエラーを防げる。