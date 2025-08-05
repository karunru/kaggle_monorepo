# 設計パターンとガイドライン

## 設計哲学

### 1. 実験駆動開発
- 各実験は独立したディレクトリで管理
- 段階的改善（exp001 → exp007）
- 再現可能性を重視した実装

### 2. 型安全性とバリデーション
- Pydantic Settingsによる設定管理
- 厳密な型ヒント（mypy strict mode）
- 実行時バリデーション

### 3. GPU最適化とスケーラビリティ
- CuPy/PyTorch統合
- 分散訓練対応（PyTorch Lightning）
- メモリ効率化

## 主要設計パターン

### Factory Pattern
**使用箇所**: models/factory.py, validation/factory.py, sampling/factory.py

```python
# モデルファクトリの例
def create_model(model_name: str, **kwargs) -> BaseModel:
    if model_name == "lightgbm":
        return LightGBMModel(**kwargs)
    elif model_name == "xgboost":
        return XGBoostModel(**kwargs)
    # ...
```

**利点**: モデル選択の柔軟性、設定からの動的生成

### Strategy Pattern  
**使用箇所**: features/配下の特徴量エンジニアリング

```python
class BaseFeature(ABC):
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

class StatisticsFeature(BaseFeature):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # 統計的特徴量の実装
        pass
```

**利点**: 特徴量の組み合わせ・切り替えが容易

### Template Method Pattern
**使用箇所**: models/base.py

```python
class BaseModel(ABC):
    def fit(self, X, y):
        X = self.preprocess(X)
        self._fit(X, y)
        return self.postprocess()
    
    @abstractmethod
    def _fit(self, X, y):
        pass
```

**利点**: 共通処理の統一、個別実装の柔軟性

### Builder Pattern
**使用箇所**: 実験設定（config.py）

```python
class ExperimentConfig(BaseSettings):
    experiment: ExperimentConfig
    model: ModelConfig  
    data: DataConfig
    training: TrainingConfig
```

**利点**: 複雑な設定の段階的構築

## コーディングガイドライン

### 1. 命名規約
- **クラス**: PascalCase (`CMISqueezeformer`)
- **関数・変数**: snake_case (`train_single_fold`)
- **定数**: UPPER_SNAKE_CASE (`EXP_NUM`)
- **実験番号**: exp{3桁数字} (`exp007`)

### 2. ドキュメント規約
```python
def train_single_fold(
    config: Config,
    fold: int,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame
) -> dict[str, Any]:
    """単一フォールドでモデルを訓練.
    
    Args:
        config: 実験設定
        fold: フォールド番号
        train_df: 訓練データ
        val_df: 検証データ
        
    Returns:
        訓練結果の辞書（loss, metrics等）
    """
```

### 3. エラーハンドリング
- 明示的な例外処理
- ログによる詳細情報記録
- ユーザフレンドリーなエラーメッセージ

### 4. テスト駆動開発
- 各実験にtest_exp{番号}.pyを必須配置
- ユニットテスト＋統合テスト
- モックを活用したGPU環境非依存テスト

## パフォーマンス最適化指針

### 1. データ処理
- **Polars使用**: pandas比で高速なデータフレーム操作
- **Arrow形式**: メモリ効率的なカラムナ形式
- **遅延評価**: 必要時まで計算を遅延

### 2. GPU活用
- **CuPy統合**: NumPy処理のGPU化
- **PyTorch Lightning**: 分散訓練の自動化
- **混合精度**: float16によるメモリ節約

### 3. I/O最適化
- **圧縮保存**: 特徴量の自動圧縮（base.py）
- **キャッシュ活用**: 計算済み特徴量の再利用
- **並列読み込み**: DataLoaderの最適化

## セキュリティ・品質保証

### 1. 型安全性
- mypy strictモードによる厳密チェック
- Pydanticによる実行時バリデーション
- 型ヒント必須化

### 2. 再現性保証
- seed_everything.pyによるシード固定
- deterministic設定
- 実験環境の記録

### 3. コード品質
- ruffによる厳格なリント
- 自動フォーマット
- 複雑度制限（mccabe）

## Kaggle特有の考慮事項

### 1. 評価システム統合
- gRPCによるKaggle評価API対応
- カスタム評価指標の実装
- リアルタイム評価フィードバック

### 2. データセット管理
- 自動アップロード（mise run update-codes）
- バージョン管理
- 依存関係追跡

### 3. サブミッション最適化
- 推論時間の最適化
- メモリ使用量制限対応
- エラー耐性の確保