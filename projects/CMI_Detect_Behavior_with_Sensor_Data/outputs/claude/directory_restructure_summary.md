# ディレクトリ構造統合実装まとめ

## 概要
`exp/`と`src/`ディレクトリを`codes/exp/`と`codes/src/`に統合し、`sys.path.append`を使用しないクリーンなimport構造を実現しました。

## 実装内容

### 1. ディレクトリ構造の変更
- `codes/`ディレクトリを作成
- `exp/`を`codes/exp/`に移動
- `src/`を`codes/src/`に移動

### 2. パッケージ設定
**pyproject.toml:**
- `package = true`に変更
- setuptools設定を追加:
  ```toml
  [tool.setuptools.packages.find]
  where = ["."]
  include = ["codes", "codes.*"]
  namespaces = true
  ```

### 3. __init__.pyファイルの作成
- `codes/__init__.py`
- `codes/exp/__init__.py`
- `codes/exp/exp007/__init__.py`

### 4. import文の修正
**絶対import（プロジェクトルートから）:**
```python
from codes.src.utils.logger import create_logger
from codes.src.utils.seed_everything import seed_everything
from codes.src.validation.factory import get_validation
```

**相対importとフォールバック:**
```python
# Local imports - use absolute when running as script
try:
    from .config import Config
    from .dataset import IMUDataModule
    from .model import CMISqueezeformer
except ImportError:
    # Fallback for running as script
    from config import Config
    from dataset import IMUDataModule
    from model import CMISqueezeformer
```

### 5. パス設定の更新
**config.py内のパス:**
- `../../outputs/exp007` → `../../../outputs/exp007`
- `../../data/` → `../../../data/`
- すべての相対パスを1階層深く修正

### 6. テストコードの修正
```python
# 変更前
from exp.exp007.dataset import IMUDataset

# 変更後
from codes.exp.exp007.dataset import IMUDataset
```

## 実行方法

### train.pyの実行
```bash
cd codes/exp/exp007/
uv run python train.py
```

### inference.pyの実行
```bash
cd codes/exp/exp007/
uv run python inference.py
```

### テストの実行
```bash
# プロジェクトルートから
uv run pytest tests/test_exp007_dataset.py
```

## 動作確認結果
- ✅ train.pyが正常に起動し、CVトレーニングが開始
- ✅ inference.pyが正常に起動（モデルなしでも初期化成功）
- ✅ pytestでテストが正常に実行され、パス

## メリット
1. **クリーンなimport構造**: `sys.path.append`ハックが不要
2. **IDE対応**: 自動補完やリファクタリングが正しく動作
3. **型チェック対応**: mypyなどの静的解析ツールが正しく動作
4. **保守性向上**: 明示的な依存関係で理解しやすい

## 注意点
- Pydantic V1スタイルの`@validator`に関する警告が出ているが、動作には影響なし
- 実行は必ず`uv run python`を使用すること（パッケージのインストールが必要）