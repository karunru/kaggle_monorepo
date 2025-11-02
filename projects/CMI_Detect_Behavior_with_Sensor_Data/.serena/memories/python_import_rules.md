# Python Import Rules

## 基本ルール
- **重要**: Pythonファイル(.py)では、importステートメントは必ずファイルの先頭に配置すること
- ファイルの途中や関数内、クラス内でのimportは禁止
- 条件分岐内でのimportも禁止

## PEP 8準拠のimport順序
1. 標準ライブラリ
2. サードパーティライブラリ  
3. ローカル/プロジェクト固有のライブラリ

各グループは空行で区切る

## 例（正しい形式）
```python
# 標準ライブラリ
import os
import sys
from pathlib import Path

# サードパーティライブラリ
import numpy as np
import pandas as pd
import torch

# ローカルライブラリ
from src.models.base import BaseModel
from src.utils.config import Config

# その後にクラスや関数定義
class MyClass:
    pass
```

## 禁止パターン
```python
def some_function():
    import pandas as pd  # ❌ 関数内でのimport禁止
    return pd.DataFrame()

if condition:
    import some_module  # ❌ 条件分岐内でのimport禁止
```

## 例外
- 動的importが必要な場合のみ、importlib.import_module()を使用
- 循環import回避のための遅延importは設計見直しを優先