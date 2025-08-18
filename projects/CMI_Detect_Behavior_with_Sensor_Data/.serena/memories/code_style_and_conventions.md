# コードスタイルと規約

## 基本設定
- **Python バージョン**: 3.12
- **コードスタイル**: PEP 8準拠
- **文字エンコーディング**: UTF-8
- **改行コード**: LF（Linux標準）

## Ruff設定（../../ruff.toml）
- **行長**: 120文字
- **インデント幅**: 4スペース
- **ターゲットバージョン**: Python 3.12
- **クォートスタイル**: ダブルクォート
- **出力フォーマット**: grouped

### 有効なルール
- Airflow, flake8-async, flake8-blind-except
- flake8-comprehensions, mccabe, flake8-datetimez
- pycodestyle (E, W), Pyflakes (F)
- isort (I), NumPy-specific rules (NPY)
- Pylint (PL), Ruff-specific rules (RUF)
- flake8-bandit (S), tryceratops (TRY)
- pyupgrade (UP)

### 無視されるルール
- D100-107: docstring関連（ある程度緩い設定）

## MyPy設定（../../mypy.ini）
- **厳密モード**: strict = true
- **表示設定**: pretty = true, show_column_numbers = true
- **エラーコード表示**: show_error_codes = true

## 型ヒント規約
- 全関数・メソッドに型ヒント必須
- pydanticモデルでデータ構造定義
- from __future__ import annotations使用推奨

## ドキュメント規約
- **形式**: Google Style Docstrings
- **言語**: 基本的に日本語コメント、英語docstring
- **必須レベル**: パブリックAPI、重要な内部関数

## ファイル命名規約
- **モジュール**: snake_case
- **クラス**: PascalCase
- **関数・変数**: snake_case
- **定数**: UPPER_SNAKE_CASE

## インポート規約
- 標準ライブラリ → サードパーティ → ローカル
- isortで自動整理（combine-as-imports = true）

## 実験コード規約
- 各実験はexp{番号}ディレクトリに独立配置
- config.py, model.py, dataset.py, train.py, inference.pyの構造統一
- 実験メタデータはpydantic設定で管理

## Pydantic設定規約（重要！）

### 禁止事項
1. **`config.get("key", default_value)` の使用禁止**
   - ❌ `config.get("learning_rate", 0.001)`
   - ✅ `config.learning_rate`
   - Pydanticクラスでは直接属性アクセスを使用

2. **辞書とPydanticオブジェクトの両対応関数の禁止**
   - ❌ `_safe_get_attr(obj, "attr", default)`
   - ❌ `_convert_dict_to_config(dict_config, ConfigClass)`
   - 型安全性を損なうため、最初からPydanticオブジェクトとして扱う

3. **`.model_dump()`でPydanticを辞書化してから渡すことの禁止**
   - ❌ `model = Model(config=config.loss.model_dump())`
   - ✅ `model = Model(config=config.loss)`
   - 直接Pydanticオブジェクトを渡す

4. **必須設定に`| None = None`デフォルト値の禁止**
   - ❌ `def __init__(self, config: Config | None = None)`
   - ✅ `def __init__(self, config: Config)`
   - 設定が不適切な場合は明示的にエラーを発生させる

### 推奨パターン
- 設定クラスは必須引数として受け取る
- 設定の検証はPydanticのバリデーション機能を活用
- デフォルト値はPydanticクラス定義内で設定
- 型ヒントを活用してIDEの補完とエラー検出を有効化