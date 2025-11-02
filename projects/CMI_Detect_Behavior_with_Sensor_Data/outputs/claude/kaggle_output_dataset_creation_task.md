# Kaggleデータセット自動作成タスクの実装

## 概要
Kaggle CLIを使用してデータセットを自動作成・管理するタスクをmise.tomlに実装。dataset-metadata.jsonの自動生成と設定により、手動設定なしでデータセット作成が可能に。

### 実装タスク
1. **create-output-dataset**: outputs/配下の実験ディレクトリからデータセット作成
2. **update-codes**: codes/ディレクトリ全体を1つのデータセットとして管理

## 実装内容

### 1. タスク名
`create-output-dataset` - 実験結果のディレクトリからKaggleデータセットを作成する

### 2. タスクの仕様
- **引数**: 実験ディレクトリ名（例：exp007）
- **処理内容**:
  1. 引数で指定されたディレクトリがoutputs/配下に存在するか確認
  2. dataset-metadata.jsonファイルの存在確認
  3. 存在しない場合は`kaggle datasets init`でメタデータファイルを生成
  4. データセットの存在確認（`kaggle d status`）
  5. 新規作成または更新処理の実行

### 3. 実装詳細

#### タスク構造
```toml
[tasks.create-output-dataset]
description="Create or update Kaggle dataset from experiment output directory"
run="""
# シェルスクリプトの内容
"""
```

#### 処理フロー
1. **環境変数とパスの設定**
   - リポジトリルートとプロジェクトディレクトリの特定
   - 引数チェック（実験ディレクトリ名が指定されているか）

2. **ディレクトリの検証**
   - outputs/{実験名}ディレクトリの存在確認
   - 空ディレクトリチェック

3. **メタデータファイルの処理**
   - dataset-metadata.jsonの存在確認
   - 存在しない場合は初期化プロンプトを表示

4. **データセットの作成/更新**
   - 既存データセットの確認
   - 新規作成または更新の実行

### 4. エラーハンドリング
- 引数未指定時のエラーメッセージ
- ディレクトリが存在しない場合のエラー
- 空ディレクトリの場合の警告
- メタデータファイルが存在しない場合の対話的処理

### 5. 使用例
```bash
# exp007の実験結果をデータセットとして作成/更新
mise run create-output-dataset exp007
```

## テスト項目
1. 引数なしでの実行時のエラー処理
2. 存在しないディレクトリ指定時のエラー処理
3. 新規データセット作成の動作確認
4. 既存データセット更新の動作確認
5. メタデータファイル初期化フローの確認

## 実装結果

### 完了した実装
1. **mise.tomlへのタスク追加**: `create-output-dataset`タスクを正常に追加
2. **引数処理の修正**: `{{arg(name="experiment_name")}}`テンプレート形式に修正
3. **エラーハンドリング**: 適切なディレクトリ存在確認とエラーメッセージ
4. **メタデータ自動設定**: `kaggle datasets init`による初期化と自動的な設定値の置換
5. **自動処理継続**: 手動編集なしで直接データセット作成/更新まで完了

### 実装詳細
```toml
[tasks.create-output-dataset]
description="Create or update Kaggle dataset from experiment output directory"
run="""
EXPERIMENT_NAME="{{arg(name="experiment_name")}}"
# リポジトリルートとプロジェクトディレクトリの特定
# ディレクトリ存在確認
# メタデータファイルの自動設定
if [ ! -f dataset-metadata.json ]; then
    uv run kaggle datasets init -p .
    DATASET_TITLE="cmi-${EXPERIMENT_NAME}"
    DATASET_ID="${USERNAME}/cmi-${EXPERIMENT_NAME}"
    jq --arg title "$DATASET_TITLE" --arg id "$DATASET_ID" \
       '.title = $title | .id = $id' dataset-metadata.json > tmp.json && \
    mv tmp.json dataset-metadata.json
fi
# データセット作成/更新処理
"""
```

### 動作確認済み機能
1. **引数認識**: `mise run create-output-dataset --help`でヘルプ表示
2. **エラーハンドリング**: 存在しないディレクトリ指定時の適切なエラー
3. **メタデータ自動設定**: dataset-metadata.json自動作成と設定値置換
4. **新規データセット作成**: `cmi-{experiment_name}`形式での自動作成
5. **既存データセット更新**: バージョン更新処理の自動実行

### 使用方法
```bash
# ヘルプの確認
mise run create-output-dataset --help

# exp007の実験結果をデータセット化
mise run create-output-dataset exp007
```

## 備考
- miseの引数テンプレート機能により、自動的な引数バリデーションが実現
- dataset-metadata.jsonが存在しない場合は自動的に`cmi-{experiment_name}`形式で設定
- データセット名は Kaggle の命名規則に準拠（ハイフンを使用、アンダースコア不可）
- `jq`コマンドを使用したJSON自動編集により、手動設定が不要

## テスト結果

### create-output-dataset
- **exp008でのテスト**: 正常にデータセット作成とバージョン更新が動作確認済み
- **作成されたデータセット**: https://www.kaggle.com/datasets/karunru/cmi-exp008
- **自動設定値**: title: "cmi-exp008", id: "karunru/cmi-exp008"

### update-codes  
- **初回実行テスト**: dataset-metadata.json自動作成と設定が正常動作
- **作成/更新されたデータセット**: https://www.kaggle.com/datasets/karunru/cmi-codes
- **自動設定値**: title: "cmi-codes", id: "karunru/cmi-codes"
- **アップロード内容**: codes/ディレクトリ内の全ファイル（__init__.py, exp.zip, src.zip等）

## update-codesタスクの詳細

### 機能
- codes/ディレクトリ全体を1つのKaggleデータセットとして管理
- 引数不要（codes全体が対象）
- dataset-metadata.jsonが存在しない場合は自動生成・設定
- 既存データセットがある場合はバージョン更新

### 使用方法
```bash
# codesディレクトリをデータセット化/更新
mise run update-codes
```

### 自動設定内容
- **title**: "cmi-codes"（固定）
- **id**: "{USERNAME}/cmi-codes"（環境変数USERNAMEを使用）
- **license**: CC0-1.0（デフォルト）