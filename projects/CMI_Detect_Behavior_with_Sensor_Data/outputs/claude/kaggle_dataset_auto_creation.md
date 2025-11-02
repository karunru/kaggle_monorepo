# Kaggleデータセット自動作成機能の実装

## 概要
`uv run kaggle d version`コマンドがデータセットが存在しない場合にエラーになる問題を解決し、存在しない場合は自動的に`create`するような機能を実装しました。

## 実装内容

### 1. 更新されたタスク

#### update-requirements タスク
`mise.toml`の`update-requirements`タスクを以下のように更新：
- `dataset-metadata.json`からデータセットIDを抽出
- `kaggle d status`でデータセットの存在を確認
- 存在しない場合（403エラー）は`kaggle d create`で新規作成
- 存在する場合は`kaggle d version`でバージョン更新

#### update-codes タスク
同様の存在確認と自動作成ロジックを追加

### 2. 実装詳細

```bash
# データセットIDの取得
DATASET_ID=$(jq -r '.id' dataset-metadata.json)

# データセット存在確認と条件分岐
if ! uv run kaggle d status $DATASET_ID 2>/dev/null; then
    echo "Dataset not found. Creating new dataset..."
    uv run kaggle d create -p . -r zip
else
    echo "Dataset exists. Updating version..."
    uv run kaggle d version -m "update" -r zip
fi
```

### 3. エラーハンドリング
- `dataset-metadata.json`が存在しない場合はエラーメッセージを表示して終了
- 標準エラー出力を`/dev/null`にリダイレクトして、403エラーの詳細を非表示に

## 使用方法

### requirements データセットの更新
```bash
mise run update-requirements
```

### codes データセットの更新
```bash
mise run update-codes
```

## 動作フロー

1. タスク実行時に`dataset-metadata.json`の存在確認
2. jqコマンドでデータセットIDを抽出
3. `kaggle d status`でデータセットの存在確認
4. 存在しない場合：
   - "Dataset not found. Creating new dataset..."を表示
   - `kaggle d create -p . -r zip`で新規作成
5. 存在する場合：
   - "Dataset exists. Updating version..."を表示
   - `kaggle d version -m "update" -r zip`でバージョン更新

## 必要な前提条件

- Kaggle APIキーが設定されていること
- jqコマンドがインストールされていること
- `dataset-metadata.json`が正しく配置されていること

## 利点

- データセットが存在しない場合でも自動的に作成されるため、初回実行時のエラーを回避
- 既存のデータセットがある場合は通常通りバージョン更新
- エラーメッセージがわかりやすく、デバッグが容易

## 更新されたファイル

- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/mise.toml`
  - `update-requirements`タスク
  - `update-codes`タスク