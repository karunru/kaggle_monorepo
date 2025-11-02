# Kaggle Code Submission用mise.tomlタスク追加の実装

## 実装概要

ブログ記事 https://ho.lc/blog/kaggle_code_submission/ を参考に、既存のmise.tomlファイルに不足していたKaggleワークフロー用のタスクを追加しました。

## 追加されたタスク

### 初期化系タスク

#### 1. `init-codes-dataset`
- **説明**: codesデータセットの初期化
- **機能**: 
  - codesディレクトリの作成
  - `kaggle d init`によるdataset-metadata.jsonの初期化
  - 初回作成時の手順案内

#### 2. `init-deps-notebook`
- **説明**: depsノートブックの初期化
- **機能**:
  - depsディレクトリの作成
  - `kaggle k init`によるkernel-metadata.jsonの初期化
  - 初回プッシュの手順案内

#### 3. `init-subs-notebook`
- **説明**: subsノートブックの初期化
- **機能**:
  - subsディレクトリの作成
  - `kaggle k init`によるkernel-metadata.jsonの初理化
  - 初回プッシュの手順案内

### 更新系タスク

#### 4. `update-codes`
- **説明**: codesデータセットの更新
- **機能**:
  - dataset-metadata.jsonの存在確認
  - `kaggle d version -m "update" -r zip`によるバージョンアップ

### ステータス確認系タスク

#### 5. `check-notebook-status`
- **説明**: ノートブックの実行ステータス確認
- **引数**: `notebook_id` - 確認したいノートブックのID
- **機能**: `kaggle k status`による実行状態確認

#### 6. `list-notebooks`
- **説明**: 自分のノートブック一覧表示
- **引数**: `username` - Kaggleユーザ名
- **機能**: `kaggle k list -m -s`による一覧表示

#### 7. `list-datasets`
- **説明**: 自分のデータセット一覧表示
- **引数**: `username` - Kaggleユーザ名
- **機能**: `kaggle d list -m -s`による一覧表示

#### 8. `list-models`
- **説明**: アップロードしたモデル一覧表示
- **引数**: `username` - Kaggleユーザ名
- **機能**: `kaggle m list --owner`による一覧表示

### 依存関係管理系タスク

#### 9. `download-deps`
- **説明**: Pythonパッケージ依存関係のwheelファイルダウンロード
- **引数**: 
  - `project_name` - プロジェクト名
  - `packages` - ダウンロードするパッケージ名
- **機能**: `pip download`による依存関係のダウンロード

#### 10. `upload-model-weights`
- **説明**: モデルの重みをKaggle Modelsにアップロード
- **引数**:
  - `model_dir` - モデルファイルのディレクトリ
  - `model_name` - モデル名
- **機能**: kagglehubまたはKaggle CLIを使用したアップロード手順の案内

#### 11. `create-requirements-dataset`
- **説明**: requirementsデータセットの作成/更新
- **引数**: `project_name` - プロジェクト名
- **機能**:
  - `uv pip freeze`による現在の環境の出力
  - requirements.txtの生成
  - データセットの初期化または更新

## 既存タスクとの関係

既存の以下のタスクと組み合わせて、完全なKaggleワークフローをカバーします：

- `update-requirements`: requirements.txtの更新とデータセットのバージョン更新
- `update-deps`: depsカーネルのプッシュ
- `update-subs`: subsカーネルのプッシュ

## 使用例

```bash
# 初期セットアップ
mise run init-codes-dataset project_name=my-project
mise run init-deps-notebook project_name=my-project
mise run init-subs-notebook project_name=my-project

# 日常的な更新作業
mise run update-codes project_name=my-project
mise run update-deps project_name=my-project
mise run update-subs project_name=my-project

# ステータス確認
mise run list-notebooks username=myusername
mise run check-notebook-status notebook_id=mynotebook/version/1

# 依存関係管理
mise run download-deps project_name=my-project packages="numpy pandas"
mise run create-requirements-dataset project_name=my-project
```

## 実装済みファイル

- `/mise.toml`: 新しいタスクを追加（既存タスクは保持）

## 次のステップ

1. 実際のプロジェクトでこれらのタスクをテストする
2. 必要に応じてタスクの調整やエラーハンドリングの改善
3. プロジェクト固有の要件に合わせたカスタマイズ