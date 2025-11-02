# mise.toml PROJECT_NAME自動取得への変更実装

## 実装概要

mise.tomlファイルのすべてのタスクで、PROJECT_NAME引数を手動で指定する代わりに、現在のディレクトリパスから自動的にプロジェクト名を取得するように変更しました。

## 変更内容

### 変更前
```bash
PROJECT_NAME={{arg(name='project_name', var=true)}}
```

### 変更後
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
CURRENT_PATH=$(pwd)
REL_PATH=${CURRENT_PATH#${REPO_ROOT}/projects/}
PROJECT_NAME=$(echo ${REL_PATH} | cut -d'/' -f1)
```

### 自動取得ロジックの説明
1. `git rev-parse --show-toplevel`: リポジトリのルートディレクトリを取得
2. `pwd`: 現在の作業ディレクトリを取得
3. `${CURRENT_PATH#${REPO_ROOT}/projects/}`: projects/ 以降の相対パスを抽出
4. `cut -d'/' -f1`: 最初のディレクトリ名（プロジェクト名）を取得

## 変更されたタスク（9個）

### 既存タスク（3個）
1. **update-requirements** - requirements.txtの更新とデータセットバージョン更新
2. **update-deps** - depsカーネルのプッシュ
3. **update-subs** - subsカーネルのプッシュ

### 新規追加タスク（6個）
4. **init-codes-dataset** - codesデータセット初期化
5. **init-deps-notebook** - depsノートブック初期化
6. **init-subs-notebook** - subsノートブック初期化
7. **update-codes** - codesデータセット更新
8. **download-deps** - 依存関係wheelファイルダウンロード
9. **create-requirements-dataset** - requirementsデータセット作成/更新

## 使用例の変更

### 変更前（引数指定が必要）
```bash
mise run update-deps project_name=CMI_Detect_Behavior_with_Sensor_Data
mise run update-subs project_name=CMI_Detect_Behavior_with_Sensor_Data
mise run download-deps project_name=CMI_Detect_Behavior_with_Sensor_Data packages="numpy pandas"
```

### 変更後（引数指定不要）
```bash
mise run update-deps
mise run update-subs
mise run download-deps packages="numpy pandas"  # packagesのみ指定
```

## 特殊ケース

### download-depsタスク
- `PROJECT_NAME`引数は削除
- `PACKAGES`引数は残存（パッケージ名の指定が必要なため）

### 引数が不要なタスク
以下のタスクは引数が全く不要になりました：
- `check-notebook-status` (notebook_id引数は残存)
- `list-notebooks` (username引数は残存)
- `list-datasets` (username引数は残存)  
- `list-models` (username引数は残存)
- `upload-model-weights` (model_dir, model_name引数は残存)

## メリット

1. **利便性向上**: 長いプロジェクト名を毎回入力する必要がなくなった
2. **エラー削減**: プロジェクト名のタイプミスが防げる
3. **一貫性**: 現在のディレクトリから確実にプロジェクト名を取得
4. **作業効率**: タスク実行時のコマンドが簡潔になった

## 注意事項

- プロジェクトディレクトリ（`projects/PROJECT_NAME/`）内で実行する必要がある
- `projects/`配下でない場所で実行すると正しく動作しない可能性がある
- gitリポジトリ内での実行が前提（`git rev-parse`を使用）

## 実装済みファイル

- `/mise.toml`: 全9タスクのPROJECT_NAME自動取得への変更

## テスト推奨

各タスクが正常に動作することを確認してください：

```bash
# プロジェクトディレクトリに移動
cd projects/CMI_Detect_Behavior_with_Sensor_Data/

# テスト実行例
mise run update-deps
mise run update-subs
mise run list-notebooks username=your_username
```