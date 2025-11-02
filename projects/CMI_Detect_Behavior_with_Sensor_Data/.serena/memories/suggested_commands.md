# 推奨コマンド一覧

## 基本的な開発コマンド

### パッケージ管理
```bash
# 依存関係追加
uv add package_name

# 開発依存関係追加  
uv add --dev package_name

# パッケージ実行
uv run command
```

### 静的解析・テスト（../../.mise.toml）
```bash
# コードフォーマット
mise run format

# リント（自動修正付き）
mise run lint

# 型チェック
mise run type-check

# テスト実行
mise run test

# 全チェック実行
mise run all
```

### CI用コマンド（チェックのみ）
```bash
# フォーマットチェック
mise run ci-format

# リントチェック
mise run ci-lint

# 型チェック
mise run ci-type-check
```

## Kaggle関連コマンド

### データセット管理
```bash
# コードをKaggleデータセットとしてアップロード
mise run update-codes

# requirements.txtを更新してアップロード
mise run update-requirements

# 依存関係ノートブックをアップロード
mise run update-deps

# サブミッション用ノートブックをアップロード  
mise run update-subs
```

### 実験結果管理
```bash
# 実験結果をKaggleデータセットとして作成/更新
mise run create-output-dataset experiment_name=exp007
```

### サブミッション追跡
```bash
# サブミッション状況をトラッキング
mise run track-submission [competition_name]
# デフォルト: cmi-detect-behavior-with-sensor-data
```

## 実験実行コマンド

### 訓練実行
```bash
# 実験ディレクトリで実行
cd codes/exp/exp007
uv run python train.py

# 推論実行
uv run python inference.py
```

### 簡易実験スクリプト
```bash
# 複数実験の一括実行
cd codes
./run_experiments_simple.sh
```

## Git操作
```bash
# 標準的なGitコマンドが利用可能
git status
git add .
git commit -m "message"
git push
```

## ファイル操作
```bash
# Linux標準コマンド
ls, cd, grep, find, cat, head, tail
mkdir, cp, mv, rm
```

## ユーティリティ
```bash
# JSON処理（miseタスクで使用）
jq

# プロセス確認
ps, nvidia-smi
```