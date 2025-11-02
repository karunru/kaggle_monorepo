# wandb_logger._prefix実装のまとめ

## 実装内容

exp007/train.pyにおいて、docs/train_pl.pyを参考に`wandb_logger._prefix`を使ったfold毎のログ分離を実装しました。

## 変更内容

### 1. train_cross_validation関数の修正
- 単一のWandBロガーを作成する形に変更
- `wandb_config`未定義のバグ修正（line 250）
- WandBロガーを各foldに渡すようにした

### 2. train_single_fold関数の修正
- WandBロガーをパラメータとして受け取るように変更
- fold開始時に`wandb_logger._prefix = f"fold_{fold}"`を設定
- モデルパラメータのログ処理を簡潔化

### 3. setup_loggers関数の修正
- WandBロガー作成部分を削除
- 引数にwandb_loggerを追加し、提供された場合のみロガーリストに追加

### 4. ログ出力の調整
- fold固有のメトリクスログを`_prefix`を使った形式に変更
- `f"fold_{fold}/final_val_loss"`から`"final_val_loss"`へ（_prefixが自動付与）

### 5. 静的解析エラーの修正
- 全角括弧を半角括弧に変更
- `logger.error`を`logger.exception`に変更
- 例外メッセージの冗長な文字列を削除

## 変更ファイル
- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/exp/exp007/train.py`

## 実装の効果
- docs/train_pl.pyと同じパターンでfold毎のログが分離される
- 単一のWandBランで全foldのログが管理される
- 既存のバグが修正され、コードがよりクリーンになった