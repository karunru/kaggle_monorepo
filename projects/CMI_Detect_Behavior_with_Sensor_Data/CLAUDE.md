# 基本設定
- 出力言語：日本語
- 思考言語: 英語
- コードスタイル：PEP 8準拠
- ドキュメント形式：Markdown

# プロジェクト構造
- プロジェクトのテンプレートは以下の通りになっています。プロジェクトによっては異なることがあるため、不明なディレクトリもしくはファイルがあれば、ユーザにCLAUDE.mdに記載を促すように指摘してください。
```
- data/       # kaggleのデータセットなどのデータはここに置く
- outputs/    # 実験の出力やclaudeの計画などの成果物をここに置く
  ├── claude/ # Claudeが作成したドキュメント・計画
  │   └── tmp/ # uv run kaggleコマンドで取得したnotebookやdiscussionなどの一時ファイルを置く
  ├── exp002/ # exp002の実験出力（モデル、ログ等）
  ├── exp003/ # exp003の実験出力
  ├── exp004/ # exp004の実験出力
  ├── exp005/ # exp005の実験出力
  ├── exp006/ # exp006の実験出力
  └── exp007/ # exp007の実験出力
- codes/      # メインのコードディレクトリ
  ├── src/    # 各実験で共通するコードはここに置く
  │   ├── ensemble/         # アンサンブル手法
  │   │   └── blending.py   # 重み付きアンサンブル最適化（ランダムサーチベース）
  │   ├── evaluation/       # 評価・最適化関連
  │   │   ├── metrics.py    # 競技用評価指標（Concordance Index for EEFS）
  │   │   ├── cat.py        # CatBoost専用評価ラッパー
  │   │   ├── lgbm.py       # LightGBM専用評価ラッパー
  │   │   ├── optimization.py # ハイパーパラメータ最適化
  │   │   └── truncate.py   # データ切り詰め処理
  │   ├── features/         # 特徴量エンジニアリング（Polars/Arrow対応）
  │   │   ├── base.py       # 抽象基底クラス（自動保存/ロード、圧縮対応）
  │   │   ├── statistics.py # 統計的特徴量
  │   │   ├── category_vectorizer.py # カテゴリカル変数ベクトル化
  │   │   ├── category_w2v.py # Word2Vecベースカテゴリ特徴量
  │   │   ├── kaplan_meier_target.py # Kaplan-Meier生存分析特徴量
  │   │   ├── groupby_concat_cat.py # グループ別カテゴリ連結特徴量
  │   │   └── [その他]      # basic_fix_le, modules など
  │   ├── kaggle_evaluation/ # Kaggle評価用のgRPCインターフェース
  │   │   ├── cmi_gateway.py # CMIコンペ用ゲートウェイ
  │   │   ├── cmi_inference_server.py # CMI推論サーバー
  │   │   └── core/         # 評価コア機能
  │   ├── models/           # 機械学習モデル（GPU最適化）
  │   │   ├── base.py       # 抽象基底クラス（CV、前処理、事後処理統合）
  │   │   ├── factory.py    # モデルファクトリ（LightGBM, XGBoost, CatBoost等）
  │   │   ├── lightgbm.py   # LightGBMモデル
  │   │   ├── xgb.py        # XGBoostモデル
  │   │   ├── cat.py        # CatBoostモデル
  │   │   ├── tabnet.py     # TabNetモデル
  │   │   ├── gnn.py        # Graph Neural Network
  │   │   ├── litnn.py      # Pairwise Ranking Neural Network
  │   │   └── [その他]      # SVM, Ridge, RGF, ERT, KTBoost等
  │   ├── sampling/         # 不均衡データ対応サンプリング
  │   │   └── factory.py    # SMOTE、アンダーサンプリング、GPU対応
  │   ├── utils/            # ユーティリティ関数
  │   │   ├── config.py     # YAML設定管理（階層化設定対応）
  │   │   ├── logger.py     # 集約ログ管理（ファイル/コンソール出力）
  │   │   ├── timer.py      # 処理時間計測
  │   │   ├── seed_everything.py # シード設定
  │   │   ├── slack.py      # Slack通知
  │   │   ├── file_io.py    # ファイルI/O
  │   │   ├── visualization.py # データ可視化
  │   │   └── [その他]      # 引数解析、チェック機能等
  │   └── validation/       # クロスバリデーション戦略
  │       └── factory.py    # 時系列対応CV（滑動窓、日付ベース、曜日調整）
  └── exp/                  # 各実験のコードはここに置く
      ├── exp001/           # 初期実験
      ├── exp002/           # Squeezeformer実装（PyTorch Lightning）
      ├── exp003/           # 損失関数改善
      ├── exp004/           # 最適化手法改善
      ├── exp005/           # 長さグループ化
      ├── exp006/           # Switch EMA実装
      └── exp007/           # 欠損値attention mask処理
- deps/       # Kaggle依存関係管理用ノートブック
- docs/       # プロジェクトドキュメント
- requirements/ # Kaggleデータセット用の要件ファイル
- sub/        # Kaggleサブミッション用ノートブック
- tests/      # テストコード
- mise.toml   # タスク自動化設定（update-requirements, update-codes, track-submission等）
- pyproject.toml # Pythonプロジェクト設定（依存関係、ツール設定）
- CLAUDE.md   # Claudeのための指示書（このファイル）
- .gitignore  # Git除外設定
```

**注記**: 
- 現在のsrc/ディレクトリは別のKaggleプロジェクト（生存分析系）のコードベースと、CMIプロジェクト用の新しいコード（kaggle_evaluation等）が混在しています。
- exp002-007は、CMIプロジェクト用のSqueezeformerベースの実装です。
- miseタスクを使用してKaggleへのコードやデータセットのアップロードが自動化されています（`mise run update-codes`など）。

# 技術スタック
- 既存のコードは親ディレクトリに存在しますが、データが多すぎるためユーザが指定した場合のみ参照してください
- ユーザの指定がない場合は、以下の技術スタックを利用してください
  - 言語：Python
  - パッケージマネージャ：uv
    - パッケージの追加は`uv add xxx`で実施してください
  - ライブラリ：pyproject.tomlを参照
  - テスト：pytest
  - 静的解析：ruff, mypy

# 行動指針
## タスクの進め方
1. ユーザの依頼に対して、そのタスクをどのように実現できるか計画を立て、その計画をまずドキュメントにまとめます
  - ドキュメントは`outputs/claude`にユーザの依頼とそれに対する計画がわかるようにmarkdown形式でまとめます
  - 再度計画を立てると数＄消費されるのを防ぐため、必ず計画はドキュメントにまとめてください
  - タスクの分割粒度ですが、各タスクごとにテストコードを記載できる程度の粒度でまとめてください
  - 注意：ユーザの指示の範囲内のみでタスクを考え、指示を超えた実装内容は絶対に含めないでください。
2. ドキュメントが完成したら、計画の通りにタスクを実装します
  - タスクのスコープから外れる実装は絶対に行わないでください
  - タスクの実装が完了したら、必ずテストコードを記載してください
3. 実装が完了したら、静的解析、テストを実行し、それらが通るように修正します
  - 静的解析、テストの行動指針は「## 静的解析」、「## テスト」のそれぞれ記載しているので、それを参照してください。
4. 静的解析、テストが通ったら、成果物のファイル名とどのような内容が記載されているかをドキュメントにまとめます
  - 記載するドキュメントは1.のドキュメントと同じものを利用してください
5. ドキュメントが完成したら、2.に戻り、次のタスクを実施します

## 静的解析
- linter/formatterにはruffを利用しています。ruffの設定は../../ruff.tomlにまとめられているので、それを参照してください。
- 型チェックにはmypyを利用しています。mypyの設定は../../mypy.iniにまとめられているので、それを参照してください。
- 静的解析が通らない場合CIが落ちるため、実装するコードは必ず静的解析が通るように実装または修正してください。
- 3回以上静的解析が落ちる場合、ユーザに対して解析エラーが発生している旨を通知し、実装を中止してください。
- 静的解析のコマンドは../../.mise.tomlに記載されています。各静的解析のコマンドは以下の通りです。
  - code format: `mise run format`
  - code lint: `mise run lint`
  - type check: `mise run type-check`

## テスト
- テストはpytestを利用しています。pytestの設定は../../pytest.iniにまとめられているので、それを参照してください。
- テストが通らない場合、CIが落ちるため、実装するコードは必ずテストが通るように実装または修正してください。
- 3回以上テストが落ちる場合、ユーザに対してテストが通らない旨を通知し、実装を中止してください。
- テストのコマンドは../../.mise.tomlに記載されています。
- `mise run test`を実行することで、`tests`配下のテストコードが実行されます。
