# 基本設定
- 出力言語：日本語
- 思考言語: 英語
- コードスタイル：PEP 8準拠
- ドキュメント形式：Markdown

# プロジェクト構造
- プロジェクトのテンプレートは以下の通りになっています。プロジェクトによっては異なることがあるため、不明なディレクトリもしくはファイルがあれば、ユーザにCLAUDE.mdに記載を促すように指摘してください。
```
- data/    # kaggleのデータセットなどのデータはここに置く
- outputs/ # 実験の出力やclaudeの計画などの成果物をここに置く (e.g., logs, model weights)
- src/     #  各実験で共通するコードはここに置く (e.g., src/dataset.py, src/model.py)
- exp/     #  各実験のコードはここに置く (e.g., exp001/train.py, exp001/config.yaml)
- tests/   # ここにテストコードを置く
```

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
