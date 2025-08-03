# kaggle_evaluation モジュールimport修正まとめ

## 概要
`codes/src/kaggle_evaluation/`配下のファイルで発生していた`ModuleNotFoundError: No module named 'kaggle_evaluation'`エラーを修正しました。

## 問題の原因
`codes/src/kaggle_evaluation/`内のファイルが絶対import（`import kaggle_evaluation.xxx`）を使用していたため、モジュールパス解決に失敗していました。

## 修正内容

### 1. cmi_inference_server.py
```python
# 修正前
import kaggle_evaluation.core.templates
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cmi_gateway

# 修正後
from . import cmi_gateway
from .core import templates
```

### 2. cmi_gateway.py
```python
# 修正前
import kaggle_evaluation.core.templates
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType

# 修正後
from .core import templates
from .core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType
```

### 3. core/templates.py
```python
# 修正前
import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.relay

# 修正後
from . import base_gateway
from . import relay
```
その他の参照箇所も相対import形式に修正

### 4. core/base_gateway.py
```python
# 修正前
import kaggle_evaluation.core.relay

# 修正後
from . import relay
```
その他の参照箇所も相対import形式に修正

### 5. core/relay.py
```python
# 修正前
import kaggle_evaluation.core.generated.kaggle_evaluation_pb2 as kaggle_evaluation_proto
import kaggle_evaluation.core.generated.kaggle_evaluation_pb2_grpc as kaggle_evaluation_grpc

# 修正後
from .generated import kaggle_evaluation_pb2 as kaggle_evaluation_proto
from .generated import kaggle_evaluation_pb2_grpc as kaggle_evaluation_grpc
```

### 6. core/generated/kaggle_evaluation_pb2_grpc.py
```python
# 修正前
import kaggle_evaluation_pb2 as kaggle__evaluation__pb2

# 修正後
from . import kaggle_evaluation_pb2 as kaggle__evaluation__pb2
```

### 7. inference.py (エイリアス追加)
```python
# 修正後
if os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local") != "local":
    import kaggle_evaluation.cmi_inference_server
    kaggle_evaluation_module = kaggle_evaluation.cmi_inference_server
else:
    import codes.src.kaggle_evaluation.cmi_inference_server
    kaggle_evaluation_module = codes.src.kaggle_evaluation.cmi_inference_server

# 使用箇所
inference_server = kaggle_evaluation_module.CMIInferenceServer(predict)
```

## 結果
- ✅ inference.pyが正常に起動
- ✅ モデル読み込み処理が正常に実行
- ✅ CMIInferenceServerの作成が成功
- ✅ run_local_gatewayの実行開始まで成功

## 注記
実行時に「invalid index to scalar variable.」エラーが発生していますが、これは訓練済みモデルが存在しないためのpredict関数内でのエラーで、import修正とは無関係です。import修正の主目的は達成されています。

## メリット
1. **相対importの使用**: パッケージ構造に依存しない
2. **sys.pathハック削除**: より安全で保守性の高いコード
3. **一貫性**: 他のプロジェクトコードと同様のimport形式