# timer.pyへのデコレータータイマー追加実装

## 概要
既存のcontextmanagerタイマーに加えて、関数デコレーターとして使えるタイマー機能を実装しました。

## 実装内容

### 1. timer_decorator関数
`src/utils/timer.py`に`timer_decorator`関数を追加：

```python
def timer_decorator(name: str = None, log: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            t0 = time.time()
            msg = f"[{timer_name}] start"
            print(msg)
            
            if log:
                logging.info(msg)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                msg = f"[{timer_name}] done in {time.time() - t0:.2f} s"
                print(msg)
                
                if log:
                    logging.info(msg)
        
        return wrapper
    return decorator
```

### 2. 主な機能
- **カスタム名称対応**: `name`パラメータで任意の名前を指定可能
- **自動名称設定**: `name`未指定時は関数名を自動使用
- **ログ制御**: `log`パラメータでログ出力のON/OFF制御
- **例外安全**: `finally`ブロックで例外発生時も時間計測を保証
- **メタデータ保持**: `functools.wraps`で元の関数の属性を保持

## 使用例

### 基本的な使用方法
```python
# カスタム名を指定
@timer_decorator("データ処理")
def process_data():
    # 処理内容
    pass

# 関数名を自動使用
@timer_decorator()
def calculate_metrics():
    # 計算処理
    pass

# ログを無効化
@timer_decorator(log=False)
def quick_operation():
    # 高速処理
    pass
```

### 引数付き関数での使用
```python
@timer_decorator("加算処理")
def add(a, b):
    return a + b

result = add(10, 20)  # [加算処理] start/done が出力される
```

### 既存のcontextmanagerとの使い分け
```python
# contextmanager版：コードブロックの計測
with timer("バッチ処理"):
    data = load_data()
    process_data(data)
    save_results(data)

# デコレーター版：関数全体の計測
@timer_decorator("データロード")
def load_data():
    # ロード処理
    return data
```

## テスト実装
`tests/test_timer.py`に包括的なテストを実装：

### contextmanagerのテスト
- 基本的な動作確認
- ログ出力の確認（有効/無効）
- 例外処理時の動作確認

### デコレーターのテスト
- カスタム名指定の動作確認
- 自動名称設定の動作確認
- 引数・キーワード引数付き関数の動作確認
- ログ出力の確認（有効/無効）
- 関数メタデータの保持確認
- 例外処理時の動作確認

## 利点

1. **コードの簡潔性**: デコレーターで関数の計測が1行で可能
2. **柔軟性**: contextmanagerとデコレーターを用途に応じて選択可能
3. **保守性**: 計測コードを関数本体から分離
4. **一貫性**: 両方式で同じ出力形式を維持

## 更新されたファイル

- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/src/utils/timer.py`
  - `timer_decorator`関数を追加
  - `functools`モジュールをインポート

- `/home/karunru/Home/Kaggle/kaggle_monorepo/projects/CMI_Detect_Behavior_with_Sensor_Data/tests/test_timer.py`
  - contextmanagerとデコレーター両方のテストを実装