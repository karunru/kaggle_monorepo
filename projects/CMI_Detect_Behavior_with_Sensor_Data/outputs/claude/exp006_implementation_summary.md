# exp006 実装サマリー：Switch EMA統合

## 概要
exp005をベースに、ema-pytorchライブラリを使用してSwitch EMA（Exponential Moving Average）機能を統合したexp006を実装しました。

## 主要な変更点

### 1. 実験番号の一元化
- `EXP_NUM = "exp006"` を定義し、全ての参照箇所で使用
- 環境変数プレフィックス、出力ディレクトリ、ログファイル名等を自動生成

### 2. EMA設定の追加
```python
class EMAConfig(BaseModel):
    """EMA（Exponential Moving Average）設定."""
    enabled: bool = Field(default=True, description="EMA使用フラグ")
    beta: float = Field(default=0.9999, description="EMA減衰係数")
    update_after_step: int = Field(default=1000, description="EMA更新開始ステップ")
    update_every: int = Field(default=10, description="EMA更新頻度")
    update_model_with_ema_every: int = Field(default=1000, description="Switch EMA更新頻度")
    use_ema_for_validation: bool = Field(default=True, description="検証時にEMAモデルを使用")
```

### 3. Length Grouping無効化
- `LengthGroupingConfig.enabled = False` （デフォルト設定）
- 実装はそのまま保持し、設定で無効化

### 4. Switch EMA実装
- `model.py` に EMA統合:
  - training_step での自動更新
  - on_train_epoch_end でのSwitch EMA適用
- `train.py` でEMA設定をモデルに伝送

### 5. テストの追加
- `test_ema_config()`: EMA設定クラスのテスト
- `test_ema_imports()`: ema-pytorchライブラリの利用可能性テスト
- `test_ema_integration()`: EMA統合の動作テスト

## 実装ファイル

### 修正されたファイル
1. **config.py**
   - EXP_NUM一元化
   - EMAConfig追加
   - 全てのexp005参照をexp006に変更

2. **model.py**
   - ema-pytorchライブラリのインポート
   - _setup_ema()メソッド追加
   - training_stepでのEMA更新
   - on_train_epoch_endでのSwitch EMA適用

3. **dataset.py**
   - ヘッダーコメント更新

4. **train.py**
   - ema_configをモデルに渡す処理追加

5. **test_exp006.py**
   - EMA関連テスト3つ追加
   - 全てのexp005参照を削除

6. **inference.py**
   - ヘッダーコメント更新

### 実行スクリプトの更新
- `run_experiments_simple.sh` にexp006を追加

## 動作確認結果

### テスト結果
```
Tests passed: 13/13
🎉 All tests passed!
```

### 統合テスト
- EMAインスタンスの正常作成確認
- Switch EMA機能の動作確認
- モデルの前向きパス正常動作

## 修正履歴

### 1. RecursionError修正（手動EMA実装）
**問題**: ema-pytorchライブラリでの自己参照によるRecursionError
```
RecursionError: maximum recursion depth exceeded
```

**解決策**: 手動EMA実装に変更
- `EMA_AVAILABLE = True`（常に利用可能）
- `_setup_manual_ema()`メソッドでEMAパラメータ辞書を作成
- `_update_ema_params()`でEMA更新処理
- `_apply_ema_to_model()`でSwitch EMA適用

### 2. デバイス不一致エラー修正
**問題**: EMAパラメータ（CPU）とモデルパラメータ（CUDA）のデバイス不一致
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**解決策**: デバイス管理の強化
- `_setup_manual_ema()`でパラメータ作成時にデバイス指定
- `_update_ema_params()`でデバイス不一致検出・修正
- `_apply_ema_to_model()`でデバイス不一致検出・修正

```python
# デバイス不一致の自動修正
if self.ema_params[name].device != param.device:
    self.ema_params[name] = self.ema_params[name].to(param.device)
```

### 最終動作確認
- RecursionError: 修正完了 ✓
- デバイスエラー: 修正完了 ✓
- 手動EMA正常動作: 確認済み ✓
- 訓練正常開始: 確認済み ✓

## 期待される効果
1. **モデルの安定化**: EMAによる重みの平滑化
2. **汎化性能の向上**: Switch EMAによる学習と推論の最適化
3. **実装の簡素化**: Length Grouping無効化による複雑性の低減
4. **安定性の向上**: RecursionErrorとデバイスエラーの修正

## 次のステップ
- 実際の訓練実行による性能評価
- EMAパラメータ（beta、更新頻度）の最適化
- 他の実験との性能比較
- 本格的な長時間訓練の実行