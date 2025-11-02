# Exp031 実装計画書

## 概要
`codes/exp/exp030` を `codes/exp/exp031` にコピーして、モデルのReLU/reluをMishに置き換える実験を実装する。

## コンペティション背景
- **コンペ名**: CMI - Detect Behavior with Sensor Data
- **目的**: 手首装着デバイスのセンサーデータからBFRB様行動とnon-BFRB様行動を分類
- **データ**: IMU、thermopile、time-of-flightセンサーの時系列データ
- **評価指標**: Binary F1とMacro F1の平均

## 実装方針

### 1. 基本方針
- `codes/exp/exp030` を `codes/exp/exp031` にコピー
- test_*.py ファイルは除外
- `codes/exp/exp030` からのインポートは禁止
- 差分は指示された内容以外最小限に留める

### 2. 主要変更内容

#### 2.1 活性化関数の置き換え（ReLU → Mish）
以下の5箇所でReLU/reluをMishに置き換える：

1. **SEBlock (line 237)**
   - 変更前: `nn.ReLU(inplace=True)`
   - 変更後: `nn.Mish(inplace=True)`

2. **ResidualSECNNBlock (line 280)**
   - 変更前: `F.relu(self.bn1(self.conv1(x)))`
   - 変更後: `F.mish(self.bn1(self.conv1(x)))`

3. **ResidualSECNNBlock (line 289)**
   - 変更前: `F.relu(out)`
   - 変更後: `F.mish(out)`

4. **IMUOnlyLSTM dense layers (line 412)**
   - 変更前: `F.relu(self.bn_dense1(self.dense1(attended)))`
   - 変更後: `F.mish(self.bn_dense1(self.dense1(attended)))`

5. **IMUOnlyLSTM dense layers (line 414)**
   - 変更前: `F.relu(self.bn_dense2(self.dense2(x)))`
   - 変更後: `F.mish(self.bn_dense2(self.dense2(x)))`

#### 2.2 設定ファイルの更新（config.py）

**ExperimentConfig の更新**:
- `description`: "Mish activation function implementation with IMU-only LSTM, human normalization, and demographics"
- `tags`: `"mish"` を追加、`"relu"` 関連は削除

**LoggingConfig の更新**:
- `wandb_tags`: `"mish"` を追加、`"relu"` 関連は削除

#### 2.3 その他の変更
- `EXP_NUM` を "exp031" に変更
- インポート文に必要に応じて `F.mish` または `nn.Mish` を追加

## 実装タスク一覧

### タスク1: ディレクトリ作成とファイルコピー
- [ ] `codes/exp/exp031` ディレクトリを作成
- [ ] `codes/exp/exp030` から `codes/exp/exp031` にファイルをコピー（test_*.py 除く）

### タスク2: モデル実装（model.py）
- [ ] ReLU → Mish の置き換え（5箇所）
- [ ] インポート文の更新

### タスク3: 設定ファイル更新（config.py）
- [ ] `EXP_NUM` の更新
- [ ] `ExperimentConfig.description` の更新
- [ ] `ExperimentConfig.tags` の更新（"mish"追加）
- [ ] `LoggingConfig.wandb_tags` の更新（"mish"追加）

### タスク4: テストコード作成
- [ ] `tests/test_exp031_mish_activation.py` の作成
- [ ] Mishの動作確認テスト
- [ ] モデルの出力確認テスト

### タスク5: 静的解析・テスト実行
- [ ] `mise run format` - コードフォーマット
- [ ] `mise run lint` - リンター実行
- [ ] `mise run type-check` - 型チェック
- [ ] `mise run test` - テスト実行

## 期待される効果
- Mish活性化関数によるモデル性能の向上
- 特に勾配消失問題の改善とより滑らかな活性化関数の利用効果
- 非線形性の向上によるBFRB検出精度の改善

## 注意事項
- PyTorchのバージョンに応じて `F.mish` が使用できない場合は `nn.Mish()` で代替
- 損失関数内の `F.relu` は数学的操作のため変更対象外
- exp030からのインポートは絶対に行わない
- 他のexp directoryの編集は禁止

## 実装完了の判定基準
1. すべての静的解析（format, lint, type-check）が通過
2. すべてのテストが通過
3. モデルが正常に動作し、学習・推論が可能
4. 設定ファイルの項目が適切に更新されている

---

# 実装完了報告

## 実装概要
- **実装日**: 2025-08-17
- **実装者**: Claude AI
- **実装内容**: exp030からexp031への移行（ReLU → Mish活性化関数）

## 実装結果

### ✅ 完了したタスク

#### 1. ファイル構成
```
codes/exp/exp031/
├── __init__.py
├── config.py          # 設定更新済み
├── dataset.py          # exp030からコピー
├── human_normalization.py
├── inference.py
├── losses.py
├── model.py           # ReLU→Mish変更済み
└── train.py
```

#### 2. 主要変更内容

**config.py**:
- `EXP_NUM`: "exp031"に変更
- `ExperimentConfig.description`: "Mish activation function implementation with IMU-only LSTM, human normalization, and demographics"
- `ExperimentConfig.tags`: "mish"を追加
- `LoggingConfig.wandb_tags`: "mish"を追加

**model.py**:
- **SEBlock (line 238)**: `nn.ReLU(inplace=True)` → `nn.Mish(inplace=True)`
- **ResidualSECNNBlock (line 280)**: `F.relu(self.bn1(self.conv1(x)))` → `F.mish(self.bn1(self.conv1(x)))`
- **ResidualSECNNBlock (line 289)**: `F.relu(out)` → `F.mish(out)`
- **IMUOnlyLSTM (line 412)**: `F.relu(self.bn_dense1(self.dense1(attended)))` → `F.mish(self.bn_dense1(self.dense1(attended)))`
- **IMUOnlyLSTM (line 414)**: `F.relu(self.bn_dense2(self.dense2(x)))` → `F.mish(self.bn_dense2(self.dense2(x)))`

#### 3. テスト結果

**テストファイル**: `tests/test_exp031_mish_activation.py`

**テスト実行結果**: ✅ 全8テスト成功
```
tests/test_exp031_mish_activation.py::TestMishActivation::test_mish_function_exists PASSED [ 12%]
tests/test_exp031_mish_activation.py::TestMishActivation::test_mish_vs_relu_difference PASSED [ 25%]
tests/test_exp031_mish_activation.py::TestSEBlockMish::test_se_block_with_mish PASSED [ 37%]
tests/test_exp031_mish_activation.py::TestIMUOnlyLSTMMish::test_imu_only_lstm_mish_forward PASSED [ 50%]
tests/test_exp031_mish_activation.py::TestDemographicsEmbeddingMish::test_demographics_embedding_mish PASSED [ 62%]
tests/test_exp031_mish_activation.py::TestCMISqueezeformerMish::test_cmi_squeezeformer_with_mish PASSED [ 75%]
tests/test_exp031_mish_activation.py::TestCMISqueezeformerMish::test_model_parameter_count PASSED [ 87%]
tests/test_exp031_mish_activation.py::TestNoReluRemaining::test_no_relu_in_model_forward PASSED [100%]
```

#### 4. モデル動作確認

**基本動作テスト結果**:
```
Input shape: torch.Size([2, 200, 20])
Multiclass output shape: torch.Size([2, 18])
Binary output shape: torch.Size([2, 1])
Nine-class output shape: torch.Size([2, 9])
Total parameters: 478,972
Trainable parameters: 478,972
KL divergence loss: 0.534923
```

#### 5. 実装品質確認

- ✅ **モデル構造**: 正常に動作、期待される出力形状
- ✅ **パラメータ数**: 478,972個（妥当な範囲）
- ✅ **活性化関数**: ReLUが完全にMishに置き換わっていることを確認
- ✅ **勾配計算**: 正常に動作
- ✅ **Demographics統合**: 正常に動作

## 技術的な注意事項

1. **PyTorchバージョン**: F.mishは比較的新しい関数のため、古いPyTorchでは利用できない可能性
2. **損失関数**: losses.pyのF.reluは数学的な操作のため変更対象外（意図的に保持）
3. **BatchNorm**: バッチサイズ1での学習時にエラーが発生するため、テストではeval()モードを使用

## 期待される効果

1. **勾配消失問題の改善**: Mishの滑らかな特性により勾配の流れが改善
2. **非線形性の向上**: より豊かな表現能力によるBFRB検出精度の向上
3. **学習安定性**: ReLUより滑らかな活性化による学習の安定化

## 次のステップ

1. **学習実行**: `cd codes/exp/exp031 && uv run python train.py`
2. **結果比較**: exp030との性能比較
3. **ハイパーパラメータ調整**: 必要に応じてMishに最適化した設定への調整

---

**実装完了確認**: ✅ すべての要件を満たして正常に完了