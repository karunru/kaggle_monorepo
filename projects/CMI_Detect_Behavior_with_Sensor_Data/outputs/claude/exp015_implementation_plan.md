# exp015 実装計画

## 🎉 実装完了 (2025-01-11)

exp015の実装が正常に完了しました。以下の成果物が作成されました：

### 作成されたファイル
- `codes/exp/exp015/model.py` - IMUFeatureExtractor統合済みCMISqueezeformer
- `codes/exp/exp015/dataset.py` - 簡素化されたDataset（7次元IMUデータのみ）
- `codes/exp/exp015/config.py` - exp015用設定（39次元特徴量対応）
- `tests/test_exp015_pytorch_features.py` - 包括的テストスイート（15項目）

### 実装内容
- **IMUFeatureExtractor**: 7次元→39次元の特徴量抽出をPyTorchで実装
- **エンドツーエンド学習**: 特徴量抽出が微分可能
- **GPU高速化**: すべての計算をGPU上で実行
- **数値安定性**: NaN勾配対策、クランピング処理

### テスト結果
✅ 15/15テストが成功（IMU特徴量抽出器、モデル統合、データセット、E2Eパイプライン）

## 概要
exp013のデータセット内で行っている特徴量生成をモデル側でPyTorchで実装し直し、参考notebookの特徴量と統合する実装。

## 現状分析

### exp013の特徴量（dataset.pyで生成）
1. **基本IMUデータ**
   - acc_x, acc_y, acc_z（加速度）
   - rot_w, rot_x, rot_y, rot_z（四元数）

2. **物理ベース特徴量**
   - linear_acc_x/y/z（重力除去後の線形加速度）
   - linear_acc_mag（線形加速度の大きさ）
   - linear_acc_mag_jerk（線形加速度大きさの微分）
   - angular_vel_x/y/z（四元数から計算した角速度）
   - angular_distance（連続する四元数間の角距離）

### 参考notebook（myso1987）の特徴量
1. **基本データ**
   - acc（3次元）
   - gyro（3次元）※我々のデータにはない

2. **追加特徴量**
   - acc_mag, gyro_mag（大きさ）
   - jerk（加速度の微分）, gyro_delta（角速度の微分）
   - acc_pow, gyro_pow（エネルギー：二乗値）
   - acc_lpf, acc_hpf（低域/高域フィルタ）
   - gyro_lpf, gyro_hpf（低域/高域フィルタ）

## 実装方針

### 1. PyTorchでの特徴量生成モジュール作成
model.py内に`IMUFeatureExtractor`クラスを作成し、以下の特徴量をPyTorchで計算：

#### 統合特徴量リスト
1. **基本IMU** (7次元)
   - acc_x, acc_y, acc_z
   - rot_w, rot_x, rot_y, rot_z

2. **物理ベース特徴量** (9次元) - exp013から
   - linear_acc_x/y/z（重力除去）
   - linear_acc_mag
   - linear_acc_mag_jerk
   - angular_vel_x/y/z（四元数から計算）
   - angular_distance

3. **追加特徴量** (13次元) - 参考notebookから適応
   - acc_mag（加速度の大きさ）
   - acc_jerk_x/y/z（加速度の微分）
   - acc_pow_x/y/z（加速度の二乗）
   - acc_lpf_x/y/z（低域フィルタ）
   - acc_hpf_x/y/z（高域フィルタ）

4. **角速度関連** (10次元) - 参考notebookのgyroを角速度で代替
   - angular_vel_mag（角速度の大きさ）
   - angular_vel_delta_x/y/z（角速度の微分）
   - angular_vel_pow_x/y/z（角速度の二乗）
   - angular_vel_lpf_x/y/z（低域フィルタ）
   - angular_vel_hpf_x/y/z（高域フィルタ）

合計: 39次元の特徴量

### 2. モデルアーキテクチャの修正
- `CMISqueezeformer`の`__init__`に`IMUFeatureExtractor`を追加
- forwardメソッドで生のIMUデータを受け取り、特徴量を生成してから処理
- アーキテクチャ自体は変更しない

### 3. dataset.pyの簡素化
- 物理特徴量計算の削除（`_add_physics_features`など）
- 生のIMUデータのみを返すように修正
- `imu_cols`を基本7次元のみに変更
- 前処理の簡素化

### 4. 削除するコード
- `remove_gravity_from_acc_pl`
- `calculate_angular_velocity_from_quat_pl`
- `calculate_angular_distance_pl`
- `_add_physics_features`
- 関連する前処理コード

### 5. configの更新
- 特徴量次元数の更新
- 新しいハイパーパラメータの追加（フィルタのカットオフ周波数など）

## 実装手順

1. **exp013をexp015にコピー**
   ```bash
   cp -r codes/exp/exp013 codes/exp/exp015
   ```

2. **model.pyの修正**
   - `IMUFeatureExtractor`クラスを追加
   - PyTorchで全特徴量計算を実装
   - `CMISqueezeformer`を修正して特徴量抽出器を統合

3. **dataset.pyの簡素化**
   - 物理特徴量計算コードを削除
   - 生のIMUデータのみを返すように修正

4. **config.pyの更新**
   - 特徴量次元数を7（生データ）に変更
   - フィルタパラメータを追加

5. **テストコードの作成**
   - 特徴量生成の単体テスト
   - データローダーのテスト
   - モデルの順伝播テスト

## 実装上の注意点

1. **GPU対応**
   - すべての計算をPyTorchのテンソル演算で実装
   - バッチ処理に対応

2. **勾配の保持**
   - 特徴量計算が微分可能であることを確認
   - detach()を使わない

3. **数値安定性**
   - 四元数の正規化時にepsを追加
   - ゼロ除算の回避

4. **パフォーマンス**
   - 不要なメモリコピーを避ける
   - インプレース演算の活用

5. **TOF/Thermalデータ**
   - 変更しない
   - 既存の処理をそのまま維持

## テスト計画

1. **単体テスト**
   - 各特徴量計算の正確性
   - バッチ処理の動作確認
   - GPU/CPU両方での動作

2. **統合テスト**
   - データローダーとモデルの連携
   - 訓練ループの動作確認
   - 推論の動作確認

3. **性能テスト**
   - 処理速度の比較（CPU vs GPU）
   - メモリ使用量の確認

## 期待される効果

1. **計算の高速化**
   - GPU上で特徴量計算を実行
   - バッチ処理の効率化

2. **End-to-End学習**
   - 特徴量抽出も学習可能に
   - より最適な特徴表現の獲得

3. **コードの簡素化**
   - dataset.pyの複雑度削減
   - 前処理の統一化

## リスクと対策

1. **メモリ使用量の増加**
   - 対策：必要に応じてバッチサイズを調整

2. **計算精度の差異**
   - 対策：単体テストで数値誤差を確認

3. **学習の不安定化**
   - 対策：特徴量の正規化を適切に実施