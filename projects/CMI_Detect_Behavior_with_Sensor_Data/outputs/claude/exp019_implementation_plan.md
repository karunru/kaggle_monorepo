# Exp019 実装計画：利き手反転データオーグメンテーション

## 概要

docs/exp019_plan.md の指示に従い、exp018をベースとして利き手反転データオーグメンテーションを実装します。

## 背景と要求事項

### 参照ドキュメント
- `docs/competition_overview.md`: CMI Detect Behavior with Sensor Data コンペティション概要
- `docs/data_description.md`: センサーデータセットの詳細構造
- `docs/handedness_augmentation_plan.md`: 利き手反転オーグメンテーションの技術詳細

### 実装要件
- **ベース**: codes/exp/exp018 を codes/exp/exp019 にコピー
- **新機能**: 右利きデータを50%の確率で左利きに反転するaugmentation
- **制約**: exp018からのインポート禁止、差分は最小限に抑制

## データの理解

### センサー構成
1. **IMU（1個）**: 
   - 加速度（acc_x/y/z）と回転（rot_w/x/y/z）
   - 利き手反転の対象
2. **Thermopile（5個）**: 
   - thm_1〜5で温度測定
   - 位置固定のため反転不可
3. **Time-of-Flight（5個）**: 
   - tof_1〜5_v0〜63（8x8グリッド×5個）
   - グリッドの左右反転が必要

### Demographics データ
- `handedness`: 0=左利き、1=右利き
- モデルの訓練に利き手情報を活用

## 技術仕様

### 利き手反転変換

#### 1. IMU加速度データ
```python
def flip_acceleration(acc_x, acc_y, acc_z):
    """Y軸の符号を反転させて左右の違いを表現"""
    return acc_x, -acc_y, acc_z
```

#### 2. IMU回転データ（四元数）
```python
def flip_rotation(rot_w, rot_x, rot_y, rot_z):
    """Y軸周りの回転を反転"""
    return rot_w, rot_x, -rot_y, rot_z
```

#### 3. Time-of-Flight データ
```python
def flip_tof_grid(tof_values):
    """8x8グリッドを左右反転（行ごとに反転）"""
    # 各センサーの64ピクセル（8x8）を行単位で反転
    return horizontally_flip_8x8_grid(tof_values)
```

## 実装計画

### タスク1: 環境構築
- [ ] exp018をexp019にコピー
- [ ] 必要な依存関係の確認
- [ ] テスト環境の準備

### タスク2: 利き手反転オーグメンテーション実装
- [ ] `HandednessAugmentation`クラスの作成
- [ ] IMUデータ反転機能の実装
- [ ] ToFデータ反転機能の実装
- [ ] Demographicsデータとの統合

### タスク3: データセット統合
- [ ] dataset.pyへのaugmentation統合
- [ ] 50%確率での適用ロジック実装
- [ ] 訓練時のみの適用制御

### タスク4: 検証と最適化
- [ ] テストコードの作成
- [ ] 反転データの妥当性確認
- [ ] 性能への影響評価

## 実装詳細

### HandednessAugmentationクラス

```python
class HandednessAugmentation:
    """利き手反転データオーグメンテーション"""
    
    def __init__(self, demographics_df: pd.DataFrame, flip_probability: float = 0.5):
        self.demographics = demographics_df
        self.flip_probability = flip_probability
        
    def __call__(self, sequence_data: Dict[str, np.ndarray], subject_id: int) -> Dict[str, np.ndarray]:
        """シーケンスデータに利き手反転を適用"""
        if random.random() > self.flip_probability:
            return sequence_data
            
        original_handedness = self.get_handedness(subject_id)
        return self.flip_sequence(sequence_data, original_handedness)
        
    def flip_sequence(self, data: Dict[str, np.ndarray], handedness: int) -> Dict[str, np.ndarray]:
        """実際の反転処理"""
        flipped_data = data.copy()
        
        # IMU加速度の反転
        flipped_data['acc_y'] = -flipped_data['acc_y']
        
        # IMU回転の反転
        flipped_data['rot_y'] = -flipped_data['rot_y']
        
        # ToFデータの反転
        for sensor_id in range(1, 6):  # tof_1 to tof_5
            flipped_data = self.flip_tof_sensor(flipped_data, sensor_id)
            
        return flipped_data
        
    def flip_tof_sensor(self, data: Dict[str, np.ndarray], sensor_id: int) -> Dict[str, np.ndarray]:
        """個別ToFセンサーの8x8グリッド反転"""
        for v in range(64):  # v0 to v63
            row = v // 8
            col = v % 8
            flipped_col = 7 - col
            flipped_v = row * 8 + flipped_col
            
            original_key = f'tof_{sensor_id}_v{v}'
            flipped_key = f'tof_{sensor_id}_v{flipped_v}'
            
            # データを交換
            temp = data[original_key].copy()
            data[original_key] = data[flipped_key]
            data[flipped_key] = temp
            
        return data
```

### データセット統合

```python
class CMIDataset:
    def __init__(self, ..., enable_handedness_aug=False, handedness_flip_prob=0.5):
        # ...既存の初期化コード...
        
        if enable_handedness_aug:
            self.handedness_aug = HandednessAugmentation(
                demographics_df, handedness_flip_prob
            )
        else:
            self.handedness_aug = None
            
    def __getitem__(self, idx):
        # ...既存のデータ取得コード...
        
        if self.handedness_aug and self.training:
            sequence_data = self.handedness_aug(sequence_data, subject_id)
            
        return sequence_data, target
```

## テスト戦略

### 単体テスト
- [ ] IMU反転の数学的正確性
- [ ] ToF 8x8グリッド反転の正確性
- [ ] Demographics統合の動作確認

### 統合テスト
- [ ] データローダーとの統合動作
- [ ] 訓練・検証時の適用制御
- [ ] メモリ使用量とパフォーマンス

### 効果測定テスト
- [ ] 左利き・右利きデータでの性能比較
- [ ] オーグメンテーション有無での性能変化
- [ ] 過学習の兆候監視

## リスクと対応策

### リスク1: 座標系の誤解
- **対応**: 小規模データでの段階的検証
- **検証**: 特定ジェスチャーの可視化による確認

### リスク2: 過学習
- **対応**: Dropoutやweight decayの調整
- **監視**: Validation性能の注意深い観測

### リスク3: ジェスチャーラベルの不整合
- **対応**: 対称性のあるジェスチャーのみを対象
- **検証**: 医療専門知識による妥当性確認

## 成功指標

### 定量的指標
- Binary F1スコアの改善（特に左利き被験者）
- Macro F1スコアの改善
- 訓練・検証損失の安定性

### 定性的指標
- 実装の保守性とコードの品質
- テスト網羅率の達成
- ドキュメント完成度

## スケジュール

1. **環境構築とコピー** (1時間)
2. **オーグメンテーション実装** (3時間)
3. **データセット統合** (2時間)
4. **テスト作成と検証** (2時間)
5. **最適化と文書化** (1時間)

**総推定時間**: 9時間

## 実装完了後の成果物

### コードファイル
- `codes/exp/exp019/dataset.py`: オーグメンテーション統合版
- `codes/exp/exp019/config.py`: 新しい設定パラメータ
- `codes/exp/exp019/train.py`: 訓練スクリプト（必要に応じて調整）

### テストファイル
- `tests/test_exp019_dataset.py`: データセットテスト
- `tests/test_exp019_handedness.py`: オーグメンテーション専用テスト

### ドキュメント
- 本ドキュメントへの実装結果追記
- パフォーマンス比較結果のレポート

---

## 注意事項

- Thermopileデータは物理的制約により反転しない
- テスト時はオーグメンテーションを無効にする
- 利き手情報がない被験者への適切な対処
- メモリ効率を考慮した実装

この計画に従って段階的に実装を進め、各ステップでの検証を徹底することで、安全で効果的な利き手反転オーグメンテーションを実現します。