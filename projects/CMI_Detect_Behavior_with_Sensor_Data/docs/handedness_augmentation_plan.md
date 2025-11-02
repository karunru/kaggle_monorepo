# 利き手反転データオーグメンテーション計画

## 概要

IMUセンサーデータとdemographicsデータの利き手情報を活用して、利き手を反転させたデータオーグメンテーションを実装します。これにより、右利きの人のデータを左利きのように、左利きの人のデータを右利きのように変換し、データの多様性を増やします。

## 背景と目的

### 現状の課題
1. **データの不均衡**: 一般的に右利きの人口が多いため、データセットも右利きに偏っている可能性
2. **汎化性能**: 少数派（左利き）のデータが不足していると、モデルの汎化性能が低下する可能性
3. **センサー配置の非対称性**: 手首のデバイス装着位置により、左右で異なるパターンが生じる

### 期待される効果
1. **データ量の倍増**: 各サンプルから反転版を生成することで、実質的にデータ量を2倍に
2. **バランスの改善**: 左利き・右利きのデータバランスを改善
3. **ロバスト性向上**: 両利きのパターンを学習することで、より頑健なモデルを構築

## IMUデータの座標系理解

### センサー配置
- デバイスは利き手の手首に装着
- IMUセンサーは1個で、3軸加速度と4元数による回転を記録

### 座標軸の定義（推定）
```
右手装着時:
- X軸: 手首から指先方向（正）
- Y軸: 親指から小指方向（正）  
- Z軸: 手の甲から手のひら方向（正）

左手装着時:
- X軸: 手首から指先方向（正）
- Y軸: 小指から親指方向（正）
- Z軸: 手の甲から手のひら方向（正）
```

## 反転変換の具体的方法

### 1. 加速度データの変換
```python
def flip_acceleration(acc_x, acc_y, acc_z, from_right_to_left=True):
    """
    加速度データの左右反転
    Y軸の符号を反転させることで左右の違いを表現
    """
    if from_right_to_left:
        return acc_x, -acc_y, acc_z
    else:
        return acc_x, -acc_y, acc_z
```

### 2. 回転データの変換
```python
def flip_rotation(rot_w, rot_x, rot_y, rot_z, from_right_to_left=True):
    """
    四元数の左右反転
    Y軸周りの回転を反転させる
    """
    if from_right_to_left:
        # Y軸周りの回転を反転
        return rot_w, rot_x, -rot_y, rot_z
    else:
        return rot_w, rot_x, -rot_y, rot_z
```

### 3. ジェスチャーラベルの考慮
一部のジェスチャーは左右で異なる可能性があるため、ラベルの調整が必要：
- 対称的なジェスチャー（例：「Forehead - scratch」）: そのまま維持
- 非対称的なジェスチャー: 必要に応じてラベルを調整

## 実装計画

### Phase 1: データ分析とバリデーション
1. **利き手分布の確認**
   - train_demographics.csvから利き手の分布を分析
   - 左利き・右利きの比率を確認

2. **座標系の検証**
   - 特定のジェスチャー（例：「Wave hello」）のIMUパターンを可視化
   - 左利き・右利きで期待される違いを確認

### Phase 2: 変換関数の実装
```python
class HandednessAugmentation:
    def __init__(self, demographics_df):
        self.demographics = demographics_df
        
    def augment_sequence(self, sequence_df, sequence_id):
        # 1. 元の利き手を取得
        subject_id = sequence_df['subject'].iloc[0]
        original_handedness = self.get_handedness(subject_id)
        
        # 2. IMUデータを反転
        augmented_df = sequence_df.copy()
        augmented_df = self.flip_imu_data(augmented_df, original_handedness)
        
        # 3. 新しいsequence_idを付与
        augmented_df['sequence_id'] = f"{sequence_id}_flipped"
        augmented_df['is_augmented'] = True
        
        return augmented_df
```

### Phase 3: 統合と検証
1. **データローダーへの統合**
   - 訓練時にランダムに反転を適用するオプションを追加
   - バッチ単位での適用も可能に

2. **効果測定**
   - 反転なし vs 反転ありでの性能比較
   - 左利きデータでの性能向上を確認

## 実装上の注意点

### 1. センサーの物理的制約
- Thermopileセンサー: 位置固定のため反転不可
- Time-of-Flightセンサー: 8x8グリッドの左右反転が必要

### 2. テスト時の扱い
- テスト時はオーグメンテーションを使用しない
- ただし、TTAとして左右反転の予測を平均化することは可能

### 3. 検証戦略
- 左利きの被験者をvalidationに含めて効果を測定
- Cross-validationで左利き・右利きのバランスを考慮

## 期待される実装コード構造

```python
# src/augmentation/handedness.py
class HandednessAugmentation:
    """利き手反転オーグメンテーション"""
    
    def __init__(self, demographics_path: str):
        self.demographics = pd.read_csv(demographics_path)
        
    def flip_imu(self, df: pd.DataFrame) -> pd.DataFrame:
        """IMUデータの左右反転"""
        
    def flip_tof(self, df: pd.DataFrame) -> pd.DataFrame:
        """ToFデータの左右反転（8x8グリッド）"""
        
    def should_flip_label(self, gesture: str) -> bool:
        """ジェスチャーラベルの反転が必要かチェック"""

# 使用例
augmenter = HandednessAugmentation("data/train_demographics.csv")
augmented_df = augmenter.augment(train_df, probability=0.5)
```

## スケジュール

1. **Week 1**: データ分析と座標系の理解
2. **Week 2**: 基本的な反転関数の実装
3. **Week 3**: データローダーへの統合
4. **Week 4**: 効果測定と最適化

## リスクと対策

### リスク
1. **座標系の誤解**: IMUの座標系を誤って理解すると逆効果
2. **過学習**: 同じデータの変形版を使うことによる過学習
3. **ラベルの不整合**: 左右で異なるジェスチャーの扱い

### 対策
1. **段階的検証**: 小規模データで効果を確認してから全体に適用
2. **正則化**: Dropoutやweight decayを調整
3. **ドメイン知識**: 医療専門家の意見を参考にジェスチャーの左右差を確認

## まとめ

利き手反転オーグメンテーションは、IMUデータの物理的特性を活用した効果的なデータ拡張手法です。適切に実装することで、モデルの汎化性能向上と、左利き・右利きの両方に対するロバスト性の向上が期待できます。