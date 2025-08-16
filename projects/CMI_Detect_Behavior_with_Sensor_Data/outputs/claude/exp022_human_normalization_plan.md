# exp022 実装計画書：Human Normalization実装

## 概要

exp022では、exp021をベースにして人体測定値（身長、肘から手首の距離、肩から手首の距離）を使用したIMUデータの正規化機能（Human Normalization）を実装します。これにより、体格の違いに依存しない、より汎化性能の高い特徴量を生成することを目指します。

## 実装方針

- **ベース**：`codes/exp/exp021`を`codes/exp/exp022`にコピー
- **差分最小化**：指示された機能以外の変更は最小限に留める
- **独立性**：`codes/exp/exp021`からのインポートは禁止
- **編集制限**：`codes/exp/exp022`以外の`codes/exp/`配下のファイル編集は禁止

## コンペティション理解

### BFRBs検出タスク
- **目標**：Body-focused repetitive behaviors（髪を抜く、皮膚をいじる等）とnon-BFRB行動の区別
- **センサー**：Heliosデバイス（IMU + thermopiles + time-of-flight）
- **評価**：Binary F1とMacro F1の平均
- **制約**：テストセットの半分はIMUデータのみ

### データ構造
- **基本識別子**：sequence_id、subject等
- **IMUデータ**：acc_[x/y/z]、rot_[w/x/y/z]
- **人体測定値**：height_cm、shoulder_to_wrist_cm、elbow_to_wrist_cm
- **フェーズ**：Transition → Pause → Gesture

## Human Normalization 設計

### 理論的背景
人体測定値を使用してIMUデータを正規化することで：
1. 体格差による特徴量のばらつきを軽減
2. より汎化性能の高いモデルを構築
3. 物理的に意味のある正規化された特徴量を生成

### 正規化対象特徴量

#### 既存の物理特徴量（exp021）
```
- linear_acc_x/y/z: 重力除去済み線形加速度
- linear_acc_mag: 線形加速度の大きさ
- angular_vel_x/y/z: 角速度
- angular_distance: 角距離
```

#### 人体測定値（メートル単位）
```
- h = height_cm / 100: 身長
- r_elbow = elbow_to_wrist_cm / 100: 前腕レバー長
- r_shoulder = shoulder_to_wrist_cm / 100: 上腕+前腕レバー長
- r_eff = clip(r_shoulder, min=0.15, max=0.9): 有効半径
```

#### 新規正規化特徴量
```python
# 身長正規化
linear_acc_mag_per_h = linear_acc_mag / h

# レバー長正規化
linear_acc_mag_per_rS = linear_acc_mag / r_shoulder
linear_acc_mag_per_rE = linear_acc_mag / r_elbow

# 無次元化比率
acc_over_centripetal_rE = linear_acc_mag / max(a_c_elbow, eps)
acc_over_centripetal_rS = linear_acc_mag / max(a_c_shoulder, eps)

# 角加速度近似
alpha_like_rE = linear_acc_mag / max(r_elbow, eps)
alpha_like_rS = linear_acc_mag / max(r_shoulder, eps)

# 速度複合特徴量
v_over_h = v_shoulder / h
v_over_rS = v_shoulder / r_shoulder
v_over_rE = v_elbow / r_elbow
```

### 欠損値処理
- 人体測定値が欠損/非正値の場合：コホート中央値で代替
- `hn_used_fallback`フラグで欠損値使用を記録
- 数値安定性：eps = 1e-6を分母に使用

## タスク分割

### Task 1: human_normalization.py作成
**目標**：Human Normalization用の共通関数を実装
**成果物**：`codes/exp/exp022/human_normalization.py`

#### 実装内容
```python
@dataclass
class HNConfig:
    hn_enabled: bool = True
    hn_eps: float = 1e-6
    hn_radius_min_max: tuple[float, float] = (0.15, 0.9)
    hn_features: list[str] = [
        "linear_acc_mag_per_h",
        "linear_acc_mag_per_rS",
        "linear_acc_mag_per_rE",
        "acc_over_centripetal_rS",
        "acc_over_centripetal_rE",
        "alpha_like_rS",
        "alpha_like_rE",
        "v_over_h",
        "v_over_rS",
        "v_over_rE"
    ]

def compute_hn_features(
    frame: pl.LazyFrame,
    demo_df: pl.DataFrame,
    cfg: HNConfig
) -> pl.LazyFrame:
    """Human Normalization特徴量を計算"""

def join_subject_anthro(
    frame: pl.LazyFrame,
    demo_df: pl.DataFrame
) -> pl.LazyFrame:
    """subjectに人体測定値をjoin"""

def derive_hn_channels(
    frame: pl.LazyFrame,
    eps: float,
    bounds: tuple[float, float]
) -> pl.LazyFrame:
    """正規化チャンネルを導出"""
```

### Task 2: exp021からexp022へのコピー
**目標**：exp021のコードをexp022にコピー
**成果物**：`codes/exp/exp022/`ディレクトリ
**実行方法**：
```bash
cp -r codes/exp/exp021 codes/exp/exp022
```

### Task 3: config.py拡張
**目標**：Human Normalization用の設定を追加
**対象ファイル**：`codes/exp/exp022/config.py`

#### 追加設定
```python
class DemographicsConfig(BaseModel):
    # 既存設定はそのまま保持
    
    # Human Normalization設定
    hn_enabled: bool = Field(default=True, description="Human Normalization機能を有効にするかどうか")
    hn_eps: float = Field(default=1e-6, description="数値安定性のためのepsilon値")
    hn_radius_min_max: tuple[float, float] = Field(
        default=(0.15, 0.9), 
        description="有効半径の最小値・最大値"
    )
    hn_features: list[str] = Field(
        default=[
            "linear_acc_mag_per_h",
            "linear_acc_mag_per_rS", 
            "linear_acc_mag_per_rE",
            "acc_over_centripetal_rS",
            "acc_over_centripetal_rE",
            "alpha_like_rS",
            "alpha_like_rE",
            "v_over_h",
            "v_over_rS",
            "v_over_rE"
        ],
        description="有効にするHuman Normalization特徴量のリスト"
    )
```

### Task 4: dataset.py統合
**目標**：IMUDatasetクラスにHuman Normalization機能を統合
**対象ファイル**：`codes/exp/exp022/dataset.py`

#### 実装手順
1. `human_normalization.py`をインポート
2. `IMUDataset.imu_cols`にHN特徴量を追加（hn_enabled=Trueの場合）
3. `_add_physics_features`メソッド内でHN特徴量を計算
4. demographics_dataとのjoinを実装

#### 修正箇所
```python
def _add_physics_features(self, df: pl.DataFrame) -> pl.DataFrame:
    """物理ベースIMU特徴量を追加（Human Normalization含む）"""
    # 既存の物理特徴量計算
    df_with_physics = ... # 既存の実装
    
    # Human Normalization特徴量の計算
    if self.demographics_config.get("hn_enabled", False):
        hn_config = HNConfig(...)
        df_with_hn = compute_hn_features(
            df_with_physics.lazy(),
            self.demographics_data,
            hn_config
        ).collect()
        return df_with_hn
    
    return df_with_physics
```

### Task 5: テストコード作成
**目標**：Human Normalization機能のユニットテスト
**成果物**：`tests/test_exp022_human_normalization.py`

#### テストケース
1. **決定論的テスト**：既知のomega、r値での計算結果検証
2. **欠損値処理**：人体測定値欠損時の中央値使用とフラグ設定
3. **数値安定性**：zero omega、負値半径での有限値出力
4. **契約テスト**：HN有効時の特徴量次元増加確認

### Task 6: 統合テスト
**目標**：exp022全体の動作確認
**検証項目**：
1. `hn_enabled=False`時のexp021との同等性
2. `hn_enabled=True`時の新特徴量生成
3. NaN/inf値の不存在
4. 特徴量統計の妥当性

### Task 7: 実験実行
**目標**：Human Normalizationの効果検証
**実行方法**：
```bash
# 従来版
python codes/exp/exp021/train.py

# Human Normalization版
python codes/exp/exp022/train.py
```

## 検証基準

### 機能要件
1. **後方互換性**：`hn_enabled=False`で既存exp021と同等結果
2. **特徴量生成**：`hn_enabled=True`で新特徴量が生成される
3. **数値安定性**：NaN/inf値が発生しない
4. **欠損値処理**：人体測定値欠損時も正常動作

### 性能要件
1. **分散減少**：体格による特徴量分散の減少
2. **汎化性能**：検証スコアの改善（期待）
3. **計算効率**：処理時間の大幅増加なし

## リスク管理

### 技術的リスク
1. **ジェスチャー特異的運動学**：実際のレバー長は動作により異なる
   - **対策**：r_elbowとr_shoulderの両方の変種を提供
2. **センサー向き変動**：既存の重力除去・角速度計算に依存
   - **対策**：必要に応じてローパスフィルタ追加検討
3. **人体測定値外れ値**：異常値の影響
   - **対策**：クリッピングとフォールバック処理

### 実装リスク
1. **メモリ使用量増加**：新特徴量による容量増大
   - **対策**：設定による特徴量選択機能
2. **計算時間増加**：Polars処理の複雑化
   - **対策**：効率的なjoin処理とベクトル化

## 完了基準

1. ✅ 全タスクの実装完了
2. ✅ ユニットテスト全通過
3. ✅ 統合テスト正常動作確認
4. ✅ `hn_enabled=False`時の既存結果との一致
5. ✅ `hn_enabled=True`時の新特徴量生成確認
6. ✅ 計算性能劣化の許容範囲内維持

## 実装スケジュール

| タスク | 推定工数 | 依存関係 |
|--------|----------|----------|
| Task 1 | 2時間 | なし |
| Task 2 | 10分 | なし |
| Task 3 | 30分 | Task 2 |
| Task 4 | 1.5時間 | Task 1, Task 3 |
| Task 5 | 1時間 | Task 1 |
| Task 6 | 30分 | Task 4, Task 5 |
| Task 7 | 1時間 | Task 6 |

**総推定工数**: 約6時間

## 成果物一覧

### 新規作成ファイル
1. `codes/exp/exp022/human_normalization.py` - HN機能実装
2. `tests/test_exp022_human_normalization.py` - ユニットテスト
3. `outputs/claude/exp022_human_normalization_plan.md` - 本計画書

### 変更ファイル
1. `codes/exp/exp022/config.py` - HN設定追加
2. `codes/exp/exp022/dataset.py` - HN機能統合
3. 必要に応じて`codes/exp/exp022/train.py` - 設定変更

### コピーファイル
1. `codes/exp/exp022/` - exp021の完全コピー

このHuman Normalization実装により、体格に依存しない特徴量でBFRBs検出の精度向上を目指します。