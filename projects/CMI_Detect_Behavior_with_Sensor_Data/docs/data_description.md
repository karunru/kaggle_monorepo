# データセット詳細

## 概要
このコンペティションでは、センサーデータを使用してBody-focused repetitive behaviors（BFRBs）とその他のジェスチャーを分類します。

このデータセットは、参加者が利き手の手首にHeliosデバイスを装着して8つのBFRB様ジェスチャーと10つのnon-BFRB様ジェスチャーを実行した際のセンサー記録を含んでいます。

## Heliosデバイスのセンサー

Heliosデバイスには3つのセンサータイプが含まれています：

### 1. 慣性測定装置（IMU） - 1個
- **モデル**: [BNO080/BNO085](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Competitions/Helios2025/IMU_Sensor.pdf)
- **機能**: 加速度計、ジャイロスコープ、磁力計の測定値をオンボード処理と組み合わせて、方向と動きのデータを提供する統合センサー

### 2. サーモパイルセンサー - 5個
- **モデル**: [MLX90632](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Competitions/Helios2025/Thermopile_Sensor.pdf)
- **機能**: 赤外線放射を測定する非接触温度センサー

### 3. Time-of-Flightセンサー - 5個
- **モデル**: [VL53L7CX](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Competitions/Helios2025/Time_of_Flight_Sensor.pdf)
- **機能**: 放射された赤外線が物体から跳ね返ってくるまでの時間を検出することで距離を測定

## データセット詳細

### 基本情報
- **ファイル数**: 15ファイル
- **総サイズ**: 1.12 GB
- **ファイル形式**: Python、CSV、Proto
- **ライセンス**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **列数**: 693列

### テストセットの特徴
- **予想テストシーケンス数**: 約3,500
- **重要な特徴**: 隠しテストセットの半分はIMUのみで記録されています
  - Thermopile（`thm_`）とTime-of-Flight（`tof_`）列は存在しますが、これらのシーケンスではnull値が含まれています
- **目的**: Time-of-FlightとThermopileセンサーの追加がBFRB検出能力を向上させるかを判断

### データの制約
- 既知のセンサー通信障害により、一部のシーケンスで一部のセンサーからのデータが欠損している場合があります

## ファイル構成

### 1. [train/test].csv
メインのセンサーデータファイル

#### 基本識別子
- `row_id`: 行ID（文字列型）
  - 形式: `SEQ_XXXXXX_YYYYYY`（シーケンスID_行番号）
  - ユニーク値数: 574,945（train）、107（test）
- `sequence_id`: センサーデータのバッチID。各シーケンスは1つのTransition、1つのPause、1つのGestureを含む（文字列型）
  - 形式: `SEQ_XXXXXX`
  - ユニーク値数: 8,151（train）、2（test）
- `sequence_type`: ジェスチャーがターゲットタイプかnon-ターゲットタイプか（文字列型、trainのみ）
  - 値: `Target`, `Non-Target`
  - Target: BFRB様ジェスチャー、Non-Target: 通常のジェスチャー
- `sequence_counter`: 各シーケンス内の行カウンター（整数型）
- `subject`: データを提供した被験者のユニークID（文字列型）
  - 形式: `SUBJ_XXXXXX`
  - ユニーク値数: 81（train）、2（test）

#### ターゲット変数
- `gesture`: 予測対象列。シーケンスのGestureの説明（文字列型、trainのみ）
  - ユニーク値数: 18種類
  - **Target（BFRB様）ジェスチャー（8種類）**:
    - `Above ear - pull hair`: 耳上の髪を引っ張る
    - `Cheek - pinch skin`: 頬の皮膚をつまむ
    - `Eyebrow - pull hair`: 眉毛を引っ張る
    - `Eyelash - pull hair`: まつ毛を引っ張る
    - `Forehead - pull hairline`: 額の生え際を引っ張る
    - `Forehead - scratch`: 額をかく
    - `Neck - pinch skin`: 首の皮膚をつまむ
    - `Neck - scratch`: 首をかく
  - **Non-Target（通常）ジェスチャー（10種類）**:
    - `Drink from bottle/cup`: ボトル/カップから飲む
    - `Feel around in tray and pull out an object`: トレイ内を探って物を取り出す
    - `Glasses on/off`: 眼鏡をかける/外す
    - `Pinch knee/leg skin`: 膝/脚の皮膚をつまむ
    - `Pull air toward your face`: 空気を顔に向かって引く
    - `Scratch knee/leg skin`: 膝/脚をかく
    - `Text on phone`: 携帯電話でテキスト入力
    - `Wave hello`: 手を振って挨拶
    - `Write name in air`: 空中に名前を書く
    - `Write name on leg`: 脚に名前を書く

#### メタデータ
- `orientation`: シーケンス中の被験者の向きの説明（文字列型、trainのみ）
  - ユニーク値数: 4種類
  - 値:
    - `Seated Straight`: 椅子にまっすぐ座った状態
    - `Lie on Back`: 仰向けに寝た状態
    - `Lie on Side - Non Dominant`: 非利き手側を下にして横向きに寝た状態
    - `Seated Lean Non Dom - FACE DOWN`: 非利き手側に寄りかかって顔を下にした座位
- `behavior`: シーケンスの現在フェーズ中の被験者の行動の説明（文字列型）
  - ユニーク値数: 4種類
  - 値:
    - `Moves hand to target location`: 手をターゲット位置に移動
    - `Hand at target location`: 手をターゲット位置で静止
    - `Performs gesture`: ジェスチャーを実行
    - `Relaxes and moves hand to target location`: リラックスして手をターゲット位置に移動
- `phase`: シーケンスの現在フェーズの説明（文字列型）
  - ユニーク値数: 2種類
  - 値:
    - `Transition`: 移行フェーズ（手の移動や休憩）
    - `Gesture`: ジェスチャーフェーズ（実際のジェスチャー実行）

#### センサーデータ

##### IMUセンサー
- `acc_[x/y/z]`: IMUセンサーからの3軸の線形加速度（メートル毎秒の二乗）
- `rot_[w/x/y/z]`: IMUのジャイロスコープ、加速度計、磁力計からの情報を組み合わせて3D空間でのデバイスの向きを記述する方向データ

##### Thermopileセンサー（5個）
- `thm_[1-5]`: 時計上の5つのthermopileセンサーで温度を摂氏で記録
- **注意**: インデックス/番号は概要タブの写真のインデックスに対応

##### Time-of-Flightセンサー（5個）
- `tof_[1-5]_v[0-63]`: 時計上の5つのtime-of-flightセンサーで距離を測定
- **データ構造**: 
  - 最初のセンサーの0番目のピクセル: `tof_1_v0`
  - 最後のピクセル: `tof_1_v63`
  - 各センサーは8x8グリッドに配置された64ピクセルを含む
- **データ収集**: 行方向で収集（左上から開始し、右へ移動し、最終的に右下で終了）
- **値の範囲**: 0-254の未較正センサー値
- **欠損値**: センサー応答がない場合（例：信号反射を引き起こす近くの物体がない場合）は-1

## 共起関係分析

### ジェスチャーと体位の関係

18種類のジェスチャーと4種類の体位の完全な共起関係：

| Gesture | Seated Straight | Lie on Back | Lie on Side - Non Dominant | Seated Lean Non Dom - FACE DOWN |
|---------|-----------------|-------------|----------------------------|----------------------------------|
| Above ear - pull hair | ✓ (10,794) | ✓ (9,838) | ✓ (9,836) | ✓ (10,092) |
| Cheek - pinch skin | ✓ (10,681) | ✓ (9,867) | ✓ (9,235) | ✓ (10,341) |
| Drink from bottle/cup | ✓ (13,093) | ✗ (0) | ✗ (0) | ✗ (0) |
| Eyebrow - pull hair | ✓ (14,233) | ✓ (10,078) | ✓ (9,835) | ✓ (10,159) |
| Eyelash - pull hair | ✓ (10,607) | ✓ (9,971) | ✓ (9,585) | ✓ (10,055) |
| Feel around in tray and pull out an object | ✓ (17,114) | ✗ (0) | ✗ (0) | ✗ (0) |
| Forehead - pull hairline | ✓ (11,309) | ✓ (9,824) | ✓ (9,676) | ✓ (9,993) |
| Forehead - scratch | ✓ (11,095) | ✓ (9,845) | ✓ (9,627) | ✓ (10,356) |
| Glasses on/off | ✓ (13,542) | ✗ (0) | ✗ (0) | ✗ (0) |
| Neck - pinch skin | ✓ (11,310) | ✓ (10,079) | ✓ (9,472) | ✓ (9,646) |
| Neck - scratch | ✓ (14,461) | ✓ (14,672) | ✓ (13,861) | ✓ (13,625) |
| Pinch knee/leg skin | ✗ (0) | ✗ (0) | ✗ (0) | ✓ (9,844) |
| Pull air toward your face | ✓ (10,469) | ✓ (10,141) | ✓ (10,133) | ✗ (0) |
| Scratch knee/leg skin | ✗ (0) | ✗ (0) | ✗ (0) | ✓ (12,328) |
| Text on phone | ✓ (15,834) | ✓ (14,340) | ✓ (13,229) | ✓ (15,059) |
| Wave hello | ✓ (12,328) | ✓ (11,186) | ✓ (10,842) | ✗ (0) |
| Write name in air | ✓ (11,250) | ✓ (10,202) | ✓ (9,815) | ✗ (0) |
| Write name on leg | ✗ (0) | ✗ (0) | ✗ (0) | ✓ (10,138) |

**体位限定ジェスチャーの特徴**:
- **Seated Straightでのみ実行**: Drink from bottle/cup, Feel around in tray and pull out an object, Glasses on/off
- **Seated Lean Non Dom - FACE DOWNでのみ実行**: Pinch knee/leg skin, Scratch knee/leg skin, Write name on leg
- **Seated Lean Non Dom - FACE DOWNで実行されない**: Pull air toward your face, Wave hello, Write name in air

### フェーズと行動の対応関係

| フェーズ | 行動 |
|---------|------|
| Transition | Moves hand to target location |
| Transition | Hand at target location |
| Transition | Relaxes and moves hand to target location |
| Gesture | Performs gesture |

### Target/Non-Targetジェスチャーの分類

- **Target（BFRB様）**: 8種類のジェスチャー - 全て身体に関連した反復行動
- **Non-Target（通常）**: 10種類のジェスチャー - 日常的な活動や意図的な動作

全81名の被験者が両方のタイプのシーケンスを実行しています。

## データセット統計情報

### シーケンス長分布
- **最小長**: 29行
- **最大長**: 700行
- **平均長**: 70.54行
- **中央値**: 59行

### センサーデータ構成
- **IMUセンサー**: 7列（`acc_x`, `acc_y`, `acc_z`, `rot_w`, `rot_x`, `rot_y`, `rot_z`）
- **Thermopileセンサー**: 5列（`thm_1` ～ `thm_5`）
- **Time-of-Flightセンサー**: 320列（`tof_1_v0` ～ `tof_5_v63`）

### データ品質
- **欠損値**: 主要なカテゴリカル変数（sequence_type, gesture, orientation等）には欠損値なし
- **センサー通信障害**: 一部シーケンスでセンサーデータの欠損あり
- **データ型**: 数値データは64bit浮動小数点、カテゴリデータは文字列型

### 被験者情報
- **総被験者数**: 81名（train）、2名（test sample）
- **年齢範囲**: 10-53歳（平均21.8歳）
- **成人/子供比**: 成人52%、子供48%
- **性別比**: 男性62%、女性38%
- **利き手**: 右利き88%、左利き12%

### 2. [train/test]_demographics.csv
参加者の人口統計学的および身体的特徴を含む表形式ファイル

#### 項目詳細
- `subject`: 被験者ID（文字列型）
  - 形式: `SUBJ_XXXXXX`
  - 訓練データ: 81名、テストデータ: 2名
- `adult_child`: 参加者の年齢区分（整数型）
  - 値: `0` = 子供（18歳未満）、`1` = 大人（18歳以上）
  - 分布（train）: 子供39名（48%）、大人42名（52%）
- `age`: データ収集時の参加者の年齢（整数型）
  - 範囲（train）: 10-53歳
  - 平均: 21.8歳
  - テストデータ: 13歳、25歳
- `sex`: 出生時に割り当てられた性別（整数型）
  - 値: `0` = 女性、`1` = 男性
  - 分布（train）: 女性31名（38%）、男性50名（62%）
  - テストデータ: 両方の性別を含む
- `handedness`: 参加者の利き手（整数型）
  - 値: `0` = 左利き、`1` = 右利き
  - 分布（train）: 左利き10名（12%）、右利き71名（88%）
  - テストデータ: 両者とも右利き
- `height_cm`: 参加者の身長（浮動小数点型）
  - 範囲（train）: 135.0-190.5cm
  - 平均: 168.0cm
  - テストデータ: 165.0cm、177.0cm
- `shoulder_to_wrist_cm`: 肩から手首までの距離（整数型）
  - 範囲（train）: 41-71cm
  - 平均: 51.6cm
  - テストデータ: 両者とも52cm
- `elbow_to_wrist_cm`: 肘から手首までの距離（浮動小数点型）
  - 範囲（train）: 18.0-44.0cm
  - 平均: 25.5cm
  - テストデータ: 23.0cm、27.0cm

### 3. sample_submission.csv
提出フォーマットのサンプル

#### 項目
- `sequence_id`: シーケンスID
- `gesture`: 予測するジェスチャー

### 4. kaggle_evaluation/
評価APIを実装するファイル群。UNIXマシンでテスト目的のローカル実行が可能。

## 提出方法

### 評価API
- 提供されたPython評価APIを使用して提出する必要があります
- APIはテストセットデータを一度に1つのシーケンスずつ提供
- [デモノートブック](https://www.kaggle.com/code/sohier/cmi-2025-demo-submission/)で使用例を確認可能

### ダウンロードコマンド
```bash
kaggle competitions download -c cmi-detect-behavior-with-sensor-data
```

## データの時系列構造

各シーケンスは以下の3つのフェーズで構成されています：

1. **Transition**: 休息位置から適切な場所への手の移動
2. **Pause**: 短い休憩（何もしない）
3. **Gesture**: BFRB様またはnon-BFRB様のジェスチャーの実行

この構造により、行動の文脈と実際のジェスチャーの両方を含む包括的なデータセットが提供されます。

## 注意事項

### データ品質に関する注意
1. **センサー通信障害**: 既知の問題により、一部のセンサーからのデータが一部のシーケンスで欠損している可能性があります
2. **欠損値パターン**: Thermopile（`thm_`）とTime-of-Flight（`tof_`）センサーは、IMUのみのシーケンスでnull値を含みます

### テストデータの構造
3. **サンプルテストセット**: 提供されているテストデータは2シーケンスのみ（被験者2名）
   - `SEQ_000001`: SUBJ_016452
   - `SEQ_000011`: SUBJ_055840
   - 両シーケンスとも全センサー（IMU + Thermopile + ToF）データが利用可能
4. **本番テストセットの特徴**:
   - 予想シーケンス数: 約3,500
   - **重要**: 隠しテストセットの半分はIMUセンサーのみで記録
   - Thermopile（`thm_`）とTime-of-Flight（`tof_`）列は存在するが、これらのシーケンスではnull値を含む
   - この構造により、Time-of-FlightとThermopileセンサーの追加がBFRB検出能力を向上させるかを判断可能

### 提出に関する制約
5. **ジェスチャー値の制約**: 提出には訓練セットで見つからない`gesture`値を含めてはいけません（エラーが発生します）
6. **予測対象**: 18種類のジェスチャーのみ（Target: 8種類、Non-Target: 10種類）

### 実験設計上の考慮点
7. **センサー依存性の評価**: 本コンペティションはセンサーの組み合わせによる検出精度向上を評価することを目的としています
8. **体位依存ジェスチャー**: 一部のジェスチャーは特定の体位でのみ実行されるため、体位情報の活用が重要です