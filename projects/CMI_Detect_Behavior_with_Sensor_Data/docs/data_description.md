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
- `row_id`: 行ID
- `sequence_id`: センサーデータのバッチID。各シーケンスは1つのTransition、1つのPause、1つのGestureを含む
- `sequence_type`: ジェスチャーがターゲットタイプかnon-ターゲットタイプか（trainのみ）
- `sequence_counter`: 各シーケンス内の行カウンター
- `subject`: データを提供した被験者のユニークID

#### ターゲット変数
- `gesture`: 予測対象列。シーケンスのGestureの説明（trainのみ）

#### メタデータ
- `orientation`: シーケンス中の被験者の向きの説明（trainのみ）
- `behavior`: シーケンスの現在フェーズ中の被験者の行動の説明

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

### 2. [train/test]_demographics.csv
参加者の人口統計学的および身体的特徴を含む表形式ファイル

#### 項目
- `subject`: 被験者ID
- `adult_child`: 参加者が子供（`0`）か大人（`1`）かを示す。大人は18歳以上と定義
- `age`: データ収集時の参加者の年齢（年）
- `sex`: 出生時に割り当てられた性別（`0` = 女性、`1` = 男性）
- `handedness`: 参加者が使用する利き手（`0` = 左利き、`1` = 右利き）
- `height_cm`: 参加者の身長（センチメートル）
- `shoulder_to_wrist_cm`: 肩から手首までの距離（センチメートル）
- `elbow_to_wrist_cm`: 肘から手首までの距離（センチメートル）

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

1. **センサー通信障害**: 既知の問題により、一部のセンサーからのデータが一部のシーケンスで欠損している可能性があります
2. **テストセットの二重構造**: 半分はIMUのみ、半分は全センサーデータ
3. **ジェスチャー値の制約**: 提出には訓練セットで見つからない`gesture`値を含めてはいけません（エラーが発生します）