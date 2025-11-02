# CMI - Detect Behavior with Sensor Data

## 基本情報
- **コンペティション名**: CMI - Detect Behavior with Sensor Data
- **副題**: Predicting Body Focused Repetitive Behaviors from a Wrist-Worn Device
- **主催者**: Child Mind Institute
- **総賞金**: $50,000
- **参加者数**: 8,284エントラント、1,474参加者、1,351チーム
- **提出回数**: 18,475
- **タグ**: Health, Time Series Analysis, Custom Metric

## 概要
Body-focused repetitive behaviors（BFRBs）は、髪を抜く、皮膚をいじる、爪を噛むなどの自己向け反復行動で、頻繁または強い場合には身体的害と心理社会的挑戦を引き起こす可能性があります。これらの行動は不安障害や強迫性障害（OCD）でよく見られ、精神的健康の課題の重要な指標となります。

このコンペティションの目標は、手首装着デバイスから収集した動き、温度、近接センサーデータを使用して、BFRB様とnon-BFRB様の活動を区別する予測モデルを開発することです。

## タイムライン
- **開始日**: 2025年5月29日
- **エントリー締切**: 2025年8月26日
- **チーム統合締切**: 2025年8月26日
- **最終提出締切**: 2025年9月2日

すべての締切は、特に明記されていない限り、該当日のUTC時間23:59です。

## 賞金
- 1位: $15,000
- 2位: $10,000
- 3位: $8,000
- 4位: $7,000
- 5位: $5,000
- 6位: $5,000

## データ説明
Child Mind InstituteはBFRBsを調査するために手首装着デバイス「Helios」を開発しました。多くの市販デバイスがIMU（慣性測定装置）を含んでいるのに対し、Heliosウォッチには追加センサーが統合されています：
- 5つのthermopiles（体温検出用）
- 5つのtime-of-flightセンサー（近接検出用）

### ジェスチャーの種類

#### BFRB様ジェスチャー（8種類）
1. Above ear - Pull hair
2. Forehead - Pull hairline
3. Forehead - Scratch
4. Eyebrow - Pull hair
5. Eyelash - Pull hair
6. Neck - Pinch skin
7. Neck - Scratch
8. Cheek - Pinch skin

#### Non-BFRB様ジェスチャー（10種類）
1. Drink from bottle/cup
2. Glasses on/off
3. Pull air toward your face
4. Pinch knee/leg skin
5. Scratch knee/leg skin
6. Write name on leg
7. Text on phone
8. Feel around in tray and pull out an object
9. Write name in air
10. Wave hello

### 実験設定
参加者は各ジェスチャーを以下の4つの異なる体位で実行：
- 座位
- 前傾座位（非利き手を脚に置く）
- 仰向け
- 横向き

各ジェスチャーは以下の3つのフェーズで構成：
1. **Transition**: 休息位置から適切な場所へ手を移動
2. **Pause**: 短い休憩（何もしない）
3. **Gesture**: BFRB様またはnon-BFRB様のジェスチャーを実行

## 評価
評価指標は、以下の2つの要素を等しく重み付けしたマクロF1のバージョンです：

1. **Binary F1**: `gesture`がターゲットタイプかnon-ターゲットタイプかの二分類
2. **Macro F1**: `gesture`に対するマクロF1（すべてのnon-targetシーケンスを単一の`non_target`クラスに統合）

最終スコアはBinary F1とMacro F1スコアの平均です。

## 提出形式
このコンペティションでは提供された評価APIを使用して提出する必要があります。これにより、モデルが一度に1つのシーケンスで推論を実行することが保証されます。テストセットの各`sequence_id`に対して、対応する`gesture`を予測する必要があります。

## コード要件
- CPU Notebook: 実行時間9時間以下
- GPU Notebook: 実行時間9時間以下
- インターネットアクセス無効
- 事前学習済みモデルを含む自由に公開されている外部データは許可

## 重要な特徴
テストセットの評価時には：
- 半分はIMUのデータのみを含む
- 残り半分はHeliosデバイスのすべてのセンサー（IMU、thermopiles、time-of-flightセンサー）のデータを含む

これにより、thermopileとtime-of-flightセンサーの追加値を判断し、IMU単体と比較してBFRB検出精度の大幅な改善が追加の費用と複雑さに値するかどうかを判断できます。

## 関連記事
- Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/
- Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/

## 謝辞
このコンペティションで使用されるデータは、Healthy Brain Network（ニューヨーク市を拠点とする画期的な精神的健康研究）との協力により提供されました。財政的支援は、Kaggleチームによる寛大な支援に加え、Children and Youth Behavioral Health Initiative（CYBHI）の一環としてカリフォルニア州保健医療サービス局（DHCS）により提供されました。

## 引用
Laura Newman, David LoBue, Arianna Zuanazzi, Florian Rupprecht, Luke Mears, Roxanne McAdams, Erin Brown, Yanyi Wang, Camilla Strauss, Arno Klein, Lauren Hendrix, Maki Koyama, Josh To, Curt White, Yuki Kotani, Michelle Freund, Michael Milham, Gregory Kiar, Martyna Plomecka, Sohier Dane, and Maggie Demkin. CMI - Detect Behavior with Sensor Data. https://kaggle.com/competitions/cmi-detect-behavior-with-sensor-data, 2025. Kaggle.