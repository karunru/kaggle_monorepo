**You:**

competition_overview.md
ファイル
data_description.md
ファイル
このコンペで使える、時系列データに対するaugmentationを調査して

* * *

**ChatGPT:**

このコンペに適した時系列データのaugmentation手法を調査するために、以下を教えてください：

1. 使用予定のモデル（例：CNN, LSTM, Transformer など）が決まっていれば教えてください。
    
2. Augmentationの目的は何ですか？（例：クラス不均衡の解消、汎化性能向上、少数クラスの強化など）
    
3. 特に扱いたいセンサーデータの種類（IMU、thermopile、ToF）やフェーズ（Transition/Pause/Gesture）に制限がありますか？
    

この情報があると、より実践的で効果的なaugmentation手法を提案できます。

* * *

**You:**

1 @codes/exp/exp035/model.py
2 学習データが少ないので過学習を抑制して汎化精度を上げたい
3 ない、全部を対象にして

* * *

**ChatGPT:**

了解しました。CMIコンペで使用されるようなIMU主体の深層学習モデル（ResidualSE-CNN + BiGRU + BERT統合）に対し、全センサーデータ（IMU, Thermopile, ToF）を対象に、汎化性能向上のために活用できる時系列データのaugmentation手法を調査します。

調査結果がまとまり次第ご連絡します。

* * *

**ChatGPT:**

# 概要

CMI主催の「Detect Behavior with Sensor Data」コンペティションでは、腕時計型デバイスのIMU（加速度・ジャイロ）、サーモパイル（熱センサ）、Time-of-Flight（距離センサ）といった**マルチモーダル時系列データ**を用いて、BFRB（身体集中反復行動）の有無や種類を分類する課題が提示されています[scribd.com](https://www.scribd.com/document/895132504/Gmail-Competition-Launch-CMI-Detect-Behavior-With-Sensor-Data#:~:text=In%20this%20competition%2C%20you%E2%80%99ll%20train,focused)[scribd.com](https://www.scribd.com/document/895132504/Gmail-Competition-Launch-CMI-Detect-Behavior-With-Sensor-Data#:~:text=repetitive%20behaviours%20,by%20the%20Child%20Mind%20Institute)。しかし、医療・ヘルスケア分野の時系列データでは**ラベル付きデータが不足**しがちであり、深層学習モデル（特にLSTMやCNN）は過学習に陥りやすいという問題があります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=In%20recent%20years%2C%20convolutional%20neural,challenge%20for%20data%20aug%02mentation%20is)[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=labeled%20datasets%20are%20available%20%28e,However%2C%20it%20is%20not)。このような少数データ環境でモデルの汎化性能を高めるために有効なのが**データ拡張（データオーギュメンテーション）**です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=match%20at%20L585%20augmentation%20aims,or%20datasets%20with%20unequal)。データ拡張とは、既存の学習データに対してラベルを保ったまま様々な変換や加工を施し、**疑似的に追加の学習サンプル**を生成する手法の総称です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=generation%20using%20data%20augmentation%20has,Data)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=Data%20augmentation%20is%20a%20well,and%20colour%20augmentation%20on%20images)。適切なデータ拡張を行うことで、**モデルの決定境界を広げて汎化性能を向上**させ、過学習の抑制につながることが多くの研究で報告されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=match%20at%20L585%20augmentation%20aims,or%20datasets%20with%20unequal)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=augmentation%20aims%20to%20decrease%20overfitting,or%20datasets%20with%20unequal)。実際、Parkinson病患者の動作状態分類にCNNを適用した研究では、データ拡張によって分類精度が **77.5% から 86.9% に大幅向上**した例もあります[arxiv.org](https://arxiv.org/abs/1706.00527#:~:text=,to%2086.88)。また、複数の身体活動データセット（CogAgeやUniMiB-SHAR）に15種類の拡張を適用した最近の研究でも、拡張なしと比べ**複合動作で5%、単純動作で30%**もの精度向上が報告されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=classification,are%20explained%20in%20the%20following)。

本レポートでは、**LSTMやCNNベースの時系列モデルで実績のあるデータ拡張手法**を概観し、特に*(1)* 少量データでの汎化精度向上、_(2)_ マルチセンサ高次元データへの適用可能性、_(3)_ 医療・ヘルスケア領域の動作分類タスクで有効であった事例、という観点で有用な手法に焦点を当てます。それぞれの手法について、**適用方法（コード例や利用可能なライブラリ）**、期待される**メリットと注意点（デメリット）**、そして**文献やベンチマークでの実績**を整理します。

## データ拡張手法一覧（主要な時系列データ拡張）

以下に、本レポートで取り上げる代表的な時系列データ拡張手法をリストします。いずれも**ラベルを変えずに**元データに変形を加えることで新たな学習サンプルを作り出すものです。

* **ノイズ付加（Jittering）** – センサ信号に微小な乱数ノイズを加える手法
    
* **スケーリング（Scaling）および振幅ワーピング** – 信号全体の振幅を拡大・縮小する、または時変的に変化させる手法
    
* **時間ワープ（Time Warping）** – 時間軸方向に信号を伸縮させて局所的に速度を変化させる手法
    
* **区間シャッフル（Permutation）** – 時系列をいくつかの区間に区切り、その順序をランダムに入れ替える手法
    
* **ランダム切り出し・スライス（Cropping）** – 時系列の一部区間をランダムに切り出したり、開始位置をずらしたりする手法
    
* **回転（Rotation）** – 主にIMUの三軸データに対し、座標軸を回転させることでセンサの向きの変動をシミュレートする手法
    
* **部分マスク（Hide-and-Seek）** – 時系列の一部区間をマスク（欠損値やゼロ埋め）して隠す手法
    
* **ミックスアップ（Mixup）** – 2つの時系列を重み付け平均して合成し、新たな合成データとする手法
    
* **カットミックス（CutMix）** – 2つの時系列の一部区間を切り貼りして1つの時系列に合成する手法
    
* **データ合成（GAN・拡散モデルなど）** – 学習した生成モデルにより新規の時系列データを人工的に生成する手法
    

以下、各手法について詳しく解説します。

## 手法別解説

### ノイズ付加（Jittering）

**図1:** センサーデータにランダムノイズを加える**ノイズ付加（jittering）**の例。青線が元データ、橙線が各時刻の値に小さなガウス雑音を加えた波形です。ノイズ付加は**最もシンプルな拡張**手法であり、センサーデータに存在し得る計測誤差や環境ノイズをシミュレートします[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Jittering%20involves%20adding%20random%20Gaussian,patterns%20in%20the%20time%20series)。微小な乱数ノイズ（例えば平均0・標準偏差0.05程度のガウス分布）を各チャネルの各時刻に加えることで、新たな時系列を生成します。実装は容易で、PythonではNumPyで乱数を加算するだけで済みます。例えば `aug_signal = original + 0.05 * np.random.randn(*original.shape)` のようにコード一行で適用可能です。また、専用ライブラリを用いる場合、`tsaug`では`AddNoise(scale=0.05)`のような関数で一括してノイズ付加を行えます。

ノイズ付加の**メリット**は、データのばらつきを人工的に増やすことで**モデルのノイズ耐性を高められる**点です[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=introducing%20small%2C%20random%20perturbations%2C%20jittering,patterns%20in%20the%20time%20series)。これにより訓練データに過度適合したモデルが、本番環境で遭遇するセンサノイズにも対応できるようになります[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Jittering%20involves%20adding%20random%20Gaussian,patterns%20in%20the%20time%20series)。実際、ウェアラブルセンサによるParkinson病モニタリング研究では、ノイズ付加を含む拡張によりモデル精度が大きく向上しています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=applied%20to%20the%20classification%20of,CCS%20CONCEPTS)。一方**デメリット・注意点**として、ノイズの強度を誤ると**信号の特徴が破壊**されラベル不一致を起こす可能性があります。特にデータ数が極端に少ない場合、過剰なノイズ付加はかえって学習を不安定にするため、ノイズレベルはタスクに応じ慎重に調整すべきです。また、医療分野の計測データではノイズ成分自体に意味がある場合もあり、一概にノイズを加えれば良いとも限らない点に留意が必要です。

### スケーリング・振幅ワーピング（Scaling & Magnitude Warping）

**スケーリング（Scaling）**は時系列全体の振幅を一様に拡大・縮小する拡張です。具体的には、元データに対し**一定のスケール係数**（例: 平均1・分散0.1程度の乱数）を乗じます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,represents%20mathematical%20notation%20of%20scaling)。こうすることで、動作の強さ（振幅）に関するバリエーションを生成できます。例えば人の動作では、同じ「髪を触る」動作でも激しく動かす人とゆっくり動かす人がいるため、スケーリングでその振幅差を再現できます。実装は簡単で、例えばNumPy配列`x`に対し`aug_x = scale * x`とするだけです。`tsaug`ライブラリでは直接のScaling関数はありませんが、乱数生成して乗算する処理をパイプラインに組み込めます。

一方、**振幅ワーピング（Magnitude Warping）**は振幅スケーリングを時間変化させた拡張です。元データに対し、時間に沿ってゆっくり変化するスケール曲線（例えば**スプライン補間でなめらかに変化する乱数係数の系列**）を乗じます[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Magnitude%20warping%20adjusts%20the%20amplitude,learn%20to%20handle%20amplitude%20variations)。これにより、時系列内で部分的に振幅が強まったり弱まったりするパターンを生成できます[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Magnitude%20warping%20adjusts%20the%20amplitude,learn%20to%20handle%20amplitude%20variations)。図式的には、元波形に対し緩やかな起伏のある補正曲線を掛け合わせるイメージです。振幅ワーピングは、単純なスケーリングよりも**現実的な強弱変化**を再現できるため、データ不足時の精度向上に寄与することが報告されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=and%20Choi%20,the%20purpose%20of%20gait%20identification)。たとえば**歩容識別（歩き方の個人識別）**の研究では、振幅の拡大縮小とノイズ付加を組み合わせた拡張で性能向上が見られました[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=discovered%20that%20using%20rotation%20data,the%20purpose%20of%20gait%20identification)。実装は、いったん長さNの乱数系列を生成しスプライン補間で滑らかにし、それを元データに要素ごと乗算します。Python実装例として、SciPy等でスプライン補間 (`scipy.interpolate.CubicSpline`) を用いて係数曲線を生成し乗じることが考えられます。

**メリット:** 振幅方向のばらつきを付与することで、モデルが**信号の絶対的な大きさに依存しすぎるのを防ぐ**効果があります。同じクラスでも強弱の差が大きい動作（例: 軽い動作と激しい動作）が存在する場合、スケーリング系の拡張は有効です[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Magnitude%20warping%20adjusts%20the%20amplitude,learn%20to%20handle%20amplitude%20variations)。特にウェアラブルの加速度データでは装着位置や個人差で振幅が変動するため、それを補完する合成データを得られます。**デメリット・注意点:** 一方で**タスクによっては振幅そのものがクラス識別に重要**な場合があります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=obvious%20how%20to%20carry%20out,paper%2C%20we%20propose%20data%20aug%02mentation)。例えば動作の**強度**でクラス分けするような場合、不用意にスケーリングするとラベルが変わってしまう恐れがあります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=obvious%20how%20to%20carry%20out,paper%2C%20we%20propose%20data%20aug%02mentation)。実際、Parkinson病の状態評価では動きの大きさ自体が症状指標となるため、著者らも「単純なスケーリングは症状ラベルを破壊しかねない」と注意を述べています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=obvious%20how%20to%20carry%20out,paper%2C%20we%20propose%20data%20aug%02mentation)。従って、振幅に関わる拡張は**ラベル不変となる条件下でのみ適用**すべきです。また、振幅ワーピングでは曲線の形状を決めるハイパーパラメータ（節点の数や変化の程度）の調整が必要であり、複雑な変形はかえって実データに存在しないパターンを生むリスクもあります。

### 時間ワープ（Time Warping）

**時間ワープ（Time Warping）**は、時系列データの時間軸方向に歪みを加える拡張手法です。具体的には、**一部の区間を圧縮したり拡大したり**して時間的な進行速度を変化させます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,mathematical%20notation%20of%20time%20warping)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=introduces%20diversity%20and%20enhances%20the,bending%29%20activity)。例えば、ある動作が本来1秒かけて行われるところを、前半を0.8秒に圧縮し後半を1.2秒に拡大するといった具合に、時間的な伸縮を滑らかに施します。図2に時間ワープのイメージを示します。

**図2:** センサーデータに対する**時間ワープ（Time Warping）**の例。青線が元データ、橙線が時間軸上でランダムに歪めたデータです。橙線では一部で時間軸が圧縮され、逆に他の部分では引き伸ばされており、局所的な動作速度の変化を再現しています。時間ワープは、人間の動作が速くなったり遅くなったりする**テンポのばらつき**をデータに与える効果があります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=match%20at%20L859%20where%20they,permutation%20on%20Parkinson%E2%80%99s%20disease%20data)。例えばBFRBの一種である「髪を引っ張る動作」でも、人によって素早く引っ張る場合とゆっくり引っ張る場合が考えられます。時間ワープしたデータを学習に含めることで、モデルは**動作の実行速度が異なっても共通のパターンを認識**しやすくなります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=match%20at%20L859%20where%20they,permutation%20on%20Parkinson%E2%80%99s%20disease%20data)。実際、ウェアラブルセンサを用いた動作認識の文献でも、時間ワープによる拡張で精度向上が確認されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=match%20at%20L859%20where%20they,permutation%20on%20Parkinson%E2%80%99s%20disease%20data)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=where%20they%20employed%20an%20ensemble,permutation%20on%20Parkinson%E2%80%99s%20disease%20data)。Parkinson病データに対しても、時間ワープで症状発現のタイミングをずらすことでモデル性能が向上したとの報告があります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。

実装面では、時間ワープは高度な補間処理を要します。代表的な方法は、元データのインデックス（時間軸）に対し**ランダムに歪んだ新しい時間軸**を生成し、元データをその新時間軸に沿って再サンプリングするやり方です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,mathematical%20notation%20of%20time%20warping)。歪みは滑らかにする必要があるため、例えば**いくつかの制御点を乱数でずらし**、それらを補間して変形関数を得ます（文献では三次スプラインで滑らかな時間歪み関数を生成しています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=Here%2C%20denotes%20a%20warping%20function,bending%29%20activity)）。Pythonで自前実装する場合、NumPy配列のインデックスを浮動小数で変換し、`numpy.interp`や`scipy.interpolate`で値を補間するとよいでしょう。より手軽には、前述の`tsaug`ライブラリに`TimeWarp(n_segments=...)`という実装があり、指定した区間数でランダムに時間軸を歪めることが可能です。

**メリット:** 時間ワープは**動作パターンの時間的位置への依存を減らし**、モデルをテンポや位相の変動に対して頑健にします[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。これにより、限られたデータから学習する際にも、例えば「動作Aが窓の前半に現れる場合と後半に現れる場合」の両方に対応できるモデルが得られます。特に**時系列内のイベント発生時刻が恣意的**なタスク（ウィンドウ区間内のどこで起きても同じクラス）では有効です[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)。Parkinson病の症状検知でも、震えなどの症状が時間窓のどの位置に現れても検出できるよう、時間ワープや窓位置シフトでデータを水増ししています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=additive%20noise%20and%20improve%20performance,the%20next%20section%2C%20the%20performance)。**デメリット・注意点:** 時間ワープは**シーケンスの時間的な意味合いを変えてしまう**リスクもあります。特に動作の順序や持続時間そのものが重要なタスクでは、無闇な時間変形は不適切です。例えば「歩行→停止→着席」といった一連の動作分類では、その順序が崩れると非現実的なデータになります。また、補間によって生じたデータ点が現実とかけ離れる可能性もあります（過度な時間圧縮で不自然な急激変化が生じるなど）。時間ワープを使う際は、**人間の動作速度の現実的な範囲内**で歪み具合を設定することが重要です。

### 区間シャッフル（Permutation）

**区間シャッフル（Permutation）**は、時系列データを複数の連続区間に分割し、その順序をランダムに並べ替える拡張です[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)。例えば長さ100の系列を10ずつの区間に区切り、[区間1,区間2,...区間10]を[区間3,区間1,...区間7]のようにシャッフルして繋ぎ直します。目的は、**時間窓内でイベントが発生する位置の影響を減らす**ことです[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)。固定長のウィンドウでラベル付けする場合、イベント（特徴的な信号変化）が窓の前半にあるか後半にあるかで学習が偏る恐れがあります。Permutationにより意図的に順序を崩すことで、モデルが**時系列内の位置によらず特徴を捉える**よう促します[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。Parkinson病データの研究では、この区間シャッフルで**震え症状の発生位置**を揺らし、モデル精度が改善したと報告されています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。

実装は比較的容易で、配列を等分割し、それらの順番を`np.random.permutation`で並べ替えてから再結合します。ただし**マルチチャネルデータの場合、チャネル間の対応を保つ**必要があります。例えば加速度センサのx,y,z軸があるとき、一つのチャネルだけ区間をシャッフルすると軸間の時間ずれが生じてしまいます。従ってPermutationは**全チャネル一括で同じ区間境界で**実施します。`tsaug`などには直接のPermutation機能はありませんが、自前で簡単にコーディング可能です。

**メリット:** 区間シャッフルにより**時系列内の局所的な順序入れ替えに対する不変性**が獲得できます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)。特に、一つのウィンドウ内で複数のイベントや動作が順不同に起こりうる場合（またはウィンドウの開始位置が恣意的な場合）に効果があります。Permutationは時間ワープほど連続性を保たない変形ですが、その分ラベルが変わらない条件で適用しやすい場合があります。**デメリット:** 一方で**時系列の持つ文脈や因果関係が崩れる**点には注意が必要です。多くの人間の動作は順序が意味を持つため、Permutationで生まれたデータが現実には起こり得ない順序になることがあります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Lastly%2C%20cropping%20,the%20next%20section%2C%20the%20performance)。例えば「手を上げる→手を下ろす」という順序が重要な場合、それを入れ替えると不自然です。したがってPermutationは**順序自体が分類に寄与しない場合に限定**すべきです。また、区間の粒度も課題です。短すぎる区間でシャッフルすると原形を留めない信号となりかねず、長すぎる区間では効果が薄れます。適切な区間長（全長に対する比率）をデータの特性に応じて選ぶ必要があります。

### ランダム切り出し・スライス（Cropping）

**ランダム切り出し（Cropping）**は、時系列データから**一部の連続区間を切り出す**ことでデータを拡張する手法です。これは画像領域のランダムクロッピングに相当し、時系列の場合は主に**開始位置をランダムにずらして一定長を抽出する**形で用いられます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Lastly%2C%20cropping%20,the%20next%20section%2C%20the%20performance)。例えばセンサデータの1分間ウィンドウを30秒に短縮するような極端な切り出しよりも、1分間ウィンドウ内で**少し前後にずらした1分間**を取り出す使い方が一般的です。実質的にはウィンドウの**ランダム時間シフト**と捉えることもできます。こうすることで、ウィンドウ境界に対する頑健性が向上し、モデルが特定の位置に現れる特徴に依存しにくくなります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Lastly%2C%20cropping%20,the%20next%20section%2C%20the%20performance)。Parkinson病データの研究でも「窓内に症状が入る位置は恣意的なので、エポック間でランダムに窓をシフトさせる（Croppingと同等）」手法が提案されています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=additive%20noise%20and%20improve%20performance,the%20next%20section%2C%20the%20performance)。

実装方法として、元の時系列データから目的の長さ分の部分配列をランダムなオフセット位置からコピーします。例えば1,000サンプルの系列から同じ長さ1,000の系列を作る場合でも、元データの100サンプル目から1,099サンプル目までを取り出す、といったことが可能です（元と長さが同じ場合は一部重複領域が外れるだけですが、開始位置が異なる分学習効果があります）。`tsaug`には`Crop(size=L)`という関数があり、指定長Lの連続区間をランダムに抽出できます。また、元データ長より短い長さでCropすることで**一部欠落した系列**を作り出すこともできます。例えばセンサ停止や欠損を模擬する目的で30秒間のデータを丸ごと切り落とすといった応用も考えられます。

**メリット:** Croppingによって**データ収集時のタイミング依存**を低減し、モデルの汎化力を高めます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=additive%20noise%20and%20improve%20performance,the%20next%20section%2C%20the%20performance)。特にセンサを一定時間ごとに区切って特徴量化する場合、区切りの境界にイベントがかかると性能に影響することがありますが、ランダム切り出しでデータを増やすことでそうした影響を平均化できます。また、データそのものを間引く効果で**過剰な特徴への依存を防ぐ正則化**にもなります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=additive%20noise%20and%20improve%20performance,the%20next%20section%2C%20the%20performance)。**デメリット:** 一方で**切り出しによって重要な情報まで削ってしまう**リスクがあります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Lastly%2C%20cropping%20,the%20next%20section%2C%20the%20performance)。例えばBFRB検知で、該当行動が窓の端に1秒だけ含まれるようなケースでは、切り出した結果その行動部分が消えてしまい誤ったラベル付きデータになる恐れがあります[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Lastly%2C%20cropping%20,the%20next%20section%2C%20the%20performance)。このため、重要なイベントが窓内に必ず収まっている状況で使う、あるいはイベントが窓内に複数回出現するような長時間データに適用するのが望ましいです。また、時系列全体の長さが変化するため、モデル側で可変長を扱える必要があります（LSTMは可変長系列を扱えますが、CNNは通常固定長入力のためパディング等が必要）。

### 回転（Rotation）

**回転（Rotation）**は、IMUなど多次元の時系列データに対して各軸を回転変換する拡張です[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=%28Fig%202%29,introduce%20variability%20is%20the%20temporal)[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=Permutation%20,labels%2C%20depend%02ing%20on%20the%20target)。特に3軸加速度やジャイロデータでは、センサの取り付け角度や身体の姿勢の違いによって軸方向の値が変化します。回転による拡張は、そうした**センサの向きの不確定性を吸収**する目的で用いられます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=%28Fig%202%29,introduce%20variability%20is%20the%20temporal)。具体的には、3次元ベクトル$(x(t), y(t), z(t))$に対し、ランダムな回転行列R（3x3の正規直交行列）を掛けて$(x',y',z')^T = R (x,y,z)^T$を得ます。この操作により、元の動きを別の向きから観測した場合のデータが生成できます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=%28Fig%202%29,introduce%20variability%20is%20the%20temporal)。例えば腕に付けたIMUが上下逆さまになった場合、加速度の符号が反転しますが、回転拡張ではそのような**装着向きのばらつき**を学習に含められます[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=%28Fig%202%29,introduce%20variability%20is%20the%20temporal)。実際、Parkinson病データに対するCNN分類では**回転拡張が単体で最も性能向上に寄与**し[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)、「センサの上下逆や左右入れ替えなどを広く想定できるようになった」と述べられています[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=%28Fig%202%29,introduce%20variability%20is%20the%20temporal)。BFRB検知でも、デバイスの装着位置・向きが人によって異なる可能性がありますし、右手・左手の差異なども回転である程度カバーできると期待されます。

実装上は、三次元の場合**ランダムなオイラー角**を生成しそれに相当する回転行列Rを作成します。そして各時刻のベクトルにRを適用します。Pythonでは`scipy.spatial.transform.Rotation`を使うと便利で、ランダムな回転を生成して`Rotation.apply(array)`でまとめて適用できます。より簡略には、特定軸についての180度回転（例えば上下反転ならx,y軸を反転、左右対称なら特定軸交換など）を実装することも考えられます。実データでどの程度の向きばらつきがあるかによりますが、**少なくとも主要軸の反転**はトライする価値があります。

**メリット:** 回転拡張により**センサ配置や装着方法の違い**によるデータ変化にモデルが対応できるようになります[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Rotation%20simulates%20different%20sensor%20placements,its%20generalization%20to%20diverse%20scenarios)。特に医療・ヘルスケア用途では、個人ごとにセンサを付ける位置・角度が完全に一定とは限らないため、回転した合成データを学習に含めることは有用です[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Rotation%20simulates%20different%20sensor%20placements,its%20generalization%20to%20diverse%20scenarios)。上記のParkinson病研究では、「上下逆につけた場合のデータ」をシミュレートした回転拡張が精度改善に大きく貢献しました[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。また2025年の転倒検知研究でも、左手首デバイスのデータを右手首に見立てて回転させることで、**センサ位置に依存しない検出**が可能になると述べています[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Rotation%20simulates%20different%20sensor%20placements,its%20generalization%20to%20diverse%20scenarios)。このように**マルチセンサ・高次元データに特有の幾何学的変換**として有効です。**デメリット・注意点:** 回転には**ドメイン知識が必要**です。すべてのタスクで無条件に有効とは限りません。例えば姿勢推定や方位そのものを当てるタスクでは、回転するとラベルが変わってしまいます。また、センサの重力軸との相対関係が重要な特徴の場合、安易に回転させるとデータの物理的意味を損ないます。さらに、回転によって生まれたデータが本当に現実にあり得るか（関節の可動範囲を超えるような向きになっていないか）も考慮すべきです。従って回転拡張は、**向き以外の要素でクラスが規定されるタスク**（動きのパターン自体に注目するタスク）に限定し、回転角度も現実的な範囲・対称性に留めるのが良いでしょう。例えばBFRB検知では手首の回転は360度あらゆる向きが理論上あり得ますが、人間の手の動きとして意味が通る範囲に限定して拡張する配慮が望ましいです。

### 部分マスク（Hide-and-Seek）

**部分マスク（Hide-and-Seek）**は、時系列データをいくつかの区間に区切り、そのうち**ランダムに選んだ一部区間の情報を隠す（マスクする）**拡張です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。具体的には、例えば時系列を5つの等分区間に割り、そのうち2区間を選んでその部分の値をゼロや平均値で埋めます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。隠す区間は逐次ではなく散発的（時系列中のランダムな場所）に選びます。こうすることで、**センサデータの一部欠損やセンサ蓄積値のドリフト**などを模擬できます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。この手法はコンピュータビジョンでいう「Cutout（画像の一部を黒塗りする）」に相当し、時系列版Cutoutとも言えます。近年の研究では“Hide-and-Seek”とも呼ばれ、複数区間を確率的にマスクする実装が提案されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。

マスクする値はゼロでもよいですが、急激な不連続を避けるため**系列全体の平均値**などで埋めると良いとされています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。実際、文献【10】では隠す部分を「データセット全体の平均値」で置き換えることで、大きな飛び値にならないよう配慮しています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。実装としては、まず何分割するかとマスクする割合を決め、各サンプルに対してマスク区間を乱択して置換します。例えば100長の系列を10区間に区切り、30%の区間（3区間）をマスクする、といった具合です。`tsaug`ライブラリでは`Dropout(p=0.1, fill=0)`のような関数で**ランダムな点や短区間のドロップアウト**を実現できます[tsaug.readthedocs.io](https://tsaug.readthedocs.io/en/stable/#:~:text=,dropped%20out%20points%20with%20zeros)[tsaug.readthedocs.io](https://tsaug.readthedocs.io/en/stable/#:~:text=,dropped%20out%20points%20with%20zeros)。Hide-and-Seekほどまとまった区間ではありませんが、類似の効果が得られます。より大きな区間をマスクするには、自前で区間選択→値置換のコードを組む必要があります。

**メリット:** 部分マスクはモデルに**頑健性と冗長性への対応力**を持たせます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。つまり、入力の一部が欠けていても残りから判断を下せるよう学習させるのです。センサデータでは実際に**一部センサが故障したり、一時的に信号が途切れたり**することがあります。例えば熱センサが一瞬遮られる、ToFセンサがノイズで飛ぶ、といった事態でも、他の情報から補完して判断できるモデルが望まれます。Hide-and-Seekで訓練データにそのような欠損を含めることで、**欠損耐性の高いモデル**が得られます[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。また、特定の短いパターン（区間）だけにモデルが過度適合するのも防げます。**デメリット:** マスクした部分に本来**重要な手がかりが存在する場合、ラベルと不整合**が生じます。例えばBFRBの決定的瞬間を含む区間をマスクすると、その擬似データではBFRBが起きていないかのように見えてしまいます。こうしたデータで学習させると混乱を招く可能性があります。ただ文献【10】では、Hide-and-Seekにより**ノイズや欠損があっても主要パターンを検出できる**性能向上が確認されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。結局のところ、マスク率はタスクに合わせ慎重に設定すべきです（例: 全長の20%程度までなど）。また、マスクは**時系列全チャネル同時**に行う必要があります。あるセンサだけ消える状況も現実にはありますが、その場合は別途「チャネルドロップアウト」として扱う方が良いでしょう（複数センサのうち一つを丸ごと欠測にする拡張も考えられますが、今回のHide-and-Seekは全チャネル横断的に部分欠損させる手法です）。

### ミックスアップ（Mixup）

**ミックスアップ（Mixup）**は、本来は画像分類で提案された拡張手法で、**2つの異なるサンプルを線形結合して新たな合成サンプルを作る**方法です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,226)。具体的には、2つの入力データ$(X_i, Y_i)$と$(X_j, Y_j)$（Yはone-hotラベルベクトル）に対し、$\lambda \in [0,1]$で重み付けして$\tilde{X} = \lambda X_i + (1-\lambda) X_j$、$\tilde{Y} = \lambda Y_i + (1-\lambda) Y_j$とします[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,226)。画像の場合はピクセル値の混合ですが、時系列でも**対応する時刻同士の値を加算平均**することで適用できます。生成された$\tilde{X}$は両方の特徴を併せ持った中間的なデータで、ラベルもソフトな重み付きになります。本来のMixupは異なるクラス同士も混合しますが、**時系列分類の場合はラベルの曖昧さを導入したくない**との理由から、**同じクラス同士のデータをMixupする**ことも考えられます（この場合$\tilde{Y}$は元と同じクラスになります）。実際、前出の拡張手法比較研究【5】では「Sub Averaging」という名称で**同一クラス・異なる被験者**の時系列を平均した拡張を試しています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,221)。これはMixup($\lambda=0.5$)の特殊ケースと言えます。結果として、**クラス内の個人差を緩和**する効果があると報告されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,221)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=)。

実装面では、まず混合させるペアを作る必要があります。データ数が少ない場合、同じデータが複数回使われることになりますが、問題ありません。$\lambda$は例えばBeta分布(α=β=0.2程度)からサンプリングするのが一般的です。時系列長が同じでないとMixupしにくいため、Padding等で長さを揃えておく必要があります。また、マルチチャネルデータでも各チャネルごとに対応する値を重み平均するだけでよいので、チャンネル構成は維持されます。PyTorch等でテンソル演算として実装することも容易です。なお、ラベルをone-hotでなく整数で管理している場合、Mixup後の中間ラベルを扱うために損失関数を工夫（ソフトターゲット対応）する必要があります。もし同一クラス同士でのみMixupするなら、ラベルはそのままで問題ありません。

**メリット:** Mixupは**学習データ空間における補間的なサンプル**を提供し、モデルに**線形関係の仮定**を促す正則化効果があります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,226)。これにより外挿的な予測を避け、未知のデータにもなだらかに対応できるモデルになるとされています。特に小さなデータセットでは訓練点間を補完する役割を果たし、汎化性能を高めます。実際、画像分野ではMixupで大幅な精度向上が確認されており、時系列でも異なる被験者のデータをMixupして**個人差を低減**する効果が報告されています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,221)。また、クラス間Mixupを許容すればデータ分布を広くカバーできます。**デメリット:** ラベルが混在するため**解釈が難しく**、特に明確なクラス境界が必要な応用では不適切です。医療データで異なる疾患クラスをMixupすると、「半分健康で半分病気」のようなデータになり、人間には意味を持たないケースもあります。モデルがその中間データでもうまく学習できるかも保証できません。したがって**Mixupは主に同一クラス内**で使うか、あるいは回帰タスク（連続値予測）には適していますが分類タスクでは慎重さが要ります。またMixup後のデータは現実には存在しない波形になるため、極端な混合だとモデルが実データからズレた特徴を学習してしまう恐れもあります。総じて、Mixupは**強力だが調整が必要な拡張**と言えます。

### カットミックス（CutMix / 部分合成）

**カットミックス（CutMix）**は、2つのデータから一部区間を切り取って**組み合わせる**拡張手法です[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,11)。画像分野のCutMixでは画像の一領域を他クラスの画像パッチで置換しますが、時系列の場合は**時間軸上のある区間を他の系列の同区間と入れ替える**操作になります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,9)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,11)。例えば、被験者Aのセンサデータ[0s～60s]のうち20s～30sの部分を、被験者B（同じクラス）の20s～30sデータに差し替える、といった具合です。この場合、生じた合成データは主にAの動作ですが一部Bの特徴も含みます。ラベルは両者が同じクラスであれば変わりません（異なるクラスの場合はMixupと同様にソフトラベルになりますが、通常は**同一クラス同士**で適用します[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,9)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=and%20are%20two%20initial%20data,bending%29%20activity)）。文献【5】では「Sub Cutmix」として**同クラス・異なる被験者**のシリーズを一部分置換する拡張を試み、データ多様性が増してモデルの頑健性が向上すると報告しています[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,9)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=variety%20by%20combining%20information%20from,9)。

CutMixの実装では、まずシリーズ長の中で**置換する区間の開始位置と長さを乱択**します。区間長は全長の何割以下、など制限を設けるのが一般的です。置換先の別シリーズも、同じ位置・長さの切片を取ります（ずらすと長さ不一致になるため通常は同期させます）。あとはその区間をスライス代入で入れ替えます。例えば長さ1000の系列について、開始300長さ200を別系列の300-500で置換、のようにします。切り貼りなので**接続部分で不連続**が生じますが、これはデータによって許容される場合とそうでない場合があります。必要であれば接続部をなだらかに繋ぐためフェードイン・アウト（重み平均で端を繋ぐ）処理を施してもよいでしょう。

**メリット:** CutMixはMixupと異なり**元データの一部そのものを利用**するため、より**現実的な変化**を与えます。例えば「歩行」のデータ同士でCutMixすれば、前半は人Aの歩行、後半は人Bの歩行というデータになりますが、どちらも現実的な歩行の波形です。モデルは**異なる個体やセッション間で生じる突然の変化にも対応**できるようになります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。特に長い継続動作を扱う場合、別々の記録を繋ぐことでデータ量を水増しできます。また、一つの動作クラスに属する多様なパターンを組み合わせることで、**クラス内のばらつき学習**にもなります[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=and%20are%20two%20initial%20data,bending%29%20activity)。**デメリット:** カットミックスで生成した系列は**境界で不自然な継ぎ目**ができます。動作が連続的であることが前提のタスクでは、その不連続がモデルに変な影響を与える恐れがあります。例えば本来一続きの動作なのに途中で別人の動きに飛ぶので、速度やリズムが不連続になる可能性があります。モデルがそこをロバストに扱えるかは未知数です。さらに、異なるクラス間でCutMixすると前半はクラスA・後半はクラスBという系列になり、ラベル付けが困難です（Mixupのようなソフトラベルを使う手もありますが複雑になります）。従って**CutMixも原則同一クラス内**で適用します。また、置換する区間長の決定も難しいです。長すぎるとそれはもはや別のサンプルを丸ごと繋いだだけになり、短すぎるとノイズに近くなります。本質的に**タスクの文脈を考慮**した調整が必要な高度な手法です。とはいえ、医療領域でも徐脈と頻脈など**状態が途中で変化する**ような時系列には、異なるケースを繋いだデータで学習させることが有効な場合があります。BFRB検知でも、一つのセッション中にBFRB動作と非BFRB動作が混在しうるなら、その連接を学習させるためにCutMix的拡張が使えるかもしれません。

### 生成モデルによるデータ合成（GAN・拡散モデル 等）

ここまで述べてきた拡張は、元データにランダムな変形を加える手法でした。一方で、**ディープラーニングを用いた生成モデル**によってデータ分布から**新規サンプルを生成**するアプローチも注目されています。代表的なものにGAN（Generative Adversarial Network）やVAE（Variational Autoencoder）、最近では拡散モデル（Diffusion Model）があります。これらは教師ありモデルとは別に**データ生成専用に訓練**し、訓練済み生成モデルから無限に疑似データをサンプリングできます。医療領域ではデータ収集が困難なため、生成モデルで補おうという研究が増えています。例えば2025年に発表されたウェアラブル転倒検出の研究では、**拡散モデル**を用いて加速度センサの人工的な「転倒」データを大量生成し、モデルに学習させたところ、従来のランダム変換による拡張よりも高い検出性能を示しました[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=distributional%20fidelity%20compared%20to%20traditional,the%20robustness%20and%20generalizability%20in)。特に3つの公開データセットすべてで拡散モデル拡張が最良となり、**他の拡張手法を一貫して上回る汎化性能**を示しています[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=distributional%20fidelity%20compared%20to%20traditional,the%20robustness%20and%20generalizability%20in)。これは生成モデルが元データ分布の複雑な特徴（転倒動作の多様なパターン）を捉え、ランダム変形では得られないリアルな合成データを供給できたためと考えられます。

実装には高度な知識が要りますが、近年は時系列用GANや拡散モデルのオープンソース実装も増えています。例えば**TimeGAN**（観測値と潜在時系列を組み合わせて学習するGAN）や、上記転倒検出で使われた**時系列拡散モデル（時系列用に1次元CNNで構築したDiffusion+U-Net）**[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=4)[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=We%20have%20integrated%20a%20Diffusion,temporal%20dependencies%20in%20the%20data)があります。これらを用いることで、元データの統計的性質を保ちつつ、新たな時系列を自由に生成できます。拡散モデルは画像ほど高解像度ではない時系列データでも効果を発揮しつつあり、実際に**従来のAugmentationではカバーしきれない微妙な動作ニュアンスも再現**できると報告されています[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=distributional%20fidelity%20compared%20to%20traditional,the%20robustness%20and%20generalizability%20in)[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=using%20the%20SmartFallMM%20dataset%20,score%20of)。

**メリット:** 生成モデルによる合成データは非常に**多様でリッチなサンプル**を提供し得ます。特に少数の実データからでも潜在分布を学習して外挿的な新データを生むため、データ空間を大きく補完できます。前述の転倒検出では、ビデオから抽出した姿勢データを使って新しい加速度波形を生成する試みもなされており、**他モーダル情報を組み合わせた高度なデータ合成**も可能です[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=data%20augmentation%20techniques%2C%20such%20as,of%20falls%20in%20older%20adults)[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=4)。また、生成モデルは**データ拡張と特徴抽出を同時に行える**ため、自己教師学習やコントラスト学習と組み合わせて精度向上を図る研究もあります[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=approaches%3A%20traditional%20augmentation%20techniques%2C%20Diffusion,different%20metrics%20to%20evaluate%20the)。**デメリット:** 最大の課題は**実装と学習の難易度**です。生成モデルを訓練するには元データ量自体ある程度必要で、かつハイパーパラメータの調整やモデリングが難しい場合があります。GANはモード崩壊（多様性が失われる）や訓練不安定性が知られていますし、拡散モデルも計算コストが高く学習に時間を要します。また、生成データが本当に有用かを評価する指標の設定も簡単ではありません[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=distributional%20fidelity%20compared%20to%20traditional,the%20robustness%20and%20generalizability%20in)（画像ではFIDなどありますが、時系列での妥当性指標は活発に研究中です）。さらに、生成モデルが**訓練データに過度適合すると**既存データのコピーのようなものを生んでしまい、汎化に寄与しない可能性もあります。そのため、生成モデル拡張を使う際は、**生成データの品質評価**（統計的類似度や識別器による見分けテスト）を行い、必要に応じて生成データにフィルタリングや加工を加える工夫も求められます。総じて、生成モデルによる拡張は強力ですが開発コストが高いため、まずは上述の**簡易なランダム変換型拡張**を試し、それでも不足する部分を生成モデルで補う、という段階的なアプローチが現実的でしょう。実際、最新の研究では**基本的な拡張 + 生成データ**の併用が効果的だったと報告されています[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=using%20the%20SmartFallMM%20dataset%20,score%20of)[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=generated%20from%20various%20techniques%2C%20the,score%20of)。

## 参考文献（および情報源）

1. **T. T. Um et al. (2017)**: _“Data Augmentation of Wearable Sensor Data for Parkinson’s Disease Monitoring using CNNs.”_ 19th ACM Int’l Conf. on Multimodal Interaction (ICMI’17). ※小規模な医療データで各種拡張（ジャitter, scaling, rotation, permutation等）を試し、**77.54%→86.88%**への精度向上を達成[arxiv.org](https://arxiv.org/abs/1706.00527#:~:text=,to%2086.88)。特に**回転**拡張の有効性を示す[mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1444915/document.pdf#:~:text=The%20best%20performance%20among%20the,major%20sources%20of%20vari%02ability%20are)。
    
2. **N. Ashfaq et al. (2024)**: _“Identification of Optimal Data Augmentation Techniques for Multimodal Time-Series Sensory Data: A Framework.”_ Information, 15(6), 343. ※HAR分野で**15種類の拡張手法**（本レポートで言及した大部分を網羅）を体系的に比較。**複合動作で5%、単一動作で30%**の精度向上[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=classification,are%20explained%20in%20the%20following)。各手法の定義や効果を詳述[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,represents%20mathematical%20notation%20of%20jittering)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,mathematical%20notation%20of%20time%20warping)[mdpi.com](https://www.mdpi.com/2078-2489/15/6/343#:~:text=,bending%29%20activity)。
    
3. **M. Debnath et al. (2025)**: _“Enhancing Wearable Fall Detection System via Synthetic Data.”_ Sensors, 25(15), 4639. ※加速度センサによる転倒検知で、拡張手法（**ジャitter, 振幅ワープ, 回転**）と**拡散モデル生成データ**・ビデオ由来データを比較。拡散モデルで生成したデータを追加学習することで全指標でベースラインを上回り、**伝統的拡張より高精度**を実現[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=distributional%20fidelity%20compared%20to%20traditional,the%20robustness%20and%20generalizability%20in)。ジャitterや回転の効果も記載[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Jittering%20involves%20adding%20random%20Gaussian,patterns%20in%20the%20time%20series)[mdpi.com](https://www.mdpi.com/1424-8220/25/15/4639#:~:text=Rotation%20simulates%20different%20sensor%20placements,its%20generalization%20to%20diverse%20scenarios)。
    
4. **tsaug (Arundo, 2019)**: _Open-source Python library for time series augmentation._ [arundo.com](https://www.arundo.com/articles/tsaug-an-open-source-python-package-for-time-series-augmentation#:~:text=Tsaug%3A%20An%20Open,learning%20models%20on%20time%20series)[tsaug.readthedocs.io](https://tsaug.readthedocs.io/en/stable/#:~:text=,)※本レポートで紹介した基本的な拡張（Jitter=AddNoise, Crop, TimeWarp, Quantize, Drift, Reverse, Dropout等）を実装。使い易いAPIで**複合的な拡張パイプライン**も構築可能[tsaug.readthedocs.io](https://tsaug.readthedocs.io/en/stable/#:~:text=,probability%2C%20reverse%20the%20sequence)[tsaug.readthedocs.io](https://tsaug.readthedocs.io/en/stable/#:~:text=,)。
    

（※その他、各種論文や資料からの引用は文中に【】付きで示しました。）

* * *

**You:**

これらのaugmentationをrandargumentの枠組みで適応するのはあり？

* * *

**ChatGPT:**

結論：**あり**です。  
RandAugment（RA）は本来画像向けですが、「時系列版RA」をそのまま使える実装があり、**tsai**ライブラリには `RandAugment(tfms, N, M)` が用意されています。任意の時系列用変換（jitter、scaling、time-warp、permutation、crop など）をリストに渡すと、各バッチで**N 個の変換をランダムに選び、強度 M で適用**できます。[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.transforms.html?utm_source=chatgpt.com)

### どう組むか（実践ポイント）

* **実装**：`tsai.data.transforms.RandAugment` に、時系列向けの tfms を渡すだけ。RA の枠組み（N/M）で回せます。元論文の思想どおり、「検索不要で少ないハイパラで強力」なのが利点です。[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.transforms.html?utm_source=chatgpt.com)[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf?utm_source=chatgpt.com)[CVFオープンアクセス](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf?utm_source=chatgpt.com)
    
* **opの候補**（例）
    
    * ノイズ付加（jitter）/ 振幅スケール / 振幅ワープ
        
    * 時間ワープ / 区間 permutation / ランダム crop（時間シフト）
        
    * （IMU限定）**3軸回転**（装着向きの違いを模擬）  
        ※tsai には時系列専用 tfms があり、`RandAugment` の `tfms` に渡せます（ドキュメントに RA のエントリあり）。[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.transforms.html?utm_source=chatgpt.com)
        
* **N/M の目安**：まずは `N=2〜3, M=6〜9` から。大きすぎると波形を壊しやすいので、学習曲線や検証スコアを見て調整。RAは**探索不要**で実務的に扱いやすいのが強みです。[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf?utm_source=chatgpt.com)
    
* **モダリティ注意**：
    
    * **IMU**：回転は**全軸を同一回転で**。センサ間の同期を崩さない。
        
    * **ToF/thermopile**：-1（欠測）や量子化値があるので、ワープ/スケール時は**マスクを尊重**し、欠測値を保持または補間ルールを統一。
        
    * **マルチセンサ**：**全チャネル同じ時間変形**（同じワープ関数・同じ切り出し）を適用して、チャネル間の整合を保つ。
        
* **ラベル不変性ガード**：振幅自体がクラス手掛かりの場合は scaling を弱める／除外するなど、**タスクに応じた op セレクション**を。RA は枠組みなので、**入れるopの吟味**が肝です（画像と同様の設計思想）。[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf?utm_source=chatgpt.com)
    
* **音声系の類例**：音声（スペクトログラム）では **SpecAugment** という「ポリシー型拡張」が標準で、**ポリシー駆動の拡張は時系列でも有効**という示唆になります（時間・周波数マスクやワープをポリシー化）。[arXiv+1](https://arxiv.org/abs/1904.08779?utm_source=chatgpt.com)
    

### 最小コード例（概念）

```python
from tsai.data.transforms import RandAugment
# 例: tsaiの時系列変換（jitter/scale/time-warp 等）を用意しておく
tfms = [TSMagAddNoise(scale_max=0.05),
        TSMagScale(max_gain=0.2),
        TSTimeWarp(max_warp=0.2),
        TSPermute(n_segments=4),
        TSCrop(size=L)]  # Lは学習ウィンドウ長
aug = RandAugment(tfms=tfms, N=2, M=7)  # N個選んで強度Mで適用
# → dataloaderの前処理やcollateで aug を噛ませる
```

（関数名は例。tsai の `RandAugment(tfms, N, M)` が時系列でも動くことが公式ドキュメントに明記されています。）[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.transforms.html?utm_source=chatgpt.com)

### 併用のすすめ

* **CutMix / Mixup は「バッチレベル」で併用**：RAは「サンプル内でのランダム op」、CutMix/Mixup は「サンプル間合成」。役割が違うので相補的です。まず RA を入れて、その上で同一クラス内Mixupや部分CutMixを薄めに。（tsai/一般実装で容易）
    
* **画像化→画像RA**：時系列を GAF/RP などで画像化し、画像用 RA（torchvision/timm 実装）を適用する手もあります。tsai には時系列→画像変換のユーティリティもあります。[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.image.html?utm_source=chatgpt.com)[docs.pytorch.org](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html?utm_source=chatgpt.com)[timm.fast.ai](https://timm.fast.ai/RandAugment?utm_source=chatgpt.com)
    

* * *

**まとめ**：  
**RandAugmentの“枠組み”を時系列で使うのはアリ**です。tsai がそのまま対応しており、N/M だけでポリシー駆動の拡張を回せます。入れる変換の選定（IMU回転や時間ワープ等の“ラベル不変”op中心）と、マルチセンサ一貫性（同一時間変形・欠測処理の統一）だけ気をつければ、**少データでの汎化向上**に効きます。[timeseriesai.github.io](https://timeseriesai.github.io/tsai/data.transforms.html?utm_source=chatgpt.com)[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf?utm_source=chatgpt.com)