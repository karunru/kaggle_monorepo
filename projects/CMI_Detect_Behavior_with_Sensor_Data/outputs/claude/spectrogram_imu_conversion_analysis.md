# IMUデータのスペクトログラム画像変換分析レポート

## 概要
本レポートでは、Kaggleノートブック「idea-with-spectrogram-cnn2d」におけるIMUセンサーデータの画像変換手法について詳細に分析します。特に、異なるsequence lengthを持つデータを同じサイズの画像に変換する手法に焦点を当てます。

## 1. IMUデータの構成

分析対象のIMUデータは以下の7つのチャンネルから構成されています：

- **加速度センサー（3軸）**: `acc_x`, `acc_y`, `acc_z`
- **回転センサー（クォータニオン）**: `rot_w`, `rot_x`, `rot_y`, `rot_z`

## 2. データ処理パイプライン

### 2.1 前処理とSequence Length正規化

#### 欠損値処理（`_prepare_sensor_data_raw`メソッド）
1. 各センサーチャンネルのデータを`float64`配列に変換
2. 欠損値（NaN）の処理：
   - 全てNaNの列：0で埋める
   - 部分的にNaNを含む列：線形補間（`interpolate`）→前方埋め（`ffill`）→後方埋め（`bfill`）→残りを0で埋める

#### Sequence Lengthの統一化（`_pad_or_truncate_final`メソッド）
異なる長さのsequenceを固定長（`seq_len=100`）に調整する手法：

```python
def _pad_or_truncate_final(self, data, target_len):
    current_len = data.shape[0]
    
    if current_len > target_len:
        # 長い場合：最後の100ステップを取る（末尾を保持）
        truncated_data = data[-target_len:]
        return truncated_data
    elif current_len < target_len:
        # 短い場合：前方にゼロパディング
        padding_rows = np.zeros((target_len - current_len, data.shape[1]))
        padded_data = np.concatenate([padding_rows, data], axis=0)
        return padded_data
    
    return data
```

**重要な特徴**：
- **末尾保持戦略**：データが長い場合、最新の情報を優先して末尾100ステップを保持
- **前方パディング**：データが短い場合、先頭にゼロを追加（時系列の順序を保持）

### 2.2 スペクトログラム変換

#### SpecFeatureExtractorクラス
時系列データを周波数領域の画像表現に変換：

```python
class SpecFeatureExtractor(nn.Module):
    def __init__(self, in_channels=7, height=12, hop_length=4, 
                 win_length=None, out_size=224):
        # n_fft = 2 * height - 1 = 2 * 12 - 1 = 23
        self.feature_extractor = nn.Sequential(
            T.Spectrogram(n_fft=23, hop_length=4, win_length=None),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.pool = nn.AdaptiveAvgPool2d((None, out_size))
```

#### 変換プロセスの詳細

1. **FFT（高速フーリエ変換）**
   - `n_fft=23`：FFTウィンドウサイズ
   - `hop_length=4`：ウィンドウの移動幅（オーバーラップ75%）
   - 入力：(batch, 7, 100) → 出力：(batch, 7, 12, time_frames)

2. **振幅のデシベル変換**
   - `AmplitudeToDB(top_db=80)`：振幅をデシベルスケールに変換
   - ダイナミックレンジを80dBに制限

3. **正規化**
   - 各チャンネルごとに最小値と最大値で正規化（0-1の範囲）
   - 画像として扱いやすいスケールに調整

4. **サイズ調整**
   - `AdaptiveAvgPool2d((None, out_size))`：時間軸方向を224に調整
   - 最終出力：(batch, 7, 12, 224)

## 3. 画像化のメリット

### 3.1 固定サイズへの変換
- **問題**：入力sequenceの長さが可変（数十〜数千ステップ）
- **解決**：
  1. Sequence lengthを100に統一（パディング/トランケート）
  2. スペクトログラム変換で周波数領域に変換
  3. Adaptive Poolingで画像サイズを224×12に固定

### 3.2 特徴抽出の効率化
- **時間領域**の局所的パターンを**周波数領域**の空間的パターンとして表現
- CNN（畳み込みニューラルネットワーク）による効率的な特徴抽出が可能
- 事前学習済みの画像認識モデル（ResNet、CAFormer等）の活用

### 3.3 周期性の捕捉
- IMUデータに含まれる周期的な動作パターン（歩行、ジェスチャー等）を周波数成分として直接表現
- 異なる周波数帯域の特徴を並列に学習可能

## 4. 実装上の工夫

### 4.1 モデル選択の柔軟性
- TIMMライブラリを使用し、様々な事前学習済みモデルを簡単に切り替え可能
- コード例では`caformer_s18.sail_in22k`を使用（Vision Transformerベースのモデル）

### 4.2 データローダーの最適化
- `pin_memory=True`：GPUメモリへの高速転送
- `persistent_workers=True`：ワーカープロセスの再利用
- `prefetch_factor=4`：データの先読み

## 5. まとめ

このアプローチの主要な特徴：

1. **Sequence Length正規化**：
   - トランケート（末尾100ステップ保持）またはパディング（前方にゼロ追加）
   - 時系列の最新情報を優先的に保持

2. **スペクトログラム変換**：
   - 時系列データを時間-周波数表現に変換
   - 7チャンネル×12周波数ビン×可変時間フレーム

3. **画像サイズ統一**：
   - Adaptive Average Poolingで224×12の固定サイズに調整
   - CNNベースの画像認識モデルの適用を可能に

この手法により、可変長のIMU時系列データを固定サイズの画像表現に効果的に変換し、最新の画像認識技術を活用した分類が可能となっています。