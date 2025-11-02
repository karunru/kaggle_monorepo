# exp034実装完了サマリー

## 概要
exp031をベースに、BiGRU+AttentionメカニズムをAFNO (Adaptive Fourier Neural Operator)ミキサーに置き換えるexp034を実装しました。

## 実装内容

### 1. ディレクトリ構成
```
codes/exp/exp034/
├── __init__.py
├── config.py           # 設定ファイル（exp034用に更新）
├── dataset.py          # データセット（変更なし）
├── human_normalization.py # ヒューマンノーマライゼーション（変更なし）
├── inference.py        # 推論（変更なし）
├── losses.py           # 損失関数（変更なし）
├── model.py            # メインモデル（AFNOMixer1D追加、IMUOnlyLSTM修正）
├── train.py            # 訓練（変更なし）
└── test_exp034.py      # 新規テストコード
```

### 2. 新規追加クラス: AFNOMixer1D

#### 特徴
- **FFTベースのトークンミキシング**: seq次元にrFFT→周波数領域でMLP→irFFT
- **計算効率**: O(T log T)でSelf-Attentionの代替
- **長距離依存**: 全シーケンス情報を効率的に混合
- **残差接続**: LayerNormと組み合わせた安定した学習

#### 実装詳細
```python
class AFNOMixer1D(nn.Module):
    def __init__(self, hidden_dim, ff_mult=2, dropout=0.1):
        # 周波数領域でのMLP（実部・虚部分離処理）
        self.ff1 = nn.Linear(hidden_dim * 2, hidden_dim * ff_mult, bias=False)
        self.ff2 = nn.Linear(hidden_dim * ff_mult, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):  # [batch, seq_len, hidden_dim]
        # FFT → 実部・虚部分離 → MLP → 復元 → 残差接続
        x_ft = torch.fft.rfft(x, dim=1)
        # ... MLP処理 ...
        output = torch.fft.irfft(y_complex, n=x.size(1), dim=1)
        return self.norm(self.dropout(output) + x)
```

### 3. IMUOnlyLSTMクラスの修正

#### 削除された要素
- `self.bigru`: BiGRU層
- `self.gru_dropout`: GRUドロップアウト
- `self.attention`: AttentionLayer

#### 追加された要素
- `self.token_proj`: 128→256次元への投影層
- `self.afno1`, `self.afno2`: AFNOMixer1D層（2層スタック）
- `self.seq_pool`: AdaptiveAvgPool1dによるシーケンス集約

#### アーキテクチャ変更
```python
# 旧: BiGRU + Attention
gru_out, _ = self.bigru(merged)
attended = self.attention(gru_out)

# 新: AFNO mixer
h = self.token_proj(merged)  # [B, T, 128] → [B, T, 256]
h = self.afno1(h)           # 1層目のAFNOミキサ
h = self.afno2(h)           # 2層目のAFNOミキサ
attended = self.seq_pool(h.transpose(1,2)).squeeze(-1)  # [B, 256]
```

### 4. 設定ファイル更新（config.py）

#### 主要変更項目
- `EXP_NUM`: "exp031" → "exp034"
- `ExperimentConfig.description`: AFNOの説明に更新
- `ExperimentConfig.tags`: ["bigru", "attention"] → ["afno", "fourier_neural_operator", "fft_mixer"]
- `LoggingConfig.wandb_tags`: 同様の変更

### 5. テストコード（test_exp034.py）

#### テスト項目
- **AFNOMixer1D**:
  - 基本的な前向き計算
  - 異なるサイズでの動作
  - 残差接続の確認
- **IMUOnlyLSTM**:
  - Demographics有無での前向き計算
  - AFNOコンポーネントの存在確認
  - 異なる入力次元での動作
- **統合テスト**: 全体的な動作確認

#### テスト結果
```
8 passed in 0.28s
```
すべてのテストが成功し、モデルが正常に動作することを確認。

## 技術的優位性

### 1. 計算効率の向上
- **BiGRU**: O(T)の逐次計算
- **Self-Attention**: O(T²)の計算量
- **AFNO**: O(T log T)のFFTベース並列計算

### 2. 長距離依存の効率的な捕捉
- FFTによりグローバルな時系列パターンを効率的に処理
- 周波数領域での情報混合により長距離相関を捉える

### 3. 解像度非依存性
- FFTベースのため、訓練時と推論時のseq_len変動に対する堅牢性
- 異なる長さのIMUシーケンスに柔軟に対応

### 4. メモリ効率
- Self-Attentionと比較してメモリ使用量が軽量
- 長いシーケンスでのスケーラビリティが向上

## 期待される効果

1. **学習速度向上**: 並列化可能なFFT演算による高速化
2. **精度向上**: 長距離依存の効果的な捕捉によるIMUパターン認識の改善
3. **安定性向上**: 勾配流の改善による学習の安定化
4. **汎用性**: 可変長シーケンスへの対応力向上

## 実装完了状況

✅ exp031のexp034へのコピー  
✅ AFNOMixer1Dクラスの実装  
✅ IMUOnlyLSTMクラスのBiGRU+Attention置換  
✅ config.pyの必須項目更新  
✅ 静的解析（ruff format/lint）の通過  
✅ テストコードの作成・実行（8/8テスト成功）  

exp034の実装が完了し、AFNO技術によるBiGRU+Attentionの効率的な置換が成功しました。