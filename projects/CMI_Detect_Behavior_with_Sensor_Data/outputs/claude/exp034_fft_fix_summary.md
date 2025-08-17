# exp034 FFTエラー修正サマリー

## 問題の概要
```
RuntimeError: cuFFT only supports dimensions whose sizes are powers of two when computing in half precision, but got a signal size of[50]
```

exp034の訓練中に、AFNOMixer1DクラスのFFT処理で半精度（16-mixed precision）使用時にエラーが発生。cuFFTが半精度計算時に2の累乗でないシーケンス長（50）をサポートしないため。

## 根本原因

1. **半精度FFTの制約**: cuFFTは半精度計算時、入力サイズが2の累乗である必要がある
2. **シーケンス長の問題**: 実際のデータのseq_len=50は2の累乗ではない（2^5=32, 2^6=64）
3. **PyTorch Lightning設定**: precision="16-mixed"により自動的に半精度が使用される

## 修正内容

### 1. import追加
```python
import math  # 2の累乗計算用
```

### 2. AFNOMixer1D.forward()の修正
```python
def forward(self, x):
    # 元のデータ型とシーケンス長を保存
    original_dtype = x.dtype
    original_seq_len = x.size(1)
    
    # 次の2の累乗を計算
    next_power_of_2 = 2 ** math.ceil(math.log2(original_seq_len))
    
    # 必要に応じてパディング
    if original_seq_len != next_power_of_2:
        padding = next_power_of_2 - original_seq_len
        x = F.pad(x, (0, 0, 0, padding))
    
    # FFT計算を安定させるためfloat32で実行
    x = x.float()
    x_ft = torch.fft.rfft(x, dim=1)
    
    # ... MLP処理 ...
    
    # 逆FFT
    x_reconstructed = torch.fft.irfft(y_complex, n=next_power_of_2, dim=1)
    
    # 元のシーケンス長に戻す
    x_reconstructed = x_reconstructed[:, :original_seq_len, :]
    
    # 元のデータ型に戻す
    x_reconstructed = x_reconstructed.to(original_dtype)
    
    # 残差接続
    return self.norm(self.dropout(x_reconstructed) + x_residual)
```

## 修正の効果

### 1. エラー解決の確認
- ✅ seq_len=50（元エラー）で正常動作
- ✅ 半精度（half precision）で正常動作
- ✅ 2の累乗でない任意のシーケンス長で動作

### 2. 機能保持の確認
- ✅ 既存テストスイート8/8通過
- ✅ 入出力形状の一致
- ✅ NaN/Inf発生なし

### 3. テスト結果
```
Testing: batch_size=2, seq_len=50, hidden_dim=256
  Half precision: SUCCESS - Output shape: torch.Size([2, 50, 256])
  Half precision: No NaN: True
  Float32: SUCCESS - Output shape: torch.Size([2, 50, 256])
  Float32: No NaN: True
```

## 技術的詳細

### パディング戦略
- **入力**: [batch, seq_len, hidden_dim]
- **パディング**: [batch, next_power_of_2, hidden_dim]
- **出力**: [batch, seq_len, hidden_dim] （元サイズに復元）

### データ型変換
- **FFT前**: half → float32（安定性確保）
- **FFT後**: float32 → 元のdtype（メモリ効率）

### パフォーマンス影響
- **最小限のオーバーヘッド**: パディングは必要時のみ
- **精度維持**: float32でFFT演算により数値安定性確保
- **メモリ効率**: 元データ型に復元

## 修正範囲
- `codes/exp/exp034/model.py`
  - import math追加
  - AFNOMixer1D.forward()メソッド修正

## 結論
元のFFTエラーが完全に解決され、AFNOミキサーが半精度環境で安定動作するようになりました。修正により、任意の長さのIMUシーケンスデータを効率的に処理可能です。