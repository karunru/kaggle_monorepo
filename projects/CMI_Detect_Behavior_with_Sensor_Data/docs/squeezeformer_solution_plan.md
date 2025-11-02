# Squeezeformer-Based Solution Plan for CMI Detect Behavior with Sensor Data

## 概要

本ドキュメントでは、CMI - Detect Behavior with Sensor Dataコンペティションに対して、Squeezeformerアーキテクチャを用いた解法方針を詳細に設計します。

## 実装状況（2025-06-28更新）

✅ **完了済み**:
- 全ての設計されたモジュールの実装
- データ前処理パイプライン (src/preprocessor.py)
- Dataset class (src/dataset.py)
- Squeezeformer model (src/model.py)
- Cross-validation strategy (src/cv_strategy.py)
- 評価指標 (src/metrics.py)
- 損失関数 (src/loss.py)
- 訓練スクリプト (exp/exp001/train.py)
- 推論パイプライン (src/inference.py)
- 設定ファイル (exp/exp001/config.yaml)
- テストコード (tests/)

📝 **主要なファイル**:
- `src/preprocessor.py`: マルチモーダルセンサーデータの前処理
- `src/dataset.py`: PyTorchデータセット（データ拡張、マルチタスク学習対応）
- `src/model.py`: Squeezeformerアーキテクチャ（IMU単体ブランチ付き）
- `src/cv_strategy.py`: StratifiedGroupKFold実装（被験者ベース分割）
- `src/metrics.py`: コンペティション用評価指標（Binary F1 + Macro F1の平均）
- `src/loss.py`: マルチタスク損失関数（複数バリエーション）
- `exp/exp001/train.py`: 5-fold CV訓練パイプライン
- `src/inference.py`: 推論パイプライン（TTA、アンサンブル対応）
- `exp/exp001/run_inference.py`: 簡易推論実行スクリプト

🚀 **使用方法**:
```bash
# 訓練実行
cd exp/exp001
python train.py

# 推論実行（訓練後）
python run_inference.py
```

## データ理解

### データセット構造
- **総行数**: 574,946行
- **総列数**: 341列
- **ユニークシーケンス数**: 8,152
- **Target vs Non-Target**: 344,058 vs 230,887（約60:40の比率）
- **ジェスチャー種類**: 18種類（BFRB様: 8種類、non-BFRB様: 10種類）

### センサー構成
1. **IMU**: 7列（acc_x/y/z, rot_w/x/y/z）
2. **Thermopile**: 5列（thm_1-5）
3. **Time-of-Flight**: 320列（tof_1-5 × 64ピクセル）

### データの特徴
- 各シーケンスは`Transition`と`Gesture`フェーズで構成
- テストセットの半分はIMUデータのみ（thermopile/ToFはnull）
- 被験者（subject）でグループ化されたクロスバリデーションが必要

## 1. データ前処理戦略

### 1.1 基本前処理
```python
class DataPreprocessor:
    def __init__(self):
        self.imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.thm_cols = [f'thm_{i}' for i in range(1, 6)]
        self.tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
        
    def preprocess_sequence(self, df):
        # 1. 欠損値処理
        df = self.handle_missing_values(df)
        
        # 2. 正規化
        df = self.normalize_features(df)
        
        # 3. フィルタリング
        df = self.apply_filters(df)
        
        # 4. 特徴量エンジニアリング
        df = self.feature_engineering(df)
        
        return df
        
    def handle_missing_values(self, df):
        # IMUデータの線形補間
        for col in self.imu_cols:
            df[col] = df[col].interpolate(method='linear')
            
        # Thermopileデータの前方充填 + 平均値補間
        for col in self.thm_cols:
            df[col] = df[col].fillna(method='ffill').fillna(df[col].mean())
            
        # ToFデータの-1を0に変換、その他は前方充填
        for col in self.tof_cols:
            df[col] = df[col].replace(-1, 0)
            df[col] = df[col].fillna(method='ffill').fillna(0)
            
        return df
        
    def normalize_features(self, df):
        # IMU: 標準化（Z-score）
        for col in self.imu_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            
        # Thermopile: Min-Max正規化（温度は範囲が限定的）
        for col in self.thm_cols:
            min_val, max_val = df[col].min(), df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                
        # ToF: Min-Max正規化（0-254の範囲）
        for col in self.tof_cols:
            df[col] = df[col] / 254.0
            
        return df
        
    def apply_filters(self, df):
        # IMUデータにローパスフィルタを適用（ノイズ除去）
        from scipy.signal import butter, filtfilt
        
        def lowpass_filter(data, cutoff=10, fs=50, order=4):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)
            
        for col in self.imu_cols:
            df[col] = lowpass_filter(df[col].values)
            
        return df
        
    def feature_engineering(self, df):
        # IMU特徴量の追加
        df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['rot_magnitude'] = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
        
        # Thermopileの統計量
        thm_data = df[self.thm_cols].values
        df['thm_mean'] = thm_data.mean(axis=1)
        df['thm_std'] = thm_data.std(axis=1)
        df['thm_max'] = thm_data.max(axis=1)
        df['thm_min'] = thm_data.min(axis=1)
        
        # ToFの統計量（8x8グリッドごと）
        for i in range(1, 6):
            tof_grid = df[[f'tof_{i}_v{j}' for j in range(64)]].values.reshape(-1, 8, 8)
            df[f'tof_{i}_center'] = tof_grid[:, 3:5, 3:5].mean(axis=(1, 2))  # 中心部
            df[f'tof_{i}_edge'] = np.concatenate([
                tof_grid[:, 0, :].flatten(),
                tof_grid[:, -1, :].flatten(),
                tof_grid[:, :, 0].flatten(),
                tof_grid[:, :, -1].flatten()
            ]).reshape(len(df), -1).mean(axis=1)  # エッジ部
            
        return df
```

### 1.2 シーケンス処理
```python
def process_sequences(df):
    sequences = []
    labels = []
    subjects = []
    
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].copy()
        
        # フェーズ別処理
        transition_data = seq_data[seq_data['phase'] == 'Transition']
        gesture_data = seq_data[seq_data['phase'] == 'Gesture']
        
        # 固定長にリサンプリング（Transition: 100steps, Gesture: 100steps）
        transition_resampled = resample_sequence(transition_data, target_length=100)
        gesture_resampled = resample_sequence(gesture_data, target_length=100)
        
        # 結合して1つのシーケンス（200 timesteps）
        full_sequence = np.concatenate([transition_resampled, gesture_resampled], axis=0)
        
        sequences.append(full_sequence)
        labels.append(seq_data['gesture'].iloc[0])
        subjects.append(seq_data['subject'].iloc[0])
        
    return np.array(sequences), np.array(labels), np.array(subjects)

def resample_sequence(data, target_length):
    from scipy.interpolate import interp1d
    
    if len(data) == 0:
        return np.zeros((target_length, data.shape[1]))
        
    current_length = len(data)
    if current_length == target_length:
        return data.values
        
    # 線形補間でリサンプリング
    old_indices = np.linspace(0, current_length - 1, current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    
    resampled_data = np.zeros((target_length, data.shape[1]))
    for i, col in enumerate(data.columns):
        if col not in ['sequence_id', 'sequence_counter', 'subject', 'orientation', 'behavior', 'phase', 'gesture']:
            f = interp1d(old_indices, data[col].values, kind='linear')
            resampled_data[:, i] = f(new_indices)
            
    return resampled_data
```

## 2. Dataset Class設計

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class CMISensorDataset(Dataset):
    def __init__(self, sequences, labels, subjects, label_encoder, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.subjects = subjects
        self.label_encoder = label_encoder
        self.augment = augment
        
        # ラベルエンコーディング
        self.encoded_labels = self.label_encoder.transform(labels)
        
        # マルチタスク用のバイナリラベル（Target vs Non-Target）
        target_gestures = [
            'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
            'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
            'Neck - scratch', 'Cheek - pinch skin'
        ]
        self.binary_labels = np.array([1 if label in target_gestures else 0 for label in labels])
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        multiclass_label = self.encoded_labels[idx]
        binary_label = self.binary_labels[idx]
        subject = self.subjects[idx]
        
        if self.augment:
            sequence = self.apply_augmentation(sequence)
            
        # Tensor変換
        sequence = torch.FloatTensor(sequence)  # [seq_len, features]
        sequence = sequence.transpose(0, 1)     # [features, seq_len] for Squeezeformer
        
        return {
            'sequence': sequence,
            'multiclass_label': torch.LongTensor([multiclass_label]),
            'binary_label': torch.LongTensor([binary_label]),
            'subject': subject
        }
        
    def apply_augmentation(self, sequence):
        augmented = sequence.copy()
        
        # 1. ガウシアンノイズ
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, sequence.shape)
            augmented += noise
            
        # 2. 時間軸スケーリング
        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.9, 1.1)
            new_length = int(len(sequence) * scale_factor)
            indices = np.linspace(0, len(sequence)-1, new_length)
            
            # 各特徴量に対して補間
            from scipy.interpolate import interp1d
            resampled = np.zeros((len(sequence), sequence.shape[1]))
            for i in range(sequence.shape[1]):
                if new_length > 1:
                    f = interp1d(np.arange(len(sequence)), sequence[:, i], kind='linear')
                    resampled[:, i] = np.interp(np.linspace(0, len(sequence)-1, len(sequence)), 
                                              np.linspace(0, len(sequence)-1, new_length),
                                              f(indices))
                else:
                    resampled[:, i] = sequence[:, i]
            augmented = resampled
            
        # 3. 部分マスキング（ToFセンサーのみ）
        if np.random.random() < 0.2:
            tof_start_idx = 12  # ToF features start after IMU and Thermopile
            tof_end_idx = tof_start_idx + 320
            mask_length = np.random.randint(5, 20)
            mask_start = np.random.randint(0, len(sequence) - mask_length)
            augmented[mask_start:mask_start+mask_length, tof_start_idx:tof_end_idx] *= 0.1
            
        return augmented

def create_dataloaders(train_sequences, train_labels, train_subjects, 
                      val_sequences, val_labels, val_subjects,
                      label_encoder, batch_size=32):
    
    train_dataset = CMISensorDataset(train_sequences, train_labels, train_subjects, 
                                   label_encoder, augment=True)
    val_dataset = CMISensorDataset(val_sequences, val_labels, val_subjects, 
                                 label_encoder, augment=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader
```

## 3. Squeezeformer Model Class設計

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Convolution module (Squeezeformer's key component)
        self.conv_module = ConvolutionModule(d_model, kernel_size=31)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Multi-Head Self-Attention
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = residual + self.dropout(attn_output)
        
        # Convolution Module
        residual = x
        x = self.ln2(x)
        x = self.conv_module(x)
        x = residual + self.dropout(x)
        
        # Feed-Forward Network
        residual = x
        x = self.ln3(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        
        # GLU (Gated Linear Unit)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Pointwise conv 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv + batch norm + activation
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return x

class CMISqueezeformer(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=8, n_heads=8, 
                 d_ff=1024, num_classes=18, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Squeezeformer blocks
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.multiclass_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # IMU-only branch for robustness
        self.imu_branch = nn.Sequential(
            nn.Linear(7, d_model // 4),  # IMU features only
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
    def forward(self, x, imu_only=False):
        # x shape: [batch, features, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, features]
        
        if imu_only:
            # Use only IMU features (first 7 features)
            imu_features = x[:, :, :7]
            imu_emb = self.imu_branch(imu_features)
            
            # Simple classification for IMU-only case
            imu_pooled = self.global_pool(imu_emb.transpose(1, 2)).squeeze(-1)
            multiclass_logits = self.multiclass_head(imu_pooled)
            binary_logits = self.binary_head(imu_pooled)
            
            return multiclass_logits, binary_logits
        
        # Full model with all sensors
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Pass through Squeezeformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # [batch, d_model]
        
        # Classification
        multiclass_logits = self.multiclass_head(x)
        binary_logits = self.binary_head(x)
        
        return multiclass_logits, binary_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
```

## 4. StratifiedGroupKFold CV設計

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CVStrategy:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, 
                                        random_state=random_state)
        
    def create_folds(self, labels, subjects):
        \"\"\"
        subjectをgroupとしたStratifiedGroupKFoldを実行
        
        Args:
            labels: ジェスチャーラベル
            subjects: 被験者ID
            
        Returns:
            folds: [(train_idx, val_idx), ...] のリスト
        \"\"\"
        # バイナリラベルでstratify（Target vs Non-Target）
        target_gestures = [
            'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
            'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
            'Neck - scratch', 'Cheek - pinch skin'
        ]
        binary_labels = np.array([1 if label in target_gestures else 0 for label in labels])
        
        folds = []
        for train_idx, val_idx in self.sgkf.split(labels, binary_labels, subjects):
            folds.append((train_idx, val_idx))
            
        return folds
        
    def validate_folds(self, folds, labels, subjects):
        \"\"\"
        Fold分割が正しく行われているかを検証
        \"\"\"
        for i, (train_idx, val_idx) in enumerate(folds):
            train_subjects = set(subjects[train_idx])
            val_subjects = set(subjects[val_idx])
            
            # 被験者の重複がないことを確認
            assert len(train_subjects & val_subjects) == 0, f\"Fold {i}: Subject overlap detected!\"
            
            print(f\"Fold {i+1}:\")
            print(f\"  Train: {len(train_idx)} samples, {len(train_subjects)} subjects\")
            print(f\"  Val: {len(val_idx)} samples, {len(val_subjects)} subjects\")
            
            # ラベル分布の確認
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            
            target_gestures = [
                'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
                'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
                'Neck - scratch', 'Cheek - pinch skin'
            ]
            
            train_target_ratio = sum(label in target_gestures for label in train_labels) / len(train_labels)
            val_target_ratio = sum(label in target_gestures for label in val_labels) / len(val_labels)
            
            print(f\"  Train Target ratio: {train_target_ratio:.3f}\")
            print(f\"  Val Target ratio: {val_target_ratio:.3f}\")
            print()

def run_cross_validation(sequences, labels, subjects, model_config, training_config):
    \"\"\"
    クロスバリデーションの実行
    \"\"\"
    cv_strategy = CVStrategy(n_splits=5)
    folds = cv_strategy.create_folds(labels, subjects)
    cv_strategy.validate_folds(folds, labels, subjects)
    
    # ラベルエンコーダー
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    fold_scores = []
    models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f\"Training Fold {fold_idx + 1}/{len(folds)}\")
        
        # データ分割
        train_sequences = sequences[train_idx]
        train_labels = labels[train_idx]
        train_subjects = subjects[train_idx]
        
        val_sequences = sequences[val_idx]
        val_labels = labels[val_idx]
        val_subjects = subjects[val_idx]
        
        # データローダー作成
        train_loader, val_loader = create_dataloaders(
            train_sequences, train_labels, train_subjects,
            val_sequences, val_labels, val_subjects,
            label_encoder, batch_size=training_config['batch_size']
        )
        
        # モデル初期化
        model = CMISqueezeformer(**model_config)
        
        # 訓練実行
        trained_model, metrics = train_model(
            model, train_loader, val_loader, training_config
        )
        
        models.append(trained_model)
        fold_scores.append(metrics)
        
        print(f\"Fold {fold_idx + 1} - Val Score: {metrics['val_score']:.4f}\")
        
    # CV結果の集計
    avg_score = np.mean([score['val_score'] for score in fold_scores])
    std_score = np.std([score['val_score'] for score in fold_scores])
    
    print(f\"CV Score: {avg_score:.4f} ± {std_score:.4f}\")
    
    return models, fold_scores, avg_score
```

## 5. 訓練戦略

### 5.1 損失関数
```python
class CMILoss(nn.Module):
    def __init__(self, alpha=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha  # バイナリとマルチクラス損失の重み
        self.multiclass_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.binary_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, multiclass_logits, binary_logits, multiclass_targets, binary_targets):
        mc_loss = self.multiclass_loss(multiclass_logits, multiclass_targets)
        binary_loss = self.binary_loss(binary_logits.squeeze(), binary_targets.float())
        
        total_loss = (1 - self.alpha) * mc_loss + self.alpha * binary_loss
        
        return total_loss, mc_loss, binary_loss
```

### 5.2 評価指標
```python
def calculate_competition_metric(multiclass_preds, binary_preds, multiclass_true, binary_true, label_encoder):
    # Binary F1
    binary_f1 = f1_score(binary_true, binary_preds, average='binary')
    
    # Macro F1 (non-targetを単一クラスとして扱う)
    # Target gestures
    target_gestures = [
        'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
        'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
        'Neck - scratch', 'Cheek - pinch skin'
    ]
    
    # Convert predictions to gesture names
    pred_gestures = label_encoder.inverse_transform(multiclass_preds)
    true_gestures = label_encoder.inverse_transform(multiclass_true)
    
    # Create modified labels for macro F1
    def convert_to_macro_labels(gestures):
        return [gesture if gesture in target_gestures else 'non_target' for gesture in gestures]
    
    macro_pred = convert_to_macro_labels(pred_gestures)
    macro_true = convert_to_macro_labels(true_gestures)
    
    macro_f1 = f1_score(macro_true, macro_pred, average='macro')
    
    # Final score
    final_score = (binary_f1 + macro_f1) / 2
    
    return final_score, binary_f1, macro_f1
```

## 6. 実装上の考慮事項

### 6.1 メモリ最適化
- グラディエントチェックポイントの使用
- Mixed Precision Training (AMP)
- バッチサイズの動的調整

### 6.2 ハードウェア対応
- IMU-onlyモードでの推論対応
- CPU推論の最適化
- 推論時間の制約対応

### 6.3 アンサンブル戦略
- 5-fold CVモデルのアンサンブル
- IMU-onlyモデルとフルモデルの適応的切り替え
- Test-time Augmentation (TTA)

## 7. 実験計画

### Phase 1: ベースライン構築
1. 基本的なSqueezeformerモデルの実装
2. データ前処理パイプラインの検証
3. CV戦略の確認

### Phase 2: モデル最適化
1. ハイパーパラメータチューニング
2. データ拡張の効果検証
3. アーキテクチャの改良

### Phase 3: アンサンブル構築
1. 複数モデルの訓練
2. アンサンブル戦略の最適化
3. 最終提出の準備

## 8. 期待される性能

- **ベースライン**: CV Score 0.75+
- **最適化後**: CV Score 0.80+
- **アンサンブル**: CV Score 0.82+

この包括的な設計により、センサーデータの時系列パターンを効果的に学習し、BFRBとnon-BFRBジェスチャーを高精度で分類することを目指します。