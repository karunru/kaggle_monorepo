**You:**

Fourier neural operatorについて調べて教えて

* * *

**ChatGPT:**

![https://www.researchgate.net/publication/380775248/figure/fig2/AS%3A11431281246708744%401716467818626/Schematic-diagram-of-a-Fourier-neural-operator-FNO-The-networks-P-and-Q-respectively.png](https://tse4.mm.bing.net/th/id/OIP.hZUXjJL6e7zp_2nfsxaWbAHaDh?w=474&h=474&c=7&p=0)

こちらは、Fourier Neural Operator (FNO) の基本アーキテクチャを示す図です。入力関数 $f$ を高次元に「リフト」し、複数のフォーリエ層（Fourier layers）を経て変換処理を行い、最後に投影ネットワークで所望の出力関数へ戻します。各フォーリエ層は、Fourier変換→スペクトル領域での線形操作→逆Fourier変換という流れで構成されています [GitHub+15ResearchGate+15arXiv+15](https://www.researchgate.net/figure/Schematic-diagram-of-a-Fourier-neural-operator-FNO-The-networks-P-and-Q-respectively_fig2_380775248?utm_source=chatgpt.com)。

* * *

## Fourier Neural Operator (FNO)とは？

### 概要と背景

FNOは、偏微分方程式（PDE）を解くための関数空間間の写像を直接学習する「ニューラルオペレーター」の一種です。従来のニューラルネットワークが有限次元空間での写像を扱うのに対し、ニューラルオペレーターは**無限次元の関数空間**間の写像を学習できるのが特徴です [ウィキペディア](https://en.wikipedia.org/wiki/Neural_operators?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)。

FNOでは、畳み込みカーネルをFourier空間で直接パラメータ化し、高い表現力と効率を持つ構造を実現。FFTを利用することで高速な演算が可能になります [ウィキペディア+1](https://en.wikipedia.org/wiki/Neural_operators?utm_source=chatgpt.com)。

* * *

## 主な利点と応用

### 高速・高精度なPDE解法

FNOは、**数値計算に比べ数桁速く**予測を行え、**精度も優れている**ことが実証されています。ベルガース方程式やDarcy流、Navier–Stokes方程式など、複数のベンチマークで性能の高さが報告されています [arXiv+1](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)。

**ゼロショット超解像**（訓練時より高い解像度での評価）が可能なのも大きなメリットです [zongyi-li.github.io](https://zongyi-li.github.io/blog/2020/fourier-pde/?utm_source=chatgpt.com)[MathWorks](https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html?utm_source=chatgpt.com)。

### スペクトルバイアスと改良手法：SpecBoost

分析によれば、FNOは低周波成分（スムースな成分）を学習するのに強い一方で、高周波成分の学習が苦手という**スペクトルバイアス**があります。この弱点を補うため、複数のFNOモデルを組み合わせる**SpecBoost**という手法が提案されていて、高周波情報の復元性能が大幅に改善されています [arXiv](https://arxiv.org/html/2404.07200v1?utm_source=chatgpt.com)。

### 様々な応用分野

* **気象予測**：FourcastNetではFNOを用いた高解像度気象モデルを構築 [arXiv+1](https://arxiv.org/html/2404.07200v1?utm_source=chatgpt.com)。
    
* **化学反応**：硬い化学反応の学習にFNOを応用する試みも報告されています [サイエンスダイレクト](https://www.sciencedirect.com/science/article/abs/pii/S001021802400556X?utm_source=chatgpt.com)。
    
* **地形変化の高速予測**：侵食や地形変化などをFNOで高速に予測する研究も進んでいます [egusphere.copernicus.org](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-307/?utm_source=chatgpt.com)。
    
* **多相流・境界インターフェース**：液–蒸気界面などの複雑な流体界面の予測にもFNOが活用可能で、高速制御への応用が期待されています [ResearchGate+15arXiv+15arXiv+15](https://arxiv.org/html/2505.13463v1?utm_source=chatgpt.com)。
    

* * *

## FNOのバリエーションと進化

| モデル名 | 特徴 |
| --- | --- |
| **IFNO (Implicit FNO)** | 展開を浅く抑え、深層学習での勾配消失を回避しつつパラメータ数削減を実現する反復構造 [サイエンスダイレクト+15ResearchGate+15MathWorks+15](https://www.researchgate.net/figure/The-architecture-of-implicit-Fourier-neural-operator-IFNO_fig1_370841747?utm_source=chatgpt.com) |
| **UFNO (U-Net enhanced FNO)** | U-Net構造を組み込んだFNOで、3D乱流の長期予測に高精度・安定性を実現 [Physical Review Link](https://link.aps.org/doi/10.1103/PhysRevFluids.9.084604?utm_source=chatgpt.com) |
| **IUFNO (Implicit U-FNO)** | IFNOとUFNOを融合した構造で、壁境界付き乱流の長期ダイナミクスにおいて最強の性能を示す [Physical Review Link](https://link.aps.org/doi/10.1103/PhysRevFluids.9.084604?utm_source=chatgpt.com) |
| **Geo-FNO** | 不規則形状・非一様格子への適用のため、ジオメトリを学習し潜在空間でFFT適用 [arXiv](https://arxiv.org/abs/2207.05209?utm_source=chatgpt.com) |
| **DAFNO (Domain Agnostic FNO)** | 幾何形状変化やトポロジーの変動に対応するため、領域判断関数を組み込んだ汎用アーキテクチャ [OpenReview](https://openreview.net/forum?id=ubap5FKbJs&noteId=UigXltwYGV&utm_source=chatgpt.com) |
| **F-FNO (Factorized FNO)** | スペクトル層や残差接続、より深い学習構造で多様なPDEに対応。誤差大幅低減、高速化を実現 [arXiv](https://arxiv.org/abs/2111.13802?utm_source=chatgpt.com) |
| **SC-FNO (Sensitivity-Constrained FNO)** | 感度解析や逆問題への適応に向けて感度項制約を加えた手法で、高精度かつデータ・学習時間を抑えつつ性能向上 [arXiv](https://arxiv.org/abs/2505.08740?utm_source=chatgpt.com) |
| **AM-FNO (Amortized FNO)** | 高周波成分を取り込むためのパラメータ削減的手法。KANや埋め込み MLP により30%ほど性能向上 [OpenReview](https://openreview.net/forum?id=a6em980M9x&utm_source=chatgpt.com) |

* * *

## 参考まとめ

* **基本的な利点**：PDEの解写像を高速かつ高精度に学習でき、解像度依存性が低い（ゼロショット超解像可能） [arXiv+4arXiv+4MathWorks+4](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)。
    
* **課題**：低周波へのバイアスがあり、高周波情報の再現は単体では弱い → SpecBoostなどで補完 [arXiv](https://arxiv.org/html/2404.07200v1?utm_source=chatgpt.com)。
    
* **応用範囲**：気象予測・乱流シミュレーション・多相流・地形変化予測など幅広い領域で成功事例あり。
    
* **進化的発展**：構造強化や幾何対応、高周波強化、逆問題対応といった多様な改良が行われ、実用性・汎用性が向上中。
    

* * *

必要であれば、特定応用への適用例や理論的な詳細（例えば損失関数の工夫や学習手法など）も解説できます。気になる分野・使いたい目的があれば、ぜひ教えてください！

* * *

**You:**

python

class DemographicsEmbedding(nn.Module):
    """Demographics特徴量の埋め込み層."""

    def __init__(
        self,
        categorical_features: list[str],
        numerical_features: list[str],
        categorical_embedding_dims: dict[str, int],
        embedding_dim: int = 16,
        dropout: float = 0.1,
        # 設定値ベースのスケーリングパラメータ
        age_min: float = 8.0,
        age_max: float = 60.0,
        height_min: float = 130.0,
        height_max: float = 195.0,
        shoulder_to_wrist_min: float = 35.0,
        shoulder_to_wrist_max: float = 75.0,
        elbow_to_wrist_min: float = 15.0,
        elbow_to_wrist_max: float = 50.0,
    ):
        """
        初期化.

        Args:
            categorical_features: カテゴリカル特徴量のリスト
            numerical_features: 数値特徴量のリスト
            categorical_embedding_dims: カテゴリカル特徴量の埋め込み次元
            embedding_dim: 最終埋め込み次元
            dropout: ドロップアウト率
            age_min: 年齢の最小値（設定値ベース）
            age_max: 年齢の最大値（設定値ベース）
            height_min: 身長の最小値（設定値ベース）
            height_max: 身長の最大値（設定値ベース）
            shoulder_to_wrist_min: 肩-手首距離の最小値（設定値ベース）
            shoulder_to_wrist_max: 肩-手首距離の最大値（設定値ベース）
            elbow_to_wrist_min: 肘-手首距離の最小値（設定値ベース）
            elbow_to_wrist_max: 肘-手首距離の最大値（設定値ベース）
        """
        super().__init__()

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        # カテゴリカル特徴量用の埋め込み層
        self.categorical_embeddings = nn.ModuleDict()
        total_categorical_dim = 0

        for feature in categorical_features:
            # カテゴリ数は2（0/1の二値）と仮定
            num_categories = 2
            embed_dim = categorical_embedding_dims.get(feature, 2)
            self.categorical_embeddings[feature] = nn.Embedding(num_categories, embed_dim)
            total_categorical_dim += embed_dim

        # 数値特徴量の次元
        numerical_dim = len(numerical_features)

        # 結合後の特徴量次元
        combined_dim = total_categorical_dim + numerical_dim

        # 最終埋め込み層（結合された特徴量を所定の次元に変換）
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        # スケーリング用のパラメータ（設定値ベース、推論時の一貫性確保）
        self.register_buffer("age_min", torch.tensor(age_min))
        self.register_buffer("age_max", torch.tensor(age_max))
        self.register_buffer("height_min", torch.tensor(height_min))
        self.register_buffer("height_max", torch.tensor(height_max))
        self.register_buffer("shoulder_to_wrist_min", torch.tensor(shoulder_to_wrist_min))
        self.register_buffer("shoulder_to_wrist_max", torch.tensor(shoulder_to_wrist_max))
        self.register_buffer("elbow_to_wrist_min", torch.tensor(elbow_to_wrist_min))
        self.register_buffer("elbow_to_wrist_max", torch.tensor(elbow_to_wrist_max))

    def _normalize_numerical(self, demographics: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """数値特徴量の正規化とクリッピング."""
        normalized_features = []

        for feature in self.numerical_features:
            values = demographics[feature].float()

            if feature == "age":
                # 範囲外値をクリッピング
                values = torch.clamp(values, self.age_min.item(), self.age_max.item())
                # MinMaxスケーリング
                normalized = (values - self.age_min) / (self.age_max - self.age_min + 1e-8)
            elif feature == "height_cm":
                values = torch.clamp(values, self.height_min.item(), self.height_max.item())
                normalized = (values - self.height_min) / (self.height_max - self.height_min + 1e-8)
            elif feature == "shoulder_to_wrist_cm":
                values = torch.clamp(values, self.shoulder_to_wrist_min.item(), self.shoulder_to_wrist_max.item())
                normalized = (values - self.shoulder_to_wrist_min) / (
                    self.shoulder_to_wrist_max - self.shoulder_to_wrist_min + 1e-8
                )
            elif feature == "elbow_to_wrist_cm":
                values = torch.clamp(values, self.elbow_to_wrist_min.item(), self.elbow_to_wrist_max.item())
                normalized = (values - self.elbow_to_wrist_min) / (
                    self.elbow_to_wrist_max - self.elbow_to_wrist_min + 1e-8
                )
            else:
                # デフォルトは標準化（平均0、標準偏差1）
                normalized = (values - values.mean()) / (values.std() + 1e-8)

            normalized_features.append(normalized.unsqueeze(-1))  # [batch, 1]

        return normalized_features

    def forward(self, demographics: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向き計算.

        Args:
            demographics: Demographics特徴量の辞書
                - カテゴリカル特徴量: [batch] (long型)
                - 数値特徴量: [batch] (float型)

        Returns:
            埋め込みベクトル [batch, embedding_dim]
        """
        batch_size = next(iter(demographics.values())).size(0)

        # カテゴリカル特徴量の埋め込み
        categorical_embeddings = []
        for feature in self.categorical_features:
            if feature in demographics:
                # カテゴリ値を0/1に制限
                cat_values = torch.clamp(demographics[feature].long(), 0, 1)
                embed = self.categorical_embeddings[feature](cat_values)  # [batch, embed_dim]
                categorical_embeddings.append(embed)

        # 数値特徴量の正規化
        numerical_embeddings = self._normalize_numerical(demographics)

        # 全ての特徴量を結合
        all_features = []

        # カテゴリカル埋め込みを結合
        if categorical_embeddings:
            cat_concat = torch.cat(categorical_embeddings, dim=-1)  # [batch, total_cat_dim]
            all_features.append(cat_concat)

        # 数値特徴量を結合
        if numerical_embeddings:
            num_concat = torch.cat(numerical_embeddings, dim=-1)  # [batch, numerical_dim]
            all_features.append(num_concat)

        # 全特徴量を結合
        if all_features:
            combined_features = torch.cat(all_features, dim=-1)  # [batch, combined_dim]
        else:
            # フォールバック: ゼロベクトル
            combined_features = torch.zeros(batch_size, 1, device=next(iter(demographics.values())).device)

        # 最終埋め込みベクトルを生成
        embedding = self.projection(combined_features)  # [batch, embedding_dim]

        return embedding


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    """Residual CNN block with Squeeze-and-Excitation attention."""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = self.shortcut(x)

        # First conv
        out = F.mish(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))

        # SE block
        out = self.se(out)

        # Add shortcut
        out += shortcut
        out = F.mish(out)

        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)

        return out


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence modeling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context


class IMUOnlyLSTM(nn.Module):
    """IMU-only LSTM model with ResidualSE-CNN and BiGRU attention (exp029: demographics + multi-head support)."""

    def __init__(
        self,
        imu_dim=20,
        n_classes=18,
        weight_decay=1e-4,
        demographics_dim=0,
        dropout=0.1,
    ):
        super().__init__()
        self.imu_dim = imu_dim
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.demographics_dim = demographics_dim

        # IMU deep branch with ResidualSE-CNN blocks
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=0.3, weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=0.3, weight_decay=weight_decay)

        # BiGRU
        self.bigru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)

        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional

        # Dense layers (基本特徴量抽出)
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        # Classification heads (IMU特徴量 + Demographics特徴量を結合)
        classification_input_dim = 128 + self.demographics_dim

        # 18クラス分類ヘッド
        self.multiclass_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # バイナリ分類ヘッド（Target vs Non-Target）
        self.binary_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 9クラス分類ヘッド（8 target + 1 non-target）
        self.nine_class_head = nn.Sequential(
            nn.LayerNorm(classification_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classification_input_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 9),
        )

    def forward(self, x, demographics_embedding=None):
        """
        前向き計算.

        Args:
            x: IMU データ [batch, seq_len, imu_dim]
            demographics_embedding: Demographics埋め込み [batch, demographics_dim] (オプション)

        Returns:
            tuple: (multiclass_logits, binary_logits, nine_class_logits)
        """
        # x shape: (batch, seq_len, imu_dim) -> (batch, imu_dim, seq_len)
        imu = x.transpose(1, 2)

        # IMU branch with ResidualSE-CNN
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)

        # Transpose back for BiGRU: (batch, seq_len, hidden_dim)
        merged = x1.transpose(1, 2)

        # BiGRU
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)

        # Attention
        attended = self.attention(gru_out)

        # Dense layers (基本特徴量抽出)
        x = F.mish(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        imu_features = F.mish(self.bn_dense2(self.dense2(x)))
        imu_features = self.drop2(imu_features)

        # Demographics特徴量との結合
        if demographics_embedding is not None and self.demographics_dim > 0:
            # IMU特徴量とDemographics特徴量を結合
            combined_features = torch.cat([imu_features, demographics_embedding], dim=-1)
        else:
            combined_features = imu_features

        # 3つの分類ヘッド
        multiclass_logits = self.multiclass_head(combined_features)
        binary_logits = self.binary_head(combined_features)
        nine_class_logits = self.nine_class_head(combined_features)

        return multiclass_logits, binary_logits, nine_class_logits


class CMISqueezeformer(pl.LightningModule):
    """IMU-only LSTM model for CMI competition (exp029: demographics + multi-head KL divergence)."""

    def __init__(
        self,
        input_dim: int = 20,  # IMU: 20次元（物理特徴量含む、jiazhuang compatible）
        num_classes: int = 18,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        scheduler_config: dict | None = None,
        loss_config: dict | None = None,
        acls_config: dict | None = None,
        schedule_free_config: dict | None = None,
        ema_config: dict | None = None,
        demographics_config: dict | None = None,
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
        id_to_gesture: dict[int, str] | None = None,
        **kwargs,  # 他の引数は無視（後方互換性のため）
    ):
        """
        初期化.

        Args:
            input_dim: 入力次元数（IMU物理特徴量: 20次元）
            num_classes: クラス数
            learning_rate: 学習率
            weight_decay: 重み減衰
            dropout: ドロップアウト率
            scheduler_config: スケジューラ設定
            loss_config: 損失関数設定
            acls_config: ACLS損失関数設定
            schedule_free_config: Schedule Free オプティマイザ設定
            ema_config: EMA設定
            demographics_config: Demographics統合設定
            target_gestures: ターゲットジェスチャーのリスト
            non_target_gestures: ノンターゲットジェスチャーのリスト
            id_to_gesture: ID→ジェスチャー名のマッピング
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.loss_config = loss_config or {}
        self.acls_config = acls_config or {}
        self.schedule_free_config = schedule_free_config or {}
        self.ema_config = ema_config or {}
        self.demographics_config = demographics_config or {}

        # Demographics統合の設定
        self.use_demographics = self.demographics_config.get("enabled", False)
        if self.use_demographics:
            self.demographics_embedding = DemographicsEmbedding(
                categorical_features=self.demographics_config.get(
                    "categorical_features", ["adult_child", "sex", "handedness"]
                ),
                numerical_features=self.demographics_config.get(
                    "numerical_features", ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]
                ),
                categorical_embedding_dims=self.demographics_config.get(
                    "categorical_embedding_dims", {"adult_child": 2, "sex": 2, "handedness": 2}
                ),
                embedding_dim=self.demographics_config.get("embedding_dim", 16),
                dropout=dropout,
                # 設定値ベースのスケーリングパラメータを追加
                age_min=self.demographics_config.get("age_min", 8.0),
                age_max=self.demographics_config.get("age_max", 60.0),
                height_min=self.demographics_config.get("height_min", 130.0),
                height_max=self.demographics_config.get("height_max", 195.0),
                shoulder_to_wrist_min=self.demographics_config.get("shoulder_to_wrist_min", 35.0),
                shoulder_to_wrist_max=self.demographics_config.get("shoulder_to_wrist_max", 75.0),
                elbow_to_wrist_min=self.demographics_config.get("elbow_to_wrist_min", 15.0),
                elbow_to_wrist_max=self.demographics_config.get("elbow_to_wrist_max", 50.0),
            )
            self.demographics_dim = self.demographics_config.get("embedding_dim", 16)
        else:
            self.demographics_embedding = None
            self.demographics_dim = 0

        # ジェスチャー関連の設定
        self.target_gestures = target_gestures or [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - pinch skin",
            "Neck - scratch",
            "Cheek - pinch skin",
        ]
        self.non_target_gestures = non_target_gestures or [
            "Drink from bottle/cup",
            "Glasses on/off",
            "Pull air toward your face",
            "Pinch knee/leg skin",
            "Scratch knee/leg skin",
            "Write name on leg",
            "Text on phone",
            "Feel around in tray and pull out an object",
            "Write name in air",
            "Wave hello",
        ]
        self.id_to_gesture = id_to_gesture

        # IMU-only LSTMモデル（Demographics統合対応）
        self.model = IMUOnlyLSTM(
            imu_dim=input_dim,
            n_classes=num_classes,
            weight_decay=weight_decay,
            demographics_dim=self.demographics_dim,
            dropout=dropout,
        )

        # KL divergence loss関連の設定
        self.kl_weight = self.loss_config.get("kl_weight", 0.1)
        self.kl_temperature = self.loss_config.get("kl_temperature", 1.0)
        self.nine_class_loss_weight = self.loss_config.get("nine_class_loss_weight", 0.2)
        self.loss_alpha = self.loss_config.get("alpha", 0.5)

        # 損失関数の設定
        self._setup_loss_functions()

        # メトリクス保存用
        self.validation_outputs = []

        # 手動EMAの初期化フラグ（自己参照問題を回避）
        self._ema_initialized = False

    def _setup_loss_functions(self):
        """損失関数の設定."""
        loss_type = self.loss_config.get("type", "cmi")

        if loss_type == "cmi":
            # 基本的なクロスエントロピー損失
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.multiclass_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.binary_criterion = nn.BCEWithLogitsLoss()
            self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        elif loss_type == "cmi_focal":
            # Focal Loss
            self.multiclass_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )
            self.binary_criterion = nn.BCEWithLogitsLoss()
            # 9クラス用もFocalLoss
            self.nine_class_criterion = FocalLoss(
                gamma=self.loss_config.get("focal_gamma", 2.0),
                alpha=self.loss_config.get("focal_alpha", 1.0),
                label_smoothing=self.loss_config.get("label_smoothing", 0.0),
            )

        elif loss_type == "soft_f1":
            # SoftF1Loss
            self.multiclass_criterion = self._create_soft_f1_criterion(self.num_classes)
            self.binary_criterion = self._create_binary_soft_f1_criterion()
            self.nine_class_criterion = self._create_soft_f1_criterion(9)

        elif loss_type == "soft_f1_acls":
            # SoftF1Loss with ACLS regularization
            self.multiclass_criterion = self._create_soft_f1_acls_criterion(self.num_classes)
            self.binary_criterion = self._create_binary_soft_f1_acls_criterion()
            self.nine_class_criterion = self._create_soft_f1_acls_criterion(9)

        elif loss_type == "acls":
            # ACLS (Adaptive and Conditional Label Smoothing)
            self.multiclass_criterion = self._create_acls_criterion(self.num_classes)
            self.binary_criterion = self._create_acls_binary_criterion()
            self.nine_class_criterion = self._create_acls_criterion(9)

        elif loss_type == "label_smoothing":
            # Label Smoothing Cross-Entropy
            alpha = self.acls_config.get("label_smoothing_alpha", 0.1)
            self.multiclass_criterion = LabelSmoothingCrossEntropy(alpha=alpha)
            self.binary_criterion = LabelSmoothingBCE(alpha=alpha)
            self.nine_class_criterion = LabelSmoothingCrossEntropy(alpha=alpha)

        elif loss_type == "mbls":
            # Margin-based Label Smoothing
            margin = self.acls_config.get("mbls_margin", 10.0)
            alpha = self.acls_config.get("mbls_alpha", 0.1)
            alpha_schedule = self.acls_config.get("mbls_schedule", None)
            self.multiclass_criterion = MbLS(margin=margin, alpha=alpha, alpha_schedule=alpha_schedule)
            self.binary_criterion = MbLSBinary(margin=margin, alpha=alpha)
            self.nine_class_criterion = MbLS(margin=margin, alpha=alpha, alpha_schedule=alpha_schedule)

        else:
            # デフォルト：基本的なクロスエントロピー
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            self.multiclass_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.binary_criterion = nn.BCEWithLogitsLoss()
            self.nine_class_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 共通設定
        self.loss_alpha = self.loss_config.get("alpha", 0.5)
        self.nine_class_loss_weight = self.loss_config.get("nine_class_loss_weight", 0.2)

        # KL divergence loss設定
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.kl_weight = self.loss_config.get("kl_weight", 0.1)
        self.kl_temperature = self.loss_config.get("kl_temperature", 1.0)

        # 学習可能重み付けの設定
        self._setup_learnable_weights()

    def _create_soft_f1_criterion(self, num_classes: int) -> "MulticlassSoftF1Loss":
        """SoftF1Loss criterion作成の共通化."""
        beta = self.loss_config.get("soft_f1_beta", 1.0)
        eps = self.loss_config.get("soft_f1_eps", 1e-6)
        return MulticlassSoftF1Loss(num_classes=num_classes, beta=beta, eps=eps)

    def _create_binary_soft_f1_criterion(self) -> "BinarySoftF1Loss":
        """Binary SoftF1Loss criterion作成の共通化."""
        beta = self.loss_config.get("soft_f1_beta", 1.0)
        eps = self.loss_config.get("soft_f1_eps", 1e-6)
        return BinarySoftF1Loss(beta=beta, eps=eps)

    def _create_acls_criterion(self, num_classes: int) -> "ACLS":
        """ACLS criterion作成の共通化."""
        return ACLS(
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
            num_classes=num_classes,
        )

    def _create_acls_binary_criterion(self) -> "ACLSBinary":
        """ACLS Binary criterion作成の共通化."""
        return ACLSBinary(
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
            num_classes=2,
        )

    def _create_soft_f1_acls_criterion(self, num_classes: int) -> "MulticlassSoftF1ACLS":
        """SoftF1ACLS criterion作成の共通化."""
        return MulticlassSoftF1ACLS(
            num_classes=num_classes,
            beta=self.loss_config.get("soft_f1_beta", 1.0),
            eps=self.loss_config.get("soft_f1_eps", 1e-6),
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
        )

    def _create_binary_soft_f1_acls_criterion(self) -> "BinarySoftF1ACLS":
        """Binary SoftF1ACLS criterion作成の共通化."""
        return BinarySoftF1ACLS(
            beta=self.loss_config.get("soft_f1_beta", 1.0),
            eps=self.loss_config.get("soft_f1_eps", 1e-6),
            pos_lambda=self.acls_config.get("acls_pos_lambda", 1.0),
            neg_lambda=self.acls_config.get("acls_neg_lambda", 0.1),
            alpha=self.acls_config.get("acls_alpha", 0.1),
            margin=self.acls_config.get("acls_margin", 10.0),
        )

    def forward(
        self,
        imu: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        demographics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き計算.

        Args:
            imu: IMUデータ [batch, input_dim, seq_len] or [batch, seq_len, input_dim]
            attention_mask: アテンションマスク（無視される）
            demographics: Demographics特徴量（オプショナル）

        Returns:
            tuple: (multiclass_logits, binary_logits, nine_class_logits)
        """
        # 入力形状の調整
        if imu.dim() == 3:
            if imu.size(1) == self.input_dim:
                # [batch, input_dim, seq_len] -> [batch, seq_len, input_dim]
                imu = imu.transpose(1, 2)
            # else: already [batch, seq_len, input_dim]

        # Demographics埋め込み
        demographics_embedding = None
        if self.use_demographics and demographics is not None:
            demographics_embedding = self.demographics_embedding(demographics)

        return self.model(imu, demographics_embedding)

    def compute_kl_loss(self, multiclass_logits: torch.Tensor, nine_class_logits: torch.Tensor) -> torch.Tensor:
        """
        18クラスを9クラスに集約してKL divergenceを計算.

        Args:
            multiclass_logits: [batch, 18] 18クラスのlogits
            nine_class_logits: [batch, 9] 9クラスのlogits

        Returns:
            KL divergence loss
        """
        temperature = self.kl_temperature

        # 18クラスの確率分布
        p_18 = F.softmax(multiclass_logits / temperature, dim=-1)

        # 18→9クラスへの集約
        # Target 8クラス（0-7）はそのまま、Non-target 10クラス（8-17）はindex 8に集約
        p_9_from_18 = torch.zeros(p_18.size(0), 9, device=p_18.device, dtype=p_18.dtype)
        p_9_from_18[:, :8] = p_18[:, :8]  # Target classes
        p_9_from_18[:, 8] = p_18[:, 8:].sum(dim=-1)  # Non-target classes

        # 9クラスのlog確率分布
        log_p_9 = F.log_softmax(nine_class_logits / temperature, dim=-1)

        # KLDivLoss: KL(p_9_from_18 || p_9)
        # nn.KLDivLossはinputがlog_p、targetがpを期待
        kl_loss = self.kl_criterion(log_p_9, p_9_from_18)

        return kl_loss

    def _setup_learnable_weights(self):
        """学習可能重み付けの設定."""
        aw_type = self.loss_config.get("auto_weighting", "none")

        if aw_type == "uncertainty":
            # 不確かさベース自動重み付け
            self.loss_s_params = nn.ParameterDict()
            init_val = self.loss_config.get("uncertainty_init_value", 0.0)

            # 基本的な損失項
            self.loss_s_params["multiclass"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            self.loss_s_params["binary"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

            # 9クラス損失項（有効な場合のみ）
            if self.loss_config.get("nine_class_head_enabled", True):
                self.loss_s_params["nine_class"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

            # KL損失項（有効な場合のみ）
            if self.kl_weight > 0:
                self.loss_s_params["kl"] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

        elif aw_type == "direct":
            # 直接的な学習可能重み
            init_alpha = self.loss_config.get("alpha", 0.5)
            init_w9 = self.loss_config.get("nine_class_loss_weight", 0.2)
            init_wkl = self.loss_config.get("kl_weight", 0.1)

            # alpha用（logit parameterization: alpha = sigmoid(alpha_raw)）
            alpha_logit = np.log(init_alpha / (1 - init_alpha + 1e-8))
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

            # その他重み用（log parameterization: weight = exp(weight_raw)）
            self.w9_raw = nn.Parameter(torch.tensor(np.log(max(init_w9, 1e-8)), dtype=torch.float32))
            self.wkl_raw = nn.Parameter(torch.tensor(np.log(max(init_wkl, 1e-8)), dtype=torch.float32))

        else:
            # "none": 従来の固定重み（学習可能パラメータなし）
            pass


このモデルアーキテクチャにおいてFNOを適応したほうが精度が高くなりそうな箇所と実装を教えて

* * *

**ChatGPT:**

めっちゃ相性いいです。IMUの**長距離時系列相関**を1D畳み込みやGRUより“安く”かつ“解像度に依存せず”に拾えるので、以下の2点でFNO/AFNOを入れるのがおすすめです。

* * *

# どこに入れる？なぜ効く？

1. **ResidualSE-CNNブロックの置換（時間方向のスペクトル畳み込み/FNO-1D）**  
    FNOは「FFT→低周波モード中心に線形変換→iFFT」で**グローバル受容野**を実現し、解像度（seq_len）が変わっても動く“演算子”を学習できます。PDEでの実証ですが、**解像度不変・長距離依存の学習**が強みで、時系列にもそのまま効きます。ゼロショット超解像や高速性がコア特性（FourCastNetでも核）です。[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)[ar5iv](https://ar5iv.labs.arxiv.org/html/2010.08895?utm_source=chatgpt.com)[zongyi-li.github.io](https://zongyi-li.github.io/blog/2020/fourier-pde/?utm_source=chatgpt.com)
    
2. **BiGRU+Attentionの置換（AFNOをトークンミキサとして）**  
    AFNOは自己注意の代替として**フーリエ空間でトークンを混合**するミキサで、大規模入力でも計算量が軽く、FourCastNetで高性能を実証済み。ViTのSelf-AttentionをAFNOに置換するノリで、「(batch, seq, hidden)」の**seq次元に1D FFT**を当てるだけでOK。長期依存の捕捉とスケール性が改善します。[arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2111.13587?utm_source=chatgpt.com)
    

> 参考：FNO/AFNOの基礎と利点（解像度非依存・高速なグローバル畳み込み、気象モデルでの成功）[arXiv+1](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)[authors.library.caltech.edu](https://authors.library.caltech.edu/records/hpbg9-9ea84/latest?utm_source=chatgpt.com)

* * *

# 実装プラン（最小侵襲で差し替え）

## A. FNO-1DブロックをCNNブロックの代わりに

### 1) スペクトル畳み込み（1D）実装例

```python
class SpectralConv1d(nn.Module):
    """FNO-1D: 時間次元にrFFT→上位modesのみ学習→irFFT"""
    def __init__(self, in_channels, out_channels, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # 複素数重み（実部・虚部）をパラメータ化
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, modes, 2) * 0.02)

    def compl_mul1d(self, x_ft, w):
        # x_ft: [B, C_in, F] 複素, w: [C_in, C_out, modes, 2]
        real = torch.einsum("bif,iof->bof", x_ft.real[:, :, :self.modes], w[...,0]) \
             - torch.einsum("bif,iof->bof", x_ft.imag[:, :, :self.modes], w[...,1])
        imag = torch.einsum("bif,iof->bof", x_ft.real[:, :, :self.modes], w[...,1]) \
             + torch.einsum("bif,iof->bof", x_ft.imag[:, :, :self.modes], w[...,0])
        out = torch.complex(real, imag)
        return out

    def forward(self, x):  # x: [B, C_in, T]
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, C_in, F_r]
        out_ft = torch.zeros(x.size(0), self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft, self.weight)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)  # [B, C_out, T]
        return x
```

### 2) FNOブロック（残差+位置線形）の定番形

```python
class FNOBlock1D(nn.Module):
    def __init__(self, channels, modes=16, dropout=0.1):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.w = nn.Conv1d(channels, channels, 1)  # 位置空間の補助線形
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        y = self.spectral(x) + self.w(x)
        y = F.mish(self.norm(y))
        return self.drop(x + y)
```

### 3) 既存モデルへの差し替え

```python
# 置換前: self.imu_block1/2 (ResidualSECNNBlock)
self.imu_block1 = nn.Sequential(
    nn.Conv1d(imu_dim, 128, 1, bias=False), nn.BatchNorm1d(128), nn.Mish(),
    FNOBlock1D(128, modes=32, dropout=0.2),
    FNOBlock1D(128, modes=32, dropout=0.2),
)
self.imu_block2 = nn.Sequential(
    FNOBlock1D(128, modes=32, dropout=0.2),
    nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.Mish(),
)
self.se_after_fno = SEBlock(128)  # そのまま流用可能
```

* 入出力は同じ `[B, C, T]` なので、以降のGRUやAttentionはそのまま使えます。
    
* **modes**は低周波重点（FNOの肝）。IMUでは `T≃512〜4096` 想定で `modes=16〜64` から探索。
    
* FNOは**解像度不依存**なので、学習時と推論時のseq_lenが違っても動きやすいのが魅力。[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)
    

> 参考実装は公開リポジトリでも同様の構成（2Dが多いですが1Dに容易に落とせます）。[GitHub](https://github.com/abelsr/Fourier-Neural-Operator?utm_source=chatgpt.com)

* * *

## B. GRU+Attentionの置換：AFNOミキサ（Self-Attentionの軽量代替）

### 1) AFNO-1Dトークンミキサ

AFNOは周波数領域でチャネルごとに学習的なしきい値・スパース化を入れた**周波数ミキサ**。自己注意の計算量を回避しつつ長距離依存を混合します。FourCastNet等で実証。[arXiv+1](https://arxiv.org/abs/2111.13587?utm_source=chatgpt.com)

```python
class AFNOMixer1D(nn.Module):
    """簡易AFNO: seq次元にFFT→周波数ごとにMLP→iFFT"""
    def __init__(self, hidden, ff_mult=2, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Linear(hidden, hidden*ff_mult, bias=False)
        self.ff2 = nn.Linear(hidden*ff_mult, hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):  # x: [B, T, H]
        x0 = x
        x_ft = torch.fft.rfft(x, dim=1)        # [B, F, H] 複素
        # 周波数毎に独立にチャネル混合（現実装は簡略版：実部・虚部を結合して線形）
        real, imag = x_ft.real, x_ft.imag
        y = torch.cat([real, imag], dim=-1)    # [B, F, 2H]
        y = self.ff2(F.gelu(self.ff1(y)))      # [B, F, H]
        # 元に戻す（虚部はゼロ近似の簡易版、必要なら2H→H+Hで分割）
        y_c = torch.complex(y, torch.zeros_like(y))
        x = torch.fft.irfft(y_c, n=x.size(1), dim=1)  # [B, T, H]
        return self.norm(self.drop(x) + x0)
```

### 2) 差し替えポイント

```python
# 置換前:
# gru_out, _ = self.bigru(merged); attended = self.attention(gru_out)

self.token_proj = nn.Linear(128, 256)  # 1x1相当
self.afno1 = AFNOMixer1D(256, ff_mult=2, dropout=0.2)
self.afno2 = AFNOMixer1D(256, ff_mult=2, dropout=0.2)
self.seq_pool = nn.AdaptiveAvgPool1d(1)  # or learnable attention

# forward内:
h = self.token_proj(merged)            # merged: [B, T, 128] -> [B, T, 256]
h = self.afno1(h); h = self.afno2(h)   # [B, T, 256]
attended = self.seq_pool(h.transpose(1,2)).squeeze(-1)  # [B, 256]
```

* Self-Attentionを使わず**O(T log T)**で長距離を混ぜられるのが売り。四天王（FourCastNet/AFNO）での実証あり。[arXiv+1](https://arxiv.org/abs/2111.13587?utm_source=chatgpt.com)
    

* * *

## C. Demographicsの“条件付け”でさらに効かせる（任意）

FNO/AFNOの出力や中間表現に**FiLM（Feature-wise Linear Modulation）**をかけ、Demographics埋め込みでチャネルをスケール・シフト。

```python
class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, in_dim)
        self.to_shift = nn.Linear(cond_dim, in_dim)
    def forward(self, x, cond):  # x: [B, C, T] or [B, T, H]
        s = self.to_scale(cond).unsqueeze(-1)  # [B, C, 1] or [B, H, 1]
        b = self.to_shift(cond).unsqueeze(-1)
        return x * (1 + s) + b
```

* 例：`x = self.film(self.fno_block(x), demographics_embedding)` のように挿入。
    
* 被験者差（身長・腕長など）で**周波数帯の利き方**が変わるケースに素直に対応できます。  
    （FNOは元々“演算子”学習なので条件付きとの相性が良い）
    

* * *

# ハイパラと学習Tips

* **modes**：16/32/48/64あたりで探索（`T`が短ければ小さめ）。低周波中心でOK。[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)
    
* **層数**：FNOBlockを2〜4層。GRUを残す構成なら2層、AFNO化で除去なら3〜4層。
    
* **正則化**：`Dropout 0.1–0.3`、`weight_decay 1e-4〜5e-4`。
    
* **可変長**：FNO/AFNOはいずれもFFTベースで**解像度非依存**に動作しやすい（パディングで2の冪に寄せると高速）。[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)
    
* **計算量**：Self-AttentionのO(T²)回避（AFNOはFFTベース）。長いシーケンスで学習が安定しやすい。[arXiv](https://arxiv.org/abs/2111.13587?utm_source=chatgpt.com)
    

* * *

# どっちを先に試す？

* **まずはA（FNO-1DをCNN置換）**：既存設計に最小変更で“長距離＋解像度不変”を注入でき、堅実に精度が伸びやすい。
    
* さらに伸ばすなら**B（AFNOでGRU置換）**：長距離依存をより強く＆計算量を抑えて扱えるので、大規模データ/長シーケンスで有利。
    

* * *

# 参考

* FNO本家（PDEでの解像度非依存・高速性、ゼロショット超解像）[arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com)[authors.library.caltech.edu](https://authors.library.caltech.edu/records/hpbg9-9ea84/latest?utm_source=chatgpt.com)[ar5iv](https://ar5iv.labs.arxiv.org/html/2010.08895?utm_source=chatgpt.com)
    
* 著者ブログ（要点の分かりやすいまとめ）[zongyi-li.github.io](https://zongyi-li.github.io/blog/2020/fourier-pde/?utm_source=chatgpt.com)
    
* AFNO（自己注意の代替となるフーリエミキサ）[arXiv+1](https://arxiv.org/abs/2111.13587?utm_source=chatgpt.com)
    
* FourCastNet（AFNO中核の地球規模気象モデル）[arXiv+1](https://arxiv.org/abs/2202.11214?utm_source=chatgpt.com)
    
* 参考実装（PyTorch FNO）[GitHub](https://github.com/abelsr/Fourier-Neural-Operator?utm_source=chatgpt.com)
    

* * *

必要なら、このコードをあなたの`CMISqueezeformer`クラスに**パッチ適用した完全版**も作ります。どっちの差し替え（A/B）から行くかだけ指定してくれれば、その方向で統合コードを出します。