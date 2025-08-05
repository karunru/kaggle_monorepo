# 技術スタックと依存関係

## パッケージマネージャー
- **uv**: 0.6.9（高速Pythonパッケージマネージャー）
- パッケージ追加: `uv add package_name`
- 実行: `uv run command`

## 主要依存関係

### 深層学習・機械学習
- **torch**: >=2.6（CUDA 12.4対応）
- **torchvision**: >=0.21
- **lightning**: >=2.5.0（PyTorch Lightning）
- **timm**: >=1.0.15（画像モデルライブラリ）
- **lightgbm**: >=4.3
- **optuna**: >=3.5（ハイパーパラメータ最適化）

### データ処理
- **polars**: >=0.20.15（高速データフレーム）
- **cupy-cuda12x**: >=13.4.1（GPU加速NumPy）
- **scipy**: >=1.15.2

### 設定・実験管理
- **hydra-core**: >=1.3.2
- **omegaconf**: >=2.3
- **pydantic**: ==2.10.6（厳密バージョン指定）
- **pydantic-settings**: >=2.2.1

### 可視化・ユーティリティ
- **rich**: >=14.0.0（リッチテキスト表示）
- **loguru**: >=0.7.2（ログ管理）
- **tqdm**: >=4.64.1（プログレスバー）

### Kaggle関連
- **grpcio**: >=1.74.0（Kaggle評価API用）
- **grpcio-tools**: >=1.74.0

### 開発依存関係
- **kaggle**: >=1.6.6（Kaggle API）
- **ruff**: >=0.12.1（リンター・フォーマッター）
- **mypy**: >=1.16.1（型チェック）
- **pytest**: >=8.4.1（テスト）
- **wandb**: >=0.16.4（実験トラッキング）

## カスタムPyTorchインデックス
- PyTorch CUDA 12.4用のカスタムインデックスを使用
- URL: https://download.pytorch.org/whl/cu124