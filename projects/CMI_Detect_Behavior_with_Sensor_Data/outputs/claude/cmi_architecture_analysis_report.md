# CMI PyTorchモデルアーキテクチャ比較解析レポート

## 解析概要
解析対象データセット数: 5
総モデル数: 30

## セキュリティチェック結果
- cmi-imu-model: 0/5 モデルが安全
- cmi-fullfeats-models: 0/10 モデルが安全
- s-offline-0-8254-15fold: 0/5 モデルが安全
- cmi-imu-only-models: 0/5 モデルが安全
- b-offline-0-8855-specialprocess: 0/5 モデルが安全

## アーキテクチャ比較
### singleNet アーキテクチャ
- 使用データセット: cmi-imu-model, s-offline-0-8254-15fold, cmi-imu-only-models
- 平均パラメータ数: 0

### BiNet アーキテクチャ
- 使用データセット: cmi-fullfeats-models, b-offline-0-8855-specialprocess
- 平均パラメータ数: 0

## 特徴量情報比較
- cmi-imu-model: 124 特徴量
- cmi-fullfeats-models: 136 特徴量
- s-offline-0-8254-15fold: 91 特徴量
- cmi-imu-only-models: 91 特徴量
- b-offline-0-8855-specialprocess: 376 特徴量

## 詳細分析結果

### cmi-imu-model
- モデル数: 5
- アーキテクチャタイプ: singleNet
- 平均パラメータ数: 0
- 特徴量数: 124

### cmi-fullfeats-models
- モデル数: 10
- アーキテクチャタイプ: BiNet
- 平均パラメータ数: 0
- 特徴量数: 136

### s-offline-0-8254-15fold
- モデル数: 5
- アーキテクチャタイプ: singleNet
- 平均パラメータ数: 0
- 特徴量数: 91

### cmi-imu-only-models
- モデル数: 5
- アーキテクチャタイプ: singleNet
- 平均パラメータ数: 0
- 特徴量数: 91

### b-offline-0-8855-specialprocess
- モデル数: 5
- アーキテクチャタイプ: BiNet
- 平均パラメータ数: 0
- 特徴量数: 376