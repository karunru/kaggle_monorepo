# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp053` を `codes/exp/exp055` にコピーしてからexp055を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp053` からのインポートは禁止
- `codes/exp/exp055` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
- 以下の特徴量で実装してないやつを実装して
  - polarsを使って実装すること
  - 実装する関数には型ヒントを記載して、必ずpolarsの型を使うこと
  - pandasを使った実装は禁止
```markdown
#### 1. 基本センサー特徴量
**加速度センサー**：
- `acc_x`, `acc_y`, `acc_z` - 生加速度データ

**回転センサー**：
- `rot_w`, `rot_x`, `rot_y`, `rot_z` - クォータニオン回転データ

#### 2. 導出物理特徴量
**線形加速度系**：
- `linear_acc_x`, `linear_acc_y`, `linear_acc_z` - 重力除去後加速度
- `linear_acc_mag`, `linear_acc_mag_jerk` - マグニチュード、変化率

**マグニチュード系**：
- `acc_mag`, `acc_mag_jerk`, `acc_mag_jerk_rate` - 合成加速度と変化率

**角度・姿勢系**：
- `roll`, `pitch`, `yaw` - オイラー角
- `rot_angle`, `rot_angle_vel`, `acc_rot_ratio` - 回転角度関連

**角速度・角加速度系**：
- `angular_vel_[x,y,z]`, `angular_vel_[x,y,z]_acc` - 角速度、角加速度
- `angular_jerk_[x,y,z]`, `angular_snap_[x,y,z]` - 角ジャーク、角スナップ
- `angular_distance` - 角距離

#### 3. 時系列統計特徴量

**3フレーム滑動窓統計**：
- 加速度系: `acc_[x,y,z]_rolling_[mean,std,std_sqrt]_3`
- 線形加速度系: `linear_acc_[x,y,z]_rolling_[median,max,min]_3`
- 回転系: `rot_[w,x,y,z]_rolling_[mean,std,std_sqrt]_3`

**5フレーム滑動窓統計**：
- 加速度系: `acc_[x,y,z]_rolling_[mean,std,std_sqrt]_5`
- 線形加速度系: `linear_acc_[x,y,z]_rolling_[median,max,min]_5`
- 回転系: `rot_[w,x,y,z]_rolling_[mean,std,std_sqrt]_5`

**10フレーム滑動窓統計**：
- 加速度系: `acc_[x,y,z]_rolling_[mean,std,std_sqrt]_10`
- 線形加速度系: `linear_acc_[x,y,z]_rolling_[median,max,min]_10`
- 回転系: `rot_[w,x,y,z]_rolling_[mean,std,std_sqrt]_10`
```