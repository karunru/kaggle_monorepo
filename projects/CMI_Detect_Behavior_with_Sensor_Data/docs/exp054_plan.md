# コンペの基本概念として以下のファイルを参照すること
  - `docs/competition_overview.md`: コンペティション概要の確認
  - `docs/data_description.md`: データセット詳細の理解
# 実装方針
- `codes/exp/exp046` を `codes/exp/exp054` にコピーしてからexp054を実装すること
  - test_*.pyはいらない
- 差分は指示されたもの以外最小限にすること 
- `codes/exp/exp046` からのインポートは禁止
- `codes/exp/exp054` 以外の `codes/exp/` 配下の他のexp directory配下のファイルの編集は禁止
- config.pyの以下の項目は必ず更新して
  - `ExperimentConfig.description`
  - `ExperimentConfig.tags`
  - `LoggingConfig.wandb_tags`
# 新規実装アイデア
```python
base_feats = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']
```
sequence_idごとの base_featsの min, max, mean, median, 25percentile, 75percentile, skew, stdを特徴量にして
`np.sum(np.diff(np.signbit(x).astype(int)) != 0)` のようにbase_featsの0を通過した回数を特徴量にして
sequence_idの長さ、`seq_df['sequence_counter'].max() - seq_df['sequence_counter'].min()` も特徴量にして
上記のプロセスで作成した新規特徴量を正規化して`combined_features = torch.cat([imu_features, demographics_embedding], dim=-1)` に追加して


