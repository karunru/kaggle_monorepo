# Human Normalization Implementation Plan (IMU × Anthropometrics)

## Objective
- Normalize IMU-derived signals using subject-specific anthropometrics (`elbow_to_wrist_cm`, `shoulder_to_wrist_cm`, `height_cm`) so motion features are comparable across body sizes and limb lengths.

## References
- Competition overview: `docs/competition_overview.md`
- Data description: `docs/data_description.md`
- Existing pipeline (baseline): `codes/exp/exp021/` (dataset construction + physics features)
- IMU sensor spec: Helios2025 IMU PDF (referenced in plan; use locally if available)
- Training data: `data/train.csv`, demographics: `data/train_demographics.csv`

## Scope
- Add “human-normalized” feature channels computed from existing physics features and demographics.
- Keep experiments isolated (no cross-imports). Gate with config flags to enable/disable.
- Zero behavior change when disabled.

---

## Feature Design

- Base signals available in exp021 after physics stage:
  - Linear acceleration (gravity removed): `linear_acc_{x,y,z}`, magnitude `linear_acc_mag`
  - Angular velocity from quaternion: `angular_vel_{x,y,z}`, angular distance `angular_distance`

- Anthropometrics per subject (meters):
  - `h = height_cm / 100`
  - `r_elbow = elbow_to_wrist_cm / 100` (forearm lever length)
  - `r_shoulder = shoulder_to_wrist_cm / 100` (upper arm + forearm lever)
  - Effective radius: `r_eff = clip(r_shoulder, min=0.15, max=0.9)` as a conservative wrist-to-shoulder distance (bounds configurable).

- Derived kinematics from angular velocity (per time step):
  - Angular speed: `omega = ||angular_vel|| = sqrt(wx^2 + wy^2 + wz^2)`
  - Tangential speed estimates: `v_elbow = omega * r_elbow`, `v_shoulder = omega * r_shoulder`
  - Centripetal acceleration estimates: `a_c_elbow = omega^2 * r_elbow`, `a_c_shoulder = omega^2 * r_shoulder`

- Normalized, near size-invariant targets (added as new channels):
  - Acc per height: `linear_acc_mag_per_h = linear_acc_mag / h`
  - Acc per shoulder span: `linear_acc_mag_per_rS = linear_acc_mag / r_shoulder`
  - Acc per elbow span: `linear_acc_mag_per_rE = linear_acc_mag / r_elbow`
  - Dimensionless circularity ratios:
    - `acc_over_centripetal_rE = linear_acc_mag / max(a_c_elbow, eps)`
    - `acc_over_centripetal_rS = linear_acc_mag / max(a_c_shoulder, eps)`
  - Angular-domain approximations (convert linear to angular by dividing length):
    - `alpha_like_rE = linear_acc_mag / max(r_elbow, eps)`
    - `alpha_like_rS = linear_acc_mag / max(r_shoulder, eps)`
  - Speed-like composites:
    - `v_over_h = v_shoulder / h`, `v_over_rS = v_shoulder / r_shoulder`, `v_over_rE = v_elbow / r_elbow`
  - Axis-wise versions (optional, configurable) using `linear_acc_{x,y,z}` and `angular_vel_{x,y,z}` with same formulas.

- Handling missing anthropometrics:
  - If any of `h`, `r_elbow`, `r_shoulder` is missing/non-positive: fall back to cohort medians (computed from demographics) and mark a per-sequence binary flag feature (e.g., `hn_used_fallback = 1`).

- Numeric stability:
  - Use `eps = 1e-6` in denominators and clamp radii to sensible [min,max].
  - Preserve existing `missing_mask` in attention to avoid learning from padded/invalid timesteps.

---

## Implementation Plan

- Add shared feature module
  - Functions:
    - `compute_hn_features(frame: pl.LazyFrame, demo_df: pl.DataFrame, cfg: HNConfig) -> pl.DataFrame`
    - `join_subject_anthro(frame, demo_df)`: join `subject` to `h`, `r_elbow`, `r_shoulder` (meters) with fallbacks.
    - `derive_hn_channels(frame, eps, bounds)`: compute channels listed above.
    - Pydantic-style dataclass `HNConfig` for toggles, bounds, and which channels to enable.

- Integrate into exp021 dataset
  - In `IMUDataset._add_physics_features` (Polars/Lazy stage), after computing physics features, perform a lazy join on `subject` with a normalized demographics table to add `h`, `r_elbow`, `r_shoulder` columns.
  - Compute normalized channels via Polars expressions; select only enabled channels.
  - Guard behind config: `demographics.hn_enabled: bool` and per-channel toggles.

- Config updates (exp021)
  - File: `codes/exp/exp021/config.py`
  - Add under `demographics`:
    - `hn_enabled: bool = True`
    - `hn_eps: float = 1e-6`
    - `hn_radius_min_max: tuple[float,float] = (0.15, 0.9)`
    - `hn_features: list[str]` default to [
      "linear_acc_mag_per_h", "linear_acc_mag_per_rS", "linear_acc_mag_per_rE",
      "acc_over_centripetal_rS", "acc_over_centripetal_rE",
      "alpha_like_rS", "alpha_like_rE", "v_over_h", "v_over_rS", "v_over_rE"
    ]

- Data flow and contracts
  - `IMUDataset.__init__` already loads demographics and creates `sequence_to_subject` mapping.
  - For vectorized path, keep computation in Polars to avoid Python loops.
  - Ensure resulting `self.imu_cols` extends to include HN channels (append to the list when enabled).

---

## Validation & Testing

- Unit tests (`tests/`)
  - File: `tests/test_hn_features.py`
  - Cases:
    - Deterministic toy sequence with known `omega`, `r_elbow`, `r_shoulder` → verify `a_c_* = omega^2 * r` and ratio channels equal expected.
    - Missing anthropometrics → uses medians and sets `hn_used_fallback` flag.
    - Numeric stability: zero `omega`, zero/negative radii → outputs finite with eps handling.
  - Quick contract test ensuring enabling HN increases feature dimension by expected count and preserves shapes/ordering.

- Smoke test on a small subset
  - Load first N sequences from `train.csv` with corresponding `train_demographics.csv`.
  - Run `IMUDataset` with `hn_enabled=True`; assert no NaNs/inf in produced tensors; log per-channel stats.

- Offline checks
  - Distribution plots for HN vs. non-HN features to confirm reduced variance across `height_cm` and `shoulder_to_wrist_cm` quantiles.

---

## Experiment Integration

- Add a new experiment `exp022_hn_baseline` cloning `exp021` config with `hn_enabled=True` and logging ablation:
  - Baseline: `exp021` (HN off)
  - HN-All: all channels on
  - HN-RadiusOnly: only per-radius scalings
  - HN-CentripetalRatio: only `acc_over_centripetal_*`

- Metrics to compare: validation `cmi_score`, per-class F1, confusion matrices.

---

## Risks & Mitigations

- Gesture-specific kinematics: true lever arm differs by motion (shoulder vs elbow). Mitigate by including both `r_elbow` and `r_shoulder` variants and letting the model learn.
- Sensor orientation drift: rely on existing gravity removal and angular velocity computation; consider adding low-pass smoothing if noisy.
- Missing or outlier anthropometrics: clamp and fallback to medians; add a binary flag.

---

## Task Breakdown

1) Add `human_normalization.py` with Polars implementation
2) Extend `exp021` dataset to compute+append HN channels
3) Wire config flags in `codes/exp/exp021/config.py`
4) Add unit tests for formulas and fallbacks
5) Run local smoke test; inspect stats/logs
6) Clone exp to `exp022_hn_baseline`; run ablations

---

## Acceptance Criteria
- `hn_enabled=False` yields identical tensors to current `exp021`.
- With HN on, new channels appear, contain finite values, and reduce across-subject variance when grouping by `height_cm`/`r_*` deciles.
- Tests pass: formulas, fallbacks, shape contracts.
