**Title**
- Make `total_loss` weights learnable in `codes/exp/exp024/model.py`

**Context**
- Current `total_loss` in `exp024` combines four terms with fixed weights:
  - `multiclass_loss` (18-class)
  - `binary_loss` (Target vs Non-target)
  - `nine_class_loss` (8 target + other)
  - `kl_loss` (18→9 distribution consistency)
- Implementation (training/validation):
  - `total_loss = alpha * L_mc + (1 - alpha) * L_bin + w9 * L_9 + wkl * L_kl`
  - Weights from `LossConfig`: `alpha`, `nine_class_loss_weight`, `kl_weight`.
- Goal: make these weights learnable during training while keeping exp024 behavior by default when disabled.

**Approach Options**
- Option A — Uncertainty weighting (Kendall & Gal 2018)
  - Learn one parameter per loss: `s_i = log(sigma_i^2)`.
  - Combine by `L_total = Σ [exp(-s_i) * L_i + s_i]`.
  - Pros: principled, scale-invariant, fewer knobs; automatically downweights noisier tasks.
  - Cons: replaces the explicit convex mix between `L_mc` and `L_bin`; if you want to preserve a strict α/(1-α) tradeoff, use Option B.

- Option B — Direct learnable weights with constraints
  - Learn raw parameters and map to valid ranges:
    - `alpha = sigmoid(alpha_raw)` ∈ (0, 1)
    - `w9 = exp(w9_raw)` > 0
    - `wkl = exp(wkl_raw)` > 0
  - Combine by `L_total = alpha * L_mc + (1 - alpha) * L_bin + w9 * L_9 + wkl * L_kl`.
  - Pros: preserves current semantics; drop-in replacement for fixed weights; can initialize from config.
  - Cons: needs regularization/clamping to avoid extreme weights; absolute loss scale matters.

**References**
- See docs/chatgpt_loss_plan.md: sections on multi-head losses, hierarchical consistency, and “自動重み付けのpytorch実装例” for uncertainty-weighted losses and KL consistency.

**Recommended Design**
- Add a configurable toggle to `LossConfig` to enable learnable weighting without breaking existing configs.
  - `auto_weighting: Literal["none", "uncertainty", "direct"] = "none"`
  - `init_weights`: optional dict to seed initial values from current config.
- Logging: always log the effective weights each step/epoch for monitoring.
- Optimizer: include the new parameters in the model’s parameter list; optionally use a smaller LR and no weight decay for them via a separate param group.

**Minimal Changes (sketch)**
- In `codes/exp/exp024/config.py` (LossConfig):
  - Add fields
    - `auto_weighting: Literal["none", "uncertainty", "direct"] = "none"`
    - `auto_weight_clamp: tuple[float, float] = (-10.0, 10.0)`  (applies to raw params)

- In `codes/exp/exp024/model.py` (within LightningModule):
  - Define parameters in `__init__` or `_setup_loss_functions` based on config.

```python
# inside __init__ or _setup_loss_functions
aw_type = self.loss_config.get("auto_weighting", "none")
if aw_type == "uncertainty":
    # s for [mc, bin, nine, kl] (include only enabled terms)
    self.loss_s_params = nn.ParameterDict()
    self.loss_s_params["mc"] = nn.Parameter(torch.tensor(0.0))
    self.loss_s_params["bin"] = nn.Parameter(torch.tensor(0.0))
    if self.loss_config.get("nine_class_head_enabled", True):
        self.loss_s_params["nine"] = nn.Parameter(torch.tensor(0.0))
    if self.kl_weight > 0:
        self.loss_s_params["kl"] = nn.Parameter(torch.tensor(0.0))

elif aw_type == "direct":
    # alpha in (0,1), others >0
    init_alpha = float(self.loss_config.get("alpha", 0.5))
    self.alpha_raw = nn.Parameter(torch.tensor(np.log(init_alpha/(1-init_alpha)), dtype=torch.float32))
    init_w9  = float(self.loss_config.get("nine_class_loss_weight", 0.2))
    init_wkl = float(self.loss_config.get("kl_weight", 0.1))
    self.w9_raw  = nn.Parameter(torch.tensor(np.log(max(init_w9, 1e-8)), dtype=torch.float32))
    self.wkl_raw = nn.Parameter(torch.tensor(np.log(max(init_wkl,1e-8)), dtype=torch.float32))
```

- In `training_step` and `validation_step` (where `total_loss` is assembled):

```python
aw_type = self.loss_config.get("auto_weighting", "none")
if aw_type == "uncertainty":
    # clamp s for stability
    s_mc  = self.loss_s_params["mc"].clamp(*self.loss_config.get("auto_weight_clamp", (-10.0, 10.0)))
    s_bin = self.loss_s_params["bin"].clamp(*self.loss_config.get("auto_weight_clamp", (-10.0, 10.0)))
    terms = [torch.exp(-s_mc) * multiclass_loss + s_mc, torch.exp(-s_bin) * binary_loss + s_bin]
    if "nine" in self.loss_s_params:
        s_nine = self.loss_s_params["nine"].clamp(*self.loss_config.get("auto_weight_clamp", (-10.0, 10.0)))
        terms.append(torch.exp(-s_nine) * nine_class_loss + s_nine)
    if "kl" in self.loss_s_params:
        s_kl = self.loss_s_params["kl"].clamp(*self.loss_config.get("auto_weight_clamp", (-10.0, 10.0)))
        terms.append(torch.exp(-s_kl) * kl_loss + s_kl)
    total_loss = sum(terms)

elif aw_type == "direct":
    alpha = torch.sigmoid(self.alpha_raw)
    w9    = torch.exp(self.w9_raw)
    wkl   = torch.exp(self.wkl_raw)
    total_loss = alpha * multiclass_loss + (1 - alpha) * binary_loss + w9 * nine_class_loss + wkl * kl_loss

else:
    total_loss = (
        self.loss_alpha * multiclass_loss
        + (1 - self.loss_alpha) * binary_loss
        + self.nine_class_loss_weight * nine_class_loss
        + self.kl_weight * kl_loss
    )

# Optional: log effective weights
if aw_type == "uncertainty":
    self.log("w_mc", torch.exp(-s_mc).detach())
    self.log("w_bin", torch.exp(-s_bin).detach())
    if "nine" in self.loss_s_params: self.log("w_nine", torch.exp(-s_nine).detach())
    if "kl"   in self.loss_s_params: self.log("w_kl", torch.exp(-s_kl).detach())
elif aw_type == "direct":
    self.log("alpha", alpha.detach())
    self.log("w9", w9.detach())
    self.log("wkl", wkl.detach())
```

- In `configure_optimizers` (optional param group):

```python
aw_type = self.loss_config.get("auto_weighting", "none")
aw_params = []
if aw_type == "uncertainty":
    aw_params = list(self.loss_s_params.parameters())
elif aw_type == "direct":
    aw_params = [self.alpha_raw, self.w9_raw, self.wkl_raw]

if aw_params:
    base_params = [p for n, p in self.named_parameters() if p not in set(aw_params)]
    optimizer = AdamW([
        {"params": base_params},
        {"params": aw_params, "lr": self.learning_rate * 0.5, "weight_decay": 0.0},
    ], lr=self.learning_rate, weight_decay=self.weight_decay)
else:
    optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
```

**Notes & Considerations**
- Term inclusion
  - Only include `nine_class_loss` if `nine_class_head_enabled` is true.
  - Include `kl_loss` only if `kl_weight > 0` OR you decide to always learn it (set initial raw to a very negative value to effectively zero it).
- Initialization
  - Uncertainty: set all `s_i = 0` so `exp(-s_i)=1` initially.
  - Direct: `alpha_raw = logit(alpha_cfg)`, `w_raw = log(max(w_cfg, eps))` to map back via `sigmoid/exp`.
- Stability
  - Clamp raw params (or `s_i`) to avoid numerical blow-ups.
  - Optionally regularize: `+ λ * ||s||^2` or `+ λ * (alpha - alpha_prior)^2`.
  - Gradient clipping already exists; keep it enabled.
- Logging
  - Add `self.log` for effective weights to track training dynamics.
  - For uncertainty, also log `s_i` for interpretability.
- Validation/Inference
  - Weight learning only affects training loss. Inference head outputs and post-processing remain unchanged.
- Tests (suggested)
  - Verify that enabling `auto_weighting` runs forward/backward without shape/device issues.
  - Check that learnable parameters receive gradients and update over a few optimizer steps.
  - Confirm that with `auto_weighting="none"` results match current fixed-weight behavior for a tiny batch.

**Why This Fits exp024**
- Matches the multi-head structure and existing KL consistency design.
- Respects current configuration defaults and avoids breaking tests when disabled.
- Builds on the prior plan in docs/chatgpt_loss_plan.md (uncertainty-based weighting and hierarchical consistency), letting the model adaptively re-balance tasks during training.

**Next Steps**
- If you want, I can wire this behind `LossConfig.auto_weighting` and add minimal unit tests under `tests/test_exp024_*` to validate gradients and logging.

