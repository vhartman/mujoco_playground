# E3 — delta_pose action mode

**Verdict: NEGATIVE for force.magnitude; unexpected winner = proprio.target (delayed PD error).**
Delta_pose (`ctrl = q + action`, so PD-error = action directly) does NOT improve force-sensing
utility — instead it **destroys training stability for direct force-feedback policies** and
produces a different failure mode from E1.

## Setup
24 runs: `{none, baseline, proprio.target, force.magnitude}` × {fixed, randomized cube_size
[0.85,1.15]} × 3 seeds. Same structure as E1 but `action_mode=delta_pose` everywhere.

## Critical result: training divergence by bundle

| Bundle | Fixed diverge | Rand diverge | Pattern |
|--------|:---:|:---:|---------|
| `none` | 0/3 | 1/3 | stable |
| `baseline` | 2/3 | 2/3 | unstable |
| `proprio.target` | 0/3 | 2/3 | stable on fixed, fragile on rand |
| `force.magnitude` | **3/3** | **3/3** | **always diverges** |

Diverged runs: v_loss ~10¹⁹–10²⁰, KL ~0.0001 (policy frozen), effective_force ~4–10 N (hand
never closes). Reward stuck at ~4.0 for all 300M steps.

## Mechanistic explanation

Under delta_pose, `ctrl_t = q_t + action_t`, so **force ≈ kp · action_t · scale** — the
current action *is* the PD error. For `force.magnitude` policies:

- The fingertip force obs at time t reflects action_t directly (same step)
- The PPO critic must assign a value to states where obs encodes the action just taken
- This creates a circular gradient: ∂V(obs)/∂action ≠ 0 through the force channel
- Value function explodes (v_loss → ∞); policy collapses to zero-entropy fixed point

For `proprio.target` (obs includes motor_targets = ctrl_{t-1} = q_{t-1} + action_{t-1}):

- The "force proxy" is 1-step delayed (prev action, not current)
- No circular gradient → stable training
- Policy learns to read delayed PD error as an implicit force signal

## Force-error at eval (healthy seeds only)

**Fixed, nominal size (scale=1.0):**
- `proprio.target` (3 seeds): **0.22 N** — best absolute performance seen across all experiments
- `none` (3 seeds): 1.20 N
- `baseline` (1 seed): 1.29 N

**Fixed, off-nominal:** `proprio.target` spikes to 2.6–3.1 N at ±5% and 5.3 N at +15% —
classic hard-coded map. `none` and `baseline` are more robust off-nominal (~1.3–1.9 N).

**Rand [0.85–1.0], healthy seeds:**
- `proprio.target` (1 seed): ~0.96 N (generalises better than fixed, limited statistics)
- `baseline` (1 seed): ~0.85–1.1 N
- `none` (2 seeds): ~1.5–1.6 N flat across sizes

All bundles spike at 1.15 (training boundary).

## Comparison to E1 (delta)

| | E1 delta (rand size, best bundle) | E3 delta_pose (rand size, best healthy) |
|---|---|---|
| `force.magnitude` | **0.34–0.43 N** ✓ | diverges ✗ |
| `proprio.target` | ~1.6 N (worse than baseline) | **~0.96 N** (but 1 seed) |
| `none` | ~1.7–1.8 N floor | ~1.5–1.6 N (slightly better?) |

E3 does not improve on E1 for the force-sensing detection goal. E1 with delta actions remains
the cleaner experiment: stable training across all bundles, clear force.magnitude > baseline >
none hierarchy.

## Caveats
- RAND results are 1-seed for `baseline` and `proprio.target` (fragile conclusions)
- The 1.15 spike is a training-distribution boundary effect across all experiments
- The `proprio.target` 0.22 N at nominal (fixed) is a strong result but reflects
  hard-coded mapping — not generalised force sensing

## Takeaway for the study
Delta_pose was intended to set the PD error directly, potentially simplifying force control.
Instead it introduces a **gradient-aliasing instability** for direct force-feedback
(`force.magnitude`): the action is simultaneously the force and the obs input, creating
degenerate PPO value estimates. The only stable force proxy under delta_pose is a
**1-step delayed** signal (`motor_targets` in `proprio.target`). E1 (delta) remains the
definitive lever for force-sensing detection. E3 is a negative result about action-mode
and obs-space interactions.

Figure: `rollout_data/force_error_vs_size_e3_dp.png`
