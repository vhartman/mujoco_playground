# Analyze Training Run

Investigate logs from a PPO training run in this MuJoCo Playground project and identify problems in the training setup.

## Instructions

The user may pass a run ID or path as `$ARGUMENTS`. If not provided, find and analyze the most recent training run.

### Step 1 — Locate the run

Runs are stored under two locations:
- `logs/<RunName-YYYYMMDD-HHMMSS>/` — checkpoints and env config
- `wandb/run-<YYYYMMDD_HHMMSS>-<id>/` — WandB logs with metrics, config, and output

If `$ARGUMENTS` is given, match it against run names in `logs/` or WandB run IDs. Otherwise use the most recently modified directory under `logs/`.

Find the corresponding WandB run by matching timestamps.

### Step 2 — Gather data

Read from the matched run:

1. **`wandb/.../files/config.yaml`** — all hyperparameters (PPO settings, env params, reward scales, network arch)
2. **`wandb/.../files/output.log`** — episode reward progression over steps
3. **`wandb/.../files/wandb-summary.json`** — final metrics (reward breakdown by component, KL, v_loss, entropy, SPS, success count)
4. **`logs/.../checkpoints/config.json`** — environment config (reward scales, noise, perturbations)
5. **`logs/.../checkpoints/<latest>/ppo_network_config.json`** — network architecture

### Step 3 — Analyze

Diagnose the following:

**Learning dynamics:**
- Is total reward improving, plateauing, or collapsing? At what step did it plateau?
- How fast is learning progressing (steps per second)?
- Is KL divergence reasonable (typical healthy range: 0.005–0.05)? Too high → unstable updates; too low → not learning.
- Is value loss decreasing over time, or stuck high?
- Is entropy loss too low (policy collapsed) or too high (not converging)?

**Reward structure:**
- Which reward components dominate (check scales vs. actual magnitudes in summary)?
- Are any penalty terms so large they swamp the task reward?
- Is `termination` reward being triggered often (high penalty = many episode resets)?
- Is `success` reward > 0? Is `consecutive_success_steps` meaningful?
- Do `wrist_pose`, `hand_pose`, `fingertip_pos` penalties make sense relative to `cube_force` and `success` rewards?

**PPO hyperparameters:**
- Learning rate: typical range 1e-4 to 5e-4
- Entropy cost: too low → premature collapse; typical 1e-3 to 1e-2
- Discounting: for 1000-step episodes, 0.97–0.99 is typical
- Clipping epsilon: 0.2–0.3 is standard
- Unroll length and minibatches: check for reasonable batch sizes
- Reward scaling: check if it matches reward magnitude (if mean reward is ~100 and scaling is 0.1, gradients may be too small)

**Network architecture:**
- Check if policy/value hidden layer sizes are appropriate for obs dimensionality
- Check `init_noise_std` — if too low initially, exploration suffers
- Check if `state_dependent_std` is appropriate

**Environment config:**
- Are `obs_noise` levels reasonable?
- Is `pert_config.enable` set appropriately for curriculum stage?
- Does `success_hold_time` match `episode_length` constraints?
- Is `force_target` achievable given the reward structure?

### Step 4 — Report

Produce a structured report with:

1. **Run summary** — env, steps trained, final reward, success rate
2. **Learning curve assessment** — improving/stuck/collapsed, where problems began
3. **Identified issues** — prioritized list, each with:
   - What the problem is
   - Evidence from the logs
   - Suggested fix (specific parameter change or range)
4. **What looks healthy** — confirm things that are working well
5. **Recommended next experiment** — the single most impactful change to try

Be specific: quote actual values from the logs, not generic advice.
