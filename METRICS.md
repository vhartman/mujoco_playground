# Logged Metrics Reference

All metrics logged to WandB (and TensorBoard) during a training run. Values are taken from run `TesolloPinch-20260508-211111` (WandB ID `ypngsypn`) as the reference example. Metrics are grouped by their WandB key prefix.

## Logging cadence overview

All logging flows through a single `progress_fn(num_steps, metrics)` callback that brax's PPO trainer calls at two distinct events. The set of keys present in `metrics` determines which event fired:

| Event | Trigger | Keys present | Flags required |
|---|---|---|---|
| **Training metrics step** | Every `training_metrics_steps` environment steps | `episode/*` | `--log_training_metrics` |
| **Eval checkpoint** | Every `num_timesteps / num_evals` steps | `eval/*`, `training/*`, and also `episode/*` | `--run_evals` (on by default) |

Concretely: with `num_timesteps=400M`, `num_evals=5`, and `training_metrics_steps=800K`, brax fires the callback roughly every 800K steps for training metrics and at 80M-step intervals for eval checkpoints. When a training-metrics step coincides with an eval checkpoint (which can happen), both key sets are present in the same `wandb.log()` call and appear as a single WandB point.

**Total datapoints per metric over a full run:**
- `episode/*` — `num_timesteps / training_metrics_steps` points (e.g. 500 points for 400M steps at 800K interval)
- `eval/*` and `training/*` — `num_evals` points (e.g. 5 points)

---

## `episode/` — Training Episode Metrics

**When logged:** Every `training_metrics_steps` environment steps, triggered by the training-metrics event. Also included at every eval checkpoint (so these metrics appear there too, reflecting the most recent training batch before the eval ran). Requires `--log_training_metrics`; if that flag is off, `episode/*` keys are never populated and this entire section goes unlogged.

**What they represent:** Averages over the stochastic training rollout batch (`num_envs` parallel envs running with action noise). These are noisier than eval metrics because the policy is still exploring.

### Reward components

| Metric | Example Value | Description |
|---|---|---|
| `episode/sum_reward` | `11.88` | Mean total undiscounted episode return across all training envs. The primary training progress signal. |
| `episode/reward/cube_force` | `509.8` | Cumulative contribution of the `cube_force` reward component per episode. High value = policy is applying force to the cube. |
| `episode/reward/success` | `0` | Cumulative success bonus per episode. Zero means no successes occurred in training rollouts. |
| `episode/reward/fingertip_pos` | `15.1` | Cumulative fingertip-proximity reward. Positive → fingers are reaching toward the cube. |
| `episode/reward/wrist_pose` | `-24.5` | Cumulative wrist-pose penalty. More negative → wrist is deviating more from the reference pose. |
| `episode/reward/hand_pose` | `-70.8` | Cumulative hand-pose penalty. Dominated by rest-pose deviation across all fingers. |
| `episode/reward/wrist_vel` | `-36.7` | Cumulative wrist velocity penalty. Indicates how much the wrist is moving. |
| `episode/reward/joint_vel` | `-1.7` | Cumulative joint velocity penalty (all joints summed). |
| `episode/reward/cube_lin_vel` | `-0.67` | Cumulative linear velocity penalty on the cube. Near-zero = cube is mostly held still. |
| `episode/reward/cube_ang_vel` | `-11.9` | Cumulative angular velocity penalty on the cube. |
| `episode/reward/action_rate` | `-38.3` | Cumulative action-rate penalty (sum of `|a_t - a_{t-1}|`). Large magnitude = policy is making jerky corrections. |
| `episode/reward/energy` | `-2.6` | Cumulative energy penalty (torque × velocity). |
| `episode/reward/termination` | `-100` | Termination penalty. `-100` means at least one episode terminated early in the batch. |

### Episode statistics

| Metric | Example Value | Description |
|---|---|---|
| `episode/length` | `409.3` | Mean episode length in steps. If well below `episode_length` (1000), episodes are terminating early (drops or constraint violations). |
| `episode/success_count` | `0` | Number of successful episodes in the training batch. |
| `episode/consecutive_success_steps` | `21.3` | Mean consecutive steps meeting the success condition. Tracks how long the policy can sustain a valid grasp. |
| `episode/steps_since_last_success` | `114975` | Mean steps elapsed since the last success event, summed across envs. Large value = policy rarely succeeds. |
| `episode/sps` | `81617` | Training steps per second (environment steps, including all parallel envs). Measure of simulation throughput. |
| `episode/learning_rate` | `3e-4` | Current learning rate. Matches `training/learning_rate`; logged here for easy correlation with episode metrics. |

### PPO losses (training rollout)

| Metric | Example Value | Description |
|---|---|---|
| `episode/total_loss` | `0.212` | Combined PPO loss: `policy_loss + entropy_loss + v_loss`. |
| `episode/policy_loss` | `-0.0098` | Policy surrogate loss. Negative = policy is improving (clipped objective is working). Near zero = no useful gradient signal. |
| `episode/v_loss` | `0.206` | Value function MSE loss. High early in training; should decrease as the critic learns the return surface. |
| `episode/entropy_loss` | `0.0163` | Entropy bonus term (negative entropy × `entropy_cost`). Tracks how random the policy is. Higher = more exploratory. |
| `episode/kl_mean` | `0.0218` | Mean KL divergence between old and new policy per update. Healthy range: `0.005`–`0.05`. Much higher → policy is changing too fast; much lower → not learning. |

### Policy distribution diagnostics

These describe the shape of the policy's action distribution (Gaussian before tanh squash).

| Metric | Example Value | Description |
|---|---|---|
| `episode/policy_dist_loc/{p25,p50,p75,mean,min,max}` | `-0.1 / 0.1 / 0.3 / 0.1 / -3.0 / 2.7` | Summary of the policy's mean outputs (loc) across batch × action dims. Rendered as a candle plot: p25–p75 band, p50 line, mean dashed line, min/max whiskers. Mean vs. median divergence = skew; whiskers near ±3 = tanh saturation. |
| `episode/policy_dist_std/{p25,p50,p75,mean,min,max}` | `0.18 / 0.30 / 0.45 / 0.34 / 0.01 / 2.7` | Summary of the policy's per-action std. min ~0 signals collapsed exploration on some dims; high max signals persistent uncertainty. |

---

## `eval/` — Evaluation Metrics

**When logged:** Only at eval checkpoints — `num_evals` times over the full run, evenly spaced at `num_timesteps / num_evals` steps apart. Requires `--run_evals` (on by default). If `--run_evals=False`, none of these keys are ever logged.

**What they represent:** Deterministic rollouts (no action noise) over `num_eval_envs` parallel environments. Because the policy is deterministic and the environments are reset fresh, these metrics are much less noisy than `episode/` and are the canonical measure of policy quality at each checkpoint.

### Episode-level summary

| Metric | Example Value | Description |
|---|---|---|
| `eval/episode_reward` | `235.4` | Mean total episode return across eval envs. The main headline metric for policy quality. Compare across runs. |
| `eval/episode_reward_std` | `8.3` | Std of episode return across eval envs. Low = consistent policy; high = high variance behavior. |
| `eval/avg_episode_length` | `1000` | Mean episode length in eval. `1000` = no early terminations; policy is stable throughout. |
| `eval/std_episode_length` | `0` | Std of episode length. `0` = all eval episodes ran to full length (no drops). |
| `eval/episode_success_count` | `40.0` | Number of eval episodes where the success condition was met (out of `num_eval_envs`). |
| `eval/episode_success_count_std` | `318.1` | Std of success count across eval batches. |
| `eval/episode_consecutive_success_steps` | `2562` | Mean consecutive steps of sustained success per eval episode. Key metric for grasp stability. |
| `eval/episode_consecutive_success_steps_std` | `954` | Std of consecutive success steps. |
| `eval/episode_steps_since_last_success` | `496882` | Summed steps since last success across eval envs. Very large = successes are rare and intermittent. |
| `eval/episode_steps_since_last_success_std` | `28736` | Std of steps-since-last-success. |

### Per-component reward breakdown (eval)

Each reward component is logged separately from eval rollouts, allowing you to see which terms are driving or limiting the total.

| Metric | Example Value | Description |
|---|---|---|
| `eval/episode_reward/cube_force` | `4873.3` | Eval cumulative cube-force reward. Much higher than training value (509) — policy applies more force in deterministic eval mode. |
| `eval/episode_reward/success` | `0.0625` | Eval success reward per episode. Small positive value = policy occasionally meets the success criterion. |
| `eval/episode_reward/fingertip_pos` | `57.3` | Eval fingertip proximity reward. Positive = fingers are approaching the cube. |
| `eval/episode_reward/wrist_pose` | `-84.7` | Eval wrist-pose penalty. |
| `eval/episode_reward/hand_pose` | `-86.1` | Eval hand-pose penalty. |
| `eval/episode_reward/wrist_vel` | `-1.77` | Eval wrist velocity penalty. Much smaller than training (−36.7) → policy is smoother in eval than training. |
| `eval/episode_reward/joint_vel` | `-2.2` | Eval joint velocity penalty. |
| `eval/episode_reward/cube_lin_vel` | `-0.44` | Eval cube linear velocity penalty. Near zero = cube is mostly stationary. |
| `eval/episode_reward/cube_ang_vel` | `-8.3` | Eval cube angular velocity penalty. |
| `eval/episode_reward/action_rate` | `-48.4` | Eval action-rate penalty. |
| `eval/episode_reward/energy` | `-3.3` | Eval energy penalty. |
| `eval/episode_reward/termination` | `0` | Eval termination penalty. Zero = no early terminations in eval (policy is stable). |

Each component also has a corresponding `_std` metric (e.g. `eval/episode_reward/cube_force_std`) measuring variance across eval envs — useful for spotting high-variance components.

### Eval infrastructure

| Metric | Example Value | Description |
|---|---|---|
| `eval/sps` | `21028` | Eval steps per second. Lower than training SPS because eval runs fewer envs and includes overhead. |
| `eval/epoch_eval_time` | `6.09` | Wall-clock seconds spent on the eval phase at this checkpoint. |
| `eval/walltime` | `269.0` | Total wall-clock seconds spent on evaluation across all checkpoints so far. |

---

## `training/` — Optimizer & Policy Diagnostics

**When logged:** At every eval checkpoint, in the same `progress_fn` call as `eval/*`. Always present when `--run_evals` is on — no separate flag required. Gives `num_evals` datapoints total, same cadence as `eval/`.

**What they represent:** Snapshot of the optimizer state and policy distribution *just before* the eval rollouts run at that checkpoint — i.e. they describe the policy that is about to be evaluated.

| Metric | Example Value | Description |
|---|---|---|
| `training/learning_rate` | `3e-4` | Current Adam learning rate. Flat unless a LR schedule is used. |
| `training/total_loss` | `0.0549` | Combined PPO loss at the checkpoint. |
| `training/policy_loss` | `-0.0047` | Policy surrogate loss at the checkpoint. |
| `training/v_loss` | `0.0472` | Value MSE loss at the checkpoint. |
| `training/entropy_loss` | `0.0124` | Entropy term at the checkpoint. |
| `training/kl_mean` | `0.0268` | KL divergence at the checkpoint. |
| `training/sps` | `81391` | Training steps per second at this checkpoint. Tracks GPU throughput over time. |
| `training/walltime` | `4977` | Total wall-clock training time in seconds (excludes eval time). |
| `training/policy_dist_loc/{p25,p50,p75,mean,min,max}` | `-0.1 / 0.1 / 0.3 / 0.1 / -2.7 / 2.8` | Per-update summary of the policy mean (loc), averaged over the SGD scan and pmap devices (mean is exact; quantiles/extrema are mean-of-per-update). Rendered as a candle plot — see `learning/wandb_charts/policy_dist_candle.json`. |
| `training/policy_dist_std/{p25,p50,p75,mean,min,max}` | `0.20 / 0.32 / 0.48 / 0.37 / 0.01 / 2.76` | Per-update summary of the policy std. Rendered as the second candle plot. |

---

## Output Log (`output.log`)

**When written:** Each call to `progress_fn` may emit up to one line per event type. The `reward=` line is written only at eval checkpoints (when `eval/episode_reward` is present in metrics). The `mean episode reward=` line is written only at training-metrics steps (when `episode/sum_reward` is present and `--log_training_metrics` is on). At a step that is both an eval checkpoint and a training-metrics step, both lines are printed together.

Two types of lines are printed to stdout and captured in `wandb/.../files/output.log`:

```
<step>: reward=<value>         # eval checkpoint reward (requires --run_evals)
<step>: mean episode reward=<value>  # training batch reward (requires --log_training_metrics)
```

| Field | Description |
|---|---|
| `step` | Environment steps elapsed (across all parallel envs). |
| `reward=` | `eval/episode_reward` at this eval checkpoint — total episode return in deterministic eval. |
| `mean episode reward=` | `episode/sum_reward` from the training batch — mean return during exploration rollouts, before any deterministic evaluation. Noisier than `reward=`. |
