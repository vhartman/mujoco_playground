# Analyze Training Run

Investigate logs from a PPO training run in this MuJoCo Playground project and identify problems in the training setup.

## How to gather data

There is a helper that does all the file resolution, pairing, and metric
extraction in one shot. **Always use it** — do not re-derive paths manually,
do not write inline python for parsing config.yaml/wandb-summary.json.

```bash
# most recent run
python learning/analyze_run.py

# N most recent (compact one-line table when N > 4)
python learning/analyze_run.py --last 6
python learning/analyze_run.py --last 20 --table   # table only, no per-run detail

# specific runs by name prefix or substring
python learning/analyze_run.py TesolloPickAndPlaceForceProprio-20260603
python learning/analyze_run.py stab-baseline-s0 stab-ent5e3-s0

# dump everything as JSON (use when you need a field the default output omits)
python learning/analyze_run.py --raw <name>
```

Output the script gives you per run, in fixed format:

- `env`, `steps`, `evals`, `completed`
- `reward`, `successes/ep`, `max_drop` (largest peak→trough after the climb),
  `low_frac` (fraction of evals below reward=15 after the climb — instability proxy)
- `train_kl`, `ep_kl`, `v_loss`, `ent_loss`, `mean_std`, `max_std`
- env knobs (`action_scale`, `ema_alpha`, `target_radius`, `target_hold_time`) +
  full `reward_scales` dict
- reward breakdown for each component (`cube_pos`, `cube_ori`, …, `action_rate`)
- 12-point sampled reward curve

Results are cached under `.claude/cache/analyze_run/` keyed by wandb run id,
invalidated by `wandb-summary.json` mtime — repeated calls are ~30 ms.

**Things to check beyond the default output** (use Bash/Read when needed):
- `logs/<run>/checkpoints/<step>/ppo_network_config.json` — network arch.
- `logs/<run>/checkpoints/config.json` — env config as actually used after `env_overrides_file` merging.
- `wandb/<run>/files/output.log` — full reward curve, hook compile timings,
  perturbation triggers, anything not numeric.
- `--raw` for any wandb metric not in the default output (e.g.
  `eval/episode_length`, `episode/policy_loss`, vision-mode metrics).

## What to diagnose

**Learning dynamics:**
- Is total reward improving, plateauing, or collapsing? At what step did it plateau?
- Is KL reasonable (healthy: 0.005–0.05)? `train_kl` is the per-update post-step value (spikes here mean the trust region was violated); `ep_kl` is the rollout-time mean (should be close to zero if the policy is stable).
- Is `v_loss` decreasing or stuck? Stuck-high while reward climbs = critic chasing a moving target.
- `ent_loss` near zero or sign-flipping = entropy collapse risk.
- `max_drop` > 5 reward → look for the cause (perturbation? task change? KL spike?).
- `low_frac` > 0.3 after a successful climb = unstable.

**Reward structure:**
- From the reward breakdown: which component dominates? Which penalty is biggest?
- Is `eval/episode_reward/success` > 0? If it should be but isn't, check that
  the success bonus is actually wired into `step()` reward (it has been
  commented out in this codebase before).
- A high `cube_ori` with low `cube_pos` (vs the other runs) means the policy is
  exploiting orientation reward in a local optimum without solving the task.
- A high `action_rate` magnitude vs `cube_pos` means the regularizer is dominant
  — usually a sign the policy is flailing because it can't solve the task.

**PPO hyperparameters** (config.yaml or `logs/<run>/checkpoints/config.json`):
- lr 1e-4 – 5e-4; entropy 1e-3 – 1e-2; discounting 0.97–0.99; clipping 0.2–0.3.
- Reward scaling vs final reward: if reward is ~400 and scaling is 0.1, advantages are ~40 — fine. If reward is ~5 and scaling is 0.1, advantages are tiny and gradients vanish.

**Environment:**
- `obs_noise.scales` — note that the level multiplier is applied on top, so the
  effective sigma is `level * scale`.
- `pert_config.enable` — perturbations off means we're not testing robustness.
- `target_radius` × `target_hold_time / ctrl_dt` = success window in steps.

## How to report

Be concrete: cite the actual numbers the script printed, not generic advice.

1. **Run summary** — env, steps, final reward, success rate, what changed vs prior runs if known.
2. **Learning curve assessment** — improving / stuck / collapsed, where the problems began.
3. **Identified issues** — prioritized list, each with: the problem, the evidence (which metric, what value), the suggested fix (specific parameter range).
4. **What looks healthy** — confirm what's working.
5. **Recommended next experiment** — the single most impactful change to try next.

Avoid: generic ranges without evidence; assertions like "v_loss is too high" without comparing to a healthy reference; recommendations that aren't tied to a specific config field.
