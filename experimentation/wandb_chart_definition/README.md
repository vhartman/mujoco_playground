# W&B custom-chart presets

These Vega-Lite specs back the custom charts used for diagnostic panels.

## `policy_dist_candle.json`

Renders an evolving distribution as a candle/band plot:
- min–max whiskers (thin rules)
- p25–p75 filled band
- p50 solid line (median)
- mean dashed line (orange)

Used for the policy mean (`policy_dist_loc`) and policy std
(`policy_dist_std`). When median and mean diverge, the distribution is
skewed — useful diagnostic.

### Data flow

The PPO loss patch in `learning/brax_compat.py` logs six scalars per
quantity per training step:

```
training/policy_dist_loc/{p25, p50, p75, mean, min, max}
training/policy_dist_std/{p25, p50, p75, mean, min, max}
```

These accumulate as normal scalar histories in W&B — no `wandb.Table`,
no per-step re-uploads. The custom chart reads from each run's scalar
history.

### One-time setup in the W&B UI

1. Open the project hosting your runs.
2. In a workspace panel grid → **Add panel** → **Custom Chart**.
3. Set the data source to the current run / run set (it defaults to
   reading scalar history).
4. Paste the contents of `policy_dist_candle.json` into the Vega-Lite
   spec editor.
5. Bind the chart's fields:
   - `step` → `_step` (or whichever step axis you use)
   - `p25`, `p50`, `p75`, `mean`, `min`, `max` → the corresponding
     `training/policy_dist_loc/...` keys
   - `title` → e.g. `"policy_dist_loc"`
6. Save. Duplicate the panel and rebind to `training/policy_dist_std/...`
   for the second chart.

You can save the configured panel as a project-level preset so it
auto-applies to new runs.

### Fallback without the custom chart

The six scalars per quantity also auto-overlay in W&B's default panel
grouping (everything under `training/policy_dist_loc/` collapses into
one line panel), so distribution evolution is already visible without
any UI setup — the custom chart is just a nicer rendering of the same
data.

---

## `mean_std_band.json`

Mean ± std shaded band for any metric that has a `{key}` + `{key}_std`
scalar pair. A single blue line with a translucent ±std band around it.

Typical use: `eval/episode_reward` (mean) + `eval/episode_reward_std`.

### Data flow

No code changes needed — brax's built-in eval loop emits
`eval/episode_reward` and `eval/episode_reward_std` as scalars at each
evaluation step.

### One-time setup in the W&B UI

1. **Add panel → Custom Chart**, paste `mean_std_band.json`.
2. Bind fields:
   - `step` → `_step`
   - `mean` → `eval/episode_reward`
   - `std` → `eval/episode_reward_std`
   - `title` → `"eval episode reward"`
3. Save / duplicate for other mean+std pairs.

---

## `reward_contribution_area.json`

Stacked area chart showing how much each reward term contributed per
evaluation checkpoint. Positive terms stack upward from zero; negative
terms (penalties) stack downward.

Supports up to six terms (`term1`–`term6`). Leave unused term fields
unbound — they are filtered out by the `null` check in the transform.

### Data flow

Each env's `step()` logs raw (unscaled) per-term rewards under
`eval/episode_metrics/reward/{term_name}`. To show true contributions
(accounting for scale) bind the **eval** metrics; to show raw
maximization progress bind the same keys directly.

### One-time setup in the W&B UI

1. **Add panel → Custom Chart**, paste `reward_contribution_area.json`.
2. Bind fields:
   - `step` → `_step`
   - `term1` → `eval/episode_metrics/reward/cube_ori` (or whichever terms apply)
   - `term2`, `term3`, … → remaining active reward terms
   - `title` → `"reward contribution"`
3. Save as a project preset to reuse across runs.
