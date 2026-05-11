# Eval Runs

Run deterministic evaluation rollouts for one or more training runs and save videos named after each run's suffix.

## Arguments

`$ARGUMENTS` can be:
- Empty — evaluate the most recent run with checkpoints in `logs/`
- A number, e.g. `4` — evaluate the last N runs with checkpoints
- A suffix or partial name, e.g. `soft_pose` or `kl_fix` — all runs whose log dir name contains that string
- A space-separated list of tokens, e.g. `soft_pose wide_tol`

## Instructions

Run this single command from the project root:

```bash
MUJOCO_GL=egl .venv/bin/python learning/eval_runs.py $ARGUMENTS
```

The script handles run discovery, evaluation, and renaming. Videos are saved as `<suffix>.mp4` in the project root. Report the summary table printed by the script, and note any failed runs.
