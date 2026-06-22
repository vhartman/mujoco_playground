"""Performance plot: success rate vs a swept force-error threshold, per policy.

For each policy, sweep the threshold and plot the fraction of rollouts that are
"successful" under a pluggable criterion. Seeds (same sensor_bundle, different run)
pool into a mean line + min-max band.

    success_fn(rollout, param) -> bool

where ``rollout`` maps every dataset column to its per-step array, so the predicate
can reach any metric/info channel. Default: |info.force_error| held < THRESH for
>= 10 consecutive steps (the 0.5 s env hold).

    python experimentation/pinch_plotting/plot_success_vs_threshold.py --group=pinch_perception
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "rollout_data"
GROUP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--group=")), "pinch_perception")


def max_consecutive_below(x: np.ndarray, thresh: float) -> int:
    below = x < thresh
    best = run = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)
    return best


_force_success = lambda r, p: max_consecutive_below(np.abs(r["info.force_error"]), p) >= 10

CONFIGS = {
    "pinch_perception": dict(
        stem="cubepinch_perception_rollouts",
        sweep=np.linspace(0.0, 3.0, 151), xlabel="force-error threshold (N)", xlim=3.0,
        ref=(0.75, "force tol (0.75 N)"), success_fn=_force_success,
        bundle_order=["baseline", "pos_only", "vel_only", "none"],
        title="CubePinch perception ablation: success rate vs force-error threshold",
    ),
    "pinch_ceiling": dict(
        stem="cubepinch_ceiling_rollouts",
        sweep=np.linspace(0.0, 3.0, 151), xlabel="force-error threshold (N)", xlim=3.0,
        ref=(0.75, "force tol (0.75 N)"), success_fn=_force_success,
        bundle_order=["baseline", "proprio.target", "proprio.delta", "force.magnitude"],
        title="CubePinch bundle ceiling: success rate vs force-error threshold",
    ),
}
CFG = CONFIGS[GROUP]


def seed_rate_curves(df, bundle):
    sub = df[df["sensor_bundle"] == bundle]
    curves = []
    for _, pg in sub.groupby("policy"):  # one policy == one seed run
        rollouts = [{c: g.sort_values("step")[c].to_numpy() for c in pg.columns}
                    for _, g in pg.groupby("rollout")]
        n = len(rollouts)
        curves.append(np.array([sum(CFG["success_fn"](r, p) for r in rollouts) / n
                                for p in CFG["sweep"]]))
    return np.array(curves)


def plot(ax, df, title):
    bundles = sorted(df["sensor_bundle"].unique(),
                     key=lambda b: CFG["bundle_order"].index(b) if b in CFG["bundle_order"] else 99)
    x = CFG["sweep"]
    for bundle in bundles:
        curves = seed_rate_curves(df, bundle) * 100.0
        line, = ax.plot(x, curves.mean(axis=0), label=f"{bundle} (n={curves.shape[0]} seeds)", linewidth=2)
        if curves.shape[0] > 1:
            ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=line.get_color(), alpha=0.15)
    ref_x, ref_lbl = CFG["ref"]
    ax.axvline(ref_x, color="grey", linestyle="--", linewidth=1, label=ref_lbl)
    ax.set_title(title)
    ax.set_xlabel(CFG["xlabel"]); ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100); ax.set_xlim(0, CFG["xlim"])
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8)


def main():
    panels = [("det", "Deterministic"), ("sto", "Stochastic")]
    avail = [(m, t) for m, t in panels if (DATA_DIR / f"{CFG['stem']}_{m}.parquet").exists()]
    if not avail:
        sys.exit(f"No parquet for stem '{CFG['stem']}' -- run collect_rollouts.py first.")
    fig, axes = plt.subplots(1, len(avail), figsize=(7 * len(avail), 5), squeeze=False)
    for ax, (mode, title) in zip(axes[0], avail):
        df = pd.read_parquet(DATA_DIR / f"{CFG['stem']}_{mode}.parquet")
        n = df.groupby(["policy", "rollout"]).ngroups // df["policy"].nunique()
        plot(ax, df, f"{title}  (n={n} rollouts/seed)")
    fig.suptitle(CFG["title"], fontsize=13)
    fig.tight_layout()
    out = DATA_DIR / f"success_vs_threshold_{GROUP}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
