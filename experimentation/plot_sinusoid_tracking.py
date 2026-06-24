"""Overlay actual vs target contact force for the sinusoid sweep.

For each chosen rollout index, a 2x2 figure shows the four bundles
(baseline, proprio.delta, proprio.target, force.magnitude) tracking the SAME
sinusoidal target (the eval phase is shared across bundles for a given rollout).
Several figures are produced for different rollout seeds/phases.
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/local/home/nstaykov/workspace/policy_analyzer/analysis/compare_cache"
OUTDIR = "/local/home/nstaykov/workspace/mujoco_playground/experimentation/sinusoid_tracking"
CTRL_DT = 0.05
TOL = 0.75

# (display name, dir token, seed) -- s3 is non-divergent for all four bundles
PANELS = [
    ("baseline",        "baseline",       "s3"),
    ("proprio.delta",   "propriodelta",   "s3"),
    ("proprio.target",  "propriotarget",  "s3"),
    ("force.magnitude", "forcemagnitude", "s3"),
]
COLORS = {
    "baseline": "#d62728", "proprio.delta": "#2ca02c",
    "proprio.target": "#1f77b4", "force.magnitude": "#9467bd",
}
ROLLOUTS = [0, 3, 11, 27, 42]   # different eval phases


def find_run(token, seed):
    g = glob.glob(f"{CACHE}/TesolloCubePinch-20260623-1[3456]*_sb_{token}_{seed}")
    if not g:
        raise FileNotFoundError(f"no cache for {token} {seed}")
    return g[0]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {}
    for name, token, seed in PANELS:
        z = np.load(find_run(token, seed) + "/det.npz", allow_pickle=True)
        data[name] = {"eff": z["effective_force"], "tgt": z["force_target"]}

    t = np.arange(80) * CTRL_DT
    for r in ROLLOUTS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        tgt = data["force.magnitude"]["tgt"][r]   # shared across bundles
        for ax, (name, _, _) in zip(axes.flat, PANELS):
            eff = data[name]["eff"][r]
            err = np.abs(eff - tgt)
            in_tol = (err[24:] < TOL).mean() * 100
            ax.fill_between(t, tgt - TOL, tgt + TOL, color="gray", alpha=0.18,
                            label=f"±{TOL} N tol")
            ax.plot(t, tgt, "--", color="black", lw=1.8, label="target")
            ax.plot(t, eff, "-", color=COLORS[name], lw=2.0, label="actual")
            ax.set_title(f"{name}   (mean|err|={err[24:].mean():.2f} N, "
                         f"{in_tol:.0f}% in-tol)", fontsize=11)
            ax.set_ylabel("contact force (N)")
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 6)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        for ax in axes[1]:
            ax.set_xlabel("time (s)")
        fig.suptitle(f"Sinusoid force tracking — rollout {r} "
                     f"(shared target, seed s3)", fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = f"{OUTDIR}/tracking_rollout{r:02d}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
