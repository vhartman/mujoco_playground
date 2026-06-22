"""Plot steady-state force error vs an unobserved cube latent (size or position),
per policy. Blind bundles dashed, force-informed solid; seeds aggregated to a mean
line + min-max band.

  --group=e2_pos      2x2 grid: rows = action_mode (delta/absolute),
                      cols = trained-fixed | trained-randomized. The E2 figure.
  --group=<single>    one panel from cubepinch_<latent>sweep_<group>.parquet.

    python experimentation/pinch_plotting/plot_latent_sweep.py --group=e2_pos
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "rollout_data"
GROUP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--group=")), "e2_pos")
FORCE_TOL = 0.75  # N

BLIND = ["none", "vel_only", "pos_only", "baseline"]
INFORMED = ["proprio.target", "proprio.delta", "force.magnitude"]
ORDER = BLIND + INFORMED
COLORS = {b: c for b, c in zip(ORDER, plt.cm.tab10(np.linspace(0, 1, len(ORDER))))}


def plot_ax(ax, df, title, xlabel):
    bundles = sorted(df["sensor_bundle"].unique(),
                     key=lambda b: ORDER.index(b) if b in ORDER else 99)
    xs = np.array(sorted(df["latent"].unique()))
    for bundle in bundles:
        sub = df[df["sensor_bundle"] == bundle]
        per_seed = (sub.groupby(["seed", "latent"])["mae_force_error"]
                    .mean().unstack().reindex(columns=xs))
        mean = per_seed.mean(axis=0).to_numpy()
        lo, hi = per_seed.min(axis=0).to_numpy(), per_seed.max(axis=0).to_numpy()
        blind = bundle in BLIND
        ax.plot(xs, mean, marker="o", lw=2, color=COLORS.get(bundle),
                ls="--" if blind else "-", label=f"{bundle}{' (blind)' if blind else ''}")
        if per_seed.shape[0] > 1:
            ax.fill_between(xs, lo, hi, color=COLORS.get(bundle), alpha=0.12)
    ax.axhline(FORCE_TOL, color="grey", ls=":", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def load(group, latent):
    p = DATA_DIR / f"cubepinch_{latent}sweep_{group}.parquet"
    if not p.exists():
        sys.exit(f"missing {p} -- run eval_latent_sweep.py --group={group} --latent={latent} first")
    return pd.read_parquet(p)


# compound group -> (fixed_group, rand_group, latent, xlabel, title, out_png)
COMPOUND = {
    "e2_pos": ("e2_pos_fixed", "e2_pos_rand", "pos", "cube x-offset (m)",
               "E2 cube-position DR", "force_error_vs_pos_e2.png"),
    "e1_size": ("e1_size_fixed", "e1_size_rand", "size", "cube size scale",
                "E1 cube-size DR", "force_error_vs_size_e1.png"),
    "e3_dp": ("e3_dp_fixed", "e3_dp_rand", "size", "cube size scale",
              "E3 delta-pose DR", "force_error_vs_size_e3_dp.png"),
}


def main():
    if GROUP in COMPOUND:
        fg, rg, lat, xlabel, title, outname = COMPOUND[GROUP]
        fixed, rand = load(fg, lat), load(rg, lat)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
        for r, am in enumerate(["delta", "absolute"]):
            plot_ax(axes[r][0], fixed[fixed.action_mode == am], f"{am} | trained FIXED", xlabel)
            plot_ax(axes[r][1], rand[rand.action_mode == am], f"{am} | trained RANDOMIZED", xlabel)
            axes[r][0].set_ylabel("hold |force_error| (N)")
        axes[0][1].legend(fontsize=8, ncol=2, loc="upper center")
        fig.suptitle(f"{title}: force error vs unobserved {lat}\n"
                     "blind (dashed) vs force-informed (solid) — randomized column is the test",
                     fontsize=13)
        out = DATA_DIR / outname
    else:
        latent = "pos" if "pos" in GROUP else "size"
        df = load(GROUP, latent)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_ax(ax, df, GROUP, "cube latent")
        ax.set_ylabel("hold |force_error| (N)")
        ax.legend(fontsize=8, ncol=2)
        out = DATA_DIR / f"force_error_vs_{latent}_{GROUP}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
