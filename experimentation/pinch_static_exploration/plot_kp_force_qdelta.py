"""3D plot of the pinch q-delta sweep: x=kp, y=force, z=q_delta.

z defaults to q_delta_from_contact_rad (||q(F) - q_at_zero_force_contact||, i.e.
how far the observed joint config has moved due to *squeezing*, measured from the
just-contact pose). Use --z to plot a different column, e.g. defl_l2_rad.

    python experimentation/plot_kp_force_qdelta.py
    python experimentation/plot_kp_force_qdelta.py --z defl_l2_rad
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "kp_force_qdelta_sweep.csv"))
    ap.add_argument("--z", default="q_delta_from_contact_rad")
    ap.add_argument("--out", default=os.path.join(HERE, "kp_force_qdelta_3d.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    kp = np.array([float(r["kp"]) for r in rows])
    F = np.array([float(r["force_N"]) for r in rows])
    Z = np.array([float(r[args.z]) for r in rows])

    fig = plt.figure(figsize=(13, 5.5))
    # Left: full range (shows the high-force grasp reconfiguration jump).
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_trisurf(kp, F, Z, cmap="viridis", alpha=0.75, linewidth=0.2)
    ax1.scatter(kp, F, Z, c="k", s=8)
    ax1.set_xlabel("kp (servo gain)"); ax1.set_ylabel("force (N)")
    ax1.set_zlabel(args.z); ax1.set_title("full range")
    ax1.view_init(elev=22, azim=-130)

    # Right: task-band zoom (F <= 5 N), where the policy actually operates.
    m = F <= 5.05
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if m.sum() >= 3:
        ax2.plot_trisurf(kp[m], F[m], Z[m], cmap="viridis", alpha=0.75, linewidth=0.2)
        ax2.scatter(kp[m], F[m], Z[m], c="k", s=8)
    ax2.set_xlabel("kp (servo gain)"); ax2.set_ylabel("force (N)")
    ax2.set_zlabel(args.z); ax2.set_title("task band (F <= 5 N)")
    ax2.view_init(elev=22, azim=-130)

    fig.suptitle(f"pinch q-delta sweep:  z = {args.z}", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
