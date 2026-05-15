"""PID_exploration.py

Evaluates PD controller quality across a grid of (kp, kd) gain values using
step-response tests on the Tesollo hand pinch joints (thumb + index, 8 DOFs).

Protocol
--------
For each (kp, kd) pair we:
  1. Build CubePinchProprio with those gains applied via _apply_pid_gains.
  2. Start from the home keyframe position.
  3. Command a fixed step target = home + 0.4 rad (clamped to joint limits).
  4. Hold that target for N_STEPS control steps (JAX jit + lax.scan).
  5. Record joint positions, velocities, and actuator forces.
  6. Compute the statistics below from the trajectory.

Statistics
----------
  rise_time_steps / rise_time_s
      Control steps / seconds until every joint first crosses 90 % of its
      commanded step amplitude.  Low = responsive.

  overshoot_pct
      Mean per-joint peak overshoot as a percentage of the step amplitude.
      Positive values only (undershoot is not penalised here).
      High overshoot → too much Kp or too little Kd.

  settling_time_steps / settling_time_s
      Control steps / seconds until the trajectory stays within ±2 % of the
      target indefinitely.  High Kp + low Kd gives long settling.

  steady_state_error_rad
      Mean |qpos − target| over the last 20 % of the episode.
      Non-zero → friction / damping overcomes the proportional gain.

  oscillation_amplitude_rad
      Mean per-joint std(qpos) over the last 30 % of the episode.
      Captures residual chatter — critical for stable contact tasks.

  joint_vel_rms
      RMS joint velocity over the whole episode.  High = jittery dynamics,
      bad for learning and for hardware.

  peak_joint_vel
      Max |qvel| seen during the episode.  Flags explosive gain combinations.

  actuator_force_rms
      RMS actuator force (Nm).  Proxy for control effort.

  energy_proxy
      Mean |qvel * force| over the episode (mechanical power, W).  Captures
      efficiency; high values mean wasted energy and thermal load.

  nan_detected
      1 if any NaN appears in qpos or qvel.  Indicates numerical blow-up
      (Kp far too high for the chosen sim_dt).

Usage
-----
  python experimentation/PID_exploration.py          # sweep + CSV
  python experimentation/PID_exploration.py --plot   # sweep + CSV + figures
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx.warp.types import GraphMode
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.tesollo_hand.base_pinch import (
    _N_ACTIVE,
    default_config,
)
from mujoco_playground._src.manipulation.tesollo_hand.pinch import CubePinchProprio

# ---------------------------------------------------------------------------
# Gain grid
# ---------------------------------------------------------------------------
KP_VALUES = list(np.linspace(1.0, 20.0, 13))
KD_VALUES = list(np.linspace(0.0, 5.0, 11))

# Rollout length (control steps).  ctrl_dt=0.05 → 10 s of simulation.
N_STEPS = 200

# Step size commanded to each active joint (rad).
STEP_SIZE = 0.4

CSV_PATH = Path(__file__).parent / "PID_test_results.csv"

# Number of physics substeps per control step (ctrl_dt=0.05 / sim_dt=0.01).
_N_SUBSTEPS = 5


# ---------------------------------------------------------------------------
# Singleton env + single JIT-compiled rollout
# ---------------------------------------------------------------------------
# Build the env once (XML parsing + model upload is expensive).  For each
# (kp, kd) pair we cheaply replace gainprm/biasprm on the already-uploaded
# MJX model rather than reconstructing everything from scratch.  The rollout
# is compiled once as a module-level JIT; subsequent calls with different
# model weights reuse the same XLA computation — no new CUDA graph per step.

_BASE_ENV: CubePinchProprio | None = None
_INIT_QPOS = None
_TARGET_CTRL = None
_INIT_DATA = None


def _get_base_env() -> CubePinchProprio:
    global _BASE_ENV, _INIT_QPOS, _TARGET_CTRL, _INIT_DATA
    if _BASE_ENV is not None:
        return _BASE_ENV

    cfg = default_config()
    # Keep warp impl (jax doesn't support contact sensors).  The module-level
    # _rollout_jit means mjx.put_model is called only once — no new CUDA graphs.
    _BASE_ENV = CubePinchProprio(config=cfg)

    # Zero passive damping and rebuild the MJX model with GraphMode.NONE.
    # In the default WARP graph mode, the CUDA graph captures GPU buffer
    # pointers on first call — later model changes are ignored. NONE disables
    # graph capture so model fields are read dynamically each kernel launch.
    _BASE_ENV.mj_model.dof_damping[:] = 0.0
    _BASE_ENV.mj_model.dof_frictionloss[:] = 0.0
    _BASE_ENV.mj_model.dof_dampingpoly[:] = 0.0
    _BASE_ENV._mjx_model = mjx.put_model(
        _BASE_ENV.mj_model, impl=cfg.impl, graph_mode=GraphMode.NONE
    )

    home_key = _BASE_ENV.mj_model.keyframe("home")
    _INIT_QPOS   = jp.array(home_key.qpos)
    _TARGET_CTRL = jp.clip(_INIT_QPOS + STEP_SIZE, _BASE_ENV._lowers, _BASE_ENV._uppers)
    _INIT_DATA   = mjx_env.make_data(
        _BASE_ENV.mj_model,
        qpos=_INIT_QPOS,
        ctrl=jp.array(home_key.ctrl),
        impl="warp",
        nconmax=256,
        njmax=128,
    )
    return _BASE_ENV


def _apply_gains(model, kp: float, kd: float):
    """Return a new MJX model with all actuator gains overwritten and joint damping zeroed."""
    kp_arr = jp.full((model.nu,), float(kp))
    kd_arr = jp.full((model.nu,), float(kd))
    gainprm = model.actuator_gainprm.at[:, 0].set(kp_arr)
    biasprm = (
        model.actuator_biasprm
        .at[:, 1].set(-kp_arr)
        .at[:, 2].set(-kd_arr)
    )
    return model.tree_replace({"actuator_gainprm": gainprm, "actuator_biasprm": biasprm})


@jax.jit
def _rollout_jit(model, data, target_ctrl):
    """Single compiled rollout reused for every gain configuration."""
    def step_fn(data, _):
        data = mjx_env.step(model, data, target_ctrl, _N_SUBSTEPS)
        return data, (data.qpos, data.qvel, data.actuator_force)
    _, trajs = jax.lax.scan(step_fn, data, None, length=N_STEPS)
    return trajs


def run_step_response(kp: float, kd: float) -> dict:
    """Run a step-response rollout for the given gains and return trajectories."""
    env = _get_base_env()
    model = _apply_gains(env.mjx_model, kp, kd)
    qpos_t, qvel_t, force_t = _rollout_jit(model, _INIT_DATA, _TARGET_CTRL)
    return {
        "qpos":     np.array(qpos_t),
        "qvel":     np.array(qvel_t),
        "force":    np.array(force_t),
        "init_pos": np.array(_INIT_QPOS),
        "target":   np.array(_TARGET_CTRL),
        "ctrl_dt":  env.dt,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(traj: dict) -> dict:
    qpos  = traj["qpos"]    # (T, 8)
    qvel  = traj["qvel"]    # (T, 8)
    force = traj["force"]   # (T, 8)
    init  = traj["init_pos"]
    target = traj["target"]
    ctrl_dt = traj["ctrl_dt"]
    T = qpos.shape[0]

    amplitude = target - init  # (8,) — always positive (step_size > 0, clipped)
    # Avoid division by zero for joints already at limit (amplitude ~ 0).
    safe_amp = np.where(np.abs(amplitude) < 1e-6, 1.0, amplitude)

    # Normalised trajectory: 0 = start, 1 = target.
    rel_pos = (qpos - init[None, :]) / safe_amp[None, :]  # (T, 8)
    error   = qpos - target[None, :]                       # (T, 8)

    # --- Rise time: first step where rel_pos >= 0.9 for every joint ----------
    above_90 = rel_pos >= 0.9                              # (T, 8)
    first_cross = np.argmax(above_90, axis=0)              # (8,)
    never = ~np.any(above_90, axis=0)
    rise_steps = np.where(never, T, first_cross).astype(float)

    # --- Overshoot -----------------------------------------------------------
    overshoot_norm = np.maximum(0.0, rel_pos - 1.0)        # only positive
    overshoot_pct = float(np.mean(np.max(overshoot_norm, axis=0)) * 100)

    # --- Settling time: last step outside ±2 % band -------------------------
    outside = np.abs(rel_pos - 1.0) > 0.02                 # (T, 8)
    # Reverse-argmax gives first violation from the end → settling index.
    rev_outside = outside[::-1]
    last_outside_rev = np.argmax(rev_outside, axis=0)      # (8,)
    never_settled = ~np.any(~outside, axis=0)
    # Convert from reversed index back to forward index.
    settling_steps = np.where(
        never_settled,
        float(T),
        (T - 1 - last_outside_rev).astype(float),
    )

    # --- Tail windows --------------------------------------------------------
    tail_20 = max(1, int(T * 0.20))
    tail_30 = max(1, int(T * 0.30))

    # --- Energy proxy --------------------------------------------------------
    energy = np.abs(qvel * force)  # element-wise mechanical power

    stats = {
        "rise_time_steps":         float(np.mean(rise_steps)),
        "rise_time_s":             float(np.mean(rise_steps)) * ctrl_dt,
        "overshoot_pct":           overshoot_pct,
        "settling_time_steps":     float(np.mean(settling_steps)),
        "settling_time_s":         float(np.mean(settling_steps)) * ctrl_dt,
        "steady_state_error_rad":  float(np.mean(np.abs(error[-tail_20:]))),
        "oscillation_amplitude_rad": float(np.mean(np.std(qpos[-tail_30:], axis=0))),
        "joint_vel_rms":           float(np.sqrt(np.mean(qvel ** 2))),
        "peak_joint_vel":          float(np.max(np.abs(qvel))),
        "actuator_force_rms":      float(np.sqrt(np.mean(force ** 2))),
        "energy_proxy":            float(np.mean(energy)),
        "nan_detected":            int(np.any(np.isnan(qpos)) or np.any(np.isnan(qvel))),
    }
    return stats


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> pd.DataFrame:
    rows = []
    total = len(KP_VALUES) * len(KD_VALUES)
    i = 0
    for kp in KP_VALUES:
        for kd in KD_VALUES:
            i += 1
            t0 = time.time()
            print(f"  [{i:2d}/{total}]  kp={kp:5.1f}  kd={kd:4.1f} ...", end="", flush=True)
            traj = run_step_response(kp, kd)
            stats = compute_statistics(traj)
            stats["kp"] = kp
            stats["kd"] = kd
            rows.append(stats)
            print(f"  done ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    # Reorder: kp, kd first.
    cols = ["kp", "kd"] + [c for c in df.columns if c not in ("kp", "kd")]
    return df[cols]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

STAT_LABELS = {
    "rise_time_s":               "Rise time (s)",
    "overshoot_pct":             "Overshoot (%)",
    "settling_time_s":           "Settling time (s)",
    "steady_state_error_rad":    "Steady-state error (rad)",
    "oscillation_amplitude_rad": "Oscillation amplitude (rad)",
    "joint_vel_rms":             "Joint velocity RMS (rad/s)",
    "peak_joint_vel":            "Peak joint velocity (rad/s)",
    "actuator_force_rms":        "Actuator force RMS (Nm)",
    "energy_proxy":              "Energy proxy (W)",
    "nan_detected":              "NaN detected",
}

HEATMAP_STATS = [
    "rise_time_s",
    "overshoot_pct",
    "settling_time_s",
    "steady_state_error_rad",
    "oscillation_amplitude_rad",
    "joint_vel_rms",
    "energy_proxy",
    "nan_detected",
]

# Gain pairs selected for step-response curves (must exist in the grid).
CURVE_CONFIGS = [
    (1.0, 0.0),
    (3.0, 0.0),
    (3.0, 2.0),
    (8.0, 2.0),
    (12.0, 0.0),
    (12.0, 3.0),
    (20.0, 5.0),
]


def visualize(df: pd.DataFrame) -> None:
    """Produce all visualisation panels and display them."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    kp_vals = sorted(df["kp"].unique())
    kd_vals = sorted(df["kd"].unique())

    # -----------------------------------------------------------------------
    # Figure 1: heatmap grid of all statistics
    # -----------------------------------------------------------------------
    n_stats = len(HEATMAP_STATS)
    ncols = 4
    nrows = (n_stats + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for ax, stat in zip(axes, HEATMAP_STATS):
        grid = np.full((len(kd_vals), len(kp_vals)), np.nan)
        for _, row in df.iterrows():
            ki = kp_vals.index(row["kp"])
            di = kd_vals.index(row["kd"])
            grid[di, ki] = row[stat]

        cmap = "RdYlGn_r" if stat != "nan_detected" else "RdYlGn"
        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks(range(len(kp_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in kp_vals], rotation=90)
        ax.set_yticks(range(len(kd_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in kd_vals])
        ax.set_xlabel("kp")
        ax.set_ylabel("kd")
        ax.set_title(STAT_LABELS[stat], fontsize=10)

        # Annotate cells with values.
        for di in range(len(kd_vals)):
            for ki in range(len(kp_vals)):
                val = grid[di, ki]
                if not np.isnan(val):
                    txt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                    ax.text(ki, di, txt, ha="center", va="center", fontsize=7,
                            color="black")

    # Hide unused axes.
    for ax in axes[n_stats:]:
        ax.set_visible(False)

    fig1.suptitle("PD Gain Sweep — Statistics Heatmaps", fontsize=14, y=1.01)
    fig1.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 2: step-response curves for selected configurations
    # -----------------------------------------------------------------------
    valid_curves = [
        (kp, kd) for kp, kd in CURVE_CONFIGS
        if kp in kp_vals and kd in kd_vals
    ]
    n_joints = _N_ACTIVE
    joint_idx = 1  # representative joint (dg_1_2 — largest range, most informative)

    fig2, axes2 = plt.subplots(
        len(valid_curves), 1, figsize=(10, 3 * len(valid_curves)), sharex=True
    )
    if len(valid_curves) == 1:
        axes2 = [axes2]

    for ax, (kp, kd) in zip(axes2, valid_curves):
        traj = run_step_response(kp, kd)
        t = np.arange(N_STEPS) * traj["ctrl_dt"]
        # Normalised position for the representative joint.
        init  = traj["init_pos"][joint_idx]
        tgt   = traj["target"][joint_idx]
        amp   = tgt - init if abs(tgt - init) > 1e-6 else 1.0
        q_norm = (traj["qpos"][:, joint_idx] - init) / amp

        ax.axhline(1.0, color="green",  lw=1.2, ls="--", label="Target")
        ax.axhline(0.9, color="orange", lw=0.8, ls=":",  label="90 % (rise)")
        ax.axhline(1.02, color="gray",  lw=0.6, ls=":",  label="±2 % band")
        ax.axhline(0.98, color="gray",  lw=0.6, ls=":")
        ax.plot(t, q_norm, color="steelblue", lw=1.5, label=f"kp={kp}, kd={kd}")
        ax.set_ylabel("Normalised position")
        ax.set_title(f"kp = {kp}, kd = {kd}", fontsize=10)
        ax.set_ylim(-0.1, 1.6)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle(
        f"Step-Response Curves — joint {joint_idx} (dg_1_2)", fontsize=13
    )
    fig2.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 3: composite scatter — rise time vs. overshoot, sized by energy
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    scatter_kp = df["kp"].values
    norm_kp = mcolors.Normalize(vmin=min(kp_vals), vmax=max(kp_vals))
    cmap_kp = plt.cm.viridis

    sc = ax3.scatter(
        df["rise_time_s"],
        df["overshoot_pct"],
        s=np.clip(df["energy_proxy"] * 200 + 30, 30, 600),
        c=scatter_kp,
        cmap=cmap_kp,
        norm=norm_kp,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax3, label="kp")

    for _, row in df.iterrows():
        ax3.annotate(
            f"kd={row['kd']:.1f}",
            (row["rise_time_s"], row["overshoot_pct"]),
            fontsize=6, textcoords="offset points", xytext=(4, 2),
        )

    ax3.set_xlabel("Rise time (s)")
    ax3.set_ylabel("Overshoot (%)")
    ax3.set_title(
        "Rise time vs. Overshoot  (bubble size ∝ energy, colour = kp)", fontsize=11
    )
    ax3.grid(alpha=0.3)
    fig3.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true",
                        help="Show visualisation after writing the CSV.")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip sweep and only plot from an existing CSV.")
    args = parser.parse_args()

    if not args.no_sweep:
        print(f"Running PD gain sweep: {len(KP_VALUES)} kp × {len(KD_VALUES)} kd "
              f"= {len(KP_VALUES)*len(KD_VALUES)} configurations")
        print(f"Each rollout: {N_STEPS} steps × ctrl_dt=0.05 s = {N_STEPS*0.05:.1f} s sim\n")
        df = run_sweep()
        df.to_csv(CSV_PATH, index=False)
        print(f"\nResults saved to {CSV_PATH}")
        print(df.to_string(index=False))
    else:
        if not CSV_PATH.exists():
            print(f"No CSV found at {CSV_PATH}. Run without --no-sweep first.")
            sys.exit(1)
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from {CSV_PATH}")

    if args.plot:
        print("\nGenerating visualisations...")
        visualize(df)


if __name__ == "__main__":
    main()
