"""PID_exploration.py

Evaluates PD controller quality across a grid of (kp, kd) gain values using
step-response tests on the Tesollo hand index finger (dg_2 — 4 DOFs).

The scene contains the hand only (no cube, no floor contacts), built in Python
by wrapping `tesollo_wrist_dof_clean.xml` in a minimal MJCF and adding
position actuators + a home keyframe.  Gravity is disabled so unactuated
fingers and the wrist do not drift while we measure the response.

Protocol
--------
For each (kp, kd) pair we:
  1. Build the hand model with those gains applied to the active actuators.
  2. Start from the home keyframe position.
  3. Command a fixed step target = home + 0.4 rad (clamped to joint limits) on
     the index-finger actuators only; the rest stay at their home ctrl.
  4. Hold that target for N_STEPS control steps (JAX jit + lax.scan).
  5. Record joint positions, velocities, and actuator forces for the active
     8 joints only.
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
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

# ---------------------------------------------------------------------------
# Scene paths
# ---------------------------------------------------------------------------
_XML_DIR = (
    PROJECT_ROOT
    / "mujoco_playground"
    / "_src"
    / "manipulation"
    / "tesollo_hand"
    / "xmls"
)

# Hand-only MJCF — wraps tesollo_wrist_dof_clean.xml (a bare <frame> fragment)
# in a complete model.  Actuators and keyframe values mirror those in
# scene_mjx_cube_pinch.xml, just without the cube.
_HAND_ONLY_XML = """<mujoco model="tesollo_hand_only">
  <include file="scene_base.xml"/>

  <worldbody>
    <frame pos="-0.18 0.04 0.084" quat="-1 1 -1 1">
      <include file="tesollo_wrist_dof_clean.xml"/>
    </frame>
  </worldbody>

  <actuator>
    <position kp="10" kv="2"  name="rj_wrist_1_1_a" joint="rj_wrist_1_1" ctrlrange="-0.3 0.3"/>
    <position kp="75" kv="10" name="rj_wrist_1_2_a" joint="rj_wrist_1_2" ctrlrange="-0.3 0.3"/>
    <position kp="10" kv="2"  name="rj_wrist_1_3_a" joint="rj_wrist_1_3" ctrlrange="-0.3 0.3"/>

    <position kp="3" name="rj_dg_1_1_a" joint="rj_dg_1_1" ctrlrange="-0.383972 0.890118"/>
    <position kp="3" name="rj_dg_1_2_a" joint="rj_dg_1_2" ctrlrange="-3.14159 0"/>
    <position kp="3" name="rj_dg_1_3_a" joint="rj_dg_1_3" ctrlrange="0 1.5708"/>
    <position kp="3" name="rj_dg_1_4_a" joint="rj_dg_1_4" ctrlrange="0 1.5708"/>

    <position kp="3" name="rj_dg_2_1_a" joint="rj_dg_2_1" ctrlrange="-0.418879 0.610865"/>
    <position kp="3" name="rj_dg_2_2_a" joint="rj_dg_2_2" ctrlrange="0 2.00713"/>
    <position kp="3" name="rj_dg_2_3_a" joint="rj_dg_2_3" ctrlrange="0 1.5708"/>
    <position kp="3" name="rj_dg_2_4_a" joint="rj_dg_2_4" ctrlrange="0 1.5708"/>

    <position kp="3" name="rj_dg_3_1_a" joint="rj_dg_3_1" ctrlrange="-0.610865 0.610865"/>
    <position kp="3" name="rj_dg_3_2_a" joint="rj_dg_3_2" ctrlrange="0 1.95477"/>
    <position kp="3" name="rj_dg_3_3_a" joint="rj_dg_3_3" ctrlrange="0 1.5708"/>
    <position kp="3" name="rj_dg_3_4_a" joint="rj_dg_3_4" ctrlrange="0 1.5708"/>

    <position kp="3" name="rj_dg_4_1_a" joint="rj_dg_4_1" ctrlrange="-0.610865 0.418879"/>
    <position kp="3" name="rj_dg_4_2_a" joint="rj_dg_4_2" ctrlrange="0 1.90241"/>
    <position kp="3" name="rj_dg_4_3_a" joint="rj_dg_4_3" ctrlrange="0 1.5708"/>
    <position kp="3" name="rj_dg_4_4_a" joint="rj_dg_4_4" ctrlrange="0 1.5708"/>

    <position kp="3" name="rj_dg_5_1_a" joint="rj_dg_5_1" ctrlrange="-0.0174533 1.0472"/>
    <position kp="3" name="rj_dg_5_2_a" joint="rj_dg_5_2" ctrlrange="-0.418879 0.610865"/>
    <position kp="3" name="rj_dg_5_3_a" joint="rj_dg_5_3" ctrlrange="0 1.5708"/>
    <position kp="3" name="rj_dg_5_4_a" joint="rj_dg_5_4" ctrlrange="0 1.5708"/>
  </actuator>

  <keyframe>
    <key name="home"
      qpos="
      0.072 0.036 0.3
      -0.167 -1.69 0.254 0.775
      -0.172 0.29 0.819 0.45
      -0.104 1.08 1.49 1.5
      0.000154 1.21 1.43 1.33
      0.00107 0.256 1.23 1.5"
      ctrl="
      0.072 0.036 0.3
      -0.167 -1.69 0.254 0.775
      -0.172 0.29 0.819 0.45
      -0.104 1.08 1.49 1.5
      0.000154 1.21 1.43 1.33
      0.00107 0.256 1.23 1.5"/>
  </keyframe>
</mujoco>
"""

# Active joints under PD test = index finger only (dg_2).  The thumb and other
# fingers stay at their home ctrl with nominal gains so they don't interfere.
_ACTIVE_JOINT_NAMES = [f"rj_dg_2_{i}" for i in range(1, 5)]
_N_ACTIVE = len(_ACTIVE_JOINT_NAMES)


def _build_hand_only_model() -> mujoco.MjModel:
    """Compile the hand-only MjModel.

    The XML is written to a temp file inside the xmls/ directory so the
    <include> directives for scene_base.xml and tesollo_wrist_dof_clean.xml
    resolve against the real assets (meshes etc.).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=str(_XML_DIR), delete=False
    ) as f:
        f.write(_HAND_ONLY_XML)
        tmp_path = f.name
    try:
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


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

# Physics substeps per control step (ctrl_dt = sim_dt * _N_SUBSTEPS).
_N_SUBSTEPS = 5
_CTRL_DT = 0.01 * _N_SUBSTEPS  # 0.05 s

# ---------------------------------------------------------------------------
# Singleton model + JIT-compiled rollout
# ---------------------------------------------------------------------------
_MJ_MODEL: mujoco.MjModel | None = None
_MJX_MODEL = None
_INIT_DATA = None
_INIT_QPOS = None          # full qpos at home
_TARGET_CTRL = None        # full ctrl with STEP_SIZE only on active actuators
_ACTIVE_DOF_IDS = None     # jp.ndarray of length 8 — indices into qpos/qvel
_ACTIVE_ACT_IDS = None     # jp.ndarray of length 8 — indices into ctrl


def _get_base_model():
    """Build/cache the hand-only model and resolve active joint/actuator IDs."""
    global _MJ_MODEL, _MJX_MODEL, _INIT_DATA, _INIT_QPOS, _TARGET_CTRL
    global _ACTIVE_DOF_IDS, _ACTIVE_ACT_IDS
    if _MJX_MODEL is not None:
        return _MJX_MODEL

    m = _build_hand_only_model()

    # No cube to push against — disable gravity so unactuated bodies stay put.
    m.opt.gravity[:] = 0.0

    # Zero passive joint dissipation so the response is pure PD.
    m.dof_damping[:] = 0.0
    m.dof_frictionloss[:] = 0.0
    m.dof_dampingpoly[:] = 0.0

    # Resolve active joint → dof index and active actuator index.
    active_dof = []
    active_act = []
    for jname in _ACTIVE_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        active_dof.append(int(m.jnt_dofadr[jid]))
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname + "_a")
        active_act.append(int(aid))
    _ACTIVE_DOF_IDS = jp.array(active_dof)
    _ACTIVE_ACT_IDS = jp.array(active_act)

    # GraphMode.NONE so model edits (gainprm/biasprm) take effect each step.
    _MJ_MODEL = m
    _MJX_MODEL = mjx.put_model(m, impl="warp", graph_mode=GraphMode.NONE)

    home_key = m.keyframe("home")
    _INIT_QPOS = jp.array(home_key.qpos)
    home_ctrl = jp.array(home_key.ctrl)

    # Clamp ctrl ranges only on the active actuators (others stay at home).
    ctrlrange = jp.array(m.actuator_ctrlrange)  # (nu, 2)
    lo = ctrlrange[_ACTIVE_ACT_IDS, 0]
    hi = ctrlrange[_ACTIVE_ACT_IDS, 1]
    stepped = jp.clip(home_ctrl[_ACTIVE_ACT_IDS] + STEP_SIZE, lo, hi)
    _TARGET_CTRL = home_ctrl.at[_ACTIVE_ACT_IDS].set(stepped)

    _INIT_DATA = mjx_env.make_data(
        m,
        qpos=_INIT_QPOS,
        ctrl=home_ctrl,
        impl="warp",
        nconmax=256,
        njmax=128,
    )
    return _MJX_MODEL


def _apply_gains(model, kp: float, kd: float):
    """Return a new MJX model with kp/kd applied to the active actuators only.

    Non-active actuators (wrist + dg_3/4/5) keep their nominal kp/kv so the
    rest of the hand stays put while we excite thumb + index.
    """
    kp_arr = jp.full((_N_ACTIVE,), float(kp))
    kd_arr = jp.full((_N_ACTIVE,), float(kd))
    gainprm = model.actuator_gainprm.at[_ACTIVE_ACT_IDS, 0].set(kp_arr)
    biasprm = (
        model.actuator_biasprm
        .at[_ACTIVE_ACT_IDS, 1].set(-kp_arr)
        .at[_ACTIVE_ACT_IDS, 2].set(-kd_arr)
    )
    return model.tree_replace({
        "actuator_gainprm": gainprm,
        "actuator_biasprm": biasprm,
    })


@jax.jit
def _rollout_jit(model, data, target_ctrl):
    """Single compiled rollout reused for every gain configuration."""
    def step_fn(data, _):
        data = mjx_env.step(model, data, target_ctrl, _N_SUBSTEPS)
        return data, (data.qpos, data.qvel, data.actuator_force)
    _, trajs = jax.lax.scan(step_fn, data, None, length=N_STEPS)
    return trajs


def run_step_response(kp: float, kd: float) -> dict:
    """Run a step-response rollout for the given gains and return trajectories
    sliced to the 8 active joints."""
    base = _get_base_model()
    model = _apply_gains(base, kp, kd)
    qpos_t, qvel_t, force_t = _rollout_jit(model, _INIT_DATA, _TARGET_CTRL)
    dof = np.array(_ACTIVE_DOF_IDS)
    act = np.array(_ACTIVE_ACT_IDS)
    return {
        "qpos":     np.array(qpos_t)[:, dof],
        "qvel":     np.array(qvel_t)[:, dof],
        "force":    np.array(force_t)[:, act],
        "init_pos": np.array(_INIT_QPOS)[dof],
        "target":   np.array(_TARGET_CTRL)[act],
        "ctrl_dt":  _CTRL_DT,
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
    """Produce all visualisation panels and save them to PNG."""

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
    plt.savefig(Path(__file__).parent / "PID_gain_sweep_heatmaps.png", dpi=300, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Figure 2: step-response curves for selected configurations
    # -----------------------------------------------------------------------
    valid_curves = [
        (kp, kd) for kp, kd in CURVE_CONFIGS
        if kp in kp_vals and kd in kd_vals
    ]
    joint_idx = 1  # representative joint (dg_2_2 — largest range, most informative)

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
        f"Step-Response Curves — joint {joint_idx} (dg_2_2)", fontsize=13
    )
    fig2.tight_layout()
    plt.savefig(Path(__file__).parent / "PID_gain_sweep_step_responses.png", dpi=300, bbox_inches="tight")
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

    plt.savefig(Path(__file__).parent / "PID_gain_sweep_scatter.png", dpi=300, bbox_inches="tight")
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
        print(f"Each rollout: {N_STEPS} steps × ctrl_dt={_CTRL_DT} s = {N_STEPS*_CTRL_DT:.1f} s sim\n")
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
