"""Plot reward component shapes for the pick_and_place environment."""

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mujoco_playground._src.manipulation.tesollo_hand.base_pick_and_place import (
    PickAndPlaceBase,
    default_config,
)

cfg = default_config()
SCALE = dict(cfg.reward_config.scales)
TARGET_RADIUS = cfg.target_radius
DT = cfg.ctrl_dt

# Vectorise scalar-input static methods over numpy arrays
r_cube_pos       = jax.vmap(lambda d: PickAndPlaceBase.r_cube_pos(d, TARGET_RADIUS))
r_fingertip_tip  = jax.vmap(PickAndPlaceBase.r_fingertip_pos_per_tip)
r_cube_ori       = jax.vmap(PickAndPlaceBase.r_cube_orientation)
r_cube_height    = jax.vmap(lambda z: PickAndPlaceBase.r_cube_height(z, INIT_Z, GOAL_Z))

# Velocity functions take an array of DOF velocities and return a sum;
# wrap so we can sweep a single uniform velocity value
r_joint_vel  = jax.vmap(lambda v: PickAndPlaceBase.r_joint_vel(jp.array([v])))
r_wrist_vel  = jax.vmap(lambda v: PickAndPlaceBase.r_wrist_vel(jp.array([v])))

# Geometry from keyframe / XML
INIT_Z = 0.05   # cube freejoint z in home keyframe
GOAL_Z = 0.11   # table_surface_z + cube_half_size

def weighted(raw, key):
    return np.asarray(raw) * SCALE[key] * DT

# ── colours ──────────────────────────────────────────────────────────────────
C_POS = "#4C8EDA"
C_ORI = "#E07B3F"
C_HGT = "#5BAD6F"
C_TIP = "#9B6DB5"
C_VEL = "#D94F4F"

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── cube_pos ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
x = jp.linspace(0, 0.30, 500)
y = weighted(r_cube_pos(x), "cube_pos")
ax.plot(np.asarray(x) * 100, y, color=C_POS, lw=2.5)
ax.axvspan(0, TARGET_RADIUS * 100, alpha=0.15, color=C_POS, label="target zone")
ax.set_xlabel("Cube–goal distance (cm)")
ax.set_ylabel("Reward per step")
ax.set_title("cube_pos")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── fingertip_pos ─────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
x = jp.linspace(0, 0.30, 500)
per_tip = r_fingertip_tip(x)
total   = per_tip * 5   # env sums over 5 fingertips
y_total = weighted(total, "fingertip_pos")
y_per   = weighted(per_tip, "fingertip_pos")
ax.plot(np.asarray(x) * 100, y_total, color=C_TIP, lw=2.5, label="sum (5 tips)")
ax.plot(np.asarray(x) * 100, y_per, color=C_TIP, lw=1.5, ls="--", alpha=0.6, label="per tip")
ax.axvspan(0, 3.5, alpha=0.15, color=C_TIP, label="contact zone")
ax.set_xlabel("Fingertip–cube distance (cm)")
ax.set_ylabel("Reward per step")
ax.set_title("fingertip_pos")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── cube_ori ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
x = jp.linspace(0, float(jp.pi), 500)
y = weighted(r_cube_ori(x), "cube_ori")
ax.plot(np.degrees(np.asarray(x)), y, color=C_ORI, lw=2.5)
ax.axvspan(0, np.degrees(0.087), alpha=0.15, color=C_ORI, label="≤5° tolerance")
ax.set_xlabel("Orientation error (°)")
ax.set_ylabel("Reward per step")
ax.set_title("cube_ori")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── cube_height ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
x = jp.linspace(INIT_Z - 0.01, GOAL_Z + 0.02, 500)
y = weighted(r_cube_height(x), "cube_height")
ax.plot(np.asarray(x) * 100, y, color=C_HGT, lw=2.5)
ax.axvline(INIT_Z * 100, color="gray", lw=1.2, ls=":", label="floor z")
ax.axvline(GOAL_Z * 100, color=C_HGT, lw=1.2, ls="--", label="table z (goal)")
ax.set_xlabel("Cube centre height (cm)")
ax.set_ylabel("Reward per step")
ax.set_title("cube_height")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── joint_vel ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
x = jp.linspace(0, 5.0, 500)
y = weighted(r_joint_vel(x), "joint_vel")
ax.plot(np.asarray(x), y, color=C_VEL, lw=2.5, label="per joint")
ax.fill_between(np.asarray(x), y, 0, alpha=0.12, color=C_VEL)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Joint velocity (rad/s)")
ax.set_ylabel("Reward per step")
ax.set_title("joint_vel  (penalty, summed over 26 joints)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── wrist_vel ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
x = jp.linspace(0, 3.0, 500)
y = weighted(r_wrist_vel(x), "wrist_vel")
ax.plot(np.asarray(x), y, color=C_VEL, lw=2.5, label="per wrist DOF")
ax.fill_between(np.asarray(x), y, 0, alpha=0.12, color=C_VEL)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Wrist velocity (rad/s)")
ax.set_ylabel("Reward per step")
ax.set_title("wrist_vel  (penalty, summed over 6 wrist DOFs)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Pick-and-Place reward components  (scale × dt applied)",
             fontsize=14, fontweight="bold", y=1.01)

out = "reward_shapes.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
