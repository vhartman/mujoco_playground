"""Static probe: measure how much the observed joint angles q move per Newton
of contact force in the pinch task, so we can size joint-position observation
noise that blurs the force signal without destroying posture.

Mechanism being characterised: the cube is welded, so at equilibrium the contact
force is a deterministic function of q (force = g(q)). The baseline policy reads
force off q. If we add Gaussian noise to observed q with std sigma, the force is
only legible to the extent that the equilibrium q-shift per Newton, ||dq/dF||,
exceeds sigma. This script measures ||dq/dF|| over the 5-15 N target range.

Method (no contact-direction guessing, no training):
  1. Gradient-descend the motor target onto the cube: at each iteration recompute
     the kinematic gradient of the sum of fingertip-to-cube-centre distances
     w.r.t. q (re-aimed every step, so the fingers converge on the cube instead
     of curling past it), step the target inward, and settle the plain-MuJoCo sim
     to equilibrium. Once the tips are on the faces, pressing further toward the
     (interior) centre raises the contact force monotonically.
  2. Log (q, contact force) along the press.
  3. From the in-contact samples, report dq/dF (Euclidean over the 8 active
     joints) and the sigma that blurs the +/-force_tolerance success band.

Usage:
    python learning/probe_pinch_force_q.py
    python learning/probe_pinch_force_q.py --width 1e-6   # probe at a stiffer contact
"""

import argparse

import mujoco
import numpy as np

from mujoco_playground._src.manipulation.tesollo_hand import pinch


def _sensor_slice(mjm, name):
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = mjm.sensor_adr[sid]
    return slice(adr, adr + mjm.sensor_dim[sid])


def _contact_force(mjm, mjd, sl):
    """Total contact-force magnitude, matching env._total_contact_force."""
    vals = mjd.sensordata[sl].reshape(-1, 3)
    return float(np.sum(np.linalg.norm(vals, axis=1)))


def _tip_dist_sum(mjm, mjd, tip_sids, cube_bid, q):
    """Sum of fingertip-to-cube-centre distances at configuration q (kinematic)."""
    mjd.qpos[:] = q
    mujoco.mj_forward(mjm, mjd)
    cube = mjd.xpos[cube_bid]
    return float(sum(np.linalg.norm(mjd.site_xpos[s] - cube) for s in tip_sids))


def _settle(mjm, mjd, ctrl, max_steps=600, vel_tol=1e-3):
    """Drive to `ctrl` and step until the hand joints stop moving."""
    mjd.ctrl[:] = ctrl
    for _ in range(max_steps):
        mujoco.mj_step(mjm, mjd)
        if np.linalg.norm(mjd.qvel) < vel_tol:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=None,
                    help="Override tip-geom solimp width (default: XML 1e-4).")
    ap.add_argument("--f-max", type=float, default=25.0,
                    help="Stop pressing once contact force exceeds this (N).")
    ap.add_argument("--ds", type=float, default=0.01,
                    help="Inward target step (rad) per iteration.")
    ap.add_argument("--max-iters", type=int, default=120,
                    help="Hard cap on press iterations.")
    ap.add_argument("--fit-lo", type=float, default=1.0,
                    help="Lower force bound (N) for the dq/dF fit (responsive regime).")
    ap.add_argument("--fit-hi", type=float, default=4.5,
                    help="Upper force bound (N) for the dq/dF fit (below saturation).")
    args = ap.parse_args()

    overrides = None
    if args.width is not None:
        overrides = {"tip_solimp.enable": True, "tip_solimp.width": args.width}
    env = pinch.CubePinch(config_overrides=overrides)
    mjm = env.mj_model
    mjd = mujoco.MjData(mjm)

    nu = mjm.nu
    assert mjm.nq == nu, f"probe assumes qpos aligns 1:1 with ctrl (nq={mjm.nq}, nu={nu})"
    lo = mjm.actuator_ctrlrange[:, 0].copy()
    hi = mjm.actuator_ctrlrange[:, 1].copy()

    cube_bid = mjm.body("cube").id
    tip_sids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f"rl_dg_{f}_tip_c")
                for f in (1, 2)]
    force_sl = _sensor_slice(mjm, "cube_force")

    home = np.array(mjm.keyframe("home").qpos, dtype=float)

    def _closing_grad(q):
        """Unit -gradient of sum(tip-cube-centre dist) w.r.t. q, evaluated at q."""
        eps = 1e-4
        g = np.zeros(nu)
        base = _tip_dist_sum(mjm, mjd, tip_sids, cube_bid, q)
        for i in range(nu):
            qp = q.copy(); qp[i] += eps
            g[i] = (_tip_dist_sum(mjm, mjd, tip_sids, cube_bid, qp) - base) / eps
        n = np.linalg.norm(g)
        return -g / n if n > 1e-12 else g

    # Press onto the cube, re-aiming the inward direction every iteration.
    width_str = f"{args.width:g}" if args.width is not None else "1e-4 (XML default)"
    print(f"\nProbing pinch contact: tip solimp width = {width_str}")
    print(f"{'iter':>5} {'F (N)':>9} {'|q-q_c| (rad)':>14}")

    mjd.qpos[:] = home
    mjd.qvel[:] = 0.0
    target = home.copy()
    _settle(mjm, mjd, target)

    samples = []  # (F, q)
    q_contact = None
    for it in range(args.max_iters):
        # mjd holds the current settled state; re-aim from there and press in.
        target = np.clip(target + args.ds * _closing_grad(mjd.qpos.copy()), lo, hi)
        mjd.qvel[:] = 0.0  # warm start from the current settled config
        _settle(mjm, mjd, target)
        F = _contact_force(mjm, mjd, force_sl)
        q = mjd.qpos.copy()
        if F > 0.1 and q_contact is None:
            q_contact = q
        dq = 0.0 if q_contact is None else float(np.linalg.norm(q - q_contact))
        if it % 5 == 0 or F > 0.1:
            print(f"{it:5d} {F:9.3f} {dq:14.5f}")
        samples.append((F, q))
        if F > args.f_max:
            break

    # 3) dq/dF in the *responsive* regime. Force saturates once the small finger
    #    actuators max out, so fit only where q still buys force: F in
    #    [fit_lo, fit_hi]. Use cumulative joint-space path length L(F) (the true
    #    rad traversed) and regress L vs F; the slope is ||dq/dF||.
    Fs = np.array([r[0] for r in samples])
    qs = np.array([r[1] for r in samples])
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(qs, axis=0), axis=1))])

    f_ceiling = float(Fs.max())
    band = (Fs >= args.fit_lo) & (Fs <= args.fit_hi)
    if band.sum() < 2:
        print(f"\n!! Fewer than 2 samples in F in [{args.fit_lo}, {args.fit_hi}] N "
              f"(force ceiling was {f_ceiling:.2f} N) — adjust --fit-lo/--fit-hi.")
        return
    slope, _ = np.polyfit(Fs[band], path[band], 1)  # rad per N
    dqdf = float(slope)

    ftol = float(env._config.force_tolerance)
    print("\n" + "=" * 64)
    print(f"static force ceiling reached  : {f_ceiling:.2f} N "
          f"(target range is {env._config.force_target_range})")
    print(f"||dq/dF|| fit over F in [{args.fit_lo:g}, {args.fit_hi:g}] N "
          f"({int(band.sum())} pts): {dqdf:.5f} rad/N")
    print(f"q-shift across tolerance band : {dqdf * ftol:.5f} rad  (+/-{ftol:g} N)")
    print("-" * 64)
    print("Recommended joint_pos noise std (obs_noise.scales.joint_pos):")
    print(f"  ~match tolerance band  sigma ~ {dqdf * ftol:.4f} rad   (force barely legible)")
    print(f"  blur a ~3 N span       sigma ~ {dqdf * 3.0:.4f} rad")
    print("=" * 64)


if __name__ == "__main__":
    main()
