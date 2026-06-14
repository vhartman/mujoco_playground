"""Characterise the pinch force->joint-angle signal along two dimensions.

The cube is welded, so at equilibrium contact force is a deterministic function
of the joint configuration. At the joints force = kp*(ctrl - q), so the part of q
that encodes force is the servo deflection ||ctrl - q|| = (J^T F)/kp. We hold
different forces at *one fixed grasp* and measure how the observed joint config q
(and the deflection) shift with force, across:

  1. kp  (actuator stiffness): expect per-Newton q-shift ~ 1/kp (kp*slope ~ const).
  2. F   (held force) at fixed kp: expect the shift to grow then saturate as the
     contact/finger geometry runs out of force-producing range (static ceiling).

To make the kp comparison clean, the contact pose and press direction are
established ONCE, kinematically (kp-independent), and reused for every kp. For
each target force we bisect the press magnitude and settle plain-MuJoCo to that
force, so all samples are at the same grasp and evenly spaced in force.

Per-sample results -> kp_force_qdelta_sweep.csv in this folder.

Usage:
    python experimentation/kp_force_qdelta_sweep.py
    python experimentation/kp_force_qdelta_sweep.py --kps 3,12,48 --width 1e-6
"""

import argparse
import csv
import os

import mujoco
import numpy as np

from mujoco_playground._src.manipulation.tesollo_hand import pinch

HERE = os.path.dirname(os.path.abspath(__file__))


def _sensor_slice(mjm, name):
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = mjm.sensor_adr[sid]
    return slice(adr, adr + mjm.sensor_dim[sid])


def _contact_force(mjm, mjd, sl):
    vals = mjd.sensordata[sl].reshape(-1, 3)
    return float(np.sum(np.linalg.norm(vals, axis=1)))


def _tip_dist_sum(mjm, mjd, tip_sids, cube_bid, q):
    mjd.qpos[:] = q
    mujoco.mj_forward(mjm, mjd)
    cube = mjd.xpos[cube_bid]
    return float(sum(np.linalg.norm(mjd.site_xpos[s] - cube) for s in tip_sids))


def _closing_grad(mjm, mjd, tip_sids, cube_bid, q):
    eps = 1e-4
    g = np.zeros(mjm.nu)
    base = _tip_dist_sum(mjm, mjd, tip_sids, cube_bid, q)
    for i in range(mjm.nu):
        qp = q.copy(); qp[i] += eps
        g[i] = (_tip_dist_sum(mjm, mjd, tip_sids, cube_bid, qp) - base) / eps
    n = np.linalg.norm(g)
    return -g / n if n > 1e-12 else g


def kinematic_contact(mjm, ds=0.005, max_iters=400):
    """Find a kp-independent (zero-force) contact command + press direction by
    marching the configuration onto the cube using only forward kinematics."""
    d = mujoco.MjData(mjm)
    cube_bid = mjm.body("cube").id
    cube_gid = mjm.geom("cube").id
    tip_sids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f"rl_dg_{f}_tip_c")
                for f in (1, 2)]
    lo = mjm.actuator_ctrlrange[:, 0].copy()
    hi = mjm.actuator_ctrlrange[:, 1].copy()
    home = np.array(mjm.keyframe("home").qpos, dtype=float)
    target = home.copy()
    prev = target.copy()
    for _ in range(max_iters):
        d.qpos[:] = target
        mujoco.mj_forward(mjm, d)
        ncon_cube = sum(1 for i in range(d.ncon)
                        if d.contact[i].geom1 == cube_gid or d.contact[i].geom2 == cube_gid)
        if ncon_cube >= 1:
            break
        prev = target.copy()
        target = np.clip(target + ds * _closing_grad(mjm, d, tip_sids, cube_bid, target),
                         lo, hi)
    direction = _closing_grad(mjm, d, tip_sids, cube_bid, prev)
    return prev, direction  # last pre-contact command (~0 force), press direction


class Presser:
    def __init__(self, kp, width, contact_target, direction):
        overrides = None
        if width is not None:
            overrides = {"tip_solimp.enable": True, "tip_solimp.width": width}
        env = pinch.CubePinch(config_overrides=overrides)
        m = env.mj_model
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.m, self.d = m, mujoco.MjData(m)
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.force_sl = _sensor_slice(m, "cube_force")
        self.home = np.array(m.keyframe("home").qpos, dtype=float)
        self.contact_target = contact_target
        self.dir = direction

    def _settle(self, ctrl, max_steps=800, vel_tol=1e-3):
        self.d.ctrl[:] = ctrl
        for _ in range(max_steps):
            mujoco.mj_step(self.m, self.d)
            if not np.all(np.isfinite(self.d.qpos)):
                return False
            if np.linalg.norm(self.d.qvel) < vel_tol:
                break
        return True

    def at_press(self, s, warm=True):
        if not warm:
            self.d.qpos[:] = self.home
        ctrl = np.clip(self.contact_target + s * self.dir, self.lo, self.hi)
        self.d.qvel[:] = 0.0
        self._settle(ctrl)
        return _contact_force(self.m, self.d, self.force_sl), self.d.qpos.copy(), self.d.ctrl.copy()

    def ceiling(self, s_hi=2.0, n=24):
        return max(self.at_press(s, warm=False)[0] for s in np.linspace(0, s_hi, n))

    def solve_force(self, F_target, s_hi=2.0, tol=0.1, iters=34):
        Fhi, _, _ = self.at_press(s_hi, warm=False)
        if Fhi < F_target - tol:
            return None
        lo, hi, res = 0.0, s_hi, None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F, q, c = self.at_press(mid)
            res = (F, q, c)
            if abs(F - F_target) < tol:
                return res
            lo, hi = (mid, hi) if F < F_target else (lo, mid)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kps", type=str, default="3,6,12,24,48")
    ap.add_argument("--width", type=float, default=None,
                    help="Override tip-geom solimp width (default: XML 1e-4).")
    ap.add_argument("--forces", type=str,
                    default="0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9",
                    help="Target contact forces (N) to hold and sample.")
    ap.add_argument("--fit-lo", type=float, default=1.0)
    ap.add_argument("--fit-hi", type=float, default=4.5)
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "kp_force_qdelta_sweep.csv"))
    args = ap.parse_args()

    kps = [float(x) for x in args.kps.split(",") if x.strip()]
    targets = [float(x) for x in args.forces.split(",") if x.strip()]

    # Establish the shared, kp-independent contact pose + press direction once.
    overrides = ({"tip_solimp.enable": True, "tip_solimp.width": args.width}
                 if args.width is not None else None)
    ref_m = pinch.CubePinch(config_overrides=overrides).mj_model
    contact_target, direction = kinematic_contact(ref_m)
    print(f"\nShared kinematic contact established. Sweeping kp = {kps}  "
          f"(solimp width = {args.width if args.width is not None else '1e-4 XML default'})\n")

    rows, summary = [], []
    for kp in kps:
        p = Presser(kp, args.width, contact_target, direction)
        ceil = p.ceiling()
        # Reference poses for q_delta:
        #   q_home    = pre-squeeze rest pose (fingers open, no contact)
        #   q_contact = zero-force just-contact pose (s=0 along the press dir)
        q_home = p.home.copy()
        _, q_contact0, _ = p.at_press(0.0, warm=False)
        recs = []
        for Ft in targets:
            if Ft > ceil - 0.1:
                continue
            r = p.solve_force(Ft)
            if r is not None and np.all(np.isfinite(r[1])):
                recs.append(r)
        if len(recs) < 2:
            print(f"  kp={kp:6.1f}  too few samples (ceiling={ceil:.2f}N)")
            continue
        Fs = np.array([r[0] for r in recs])
        qs = np.array([r[1] for r in recs])
        cs = np.array([r[2] for r in recs])
        defl = np.linalg.norm(cs - qs, axis=1)
        q_from_contact = np.linalg.norm(qs - q_contact0, axis=1)
        q_from_home = np.linalg.norm(qs - q_home, axis=1)
        band = (Fs >= args.fit_lo) & (Fs <= args.fit_hi)
        if band.sum() >= 2:
            slopes = [np.polyfit(Fs[band], qs[band, j], 1)[0] for j in range(qs.shape[1])]
            dqdf = float(np.linalg.norm(slopes))
            ddefl = float(np.polyfit(Fs[band], defl[band], 1)[0])
        else:
            dqdf = ddefl = float("nan")
        f_sat = float("nan")
        if len(Fs) >= 4 and np.isfinite(ddefl) and ddefl > 0:
            for i in range(1, len(Fs) - 1):
                df = Fs[i + 1] - Fs[i - 1]
                if df <= 0:
                    continue
                local = (defl[i + 1] - defl[i - 1]) / df
                if local < 0.3 * ddefl and Fs[i] > args.fit_hi:
                    f_sat = float(Fs[i]); break
        for (F, q, c), dl, qc, qh in zip(recs, defl, q_from_contact, q_from_home):
            rows.append({
                "kp": kp, "force_N": F, "defl_l2_rad": float(dl),
                "q_delta_from_contact_rad": float(qc),
                "q_delta_from_home_rad": float(qh),
                **{f"q{i}": float(q[i]) for i in range(len(q))},
                **{f"ctrl{i}": float(c[i]) for i in range(len(c))},
            })
        summary.append((kp, ceil, dqdf, ddefl, f_sat))
        print(f"  kp={kp:6.1f}  F_ceiling={ceil:6.2f}N  d(defl)/dF={ddefl:.5f} rad/N  "
              f"kp*d(defl)/dF={kp*ddefl:6.3f}  dq/dF={dqdf:.5f}  "
              f"sat_F={f_sat if np.isfinite(f_sat) else float('nan'):6.2f}N")

    fieldnames = (["kp", "force_N", "defl_l2_rad",
                   "q_delta_from_contact_rad", "q_delta_from_home_rad"]
                  + [f"q{i}" for i in range(8)] + [f"ctrl{i}" for i in range(8)])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 80)
    print(f"wrote {len(rows)} samples to {args.out}")
    print("-" * 80)
    print(f"{'kp':>7} {'F_ceil(N)':>10} {'d(defl)/dF':>12} {'kp*d(defl)/dF':>14} "
          f"{'dq/dF':>9} {'sat_F(N)':>9}")
    for kp, ceil, dqdf, dd, fsat in summary:
        print(f"{kp:7.1f} {ceil:10.2f} {dd:12.5f} {kp*dd:14.4f} {dqdf:9.5f} "
              f"{fsat if np.isfinite(fsat) else float('nan'):9.2f}")
    print("=" * 80)

    # Clean cross-sections straight from the written CSV (robust to fit noise):
    # (a) deflection at a fixed force vs kp -> the 1/kp servo signal (proprio sees
    #     this via motor_targets). (b) observed q-shift vs force at the lowest kp
    #     -> the small, kp-robust signal baseline must read, plus its saturation.
    from collections import defaultdict
    grp = defaultdict(list)
    for r in csv.DictReader(open(args.out)):
        grp[float(r["kp"])].append({k: float(v) for k, v in r.items()})
    print("\n(a) servo deflection ||ctrl-q|| at fixed force vs kp  (expect ~1/kp):")
    for F in (2.0, 4.0):
        cells = []
        for kp in sorted(grp):
            r = min(grp[kp], key=lambda r: abs(r["force_N"] - F))
            cells.append(f"kp{kp:.0f}:{r['defl_l2_rad']:.4f}")
        print(f"      @~{F:.0f}N  " + "  ".join(cells))
    if grp:
        kp0 = min(grp)
        print(f"(b) observed q-shift ||q-q_contact|| vs force at kp={kp0:.0f} (saturation):")
        for r in sorted(grp[kp0], key=lambda r: r["force_N"]):
            print(f"      F={r['force_N']:5.2f}N  q_shift={r['q_delta_from_contact_rad']:.4f} rad")
    print("=" * 80)


if __name__ == "__main__":
    main()
