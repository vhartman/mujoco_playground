"""Record a video of a fixed-kp pinch squeeze to sanity-check the q-delta probe.

Drives the two fingers from the open rest pose onto the welded cube and keeps
pressing (ramping the command along the kinematic closing direction), rendering
plain-MuJoCo with the contact force overlaid. Lets you visually confirm the
fingers press cleanly, then curl/roll on the faces as force approaches the
~9 N grasp ceiling.

    python experimentation/record_pinch_squeeze.py --kp 3
    python experimentation/record_pinch_squeeze.py --kp 48 --out experimentation/squeeze_kp48.mp4
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import imageio.v2 as imageio
import kp_force_qdelta_sweep as S
from mujoco_playground._src.manipulation.tesollo_hand import pinch

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from PIL import Image, ImageDraw, ImageFont

    def _load_font(size):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    _FONT_LG = _load_font(18)
    _PURPLE = (180, 50, 255)

    def _label(frame, text):
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text((8, 6), text, fill=_PURPLE, font=_FONT_LG)
        return np.asarray(im)

except Exception:
    def _label(frame, text):
        return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--s-max", type=float, default=1.2, help="max press magnitude (rad along dir).")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--hold", type=int, default=30, help="extra frames held at full squeeze.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "pinch_squeeze.mp4"))
    args = ap.parse_args()

    m0 = pinch.CubePinch().mj_model
    ct, dr = S.kinematic_contact(m0)
    p = S.Presser(args.kp, args.width, ct, dr)
    m, d = p.m, p.d
    f1_sl = S._sensor_slice(m, "rl_dg_1_tip_cube_force")
    f1b_sl = S._sensor_slice(m, "rl_dg_1_tip_2_cube_force")
    f2_sl = S._sensor_slice(m, "rl_dg_2_tip_cube_force")
    f2b_sl = S._sensor_slice(m, "rl_dg_2_tip_2_cube_force")

    W, H = 640, 480
    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.06]
    cam.distance, cam.azimuth, cam.elevation = 0.32, 140.0, -18.0

    # start at the open rest pose, settle
    d.qpos[:] = p.home; d.qvel[:] = 0.0
    p._settle(p.home)

    ss = np.concatenate([np.linspace(0.0, args.s_max, args.frames),
                         np.full(args.hold, args.s_max)])
    frames = []
    for s in ss:
        ctrl = np.clip(ct + s * dr, p.lo, p.hi)
        d.ctrl[:] = ctrl
        for _ in range(8):  # a few sim steps per frame -> smooth motion
            mujoco.mj_step(m, d)
        F1 = S._contact_force(m, d, f1_sl) + S._contact_force(m, d, f1b_sl)
        F2 = S._contact_force(m, d, f2_sl) + S._contact_force(m, d, f2b_sl)
        renderer.update_scene(d, camera=cam)
        frame = _label(renderer.render(),
                       f"kp={args.kp:g}  press={s:.2f}rad  F1={F1:.2f} N  F2={F2:.2f} N")
        frames.append(frame)

    F1_peak = S._contact_force(m, d, f1_sl) + S._contact_force(m, d, f1b_sl)
    F2_peak = S._contact_force(m, d, f2_sl) + S._contact_force(m, d, f2b_sl)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames, peak F1={F1_peak:.2f} N  F2={F2_peak:.2f} N)")


if __name__ == "__main__":
    main()
