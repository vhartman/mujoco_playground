"""Export interactive frontend artifacts for a rollout.

Artifacts land directly in run_dir (analysis/<suffix>/):
  index.html     interactive viewer (copied from frontend_template.html)
  data.json      schema output_groups + DOF arrays + meta
  frames/        frame_0000.png … frame_NNNN.png (one per timestep)

analysis/index.html is regenerated as a landing page listing all available runs.

Serve the whole analysis/ directory once:
  python -m experimentation.policy_analyzer --serve 8000
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

_TANH_SAT = 2.0
_TEMPLATE = Path(__file__).parent / "frontend_template.html"


def _render_frames(npz, mj_model, height: int = 360, width: int = 480) -> list:
    """Render all T frames from stored traj fields in the npz."""
    import mujoco
    from mujoco_playground._src.mjx_env import render_array

    T = int(npz["traj_qpos"].shape[0])

    class _RS:
        class _D:
            __slots__ = ("qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied")
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        def __init__(self, **kw):
            self.data = self._D(**kw)

    states = [
        _RS(
            qpos=npz["traj_qpos"][t],
            qvel=npz["traj_qvel"][t],
            mocap_pos=npz["traj_mocap_pos"][t],
            mocap_quat=npz["traj_mocap_quat"][t],
            xfrc_applied=npz["traj_xfrc_applied"][t],
        )
        for t in range(T)
    ]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    return render_array(mj_model, states, height=height, width=width, scene_option=scene_option)


def update_root_index(analysis_dir: Path) -> None:
    """Regenerate analysis/index.html as a landing page listing all available runs."""
    runs = []
    for d in analysis_dir.iterdir():
        if not d.is_dir() or not (d / "index.html").exists():
            continue
        meta: dict = {}
        data_json = d / "data.json"
        if data_json.exists():
            try:
                with open(data_json) as f:
                    meta = json.load(f).get("meta", {})
            except Exception:
                pass
        runs.append((d.name, meta, data_json.stat().st_mtime if data_json.exists() else 0))

    runs.sort(key=lambda r: r[2], reverse=True)

    rows = ""
    for name, meta, _ in runs:
        ckpt = (meta.get("checkpoint") or "").replace("\\", "/").split("/")[-1]
        parts = [
            meta.get("sensor_bundle", ""),
            f"ckpt {ckpt}" if ckpt else "",
            f"T={meta['T']}" if "T" in meta else "",
            "det" if meta.get("deterministic") else ("sto" if "deterministic" in meta else ""),
        ]
        detail = " / ".join(p for p in parts if p)
        rows += f'<li><a href="{name}/">{name}</a>'
        if detail:
            rows += f' <span class="d">{detail}</span>'
        rows += "</li>\n"

    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <title>Policy Analyzer</title>\n"
        "  <style>\n"
        "    body{background:#111827;color:#f3f4f6;font:13px ui-monospace,monospace;padding:2rem}\n"
        "    h1{font-size:13px;color:#9ca3af;margin-bottom:1.2rem}\n"
        "    ul{list-style:none;padding:0}\n"
        "    li{margin:5px 0}\n"
        "    a{color:#60a5fa;text-decoration:none}\n"
        "    a:hover{text-decoration:underline}\n"
        "    .d{color:#6b7280;margin-left:10px;font-size:11px}\n"
        "  </style>\n"
        "</head>\n<body>\n"
        f"  <h1>Policy Analyzer — {len(runs)} run{'s' if len(runs) != 1 else ''}</h1>\n"
        f"  <ul>\n{rows}  </ul>\n"
        "</body>\n</html>\n"
    )
    (analysis_dir / "index.html").write_text(html)


def export_frontend(
    run_dir: Path,
    schema: Optional[dict] = None,
    render_height: int = 360,
    render_width: int = 480,
) -> Path:
    """Generate frontend artifacts in run_dir from rollout.npz.

    Also regenerates analysis/index.html (run_dir.parent) as the landing page.
    """
    import PIL.Image
    from experimentation.policy_analyzer.collect import load_env_from_checkpoint
    from mujoco_playground._src.manipulation.tesollo_hand import io_schema as _io

    npz = np.load(run_dir / "rollout.npz", allow_pickle=False)

    env_name, cfg, env = load_env_from_checkpoint(str(npz["id_checkpoint"]))

    if schema is None:
        schema = _io.build_io_schema(env, env_name=env_name, sensor_bundle=cfg.sensor_bundle)

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    T = int(npz["pre_squash"].shape[0])
    print(f"Rendering {T} frames …")
    frames = _render_frames(npz, env.mj_model, height=render_height, width=render_width)
    for i, frame in enumerate(frames):
        PIL.Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
    print(f"Wrote {T} frames to {frames_dir}")

    data = {
        "meta": {
            "env_name": env_name,
            "sensor_bundle": cfg.sensor_bundle,
            "checkpoint": str(npz["id_checkpoint"]),
            "T": T,
            "dt": float(npz["id_dt"]),
            "deterministic": bool(npz["id_deterministic"]),
            "seed": int(npz["id_seed"]),
            "tanh_sat": _TANH_SAT,
        },
        "input_groups": schema["input_groups"],
        "output_groups": schema["output_groups"],
        "obs": np.asarray(npz["obs"]).tolist(),
        "dof": {
            "pre_squash":   np.asarray(npz["pre_squash"]).tolist(),
            "action_scale": np.asarray(npz["action_scale"]).tolist(),
            "command":      np.asarray(npz["command"]).tolist(),
        },
    }
    data_path = run_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote {data_path}")

    shutil.copy(_TEMPLATE, run_dir / "index.html")

    update_root_index(run_dir.parent)
    return run_dir
