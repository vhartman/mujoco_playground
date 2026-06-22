"""Environment visualizer — works with preset names, registry names, XML paths, or dotted RL env class paths."""
import argparse
import importlib
import pathlib

import jax
import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_playground._src import manipulation
from mujoco_playground._src import registry
from mujoco_playground._src.manipulation.tesollo_hand.base_wrist import get_assets


_ROOT = pathlib.Path(__file__).parent.parent

_PRESET_XML = {
    "pinch_full": lambda: mujoco.MjModel.from_xml_path(
        str(_ROOT / "mujoco_playground/_src/manipulation/tesollo_hand/xmls/scene_mjx_cube_pinch.xml")
    ),
    # "pinch_restricted": lambda: mujoco.MjModel.from_xml_string(
    #     build_reduced_pinch_scene(), assets=get_assets()
    # ),
}

_RL_ENV_MODULE_ROOT = "mujoco_playground._src.manipulation.tesollo_hand"


def load_env(env_arg: str, impl: str = "warp"):
    """Returns (env_instance | None, mj_model, env_name | None).

    env_name is the registry key (e.g. "TesolloCubePinch") when the env was
    loaded via the registry; None otherwise.
    """
    if env_arg in _PRESET_XML:
        return None, _PRESET_XML[env_arg](), None

    p = pathlib.Path(env_arg)
    if p.suffix == ".xml":
        return None, mujoco.MjModel.from_xml_path(str(p.resolve())), None

    if env_arg in registry.ALL_ENVS:
        env = registry.load(env_arg, config_overrides={"impl": impl})
        return env, env.mj_model, env_arg

    # Dotted Python path: some.module.ClassName
    module_path, _, class_name = env_arg.rpartition(".")
    if not module_path:
        raise ValueError(f"Cannot interpret env argument: {env_arg!r}")
    if _RL_ENV_MODULE_ROOT not in module_path:
        module_path = f"{_RL_ENV_MODULE_ROOT}.{module_path}"
    cls = getattr(importlib.import_module(module_path), class_name)
    env = cls(config_overrides={"impl": impl})
    _class_to_name = {v: k for k, v in manipulation._envs.items()}
    env_name = _class_to_name.get(cls)
    return env, env.mj_model, env_name


def load_model(env_arg: str, impl: str = "warp") -> mujoco.MjModel:
    _, m, _ = load_env(env_arg, impl)
    return m


def _apply_dr_to_spec(spec: mujoco.MjSpec, m_ref: mujoco.MjModel,
                      geom_size: np.ndarray, body_pos: np.ndarray):
    """Write DR'd cube geom sizes and body position back into the spec.

    m_ref is the current compiled model, used only for name→id lookups.
    """
    cube_bid = m_ref.body("cube").id
    for body in spec.bodies:
        if body.name != "cube":
            continue
        body.pos = body_pos[cube_bid]
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX and geom.name:
                geom.size = geom_size[m_ref.geom(geom.name).id]
            elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_gid = next(
                    g for g in range(m_ref.ngeom)
                    if m_ref.geom_bodyid[g] == cube_bid
                    and m_ref.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                )
                for mesh in spec.meshes:
                    if mesh.name == "cube_mesh":
                        mesh.scale = geom_size[mesh_gid]
                        break
        break


def make_key_callback(m, data, env=None, randomize_fn=None):
    """P: print qpos. R: call env.reset() and update the viewer (only when env is provided).

    When randomize_fn is set, R also applies domain randomization to the MJX
    model before resetting, recompiles the spec for correct mesh rendering,
    and re-uploads changed meshes to the GPU via the viewer handle.

    Call set_viewer() after launch_passive to enable mesh re-upload.
    """
    rng = [None]
    base_mjx_model = [None]
    pending_mesh_update = [False]
    if env is not None:
        rng[0] = jax.random.PRNGKey(0)
    if randomize_fn is not None:
        base_mjx_model[0] = env.mjx_model

    def _cb(keycode):
        ch = chr(keycode)

        if ch == "P":
            jnt_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
            print("\n--- current qpos ---")
            print(" ".join(f"{v:.6f}" for v in data.qpos))
            print("--- per joint ---")
            qi = 0
            for i, name in enumerate(jnt_names):
                jtype = m.jnt_type[i]
                nq = 4 if jtype == mujoco.mjtJoint.mjJNT_FREE else (
                     3 if jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
                print(f"  {name}: {' '.join(f'{v:.6f}' for v in data.qpos[qi:qi+nq])}")
                qi += nq
            print()

        elif ch == "R" and env is not None:
            rng[0], key, dr_key = jax.random.split(rng[0], 3)

            if randomize_fn is not None:
                model_dr, in_axes = randomize_fn(base_mjx_model[0], dr_key[None])
                env._mjx_model = jax.tree.map(
                    lambda a, ax: a[0] if ax == 0 else a, model_dr, in_axes,
                )
                new_geom_size = np.array(env._mjx_model.geom_size)
                new_body_pos = np.array(env._mjx_model.body_pos)
                changed = {}
                for name, old, new in [
                    ("geom_size", m.geom_size, new_geom_size),
                    ("body_pos", m.body_pos, new_body_pos),
                ]:
                    diff_mask = ~np.isclose(old, new)
                    if diff_mask.any():
                        idxs = np.argwhere(diff_mask)
                        changed[name] = {
                            tuple(idx): f"{old[tuple(idx)]:.4f} -> {new[tuple(idx)]:.4f}"
                            for idx in idxs[:8]
                        }
                _apply_dr_to_spec(env._mj_spec, m, new_geom_size, new_body_pos)
                m_new = env._mj_spec.compile()
                m.geom_size[:] = m_new.geom_size
                m.body_pos[:] = m_new.body_pos
                m.body_mass[:] = m_new.body_mass
                m.body_inertia[:] = m_new.body_inertia
                m.mesh_vert[:] = m_new.mesh_vert
                m.mesh_normal[:] = m_new.mesh_normal
                pending_mesh_update[0] = True
                if changed:
                    print("--- domain_randomize ---")
                    for name, diffs in changed.items():
                        print(f"  {name}:")
                        for idx, val in diffs.items():
                            print(f"    [{', '.join(str(i) for i in idx)}]: {val}")
                else:
                    print("domain_randomize: no values changed")

            state = env.reset(key)
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)
            data.mocap_pos[:] = np.array(state.data.mocap_pos)
            data.mocap_quat[:] = np.array(state.data.mocap_quat)
            mujoco.mj_forward(m, data)
            print("env.reset() applied")

    _cb.pending_mesh_update = pending_mesh_update
    return _cb


def render_video(
    m: mujoco.MjModel,
    data: mujoco.MjData,
    path: str,
    steps: int = 600,
    fps: float = 30.0,
    height: int = 480,
    width: int = 640,
) -> None:
    """Simulate `steps` steps and write a video to `path` at `fps` frames per second."""
    renderer = mujoco.Renderer(m, height=height, width=width)
    render_every = max(1, round(1.0 / (fps * m.opt.timestep)))
    frames = []
    for i in range(steps):
        mujoco.mj_step(m, data)
        if i % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render())
    media.write_video(path, frames, fps=fps)
    print(f"Video saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="proprio",
        help=(
            "Preset name %(choices)s, path to an .xml file, "
            "or dotted Python class path (e.g. my.module.MyEnv)"
        ),
        metavar="ENV",
    )
    parser.add_argument("--output", default="viewer", choices=["viewer", "video"],
                        help="Output mode: launch interactive viewer or render to video file (default: viewer)")
    parser.add_argument("--mode", default="active", choices=["passive", "active"],
                        help="Viewer mode, ignored when --output video (default: active)")
    parser.add_argument("--impl", default="warp", choices=["jax", "warp"],
                        help="MJX backend for RL envs (default: warp)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Number of simulation steps for video output (default: 600)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second for video output (default: 30)")
    parser.add_argument("--video-path", default=None,
                        help="Output path for video (default: <env>.mp4)")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Apply domain randomization on each R press (requires a registry env name)")
    args = parser.parse_args()

    env, m, env_name = load_env(args.env, impl=args.impl)
    data = mujoco.MjData(m)

    randomize_fn = None
    if args.domain_rand:
        if env_name is None:
            parser.error("--domain-rand requires a registry env name (e.g. TesolloCubePinch)")
        randomize_fn = registry.get_domain_randomizer(env_name)
        if randomize_fn is None:
            parser.error(f"No domain randomizer registered for {env_name!r}")
        print(f"Domain randomization enabled for {env_name}")

    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(m, data, key_id)

    if args.output == "video":
        video_path = args.video_path or f"{args.env}.mp4"
        render_video(m, data, path=video_path, steps=args.steps, fps=args.fps)
    elif args.mode == "active":
        mujoco.viewer.launch(m, data)
    else:
        cb = make_key_callback(m, data, env=env, randomize_fn=randomize_fn)
        if env is not None:
            print("Press R to call env.reset() and update the viewer.")
            if randomize_fn is not None:
                print("  (domain randomization will be applied on each R press)")
        with mujoco.viewer.launch_passive(m, data, key_callback=cb) as v:
            while v.is_running():
                if cb.pending_mesh_update[0]:
                    cb.pending_mesh_update[0] = False
                    cube_mesh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "cube_mesh")
                    if cube_mesh_id >= 0:
                        v.update_mesh(cube_mesh_id)
                mujoco.mj_forward(m, data)
                v.sync()
