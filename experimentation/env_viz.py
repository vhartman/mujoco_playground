"""Environment visualizer — works with preset names, XML paths, or dotted RL env class paths."""
import argparse
import importlib
import mujoco
import mujoco.viewer
import pathlib

from mujoco_playground._src.manipulation.tesollo_hand.pinch import (
    CubePinchForce,
    CubePinchProprio,
    CubePinchBaseline,
)
from mujoco_playground._src.manipulation.tesollo_hand.base_wrist import get_assets
from mujoco_playground._src.manipulation.tesollo_hand.scene_builders.static_grasp import (
    build_static_grasp_scene,
)

_ROOT = pathlib.Path(__file__).parent.parent

_PRESET_XML = {
    "pinch_full": lambda: mujoco.MjModel.from_xml_path(
        str(_ROOT / "mujoco_playground/_src/manipulation/tesollo_hand/xmls/scene_mjx_cube_pinch.xml")
    ),
    "pinch_restricted": lambda: mujoco.MjModel.from_xml_string(
        build_static_grasp_scene(), assets=get_assets()
    ),
}

_PRESET_RL = {
    "force": CubePinchForce,
    "proprio": CubePinchProprio,
    "baseline": CubePinchBaseline,
}


def load_model(env_arg: str) -> mujoco.MjModel:
    if env_arg in _PRESET_RL:
        return _PRESET_RL[env_arg]().mj_model

    if env_arg in _PRESET_XML:
        return _PRESET_XML[env_arg]()

    # XML file path
    p = pathlib.Path(env_arg)
    if p.suffix == ".xml":
        return mujoco.MjModel.from_xml_path(str(p.resolve()))

    # Dotted Python path: some.module.ClassName
    module_path, _, class_name = env_arg.rpartition(".")
    if not module_path:
        raise ValueError(f"Cannot interpret env argument: {env_arg!r}")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls().mj_model


def print_qpos(m, data):
    def _cb(keycode):
        if chr(keycode) != "P":
            return
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
    return _cb


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
parser.add_argument("--mode", default="active", choices=["passive", "active"])
args = parser.parse_args()

m = load_model(args.env)
data = mujoco.MjData(m)

key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(m, data, key_id)

if args.mode == "active":
    mujoco.viewer.launch(m, data)
else:
    with mujoco.viewer.launch_passive(m, data, key_callback=print_qpos(m, data)) as v:
        while v.is_running():
            mujoco.mj_forward(m, data)
            v.sync()
