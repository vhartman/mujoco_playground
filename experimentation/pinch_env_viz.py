import argparse
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

_XML_ENVS = {"pinch_full", "pinch_restricted"}

_RL_ENVS = {
    "cube_pinch_force": CubePinchForce,
    "cube_pinch_proprio": CubePinchProprio,
    "cube_pinch_baseline": CubePinchBaseline,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    choices=[*_XML_ENVS, *_RL_ENVS],
    default="cube_pinch_proprio",
)
parser.add_argument("--mode", default="active", choices=["passive", "active"])
args = parser.parse_args()

if args.env in _XML_ENVS:
    if args.env == "pinch_full":
        mujoco_dir = pathlib.Path(__file__).parent.parent
        xml_dir = mujoco_dir / "mujoco_playground/_src/manipulation/tesollo_hand/xmls/"
        m = mujoco.MjModel.from_xml_path(str(xml_dir / "scene_mjx_cube_pinch.xml"))
    else:
        m = mujoco.MjModel.from_xml_string(build_static_grasp_scene(), assets=get_assets())
    data = mujoco.MjData(m)
else:
    env = _RL_ENVS[args.env]()
    m = env.mj_model
    data = mujoco.MjData(m)


# Load the home keyframe as starting pose
mujoco.mj_resetDataKeyframe(m, data, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))

def print_qpos(keycode):
    if chr(keycode) == 'P':
        jnt_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        print('\n--- current qpos ---')
        print(' '.join(f'{v:.6f}' for v in data.qpos))
        print('--- per joint ---')
        qi = 0
        for i, name in enumerate(jnt_names):
            jtype = m.jnt_type[i]
            nq = 4 if jtype == mujoco.mjtJoint.mjJNT_FREE else (
                 3 if jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
            vals = data.qpos[qi:qi+nq]
            print(f'  {name}: {" ".join(f"{v:.6f}" for v in vals)}')
            qi += nq
        print()

if args.mode == "active":
    mujoco.viewer.launch(m, data)
elif args.mode == "passive":
    with mujoco.viewer.launch_passive(m, data, key_callback=print_qpos) as v:
        while v.is_running():
            mujoco.mj_forward(m, data)
            v.sync()
