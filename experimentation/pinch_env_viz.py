import argparse
import mujoco
import mujoco.viewer
import pathlib
from mujoco_playground._src.manipulation.tesollo_hand.pinch import CubePinch

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["xml", "rl_env"], default="xml")
args = parser.parse_args()
source = args.source

if source == "xml":
    f = 'scene_mjx_cube_pinch.xml'

    mujoco_dir = pathlib.Path(__file__).parent.parent
    XML_DIR = mujoco_dir / "mujoco_playground/_src/manipulation/tesollo_hand/xmls/"

    m = mujoco.MjModel.from_xml_path(str(XML_DIR / f))
    hand_object = m.body("rh")

    data = mujoco.MjData(m)
elif source == "rl_env":
    env = CubePinch()
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

mujoco.viewer.launch(m, data)
# with mujoco.viewer.launch_passive(m, data, key_callback=print_qpos) as v:
#     while v.is_running():
#         mujoco.mj_forward(m, data)
#         v.sync()
