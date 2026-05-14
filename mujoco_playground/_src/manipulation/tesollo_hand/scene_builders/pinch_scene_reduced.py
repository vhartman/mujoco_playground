"""Scene builder for the static-grasp variant of the pinch scene.

Loads scene_mjx_cube_pinch and produces a reduced model where:
  - The cube's free joint is removed (cube fixed at its keyframe pose).
  - Only thumb (dg_1) and index (dg_2) actuators remain.
  - Wrist and middle/ring/pinky finger joints are deleted; their bodies are
    frozen by baking the keyframe poses into the body-frame transforms before
    removing the joints.
"""

import mujoco
import numpy as np

from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_wrist_constants as consts,
)
from mujoco_playground._src.manipulation.tesollo_hand.scene_builders.scene_builder import (
    SceneBuilder,
)

_XML_PATH = consts.ROOT_PATH / "xmls" / "scene_mjx_cube_pinch.xml"

_JOINTS_TO_FREEZE = (
    [f"rj_wrist_1_{i}" for i in range(1, 4)]
    + [f"rj_dg_{f}_{i}" for f in (3, 4, 5) for i in range(1, 5)]
    + ["cube_freejoint"]
)

_ACTUATORS_TO_FREEZE = (
    [f"rj_wrist_1_{i}_a" for i in range(1, 4)]
    + [f"rj_dg_{f}_{i}_a" for f in (3, 4, 5) for i in range(1, 5)]
)

_BODIES_TO_BAKE = [f"rl_dg_{f}_{i}" for f in (3, 4, 5) for i in range(1, 5)] + ["cube"]


def _rh_pose_overrides(
    spec: mujoco.MjSpec,
    data: mujoco.MjData,
    model: mujoco.MjModel,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute rh pose relative to its wrapping <frame> (not the worldbody).

    The rh body lives inside a <frame> in the XML, so spec rh.pos/quat are
    expressed relative to that frame — not relative to the worldbody.  Using
    the worldbody as parent would double-count the frame offset.
    """
    rh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh")
    xpos_rh = data.xpos[rh_id]
    xquat_rh = data.xquat[rh_id]

    frame = spec.worldbody.frames[0]
    frame_quat = np.array(frame.quat, dtype=float)
    frame_quat /= np.linalg.norm(frame_quat)
    frame_pos = np.array(frame.pos, dtype=float)

    w, x, y, z = frame_quat
    frame_mat = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

    pos_in_frame = frame_mat.T @ (xpos_rh - frame_pos)

    frame_quat_inv = np.array([frame_quat[0], -frame_quat[1], -frame_quat[2], -frame_quat[3]])
    quat_in_frame = np.zeros(4)
    mujoco.mju_mulQuat(quat_in_frame, frame_quat_inv, xquat_rh)

    return {"rh": (pos_in_frame, quat_in_frame)}


def build_reduced_pinch_scene(keyframe_name: str = "home") -> str:
    """Return an XML string for the static-grasp scene."""
    return SceneBuilder(_XML_PATH).build(
        keyframe_name=keyframe_name,
        bodies_to_bake=_BODIES_TO_BAKE,
        joints_to_remove=_JOINTS_TO_FREEZE,
        actuators_to_remove=_ACTUATORS_TO_FREEZE,
        pose_overrides=_rh_pose_overrides,
    )
