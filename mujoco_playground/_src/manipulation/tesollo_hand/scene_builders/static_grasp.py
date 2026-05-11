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

# Bodies (excluding rh) whose joints are removed; poses baked from parent body frame.
_BODIES_TO_BAKE = [f"rl_dg_{f}_{i}" for f in (3, 4, 5) for i in range(1, 5)] + ["cube"]

# Joints and actuators that survive into the reduced model.
_ACTIVE_JOINTS = [f"rj_dg_{f}_{i}" for f in (1, 2) for i in range(1, 5)]
_ACTIVE_ACTUATORS = [f"rj_dg_{f}_{i}_a" for f in (1, 2) for i in range(1, 5)]


def _compute_body_pose_relative_to_parent(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat) of body expressed in its parent body's frame."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pid = model.body_parentid[bid]
    xpos_b = data.xpos[bid].copy()
    xquat_b = data.xquat[bid].copy()
    xpos_p = data.xpos[pid].copy()
    xmat_p = data.xmat[pid].reshape(3, 3)
    xquat_p = data.xquat[pid].copy()

    pos_rel = xmat_p.T @ (xpos_b - xpos_p)

    quat_p_inv = np.array(
        [xquat_p[0], -xquat_p[1], -xquat_p[2], -xquat_p[3]], dtype=float
    )
    quat_rel = np.zeros(4)
    mujoco.mju_mulQuat(quat_rel, quat_p_inv, xquat_b)

    return pos_rel, quat_rel


def _rh_pose_relative_to_frame(
    spec: mujoco.MjSpec,
    data: mujoco.MjData,
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rh pose relative to its wrapping frame (not the worldbody).

    The rh body lives inside a <frame> in the XML, so spec rh.pos/quat are
    expressed relative to that frame — not relative to the worldbody.  Using
    the worldbody as parent would double-count the frame offset.
    """
    rh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh")
    xpos_rh = data.xpos[rh_id].copy()
    xquat_rh = data.xquat[rh_id].copy()

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

    frame_quat_inv = np.array(
        [frame_quat[0], -frame_quat[1], -frame_quat[2], -frame_quat[3]]
    )
    quat_in_frame = np.zeros(4)
    mujoco.mju_mulQuat(quat_in_frame, frame_quat_inv, xquat_rh)

    return pos_in_frame, quat_in_frame


def build_static_grasp_scene(keyframe_name: str = "home") -> str:
    """Return an XML string for the static-grasp scene.

    The scene is derived from scene_mjx_cube_pinch with the following changes:
      - Cube free joint removed; cube fixed at its keyframe pose.
      - Wrist, middle, ring and pinky joints removed; bodies frozen at their
        keyframe poses by baking the transforms into body-frame pos/quat.
      - All actuators except thumb (dg_1) and index (dg_2) removed.
      - A new keyframe is added with the thumb and index joint values from the
        original keyframe.
    """
    spec = mujoco.MjSpec.from_file(str(_XML_PATH))
    model = spec.compile()
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Bake keyframe poses into body-frame transforms before removing joints.
    rh_pos, rh_quat = _rh_pose_relative_to_frame(spec, data, model)
    rh_body = spec.worldbody.find_child("rh")
    rh_body.pos = rh_pos
    rh_body.quat = rh_quat

    for name in _BODIES_TO_BAKE:
        pos_rel, quat_rel = _compute_body_pose_relative_to_parent(data, model, name)
        spec.worldbody.find_child(name).pos = pos_rel
        spec.worldbody.find_child(name).quat = quat_rel

    # Extract thumb+index qpos/ctrl before joint/actuator addresses are gone.
    new_qpos = [
        float(model.key_qpos[key_id, model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        ]])
        for j in _ACTIVE_JOINTS
    ]
    new_ctrl = [
        float(model.key_ctrl[key_id, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)])
        for a in _ACTIVE_ACTUATORS
    ]

    joints_by_name = {j.name: j for j in spec.joints}
    for name in _JOINTS_TO_FREEZE:
        spec.delete(joints_by_name[name])

    actuators_by_name = {a.name: a for a in spec.actuators}
    for name in _ACTUATORS_TO_FREEZE:
        spec.delete(actuators_by_name[name])

    for key in list(spec.keys):
        spec.delete(key)

    spec.add_key(name=keyframe_name, qpos=new_qpos, ctrl=new_ctrl)

    return spec.to_xml()
