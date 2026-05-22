"""Generic MuJoCo scene builder: freeze joints, bake poses, trim actuators."""

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np


def _body_pose_in_parent_frame(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat) of body expressed in its parent body's frame."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pid = model.body_parentid[bid]

    xmat_p = data.xmat[pid].reshape(3, 3)
    pos_rel = xmat_p.T @ (data.xpos[bid] - data.xpos[pid])

    xquat_p = data.xquat[pid]
    quat_p_inv = np.array([xquat_p[0], -xquat_p[1], -xquat_p[2], -xquat_p[3]])
    quat_rel = np.zeros(4)
    mujoco.mju_mulQuat(quat_rel, quat_p_inv, data.xquat[bid])

    return pos_rel, quat_rel


# Signature for callers that need custom pose computation (e.g. bodies inside frames).
PoseOverrideFn = Callable[
    [mujoco.MjSpec, mujoco.MjData, mujoco.MjModel],
    dict[str, tuple[np.ndarray, np.ndarray]],
]


class SceneBuilder:
    def __init__(self, xml_path: str | Path):
        self._xml_path = Path(xml_path)

    def build(
        self,
        *,
        keyframe_name: str = "home",
        bodies_to_bake: list[str] = (),
        joints_to_remove: list[str] = (),
        actuators_to_remove: list[str] = (),
        pose_overrides: PoseOverrideFn | None = None,
    ) -> str:
        """Return a modified XML string with the requested joints/actuators removed.

        Bodies in `bodies_to_bake` have their current keyframe pose baked into
        their parent-relative pos/quat before their joints are removed, so they
        stay in place in the frozen scene.  `pose_overrides` can supply custom
        (pos, quat) pairs for bodies that need non-standard computation (e.g. a
        body that is a child of a <frame> rather than another body).
        """
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        model = spec.compile()
        data = mujoco.MjData(model)

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        # Compute and apply any caller-supplied pose overrides first.
        overrides = pose_overrides(spec, data, model) if pose_overrides else {}
        for body_name, (pos, quat) in overrides.items():
            body = spec.worldbody.find_child(body_name)
            body.pos = pos
            body.quat = quat

        # Bake remaining bodies using the generic parent-frame computation.
        for body_name in bodies_to_bake:
            pos, quat = _body_pose_in_parent_frame(data, model, body_name)
            spec.worldbody.find_child(body_name).pos = pos
            spec.worldbody.find_child(body_name).quat = quat

        # Snapshot surviving joint qpos and actuator ctrl before addresses change.
        remove_joints = set(joints_to_remove)
        remove_actuators = set(actuators_to_remove)
        new_qpos = [
            float(model.key_qpos[key_id, model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            ]])
            for j in spec.joints if j.name not in remove_joints
        ]
        new_ctrl = [
            float(model.key_ctrl[key_id, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a.name)])
            for a in spec.actuators if a.name not in remove_actuators
        ]

        joints_by_name = {j.name: j for j in spec.joints}
        for name in joints_to_remove:
            spec.delete(joints_by_name[name])

        actuators_by_name = {a.name: a for a in spec.actuators}
        for name in actuators_to_remove:
            spec.delete(actuators_by_name[name])

        for key in list(spec.keys):
            spec.delete(key)
        spec.add_key(name=keyframe_name, qpos=new_qpos, ctrl=new_ctrl)

        return spec.to_xml()

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
