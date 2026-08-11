"""Generic MuJoCo scene builder: freeze joints, bake poses, trim actuators."""

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




class SceneBuilder:
    def __init__(self, xml_path: str | Path):
        self._xml_path = Path(xml_path)

    def build_spec(
        self,
        *,
        keyframe_name: str = "home",
        bodies_to_bake: list[str] = (),
        joints_to_remove: list[str] = (),
        actuators_to_remove: list[str] = (),
    ) -> mujoco.MjSpec:
        """Return a modified MjSpec with the requested joints/actuators removed.

        Bodies in `bodies_to_bake` have their current keyframe pose baked into
        their parent-relative pos/quat before their joints are removed, so they
        stay in place in the frozen scene.

        Returns MjSpec directly to avoid the to_xml() → from_string() roundtrip,
        which can fail when the spec merges multiple model assets with conflicting
        default class hierarchies.
        """
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        model = spec.compile()
        data = mujoco.MjData(model)

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        for body_name in bodies_to_bake:
            pos, quat = _body_pose_in_parent_frame(data, model, body_name)
            spec.worldbody.find_child(body_name).pos = pos
            spec.worldbody.find_child(body_name).quat = quat

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

        return spec

    def build(
        self,
        *,
        keyframe_name: str = "home",
        bodies_to_bake: list[str] = (),
        joints_to_remove: list[str] = (),
        actuators_to_remove: list[str] = (),
    ) -> str:
        """Like build_spec() but returns an XML string instead of an MjSpec."""
        return self.build_spec(
            keyframe_name=keyframe_name,
            bodies_to_bake=bodies_to_bake,
            joints_to_remove=joints_to_remove,
            actuators_to_remove=actuators_to_remove,
        ).to_xml()
