"""Scene builder for the pinch task: freeze wrist + middle/ring/pinky, fix cube."""

import mujoco

from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_pinch_constants as consts,
)
from mujoco_playground._src.manipulation.tesollo_hand.scene_builders.scene_modifier import (
    SceneBuilder,
)

_JOINTS_TO_REMOVE = [
    # wrist
    "rj_wrist_0_1", "rj_wrist_0_2", "rj_wrist_0_3",
    "rj_wrist_1_1", "rj_wrist_1_2", "rj_wrist_1_3",
    # cube
    "cube_freejoint",
    # middle finger
    "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
    # ring finger
    "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    # pinky
    "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
]

_ACTUATORS_TO_REMOVE = [
    # wrist
    "wrist_tx", "wrist_ty", "wrist_tz",
    "wrist_rx", "wrist_ry", "wrist_rz",
    # middle
    "dg_3_j1_rx", "dg_3_j2_ry", "dg_3_j3_ry", "dg_3_j4_ry",
    # ring
    "dg_4_j1_rx", "dg_4_j2_ry", "dg_4_j3_ry", "dg_4_j4_ry",
    # pinky
    "dg_5_j1_rz", "dg_5_j2_rx", "dg_5_j3_ry", "dg_5_j4_ry",
]

# Bodies whose keyframe pose must be baked into spec pos/quat before their
# joints are removed. List in parent-first order within each kinematic chain.
_BODIES_TO_BAKE = [
    "rh",          # wrist translation + rotation baked here
    "cube",        # cube fixed at keyframe world position
    # middle finger segments
    "rl_dg_3_1", "rl_dg_3_2", "rl_dg_3_3", "rl_dg_3_4",
    # ring finger segments
    "rl_dg_4_1", "rl_dg_4_2", "rl_dg_4_3", "rl_dg_4_4",
    # pinky segments
    "rl_dg_5_1", "rl_dg_5_2", "rl_dg_5_3", "rl_dg_5_4",
]


def build_pinch_spec(weld_cube: bool = True) -> mujoco.MjSpec:
    """Return a reduced MjSpec for the pinch task.

    Bakes the wrist and frozen-finger poses from the 'home' keyframe and strips
    all joints/actuators that are not thumb (dg_1) or index (dg_2). When
    ``weld_cube`` is True (default) the cube freejoint is removed and its home
    pose baked (static squeeze); when False the cube keeps its freejoint and is a
    free rigid body the policy must actively hold.
    """
    joints_to_remove = list(_JOINTS_TO_REMOVE)
    bodies_to_bake = list(_BODIES_TO_BAKE)
    if not weld_cube:
        joints_to_remove.remove("cube_freejoint")
        bodies_to_bake.remove("cube")
    return SceneBuilder(consts.SCENE_XML).build_spec(
        keyframe_name="home",
        bodies_to_bake=bodies_to_bake,
        joints_to_remove=joints_to_remove,
        actuators_to_remove=_ACTUATORS_TO_REMOVE,
    )
