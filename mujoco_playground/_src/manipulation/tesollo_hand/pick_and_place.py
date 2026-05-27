# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pick-and-place task variants: different observation configurations."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.tesollo_hand.base_pick_and_place import (
    PickAndPlaceBase,
    default_config,
)
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_grasp_constants as consts,
)

__all__ = [
    "default_config",
    "PickAndPlaceProprio",
    "PickAndPlaceBaseline",
    "PickAndPlaceForce",
    "domain_randomize",
]

class PickAndPlaceBaseline(PickAndPlaceBase):
    """Pick-and-place: q(26) + qdot(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) = 65."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, cube_pos = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )
        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

class PickAndPlaceProprio(PickAndPlaceBase):
    """Pick-and-place: q(26) + qdot(26) + ctrl_targets(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) = 91."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, cube_pos = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )
        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_motor_targets(info),              # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

class PickAndPlaceProprioDelta(PickAndPlaceBase):
    """Pick-and-place: q(26) + qdot(26) + ctrl_targets(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) = 91."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, cube_pos = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )

        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_motor_deltas(data, info),         # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

class PickAndPlaceForce(PickAndPlaceBase):
    """Pick-and-place with ground-truth contact forces: q(26) + qdot(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) + fingertip_forces(5) + total_force(1) = 97."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, _ = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )
        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
            self._obs_fingertip_forces(data),           # 5
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

class PickAndPlaceForceProprio(PickAndPlaceBase):
    """Pick-and-place with ground-truth contact forces: q(26) + qdot(26) + ctrl_targets(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) + fingertip_forces(5) + total_force(1) = 97."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, _ = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )
        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_motor_targets(info),              # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
            self._obs_fingertip_forces(data),           # 5
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

class PickAndPlaceForceProprioDelta(PickAndPlaceBase):
    """Pick-and-place with ground-truth contact forces: q(26) + qdot(26) + ctrl_targets(26) + cube_pos(3) + last_ground_cube_pos(3) + goal_pos(3) + goal_quat(4) + fingertip_forces(5) + total_force(1) = 97."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles, joint_vel, _ = self._maybe_apply_obs_noise(
            self._obs_joint_angles(data),
            self._obs_joint_velocities(data),
            self._obs_cube_pos(data),
            info,
        )
        state = jp.concatenate([
            joint_angles,                               # 26
            joint_vel,                                  # 26
            self._obs_motor_deltas(data, info),         # 26
            self._obs_last_ground_cube_pos(info),       # 3
            self._obs_goal_pos(info),                   # 3
            self._obs_goal_quat(info),                  # 4
            self._obs_fingertip_forces(data),           # 5
        ])
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = PickAndPlaceProprio().mj_model
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4",
        "rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4",
        "rl_dg_3_1", "rl_dg_3_2", "rl_dg_3_3", "rl_dg_3_4",
        "rl_dg_4_1", "rl_dg_4_2", "rl_dg_4_3", "rl_dg_4_4",
        "rl_dg_5_1", "rl_dg_5_2", "rl_dg_5_3", "rl_dg_5_4",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    silicone_geom_ids = [
        mj_model.geom(g).id for g in [
            "rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip",
        ]
    ]

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        silicone_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[silicone_geom_ids, 0].set(silicone_friction)

        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0.at[hand_qids].set(
            model.qpos0[hand_qids]
            + jax.random.uniform(key, shape=(consts.NQ,), minval=-0.05, maxval=0.05)
        )

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (consts.NQ,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction, body_mass, body_inertia, body_ipos, qpos0,
            dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
        )

    (
        geom_friction, body_mass, body_inertia, body_ipos, qpos0,
        dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "body_mass": 0, "body_inertia": 0, "body_ipos": 0,
        "qpos0": 0, "dof_frictionloss": 0, "dof_armature": 0, "dof_damping": 0,
        "actuator_gainprm": 0, "actuator_biasprm": 0,
    })
    model = model.tree_replace({
        "geom_friction": geom_friction, "body_mass": body_mass,
        "body_inertia": body_inertia, "body_ipos": body_ipos, "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss, "dof_armature": dof_armature,
        "dof_damping": dof_damping, "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })
    return model, in_axes
