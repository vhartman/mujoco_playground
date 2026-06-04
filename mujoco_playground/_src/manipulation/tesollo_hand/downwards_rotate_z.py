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
"""Downwards-facing hand in-hand rotation environment for the Tesollo hand.

The hand is fixed in the pick_and_place orientation (palm facing downward) with
the wrist x-rotation restricted to keep it facing down. The cube starts on the
floor; the policy must learn to grasp and then rotate the cube around the world
z-axis to match a randomly sampled target orientation.
"""

__all__ = [
    "DownwardsRotateZ",
    "default_config",
    "domain_randomize",
]

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_grasp as tesollo_hand_base
from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_downwards_rotate_z_constants as consts,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=120,
        target_hold_time=1.0,
        sensor_bundle="proprio",
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.0,
                joint_vel=0.0,
                motor_targets=0.0,
                goal_quat=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                fingertip_pos=0.3,
                cube_ori=4.0,
                joint_vel=-0.002,
                wrist_vel=-0.02,
                action_rate=-0.005,
                cube_on_floor=-1.0,
            ),
            success_reward=5.0,
        ),
        pert_config=config_dict.create(
            enable=False,
            linear_velocity_pert=[0.0, 3.0],
            angular_velocity_pert=[0.0, 0.5],
            pert_duration_steps=[1, 100],
            pert_wait_steps=[60, 150],
        ),
        kp_scale=1.0,
        scene=config_dict.create(
            cube_mass=0.108,
        ),
        impl="warp",
        nconmax=200 * 8192,
        njmax=2200,
    )


class DownwardsRotateZ(tesollo_hand_base.TesolloHandGraspEnv):
    """In-hand rotation with a downward-facing hand.

    The cube starts on the floor. The policy must grasp it and rotate it around
    the world z-axis to match a randomly sampled target orientation. The wrist
    x-rotation DOF is restricted via ctrlrange to keep the palm facing down.
    """

    _TASK_KEYS: tuple[str, ...] = ("goal_quat",)

    def _task_obs_keys(self) -> tuple[str, ...]:
        return self._TASK_KEYS

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(
            xml_path=consts.SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        model_dirty = False
        if self._config.kp_scale != 1.0:
            s = self._config.kp_scale
            self._mj_model.actuator_gainprm[:, 0] *= s
            self._mj_model.actuator_biasprm[:, 1] *= s
            self._mj_model.actuator_biasprm[:, 2] *= s
            model_dirty = True
        cube_body_id = self._mj_model.body("cube").id
        xml_cube_mass = float(self._mj_model.body_mass[cube_body_id])
        cfg_cube_mass = float(self._config.scene.cube_mass)
        if cfg_cube_mass != xml_cube_mass:
            scale = cfg_cube_mass / xml_cube_mass
            self._mj_model.body_mass[cube_body_id] = cfg_cube_mass
            self._mj_model.body_inertia[cube_body_id] *= scale
            model_dirty = True
        if model_dirty:
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        obs_module.validate_spec(
            self._config.sensor_bundle,
            self._task_obs_keys(),
            self._config.obs_noise.scales,
        )
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos, dtype=float)
        self._init_mpos = jp.array(home_key.mpos, dtype=float)
        self._init_mquat = jp.array(home_key.mquat, dtype=float)
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._wrist_qids = mjx_env.get_qpos_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._wrist_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        non_hand_bodies = {"world", "cube", "goal"}
        self._hand_geom_ids = jp.array([
            g for g in range(self._mj_model.ngeom)
            if self._mj_model.geom(g).contype != 0
            and self._mj_model.body(self._mj_model.geom_bodyid[g]).name not in non_hand_bodies
            and g != self._floor_geom_id
        ])
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]
        self._cube_init = self._init_q[self._cube_qids]
        self._geom = consts.SceneGeometry.from_mj_model(self._mj_model)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        state = self._build_obs(
            self._config.sensor_bundle, self._task_obs_keys(), data, info
        )
        return {
            "state": state,
            "privileged_state": self._obs_privileged(data, info),
        }

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, p_rng = jax.random.split(rng)
        start_pos = self._cube_init[:2] + jax.random.uniform(
            p_rng, (2,), minval=-0.01, maxval=0.01
        )
        start_pos = jp.array([start_pos[0], start_pos[1], self._cube_init[2]])

        rng, goal_rot_rng = jax.random.split(rng)
        goal_angle = jax.random.uniform(goal_rot_rng, minval=0.0, maxval=2 * jp.pi)
        goal_quat = jp.array([jp.cos(goal_angle / 2), 0.0, 0.0, jp.sin(goal_angle / 2)])

        qpos = self._init_q.at[self._cube_qids[:3]].set(start_pos)
        qvel = jp.zeros(self.mj_model.nv)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=self._default_pose,
            qvel=qvel,
            mocap_pos=self._init_mpos,
            mocap_quat=goal_quat,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
        pert_wait_steps = jax.random.randint(
            pert1, (1,),
            minval=self._config.pert_config.pert_wait_steps[0],
            maxval=self._config.pert_config.pert_wait_steps[1],
        )
        pert_duration_steps = jax.random.randint(
            pert2, (1,),
            minval=self._config.pert_config.pert_duration_steps[0],
            maxval=self._config.pert_config.pert_duration_steps[1],
        )
        pert_lin = jax.random.uniform(
            pert3,
            minval=self._config.pert_config.linear_velocity_pert[0],
            maxval=self._config.pert_config.linear_velocity_pert[1],
        )
        pert_ang = jax.random.uniform(
            pert3,
            minval=self._config.pert_config.angular_velocity_pert[0],
            maxval=self._config.pert_config.angular_velocity_pert[1],
        )

        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "at_target_step_counter": jp.zeros((), dtype=jp.int32),
            "motor_targets": data.ctrl,
            "goal_quat": goal_quat,
            "pert_wait_steps": pert_wait_steps,
            "pert_duration_steps": pert_duration_steps,
            "pert_vel": jp.array([pert_lin] * 3 + [pert_ang] * 3),
            "pert_dir": jp.zeros(6, dtype=float),
            "last_pert_step": jp.array([-jp.inf], dtype=float),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/success"] = jp.zeros((), dtype=float)
        metrics["success_count"] = jp.zeros((), dtype=float)

        obs = self._get_obs(data, info)
        rew, done = jp.zeros(2)
        return mjx_env.State(data, obs, rew, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        delta = action * self._config.action_scale
        motor_targets = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)
        motor_targets = (
            self._config.ema_alpha * motor_targets
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        cube_off_floor = ~self._cube_in_contact_with_floor(data)
        ori_error = self._cube_orientation_error(data)
        at_goal = cube_off_floor & (ori_error < self._ORI_TOLERANCE_RAD)

        state.info["at_target_step_counter"] = jp.where(
            at_goal,
            state.info["at_target_step_counter"] + 1,
            jp.zeros((), dtype=jp.int32),
        )
        hold_steps = jp.asarray(self._config.target_hold_time / self.dt, dtype=jp.int32)
        success = state.info["at_target_step_counter"] > hold_steps

        done = self._get_termination(data)
        obs = self._get_obs(data, state.info)
        rewards = self._get_reward(data, state.info, action)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}

        state.info["steps_since_last_success"] = jp.where(
            success, 0, state.info["steps_since_last_success"] + 1
        )
        state.info["success_count"] = jp.where(
            done,
            jp.zeros((), dtype=jp.int32),
            jp.where(success, state.info["success_count"] + 1, state.info["success_count"]),
        )
        state.metrics["success_count"] = success.astype(float)
        state.info["step"] += 1
        state.metrics["reward/success"] = success.astype(float)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        rew = sum(rewards.values()) * self.dt
        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        return jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @staticmethod
    def r_fingertip_pos_per_tip(fingertip_dist: jax.Array, cube_half_size: float) -> jax.Array:
        return reward.tolerance(
            fingertip_dist, (0, cube_half_size), margin=0.1, sigmoid="reciprocal",
        )

    @staticmethod
    def r_joint_vel(hand_qvel: jax.Array) -> jax.Array:
        vel_tolerance = 0.3
        max_velocity = 0.8
        excess = jp.maximum(0.0, jp.abs(hand_qvel) - vel_tolerance)
        return jp.sum((excess / (max_velocity - vel_tolerance)) ** 2)

    @staticmethod
    def r_wrist_vel(wrist_qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(wrist_qvel))

    @staticmethod
    def r_cube_orientation(ori_error: jax.Array, tolerance_rad: float) -> jax.Array:
        return reward.tolerance(ori_error, (0, tolerance_rad), margin=1.0, sigmoid="gaussian")

    @staticmethod
    def r_cube_on_floor(cube_on_floor: jax.Array) -> jax.Array:
        return cube_on_floor.astype(float)

    @staticmethod
    def r_action_rate(action: jax.Array) -> jax.Array:
        return jp.sum(jp.square(action))

    def _get_reward(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        action: jax.Array,
    ) -> dict[str, jax.Array]:
        cube_pos = self.get_cube_position(data)
        cube_ori_error = self._cube_orientation_error(data)
        cube_on_floor = self._cube_in_contact_with_floor(data)
        cube_off_floor = (~cube_on_floor).astype(float)

        fingertip_distances = jp.linalg.norm(
            self.get_fingertip_positions(data).reshape(-1, 3) - cube_pos, axis=1
        )
        fingertip_reward = jp.sum(
            self.r_fingertip_pos_per_tip(fingertip_distances, self._geom.cube_half_size)
        )

        return {
            "fingertip_pos": fingertip_reward,
            "cube_ori": cube_off_floor * self.r_cube_orientation(cube_ori_error, self._ORI_TOLERANCE_RAD),
            "joint_vel": self.r_joint_vel(data.qvel[self._hand_dqids]),
            "wrist_vel": self.r_wrist_vel(data.qvel[self._wrist_dqids]),
            "action_rate": self.r_action_rate(action),
            "cube_on_floor": self.r_cube_on_floor(cube_on_floor),
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    _TIP_FORCE_SCALE = 10.0
    _ORI_TOLERANCE_RAD = 5.0 * jp.pi / 180.0

    _TIP_FORCE_SENSORS = [
        "rl_dg_1_tip_cube_force",
        "rl_dg_2_tip_cube_force",
        "rl_dg_3_tip_cube_force",
        "rl_dg_4_tip_cube_force",
        "rl_dg_5_tip_cube_force",
    ]

    def _obs_fingertip_forces(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        forces = jp.array([
            jp.sum(jp.linalg.norm(
                mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
                axis=1,
            ))
            for name in self._TIP_FORCE_SENSORS
        ])
        return forces / self._TIP_FORCE_SCALE

    def _obs_fingertip_force_dirs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        dirs = []
        for name in self._TIP_FORCE_SENSORS:
            net = jp.sum(
                mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
                axis=0,
            )
            magnitude = jp.linalg.norm(net)
            dirs.append(jp.where(magnitude > 1e-3, net / magnitude, jp.zeros(3)))
        return jp.concatenate(dirs)

    def _obs_goal_quat(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return info["goal_quat"]

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return jp.concatenate([
            data.qpos[self._hand_qids],        # 26: true joint angles
            data.qvel[self._hand_dqids],        # 26: true joint velocities
            self.get_cube_position(data),       # 3:  true cube position
            self.get_cube_linvel(data),         # 3:  cube linear velocity
            self.get_cube_angvel(data),         # 3:  cube angular velocity
            self.get_fingertip_positions(data), # 15: fingertip positions
            self.get_palm_position(data),       # 3:  palm position
            info["motor_targets"],              # 26: motor targets
            info["goal_quat"],                  # 4:  goal orientation
            self._obs_fingertip_forces(data, info),  # 5: per-tip contact force magnitudes
        ])  # total: 114

    # ------------------------------------------------------------------
    # Contact helpers
    # ------------------------------------------------------------------

    def _contact_geoms(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        impl = data._impl
        if hasattr(impl, 'contact'):
            return impl.contact.geom1, impl.contact.geom2
        else:
            g = impl.contact__geom
            return g[:, 0], g[:, 1]

    def _cube_in_contact_with_floor(self, data: mjx.Data) -> jax.Array:
        g1, g2 = self._contact_geoms(data)
        floor_id, cube_id = self._floor_geom_id, self._cube_geom_id
        return jp.any(
            ((g1 == floor_id) & (g2 == cube_id))
            | ((g1 == cube_id) & (g2 == floor_id))
        )

    def _cube_lin_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_linvel(data))

    def _cube_ang_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_angvel(data))

    def _maybe_apply_perturbation(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            d = jax.random.normal(rng, (6,))
            return d / jp.linalg.norm(d)

        def get_xfrc(
            state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
        ) -> jax.Array:
            u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
            force = (
                u_t * self._cube_mass * state.info["pert_vel"]
                / (state.info["pert_duration_steps"] * self.dt)
            )
            xfrc = jp.zeros((self.mjx_model.nbody, 6))
            return xfrc.at[self._cube_body_id].set(force * pert_dir)

        step, last_pert_step = state.info["step"], state.info["last_pert_step"]
        start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
        start_pert &= step != 0
        last_pert_step = jp.where(start_pert, step, last_pert_step)
        duration = jp.clip(step - last_pert_step, 0, 100_000)
        in_pert_interval = duration < state.info["pert_duration_steps"]

        pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
        xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

        state.info["pert_dir"] = pert_dir
        state.info["last_pert_step"] = last_pert_step
        return state.replace(data=state.data.replace(xfrc_applied=xfrc))


# Register "goal_quat" if not already registered (pick_and_place registers it
# when imported; this guard handles standalone use of this module).
try:
    obs_module.register(obs_module.ObsComponent(
        "goal_quat",
        DownwardsRotateZ._obs_goal_quat,
        size=4,
        description="goal orientation quaternion",
    ))
except ValueError:
    pass  # already registered by pick_and_place


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = DownwardsRotateZ().mj_model
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
