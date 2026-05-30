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
"""Base class for pick-and-place task with the Tesollo hand."""

import abc
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
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_pick_and_place_constants as consts,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=180,
        target_hold_time=1.5,
        target_radius=0.008,
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.0,
                joint_vel=0.0,
                cube_pos=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                fingertip_pos=0.2,
                cube_pos=4.0,
                cube_ori=0.5,
                cube_height=0.0,
                joint_vel=-0.01,
                wrist_vel=-0.01,
                cube_dropped=0.0,
            ),
            success_reward=10.0,
        ),
        pert_config=config_dict.create(
            enable=False,
            linear_velocity_pert=[0.0, 3.0],
            angular_velocity_pert=[0.0, 0.5],
            pert_duration_steps=[1, 100],
            pert_wait_steps=[60, 150],
        ),
        kp_scale=1.0,
        impl="warp",
        nconmax=200 * 8192,
        njmax=2200,
    )


class PickAndPlaceBase(tesollo_hand_base.TesolloHandGraspEnv, abc.ABC):
    """Base for pick-and-place: grasp a cube and transport it to a goal position.

    Subclasses implement _get_obs to define the observation space.
    All task logic (reset, step, reward, termination, domain randomization)
    lives here.
    """

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
        if self._config.kp_scale != 1.0:
            s = self._config.kp_scale
            self._mj_model.actuator_gainprm[:, 0] *= s
            self._mj_model.actuator_biasprm[:, 1] *= s
            self._mj_model.actuator_biasprm[:, 2] *= s
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
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
        self._table_geom_id = self._mj_model.geom("table_top").id
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        non_hand_bodies = {"world", "cube", "table", "goal"}
        self._hand_geom_ids = jp.array([
            g for g in range(self._mj_model.ngeom)
            if self._mj_model.geom(g).contype != 0
            and self._mj_model.body(self._mj_model.geom_bodyid[g]).name not in non_hand_bodies
            and g != self._floor_geom_id
        ])
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]
        self._cube_init = self._init_q[self._cube_qids]
        self._cube_init_z = float(self._cube_init[2])
        self._geom = consts.SceneGeometry.from_mj_model(self._mj_model)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation: ...

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        # q_hand = jp.clip(
        #     self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        #     self._lowers,
        #     self._uppers,
        # )
        q_hand = self._default_pose
        v_hand = jp.zeros(consts.NV)

        rng, p_rng = jax.random.split(rng)
        start_pos = self._cube_init[:3] + jax.random.uniform(
            p_rng, (3,), minval=-0.01, maxval=0.01
        )
        # q_cube = jp.concatenate([start_pos, self._cube_init[3:]])
        q_cube = self._cube_init
        v_cube = jp.zeros(6)

        rng, goal_rng, goal_rot_rng = jax.random.split(rng, 3)
        goal_xy = jax.random.uniform(
            goal_rng, (2,),
            minval=jp.array([self._geom.goal_x_min, self._geom.goal_y_min]),
            maxval=jp.array([self._geom.goal_x_max, self._geom.goal_y_max]),
        )
        # goal_pos = jp.array([goal_xy[0], goal_xy[1], self._geom.goal_z])
        random_z = jax.random.uniform(
            goal_rng, (), minval=self._geom.goal_z + 0.08, maxval=self._geom.goal_z + 0.15
        )
        # goal_pos = jp.array([goal_xy[0], goal_xy[1], random_z])
        goal_pos = jp.array([self._cube_init[0], self._cube_init[1], 0.35])

        goal_angle = jax.random.uniform(goal_rot_rng, minval=0.0, maxval=2 * jp.pi)
        goal_quat = jp.array([jp.cos(goal_angle / 2), 0.0, 0.0, jp.sin(goal_angle / 2)])
        # goal_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])
        print("reset: goal_pos =", goal_pos)
        print("reset: goal_quat =", goal_quat)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=q_hand,
            qvel=qvel,
            mocap_pos=goal_pos,
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
            "goal_pos": goal_pos,
            "goal_quat": goal_quat,
            "last_ground_cube_pos": start_pos,
            "cube_left_floor": jp.zeros((), dtype=bool),
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

        cube_pos = self.get_cube_position(data)
        cube_on_ground = self._cube_in_contact_with_floor(data)
        state.info["last_ground_cube_pos"] = jp.where(
            cube_on_ground, cube_pos, state.info["last_ground_cube_pos"]
        )
        state.info["cube_left_floor"] = state.info["cube_left_floor"] | ~cube_on_ground
 

        done = self._get_termination(data)
        obs = self._get_obs(data, state.info)
        rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}

        at_goal = jp.linalg.norm(cube_pos - state.info["goal_pos"]) < self._config.target_radius
        state.info["at_target_step_counter"] = jp.where(
            at_goal,
            state.info["at_target_step_counter"] + 1,
            jp.zeros((), dtype=jp.int32),
        )
        hold_steps = jp.asarray(self._config.target_hold_time / self.dt, dtype=jp.int32)
        success = state.info["at_target_step_counter"] > hold_steps

        rew = sum(rewards.values()) * self.dt
        # rew += success * self._config.reward_config.success_reward

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

        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        return jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

    # ------------------------------------------------------------------
    # Reward-term formulas (scalar inputs → scalar output)
    #
    # These are the single source of truth for the reward shape.
    # `_get_reward` extracts the relevant scalars from `data`/`info` and
    # delegates here; visualisation/analysis code can call them directly.
    # ------------------------------------------------------------------

    @staticmethod
    def r_cube_pos(cube_target_error: jax.Array, target_radius: float) -> jax.Array:
        # margin=0.032, value_at_margin=0.5: yields reward ≈ 0.10 at the nearest
        # possible start (~0.285 m away) via 1/(1 + d/margin).
        return reward.tolerance(
            cube_target_error, (0, target_radius), margin=0.032,
            sigmoid="reciprocal", value_at_margin=0.5,
        )

    @staticmethod
    def r_fingertip_pos_per_tip(fingertip_dist: jax.Array) -> jax.Array:
        return reward.tolerance(
            fingertip_dist, (0, 0.035), margin=0.1, sigmoid="reciprocal",
        )

    @classmethod
    def r_cube_still_at_goal(
        cls,
        cube_target_error: jax.Array,
        cube_lin_vel: jax.Array,
    ) -> jax.Array:
        return (
            cls.r_at_goal_factor(cube_target_error)
            * jp.exp(-cube_lin_vel * 12.0)
        )

    @staticmethod
    def r_release_at_goal(
        cube_target_error: jax.Array, hand_open: jax.Array
    ) -> jax.Array:
        near_goal = (cube_target_error < 0.1).astype(float)
        return near_goal * hand_open

    @staticmethod
    def r_joint_vel(hand_qvel: jax.Array) -> jax.Array:
        max_velocity = 5.0
        vel_tolerance = 1.0
        return jp.sum((hand_qvel / (max_velocity - vel_tolerance)) ** 2)

    @staticmethod
    def r_wrist_vel(wrist_qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(wrist_qvel))

    @staticmethod
    def r_cube_orientation(ori_error: jax.Array) -> jax.Array:
        # tolerance band: ≤5° (0.087 rad); gaussian gives steep decay beyond it.
        # margin=1.0 rad (≈57°): reward ≈ 0.1 at 62° total error.
        return reward.tolerance(ori_error, (0, 0.087), margin=1.0, sigmoid="gaussian")

    @staticmethod
    def r_cube_height(cube_z: jax.Array, init_z: float, goal_z: float) -> jax.Array:
        """1 at goal_z, linear ramp from 0 at init_z, hard-zero above goal_z."""
        ramp = jp.clip((cube_z - init_z) / (goal_z - init_z), 0.0, 1.0)
        return jp.where(cube_z > goal_z, 0.0, ramp)

    @staticmethod
    def r_cube_dropped(cube_on_floor: jax.Array) -> jax.Array:
        return cube_on_floor.astype(float)

    def _get_reward(
        self,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        cube_pos = self.get_cube_position(data)
        goal_pos = info["goal_pos"]
        cube_pos_error = jp.linalg.norm(cube_pos - goal_pos)
        cube_ori_error = self._cube_orientation_error(data)
        fingertip_distances = jp.linalg.norm(
            self.get_fingertip_positions(data).reshape(-1, 3) - cube_pos, axis=1
        )

        fingertip_reward = jp.sum(self.r_fingertip_pos_per_tip(fingertip_distances))

        return {
            "fingertip_pos": fingertip_reward,
            "cube_pos": self.r_cube_pos(cube_pos_error, self._config.target_radius),
            "cube_ori": self.r_cube_orientation(cube_ori_error),
            "cube_height": self.r_cube_height(cube_pos[2], self._cube_init_z, self._geom.goal_z),
            "joint_vel": self.r_joint_vel(data.qvel[self._hand_dqids]),
            "wrist_vel": self.r_wrist_vel(data.qvel[self._wrist_dqids]),
            "cube_dropped": self.r_cube_dropped(
                self._cube_in_contact_with_floor(data) & info["cube_left_floor"]
            )
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _maybe_apply_obs_noise(
        self,
        joint_angles: jax.Array,
        joint_vel: jax.Array,
        cube_pos: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        level = self._config.obs_noise.level
        scales = self._config.obs_noise.scales
        info["rng"], k1, k2, k3 = jax.random.split(info["rng"], 4)

        def add_noise(arr, key, scale):
            return arr + (2 * jax.random.uniform(key, arr.shape) - 1) * level * scale

        return (
            add_noise(joint_angles, k1, scales.joint_pos),
            add_noise(joint_vel, k2, scales.joint_vel),
            add_noise(cube_pos, k3, scales.cube_pos),
        )

    def _obs_joint_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._hand_qids]

    def _obs_joint_velocities(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._hand_dqids]

    def _obs_motor_targets(self, info: dict[str, Any]) -> jax.Array:
        return info["motor_targets"]
    
    def _obs_motor_deltas(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return info["motor_targets"] - data.qpos[self._hand_qids]

    def _obs_cube_pos(self, data: mjx.Data) -> jax.Array:
        return self.get_cube_position(data)

    def _obs_cube_to_goal(self, cube_pos: jax.Array, info: dict[str, Any]) -> jax.Array:
        return info["goal_pos"] - cube_pos

    def _cube_orientation_error(self, data: mjx.Data):
        cube_ori = self.get_cube_orientation(data)
        cube_goal_ori = self.get_cube_goal_orientation(data)
        quat_diff = math.quat_mul(cube_ori, math.quat_inv(cube_goal_ori))
        quat_diff = math.normalize(quat_diff)
        return 2.0 * jp.asin(jp.clip(jp.linalg.norm(quat_diff[1:]), max=1.0))

    def _obs_goal_pos(self, info: dict[str, Any]) -> jax.Array:
        return info["goal_pos"]

    def _obs_goal_quat(self, info: dict[str, Any]) -> jax.Array:
        return info["goal_quat"]

    def _obs_last_ground_cube_pos(self, info: dict[str, Any]) -> jax.Array:
        """Last observed cube position while it was resting on the floor."""
        return info["last_ground_cube_pos"]

    def _obs_fingertip_forces(self, data: mjx.Data) -> jax.Array:
        """Per-fingertip contact force magnitude vs cube, normalized by 10 N. Shape: (5,)."""
        tip_sensors = [
            "rl_dg_1_tip_cube_force",
            "rl_dg_2_tip_cube_force",
            "rl_dg_3_tip_cube_force",
            "rl_dg_4_tip_cube_force",
            "rl_dg_5_tip_cube_force",
        ]
        forces = jp.array([
            jp.sum(jp.linalg.norm(
                mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
                axis=1,
            ))
            for name in tip_sensors
        ])
        return forces / 10.0

    def _obs_total_contact_force(self, data: mjx.Data) -> jax.Array:
        """Sum of all hand-cube contact force magnitudes, normalized by 10 N. Shape: (1,)."""
        forces = mjx_env.get_sensor_data(
            self.mj_model, data, "cube_force"
        ).reshape(-1, 3)
        return jp.sum(jp.linalg.norm(forces, axis=1), keepdims=True) / 10.0

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return jp.concatenate([
            data.qpos[self._hand_qids],       # 26: true joint angles
            data.qvel[self._hand_dqids],       # 26: true joint velocities
            self.get_cube_position(data),      # 3:  true cube position
            self.get_cube_linvel(data),        # 3:  cube linear velocity
            self.get_cube_angvel(data),        # 3:  cube angular velocity
            self.get_fingertip_positions(data),# 15: fingertip positions
            self.get_palm_position(data),      # 3:  palm position
            info["motor_targets"],             # 26: motor targets
            info["goal_pos"],                  # 3:  goal position
            info["goal_quat"],                 # 4:  goal orientation
        ])  # total: 112

    # ------------------------------------------------------------------
    # Cost and utility methods
    # ------------------------------------------------------------------

    def _contact_geoms(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Returns (geom1_ids, geom2_ids) arrays for all contact slots."""
        impl = data._impl
        if hasattr(impl, 'contact'):          # JAX backend
            return impl.contact.geom1, impl.contact.geom2
        else:                                  # Warp backend
            g = impl.contact__geom            # (nconmax, 2)  vec2i → int32
            return g[:, 0], g[:, 1]

    def _hand_in_contact_with_cube(self, data: mjx.Data) -> jax.Array:
        g1, g2 = self._contact_geoms(data)
        cube_id = self._cube_geom_id
        is_cube = (g1 == cube_id) | (g2 == cube_id)
        other = jp.where(g1 == cube_id, g2, g1)
        is_hand = jp.any(other[:, None] == self._hand_geom_ids[None, :], axis=1)
        return jp.any(is_cube & is_hand)

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
