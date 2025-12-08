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
"""Reorient task for tesollo hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_reach as tesollo_hand_reach
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_reach_constants as consts,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=1000,
        success_threshold=0.1,
        joint_vel_threshold=0.5,
        vel_threshold=0.5,
        ang_vel_threshold=0.5,
        history_len=5,
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.025,
            ),
            random_ori_injection_prob=0.0,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                termination=-100.0,
                hand_pose=-0.5,
                wrist_pose=-1.0,
                action_rate=-0.005,
                joint_vel=-0.01,
                energy=-1e-3,
                wrist_vel=-0.1,
            ),
            success_reward=100.0,
        ),
        pert_config=config_dict.create(
            enable=False,
            linear_velocity_pert=[0.0, 3.0],
            angular_velocity_pert=[0.0, 0.5],
            pert_duration_steps=[1, 100],
            pert_wait_steps=[60, 150],
        ),
        impl="jax",
        nconmax=200 * 8192,
        njmax=1024,
    )


class KeyboardReach(tesollo_hand_reach.TesolloHandReachEnv):
    """Reach a series of points with the fingertips."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
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
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # randomize the next goals

        # Randomize the hand pose.
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        data = mjx_env.make_data(
            self._mj_model,
            qpos=q_hand,
            ctrl=q_hand,
            qvel=v_hand,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
        pert_wait_steps = jax.random.randint(
            pert1,
            (1,),
            minval=self._config.pert_config.pert_wait_steps[0],
            maxval=self._config.pert_config.pert_wait_steps[1],
        )
        pert_duration_steps = jax.random.randint(
            pert2,
            (1,),
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
        pert_velocity = jp.array([pert_lin] * 3 + [pert_ang] * 3)

        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": data.ctrl,
            "qpos_error_history": jp.zeros(self._config.history_len * 24),
            "goal_quat_dquat": jp.zeros(3),
            # Perturbation.
            "pert_wait_steps": pert_wait_steps,
            "pert_duration_steps": pert_duration_steps,
            "pert_vel": pert_velocity,
            "pert_dir": jp.zeros(6, dtype=float),
            "last_pert_step": jp.array([-jp.inf], dtype=float),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/success"] = jp.zeros((), dtype=float)
        metrics["steps_since_last_success"] = 0
        metrics["success_count"] = 0

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        # Apply control and step the physics.
        delta = action * self._config.action_scale
        motor_targets = state.data.ctrl + delta
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        motor_targets = (
            self._config.ema_alpha * motor_targets
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        # hand_qvel = data.qvel[self._hand_dqids]
        # hand_qvel_norm = jp.sum(hand_qvel ** 2)

        success = False
        state.info["steps_since_last_success"] = jp.where(
            success, 0, state.info["steps_since_last_success"] + 1
        )
        state.info["success_count"] = jp.where(
            success, state.info["success_count"] + 1, state.info["success_count"]
        )
        state.metrics["steps_since_last_success"] = state.info[
            "steps_since_last_success"
        ]
        state.metrics["success_count"] = state.info["success_count"]

        done = self._get_termination(data, state.info)
        obs = self._get_obs(data, state.info)

        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

        # Sample a new goal orientation.
        state.info["rng"], goal_rng = jax.random.split(state.info["rng"])
        state.info["goal_quat_dquat"] = jp.where(
            success,
            3 + jax.random.uniform(goal_rng, (3,), minval=-2, maxval=2),
            state.info["goal_quat_dquat"] * 0.8,
        )
        goal_quat = math.quat_integrate(
            state.data.mocap_quat[0],
            state.info["goal_quat_dquat"],
            2 * jp.array(self.dt),
        )
        data = data.replace(mocap_quat=jp.array([goal_quat]))
        state.metrics["reward/success"] = success.astype(float)
        reward += success * self._config.reward_config.success_reward

        # Update info and metrics.
        state.info["step"] += 1
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        return False

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        # Hand joint angles.
        joint_angles = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.obs_noise.level
            * self._config.obs_noise.scales.joint_pos
        )

        # Joint position error history.
        qpos_error_history = (
            jp.roll(info["qpos_error_history"], 24)
            .at[:24]
            .set(noisy_joint_angles - info["motor_targets"])
        )
        info["qpos_error_history"] = qpos_error_history

        state = jp.concatenate(
            [
                noisy_joint_angles,  # 24
                qpos_error_history,  # 24 * history_len
                info["last_act"],  # 24
            ]
        )

        privileged_state = jp.concatenate(
            [
                state,
                data.qpos[self._hand_qids],
                data.qvel[self._hand_dqids],
                self.get_fingertip_positions(data),
                info["pert_dir"],
                data.xfrc_applied[self._cube_body_id],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    # Reward terms.

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del done, metrics  # Unused.

        terminated = self._get_termination(data, info)

        hand_pose_reward = jp.sum(
            jp.square(data.qpos[self._hand_qids] - self._default_pose)
        )

        wrist_pose_reward = jp.sum(
            jp.square(data.qpos[self._wrist_qids] - self._default_wrist_pose)
        )

        cost_wrist_vel = jp.sum(
            jp.square(data.qvel[self._wrist_qids])
        )

        return {
            "termination": terminated,
            "hand_pose": hand_pose_reward,
            "wrist_pose": wrist_pose_reward,
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "joint_vel": self._cost_joint_vel(data),
            "wrist_vel": cost_wrist_vel,
            "energy": self._cost_energy(
                data.qvel[self._hand_dqids], data.actuator_force
            )
        }

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        c1 = jp.sum(jp.square(act - last_act))
        c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
        return c1 + c2

    def _cost_joint_vel(self, data: mjx.Data) -> jax.Array:
        max_velocity = 5.0
        vel_tolerance = 1.0
        hand_qvel = data.qvel[self._hand_dqids]
        return jp.sum((hand_qvel / (max_velocity - vel_tolerance)) ** 2)

    # Perturbation.

    def _maybe_apply_perturbation(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            directory = jax.random.normal(rng, (6,))
            return directory / jp.linalg.norm(directory)

        def get_xfrc(
            state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
        ) -> jax.Array:
            u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
            force = (
                u_t
                * self._cube_mass
                * state.info["pert_vel"]
                / (state.info["pert_duration_steps"] * self.dt)
            )
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
            return xfrc_applied

        step, last_pert_step = state.info["step"], state.info["last_pert_step"]
        start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
        start_pert &= step != 0  # No perturbation at the beginning of the episode.
        last_pert_step = jp.where(start_pert, step, last_pert_step)
        duration = jp.clip(step - last_pert_step, 0, 100_000)
        in_pert_interval = duration < state.info["pert_duration_steps"]

        pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
        xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

        state.info["pert_dir"] = pert_dir
        state.info["last_pert_step"] = last_pert_step
        data = state.data.replace(xfrc_applied=xfrc)
        return state.replace(data=data)


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = KeyboardReach().mj_model
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        #   "palm",
        "rl_dg_1_1",
        "rl_dg_1_2",
        "rl_dg_1_3",
        "rl_dg_1_4",
        "rl_dg_2_1",
        "rl_dg_2_2",
        "rl_dg_2_3",
        "rl_dg_2_4",
        "rl_dg_3_1",
        "rl_dg_3_2",
        "rl_dg_3_3",
        "rl_dg_3_4",
        "rl_dg_4_1",
        "rl_dg_4_2",
        "rl_dg_4_3",
        "rl_dg_4_4",
        "rl_dg_5_1",
        "rl_dg_5_2",
        "rl_dg_5_3",
        "rl_dg_5_4",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    silicone_geoms = [
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    ]
    silicone_geom_ids = [mj_model.geom(g).id for g in silicone_geoms]

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        # Fingertip friction: =U(0.5, 1.0).
        silicone_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[silicone_geom_ids, 0].set(
            silicone_friction
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[hand_qids].set(
            qpos0[hand_qids]
            + jax.random.uniform(key, shape=(24,), minval=-0.05, maxval=0.05)
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(24,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(24,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        # Joint stiffness: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # Joint damping: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (24,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "dof_damping": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes
