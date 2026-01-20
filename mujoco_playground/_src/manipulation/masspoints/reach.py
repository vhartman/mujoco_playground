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
"""Pressing task with two masspoints."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.masspoints import (
    base_reach as masspoints_reach,
)
from mujoco_playground._src.manipulation.masspoints import (
    masspoint_reach_constants as consts,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=1000,
        success_threshold=0.01,
        history_len=5,
        future_goal_len=4,
        obs_noise=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.025,
            ),
            random_ori_injection_prob=0.0,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                termination=-100.0,
                position=0.0,
                neg_goal_distance=0.0,
                hand_pose=-100.,
                action_rate=-1.,
                joint_vel=-100.,
                energy=-0.001,
                pressing_cost=-0.0,
            ),
            success_reward=100.0,
        ),
        impl="jax",
        nconmax=200 * 8192,
        njmax=1024,
    )


class KeyboardMasspointReach(masspoints_reach.MasspointsReachEnv):
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
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._default_pose = self._init_q[self._hand_qids]

        KEY_NAMES = []
        for i in range(len(self._goal_ids)):
            KEY_NAMES.append(f"key_joint_{i}")

        self._key_ids = mjx_env.get_qpos_ids(self.mj_model, KEY_NAMES)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # randomize the next goals
        rng, goal_rng = jax.random.split(rng, 2)
        goal_order = jax.random.randint(
            goal_rng, self._config.future_goal_len, 0, len(self._goal_ids)
        )

        # Randomize the hand pose.
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        q_toggles = jp.zeros(len(self._goal_ids))

        qpos = jp.concatenate([q_hand, q_toggles])
        qvel = jp.concatenate([v_hand, q_toggles])

        goal_indicators = jp.zeros((2, 3))
        if self._config.future_goal_len == 1:
            goal_indicators = goal_indicators.at[0].set(self._goal_locations[0])
        else:
            goal_indicators = self._goal_locations[goal_order[:2]]

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=q_hand,
            qvel=qvel,
            mocap_pos=goal_indicators,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": data.ctrl,
            "qpos_error_history": jp.zeros(self._config.history_len * 6),
            "goal_order": goal_order,
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

    def get_goal_reached(self, data, current_goal):
        # fingertip_positions = self.get_fingertip_positions(data).reshape(-1, 3)
        # goal_distance = jp.linalg.norm(
        #     fingertip_positions - self._goal_locations[current_goal], axis=1
        # )

        # return jp.any(goal_distance < self._config.success_threshold)
        keys_status = data.qpos[self._key_ids]
        return keys_status[current_goal] < -0.015

    def get_nothing_else_pressed(self, data, current_goal):
        # fingertip_positions = self.get_fingertip_positions(data).reshape(-1, 3)
        # z_positions = fingertip_positions[:, 2]

        # goal_distance = jp.linalg.norm(
        #     fingertip_positions - self._goal_locations[current_goal], axis=1
        # )

        # return jp.all(
        #     (z_positions > -0.03) | (goal_distance < self._config.success_threshold)
        # )
        # return jp.all((z_positions > 0.1) | (goal_distance < self._config.success_threshold))
        # keys_status = data.qpos[self._key_ids]
        # return jp.all((keys_status > -0.1) | (self._goal_ids == current_goal))
        key_ids = self._key_ids
        mask = key_ids != current_goal  # exclude that key
        # masked_keys_status = data.qpos[key_ids[mask]]
        # return jp.all(masked_keys_status > -0.1)
        valid = jp.where(mask, data.qpos[key_ids] > -0.01, True)

        return jp.all(valid)

    # def get_pressing_cost(self, data):
    #     keys_status = data.qpos[self._key_ids]

    #     return jp.sum(keys_status < 0.001)
    def get_pressing_cost(self, data, current_goal):
        key_ids = self._key_ids
        mask = key_ids != current_goal  # exclude that key
        keys_status = data.qpos[key_ids] * mask
        # return jp.sum((keys_status < -0.005) * jp.abs(keys_status))
        return jp.sum(jp.abs(keys_status) + jp.abs(keys_status)**2)
        # return jp.sum((keys_status < -0.005) * (jp.abs(keys_status) + jp.abs(keys_status)**2))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
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

        success = self.get_goal_reached(
            data, state.info["goal_order"][0]
        )#  & self.get_nothing_else_pressed(data, state.info["goal_order"][0])

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

        # Sample a new goal and move the existing ones oen up.
        state.info["rng"], goal_rng = jax.random.split(state.info["rng"])

        new_goal = jax.random.randint(goal_rng, (), 0, len(self._goal_ids))

        shifted_goals = jp.roll(state.info["goal_order"], -1)
        shifted_goals_with_new_goal = shifted_goals
        shifted_goals_with_new_goal = shifted_goals_with_new_goal.at[-1].set(new_goal)

        # print("success", success.shape)
        # print("goal_order", state.info["goal_order"].shape)

        new_goal_order = jp.where(
            success, shifted_goals_with_new_goal, state.info["goal_order"]
        )

        state.info["goal_order"] = new_goal_order

        # print(state.data.mocap_pos)

        goal_indicators = jp.zeros((2, 3))
        if self._config.future_goal_len == 1:
            goal_indicators = goal_indicators.at[0, :].set(
                self._goal_locations[new_goal_order[0]]
            )
        else:
            goal_indicators = self._goal_locations[new_goal_order[:2]]

        data = data.replace(mocap_pos=goal_indicators)
        # data = data.replace(mocap_pos=self._goal_locations[new_goal_order[:2]])

        # state.info["goal_order"] = jp.where(
        #     success,
        #     new_goal,
        #     state.info["goal_order"],
        # )

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
        # del info  # Unused.
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        return nans

        # current_goal = info["goal_order"][0]
        # mask = self._key_ids != current_goal  # exclude that key
        # keys_status = data.qpos[self._key_ids] * mask

        # return nans | jp.any(keys_status < -0.015)

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
            jp.roll(info["qpos_error_history"], 6)
            .at[:6]
            .set(noisy_joint_angles - info["motor_targets"])
        )
        info["qpos_error_history"] = qpos_error_history

        fingertip_positions = self.get_fingertip_positions(data).reshape(-1, 3)
        # TODO: Add distances for all goals

        all_goal_distances = []
        for goal_id in info["goal_order"]:
            goal_distance = jp.linalg.norm(
                fingertip_positions - self._goal_locations[goal_id], axis=1
            )
            all_goal_distances.append(goal_distance)

        # current_goal = info["goal_order"][0]
        # goal_distance = jp.linalg.norm(
        #     fingertip_positions - self._goal_locations[current_goal], axis=1
        # )

        goal_position_error = jp.min(goal_distance, keepdims=True)

        future_goal_poses = self._goal_locations[info["goal_order"]].flatten()

        state = jp.concatenate(
            [
                noisy_joint_angles,  # 6
                qpos_error_history,  # 6 * history_len
                info["last_act"],  # 6
                self.get_fingertip_positions(data),
                # goal_position_error,
                # jp.array(all_goal_distances).flatten(),
                future_goal_poses,
            ]
        )

        privileged_state = jp.concatenate(
            [
                state,
                data.qpos[self._hand_qids],
                data.qvel[self._hand_dqids],
                self.get_fingertip_positions(data),
                goal_position_error,
                future_goal_poses,
            ]
        )

        # return state
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

        fingertip_positions = self.get_fingertip_positions(data).reshape(-1, 3)
        # current_goal = info["goal_order"][0]
        # goal_distance = jp.linalg.norm(
        #     fingertip_positions - self._goal_locations[current_goal], axis=1
        # )
        # fingertip_goal_reward = jp.max(
        #     reward.tolerance(goal_distance, (0, 0.005), margin=0.1, sigmoid="linear")
        # )
        goal_distances = []
        fingertip_goal_rewards = []
        for i, goal_id in enumerate(info["goal_order"]):
            # current_goal = info["goal_order"][0]
            weight = 1.0 / (1.0 + i)
            goal_distance = jp.linalg.norm(
                fingertip_positions - self._goal_locations[goal_id], axis=1
            )
            fingertip_goal_reward = weight * jp.max(
                reward.tolerance(
                    goal_distance, (0, 0.005), margin=0.05, sigmoid="linear"
                )
            )

            xy_goal_distance = jp.linalg.norm(
                (fingertip_positions - self._goal_locations[goal_id, :])[:, :2], axis=1
            )
            goal_distances.append(
                weight
                * reward.tolerance(
                    xy_goal_distance, (0, 0.005), margin=0.5, sigmoid="linear"
                )
            )
            # goal_distances.append(-weight * xy_goal_distance)

            fingertip_goal_rewards.append(fingertip_goal_reward)

        pressing_cost = self.get_pressing_cost(data, info["goal_order"][0])

        return {
            "termination": terminated,
            "hand_pose": hand_pose_reward,
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "joint_vel": self._cost_joint_vel(data),
            "energy": self._cost_energy(
                data.qvel[self._hand_dqids], data.actuator_force
            ),
            "position": jp.sum(jp.array(fingertip_goal_rewards)),
            "neg_goal_distance": jp.sum(jp.array(goal_distances)),
            "pressing_cost": pressing_cost,
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
            + jax.random.uniform(key, shape=(6,), minval=-0.05, maxval=0.05)
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(6,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(6,), minval=1.0, maxval=1.05
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
            key, (6,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction,
            body_mass,
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
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes
