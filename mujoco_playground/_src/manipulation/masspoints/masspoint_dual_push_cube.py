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
"""Panda robotiq push cube environment."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import types

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.masspoints import base_dual_push
import numpy as np

WORKSPACE_MIN = (0.3, -0.5, 0.0)
WORKSPACE_MAX = (0.75, 0.7, 0.5)
OBJ_SAMPLE_MIN = (0.4, -0.4, -0.005)
# OBJ_SAMPLE_MAX = (0.5, 0.2, 0.04)
OBJ_SAMPLE_MAX = (0.65, 0.4, 0.04)


def default_config():
  """Returns reward config for the environment."""

  return config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.005,
      episode_length=3000,
      action_repeat=4,
      action_scale=0.5,
      action_history_len=5,
      obs_history_len=1,
      noise_config=config_dict.create(
          action_min_delay=1,  # env steps
          action_max_delay=2,  # env steps
          obs_min_delay=1,  # env steps
          obs_max_delay=2,  # env steps
          noise_scales=config_dict.create(
              obj_pos=0.00,  # meters
              obj_angle=0.0,  # degrees
              robot_qpos=0.000,  # radians
              robot_qvel=0.000,  # radians/s
          ),
      ),
      reward_config=config_dict.create(
          termination_reward=-50.0,
          success_reward=500.0,
          success_wait_reward=0.001,
          success_step_count=10,
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=0.5,
              # Box goes to the target mocap.
              box_target=0.5,
              box_orientation=0.5,
              # Gripper collides with side of object, instead of top.
            #   gripper_collision_side=1.0,
              # Arm stays close to target pose.
              # Reduce joint velocity.
              joint_vel=0.0,
              # Avoid joint vel limits.
              joint_vel_limit=0.0,
              # Torque penalty of the arm.
              total_command=-0.0,
              # Reduce action rate.
              action_rate=-0.0,
              # penalty for closeness
              collision_penalty=0.
          ),
      ),
      impl="jax",
      nconmax=32 * 8192,
      njmax=256,
  )


def get_rand_dir(rng: jax.Array) -> jax.Array:
  key1, key2 = jax.random.split(rng)
  theta = jax.random.normal(key1) * 2 * jp.pi
  phi = jax.random.normal(key2) * jp.pi
  x = jp.sin(phi) * jp.cos(theta)
  y = jp.sin(phi) * jp.sin(theta)
  z = jp.cos(phi)
  return jp.array([x, y, z])

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "masspoints"
SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx_dual_box_pushing.xml"

class MasspointDualPushCube(base_dual_push.MasspointsDualPushEnv):
  """Environment for pushing a cube with a Panda robot and Robotiq gripper."""

  def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

  def _get_rand_target_pos(
      self, rng: jax.Array, offset: jax.Array, init_pos: jax.Array
  ) -> jax.Array:
    min_pos = jp.array([-0.56 + 0.4, -offset*0.5, -0.005]) + init_pos
    max_pos = jp.array([-0.56 + 0.4 + offset * 0.7, offset*0.5, 0.005]) + init_pos
    pos = jax.random.uniform(rng, (3,), minval=min_pos, maxval=max_pos)
    return jp.clip(pos, np.array(OBJ_SAMPLE_MIN), np.array(OBJ_SAMPLE_MAX))

  def _get_rand_target_quat(
      self, rng: jax.Array, max_angle: jax.Array
  ) -> jax.Array:
    perturb_axis = jp.array([0.0, 0.0, 1.0], dtype=float)
    max_angle_rad = max_angle * jp.pi / 180
    perturb_theta = jax.random.uniform(rng, maxval=max_angle_rad)
    target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)
    return target_quat

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_box1_pos, rng_box1_quat, rng_target_1, rng_robot_arm, rng_target1_theta, rng_box1, rng_box2, rng_target, rng_theta = (
        jax.random.split(rng, 10)
    )

    # intialize box position
    box_1_pos = self._get_rand_target_pos(rng_box1_pos, jp.array(0.15), self._init_obj_1_pos)
    box_1_pos = box_1_pos.at[2].set(self._init_obj_1_pos[2])
    box_1_quat = self._get_rand_target_quat(rng_box1_quat, jp.array(360))

    # initialize target position
    target_1_pos = self._get_rand_target_pos(rng_target_1, jp.array(0.05), self._init_obj_1_pos)

    # initialize target orientation
    target_1_quat = self._get_rand_target_quat(rng_target1_theta, jp.array(45))
    target_1_quat = math.quat_mul(box_1_quat, target_1_quat)
    
    # intialize box position
    box_2_pos = self._get_rand_target_pos(rng_box1, jp.array(0.15), self._init_obj_2_pos)
    box_2_pos = box_2_pos.at[2].set(self._init_obj_2_pos[2])
    box_2_quat = self._get_rand_target_quat(rng_box2, jp.array(360))

    # initialize target position
    target_2_pos = self._get_rand_target_pos(rng_target, jp.array(0.05), self._init_obj_2_pos)

    # initialize target orientation
    target_2_quat = self._get_rand_target_quat(rng_theta, jp.array(45))
    target_2_quat = math.quat_mul(box_2_quat, target_2_quat)

    # initialize mjx.Data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_1_qposadr : self._obj_1_qposadr + 7]
        .set(jp.concatenate([box_1_pos, box_1_quat]))
        .at[self._obj_2_qposadr : self._obj_2_qposadr + 7]
        .set(jp.concatenate([box_2_pos, box_2_quat]))
    )
    # sample random joint position for robot arm
    init_q = init_q.at[self._masspoints_qids].set(
        jax.random.uniform(
            rng_robot_arm,
            (6,),
            minval=self._jnt_range[:, 0] * self._joint_range_init_percent_limit,
            maxval=self._jnt_range[:, 1] * self._joint_range_init_percent_limit,
        )
    )
    # jax.debug.print("{q}", q=init_q)
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        mocap_pos=jp.array([target_1_pos, target_2_pos]),
        mocap_quat=jp.array([target_1_quat, target_2_quat]),
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "success": jp.array(0.0, dtype=float),
        "success_1_box_1": jp.array(0.0, dtype=float),
        "success_2_box_1": jp.array(0.0, dtype=float),
        "success_1_box_2": jp.array(0.0, dtype=float),
        "success_2_box_2": jp.array(0.0, dtype=float),
        "subsuccess": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.reward_scales.keys()},
    }
    info = {
        "rng": rng,
        "box_1_success": jp.array(0, dtype=float),
        "box_2_success": jp.array(0, dtype=float),
        "last_action": jp.zeros(6, dtype=float),
        "action_history": jp.zeros(self._config.action_history_len * 6),
        "success_step_count": jp.array(0, dtype=int),
        "prev_step_success": jp.array(0, dtype=int),
        "curriculum_id": jp.array(0, dtype=int),
        "angle_curriculum": jp.array([20, 30, 45, 90, 135, 180], dtype=float),
        "pos_curriculum": jp.array([0.05, 0.05, 0.1, 0.2, 0.4, 0.4], dtype=float),
    }
    obs = self._get_single_obs(data, info)
    info["obs_history"] = jp.zeros(self._config.obs_history_len * obs.shape[0])

    reward, done = jp.zeros(2)
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action_history = jp.roll(state.info["action_history"], 6).at[:6].set(action)
    state.info["action_history"] = action_history

    # add action delay
    state.info["rng"], key = jax.random.split(state.info["rng"])
    action_idx = jax.random.randint(
        key,
        (1,),
        minval=self._config.noise_config.action_min_delay,
        maxval=self._config.noise_config.action_max_delay,
    )
    action_w_delay = action_history.reshape((-1, 6))[action_idx[0]]

    # get the ctrl
    ctrl = action_w_delay * self._config.action_scale
    # ctrl = jp.clip(
    #     ctrl, -self._max_torque / self._gear, self._max_torque / self._gear
    # )

    # step the physics
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    state = state.replace(data=data)

    # calculate rewards
    rewards = self._get_reward(state.data, state.info, action)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in rewards.items()
    }
    reward = sum(rewards.values())
    # reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    reward_scale_sum = sum(
        self._config.reward_config.reward_scales[k] for k in rewards
    )
    reward /= reward_scale_sum

    # termination reward
    termination = self._get_termination(state.data)
    reward += self._config.reward_config.termination_reward * termination

    # success reward
    state, success = self._get_success(state)
    sub_success = state.info["box_1_success"] + state.info["box_2_success"]
    reward += self._config.reward_config.success_wait_reward * sub_success
    reward += self._config.reward_config.success_reward * success

    rewards["subsuccess"] = sub_success * 1.0

    # finalize reward
    reward *= self.dt

    # reset target mocap if success
    state = self._reset_if_success(state, success)

    # calculate done
    state.metrics.update(out_of_bounds=termination.astype(float), **rewards)
    done = (
        termination
        | jp.isnan(state.data.qpos).any()
        | jp.isnan(state.data.qvel).any()
    )
    done = done.astype(float)

    # get observations
    obs = self._get_obs(state)

    # store info for the next step
    state.info["last_action"] = action_w_delay
    state.info["rng"], _ = jax.random.split(state.info["rng"])

    state = mjx_env.State(
        state.data, obs, reward, done, state.metrics, state.info
    )
    return state

  def _get_termination(self, data: mjx.Data):
    box_1_pos = data.xpos[self._obj_1_body]
    box_1_oob = box_1_pos[2] < -0.01
    box_1_oob |= (box_1_pos[0] > WORKSPACE_MAX[0]) | (box_1_pos[0] < WORKSPACE_MIN[0])
    box_1_oob |= (box_1_pos[1] > WORKSPACE_MAX[1]) | (box_1_pos[1] < WORKSPACE_MIN[1])
    
    box_2_pos = data.xpos[self._obj_2_body]
    box_2_oob = box_2_pos[2] < -0.01
    box_2_oob |= (box_2_pos[0] > WORKSPACE_MAX[0]) | (box_2_pos[0] < WORKSPACE_MIN[0])
    box_2_oob |= (box_2_pos[1] > WORKSPACE_MAX[1]) | (box_2_pos[1] < WORKSPACE_MIN[1])
    
    return box_1_oob | box_2_oob

  def _get_success(
      self, state: mjx_env.State
  ) -> Tuple[mjx_env.State, jax.Array, jax.Array]:
    data = state.data

    for box_name, mocap_target, obj_body in zip(["box_1", "box_2"], [self._mocap_target_1, self._mocap_target_2], [self._obj_1_body, self._obj_2_body]):
    # for box_name, mocap_target, obj_body in zip(["box_1"], [self._mocap_target_1], [self._obj_1_body]):
        target_pos = data.mocap_pos[mocap_target, :].ravel()
        target_quat = data.mocap_quat[mocap_target, :].ravel()
        box_pos = data.xpos[obj_body]
        box_quat = data.xquat[obj_body]
        ori_error = self._orientation_error(box_quat, target_quat)

        # get success condition
        success_cond_1 = jp.linalg.norm(target_pos - box_pos) < 0.05  # 3cm
        success_cond_2 = ori_error < (180 / 180 * jp.pi)  # 10 degrees
        # success_cond_3 = (
        #     state.info["success_step_count"]
        #     >= self._config.reward_config.success_step_count
        # )
        box_success = success_cond_1 & success_cond_2

        # if we have already achieved success for a box, we keep it
        state.info[f"{box_name}_success"] = jp.maximum(box_success.astype(float), state.info[f"{box_name}_success"])

        state.metrics[f"success_1_{box_name}"] = 1. * success_cond_1.astype(float)
        state.metrics[f"success_2_{box_name}"] = 1. * success_cond_2.astype(float)

    success = (state.info["box_1_success"] + state.info["box_2_success"]) > 1.5
    # success = state.info["box_1_success"]
    # success = state.info["box_2_success"]

    # report metrics
    state.metrics["success"] = success.astype(float)
    # state.metrics["success_1"] = success_cond_1.astype(float)
    # state.metrics["success_2"] = success_cond_2.astype(float)

    # # calculate success counter for next step
    # state.info["prev_step_success"] = (success_cond_1 & success_cond_2).astype(
    #     int
    # )
    # state.info["success_step_count"] = jp.where(
    #     state.info["prev_step_success"], state.info["success_step_count"] + 1, 0
    # )
    # state.info["prev_step_success"] *= 1 - success
    # state.info["success_step_count"] *= 1 - success

    # sub_success = success_cond_1 & success_cond_2
    return state, success

  def _reset_if_success(
      self, state: mjx_env.State, success: jax.Array
  ) -> mjx_env.State:
    # increase curriculum step
    state.info["curriculum_id"] += success.astype(int)
    state.info["curriculum_id"] = jp.minimum(state.info["curriculum_id"], len(state.info["pos_curriculum"])-1)
    max_pos = state.info["pos_curriculum"][state.info["curriculum_id"]]
    max_angle = state.info["angle_curriculum"][state.info["curriculum_id"]]
    
    state.info["rng"], key1, key2, key3, key4 = jax.random.split(state.info["rng"], 5)

    # sample new target position and orientation
    target_1_pos = state.data.mocap_pos[self._mocap_target_1, :].ravel()
    target_1_quat = state.data.mocap_quat[self._mocap_target_1, :].squeeze()
    new_target_1_pos = jp.where(
        success,
        self._get_rand_target_pos(key1, max_pos, self._init_obj_1_pos),
        target_1_pos,
    )
    new_target_1_quat = jp.where(
        success,
        math.quat_mul(
            state.data.xquat[self._obj_1_body],
            self._get_rand_target_quat(key2, max_angle),
        ),
        target_1_quat,
    )

    target_2_pos = state.data.mocap_pos[self._mocap_target_2, :].ravel()
    target_2_quat = state.data.mocap_quat[self._mocap_target_2, :].squeeze()
    new_target_2_pos = jp.where(
        success,
        self._get_rand_target_pos(key3, max_pos, self._init_obj_2_pos),
        target_2_pos,
    )
    new_target_2_quat = jp.where(
        success,
        math.quat_mul(
            state.data.xquat[self._obj_2_body],
            self._get_rand_target_quat(key4, max_angle),
        ),
        target_2_quat,
    )

    data = state.data.replace(
        mocap_pos=jp.array([new_target_1_pos, new_target_2_pos]),
        mocap_quat=jp.array([new_target_1_quat, new_target_2_quat]),
    )

    # resets box-wise success to zero if success == 1, keeps it at the same value otherwise
    state.info["box_1_success"] = state.info["box_1_success"] * (1 - success.astype(int))
    state.info["box_2_success"] = state.info["box_2_success"] * (1 - success.astype(int))

    return state.replace(data=data)

  def _get_reward(
      self, data: mjx.Data, info: dict[str, Any], action: jax.Array
  ) -> dict[str, jax.Array]:
    total_box_target = 0.
    total_box_orientation = 0.
    total_gripper_box = 0.

    gripper_box_rewards = jp.zeros((2,2), dtype=float)
    
    masspoint_pos = self.get_fingertip_positions(data).reshape(-1, 3)
        
    # Target, gripper, and object rewards.
    for i, (mocap_target, obj_body) in enumerate(zip([self._mocap_target_1, self._mocap_target_2], [self._obj_1_body, self._obj_2_body])):
    # for mocap_target, obj_body in zip([self._mocap_target_1], [self._obj_1_body]):
    # for mocap_target, obj_body in zip([self._mocap_target_2], [self._obj_2_body]):
        target_pos = data.mocap_pos[mocap_target, :].ravel()
        box_pos = data.xpos[obj_body]
        side_dir = box_pos - target_pos
        side_dir = math.normalize(side_dir) * 0.05 * (math.norm(side_dir) > 1e-3)
        box_side_pos = side_dir + box_pos
        # gripper_pos = data.site_xpos[self._gripper_site]
        # masspoint_pos = self.get_fingertip_positions(data)[:3]

        # print(box_pos)
        # print(masspoint_pos)
        # print(jp.linalg.norm(box_pos - masspoint_pos, axis=1))

        gripper_box = reward_util.tolerance(
                # jp.linalg.norm(box_pos - masspoint_pos, axis=1),
                jp.linalg.norm(box_side_pos - masspoint_pos, axis=1),
                # jp.linalg.norm(box_side_pos - masspoint_pos),
                (0, 0.01),
                margin=0.5,
                sigmoid="linear",
            )

        gripper_box_rewards = gripper_box_rewards.at[i, :].set(gripper_box)
        # jax.debug.print("{x}", x=gripper_box_rewards)

        box_target = reward_util.tolerance(
            jp.linalg.norm(box_pos[:2] - target_pos[:2]),
            (0, 0.005),
            margin=0.4,
            sigmoid="reciprocal",
        )

        target_quat = data.mocap_quat[mocap_target, :].squeeze()
        ori_error = self._orientation_error(data.xquat[obj_body], target_quat)
        box_orientation = reward_util.tolerance(
            ori_error, (0, 0.2), margin=jp.pi, sigmoid="reciprocal"
        )

        total_box_target += box_target
        total_box_orientation += box_orientation
        # total_gripper_box += gripper_box

    # jax.debug.print("{x}", x=gripper_box_rewards)

    total_gripper_box = jp.sum(jp.max(gripper_box_rewards, axis=0))

    # Action regularization.
    joint_vel_mse = jp.linalg.norm(
        data.qvel[self._massspoints_dqids]
    )
    joint_vel = reward_util.tolerance(
        joint_vel_mse, (0, 0.5), margin=2.0, sigmoid="reciprocal"
    )
    total_command = jp.linalg.norm(action)
    action_rate = jp.linalg.norm(action - info["last_action"])

    joints_near_vel_limits = jp.any(
        jp.logical_or(
            data.qvel[self._massspoints_dqids]
            > (self._jnt_vel_range[:, 1] * self._joint_vel_limit_percentage),
            data.qvel[self._massspoints_dqids]
            < (self._jnt_vel_range[:, 0] * self._joint_vel_limit_percentage),
        )
    )

    distance = jp.linalg.norm(masspoint_pos[0, :] - masspoint_pos[1, :])
    distance_penalty = reward_util.tolerance(
        distance,
        (0, 0.02),
        margin=0.1,
        sigmoid="linear",
    )

    return {
        "box_target": total_box_target,
        "box_orientation": total_box_orientation,
        "gripper_box": total_gripper_box,
        
        "joint_vel": joint_vel,
        "joint_vel_limit": 1 - joints_near_vel_limits,
        "total_command": total_command,
        "action_rate": action_rate,
        "collision_penalty": distance_penalty
    }

  def _orientation_error(self, object_quat, target_quat) -> jax.Array:
    quat_diff = math.quat_mul(object_quat, math.quat_inv(target_quat))
    quat_diff = math.normalize(quat_diff)
    ori_error = 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), a_max=1.0))
    return ori_error

  def _get_obs(self, state: mjx_env.State) -> jax.Array:
    obs = self._get_single_obs(state.data, state.info)
    obs_size = obs.shape[0]

    # fill the buffer
    obs_history = (
        jp.roll(state.info["obs_history"], obs_size).at[:obs_size].set(obs)
    )
    state.info["obs_history"] = obs_history

    # add observation delay
    state.info["rng"], key = jax.random.split(state.info["rng"])
    obs_idx = jax.random.randint(
        key,
        (1,),
        minval=self._config.noise_config.obs_min_delay,
        maxval=self._config.noise_config.obs_max_delay,
    )
    obs = obs_history.reshape((-1, obs_size))[obs_idx[0]]

    return obs

  def _get_single_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    target_1_pos = data.mocap_pos[self._mocap_target_1, :].ravel()
    target_1_quat = data.mocap_quat[self._mocap_target_1, :].ravel()
    target_1_mat = math.quat_to_mat(target_1_quat)
    target_1_orientation = target_1_mat.ravel()[3:]

    target_2_pos = data.mocap_pos[self._mocap_target_2, :].ravel()
    target_2_quat = data.mocap_quat[self._mocap_target_2, :].ravel()
    target_2_mat = math.quat_to_mat(target_2_quat)
    target_2_orientation = target_2_mat.ravel()[3:]

    # Add noise to object position and orientation.
    info["rng"], key1, key2, key3, key4, key5, key6 = jax.random.split(info["rng"], 7)
    angle = jax.random.uniform(
        key1,
        minval=0,
        maxval=self._config.noise_config.noise_scales.obj_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key2), angle)
    obj_1_quat = data.xquat[self._obj_1_body]
    obj_1_quat_w_noise = math.quat_mul(rand_quat, obj_1_quat)
    obj_1_pos = data.xpos[self._obj_1_body]
    obj_1_pos_w_noise = obj_1_pos + jax.random.uniform(
        key3, (3,), minval=-1, maxval=1
    ) * self._config.noise_config.noise_scales.obj_pos
    obj_1_orientation_w_noise = math.quat_to_mat(obj_1_quat_w_noise).ravel()[3:]
    
    angle = jax.random.uniform(
        key4,
        minval=0,
        maxval=self._config.noise_config.noise_scales.obj_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key5), angle)
    obj_2_quat = data.xquat[self._obj_2_body]
    obj_2_quat_w_noise = math.quat_mul(rand_quat, obj_2_quat)
    obj_2_pos = data.xpos[self._obj_2_body]
    obj_2_pos_w_noise = obj_2_pos + jax.random.uniform(
        key6, (3,), minval=-1, maxval=1
    ) * self._config.noise_config.noise_scales.obj_pos
    obj_2_orientation_w_noise = math.quat_to_mat(obj_2_quat_w_noise).ravel()[3:]

    # Add noise to robot proprio observation.
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    robot_qpos = data.qpos[
        self._masspoints_qids
    ]
    robot_qpos_w_noise = robot_qpos + jax.random.uniform(
        key1, minval=0, maxval=self._config.noise_config.noise_scales.robot_qpos
    )
    robot_qvel = data.qvel[
        self._massspoints_dqids
    ]
    robot_qvel_w_noise = robot_qvel + jax.random.uniform(
        key2, minval=0, maxval=self._config.noise_config.noise_scales.robot_qvel
    )

    obs = jp.concatenate([
        target_1_pos,
        target_1_orientation,
        target_2_pos,
        target_2_orientation,
        info["last_action"],
        # Robot joint angles and velocities.
        robot_qpos_w_noise,
        robot_qvel_w_noise,
        # Object position and orientation.
        obj_1_orientation_w_noise,
        obj_1_pos_w_noise,
        obj_2_orientation_w_noise,
        obj_2_pos_w_noise,
        info["box_1_success"].reshape(-1),
        info["box_2_success"].reshape(-1),
    ])
    return obs

  @property
  def action_size(self):
    return 6
