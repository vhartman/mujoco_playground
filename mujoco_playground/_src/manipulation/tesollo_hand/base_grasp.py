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
"""Base classes for tesollo hand."""

import abc
import functools
from typing import Any, Dict, Optional, Union

import numpy as np
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import tesollo_hand_grasp_constants as consts
from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module

import mujoco.viewer
import time


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "tesollo_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class TesolloHandGraspEnv(mjx_env.MjxEnv):
  """Base class for TESOLLO hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_spec = mujoco.MjSpec.from_file(xml_path, assets=self._model_assets)

    self._mj_model = self._mj_spec.compile()

    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def _build_obs(
      self,
      sensor_bundle: str,
      task_keys: tuple[str, ...],
      data,
      info: dict,
  ) -> jax.Array:
    all_keys = obs_module.SENSOR_BUNDLES[sensor_bundle] + task_keys
    rng_keys = jax.random.split(info["rng"], len(all_keys) + 1)
    info["rng"] = rng_keys[0]
    parts = []
    for i, key in enumerate(all_keys):
      component = obs_module.get(key)
      vec = component.fn(self, data, info)
      scale = self._config.obs_noise.level * getattr(
          self._config.obs_noise.scales, key, 0.0
      )
      if scale > 0.0:
        vec = vec + scale * jax.random.normal(rng_keys[i + 1], vec.shape)
      parts.append(vec)
    return jp.concatenate(parts)

  @functools.cached_property
  def obs_size(self) -> int:
    all_keys = (
        obs_module.SENSOR_BUNDLES[self._config.sensor_bundle]
        + self._task_obs_keys()
    )
    return sum(obs_module.get(k).size for k in all_keys)

  def _task_obs_keys(self) -> tuple[str, ...]:
    return ()

  # Sensor readings.
  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angacc")

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_upvector")

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_upvector")

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  def get_palm_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_orientation")

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])

  # ------------------------------------------------------------------
  # Obs component methods — registered below as ObsComponents so that
  # _build_obs can call them uniformly via (env, data, info) -> jax.Array.
  # ------------------------------------------------------------------

  def _obs_joint_pos(self, data, info) -> jax.Array:
    return data.qpos[self._hand_qids]

  def _obs_joint_vel(self, data, info) -> jax.Array:
    return data.qvel[self._hand_dqids]

  def _obs_motor_targets(self, data, info) -> jax.Array:
    return info["motor_targets"]

  def _obs_motor_deltas(self, data, info) -> jax.Array:
    return info["motor_targets"] - data.qpos[self._hand_qids]

  def _obs_fingertip_pos(self, data, info) -> jax.Array:
    return self.get_fingertip_positions(data)

  def _obs_palm_pos(self, data, info) -> jax.Array:
    return self.get_palm_position(data)

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


# Per-element label tuples shared by the generic hand-sensor components.
# joint_pos/vel/motor_targets/deltas all index the 26 DOFs in actuator order;
# ACTUATOR_NAMES encodes the axis (tx/ty/rz etc.) so are more readable than the
# raw XML joint names. fingertip_pos is 5 tips × (x, y, z).
_XYZ = ("x", "y", "z")
_FINGER_ANAT = ("thumb", "index", "middle", "ring", "pinky")
_JOINT_LABELS = tuple(consts.ACTUATOR_NAMES)
_FINGERTIP_XYZ_LABELS = tuple(f"{f}_{a}" for f in _FINGER_ANAT for a in _XYZ)
_PALM_LABELS = ("palm_x", "palm_y", "palm_z")

# Register generic hand-sensor obs components using the methods above.
# fn signature is (env, data, info) which matches an unbound method call.
obs_module.register(obs_module.ObsComponent("joint_pos",    TesolloHandGraspEnv._obs_joint_pos,    size=26, description="hand joint positions",            labels=_JOINT_LABELS))
obs_module.register(obs_module.ObsComponent("joint_vel",    TesolloHandGraspEnv._obs_joint_vel,    size=26, description="hand joint velocities",           labels=_JOINT_LABELS))
obs_module.register(obs_module.ObsComponent("motor_targets",TesolloHandGraspEnv._obs_motor_targets, size=26, description="current actuator targets",         labels=_JOINT_LABELS))
obs_module.register(obs_module.ObsComponent("motor_deltas", TesolloHandGraspEnv._obs_motor_deltas, size=26, description="motor targets minus current qpos",   labels=_JOINT_LABELS))
obs_module.register(obs_module.ObsComponent("fingertip_pos",TesolloHandGraspEnv._obs_fingertip_pos, size=15, description="fingertip positions (world frame)", labels=_FINGERTIP_XYZ_LABELS))
obs_module.register(obs_module.ObsComponent("palm_pos",     TesolloHandGraspEnv._obs_palm_pos,     size=3,  description="palm/grasp site position",          labels=_PALM_LABELS))


# ---------------------------------------------------------------------------
# Task-level base: grasp a cube held in the hand (reorientation-style).
# Subclasses implement _get_obs; everything else lives here.
# ---------------------------------------------------------------------------

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=300,
        success_threshold=0.1,
        joint_vel_threshold=0.5,
        vel_threshold=0.5,
        ang_vel_threshold=0.5,
        history_len=2,
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.0,
                cube_pos=0.0,
                cube_ori=0.0,
            ),
            random_ori_injection_prob=0.0,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                cube_ang_vel=-0.0,
                cube_lin_vel=-0.0,
                fingertip_pos=1.,
                palm_pos=-1.,
                orientation=0.5,
                cube_palm_position=0.5,
                cube_goal_position=0.5,
                height=5.0,
                termination=0.0,
                hand_pose=-0.00,
                wrist_pose=-0.00,
                action_rate=-0.000,
                joint_vel=-0.01,
                energy=-0.00,
                wrist_vel=-0.01,
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
        impl="jax",
        nconmax=200 * 8192,
        njmax=1024,
    )


class GraspBase(TesolloHandGraspEnv, abc.ABC):
    """Base for in-hand grasp/reorientation tasks.

    Subclasses implement _get_obs to define the observation space.
    All task logic (reset, step, reward, termination, perturbation) lives here.
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
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]
        self._cube_init = self._init_q[self._cube_qids]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation: ...

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, goal_rng = jax.random.split(rng)
        goal_quat = uniform_quat(goal_rng)

        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jp.array([0.0, 0.0, -0.2]) + 0.1 * jax.random.uniform(
            p_rng, (3,), minval=-0.01, maxval=0.01
        )
        q_cube = jp.array([*start_pos, *self._cube_init[3:]])
        v_cube = jp.zeros(6)

        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=q_hand,
            qvel=qvel,
            mocap_pos=self._init_mpos,
            mocap_quat=goal_quat,
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
            "action_delta": jp.zeros(self.mjx_model.nu),
            "qpos_error_history": jp.zeros(self._config.history_len * consts.NQ),
            "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
            "cube_ori_error_history": jp.zeros(self._config.history_len * 6),
            "goal_quat_dquat": jp.zeros(3),
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
        reward_, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward_, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        delta = action * self._config.action_scale
        state.info["action_delta"] = delta
        motor_targets = state.data.ctrl + delta
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        motor_targets = (
            self._config.ema_alpha * motor_targets
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        cube_pos = self.get_cube_position(data)
        cube_goal_error = jp.linalg.norm(cube_pos)

        ori_error = self._cube_orientation_error(data)
        cube_lin_vel = self._cube_lin_velocity(data)
        cube_ang_vel = self._cube_ang_velocity(data)

        cube_height = self.get_cube_position(data)[2]

        success = (
            (ori_error < self._config.success_threshold)
            & (cube_goal_error < 0.05)
            & (cube_lin_vel < self._config.vel_threshold)
            & (cube_ang_vel < self._config.ang_vel_threshold)
        )
        state.info["steps_since_last_success"] = jp.where(
            success, 0, state.info["steps_since_last_success"] + 1
        )
        state.info["success_count"] = jp.where(
            success, state.info["success_count"] + 1, state.info["success_count"]
        )
        state.metrics["steps_since_last_success"] = state.info["steps_since_last_success"]
        state.metrics["success_count"] = state.info["success_count"]

        done = self._get_termination(data, state.info)
        obs = self._get_obs(data, state.info)

        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward_ = sum(rewards.values()) * self.dt

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
        reward_ += success * self._config.reward_config.success_reward

        state.info["step"] += 1
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward_.dtype)
        return state.replace(data=data, obs=obs, reward=reward_, done=done)

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        return nans

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del done, metrics

        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pose_mse = jp.linalg.norm(palm_pos - cube_pos)
        cube_pos_reward = reward.tolerance(
            cube_pose_mse, (0, 0.02), margin=0.1, sigmoid="linear"
        )

        cube_goal_error = jp.linalg.norm(cube_pos)
        cube_goal_reward = reward.tolerance(
            cube_goal_error, (0, 0.02), margin=0.3, sigmoid="linear"
        )
        cube_height_reward = jp.clip(cube_pos[2] + 0.21, 0, 0.15)

        fingertip_distances = self.get_fingertip_positions(data).reshape(-1, 3) - cube_pos
        fingertip_reward = jp.sum(
            reward.tolerance(
                jp.linalg.norm(fingertip_distances, axis=1), (0, 0.035), margin=0.1, sigmoid="reciprocal"
            )
        )

        palm_distance = self.get_palm_position(data) - cube_pos
        palm_reward = jp.linalg.norm(palm_distance)

        terminated = self._get_termination(data, info)

        hand_pose_reward = jp.sum(
            jp.square(data.qpos[self._hand_qids] - self._default_pose)
        )

        wrist_pose_reward = jp.sum(
            jp.square(data.qpos[self._wrist_qids] - self._default_wrist_pose)
        )

        cost_wrist_vel = jp.sum(jp.square(data.qvel[self._wrist_qids]))

        cube_lin_vel = self._cube_lin_velocity(data)
        cube_ang_vel = self._cube_ang_velocity(data)

        return {
            "cube_lin_vel": cube_lin_vel,
            "cube_ang_vel": cube_ang_vel,
            "fingertip_pos": fingertip_reward,
            "palm_pos": palm_reward,
            "orientation": self._reward_cube_orientation(data),
            "cube_palm_position": cube_pos_reward,
            "cube_goal_position": cube_goal_reward,
            "height": cube_height_reward,
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
            ),
        }

    # ------------------------------------------------------------------
    # Observation helpers — call these from subclass _get_obs
    # ------------------------------------------------------------------

    def _obs_noisy_joint_angles(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        joint_angles = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        return (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.obs_noise.level
            * self._config.obs_noise.scales.joint_pos
        )

    def _obs_qpos_error_history(
        self, noisy_joint_angles: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        qpos_error_history = (
            jp.roll(info["qpos_error_history"], consts.NQ)
            .at[:consts.NQ]
            .set(noisy_joint_angles - info["motor_targets"])
        )
        info["qpos_error_history"] = qpos_error_history
        return qpos_error_history

    def _obs_noisy_cube_pose(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Returns a noisy 7-vector (xyz, wxyz) with optional random-ori injection."""
        cube_pos = self.get_cube_position(data)
        cube_quat = self.get_cube_orientation(data)
        info["rng"], pos_rng, ori_rng = jax.random.split(info["rng"], 3)
        noisy_cube_quat = math.normalize(
            cube_quat
            + jax.random.normal(ori_rng, shape=(4,))
            * self._config.obs_noise.level
            * self._config.obs_noise.scales.cube_ori
        )
        noisy_cube_pos = (
            cube_pos
            + (2 * jax.random.uniform(pos_rng, shape=cube_pos.shape) - 1)
            * self._config.obs_noise.level
            * self._config.obs_noise.scales.cube_pos
        )
        noisy_pose = jp.concatenate([noisy_cube_pos, noisy_cube_quat])

        info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
        rand_quat = uniform_quat(key1)
        rand_pos = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
        rand_pose = jp.concatenate([rand_pos, rand_quat])
        m = self._config.obs_noise.level * jax.random.bernoulli(
            key3, self._config.obs_noise.random_ori_injection_prob
        )
        return noisy_pose * (1 - m) + rand_pose * m

    def _obs_cube_pos_error_history(
        self,
        noisy_pose: jax.Array,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> jax.Array:
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - noisy_pose[:3]
        cube_pos_error_history = (
            jp.roll(info["cube_pos_error_history"], 3).at[:3].set(cube_pos_error)
        )
        info["cube_pos_error_history"] = cube_pos_error_history
        return cube_pos_error_history

    def _obs_cube_ori_error_history(
        self,
        noisy_pose_quat: jax.Array,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> jax.Array:
        goal_quat = self.get_cube_goal_orientation(data)
        quat_diff = math.quat_mul(noisy_pose_quat, math.quat_inv(goal_quat))
        xmat_diff = math.quat_to_mat(quat_diff).ravel()[3:]
        cube_ori_error_history = (
            jp.roll(info["cube_ori_error_history"], 6).at[:6].set(xmat_diff)
        )
        info["cube_ori_error_history"] = cube_ori_error_history
        return cube_ori_error_history

    def _obs_last_act(self, info: dict[str, Any]) -> jax.Array:
        return info["last_act"]

    def _obs_privileged_grasp(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Uncorrupted quantities for the critic/privileged observer."""
        palm_pos = self.get_palm_position(data)
        cube_pos_error_uncorrupted = palm_pos - self.get_cube_position(data)
        cube_quat_uncorrupted = self.get_cube_orientation(data)
        goal_quat = self.get_cube_goal_orientation(data)
        quat_diff_uncorrupted = math.quat_mul(
            cube_quat_uncorrupted, math.quat_inv(goal_quat)
        )
        xmat_diff_uncorrupted = math.quat_to_mat(quat_diff_uncorrupted).ravel()[3:]
        return jp.concatenate([
            data.qpos[self._hand_qids],
            data.qvel[self._hand_dqids],
            self.get_fingertip_positions(data),
            cube_pos_error_uncorrupted,
            xmat_diff_uncorrupted,
            self.get_cube_linvel(data),
            self.get_cube_angvel(data),
            info["pert_dir"],
            data.xfrc_applied[self._cube_body_id],
        ])

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cube_lin_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_linvel(data))

    def _cube_ang_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_angvel(data))

    def _cube_orientation_error(self, data: mjx.Data) -> jax.Array:
        cube_ori = self.get_cube_orientation(data)
        cube_goal_ori = self.get_cube_goal_orientation(data)
        quat_diff = math.quat_mul(cube_ori, math.quat_inv(cube_goal_ori))
        quat_diff = math.normalize(quat_diff)
        return 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), 0.0, 1.0))

    def _reward_cube_orientation(self, data: mjx.Data) -> jax.Array:
        ori_error = self._cube_orientation_error(data)
        return reward.tolerance(ori_error, (0, 0.2), margin=jp.pi, sigmoid="linear")

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

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

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
        start_pert &= step != 0
        last_pert_step = jp.where(start_pert, step, last_pert_step)
        duration = jp.clip(step - last_pert_step, 0, 100_000)
        in_pert_interval = duration < state.info["pert_duration_steps"]

        pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
        xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

        state.info["pert_dir"] = pert_dir
        state.info["last_pert_step"] = last_pert_step
        data = state.data.replace(xfrc_applied=xfrc)
        return state.replace(data=data)


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
