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

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module
from mujoco_playground._src.manipulation.tesollo_hand import tesollo_hand_grasp_constants as consts


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

  # ------------------------------------------------------------------
  # Observation spec
  # ------------------------------------------------------------------

  def _build_obs_components(self) -> dict[str, obs_module.ObsComponent]:
    """The obs components every hand env has, sized for this instance.

    Subclasses call super() and add their task-specific entries.
    """
    n = len(self._hand_qids)
    ntips = 3 * len(self._fingertip_names)
    components = {
        "joint_pos": obs_module.ObsComponent(
            "joint_pos", self._obs_joint_pos, size=n,
            description="hand joint positions"),
        "joint_vel": obs_module.ObsComponent(
            "joint_vel", self._obs_joint_vel, size=n,
            description="hand joint velocities"),
        "motor_targets": obs_module.ObsComponent(
            "motor_targets", self._obs_motor_targets, size=n,
            description="current actuator targets"),
        "motor_deltas": obs_module.ObsComponent(
            "motor_deltas", self._obs_motor_deltas, size=n,
            description="motor targets minus current qpos"),
        "fingertip_pos": obs_module.ObsComponent(
            "fingertip_pos", self._obs_fingertip_pos, size=ntips,
            description="fingertip positions (world frame)"),
        "palm_pos": obs_module.ObsComponent(
            "palm_pos", self._obs_palm_pos, size=3,
            description="palm/grasp site position", labels=_PALM_LABELS),
    }
    if self._TIP_FORCE_SENSORS:
      nforce = len(self._TIP_FORCE_SENSORS)
      components.update({
          "fingertip_forces": obs_module.ObsComponent(
              "fingertip_forces", self._obs_fingertip_forces, size=nforce,
              description="per-tip contact force magnitude vs cube, raw Newtons"),
          "fingertip_force_dirs": obs_module.ObsComponent(
              "fingertip_force_dirs", self._obs_fingertip_force_dirs, size=nforce * 3,
              description="per-tip normalized net force direction vs cube"),
      })
    return components

  def _task_obs_keys(self) -> tuple[str, ...]:
    """Task-specific obs keys, appended after the bundle's keys."""
    return ()

  @functools.cached_property
  def obs_size(self) -> int:
    all_keys = (
        obs_module.resolve_bundle(self._config.sensor_bundle)
        + self._task_obs_keys()
    )
    return sum(self._obs_components[k].size for k in all_keys)

  def _build_obs(
      self,
      sensor_bundle: str,
      task_keys: tuple[str, ...],
      data,
      info: dict,
  ) -> jax.Array:
    all_keys = obs_module.resolve_bundle(sensor_bundle) + task_keys
    rng_keys = jax.random.split(info["rng"], len(all_keys) + 1)
    info["rng"] = rng_keys[0]
    # Per-episode constant bias, sampled once in reset (see _sample_obs_bias).
    # Absent for envs that never populate it -> no bias added.
    obs_bias = info.get("obs_bias", {})
    parts = []
    for i, key in enumerate(all_keys):
      comp = self._obs_components[key]
      vec = comp.fn(data, info)
      scale = self._config.obs_noise.level * getattr(
          self._config.obs_noise.scales, key, 0.0
      )
      if scale > 0.0:
        vec = vec + scale * jax.random.normal(rng_keys[i + 1], vec.shape)
      bias = obs_bias.get(key)
      if bias is not None:
        vec = vec + bias
      parts.append(vec)
    return jp.concatenate(parts)

  def _sample_obs_bias(
      self, rng, sensor_bundle: str, task_keys: tuple[str, ...]
  ) -> dict:
    """Per-episode constant observation bias, one offset per observed channel.

    Unlike the per-step white noise in obs_noise.scales, a constant offset is not
    zero-mean over the episode, so it does not average out: it shifts the
    policy's state estimate by a fixed, unobservable amount for the whole
    episode. Returns {} when the env config has no bias_scales block.
    """
    cfg = self._config.obs_noise
    bias_scales = getattr(cfg, "bias_scales", None)
    if bias_scales is None:
      return {}
    all_keys = obs_module.resolve_bundle(sensor_bundle) + task_keys
    rng_keys = jax.random.split(rng, len(all_keys))
    bias = {}
    for rk, key in zip(rng_keys, all_keys):
      scale = cfg.level * getattr(bias_scales, key, 0.0)
      size = self._obs_components[key].size
      bias[key] = scale * jax.random.normal(rk, (size,))
    return bias

  # Obs component implementations: (data, info) -> jax.Array.

  _TIP_FORCE_SENSORS: list[str] = []

  @property
  def _fingertip_names(self) -> tuple[str, ...]:
    """Active fingertip site names. Override to use a reduced set."""
    return consts.FINGERTIP_NAMES

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

  def _obs_fingertip_forces(self, data, info) -> jax.Array:
    forces = jp.array([
        jp.sum(jp.linalg.norm(
            mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
            axis=1,
        ))
        for name in self._TIP_FORCE_SENSORS
    ])
    return forces

  def _obs_fingertip_force_dirs(self, data, info) -> jax.Array:
    dirs = []
    for name in self._TIP_FORCE_SENSORS:
      net = jp.sum(
          mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
          axis=0,
      )
      magnitude = jp.linalg.norm(net)
      dirs.append(jp.where(magnitude > 1e-3, net / magnitude, jp.zeros(3)))
    return jp.concatenate(dirs)

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
        for name in self._fingertip_names
    ])

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


# Per-element label tuples used in _build_obs_components().
_PALM_LABELS = ("palm_x", "palm_y", "palm_z")


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
