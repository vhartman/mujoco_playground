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

from typing import Any, Dict, Optional, Union

import numpy as np
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.masspoints import masspoint_reach_constants as consts

import mujoco.viewer
import time

def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "tesollo_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "assets"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class MasspointsPushEnv(mjx_env.MjxEnv):
  """Base class for masspoint push environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_spec = mujoco.MjSpec.from_file(xml_path, assets=self._model_assets)

    self._goal_locations = []
    self._goal_ids = []

    self._mj_model = self._mj_spec.compile()

    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def _post_init(self):
    self._masspoints_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._massspoints_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._jnt_range = np.array(self.jnt_range())
    self._jnt_vel_range = np.array(self.jnt_vel_range())

    self._joint_range_init_percent_limit = np.array(
        [1., 1., 1., 1., 1., 1.0]
    )
    self._joint_vel_limit_percentage = 0.9

    obj_name="box"
    keyframe="home"
    
    self._obj_body = self.mj_model.body(obj_name).id
    self._obj_geom = self.mj_model.geom(obj_name).id
    self._obj_qposadr = self.mj_model.jnt_qposadr[
        self.mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self.mj_model.body("mocap_target").mocapid
    self._floor_geom = self.mj_model.geom("floor").id
    self._wall_geom = self.mj_model.geom("wall").id
    self._init_q = self.mj_model.keyframe(keyframe).qpos.copy()
    self._init_obj_pos = np.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=np.float32,
    )
    self._init_obj_quat = np.array(
        self._init_q[self._obj_qposadr + 3 : self._obj_qposadr + 7],
        dtype=np.float32,
    )
    self._init_ctrl = self.mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
  
  def jnt_vel_range(self):
    return [
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ]

  def jnt_range(self):
    return [
        [0.2, 0.25],
        [-0.1, 0.1],
        [-0.001, 0.001],
        [0.2, 0.25],
        [-0.1, 0.1],
        [-0.001, 0.001],
    ]
  # Sensor readings.
  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
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


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
