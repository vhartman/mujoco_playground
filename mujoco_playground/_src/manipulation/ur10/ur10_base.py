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
"""Franka Emika Panda base class."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

import jax
import jax.numpy as jp

from mujoco_playground._src import mjx_env

ARM_1_JOINTS = [
    "ur10_1:shoulder_pan_joint",
    "ur10_1:shoulder_lift_joint",
    "ur10_1:elbow_joint",
    "ur10_1:wrist_1_joint",
    "ur10_1:wrist_2_joint",
    "ur10_1:wrist_3_joint",
]

ARM_2_JOINTS = [
    "ur10_2:shoulder_pan_joint",
    "ur10_2:shoulder_lift_joint",
    "ur10_2:elbow_joint",
    "ur10_2:wrist_1_joint",
    "ur10_2:wrist_2_joint",
    "ur10_2:wrist_3_joint",
]

_ENV_DIR = mjx_env.ROOT_PATH / "manipulation/ur10"

def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, _ENV_DIR / "xmls/assets")
  mjx_env.update_assets(assets, _ENV_DIR / "xmls/mujoco_ur10")
  mjx_env.update_assets(assets, _ENV_DIR / "xmls/mujoco_ur10/assets")
 
  return assets

class DualUR10Base(mjx_env.MjxEnv):
  """Base environment for Franka Emika Panda and Robotiq gripper."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      xml_path: epath.Path,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_spec = mujoco.MjSpec.from_file(xml_path, assets=self._model_assets)

    self._mj_model = self._mj_spec.compile()

    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def _post_init(self):
    self._robots_qids = mjx_env.get_qpos_ids(self.mj_model, ARM_1_JOINTS + ARM_2_JOINTS)
    self._robots_dqids = mjx_env.get_qvel_ids(self.mj_model, ARM_1_JOINTS + ARM_2_JOINTS)
    
    self._mocap_target_1 = self.mj_model.body("mocap_target_1").mocapid
    self._mocap_target_2 = self.mj_model.body("mocap_target_2").mocapid

    self._floor_geom = self.mj_model.geom("floor").id

    keyframe = "home"

    self.workspace_center = jp.array([0., 0., 0.5])

    self._init_q = self.mj_model.keyframe(keyframe).qpos.copy()

    self._init_ctrl = self.mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

    self._joint_limit_percentage = 0.9
    self._joint_vel_limit_percentage = 0.9
    self._jnt_range = np.array(self.jnt_range())
    self._jnt_vel_range = np.array(self.jnt_vel_range())
    self._joint_range_init_percent_limit = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
  
  def get_ee_positions(self, data: mjx.Data) -> jax.Array:
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_ee_pos")
        for name in ["ur10_1", "ur10_2"]
    ])
  
  def get_ee_quats(self, data: mjx.Data) -> jax.Array:
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_ee_quat")
        for name in ["ur10_1", "ur10_2"]
    ])

  def jnt_range(self):
    # TODO(siholt): Use joint limits from XML.
    return [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
    ]

  def jnt_vel_range(self):
    return [
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.6100, 1.6100],
        [-1.6100, 1.6100],
        
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.1750, 1.1750],
        [-1.6100, 1.6100],
        [-1.6100, 1.6100],
    ]

  def ctrl_range(self):
    return [
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],

        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ]

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
