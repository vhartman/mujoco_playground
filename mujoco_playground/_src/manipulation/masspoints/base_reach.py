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
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class MasspointsReachEnv(mjx_env.MjxEnv):
  """Base class for masspoint reach environments."""

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

    # self._mj_spec.copy_during_attach = True

    cnt = 0
    self._num_rows = 3
    self._num_cols = 6
    for i in range(self._num_rows):
        for j in range(self._num_cols):
            # key_mjcf = mujoco.MjSpec.from_file(
            #     "/home/duplo/git/robohand/src/mujoco_playground/mujoco_playground/_src/manipulation/tesollo_hand/xmls/key.xml",
            # )

            pos_x = i * 0.03 + 0.17
            pos_y = -j * 0.03 + 0.06

            # key_mjcf.body('key_0').pos[:2] = np.array([pos_x, pos_y])
            # key_mjcf.body('key_0').name = key_mjcf.body('key_0').name[:-2] + f'_{i * self._num_cols + j}'
            # key_mjcf.joint('key_joint_0').name = key_mjcf.joint('key_joint_0').name[:-2] + f'_{i * self._num_cols + j}'

            # self._mj_spec.attach(key_mjcf, site=self._mj_spec.site("attachment_site"))

            self._goal_ids.append(cnt)
            self._goal_locations.append([pos_x, pos_y, -0.05])

            cnt += 1

    self._goal_ids = jp.array(self._goal_ids)

    self._goal_locations = jp.array(self._goal_locations)

    self._mj_model = self._mj_spec.compile()

    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # self._goal_locations = jp.array(
    #     [
    #         [0.2,  -0.03, -0.05],
    #         [0.2,   0.00, -0.05],
    #         [0.2,   0.03, -0.05],
    #         [0.2,  -0.06, -0.05],
    #         [0.23, -0.03, -0.05],
    #         [0.23,  0.00, -0.05],
    #         [0.23,  0.03, -0.05],
    #         [0.23, -0.06, -0.05],
    #     ]
    # )

    # self._goal_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

    # d = mujoco.MjData(self._mj_model)
    
    # with mujoco.viewer.launch_passive(self._mj_model, d) as viewer:
    #   # Close the viewer automatically after 30 wall-seconds.
    #   while viewer.is_running():
    #     # mj_step can be replaced with code that also evaluates
    #     # a policy and applies a control signal before stepping the physics.
    #     mujoco.mj_step(self._mj_model, d)

    #     # Example modification of a viewer option: toggle contact points every two seconds.
    #     with viewer.lock():
    #       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    #     # Pick up changes to the physics state, apply perturbations, update options from GUI.
    #     viewer.sync()

    #     # Rudimentary time keeping, will drift relative to wall clock.
    #     time.sleep(0.01)

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
