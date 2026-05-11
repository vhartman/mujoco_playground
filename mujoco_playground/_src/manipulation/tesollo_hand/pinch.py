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
"""Pinch task variants for the Tesollo hand: three observation configurations."""

from typing import Any, Dict, Optional, Union

import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src.manipulation.tesollo_hand.base_pinch import (
    CubePinchBase,
    _N_ACTIVE,
    default_config,
)
from mujoco_playground._src.manipulation.tesollo_hand.scene_builders.static_grasp import (
    build_static_grasp_scene,
)
from mujoco_playground._src import mjx_env

# Re-export so existing callers of `from pinch import default_config` still work.
__all__ = [
    "default_config",
    "CubePinchForce",
    "CubePinchProprio",
    "CubePinchBaseline",
]


class CubePinchForce(CubePinchBase):
    """Pinch variant: q + qdot + ctrl_targets + force(3) + object = 67-dim state."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(build_static_grasp_scene, config, config_overrides)

    @property
    def action_size(self) -> int:
        return _N_ACTIVE

    def _get_obs(self, data, info) -> mjx_env.Observation:
        state = jp.concatenate([
            self._obs_joint_angles(data, info),      # 23
            self._obs_joint_velocities(data, info),  # 23
            self._obs_motor_targets(info),            # 11
            self._obs_force(data),                    # 3
            self._obs_object(data, info),             # 7
        ])  # total = 67
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}


class CubePinchProprio(CubePinchBase):
    """Pinch variant: q + qdot + ctrl_targets_or_delta + object = 64-dim state."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(build_static_grasp_scene, config, config_overrides)

    @property
    def action_size(self) -> int:
        return _N_ACTIVE

    def _get_obs(self, data, info) -> mjx_env.Observation:
        ctrl_obs = (
            self._obs_ctrl_delta(data, info)
            if self._config.use_ctrl_delta
            else self._obs_motor_targets(info)
        )
        state = jp.concatenate([
            self._obs_joint_angles(data, info),      # 23
            self._obs_joint_velocities(data, info),  # 23
            ctrl_obs,                                 # 11
            self._obs_object(data, info),             # 7
        ])  # total = 64
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}


class CubePinchBaseline(CubePinchBase):
    """Pinch variant: q + qdot + object = 53-dim state (no ctrl info)."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(build_static_grasp_scene, config, config_overrides)

    @property
    def action_size(self) -> int:
        return _N_ACTIVE

    def _get_obs(self, data, info) -> mjx_env.Observation:
        state = jp.concatenate([
            self._obs_joint_angles(data, info),      # 23
            self._obs_joint_velocities(data, info),  # 23
            self._obs_object(data, info),             # 7
        ])  # total = 53
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}
