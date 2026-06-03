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
"""Observation component registry for Tesollo hand environments.

Infrastructure only — no component registrations live here.
Each environment class registers its own _obs_* methods after its class body.
"""

import dataclasses
import warnings
from typing import Callable


@dataclasses.dataclass(frozen=True)
class ObsComponent:
    key: str
    fn: Callable   # (env, data, info) -> jax.Array
    size: int
    description: str


_REGISTRY: dict[str, "ObsComponent"] = {}


def register(component: ObsComponent) -> ObsComponent:
    if component.key in _REGISTRY:
        raise ValueError(f"Obs key {component.key!r} already registered.")
    _REGISTRY[component.key] = component
    return component


def get(key: str) -> ObsComponent:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown obs key {key!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


SENSOR_BUNDLES: dict[str, tuple[str, ...]] = {
    "baseline":      ("joint_pos", "joint_vel"),
    "proprio":       ("joint_pos", "joint_vel", "motor_targets"),
    "force":         ("joint_pos", "joint_vel", "fingertip_forces"),
    "force_vec":     ("joint_pos", "joint_vel", "fingertip_forces", "fingertip_force_dirs"),
    "force_proprio": ("joint_pos", "joint_vel", "motor_targets", "fingertip_forces"),
}


def validate_spec(sensor_bundle: str, task_keys: tuple[str, ...], noise_scales) -> None:
    if sensor_bundle not in SENSOR_BUNDLES:
        raise ValueError(
            f"Unknown sensor_bundle {sensor_bundle!r}. Valid: {sorted(SENSOR_BUNDLES)}"
        )
    all_keys = SENSOR_BUNDLES[sensor_bundle] + task_keys
    for key in all_keys:
        get(key)
    stale_noise_keys = set(noise_scales.keys()) - set(all_keys)
    if stale_noise_keys:
        warnings.warn(
            f"obs_noise.scales has keys not in active obs spec: {stale_noise_keys}"
        )
