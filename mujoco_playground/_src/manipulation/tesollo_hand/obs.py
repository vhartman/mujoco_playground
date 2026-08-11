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
"""Observation components and the sensor-bundle spec for Tesollo hand envs.

Two things live here:

1. ObsComponent — one named slice of the observation vector (key, bound
   callable, concrete size, description, optional per-element labels). Each env
   builds its own ``_obs_components`` dict in _post_init(); there is no global
   registry, so two envs with different configs can coexist in one process.

2. The sensor-bundle spec — a pure string -> tuple[str, ...] parser that turns a
   config value such as "proprio.target+force.magnitude" into an ordered list of
   obs keys. Because it is a config string, sensor selection is sweepable by the
   queue runner rather than being a code edit.
"""

import dataclasses
import enum
import warnings
from typing import Callable


@dataclasses.dataclass(frozen=True)
class ObsComponent:
    key: str
    fn: Callable                          # bound method: (data, info) -> jax.Array
    size: int                             # concrete element count, known at build time
    description: str
    labels: tuple[str, ...] | None = None # per-element names; len must equal size

    def __post_init__(self):
        if self.labels is not None and len(self.labels) != self.size:
            raise ValueError(
                f"Obs key {self.key!r}: got {len(self.labels)} labels "
                f"but size is {self.size}."
            )

    def element_labels(self) -> tuple[str, ...]:
        """Per-element label strings, falling back to '<key>[i]' when unset."""
        if self.labels:
            return self.labels
        return tuple(f"{self.key}[{i}]" for i in range(self.size))


class SensorGroup(enum.Enum):
    """A choice the bundle makes exactly once. Declaration order is the order
    the groups contribute to the observation vector."""

    BASELINE = "baseline"
    PROPRIO = "proprio"
    FORCE = "force"


class SensorOption(enum.Enum):
    """One (group, representation) pair, with the obs keys it expands to.

    `spec` is how the option is written in a config string.
    """

    BASELINE_FULL = ("baseline", SensorGroup.BASELINE, ("joint_pos", "joint_vel"))
    BASELINE_NONE = ("none", SensorGroup.BASELINE, ())
    PROPRIO_DELTA = ("proprio.delta", SensorGroup.PROPRIO, ("motor_deltas",))
    PROPRIO_TARGET = ("proprio.target", SensorGroup.PROPRIO, ("motor_targets",))
    FORCE_MAGNITUDE = ("force.magnitude", SensorGroup.FORCE, ("fingertip_forces",))
    FORCE_FULL = (
        "force.full",
        SensorGroup.FORCE,
        ("fingertip_forces", "fingertip_force_dirs"),
    )

    def __init__(self, spec: str, group: SensorGroup, keys: tuple[str, ...]):
        self.spec = spec
        self.group = group
        self.keys = keys

    @classmethod
    def from_spec(cls, spec: str) -> "SensorOption":
        for option in cls:
            if option.spec == spec:
                return option
        raise ValueError(
            f"Unknown sensor-bundle token {spec!r}. "
            f"Valid tokens: {sorted(o.spec for o in cls)}."
        )


# Selected when a bundle names no option for a group.
_DEFAULTS: dict[SensorGroup, SensorOption] = {
    SensorGroup.BASELINE: SensorOption.BASELINE_FULL,
}


def resolve_bundle(sensor_bundle: str) -> tuple[str, ...]:
    """Expand a '+'-composed bundle spec into an ordered tuple of obs keys.

    Every token is a SensorOption spec, and each group may be chosen at most
    once: "proprio.target+force.magnitude" is fine, "proprio.delta+proprio.target"
    is not. Groups with no token fall back to _DEFAULTS, so joint_pos/joint_vel
    are present unless "none" replaces them.
    """
    selected = dict(_DEFAULTS)
    chosen: set[SensorGroup] = set()

    for token in (t.strip() for t in sensor_bundle.split("+")):
        if not token:
            continue
        option = SensorOption.from_spec(token)
        if option.group in chosen:
            raise ValueError(
                f"Group {option.group.value!r} selected more than once in "
                f"{sensor_bundle!r}; choose a single option per group."
            )
        chosen.add(option.group)
        selected[option.group] = option

    return tuple(
        key
        for group in SensorGroup
        if group in selected
        for key in selected[group].keys
    )


def validate_spec(
    sensor_bundle: str,
    task_keys: tuple[str, ...],
    obs_components: dict,
    noise_scales,
) -> None:
    """Check that all keys in bundle+task are present in obs_components."""
    all_keys = resolve_bundle(sensor_bundle) + task_keys
    missing = [k for k in all_keys if k not in obs_components]
    if missing:
        raise KeyError(
            f"Obs keys {missing} not found in env obs_components. "
            f"Available: {sorted(obs_components)}"
        )
    stale_noise_keys = set(noise_scales.keys()) - set(all_keys)
    if stale_noise_keys:
        warnings.warn(
            f"obs_noise.scales has keys not in active obs spec: {stale_noise_keys}"
        )
