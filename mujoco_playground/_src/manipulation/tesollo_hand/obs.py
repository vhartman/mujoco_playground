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
"""Observation DSL and component descriptor for Tesollo hand environments.

obs.py owns two things:

1. ObsComponent — a plain dataclass describing one named obs slice (key, bound
   callable, concrete size, description, optional per-element labels).  Each env
   builds its own ``_obs_components: dict[str, ObsComponent]`` in _post_init();
   there is no global registry.

2. resolve_bundle / sensor-bundle DSL — a pure string → tuple[str, ...] parser
   that expands a '+'-joined bundle spec into an ordered list of obs key names.
   The baseline group (joint_pos, joint_vel) is always first.
"""

import dataclasses
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


# ---------------------------------------------------------------------------
# Compositional sensor-bundle DSL
#
# A bundle is a '+'-joined set of sensor groups. The "baseline" group
# (joint_pos, joint_vel) is always included. Each other group selects exactly
# one representation type:
#
#   baseline                joint_pos, joint_vel                 (always on)
#   proprio.delta           motor_deltas
#   proprio.target          motor_targets
#   force.magnitude         fingertip_forces
#   force.full              fingertip_forces, fingertip_force_dirs
#
# A group may appear at most once. Example: "proprio.target+force.magnitude".
# ---------------------------------------------------------------------------

_BASELINE_KEYS: tuple[str, ...] = ("joint_pos", "joint_vel")

# group -> {type -> obs component keys the (group, type) pair expands to}.
_SENSOR_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "proprio": {
        "delta":  ("motor_deltas",),
        "target": ("motor_targets",),
    },
    "force": {
        "magnitude": ("fingertip_forces",),
        "full":      ("fingertip_forces", "fingertip_force_dirs"),
    },
}

# Order in which groups contribute to the observation vector (after baseline).
_GROUP_ORDER: tuple[str, ...] = ("proprio", "force")


def resolve_bundle(sensor_bundle: str) -> tuple[str, ...]:
    """Expand a '+'-composed bundle spec into an ordered tuple of obs keys.

    "baseline" is always included first. Each remaining token is
    "<group>.<type>" (e.g. "force.full"); a group may be selected at most once.
    Raises ValueError on a malformed token, an unknown group/type, or a group
    given more than once.

    Special cases (whole-spec tokens, no '+' composition):
      "none"     -> EMPTY proprioceptive state; policy sees only the task keys
                    (e.g. target_force). Sanity check for solving with no proprio.
      "pos_only" -> joint_pos only, NO joint_vel; isolates q from q_dot.
      "vel_only" -> joint_vel only, NO joint_pos; isolates q_dot from q.
    """
    if sensor_bundle.strip() == "none":
        return ()
    if sensor_bundle.strip() == "pos_only":
        return ("joint_pos",)
    if sensor_bundle.strip() == "vel_only":
        return ("joint_vel",)

    selected: dict[str, str] = {}
    for tok in (t.strip() for t in sensor_bundle.split("+")):
        if not tok or tok == "baseline":
            continue
        group, sep, type_ = tok.partition(".")
        if not sep:
            raise ValueError(
                f"Malformed sensor-bundle token {tok!r} in {sensor_bundle!r}: "
                "expected 'baseline' or '<group>.<type>' (e.g. 'force.full')."
            )
        if group not in _SENSOR_GROUPS:
            raise ValueError(
                f"Unknown sensor group {group!r} in {sensor_bundle!r}. "
                f"Valid groups: {sorted(_SENSOR_GROUPS)} (plus 'baseline')."
            )
        if type_ not in _SENSOR_GROUPS[group]:
            raise ValueError(
                f"Unknown type {type_!r} for group {group!r} in {sensor_bundle!r}. "
                f"Valid types: {sorted(_SENSOR_GROUPS[group])}."
            )
        if group in selected:
            raise ValueError(
                f"Group {group!r} selected more than once in {sensor_bundle!r}; "
                "choose a single type per group."
            )
        selected[group] = type_

    keys: list[str] = list(_BASELINE_KEYS)
    for group in _GROUP_ORDER:
        if group in selected:
            keys.extend(_SENSOR_GROUPS[group][selected[group]])
    return tuple(keys)


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
