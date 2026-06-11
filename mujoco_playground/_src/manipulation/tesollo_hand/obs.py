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
    labels: tuple[str, ...] | None = None  # per-element names; len must == size


_REGISTRY: dict[str, "ObsComponent"] = {}


def register(component: ObsComponent) -> ObsComponent:
    if component.key in _REGISTRY:
        raise ValueError(f"Obs key {component.key!r} already registered.")
    if component.labels is not None and len(component.labels) != component.size:
        raise ValueError(
            f"Obs key {component.key!r}: got {len(component.labels)} labels "
            f"but size is {component.size}."
        )
    _REGISTRY[component.key] = component
    return component


def get(key: str) -> ObsComponent:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown obs key {key!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def element_labels(key: str) -> tuple[str, ...]:
    """Per-element labels for a component; defaults to '<key>[i]' when unset."""
    c = get(key)
    return c.labels or tuple(f"{key}[{i}]" for i in range(c.size))


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
# A group may appear at most once. Example: "proprio.target+force.full".
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
    """
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


def validate_spec(sensor_bundle: str, task_keys: tuple[str, ...], noise_scales) -> None:
    all_keys = resolve_bundle(sensor_bundle) + task_keys
    for key in all_keys:
        get(key)
    stale_noise_keys = set(noise_scales.keys()) - set(all_keys)
    if stale_noise_keys:
        warnings.warn(
            f"obs_noise.scales has keys not in active obs spec: {stale_noise_keys}"
        )
