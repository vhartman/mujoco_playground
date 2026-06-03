# Observation System Refactor — Spec & Implementation Plan

## Motivation

Every task env currently owns its `_get_obs` entirely: obs component selection,
noise application, and privileged state construction are all hand-written per
class, with significant duplication across `pick_and_place.py`, `grasp.py`,
`reorient.py`, and `reach.py`. Adding a new input variant (e.g. adding
fingertip forces to an existing task) requires a new subclass. Noise scales live
in the config but are applied ad-hoc with no central contract.

The goal is a system where:
- The **obs registry** defines what components are available and how to extract them.
- The **sensor bundle** config knob selects which hand-sensor components go to the actor network.
- Each **task base** declares its task-specific context components (fixed per task).
- **Noise** is applied uniformly via the existing `obs_noise.scales` config, keyed by component name.
- **Privileged state** is fixed per task base — it always contains the full ground-truth view.
- New obs variants require **no new classes**, only a different `sensor_bundle` string in the config.

---

## Non-goals

- Migrating `GraspBase`, `reach`, or `reorient` task envs in this PR. The
  mechanism is implemented to be ready for them, but only `PickAndPlaceBase` is
  migrated now.
- Making privileged state configurable. It is fixed per task base.
- Changing the `obs_noise` config schema — `level` × `scales.<key>` stays as-is.

---

## New file: `tesollo_hand/obs.py`

Single source of truth for all obs infrastructure. Nothing in this file imports
from any task env.

### `ObsComponent`

```python
@dataclasses.dataclass(frozen=True)
class ObsComponent:
    key: str        # unique name used in config and registry
    fn: Callable    # (env: TesolloHandGraspEnv, data: mjx.Data, info: dict) -> jax.Array
    size: int       # output length — statically known, no forward pass needed
    description: str
```

`fn` is not stored in the config (not serializable); it lives only in the
registry. The config stores only the string `key`.

### Registry

Module-level ordered dict. Provides collision detection and a stable iteration
order (= stable obs vector layout).

```python
_REGISTRY: dict[str, ObsComponent] = {}

def register(component: ObsComponent) -> ObsComponent:
    if component.key in _REGISTRY:
        raise ValueError(f"Obs key {component.key!r} already registered.")
    _REGISTRY[component.key] = component
    return component

def get(key: str) -> ObsComponent:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown obs key {key!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
```

### Hand-sensor components (registered at module import)

These are generic across all tesollo-hand tasks:

| key | size | description |
|---|---|---|
| `joint_pos` | 26 | hand joint positions |
| `joint_vel` | 26 | hand joint velocities |
| `motor_targets` | 26 | current actuator targets |
| `motor_deltas` | 26 | targets − current qpos |
| `fingertip_pos` | 15 | fingertip positions (world frame) |
| `palm_pos` | 3 | palm/grasp site position |
| `fingertip_forces` | 5 | per-tip contact force magnitude vs cube, /10N |
| `fingertip_force_dirs` | 15 | per-tip normalized net force direction vs cube |
| `total_contact_force` | 1 | sum of all hand-cube contact force magnitudes, /10N |

Each is registered with a named module-level function (no lambdas):

```python
def _joint_pos(env, data, info):
    return data.qpos[env._hand_qids]

register(ObsComponent("joint_pos", _joint_pos, size=26, description="..."))
```

### Sensor bundles

Named presets for the hand-sensor portion of the obs. These are cross-task.

```python
SENSOR_BUNDLES: dict[str, tuple[str, ...]] = {
    "baseline":   ("joint_pos", "joint_vel"),
    "proprio":    ("joint_pos", "joint_vel", "motor_targets"),
    "force":      ("joint_pos", "joint_vel", "fingertip_forces"),
    "force_vec":  ("joint_pos", "joint_vel", "fingertip_forces", "fingertip_force_dirs"),
    "force_proprio": ("joint_pos", "joint_vel", "motor_targets", "fingertip_forces"),
}
```

All keys in every bundle must be registered hand-sensor components. Validated at
module import time (not at runtime).

### Validation helper

Called from `TesolloHandGraspEnv.__init__` for any env using the new system:

```python
def validate_spec(sensor_bundle: str, task_keys: tuple[str, ...], noise_scales) -> None:
    if sensor_bundle not in SENSOR_BUNDLES:
        raise ValueError(f"Unknown sensor_bundle {sensor_bundle!r}. Valid: {sorted(SENSOR_BUNDLES)}")
    all_keys = SENSOR_BUNDLES[sensor_bundle] + task_keys
    for key in all_keys:
        get(key)  # raises KeyError with a clear message if missing
    stale_noise_keys = set(noise_scales.keys()) - set(all_keys)
    if stale_noise_keys:
        import warnings
        warnings.warn(f"obs_noise.scales has keys not in active obs spec: {stale_noise_keys}")
```

---

## Changes to `TesolloHandGraspEnv` (`base_grasp.py`)

Add one method that any task base (that opts in) calls to build its obs:

```python
def _build_obs(
    self,
    sensor_bundle: str,
    task_keys: tuple[str, ...],
    data: mjx.Data,
    info: dict,
) -> jax.Array:
    all_keys = obs.SENSOR_BUNDLES[sensor_bundle] + task_keys
    # Pre-split one key per component for RNG stability:
    # adding/removing a zero-noise component does not shift other components' keys.
    rng_keys = jax.random.split(info["rng"], len(all_keys) + 1)
    info["rng"] = rng_keys[0]

    parts = []
    for i, key in enumerate(all_keys):
        component = obs.get(key)
        vec = component.fn(self, data, info)
        scale = self._config.obs_noise.level * getattr(
            self._config.obs_noise.scales, key, 0.0
        )
        if scale > 0.0:
            vec = vec + scale * jax.random.normal(rng_keys[i + 1], vec.shape)
        parts.append(vec)
    return jp.concatenate(parts)
```

Also add a property so the obs dimension is always statically known:

```python
@functools.cached_property
def obs_size(self) -> int:
    all_keys = obs.SENSOR_BUNDLES[self._config.sensor_bundle] + self._task_obs_keys()
    return sum(obs.get(k).size for k in all_keys)
```

`TesolloHandGraspEnv` itself does **not** call `_build_obs` — that is the task
base's responsibility. This keeps `TesolloHandGraspEnv` non-opinionated about
obs structure.

---

## Changes to `PickAndPlaceBase` (`base_pick_and_place.py`)

### Register task-specific components at module level

```python
# registered once when the module is imported
obs.register(ObsComponent("goal_pos",            _goal_pos,            size=3,  description="..."))
obs.register(ObsComponent("goal_quat",           _goal_quat,           size=4,  description="..."))
obs.register(ObsComponent("last_ground_cube_pos",_last_ground_cube_pos,size=3,  description="..."))
obs.register(ObsComponent("cube_pos",            _cube_pos,            size=3,  description="..."))
obs.register(ObsComponent("cube_to_goal",        _cube_to_goal,        size=3,  description="..."))
```

### Declare task context

```python
class PickAndPlaceBase(TesolloHandGraspEnv, abc.ABC):
    _TASK_KEYS: tuple[str, ...] = (
        "last_ground_cube_pos", "goal_pos", "goal_quat",
    )

    def _task_obs_keys(self) -> tuple[str, ...]:
        return self._TASK_KEYS
```

### `_get_obs` — no longer abstract

```python
def _get_obs(self, data, info):
    state = self._build_obs(
        self._config.sensor_bundle, self._task_obs_keys(), data, info
    )
    return {
        "state": state,
        "privileged_state": self._obs_privileged(data, info),
    }
```

`_obs_privileged` remains hardcoded on `PickAndPlaceBase`. It is not
config-driven.

### Validation in `_post_init`

```python
def _post_init(self):
    obs.validate_spec(
        self._config.sensor_bundle,
        self._task_obs_keys(),
        self._config.obs_noise.scales,
    )
    ...
```

### Config changes

Replace:
```python
obs_noise=config_dict.create(
    level=1.0,
    scales=config_dict.create(joint_pos=0.0, joint_vel=0.0, cube_pos=0.0),
),
```

With:
```python
sensor_bundle="proprio",
obs_noise=config_dict.create(
    level=1.0,
    scales=config_dict.create(
        joint_pos=0.0,
        joint_vel=0.0,
        motor_targets=0.0,
        last_ground_cube_pos=0.0,
        goal_pos=0.0,
        goal_quat=0.0,
        fingertip_forces=0.0,
        fingertip_force_dirs=0.0,
    ),
),
```

All components that can ever appear in any bundle are listed with default 0.
A user tunes noise by overriding specific scale values — no structural change
to the config needed.

---

## Changes to `pick_and_place.py`

Remove all subclasses. Replace with one class and named config factories:

```python
class PickAndPlace(PickAndPlaceBase):
    """Single concrete pick-and-place env. Obs layout set by config.sensor_bundle."""
    pass


# --- Config factories (these are the "variants") ---

def baseline_config(**overrides) -> ConfigDict:
    cfg = default_config()
    cfg.sensor_bundle = "baseline"
    cfg.update(overrides)
    return cfg

def proprio_config(**overrides) -> ConfigDict:
    cfg = default_config()
    cfg.sensor_bundle = "proprio"
    cfg.update(overrides)
    return cfg

def force_config(**overrides) -> ConfigDict:
    cfg = default_config()
    cfg.sensor_bundle = "force"
    cfg.update(overrides)
    return cfg

def force_proprio_config(**overrides) -> ConfigDict:
    cfg = default_config()
    cfg.sensor_bundle = "force_proprio"
    cfg.update(overrides)
    return cfg
```

Usage:
```python
env = PickAndPlace(config=proprio_config())
# tune noise for a sweep:
env = PickAndPlace(config=proprio_config(**{"obs_noise.scales.joint_pos": 0.05}))
```

---

## Training pipeline integration

### Registry and env naming

The old class-per-variant design exposed each variant as a separately registered
env name (`TesolloPickAndPlaceProprio`, `TesolloPickAndPlaceForce`, etc.).
The new design registers **one** env: `TesolloPickAndPlace`.

- The old variant names are **removed without aliases**. Any queue file
  referencing them must be updated (see below).
- `manipulation_params.brax_ppo_config` simplifies from
  `env_name.startswith("TesolloPickAndPlace")` to `env_name == "TesolloPickAndPlace"`.
  No per-variant PPO parameter differences are expected.

### Queue YAML

`sensor_bundle` is a plain config key, overridable via `env_overrides` exactly
like any other. Sweeping over obs variants uses the existing sweep mechanism:

```yaml
defaults:
  flags:
    use_wandb: true
    num_timesteps: 500000000

sweep:
  env_names: [TesolloPickAndPlace]
  params:
    sensor_bundle:
      values: [baseline, proprio, force, force_proprio]
    obs_noise.level:
      range: [0.5, 2.0]
      steps: 3
```

Use the `suffix` flag to label runs when the auto-generated name needs
disambiguation beyond what the sweep label provides.

### Experiment naming

`train_jax_ppo.py` is **not modified**. Experiment names remain
`{env_name}-{timestamp}-{suffix}`. The `sensor_bundle` value is logged to W&B
via `wandb.config.update(env_cfg.to_dict())` automatically since it is part of
the env config.

### `run_queue.py` — `_param_label`

`_param_label` now handles string-valued params and has a short-alias table so
`sensor_bundle=proprio` produces `sb_proprio` instead of a format error:

```python
_KEY_ALIASES = {"sensor_bundle": "sb"}

def _param_label(key, value):
    short = _KEY_ALIASES.get(key, key.split(".")[-1].replace("finger_", ""))
    if isinstance(value, str):
        return f"{short}_{value}"
    return f"{short}{value:g}"
```

This change is **already applied** to `learning/run_queue.py`.

---

## Implementation plan

### Step 1 — `obs.py`: registry + hand-sensor components + bundles
- Create `tesollo_hand/obs.py`
- Define `ObsComponent`, `register`, `get`, `validate_spec`
- Register all hand-sensor components with named functions (no lambdas)
- Define `SENSOR_BUNDLES`
- Validate all bundle keys exist at module import time
- No imports from any task env

### Step 2 — `TesolloHandGraspEnv`: `_build_obs` + `obs_size`
- Add `_build_obs` method to `TesolloHandGraspEnv` in `base_grasp.py`
- Add `obs_size` cached property
- Import `obs.py` — no other changes to existing task envs yet

### Step 3 — `base_pick_and_place.py`: register task components + wire `_get_obs`
- Register pick-and-place-specific obs components at module level
- Add `_TASK_KEYS` class variable and `_task_obs_keys()` method
- Replace abstract `_get_obs` with the concrete implementation calling `_build_obs`
- Add `validate_spec` call in `_post_init`
- Update `default_config`: add `sensor_bundle`, expand `obs_noise.scales`
- Remove `_maybe_apply_obs_noise`, `_obs_joint_angles`, `_obs_joint_velocities`,
  `_obs_motor_targets`, `_obs_motor_deltas`, `_obs_fingertip_forces`,
  `_obs_fingertip_force_dirs`, `_obs_total_contact_force` (now in `obs.py`)
- Keep task-specific helpers: `_obs_cube_pos`, `_obs_cube_to_goal`,
  `_obs_goal_pos`, `_obs_goal_quat`, `_obs_last_ground_cube_pos`, `_obs_privileged`

### Step 4 — `pick_and_place.py`: replace subclasses with config factories
- Delete `PickAndPlaceProprio`, `PickAndPlaceBaseline`, `PickAndPlaceForce`,
  `PickAndPlaceForceVec`, `PickAndPlaceForceProprio`, `PickAndPlaceProprioDelta`,
  `PickAndPlaceForceProprioDelta`
- Add single `PickAndPlace(PickAndPlaceBase)` class
- Add `baseline_config`, `proprio_config`, `force_config`, `force_proprio_config`
  factory functions
- Update `__all__`

### Step 5 — Update call sites
- Replace `startswith("TesolloPickAndPlace")` with `== "TesolloPickAndPlace"` in
  `manipulation_params.py`
- Update `__init__.py` exports: remove old class names, export `PickAndPlace`
  and config factories
- Update registry: register `TesolloPickAndPlace` backed by `PickAndPlace` with
  `default_config()`; remove all old variant registrations
- Update any existing queue YAML files that reference the old variant names

### Step 6 — Verify
- Instantiate each config factory, assert `env.obs_size` matches expected value
- Assert `validate_spec` raises on bad bundle name
- Assert adding a zero-noise component does not shift other components' RNG draws
  (fix seed, compare obs vectors before and after adding an extra zero-noise key)
- Run `env_viz` with `PickAndPlace` to confirm reset and rendering still work

---

## What is explicitly not changing

- `GraspBase`, `reorient.py`, `reach.py` — keep their existing `_get_obs`.
  `_build_obs` is available to them when ready to migrate; nothing forces it now.
- `obs_noise.level` × `scales.<key>` schema — unchanged.
- Privileged state — hardcoded on `PickAndPlaceBase`, not config-driven.
- `train_jax_ppo.py` — no changes.
- The MuJoCo XML, reset logic, reward, termination — untouched.
