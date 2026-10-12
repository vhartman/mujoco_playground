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
"""In-hand z rotation to any angle with a downward-facing hand, minimal reward.

Episodes start from a cache of stable lifted grasps (hand + cube states recorded
from an existing policy), so there is no pickup phase. The goal is a z rotation
of the cube relative to its starting yaw, uniform in [-max, max].

Reward (per control step), four terms:
  progress  decrease of the 3D orientation error to the goal (potential-based: an
            episode's total is initial minus final error, whatever the path, so
            tilted or oscillating motions cannot farm it)
  success   once per goal, when the 3D error has stayed within tolerance for
            target_hold_time (an event, not a per-step rate: hovering in and out
            of tolerance earns nothing)
  action    -|a_t - a_{t-1}|^2
  drop      a constant cost when the cube falls (its lowest point touches the
            floor); the episode ends. Constant, not tied to the current error: with
            pi - error, dropping right after a new large goal was nearly free, and
            the policy learned to give up on hard goals by dropping the cube

With task.resample_goal, holding a goal for target_hold_time draws a new goal relative
to the current yaw (consecutive goals, as in Dactyl / DeXtreme). The progress potential
is re-based on the new goal, so the jump in error is neither rewarded nor charged.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.tesollo_hand import downwards_rotate_z


def default_config() -> config_dict.ConfigDict:
    cfg = downwards_rotate_z.default_config()
    cfg.episode_length = 200
    cfg.ema_alpha = 0.2
    cfg.sensor_bundle = "baseline"
    cfg.grasp_cache_path = ""  # .npz with qpos, qvel, ctrl (see make_grasp_cache.py)
    cfg.task.max_target_angle = float(np.pi)  # goal yaw = start yaw + U(-max, max)
    cfg.task.resample_goal = False  # new goal after each success (training); off for evals
    # Exploration aid (not a reward term): cancel (1 - s) of the cube weight with an
    # upward force, s ramping start_scale -> 1 over ramp_steps per-env steps. The
    # counter survives episode resets under --full_reset (use --num_resets_per_eval=0).
    cfg.gravity_curriculum = config_dict.create(enable=False, start_scale=0.0, ramp_steps=100_000)
    cfg.rotate_reward = config_dict.create(
        progress=1.0,
        success=1.0,
        action_rate=0.01,
        drop=5.0,
    )
    return cfg


class RotateInHand(downwards_rotate_z.DownwardsRotateZ):
    """In-hand rotation from a grasp cache with a four-term reward."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config=config or default_config(), config_overrides=config_overrides)
        cache = np.load(self._config.grasp_cache_path)
        self._cache_qpos = jp.asarray(cache["qpos"])
        self._cache_qvel = jp.asarray(cache["qvel"])
        self._cache_ctrl = jp.asarray(cache["ctrl"])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_idx, k_goal = jax.random.split(rng, 3)
        i = jax.random.randint(k_idx, (), 0, self._cache_qpos.shape[0])
        qpos, qvel, ctrl = self._cache_qpos[i], self._cache_qvel[i], self._cache_ctrl[i]
        q = qpos[self._cube_qids[3]:self._cube_qids[3] + 4]
        start_yaw = 2.0 * jp.arctan2(q[3], q[0])
        delta = jax.random.uniform(k_goal, minval=-1.0, maxval=1.0) * self._config.task.max_target_angle
        goal_yaw = start_yaw + delta
        goal_quat = jp.array([jp.cos(goal_yaw / 2), 0.0, 0.0, jp.sin(goal_yaw / 2)])
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl,
            mocap_pos=self._init_mpos, mocap_quat=goal_quat,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)  # sensors (cube pose) are read below
        last_action = 2.0 * (ctrl - self._lowers) / (self._uppers - self._lowers) - 1.0
        info = {
            "rng": rng,
            "goal_quat": goal_quat,
            "goal_delta": delta,
            "motor_targets": ctrl,
            "last_action": last_action,
            "ori_error": self._cube_orientation_error(data, goal_quat),
            "success_count": jp.zeros((), dtype=jp.int32),
            "hold_steps": jp.zeros((), dtype=jp.int32),
            "goals_reached": jp.zeros((), dtype=jp.int32),
            # gravity-curriculum counter; survives resets under --full_reset
            "AutoResetWrapper_preserve_info": {"grav_steps": jp.zeros((), dtype=jp.int32)},
        }
        metrics = {k: jp.zeros(()) for k in (
            "reward/progress", "reward/success", "reward/action_rate", "reward/drop",
            "ori_error_deg", "goal_delta_deg", "gravity_scale", "goals_reached")}
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        target = self._lowers + 0.5 * (action + 1.0) * (self._uppers - self._lowers)
        motor_targets = (
            self._config.ema_alpha * jp.clip(target, self._lowers, self._uppers)
            + (1.0 - self._config.ema_alpha) * state.info["motor_targets"]
        )
        info = state.info
        data_in, gravity_scale = state.data, jp.ones(())
        if self._config.gravity_curriculum.enable:
            # Exploration aid, not a reward term: cancel (1 - s) of the cube weight
            # with an upward force, s ramping start_scale -> 1 over ramp_steps.
            gc = self._config.gravity_curriculum
            n = info["AutoResetWrapper_preserve_info"]["grav_steps"]
            gravity_scale = gc.start_scale + (1.0 - gc.start_scale) * jp.clip(n / gc.ramp_steps, 0.0, 1.0)
            lift = (1.0 - gravity_scale) * self._cube_mass * (-self._mj_model.opt.gravity[2])
            data_in = state.data.replace(
                xfrc_applied=state.data.xfrc_applied.at[self._cube_body_id, 2].set(lift))
            info["AutoResetWrapper_preserve_info"] = {"grav_steps": n + 1}
        data = mjx_env.step(self.mjx_model, data_in, motor_targets, self.n_substeps)
        info["motor_targets"] = motor_targets
        state.metrics["gravity_scale"] = gravity_scale

        err = self._cube_orientation_error(data, info["goal_quat"])
        success = err < self._config.task.ori_tolerance_rad
        # The hand is above the floor, so a dropped cube ends up touching it.
        dropped = self._cube_lowest_point(data) < 0.003
        nan = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        hold = (info["hold_steps"] + 1) * success.astype(jp.int32)
        reached = hold == int(round(self._config.task.target_hold_time / self.dt))

        w = self._config.rotate_reward
        terms = {
            "progress": w.progress * (info["ori_error"] - err),
            "success": w.success * reached.astype(float),
            "action_rate": -w.action_rate * jp.sum(jp.square(action - info["last_action"])),
            "drop": -w.drop * dropped.astype(float),
        }
        info["ori_error"] = err
        info["last_action"] = action
        info["success_count"] = info["success_count"] + success.astype(jp.int32)
        info["hold_steps"] = hold
        info["goals_reached"] = info["goals_reached"] + reached.astype(jp.int32)
        if self._config.task.resample_goal:
            rng, k_goal = jax.random.split(info["rng"])
            q = self.get_cube_orientation(data)
            delta = jax.random.uniform(k_goal, minval=-1.0, maxval=1.0) * self._config.task.max_target_angle
            yaw = 2.0 * jp.arctan2(q[3], q[0]) + delta
            new_goal = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
            info["rng"] = rng
            info["goal_quat"] = jp.where(reached, new_goal, info["goal_quat"])
            info["goal_delta"] = jp.where(reached, delta, info["goal_delta"])
            info["hold_steps"] = jp.where(reached, 0, info["hold_steps"])
            info["ori_error"] = self._cube_orientation_error(data, info["goal_quat"])
            data = data.replace(mocap_quat=jp.broadcast_to(info["goal_quat"], data.mocap_quat.shape))

        for k, v in terms.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["ori_error_deg"] = jp.rad2deg(err)
        state.metrics["goal_delta_deg"] = jp.rad2deg(info["goal_delta"])
        state.metrics["goals_reached"] = info["goals_reached"].astype(float)
        obs = self._get_obs(data, info)
        done = (dropped | nan).astype(float)
        return state.replace(data=data, obs=obs, reward=sum(terms.values()), done=done, info=info)

    def _cube_lowest_point(self, data) -> jax.Array:
        """Height of the cube box's lowest point above the floor (z = 0)."""
        q = self.get_cube_orientation(data)
        w, x, y, z = q[0], q[1], q[2], q[3]
        r2 = jp.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        half = self.mjx_model.geom_size[self._cube_geom_id]
        return self.get_cube_position(data)[2] - jp.sum(half * jp.abs(r2))

    def _obs_privileged(self, data, info) -> jax.Array:
        # The parent's critic input lacks the cube orientation; the value of this
        # task depends on it directly.
        return jp.concatenate([
            super()._obs_privileged(data, info),
            self.get_cube_orientation(data),
            jp.atleast_1d(info["ori_error"]),
        ])


def domain_randomize(model, rng):
    return downwards_rotate_z.domain_randomize(model, rng)
