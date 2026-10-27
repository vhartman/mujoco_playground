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
            the policy learned to give up on hard goals by dropping the cube. With task.max_tilt_deg > 0,
            tilting the cube past it counts as the same failure (the rotation must stay about the vertical)

With task.resample_goal, holding a goal for target_hold_time draws a new goal relative
to the current yaw (consecutive goals, as in Dactyl / DeXtreme). The progress potential
is re-based on the new goal, so the jump in error is neither rewarded nor charged.
With task.goal_sequence = "monotonic", every goal (the first and each new one) is goal_step
further in one direction, drawn once per episode: the total rotation keeps growing, so past
about one grasp's range (~180 deg) further goals need a regrasp (finger gaiting).

Velocity mode (velocity.enable): instead of goal angles, a command c in {-1, 0, +1}
(redrawn on average every switch_time seconds) asks for rotation about world z at c * omega,
or for holding still (c = 0). The reward replaces progress + success with one term:
  rotate    c != 0: min(c * dyaw / (omega * dt), 1), with dyaw the change of the cube's
            twist about world z over the control step (a displacement, not a sampled
            angular velocity). Capped above, linear below, so a back-and-forth motion
            with zero net rotation never earns a positive total.
            c == 0: decrease of |yaw - yaw at the hold command| / (omega * dt), capped at 1
            (potential-based: returning to the held yaw pays back what drifting cost).
The command reaches the policy through the existing goal_quat observation as a moving
target: lookahead * omega ahead of the cube in the commanded direction, or the held yaw.
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
    cfg.task.goal_sequence = "random"  # "random": U(-max, max); "monotonic": sign * goal_step, sign per episode
    cfg.task.goal_step = float(np.pi / 2)
    # > 0: tilting the cube beyond this ends the episode with the drop cost (rotate about the vertical, upright)
    cfg.task.max_tilt_deg = 0.0
    # Exploration aid (not a reward term): cancel (1 - s) of the cube weight with an
    # upward force, s ramping start_scale -> 1 over ramp_steps per-env steps. The
    # counter survives episode resets under --full_reset (use --num_resets_per_eval=0).
    cfg.gravity_curriculum = config_dict.create(enable=False, start_scale=0.0, ramp_steps=100_000)
    cfg.velocity = config_dict.create(
        enable=False, omega=0.6, lookahead=0.5, switch_time=3.0, p_hold=0.2)
    cfg.rotate_reward = config_dict.create(
        progress=1.0,
        success=1.0,
        action_rate=0.01,
        drop=5.0,
        rotate=1.0,  # velocity mode only
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
        assert self._config.task.goal_sequence in ("random", "monotonic"), self._config.task.goal_sequence
        cache = np.load(self._config.grasp_cache_path)
        self._cache_qpos = jp.asarray(cache["qpos"])
        self._cache_qvel = jp.asarray(cache["qvel"])
        self._cache_ctrl = jp.asarray(cache["ctrl"])
        # optional per-entry spin command (velocity mode), e.g. for restarts from mid-episode states
        self._cache_cmd = jp.asarray(cache["cmd"], dtype=jp.int32) if "cmd" in cache.files else None

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_idx, k_goal = jax.random.split(rng, 3)
        i = jax.random.randint(k_idx, (), 0, self._cache_qpos.shape[0])
        qpos, qvel, ctrl = self._cache_qpos[i], self._cache_qvel[i], self._cache_ctrl[i]
        q = qpos[self._cube_qids[3]:self._cube_qids[3] + 4]
        start_yaw = 2.0 * jp.arctan2(q[3], q[0])
        delta = jax.random.uniform(k_goal, minval=-1.0, maxval=1.0) * self._config.task.max_target_angle
        sign = jp.ones(())
        if self._config.task.goal_sequence == "monotonic":
            sign = jp.where(jax.random.uniform(jax.random.fold_in(k_goal, 1)) < 0.5, -1.0, 1.0)
            delta = sign * self._config.task.goal_step
        goal_yaw = start_yaw + delta
        goal_quat = _yaw_quat(goal_yaw)
        rng, k_cmd = jax.random.split(rng)
        cmd = self._sample_cmd(k_cmd)
        if self._cache_cmd is not None:
            cmd = self._cache_cmd[i]
        if self._config.velocity.enable:
            goal_quat = self._carrot(start_yaw, cmd, start_yaw)
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
            "goal_sign": sign,
            "motor_targets": ctrl,
            "last_action": last_action,
            "ori_error": self._cube_orientation_error(data, goal_quat),
            "success_count": jp.zeros((), dtype=jp.int32),
            "hold_steps": jp.zeros((), dtype=jp.int32),
            "goals_reached": jp.zeros((), dtype=jp.int32),
            "cmd": cmd,
            "cache_idx": i,  # which cache entry this episode started from
            "ref_yaw": start_yaw,
            "prev_yaw": start_yaw,
            # gravity-curriculum counter; survives resets under --full_reset
            "AutoResetWrapper_preserve_info": {"grav_steps": jp.zeros((), dtype=jp.int32)},
        }
        metrics = {k: jp.zeros(()) for k in (
            "reward/progress", "reward/success", "reward/action_rate", "reward/drop",
            "reward/rotate", "ori_error_deg", "goal_delta_deg", "gravity_scale", "goals_reached",
            "cmd", "yaw_rate_dps", "tilt_deg", "tilted")}
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
        q = self.get_cube_orientation(data)
        tilt = jp.arccos(jp.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))
        tilted = jp.zeros((), dtype=bool)
        if self._config.task.max_tilt_deg > 0:
            tilted = tilt > np.deg2rad(self._config.task.max_tilt_deg)
        failed = dropped | tilted
        nan = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        hold = (info["hold_steps"] + 1) * success.astype(jp.int32)
        reached = hold == int(round(self._config.task.target_hold_time / self.dt))

        yaw = _cube_yaw(q)
        dyaw = _wrap(yaw - info["prev_yaw"])
        w = self._config.rotate_reward
        terms = {
            "progress": w.progress * (info["ori_error"] - err),
            "success": w.success * reached.astype(float),
            "rotate": jp.zeros(()),
            "action_rate": -w.action_rate * jp.sum(jp.square(action - info["last_action"])),
            "drop": -w.drop * failed.astype(float),
        }
        if self._config.velocity.enable:
            v = self._config.velocity
            unit = v.omega * self.dt
            c = info["cmd"].astype(float)
            spin = jp.minimum(c * dyaw / unit, 1.0)
            back = (jp.abs(_wrap(info["prev_yaw"] - info["ref_yaw"]))
                    - jp.abs(_wrap(yaw - info["ref_yaw"]))) / unit
            terms["rotate"] = w.rotate * jp.where(info["cmd"] != 0, spin, jp.minimum(back, 1.0))
            terms["progress"] = jp.zeros(())
            terms["success"] = jp.zeros(())
        info["ori_error"] = err
        info["last_action"] = action
        info["success_count"] = info["success_count"] + success.astype(jp.int32)
        info["hold_steps"] = hold
        info["goals_reached"] = info["goals_reached"] + reached.astype(jp.int32)
        state.metrics["cmd"] = info["cmd"].astype(float)
        state.metrics["yaw_rate_dps"] = jp.rad2deg(dyaw / self.dt)
        state.metrics["tilt_deg"] = jp.rad2deg(tilt)
        state.metrics["tilted"] = tilted.astype(float)
        info["prev_yaw"] = yaw
        if self._config.velocity.enable:
            rng, k_sw, k_cmd = jax.random.split(info["rng"], 3)
            switch = jax.random.uniform(k_sw) < self.dt / self._config.velocity.switch_time
            info["rng"] = rng
            info["cmd"] = jp.where(switch, self._sample_cmd(k_cmd), info["cmd"])
            info["ref_yaw"] = jp.where(switch, yaw, info["ref_yaw"])
            info["goal_quat"] = self._carrot(yaw, info["cmd"], info["ref_yaw"])
            info["ori_error"] = self._cube_orientation_error(data, info["goal_quat"])
            data = data.replace(mocap_quat=jp.broadcast_to(info["goal_quat"], data.mocap_quat.shape))
        elif self._config.task.resample_goal:
            rng, k_goal = jax.random.split(info["rng"])
            delta = jax.random.uniform(k_goal, minval=-1.0, maxval=1.0) * self._config.task.max_target_angle
            if self._config.task.goal_sequence == "monotonic":
                delta = info["goal_sign"] * self._config.task.goal_step
            yaw = yaw + delta
            new_goal = _yaw_quat(yaw)
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
        done = (failed | nan).astype(float)
        return state.replace(data=data, obs=obs, reward=sum(terms.values()), done=done, info=info)

    def _sample_cmd(self, key) -> jax.Array:
        p = self._config.velocity.p_hold
        return jax.random.choice(key, jp.array([-1, 0, 1]), p=jp.array([(1 - p) / 2, p, (1 - p) / 2]))

    def _carrot(self, yaw, cmd, ref_yaw) -> jax.Array:
        """Velocity-mode target: lookahead * omega ahead of the cube, or the held yaw."""
        v = self._config.velocity
        return _yaw_quat(jp.where(cmd == 0, ref_yaw, yaw + cmd * v.omega * v.lookahead))

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


def _yaw_quat(yaw):
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])


def _cube_yaw(q):
    """Twist of quaternion q (w, x, y, z) about world z (swing-twist decomposition)."""
    return 2.0 * jp.arctan2(q[3], q[0])


def _wrap(a):
    return (a + jp.pi) % (2 * jp.pi) - jp.pi


def domain_randomize(model, rng):
    return downwards_rotate_z.domain_randomize(model, rng)
