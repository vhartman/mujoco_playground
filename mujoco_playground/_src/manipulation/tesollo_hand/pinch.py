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
"""Pinch task for tesollo hand: reach a target contact force on cube sides."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_wrist as tesollo_hand_base
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_wrist_constants as consts,
)

_PINCH_XML = consts.ROOT_PATH / "xmls" / "scene_mjx_cube_pinch.xml"

# Wrist (3) + thumb/dg_1 (4) + index/dg_2 (4) = 11 controlled DOFs.
# Middle/ring/pinky are held at their keyframe defaults.
_N_ACTIVE = 11


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=1000,
        # Target total contact force from hand on cube sides, in Newtons.
        force_target=10.0,
        # Force must stay within ±force_tolerance N of target for success_hold_time seconds.
        force_tolerance=0.5,
        success_hold_time=3.0,
        obs_noise=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.001,
                joint_vel=0.01,
                cube_pos=0.005,
                cube_quat=0.02,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                cube_force=5.0,
                fingertip_reach=0.5,
                cube_lin_vel=-0.1,
                cube_ang_vel=-0.8,
                hand_pose=-0.2,
                wrist_pose=-0.5,
                action_rate=-0.005,
                joint_vel=-0.01,
                energy=-1e-3,
                wrist_vel=-0.1,
                termination=-100.0,
            ),
            success_reward=10.0,
        ),
        pert_config=config_dict.create(
            enable=False,
            linear_velocity_pert=[0.0, 3.0],
            angular_velocity_pert=[0.0, 0.5],
            pert_duration_steps=[1, 100],
            pert_wait_steps=[60, 150],
        ),
        # PID gains for all 23 actuators (3 wrist + 5 fingers x 4).
        # When enable=True these override the XML-baked values at load time.
        # kp_per_actuator / kv_per_actuator: optional length-23 lists that
        # override the group defaults on a per-actuator basis.
        pid_gains=config_dict.create(
            enable=False,
            wrist_kp=[10.0, 75.0, 10.0],
            wrist_kv=[2.0, 10.0, 2.0],
            finger_kp=3.0,
            finger_kv=0.0,
            kp_per_actuator=[],
            kv_per_actuator=[],
        ),
        impl="warp",
        nconmax=200 * 8192,
        njmax=1024,
    )


class CubePinch(tesollo_hand_base.TesolloHandWristEnv):
    """Pinch a cube resting on the floor with a target contact force of 10 N."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=_PINCH_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        if self._config.pid_gains.enable:
            self._apply_pid_gains()

        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos, dtype=float)
        # Active: wrist(3) + thumb(4) + index(4). Fixed: middle + ring + pinky.
        self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:_N_ACTIVE, 0])
        self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:_N_ACTIVE, 1])
        self._fixed_ctrl_vals = jp.array(home_key.ctrl[_N_ACTIVE:])  # 12-dim
        self._wrist_qids = mjx_env.get_qpos_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._wrist_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._finger_qids = mjx_env.get_qpos_ids(self.mj_model, consts.FINGER_NAMES)
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]
        # Initial cube xy used for drift-based termination.
        self._init_cube_pos = jp.array(home_key.qpos[self._cube_qids[:3]])

    def _apply_pid_gains(self) -> None:
        """Override actuator PID gains from config, replacing XML-baked values.

        MuJoCo <position kp kv> maps to:
          gainprm[:, 0] = kp
          biasprm[:, 1] = -kp  (position error term)
          biasprm[:, 2] = -kv  (velocity error term)
        """
        cfg = self._config.pid_gains
        nu = self._mjx_model.nu  # 23

        if cfg.kp_per_actuator:
            kp = jp.array(cfg.kp_per_actuator, dtype=float)
        else:
            kp = jp.concatenate([
                jp.array(cfg.wrist_kp, dtype=float),      # (3,)
                jp.full((nu - 3,), cfg.finger_kp),        # (20,)
            ])

        if cfg.kv_per_actuator:
            kv = jp.array(cfg.kv_per_actuator, dtype=float)
        else:
            kv = jp.concatenate([
                jp.array(cfg.wrist_kv, dtype=float),      # (3,)
                jp.full((nu - 3,), cfg.finger_kv),        # (20,)
            ])

        gainprm = self._mjx_model.actuator_gainprm.at[:, 0].set(kp)
        biasprm = (
            self._mjx_model.actuator_biasprm
            .at[:, 1].set(-kp)
            .at[:, 2].set(-kv)
        )
        self._mjx_model = self._mjx_model.tree_replace({
            "actuator_gainprm": gainprm,
            "actuator_biasprm": biasprm,
        })

    @property
    def action_size(self) -> int:
        return _N_ACTIVE

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pos_rng, pert1, pert2, pert3 = jax.random.split(rng, 5)

        # Randomise only the 11 active joints; fixed joints stay at keyframe.
        q_active = jp.clip(
            self._default_pose[:_N_ACTIVE] + 0.05 * jax.random.normal(pos_rng, (_N_ACTIVE,)),
            self._lowers,
            self._uppers,
        )
        q_hand = jp.concatenate([q_active, self._default_pose[_N_ACTIVE:]])
        # Cube starts at keyframe position (on floor).
        q_cube = self._init_q[self._cube_qids]
        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.zeros(self._mj_model.nv)
        # Full ctrl: active joints + fixed joints locked at keyframe values.
        ctrl = jp.concatenate([q_active, self._fixed_ctrl_vals])

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=ctrl,
            qvel=qvel,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        pert_wait_steps = jax.random.randint(
            pert1, (1,),
            minval=self._config.pert_config.pert_wait_steps[0],
            maxval=self._config.pert_config.pert_wait_steps[1],
        )
        pert_duration_steps = jax.random.randint(
            pert2, (1,),
            minval=self._config.pert_config.pert_duration_steps[0],
            maxval=self._config.pert_config.pert_duration_steps[1],
        )
        pert_lin = jax.random.uniform(
            pert3,
            minval=self._config.pert_config.linear_velocity_pert[0],
            maxval=self._config.pert_config.linear_velocity_pert[1],
        )
        pert_ang = jax.random.uniform(
            pert3,
            minval=self._config.pert_config.angular_velocity_pert[0],
            maxval=self._config.pert_config.angular_velocity_pert[1],
        )

        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "consecutive_success_steps": jp.zeros(()),
            "last_act": jp.zeros(_N_ACTIVE),
            "last_last_act": jp.zeros(_N_ACTIVE),
            "motor_targets": data.ctrl,  # full 23-dim
            "pert_wait_steps": pert_wait_steps,
            "pert_duration_steps": pert_duration_steps,
            "pert_vel": jp.array([pert_lin] * 3 + [pert_ang] * 3),
            "pert_dir": jp.zeros(6, dtype=float),
            "last_pert_step": jp.array([-jp.inf], dtype=float),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/success"] = jp.zeros((), dtype=float)
        metrics["consecutive_success_steps"] = jp.zeros(())
        metrics["steps_since_last_success"] = 0
        metrics["success_count"] = 0
        metrics["f_thumb"] = jp.zeros(())
        metrics["f_index"] = jp.zeros(())
        metrics["opposing"] = jp.zeros(())
        metrics["termination/drift"] = jp.zeros(())
        metrics["termination/nan"] = jp.zeros(())
        metrics["termination/tip_on_ground"] = jp.zeros(())

        obs = self._get_obs(data, info)
        rew, done = jp.zeros(2)
        return mjx_env.State(data, obs, rew, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        # action is _N_ACTIVE-dim; expand to full 23-dim ctrl.
        delta = action * self._config.action_scale
        active_ctrl = jp.clip(
            state.data.ctrl[:_N_ACTIVE] + delta, self._lowers, self._uppers
        )
        full_ctrl = jp.concatenate([active_ctrl, self._fixed_ctrl_vals])
        motor_targets = (
            self._config.ema_alpha * full_ctrl
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )
        # Fixed joints must not drift under EMA.
        motor_targets = jp.concatenate([motor_targets[:_N_ACTIVE], self._fixed_ctrl_vals])

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        # Success: pinch force held within ±force_tolerance N of target for
        # success_hold_time seconds. Pinch force only fires when thumb and index
        # touch opposing sides of the cube.
        f_thumb, f_index, opposing = self._pinch_components(data)
        total_force = jp.minimum(f_thumb, f_index) * opposing
        in_tolerance = (
            (total_force >= self._config.force_target - self._config.force_tolerance)
            & (total_force <= self._config.force_target + self._config.force_tolerance)
        )
        hold_steps_required = jp.round(self._config.success_hold_time / self._config.ctrl_dt)
        consecutive = jp.where(
            in_tolerance,
            state.info["consecutive_success_steps"] + 1.0,
            jp.zeros(()),
        )
        state.info["consecutive_success_steps"] = consecutive
        success = consecutive >= hold_steps_required

        state.info["steps_since_last_success"] = jp.where(
            success, 0, state.info["steps_since_last_success"] + 1
        )
        state.info["success_count"] = jp.where(
            success, state.info["success_count"] + 1, state.info["success_count"]
        )
        state.metrics["steps_since_last_success"] = state.info["steps_since_last_success"]
        state.metrics["success_count"] = state.info["success_count"]
        state.metrics["consecutive_success_steps"] = consecutive

        done, term_reasons = self._get_termination(data, state.info)
        state.metrics["termination/drift"] = term_reasons["drift"].astype(float)
        state.metrics["termination/nan"] = term_reasons["nan"].astype(float)
        state.metrics["termination/tip_on_ground"] = term_reasons["tip_on_ground"].astype(float)
        obs = self._get_obs(data, state.info)
        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        rew = sum(rewards.values()) * self.dt
        rew += success * self._config.reward_config.success_reward

        state.info["step"] += 1
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.metrics["reward/success"] = success.astype(float)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["f_thumb"] = f_thumb
        state.metrics["f_index"] = f_index
        state.metrics["opposing"] = opposing

        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del info
        cube_xy = self.get_cube_position(data)[:2]
        drift = jp.linalg.norm(cube_xy - self._init_cube_pos[:2]) > 0.15
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        tip_on_ground = jp.any(tips[:2, 2] < 0.005)
        reasons = {"drift": drift, "nan": nans, "tip_on_ground": tip_on_ground}
        return drift | nans | tip_on_ground, reasons

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        joint_angles = data.qpos[self._hand_qids]
        joint_vel = data.qvel[self._hand_dqids]
        cube_pos = self.get_cube_position(data)
        cube_quat = self.get_cube_orientation(data)

        # Noisy proprioception + cube pose for the actor.
        info["rng"], pos_rng, vel_rng, cpos_rng, cquat_rng = jax.random.split(
            info["rng"], 5
        )
        noisy_q = joint_angles + (
            2 * jax.random.uniform(pos_rng, shape=joint_angles.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_pos
        noisy_qdot = joint_vel + (
            2 * jax.random.uniform(vel_rng, shape=joint_vel.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_vel
        noisy_cube_pos = cube_pos + (
            2 * jax.random.uniform(cpos_rng, shape=cube_pos.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.cube_pos
        noisy_cube_quat = cube_quat + (
            2 * jax.random.uniform(cquat_rng, shape=cube_quat.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.cube_quat
        noisy_cube_quat = noisy_cube_quat / (
            jp.linalg.norm(noisy_cube_quat) + 1e-6
        )

        state = jp.concatenate(
            [noisy_q, noisy_qdot, noisy_cube_pos, noisy_cube_quat]
        )  # 53

        # Privileged critic state: ground truth proprioception + cube kinematics
        # + fingertip positions. Contact forces excluded (reward signal only).
        privileged_state = jp.concatenate([
            joint_angles,
            joint_vel,
            self.get_cube_position(data),
            self.get_cube_linvel(data),
            self.get_cube_angvel(data),
            self.get_fingertip_positions(data),
        ])

        return {"state": state, "privileged_state": privileged_state}

    def _total_contact_force(self, data: mjx.Data) -> jax.Array:
        """Sum of contact force norms: hand touching cube sides (floor excluded by sensor)."""
        forces = mjx_env.get_sensor_data(
            self.mj_model, data, "cube_force"
        ).reshape(-1, 3)
        return jp.sum(jp.linalg.norm(forces, axis=1))

    def _fingertip_force(self, data: mjx.Data, sensor_name: str) -> jax.Array:
        """Total contact force magnitude on a single fingertip vs. the cube."""
        f = mjx_env.get_sensor_data(self.mj_model, data, sensor_name).reshape(-1, 3)
        return jp.sum(jp.linalg.norm(f, axis=1))

    def _pinch_components(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return (pinch_force, f_thumb, f_index, opposing) for logging and reward."""
        f_thumb = self._fingertip_force(data, "rl_dg_1_tip_cube_force")
        f_index = self._fingertip_force(data, "rl_dg_2_tip_cube_force")

        cube_pos = self.get_cube_position(data)
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        d_thumb = tips[0] - cube_pos
        d_index = tips[1] - cube_pos
        d_thumb = d_thumb / (jp.linalg.norm(d_thumb) + 1e-6)
        d_index = d_index / (jp.linalg.norm(d_index) + 1e-6)

        opposing = jp.clip(-jp.dot(d_thumb, d_index), 0.0, 1.0)
        return f_thumb, f_index, opposing

    def _pinch_force(self, data: mjx.Data) -> jax.Array:
        return jp.minimum(*self._pinch_components(data)[:2]) * self._pinch_components(data)[2]

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics

        # Primary: tolerance reward peaked at force_target (10 N) using the
        # opposing-side pinch force between thumb and index. Fires only when the
        # two fingers contact the cube on opposite sides.
        total_force = self._pinch_force(data)
        force_reward = reward.tolerance(
            total_force,
            bounds=(self._config.force_target, self._config.force_target),
            margin=self._config.force_target,
            sigmoid="gaussian",
        )

        cube_pos = self.get_cube_position(data)
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        reach_dists = jp.linalg.norm(tips[:2] - cube_pos, axis=1)
        fingertip_reach = jp.sum(
            reward.tolerance(reach_dists, bounds=(0, 0.035), margin=0.1, sigmoid="reciprocal")
        )

        return {
            "cube_force": force_reward,
            "fingertip_reach": fingertip_reach,
            "cube_lin_vel": self._cube_lin_velocity(data),
            "cube_ang_vel": self._cube_ang_velocity(data),
            "hand_pose": jp.sum(jp.square(data.qpos[self._hand_qids] - self._default_pose)),
            "wrist_pose": jp.sum(jp.square(data.qpos[self._wrist_qids] - self._default_wrist_pose)),
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "joint_vel": self._cost_joint_vel(data),
            "wrist_vel": jp.sum(jp.square(data.qvel[self._wrist_dqids])),
            "energy": self._cost_energy(data.qvel[self._hand_dqids], data.actuator_force),
            "termination": done,
        }

    def _cube_lin_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_linvel(data))

    def _cube_ang_velocity(self, data: mjx.Data) -> jax.Array:
        return math.norm(self.get_cube_angvel(data))

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        c1 = jp.sum(jp.square(act - last_act))
        c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
        return c1 + c2

    def _cost_joint_vel(self, data: mjx.Data) -> jax.Array:
        max_velocity = 5.0
        vel_tolerance = 1.0
        return jp.sum((data.qvel[self._hand_dqids] / (max_velocity - vel_tolerance)) ** 2)

    def _maybe_apply_perturbation(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            d = jax.random.normal(rng, (6,))
            return d / jp.linalg.norm(d)

        def get_xfrc(state, pert_dir, i):
            u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
            force = (
                u_t
                * self._cube_mass
                * state.info["pert_vel"]
                / (state.info["pert_duration_steps"] * self.dt)
            )
            xfrc = jp.zeros((self.mjx_model.nbody, 6))
            return xfrc.at[self._cube_body_id].set(force * pert_dir)

        step, last_pert_step = state.info["step"], state.info["last_pert_step"]
        start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
        start_pert &= step != 0
        last_pert_step = jp.where(start_pert, step, last_pert_step)
        duration = jp.clip(step - last_pert_step, 0, 100_000)
        in_pert_interval = duration < state.info["pert_duration_steps"]

        pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
        xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

        state.info["pert_dir"] = pert_dir
        state.info["last_pert_step"] = last_pert_step
        return state.replace(data=state.data.replace(xfrc_applied=xfrc))


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = CubePinch().mj_model
    cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4",
        "rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4",
        "rl_dg_3_1", "rl_dg_3_2", "rl_dg_3_3", "rl_dg_3_4",
        "rl_dg_4_1", "rl_dg_4_2", "rl_dg_4_3", "rl_dg_4_4",
        "rl_dg_5_1", "rl_dg_5_2", "rl_dg_5_3", "rl_dg_5_4",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    silicone_geom_ids = [
        mj_model.geom(g).id for g in [
            "rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip",
        ]
    ]

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        silicone_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=2.0)
        geom_friction = model.geom_friction.at[silicone_geom_ids, 0].set(silicone_friction)

        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.5, maxval=1.5)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0.at[hand_qids].set(
            model.qpos0[hand_qids]
            + jax.random.uniform(key, shape=(23,), minval=-0.05, maxval=0.05)
        )

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(23,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(23,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (23,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction, body_mass, body_inertia, body_ipos, qpos0,
            dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
        )

    (
        geom_friction, body_mass, body_inertia, body_ipos, qpos0,
        dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "body_mass": 0, "body_inertia": 0, "body_ipos": 0,
        "qpos0": 0, "dof_frictionloss": 0, "dof_armature": 0, "dof_damping": 0,
        "actuator_gainprm": 0, "actuator_biasprm": 0,
    })
    model = model.tree_replace({
        "geom_friction": geom_friction, "body_mass": body_mass,
        "body_inertia": body_inertia, "body_ipos": body_ipos, "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss, "dof_armature": dof_armature,
        "dof_damping": dof_damping, "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })
    return model, in_axes
