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
"""Abstract base for all CubePinch environment variants."""

import abc
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_wrist as tesollo_hand_base

# Thumb (dg_1, 4) + index (dg_2, 4) = 8 controlled DOFs.
# Wrist, middle, ring, pinky and cube are frozen by the scene builder.
_N_ACTIVE = 8
_ACTIVE_JOINT_NAMES = [f"rj_dg_{f}_{i}" for f in (1, 2) for i in range(1, 5)]


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
                fingertip_reach=2.0,
                fingertip_height=2.0,
                palm_height=1.5,
                action_rate=-0.005,
                joint_vel=-0.01,
                energy=-1e-3,
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
        pid_gains=config_dict.create(
            enable=False,
            finger_kp=3.0,
            finger_kv=0.0,
            kp_per_actuator=[],
            kv_per_actuator=[],
        ),
        use_ctrl_delta=False,
        impl="warp",
        nconmax=200 * 8192,
        njmax=1024,
    )


class CubePinchBase(tesollo_hand_base.TesolloHandWristEnv, abc.ABC):
    """Abstract base class for all CubePinch environment variants."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(
            xml_source=xml_path,
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
        # Thumb (dg_1) + index (dg_2) = 8 active DOFs.
        self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:_N_ACTIVE, 0])
        self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:_N_ACTIVE, 1])
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, _ACTIVE_JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, _ACTIVE_JOINT_NAMES)
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        self._default_pose = self._init_q  # 8-dim: thumb+index keyframe qpos
        # Cube freejoint removed; get fixed world position via forward kinematics.
        _init_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, _init_data)
        self._init_cube_pos = jp.array(_init_data.xpos[self._cube_body_id])

    def _apply_pid_gains(self) -> None:
        """Override actuator PID gains from config, replacing XML-baked values.

        MuJoCo <position kp kv> maps to:
          gainprm[:, 0] = kp
          biasprm[:, 1] = -kp  (position error term)
          biasprm[:, 2] = -kv  (velocity error term)
        """
        cfg = self._config.pid_gains
        nu = self._mjx_model.nu  # 8

        if cfg.kp_per_actuator:
            kp = jp.array(cfg.kp_per_actuator, dtype=float)
        else:
            kp = jp.full((nu,), cfg.finger_kp)

        if cfg.kv_per_actuator:
            kv = jp.array(cfg.kv_per_actuator, dtype=float)
        else:
            kv = jp.full((nu,), cfg.finger_kv)

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

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def action_size(self) -> int: ...

    @abc.abstractmethod
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation: ...

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_joint_angles(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Noisy joint angles (23-dim)."""
        joint_angles = data.qpos[self._hand_qids]
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=joint_angles.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_pos
        return joint_angles + noise

    def _obs_joint_velocities(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Noisy joint velocities (23-dim)."""
        joint_vel = data.qvel[self._hand_dqids]
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=joint_vel.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_vel
        return joint_vel + noise

    def _obs_cube_pos(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Noisy cube position (3-dim)."""
        cube_pos = self.get_cube_position(data)
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=cube_pos.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.cube_pos
        return cube_pos + noise

    def _obs_cube_quat(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Noisy cube quaternion, renormalized (4-dim)."""
        cube_quat = self.get_cube_orientation(data)
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=cube_quat.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.cube_quat
        noisy_quat = cube_quat + noise
        return noisy_quat / (jp.linalg.norm(noisy_quat) + 1e-6)

    def _obs_motor_targets(self, info: dict[str, Any]) -> jax.Array:
        """Active motor targets (11-dim), no noise."""
        return info["motor_targets"][:_N_ACTIVE]

    def _obs_ctrl_delta(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Position error of active joints to their targets (11-dim), no noise."""
        return info["motor_targets"][:_N_ACTIVE] - data.qpos[self._hand_qids[:_N_ACTIVE]]

    def _obs_force(self, data: mjx.Data) -> jax.Array:
        """Pinch force terms: [f_thumb, f_index, total_force_gated] (3-dim), no noise."""
        f_thumb = self._fingertip_force(data, "rl_dg_1_tip_cube_force")
        f_index = self._fingertip_force(data, "rl_dg_2_tip_cube_force")
        contact_gate = jp.clip(jp.minimum(f_thumb, f_index) / 0.5, 0.0, 1.0)
        total_force_gated = self._total_contact_force(data) * contact_gate
        return jp.array([f_thumb, f_index, total_force_gated])

    def _obs_object(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Default object observation: cube_pos(3) + cube_quat(4) = 7-dim.

        Subclasses can override to change the object representation.
        Note: this consumes two RNG keys from info["rng"].
        """
        return jp.concatenate([
            self._obs_cube_pos(data, info),
            self._obs_cube_quat(data, info),
        ])

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Ground-truth privileged critic state (48-dim), no noise.

        q(8) + q_dot(8) + cube_pos(3) + cube_linvel(3) + cube_angvel(3)
        + fingertips_palm(15) + ctrl_targets(8) = 48
        """
        return jp.concatenate([
            data.qpos[self._hand_qids],
            data.qvel[self._hand_dqids],
            self.get_cube_position(data),
            self.get_cube_linvel(data),
            self.get_cube_angvel(data),
            self.get_fingertip_positions(data),
            info["motor_targets"][:_N_ACTIVE],
        ])

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pos_rng, pert1, pert2, pert3 = jax.random.split(rng, 5)

        # Randomise all 8 active joints (thumb + index); cube is fixed.
        qpos = jp.clip(
            self._default_pose + 0.05 * jax.random.normal(pos_rng, (_N_ACTIVE,)),
            self._lowers,
            self._uppers,
        )
        qvel = jp.zeros(self._mj_model.nv)
        ctrl = qpos

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

        delta = action * self._config.action_scale
        active_ctrl = jp.clip(
            state.data.ctrl[:_N_ACTIVE] + delta, self._lowers, self._uppers
        )
        motor_targets = (
            self._config.ema_alpha * active_ctrl
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

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

        done, term_reasons = self._get_termination(data)
        state.metrics["termination/drift"] = term_reasons["drift"].astype(float)
        state.metrics["termination/nan"] = term_reasons["nan"].astype(float)
        state.metrics["termination/tip_on_ground"] = term_reasons["tip_on_ground"].astype(float)
        obs = self._get_obs(data, state.info)
        rewards = self._get_reward(
            data, action, state.info, done,
            f_thumb=f_thumb, f_index=f_index, opposing=opposing,
        )
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
        self, data: mjx.Data
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cube_xy = self.get_cube_position(data)[:2]
        drift = jp.linalg.norm(cube_xy - self._init_cube_pos[:2]) > 0.15
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        tip_on_ground = jp.any(tips[:2, 2] < 0.005)
        reasons = {"drift": drift, "nan": nans, "tip_on_ground": tip_on_ground}
        return drift | nans | tip_on_ground, reasons

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return (f_thumb, f_index, opposing) for logging and reward."""
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

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        f_thumb: jax.Array,
        f_index: jax.Array,
        opposing: jax.Array,
    ) -> dict[str, jax.Array]:

        # Primary: tolerance reward peaked at force_target (10 N) using the
        # opposing-side pinch force between thumb and index. Fires only when the
        # two fingers contact the cube on opposite sides.
        total_force = jp.minimum(f_thumb, f_index) * opposing
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

        # Dense gradient before the hard ground-termination cliff.
        min_tip_z = jp.min(tips[:2, 2])
        fingertip_height = reward.tolerance(
            min_tip_z, bounds=(0.025, jp.inf), margin=0.025, sigmoid="linear"
        )

        # Reward keeping the palm up; directly counteracts gravity-induced drooping.
        palm_z = self.get_palm_position(data)[2]
        palm_height = reward.tolerance(
            palm_z, bounds=(0.04, jp.inf), margin=0.04, sigmoid="linear"
        )

        return {
            "cube_force": force_reward,
            "fingertip_reach": fingertip_reach,
            "fingertip_height": fingertip_height,
            "palm_height": palm_height,
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "joint_vel": self._cost_joint_vel(data),
            "energy": self._cost_energy(data.qvel[self._hand_dqids], data.actuator_force),
            "termination": done,
        }

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
    from mujoco_playground._src.manipulation.tesollo_hand.pinch import CubePinchProprio
    mj_model = CubePinchProprio().mj_model
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, _ACTIVE_JOINT_NAMES)
    n_hand = len(hand_qids)
    hand_body_names = [
        "rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4",
        "rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4",
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
            + jax.random.uniform(key, shape=(n_hand,), minval=-0.05, maxval=0.05)
        )

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(n_hand,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(n_hand,), minval=1.0, maxval=1.05
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
            key, (n_hand,), minval=0.8, maxval=1.2
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
