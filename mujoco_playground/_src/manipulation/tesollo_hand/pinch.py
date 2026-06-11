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
"""Static-grasp pinch environment for the Tesollo hand (thumb + index only)."""

__all__ = [
    "CubePinch",
    "default_config",
    "domain_randomize",
]

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_grasp as tesollo_hand_base
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_pinch_constants as consts,
)
from mujoco_playground._src.manipulation.tesollo_hand.scene_builders import (
    pinch_scene_reduced,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=400,
        # "baseline" | "proprio" | "force"
        sensor_bundle="proprio",
        # Target total contact force from hand on cube sides, in Newtons.
        force_target=10.0,
        # Range [min, max] for randomizing force_target on each reset.
        force_target_range=[10.0, 10.0],
        # Force must stay within ±force_tolerance N of target for success_hold_time seconds.
        force_tolerance=2.0,
        success_hold_time=1.0,
        obs_noise=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.001,
                joint_vel=0.01,
                motor_targets=0.0,
                force=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                cube_force=5.0,
                fingertip_reach=2.0,
                fingertip_pos_per_tip=2.0,
                pinch_alignment=2.0,
                action_rate=-0.005,
                joint_vel=-0.01,
                energy=-1e-3,
                termination=-100.0,
            ),
            shaping_scale=9.0,
            shaping_floor=0.5,
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


class CubePinch(tesollo_hand_base.TesolloHandGraspEnv):
    """Static-grasp pinch task: squeeze a fixed cube with thumb and index finger.

    The wrist, middle, ring, and pinky are frozen by the scene builder.
    Only the 8 thumb+index DOFs are controlled. The cube has no freejoint.

    Observation layout is controlled by config.sensor_bundle:
      "baseline" → joint_pos(8) + joint_vel(8) + force_target(1) = 17
      "proprio"  → + motor_targets(8)                             = 25
      "force"    → + motor_targets(8) + force(3)                  = 28
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        # Call the MjxEnv grandparent directly so we can build the reduced
        # scene from an XML string rather than a fixed file path.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._model_assets = tesollo_hand_base.get_assets()
        self._mj_spec = pinch_scene_reduced.build_pinch_spec()
        self._mj_model = self._mj_spec.compile()
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._xml_path = consts.SCENE_XML.as_posix()

        if self._config.sensor_bundle not in consts.SENSOR_BUNDLES:
            raise ValueError(
                f"Unknown sensor_bundle {self._config.sensor_bundle!r}. "
                f"Valid: {sorted(consts.SENSOR_BUNDLES)}"
            )
        if self._config.pid_gains.enable:
            self._apply_pid_gains()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos, dtype=float)
        self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1])
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        self._cube_half_size = float(self._mj_model.geom_size[self._cube_geom_id, 0])
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        self._default_pose = self._init_q
        # Cube freejoint removed; get fixed world position via forward kinematics.
        _init_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, _init_data)
        self._init_cube_pos = jp.array(_init_data.xpos[self._cube_body_id])

    def _apply_pid_gains(self) -> None:
        """Override actuator PID gains from config, replacing XML-baked values."""
        cfg = self._config.pid_gains
        nu = self._mj_model.nu

        kp = (
            jp.array(cfg.kp_per_actuator, dtype=float)
            if cfg.kp_per_actuator
            else jp.full((nu,), cfg.finger_kp)
        )
        kv = (
            jp.array(cfg.kv_per_actuator, dtype=float)
            if cfg.kv_per_actuator
            else jp.full((nu,), cfg.finger_kv)
        )
        self._mj_model.actuator_gainprm[:, 0] = np.array(kp)
        self._mj_model.actuator_biasprm[:, 1] = -np.array(kp)
        self._mj_model.actuator_biasprm[:, 2] = -np.array(kv)

    # ------------------------------------------------------------------
    # Properties required by TesolloHandGraspEnv
    # ------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def _fingertip_names(self) -> tuple[str, ...]:
        return tuple(consts.FINGERTIP_NAMES)

    def get_fingertip_global_positions(self, data: mjx.Data) -> jax.Array:
        tip_ids = [
            self._mj_model.site(f"rl_dg_{f}_tip_c").id
            for f in (1, 2)
        ]
        return jp.concatenate([data.site_xpos[sid] for sid in tip_ids])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def obs_size(self) -> int:
        n = consts.N_ACTIVE
        base = n + n + 1  # joint_pos + joint_vel + force_target
        bundle = self._config.sensor_bundle
        if bundle == "proprio":
            return base + n
        if bundle == "force":
            return base + n + 3
        return base  # "baseline"

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        sensor_bundle = self._config.sensor_bundle
        joint_pos = self._obs_joint_pos(data, info)
        joint_vel = self._obs_joint_vel(data, info)
        force_target_obs = self._obs_force_target(data, info)

        if sensor_bundle == "baseline":
            state = jp.concatenate([joint_pos, joint_vel, force_target_obs])
        elif sensor_bundle == "proprio":
            state = jp.concatenate([joint_pos, joint_vel, self._obs_motor_targets(data, info), force_target_obs])
        else:  # "force"
            state = jp.concatenate([joint_pos, joint_vel, self._obs_motor_targets(data, info), self._obs_force(data, info), force_target_obs])

        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

    # ------------------------------------------------------------------
    # Obs component methods (8-DOF pinch-specific, override base 26-DOF)
    # ------------------------------------------------------------------

    def _obs_joint_pos(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        angles = data.qpos[self._hand_qids]
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=angles.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_pos
        return angles + noise

    def _obs_joint_vel(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        vel = data.qvel[self._hand_dqids]
        info["rng"], key = jax.random.split(info["rng"])
        noise = (
            2 * jax.random.uniform(key, shape=vel.shape) - 1
        ) * self._config.obs_noise.level * self._config.obs_noise.scales.joint_vel
        return vel + noise

    def _obs_motor_targets(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return info["motor_targets"]

    def _obs_force(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """[f_thumb, f_index, total_force] / (2 * force_target), shape (3,)."""
        norm = info["force_target"] * 2.0
        f_thumb = self._fingertip_force(data, "rl_dg_1_tip_cube_force")
        f_index = self._fingertip_force(data, "rl_dg_2_tip_cube_force")
        total_force = self._total_contact_force(data)
        return jp.array([f_thumb, f_index, total_force]) / norm

    def _obs_force_target(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Normalized force target command, shape (1,)."""
        return jp.array([info["force_target"] / self._config.force_target])

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Ground-truth privileged critic state (no noise).

        q(8) + qdot(8) + cube_pos(3) + cube_linvel(3) + cube_angvel(3)
        + fingertips_global(6) + motor_targets(8) + force_target(1) = 40
        """
        return jp.concatenate([
            data.qpos[self._hand_qids],
            data.qvel[self._hand_dqids],
            self.get_cube_position(data),
            self.get_cube_linvel(data),
            self.get_cube_angvel(data),
            self.get_fingertip_global_positions(data),
            info["motor_targets"],
            jp.array([info["force_target"] / self._config.force_target]),
        ])

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pos_rng, pert1, pert2, pert3, force_rng = jax.random.split(rng, 6)

        qpos = jp.clip(
            self._default_pose + 0.05 * jax.random.normal(pos_rng, (consts.N_ACTIVE,)),
            self._lowers,
            self._uppers,
        )
        qvel = jp.zeros(self._mj_model.nv)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=qpos,
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

        force_target = jax.random.uniform(
            force_rng,
            minval=self._config.force_target_range[0],
            maxval=self._config.force_target_range[1],
        )

        info = {
            "rng": rng,
            "step": 0,
            "steps_since_last_success": 0,
            "success_count": 0,
            "consecutive_success_steps": jp.zeros(()),
            "last_act": jp.zeros(consts.N_ACTIVE),
            "last_last_act": jp.zeros(consts.N_ACTIVE),
            "motor_targets": data.ctrl,
            "force_target": force_target,
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
        metrics["effective_force"] = jp.zeros(())
        metrics["force_target"] = jp.zeros(())
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
        active_ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)
        motor_targets = (
            self._config.ema_alpha * active_ctrl
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        f_thumb = self._fingertip_force(data, "rl_dg_1_tip_cube_force")
        f_index = self._fingertip_force(data, "rl_dg_2_tip_cube_force")
        contact_gate = jp.clip(jp.minimum(f_thumb, f_index) / 0.5, 0.0, 1.0)
        effective_force = self._total_contact_force(data) * contact_gate
        force_target = state.info["force_target"]
        in_tolerance = (
            (effective_force >= force_target - self._config.force_tolerance)
            & (effective_force <= force_target + self._config.force_tolerance)
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
        rewards = self._get_reward(data, action, state.info, done, effective_force)

        a = self._config.reward_config.shaping_floor
        rf = a + (1.0 - a) * rewards["cube_force"]
        rr = a + (1.0 - a) * rewards["fingertip_reach"] / 2.0
        shaping = self._config.reward_config.shaping_scale * rf * rr

        sc = self._config.reward_config.scales
        penalties = (
            sc.action_rate * rewards["action_rate"]
            + sc.joint_vel * rewards["joint_vel"]
            + sc.energy * rewards["energy"]
            + sc.termination * rewards["termination"]
            + sc.fingertip_pos_per_tip * rewards["fingertip_pos_per_tip"]
        )

        rew = (shaping + penalties) * self.dt
        rew += in_tolerance * self._config.reward_config.success_reward

        state.info["step"] += 1
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.metrics["reward/success"] = success.astype(float)
        state.metrics["reward/cube_force"] = rewards["cube_force"]
        state.metrics["reward/fingertip_reach"] = rewards["fingertip_reach"]
        state.metrics["reward/fingertip_pos_per_tip"] = sc.fingertip_pos_per_tip * rewards["fingertip_pos_per_tip"]
        state.metrics["reward/pinch_alignment"] = rewards["pinch_alignment"]
        state.metrics["reward/action_rate"] = sc.action_rate * rewards["action_rate"]
        state.metrics["reward/joint_vel"] = sc.joint_vel * rewards["joint_vel"]
        state.metrics["reward/energy"] = sc.energy * rewards["energy"]
        state.metrics["reward/termination"] = sc.termination * rewards["termination"]
        state.metrics["f_thumb"] = f_thumb
        state.metrics["f_index"] = f_index
        state.metrics["effective_force"] = effective_force
        state.metrics["force_target"] = state.info["force_target"]

        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(
        self, data: mjx.Data
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cube_xy = self.get_cube_position(data)[:2]
        drift = jp.linalg.norm(cube_xy - self._init_cube_pos[:2]) > 0.15
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        tip_on_ground = jp.any(tips[:, 2] < 0.005)
        reasons = {"drift": drift, "nan": nans, "tip_on_ground": tip_on_ground}
        return drift | nans | tip_on_ground, reasons

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        effective_force: jax.Array,
    ) -> dict[str, jax.Array]:
        force_target = info["force_target"]
        force_reward = reward.tolerance(
            effective_force,
            bounds=(force_target, force_target),
            margin=force_target,
            sigmoid="gaussian",
        )

        cube_pos = self.get_cube_position(data)
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        reach_dists = jp.linalg.norm(tips - cube_pos, axis=1)
        fingertip_reach = jp.sum(
            reward.tolerance(reach_dists, bounds=(0, 0.035), margin=0.1, sigmoid="reciprocal")
        )
        fingertip_pos_per_tip = jp.sum(
            reward.tolerance(reach_dists, bounds=(0, self._cube_half_size), margin=0.1, sigmoid="reciprocal")
        )

        thumb_dist = jp.linalg.norm(tips[0] - cube_pos)
        index_dist = jp.linalg.norm(tips[1] - cube_pos)
        pinch_alignment = jp.sum(
            reward.tolerance(
                jp.array([thumb_dist, index_dist]),
                bounds=(0, 0.01),
                margin=0.08,
                sigmoid="reciprocal",
            )
        )

        return {
            "cube_force": force_reward,
            "fingertip_reach": fingertip_reach,
            "fingertip_pos_per_tip": fingertip_pos_per_tip,
            "pinch_alignment": pinch_alignment,
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "joint_vel": self._cost_joint_vel(data),
            "energy": self._cost_energy(data.qvel[self._hand_dqids], data.actuator_force),
            "termination": done,
        }

    def _total_contact_force(self, data: mjx.Data) -> jax.Array:
        forces = mjx_env.get_sensor_data(self.mj_model, data, "cube_force").reshape(-1, 3)
        return jp.sum(jp.linalg.norm(forces, axis=1))

    def _fingertip_force(self, data: mjx.Data, sensor_name: str) -> jax.Array:
        f = mjx_env.get_sensor_data(self.mj_model, data, sensor_name).reshape(-1, 3)
        return jp.sum(jp.linalg.norm(f, axis=1))

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
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4",
        "rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    silicone_geom_ids = [
        mj_model.geom(g).id for g in ["rl_dg_1_tip", "rl_dg_2_tip"]
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
            + jax.random.uniform(key, shape=(consts.N_ACTIVE,), minval=-0.05, maxval=0.05)
        )

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(consts.N_ACTIVE,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(consts.N_ACTIVE,), minval=1.0, maxval=1.05
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
            key, (consts.N_ACTIVE,), minval=0.8, maxval=1.2
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
