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

import logging
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
from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module
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
        weld_cube=True,
        domain_rand=config_dict.create(
            cube_size=[0.85, 1.15],
            cube_pos=[0.0, 0.0],
            cube_mass=[1.0, 1.0],
            actuator_kp=[1.0, 1.0],
        ),
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=80,
        sensor_bundle="proprio.target",
        force_target=config_dict.create(
            range=[2.0, 5.0],
            sinusoid=False,
            frequency=0.5,
            phase=[0.0, 2.0 * np.pi],
            amplitude_scale=1.0,
            frequency_log2_range=[0.0, 0.0],
            amplitude_scale_range=[0.0, 0.0],
        ),
        force_reward_margin=4.5,  # width of the force reward (N)
        force_reward_sigmoid="reciprocal",  # r = 1/(1 + 2|force_error|)
        force_reward_contact_gated=True,
        obs_noise=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.001,
                joint_vel=0.01,
                motor_targets=0.0,
                fingertip_forces=0.0,
                fingertip_force_dirs=0.0,
            ),
            bias_scales=config_dict.create(
                joint_pos=0.0,
                joint_vel=0.0,
                motor_targets=0.0,
                fingertip_forces=0.0,
                fingertip_force_dirs=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Two normalized [0, 1] reward components ...
                cube_force=5.0,            # contact force matching the target
                fingertip_pos_per_tip=2.0,  # fingertips reaching the cube centre
                # ... plus smoothness penalties.
                action_rate=-0.005,
                joint_vel=-0.01,
            ),
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
      "baseline"                       → joint_pos(8) + joint_vel(8) + target_force(1) = 17
      "proprio.target"                 → + motor_targets(8)                             = 25
      "proprio.target+force.magnitude" → + motor_targets(8) + fingertip_forces(4)      = 29
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
        self._mj_spec = pinch_scene_reduced.build_pinch_spec(weld_cube=self._config.weld_cube)
        self._mj_model = self._mj_spec.compile()
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._xml_path = consts.SCENE_XML.as_posix()

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
        self._default_pose = self._init_q[self._hand_qids]
        # Cube freejoint removed; get fixed world position via forward kinematics.
        _init_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, _init_data)
        self._init_cube_pos = jp.array(_init_data.xpos[self._cube_body_id])
        self._obs_components = self._build_obs_components()
        obs_module.validate_spec(
            self._config.sensor_bundle,
            self._task_obs_keys(),
            self._obs_components,
            self._config.obs_noise.scales,
        )

    def _apply_pid_gains(self) -> None:
        """Override actuator PID gains from config, replacing XML-baked values."""
        cfg = self._config.pid_gains
        nu = self._mj_model.nu
        # Writes every actuator, so only valid in the finger-only scene.
        assert nu == consts.N_ACTIVE, (
            f"pid_gains overrides all {nu} actuators; expected the reduced"
            f" finger-only scene with {consts.N_ACTIVE}"
        )
        kp = np.full((nu,), cfg.finger_kp)
        kv = np.full((nu,), cfg.finger_kv)
        self._mj_model.actuator_gainprm[:, 0] = kp
        self._mj_model.actuator_biasprm[:, 1] = -kp
        self._mj_model.actuator_biasprm[:, 2] = -kv

    def domain_randomizer(self, model: mjx.Model, rng: jax.Array):
        """Per-env domain randomization, closing over THIS env's
        config.domain_rand. train_jax_ppo prefers this over the registry's
        module-level entry point."""
        spec = {k: list(v) for k, v in self._config.domain_rand.items()}
        return _domain_randomize_impl(model, rng, spec)

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

    def sync_mj_model_meshes(self) -> None:
        """Recompile the spec so mj_model meshes match the current mjx_model's
        cube geom sizes.  Call before render() when domain randomization has
        changed geom_size on the MJX model."""
        mjx_geom_size = np.array(self._mjx_model.geom_size)
        mjx_body_pos = np.array(self._mjx_model.body_pos)
        cube_bid = self._mj_model.body("cube").id
        if (np.allclose(self._mj_model.geom_size, mjx_geom_size)
                and np.allclose(self._mj_model.body_pos, mjx_body_pos)):
            return
        for body in self._mj_spec.bodies:
            if body.name != "cube":
                continue
            body.pos = mjx_body_pos[cube_bid]
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_BOX and geom.name:
                    geom.size = mjx_geom_size[self._mj_model.geom(geom.name).id]
                elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_gid = next(
                        g for g in range(self._mj_model.ngeom)
                        if self._mj_model.geom_bodyid[g] == cube_bid
                        and self._mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    )
                    for mesh in self._mj_spec.meshes:
                        if mesh.name == "cube_mesh":
                            mesh.scale = mjx_geom_size[mesh_gid]
                            break
            break
        m_new = self._mj_spec.compile()
        self._mj_model.geom_size[:] = m_new.geom_size
        self._mj_model.body_pos[:] = m_new.body_pos
        self._mj_model.body_mass[:] = m_new.body_mass
        self._mj_model.body_inertia[:] = m_new.body_inertia
        self._mj_model.mesh_vert[:] = m_new.mesh_vert
        self._mj_model.mesh_normal[:] = m_new.mesh_normal

    def render(self, trajectory, height=240, width=320, camera=None,
               scene_option=None, modify_scene_fns=None):
        self.sync_mj_model_meshes()
        return mjx_env.render_array(
            self._mj_model, trajectory, height, width, camera,
            scene_option=scene_option, modify_scene_fns=modify_scene_fns,
        )

    _TASK_KEYS: tuple[str, ...] = ("target_force",)

    _FINGER_FORCE_SENSORS: tuple[tuple[str, ...], ...] = (
        ("rl_dg_1_tip_cube_force", "rl_dg_1_tip_2_cube_force"),  # thumb
        ("rl_dg_2_tip_cube_force", "rl_dg_2_tip_2_cube_force"),  # index
    )
    # One channel per tip geom, so len() is the fingertip_forces obs size (4).
    # The box-pad (tip_2) geoms sit recessed and read ~0 in a pinch.
    _TIP_FORCE_SENSORS: list[str] = [s for g in _FINGER_FORCE_SENSORS for s in g]

    def _task_obs_keys(self) -> tuple[str, ...]:
        return self._TASK_KEYS

    def _build_obs_components(self) -> dict:
        c = super()._build_obs_components()
        c.update({
            "target_force": obs_module.ObsComponent(
                "target_force", self._obs_target_force,
                size=1,
                description="randomised force target, raw Newtons (PPO normalizes obs)",
                labels=("target_force",),
            ),
        })
        return c

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

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        state = self._build_obs(
            self._config.sensor_bundle, self._task_obs_keys(), data, info
        )
        return {"state": state, "privileged_state": self._obs_privileged(data, info)}

    # ------------------------------------------------------------------
    # Pinch-specific obs component methods
    # ------------------------------------------------------------------

    def _obs_target_force(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Raw force target command in Newtons, shape (1,). Standardised by the
        PPO running-observation normalizer, so no manual scaling is applied."""
        return jp.array([info["force_target"]])

    def _force_target_at(
        self,
        step: jax.Array,
        phase: jax.Array,
        frequency: jax.Array,
        amplitude_scale: jax.Array,
    ) -> jax.Array:
        """Sinusoidal force target (N) at a given step, centred in force_target.range.

        target(t) = mid + amp * sin(2π f t + phase), with t = step * ctrl_dt.
        The sine is always centred on the midpoint of force_target.range;
        `amplitude_scale` sets what fraction of the range it spans, so scale=1
        touches both ends and scale=0 is a constant mid. `phase`, `frequency`
        and `amplitude_scale` are per-episode values carried in info (see reset),
        which prevents the policy from memorizing a fixed step→target schedule.
        """
        lo, hi = self._config.force_target.range
        mid = 0.5 * (lo + hi)
        amp = 0.5 * (hi - lo) * amplitude_scale
        omega = 2.0 * jp.pi * frequency
        return mid + amp * jp.sin(omega * step * self._config.ctrl_dt + phase)


    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Ground-truth privileged critic state (no noise). The cube is static
        (no freejoint), but its per-env DR'd half-size and world pose/orientation
        are exposed so the critic sees the unobserved cube-size latent directly.

        Under the DR vmap wrapper ``self._mjx_model`` is the per-env randomized
        model at step time, so ``geom_size`` reflects this env's cube size.

        q(8) + qdot(8) + fingertips_global(6) + motor_targets(8)
        + force_target(1) + fingertip_forces(4)
        + cube_size(1) + cube_pos(3) + cube_quat(4) = 43
        """
        cube_size = self._mjx_model.geom_size[self._cube_geom_id, 0]
        return jp.concatenate([
            data.qpos[self._hand_qids],
            data.qvel[self._hand_dqids],
            self.get_fingertip_global_positions(data),
            info["motor_targets"],
            jp.array([info["force_target"]]),
            self._obs_fingertip_forces(data, info),
            jp.array([cube_size]),
            self.get_cube_position(data),
            self.get_cube_orientation(data),
        ])

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pert1, pert2, pert3, force_rng, bias_rng = jax.random.split(rng, 6)

        hand_q = jp.clip(
            self._default_pose,
            self._lowers,
            self._uppers,
        )
        qpos = self._init_q.at[self._hand_qids].set(hand_q)
        qvel = jp.zeros(self._mj_model.nv)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=hand_q,
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
        # Sub-split so lin/ang magnitudes are independent (sharing pert3 made
        # them affinely correlated). Derived from pert3 only, so the other
        # reset keys (force target, obs bias) are unchanged.
        pert_lin_rng, pert_ang_rng = jax.random.split(pert3)
        pert_lin = jax.random.uniform(
            pert_lin_rng,
            minval=self._config.pert_config.linear_velocity_pert[0],
            maxval=self._config.pert_config.linear_velocity_pert[1],
        )
        pert_ang = jax.random.uniform(
            pert_ang_rng,
            minval=self._config.pert_config.angular_velocity_pert[0],
            maxval=self._config.pert_config.angular_velocity_pert[1],
        )

        ft_cfg = self._config.force_target
        force_rng, phase_rng = jax.random.split(force_rng)
        # Per-episode sinusoid shape. Whether a range is active is decided from
        # the static config at trace time, so degenerate ranges draw no RNG.
        freq_lo, freq_hi = ft_cfg.frequency_log2_range
        if freq_hi > freq_lo:
            phase_rng, freq_rng = jax.random.split(phase_rng)
            force_frequency = jp.exp2(
                jax.random.uniform(freq_rng, minval=freq_lo, maxval=freq_hi)
            )
        else:
            force_frequency = jp.asarray(ft_cfg.frequency)

        amp_lo, amp_hi = ft_cfg.amplitude_scale_range
        if amp_hi > amp_lo:
            phase_rng, amp_rng = jax.random.split(phase_rng)
            force_amplitude_scale = jax.random.uniform(
                amp_rng, minval=amp_lo, maxval=amp_hi
            )
        else:
            force_amplitude_scale = jp.asarray(ft_cfg.amplitude_scale)

        force_phase = jp.where(
            ft_cfg.sinusoid,
            jax.random.uniform(
                phase_rng,
                minval=ft_cfg.phase[0],
                maxval=ft_cfg.phase[1],
            ),
            0.0,
        )
        force_target = jp.where(
            ft_cfg.sinusoid,
            self._force_target_at(
                0, force_phase, force_frequency, force_amplitude_scale
            ),
            jax.random.uniform(
                force_rng,
                minval=ft_cfg.range[0],
                maxval=ft_cfg.range[1],
            ),
        )

        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(consts.N_ACTIVE),
            "last_last_act": jp.zeros(consts.N_ACTIVE),
            "motor_targets": data.ctrl,
            "force_target": force_target,
            "force_phase": force_phase,
            "force_frequency": force_frequency,
            "force_amplitude_scale": force_amplitude_scale,
            "force_error": jp.zeros(()),
            "pert_wait_steps": pert_wait_steps,
            "pert_duration_steps": pert_duration_steps,
            "pert_vel": jp.array([pert_lin] * 3 + [pert_ang] * 3),
            "pert_dir": jp.zeros(6, dtype=float),
            "last_pert_step": jp.array([-jp.inf], dtype=float),
            "obs_bias": self._sample_obs_bias(
                bias_rng, self._config.sensor_bundle, self._task_obs_keys()
            ),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}_per_step"] = jp.zeros(())
        metrics["f_thumb"] = jp.zeros(())
        metrics["f_index"] = jp.zeros(())
        metrics["effective_force"] = jp.zeros(())
        metrics["force_target"] = jp.zeros(())
        metrics["force_error_abs_per_step"] = jp.zeros(())
        metrics["termination/drift"] = jp.zeros(())
        metrics["termination/nan"] = jp.zeros(())
        metrics["termination/tip_on_ground"] = jp.zeros(())

        obs = self._get_obs(data, info)
        rew, done = jp.zeros(2)
        return mjx_env.State(data, obs, rew, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        # Absolute joint-position targets: action in [-1, 1] maps linearly onto the
        # full actuator ctrl range, so the policy can command any reachable target.
        active_ctrl = self._lowers + 0.5 * (action + 1.0) * (self._uppers - self._lowers)
        active_ctrl = jp.clip(active_ctrl, self._lowers, self._uppers)
        motor_targets = (
            self._config.ema_alpha * active_ctrl
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        f_thumb = self._finger_contact_force(data, self._FINGER_FORCE_SENSORS[0])
        f_index = self._finger_contact_force(data, self._FINGER_FORCE_SENSORS[1])
        contact_gate = jp.clip(jp.minimum(f_thumb, f_index) / 0.5, 0.0, 1.0)
        effective_force = self._total_contact_force(data) * contact_gate
        force_target = state.info["force_target"]
        state.info["force_error"] = effective_force - force_target

        done, term_reasons = self._get_termination(data)
        state.metrics["termination/drift"] = term_reasons["drift"].astype(float)
        state.metrics["termination/nan"] = term_reasons["nan"].astype(float)
        state.metrics["termination/tip_on_ground"] = term_reasons["tip_on_ground"].astype(float)

        raw_rewards = self._get_reward(
            data, action, state.info, effective_force, contact_gate
        )
        scaled_rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()
        }

        rew = sum(scaled_rewards.values()) * self.dt

        state.info["step"] += 1
        if self._config.force_target.sinusoid:
            state.info["force_target"] = self._force_target_at(
                state.info["step"],
                state.info["force_phase"],
                state.info["force_frequency"],
                state.info["force_amplitude_scale"],
            )
        obs = self._get_obs(data, state.info)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for k, v in raw_rewards.items():
            state.metrics[f"reward/{k}_per_step"] = v
        state.metrics["f_thumb"] = f_thumb
        state.metrics["f_index"] = f_index
        state.metrics["effective_force"] = effective_force
        # Log the target this step's force was scored against (the pre-advance
        # value), so effective_force and force_target line up in the dashboard.
        state.metrics["force_target"] = force_target
        # `_per_step` suffix: brax divides these by the episode length.
        state.metrics["force_error_abs_per_step"] = jp.abs(state.info["force_error"])

        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(
        self, data: mjx.Data
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cube_pos = self.get_cube_position(data)
        # Drift is measured against this env's own body position, so a cube_pos
        # DR offset does not eat into the budget and terminate immediately.
        if self._config.weld_cube:
            ref_xy = self.mjx_model.body_pos[self._cube_body_id, :2]
        else:
            ref_xy = self._init_cube_pos[:2]
        drift = jp.linalg.norm(cube_pos[:2] - ref_xy) > 0.15
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
        effective_force: jax.Array,
        contact_gate: jax.Array,
    ) -> dict[str, jax.Array]:
        # Contact force matching the (randomized) target, normalized to [0, 1].
        # tolerance() rejects traced bounds under jit, so the error is divided
        # by the margin before the call and bounds/margin stay constant.
        force_target = info["force_target"]
        force_error = (effective_force - force_target) / self._config.force_reward_margin
        force_reward = reward.tolerance(
            force_error,
            bounds=(0.0, 0.0),
            margin=1.0,
            sigmoid=self._config.force_reward_sigmoid,
        )
        if self._config.force_reward_contact_gated:
            # Without this the no-contact state is scored as a force error, which
            # still pays out when the target sits at the bottom of its range.
            force_reward = force_reward * contact_gate

        # Fingertips reaching the cube centre, normalized to [0, 1]: the mean
        # over tips of a per-tip closeness reward.
        cube_pos = self.get_cube_position(data)
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        reach_dists = jp.linalg.norm(tips - cube_pos, axis=1)
        # Distance beyond this env's DR'd cube surface: tolerance() rejects
        # traced bounds, and for dist >= 0 this equals bounds=(0, half).
        half = self.mjx_model.geom_size[self._cube_geom_id, 0]
        per_tip = reward.tolerance(
            jp.maximum(reach_dists - half, 0.0),
            bounds=(0.0, 0.0), margin=0.1, sigmoid="reciprocal",
        )
        fingertip_pos_per_tip = jp.mean(per_tip)

        return {
            "cube_force": force_reward,
            "fingertip_pos_per_tip": fingertip_pos_per_tip,
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "joint_vel": self._cost_joint_vel(data),
        }

    def _total_contact_force(self, data: mjx.Data) -> jax.Array:
        """Total cube contact-force magnitude (N), summed over both fingers'
        tip-sphere and box-pad sensors. Replaces the whole-hand `cube_force`
        subtree sensor so the rewarded force equals the sum of the per-finger
        forces the policy/critic observe."""
        return sum(
            self._finger_contact_force(data, group)
            for group in self._FINGER_FORCE_SENSORS
        )

    def _finger_contact_force(
        self, data: mjx.Data, sensor_names: tuple[str, ...]
    ) -> jax.Array:
        """Contact-force magnitude (N) a single finger exerts on the cube, summed
        over its tip-sphere and box-pad contact sensors."""
        return sum(
            jp.sum(jp.linalg.norm(
                mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
                axis=1,
            ))
            for name in sensor_names
        )

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


def _get_scene_ids():
    """Lightweight ID lookup from a throwaway scene compile (no CubePinch)."""
    mj_model = pinch_scene_reduced.build_pinch_spec(weld_cube=True).compile()
    cube_bid = mj_model.body("cube").id
    cube_gids = np.array(
        [g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == cube_bid]
    )
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4",
        "rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    silicone_geom_ids = [
        mj_model.geom(g).id for g in ["rl_dg_1_tip", "rl_dg_2_tip"]
    ]
    return cube_bid, cube_gids, hand_qids, hand_body_ids, silicone_geom_ids


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Registry entry point: randomize with the ranges from default_config().

    Same implementation as CubePinch.domain_randomizer, which is what callers
    holding an env should use: the ranges live in config.domain_rand, so only
    the instance knows whether they were overridden. This module-level form has
    no instance to ask and therefore states the defaults.
    """
    return _domain_randomize_impl(model, rng, _default_dr_spec())


def _default_dr_spec() -> dict[str, list[float]]:
    return {k: list(v) for k, v in default_config().domain_rand.items()}


def _domain_randomize_impl(model: mjx.Model, rng: jax.Array, spec):
    """Per-env domain randomization for hand dynamics and (optionally) cube
    geometry.  Cube size/pos/mass branches are controlled by `spec` (populated
    from config.domain_rand); they trace away when lo==hi so there is no
    runtime cost when disabled.
    """
    cube_bid, cube_gids, hand_qids, hand_body_ids, silicone_geom_ids = (
        _get_scene_ids()
    )

    (size_lo, size_hi) = spec["cube_size"]
    (pos_lo, pos_hi) = spec["cube_pos"]
    (mass_lo, mass_hi) = spec["cube_mass"]
    (kp_lo, kp_hi) = spec["actuator_kp"]
    do_size = size_lo != size_hi
    do_pos = pos_lo != pos_hi
    do_mass = mass_lo != mass_hi
    do_kp = kp_lo != kp_hi

    _log = logging.getLogger(__name__)
    randomized_keys: dict[str, str] = {}

    @jax.vmap
    def rand(rng):
        geom_size = model.geom_size
        body_pos = model.body_pos
        body_mass = model.body_mass
        body_inertia = model.body_inertia
        actuator_gainprm = model.actuator_gainprm
        actuator_biasprm = model.actuator_biasprm
        if do_kp:
            rng, k = jax.random.split(rng)
            s = jax.random.uniform(k, (), minval=kp_lo, maxval=kp_hi)
            kp = model.actuator_gainprm[:, 0] * s
            actuator_gainprm = actuator_gainprm.at[:, 0].set(kp)
            actuator_biasprm = actuator_biasprm.at[:, 1].set(-kp)
        if do_size:
            rng, k = jax.random.split(rng)
            s = jax.random.uniform(k, (), minval=size_lo, maxval=size_hi)
            geom_size = geom_size.at[cube_gids].set(geom_size[cube_gids] * s)
            body_pos = body_pos.at[cube_bid, 2].set(geom_size[cube_gids[0], 2])
        if do_pos:
            rng, k = jax.random.split(rng)
            dxy = jax.random.uniform(k, (2,), minval=pos_lo, maxval=pos_hi)
            body_pos = body_pos.at[cube_bid, :2].set(
                model.body_pos[cube_bid, :2] + dxy
            )
        if do_mass:
            rng, k = jax.random.split(rng)
            m = jax.random.uniform(k, (), minval=mass_lo, maxval=mass_hi)
            body_mass = body_mass.at[cube_bid].set(model.body_mass[cube_bid] * m)
            body_inertia = body_inertia.at[cube_bid].set(
                model.body_inertia[cube_bid] * m
            )
        return (
            geom_size, body_pos, body_mass, body_inertia,
            actuator_gainprm, actuator_biasprm,
        )

    if do_size:
        randomized_keys["geom_size"] = f"[{size_lo}, {size_hi}]"
    if do_pos:
        randomized_keys["body_pos"] = f"[{pos_lo}, {pos_hi}]"
    if do_mass:
        randomized_keys["body_mass"] = f"[{mass_lo}, {mass_hi}]"
        randomized_keys["body_inertia"] = f"[{mass_lo}, {mass_hi}]"
    if do_kp:
        randomized_keys["actuator_gainprm"] = f"[{kp_lo}, {kp_hi}]"
        randomized_keys["actuator_biasprm"] = f"[{kp_lo}, {kp_hi}]"

    if randomized_keys:
        _log.debug(
            "domain_randomize: %s",
            ", ".join(f"{k}={v}" for k, v in randomized_keys.items()),
        )
    else:
        _log.debug("domain_randomize: no keys randomized (all ranges degenerate)")

    (
        geom_size, body_pos, body_mass, body_inertia,
        actuator_gainprm, actuator_biasprm,
    ) = rand(rng)

    replace_dict = {}
    axes_dict = {}
    if do_size:
        replace_dict["geom_size"] = geom_size
        axes_dict["geom_size"] = 0
    if do_pos or do_size:
        replace_dict["body_pos"] = body_pos
        axes_dict["body_pos"] = 0
    if do_mass:
        replace_dict["body_mass"] = body_mass
        replace_dict["body_inertia"] = body_inertia
        axes_dict["body_mass"] = 0
        axes_dict["body_inertia"] = 0
    if do_kp:
        replace_dict["actuator_gainprm"] = actuator_gainprm
        replace_dict["actuator_biasprm"] = actuator_biasprm
        axes_dict["actuator_gainprm"] = 0
        axes_dict["actuator_biasprm"] = 0

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    if replace_dict:
        in_axes = in_axes.tree_replace(axes_dict)
        model = model.tree_replace(replace_dict)
    return model, in_axes
