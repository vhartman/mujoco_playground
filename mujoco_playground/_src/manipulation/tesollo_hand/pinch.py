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
        action_scale=0.5,
        action_mode="delta",
        ghost_cube=False,
        cube_size_scale=1.0,
        cube_pos_offset=[0.0, 0.0],
        weld_cube=True,
        domain_rand=config_dict.create(
            cube_size=[0.85, 1.15],
            cube_pos=[0.0, 0.0],
            cube_mass=[1.0, 1.0],
        ),
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=80,
        # "baseline" | "proprio.target" | "proprio.target+force.magnitude"
        sensor_bundle="proprio.target",
        force_target_range=[2.0, 5.0],
        force_tolerance=0.75,
        success_hold_time=0.5,
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
      "baseline"                       → joint_pos(8) + joint_vel(8) + target_force(1) = 17
      "proprio.target"                 → + motor_targets(8)                             = 25
      "proprio.target+force.magnitude" → + motor_targets(8) + fingertip_forces(2)      = 27
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
        _cube_dr_spec.update({k: list(v) for k, v in self._config.domain_rand.items()})
        self._model_assets = tesollo_hand_base.get_assets()
        self._mj_spec = pinch_scene_reduced.build_pinch_spec(weld_cube=self._config.weld_cube)
        self._mj_model = self._mj_spec.compile()
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._xml_path = consts.SCENE_XML.as_posix()

        if self._config.ghost_cube:
            _ghost_alpha = 0.32
            _cube_bid = self._mj_model.body("cube").id
            for _g in range(self._mj_model.ngeom):
                if self._mj_model.geom_bodyid[_g] != _cube_bid:
                    continue
                self._mj_model.geom_contype[_g] = 0
                self._mj_model.geom_conaffinity[_g] = 0
                self._mj_model.geom_rgba[_g, 3] = _ghost_alpha
                _mid = self._mj_model.geom_matid[_g]
                if _mid >= 0:
                    self._mj_model.mat_rgba[_mid, 3] = _ghost_alpha

        if self._config.cube_size_scale != 1.0:
            _scale = float(self._config.cube_size_scale)
            _cube_bid = self._mj_model.body("cube").id
            for _g in range(self._mj_model.ngeom):
                if self._mj_model.geom_bodyid[_g] == _cube_bid:
                    self._mj_model.geom_size[_g] *= _scale
            self._mj_model.body_pos[_cube_bid, 2] = self._mj_model.geom_size[
                self._mj_model.geom("cube").id, 2
            ]

        if any(self._config.cube_pos_offset):
            _cb = self._mj_model.body("cube").id
            self._mj_model.body_pos[_cb, :2] += np.array(
                self._config.cube_pos_offset, dtype=float
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
    # One entry per finger; its length sets the fingertip_forces obs size (2).
    _TIP_FORCE_SENSORS: list[str] = [
        "rl_dg_1_tip_cube_force",
        "rl_dg_2_tip_cube_force",
    ]
    _TIP_FORCE_SCALE: float = 10.0

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

    def _obs_fingertip_forces(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Per-finger contact-force magnitude (thumb, index), each summed over the
        finger's tip-sphere and box-pad sensors, scaled by 1/_TIP_FORCE_SCALE.

        Overrides the base per-sensor version so the oracle force observation
        includes the pad and matches the rewarded `effective_force`."""
        forces = jp.array([
            self._finger_contact_force(data, group)
            for group in self._FINGER_FORCE_SENSORS
        ])
        return forces / self._TIP_FORCE_SCALE

    def _obs_fingertip_force_dirs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Per-finger normalized net contact-force direction, summing the tip
        sphere and box pad of each finger before normalizing."""
        dirs = []
        for group in self._FINGER_FORCE_SENSORS:
            net = jp.sum(
                jp.stack([
                    jp.sum(
                        mjx_env.get_sensor_data(self.mj_model, data, name).reshape(-1, 3),
                        axis=0,
                    )
                    for name in group
                ]),
                axis=0,
            )
            magnitude = jp.linalg.norm(net)
            dirs.append(jp.where(magnitude > 1e-3, net / magnitude, jp.zeros(3)))
        return jp.concatenate(dirs)

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Ground-truth privileged critic state (no noise). The cube is static
        (no freejoint), so no cube pose/velocity terms are included.

        q(8) + qdot(8) + fingertips_global(6) + motor_targets(8)
        + force_target(1) + fingertip_forces(2) = 33
        """
        return jp.concatenate([
            data.qpos[self._hand_qids],
            data.qvel[self._hand_dqids],
            self.get_fingertip_global_positions(data),
            info["motor_targets"],
            jp.array([info["force_target"]]),
            self._obs_fingertip_forces(data, info),
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
        metrics["reward/success_per_step"] = jp.zeros((), dtype=float)
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

        if self._config.action_mode == "delta":
            active_ctrl = state.data.ctrl + action * self._config.action_scale
        elif self._config.action_mode == "delta_pose":
            active_ctrl = state.data.qpos[self._hand_qids] + action * self._config.action_scale
        elif self._config.action_mode == "absolute":
            active_ctrl = self._lowers + 0.5 * (action + 1.0) * (self._uppers - self._lowers)
        else:
            raise ValueError(f"unknown action_mode: {self._config.action_mode!r}")
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
        raw_rewards = self._get_reward(data, action, state.info, effective_force)
        scaled_rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()
        }

        # Weighted sum of the two normalized reward components plus penalties,
        # plus a per-step success bonus paid on every step once the force has been
        # held within tolerance for the required hold time (`success`). Both parts
        # are scaled by dt so the bonus stays commensurate with the shaped terms
        # regardless of control rate.
        rew = (
            sum(scaled_rewards.values())
            + success * self._config.reward_config.success_reward
        ) * self.dt

        state.info["step"] += 1
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        # Log the unscaled, per-step raw reward components (each in [0, 1] for the
        # normalized terms) so the dashboard reads the actual component value
        # rather than a scale- and dt-weighted quantity.
        state.metrics["reward/success_per_step"] = success.astype(float)
        for k, v in raw_rewards.items():
            state.metrics[f"reward/{k}_per_step"] = v
        state.metrics["f_thumb"] = f_thumb
        state.metrics["f_index"] = f_index
        state.metrics["effective_force"] = effective_force
        state.metrics["force_target"] = state.info["force_target"]

        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(
        self, data: mjx.Data
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cube_pos = self.get_cube_position(data)
        drift = jp.linalg.norm(cube_pos[:2] - self._init_cube_pos[:2]) > 0.15
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
    ) -> dict[str, jax.Array]:
        # Contact force matching the (randomized) target, normalized to [0, 1].
        # Use the relative force error so tolerance() gets constant bounds/margin
        # (passing the traced target as bounds breaks under jit). With error e =
        # (f - target) / target this is identical to a Gaussian tolerance on f
        # with bounds=(target, target), margin=target.
        force_target = info["force_target"]
        force_error = (effective_force - force_target) / force_target
        force_reward = reward.tolerance(
            force_error, bounds=(0.0, 0.0), margin=1.0, sigmoid="gaussian"
        )

        # Fingertips reaching the cube centre, normalized to [0, 1]: the mean
        # over tips of a per-tip closeness reward.
        cube_pos = self.get_cube_position(data)
        tips = self.get_fingertip_global_positions(data).reshape(-1, 3)
        reach_dists = jp.linalg.norm(tips - cube_pos, axis=1)
        fingertip_pos_per_tip = jp.mean(
            reward.tolerance(reach_dists, bounds=(0, self._cube_half_size), margin=0.1, sigmoid="reciprocal")
        )

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


_cube_dr_spec: dict[str, list[float]] = {
    "cube_size": [1.0, 1.0],
    "cube_pos": [0.0, 0.0],
    "cube_mass": [1.0, 1.0],
}


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
    """Per-env domain randomization for hand dynamics and (optionally) cube
    geometry.  Cube size/pos/mass branches are controlled by _cube_dr_spec
    (populated from config.domain_rand at env construction time); they trace
    away when lo==hi so there is no runtime cost when disabled.
    """
    cube_bid, cube_gids, hand_qids, hand_body_ids, silicone_geom_ids = (
        _get_scene_ids()
    )

    spec = _cube_dr_spec
    (size_lo, size_hi) = spec["cube_size"]
    (pos_lo, pos_hi) = spec["cube_pos"]
    (mass_lo, mass_hi) = spec["cube_mass"]
    do_size = size_lo != size_hi
    do_pos = pos_lo != pos_hi
    do_mass = mass_lo != mass_hi

    _log = logging.getLogger(__name__)
    randomized_keys: dict[str, str] = {}

    # @jax.vmap
    # def rand_hand(rng):
    #     rng, key = jax.random.split(rng)
    #     silicone_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=2.0)
    #     geom_friction = model.geom_friction.at[silicone_geom_ids, 0].set(silicone_friction)
    #
    #     rng, key1, key2 = jax.random.split(rng, 3)
    #     dmass = jax.random.uniform(key1, minval=0.5, maxval=1.5)
    #     body_inertia = model.body_inertia.at[cube_bid].set(
    #         model.body_inertia[cube_bid] * dmass
    #     )
    #     dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    #     body_ipos = model.body_ipos.at[cube_bid].set(
    #         model.body_ipos[cube_bid] + dpos
    #     )
    #
    #     rng, key = jax.random.split(rng)
    #     qpos0 = model.qpos0.at[hand_qids].set(
    #         model.qpos0[hand_qids]
    #         + jax.random.uniform(key, shape=(consts.N_ACTIVE,), minval=-0.05, maxval=0.05)
    #     )
    #
    #     rng, key = jax.random.split(rng)
    #     frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
    #         key, shape=(consts.N_ACTIVE,), minval=0.5, maxval=2.0
    #     )
    #     dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)
    #
    #     rng, key = jax.random.split(rng)
    #     armature = model.dof_armature[hand_qids] * jax.random.uniform(
    #         key, shape=(consts.N_ACTIVE,), minval=1.0, maxval=1.05
    #     )
    #     dof_armature = model.dof_armature.at[hand_qids].set(armature)
    #
    #     rng, key = jax.random.split(rng)
    #     dm = jax.random.uniform(key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1)
    #     body_mass = model.body_mass.at[hand_body_ids].set(
    #         model.body_mass[hand_body_ids] * dm
    #     )
    #
    #     rng, key = jax.random.split(rng)
    #     kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
    #         key, (model.nu,), minval=0.8, maxval=1.2
    #     )
    #     actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    #     actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
    #
    #     rng, key = jax.random.split(rng)
    #     kd = model.dof_damping[hand_qids] * jax.random.uniform(
    #         key, (consts.N_ACTIVE,), minval=0.8, maxval=1.2
    #     )
    #     dof_damping = model.dof_damping.at[hand_qids].set(kd)
    #
    #     return (
    #         geom_friction, body_mass, body_inertia, body_ipos, qpos0,
    #         dof_frictionloss, dof_armature, dof_damping,
    #         actuator_gainprm, actuator_biasprm,
    #     )

    @jax.vmap
    def rand(rng):
        geom_size = model.geom_size
        body_pos = model.body_pos
        body_mass = model.body_mass
        body_inertia = model.body_inertia
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
        return geom_size, body_pos, body_mass, body_inertia

    if do_size:
        randomized_keys["geom_size"] = f"[{size_lo}, {size_hi}]"
    if do_pos:
        randomized_keys["body_pos"] = f"[{pos_lo}, {pos_hi}]"
    if do_mass:
        randomized_keys["body_mass"] = f"[{mass_lo}, {mass_hi}]"
        randomized_keys["body_inertia"] = f"[{mass_lo}, {mass_hi}]"

    if randomized_keys:
        _log.debug(
            "domain_randomize: %s",
            ", ".join(f"{k}={v}" for k, v in randomized_keys.items()),
        )
    else:
        _log.debug("domain_randomize: no keys randomized (all ranges degenerate)")

    geom_size, body_pos, body_mass, body_inertia = rand(rng)

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

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    if replace_dict:
        in_axes = in_axes.tree_replace(axes_dict)
        model = model.tree_replace(replace_dict)
    return model, in_axes
