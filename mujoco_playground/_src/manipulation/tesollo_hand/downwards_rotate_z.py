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
"""Downwards-facing hand in-hand rotation environment for the Tesollo hand.

The hand is fixed in the pick_and_place orientation (palm facing downward) with
the wrist x-rotation restricted to keep it facing down. The cube starts on the
floor; the policy must learn to grasp and then rotate the cube around the world
z-axis to match a randomly sampled target orientation.
"""

__all__ = [
    "DownwardsRotateZ",
    "default_config",
    "domain_randomize",
]

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.tesollo_hand import base_grasp as tesollo_hand_base
from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module
from mujoco_playground._src.manipulation.tesollo_hand import (
    tesollo_hand_downwards_rotate_z_constants as consts,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        action_scale=0.5,
        action_mode="absolute",  # "absolute" (target = action mapped to ctrl range, default) or "delta" (incremental)
        action_repeat=1,
        ema_alpha=1.0,
        episode_length=150,
        # Task difficulty: the goal and what counts as reaching it.
        task=config_dict.create(
            min_target_angle=float(np.deg2rad(0.0)),
            max_target_angle=float(np.deg2rad(90.0)),
            symmetric_target=True,  # goal in either direction, else positive
            ori_tolerance_rad=float(np.deg2rad(3.0)),
            target_hold_time=0.25,  # seconds in tolerance to count a success
        ),
        sensor_bundle="proprio.target",
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.0,
                joint_vel=0.0,
                motor_targets=0.0,
                goal_quat=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                fingertip_pos=0.5,
                cube_ori=2.0,
                joint_vel=-0.002,
                wrist_vel=-0.02,
                action_rate=-0.5,
                cube_on_floor=-0.5,
            ),
            success_reward=2.0,
            ori_reward_margin_rad=float(np.deg2rad(30.0)),
            cube_ori_mode="orientation",  # "orientation" | "speed"
            target_spin_speed=1.0,  # rad/s (speed mode)
        ),
        pert_config=config_dict.create(
            enable=False,
            linear_velocity_pert=[0.0, 3.0],
            angular_velocity_pert=[0.0, 0.5],
            pert_duration_steps=[1, 100],
            pert_wait_steps=[60, 150],
        ),
        scene=config_dict.create(
            cube_mass=0.108,
            cube_size=0.07,  # edge length (m)
        ),
        constrain_wrist_translation=False,
        terminate_on_redrop=False,
        cube_oob_radius=0.0,  # terminate past this xy radius (m); 0 disables
        impl="warp",
        nconmax=200 * 8192,
        njmax=2200,
    )


class DownwardsRotateZ(tesollo_hand_base.TesolloHandGraspEnv):
    """In-hand rotation with a downward-facing hand.

    The cube starts on the floor. The policy must grasp it and rotate it around
    the world z-axis to match a randomly sampled target orientation. The wrist
    x-rotation DOF is restricted via ctrlrange to keep the palm facing down.
    """

    _TASK_KEYS: tuple[str, ...] = ("goal_quat",)
    _VEL_TOLERANCE: float = 0.3
    _TIP_FORCE_SENSORS: list[str] = [
        "rl_dg_1_tip_cube_force",
        "rl_dg_2_tip_cube_force",
        "rl_dg_3_tip_cube_force",
        "rl_dg_4_tip_cube_force",
        "rl_dg_5_tip_cube_force",
    ]

    def _task_obs_keys(self) -> tuple[str, ...]:
        if self._config.reward_config.cube_ori_mode == "speed":
            return ()
        return self._TASK_KEYS

    def _build_obs_components(self) -> dict:
        c = super()._build_obs_components()
        c["goal_quat"] = obs_module.ObsComponent(
            "goal_quat", self._obs_goal_quat, size=4,
            description="goal orientation quaternion",
            labels=("goal_qw", "goal_qx", "goal_qy", "goal_qz"),
        )
        return c

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(
            xml_path=consts.SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        model_dirty = False
        cube_body_id = self._mj_model.body("cube").id
        xml_cube_mass = float(self._mj_model.body_mass[cube_body_id])
        cfg_cube_mass = float(self._config.scene.cube_mass)
        if cfg_cube_mass != xml_cube_mass:
            scale = cfg_cube_mass / xml_cube_mass
            self._mj_model.body_mass[cube_body_id] = cfg_cube_mass
            self._mj_model.body_inertia[cube_body_id] *= scale
            model_dirty = True
        cube_geom_ids = np.array(
            [g for g in range(self._mj_model.ngeom)
             if self._mj_model.geom_bodyid[g] == cube_body_id]
        )
        xml_cube_size = 2.0 * float(
            self._mj_model.geom_size[self._mj_model.geom("cube").id, 0]
        )
        cfg_cube_size = float(self._config.scene.cube_size)
        if cfg_cube_size != xml_cube_size:
            self._mj_model.geom_size[cube_geom_ids] *= cfg_cube_size / xml_cube_size
            model_dirty = True
        # Snapshot pre-constraint ctrlrange so _joint_max_vel is not affected
        # by the narrower actuator range set by _constrain_wrist_translation.
        self._xml_ctrlrange = self._mj_model.actuator_ctrlrange.copy()
        if self._config.constrain_wrist_translation:
            self._constrain_wrist_translation(
                global_box=np.array([[-0.25, 0.25],  # global x, centred on cube
                                     [-0.25, 0.25],  # global y, centred on cube
                                     [0.0, 1.0]]),   # global z, absolute height
                cube_relative_axes=(0, 1),
            )
            model_dirty = True
        if model_dirty:
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _constrain_wrist_translation(
        self, global_box: np.ndarray, cube_relative_axes: tuple[int, ...] = ()
    ) -> None:
        """Limit the 3 wrist slide joints to a global-frame box.

        global_box is a (3, 2) array of [lo, hi] limits, in metres, for the hand
        base in the global (x, y, z) frame. Axes listed in cube_relative_axes are
        offset by the cube's global position (so e.g. x/y can track the cube);
        the remaining axes are interpreted as absolute global coordinates.

        The wrist slides are global-axis-aligned translations of the hand base,
        but the downward hand mounting permutes/offsets their joint-local axes,
        so the desired global box is mapped back to joint qpos limits by probing
        the compiled model.
        """
        slides = ["rj_wrist_0_1", "rj_wrist_0_2", "rj_wrist_0_3"]
        m = self._mj_model
        probe = mujoco.MjData(m)
        rh = m.body("rh").id
        cube = m.body("cube").id
        qadr = {j: m.jnt_qposadr[m.joint(j).id] for j in slides}
        # joint id -> actuator id, to keep ctrlrange in sync with jnt_range.
        act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}

        def hand_xpos(overrides: dict) -> np.ndarray:
            probe.qpos[:] = m.keyframe("home").qpos
            for j, v in overrides.items():
                probe.qpos[qadr[j]] = v
            mujoco.mj_forward(m, probe)
            return probe.xpos[rh].copy()

        zero = {j: 0.0 for j in slides}
        p0 = hand_xpos(zero)                  # hand base position at all slides = 0
        cube_pos = probe.xpos[cube].copy()    # cube global position (static at x=y=0)
        for j in slides:
            disp = hand_xpos({**zero, j: 1.0}) - p0  # global displacement / unit qpos
            g = int(np.argmax(np.abs(disp)))         # global axis index (0=x, 1=y, 2=z)
            sign = float(np.sign(disp[g]))
            center = cube_pos[g] if g in cube_relative_axes else 0.0
            lo_g = center + global_box[g, 0]         # desired global range on this axis
            hi_g = center + global_box[g, 1]
            # global_coord = p0[g] + sign * q  ->  q = sign * (global_coord - p0[g])
            qa = sign * (lo_g - p0[g])
            qb = sign * (hi_g - p0[g])
            lo, hi = sorted((qa, qb))
            jid = m.joint(j).id
            m.jnt_range[jid] = [lo, hi]
            m.jnt_limited[jid] = 1
            aid = act_of.get(jid)
            if aid is not None:
                m.actuator_ctrlrange[aid] = [lo, hi]

    def _post_init(self) -> None:
        assert self._config.reward_config.cube_ori_mode in ("orientation", "speed"), (
            f"cube_ori_mode must be 'orientation' or 'speed', got "
            f"{self._config.reward_config.cube_ori_mode!r}"
        )
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos, dtype=float)
        self._init_mpos = jp.array(home_key.mpos, dtype=float)
        self._init_mquat = jp.array(home_key.mquat, dtype=float)
        # jp.array copies, not numpy views: a view would silently track any
        # later in-place edit of actuator_ctrlrange on mj_model.
        self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1])
        self._wrist_qids = mjx_env.get_qpos_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._wrist_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.WRIST_JOINT_NAMES)
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._cube_body_id = self._mj_model.body("cube").id
        # Nominal cube half-height; reset() scales the free-body resting z by the
        # per-env cube size (under cube-size DR) so it always rests on the floor.
        self._cube_half_z = float(self._mj_model.geom_size[self._cube_geom_id, 2])
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
        # Weight of the cube (N), used to normalize the floor-support penalty:
        # when the floor bears the cube's full weight the penalty saturates at 1.
        g = float(np.linalg.norm(self._mj_model.opt.gravity))
        self._cube_weight = float(self._cube_mass) * g
        ctrl_span = self._xml_ctrlrange[:, 1] - self._xml_ctrlrange[:, 0]
        self._joint_max_vel = jp.array(ctrl_span)
        self._default_wrist_pose = self._init_q[self._wrist_qids]
        self._default_pose = self._init_q[self._hand_qids]
        self._cube_init = self._init_q[self._cube_qids]
        self._geom = consts.SceneGeometry.from_mj_model(self._mj_model)
        self._obs_components = self._build_obs_components()
        obs_module.validate_spec(
            self._config.sensor_bundle,
            self._task_obs_keys(),
            self._obs_components,
            self._config.obs_noise.scales,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        state = self._build_obs(
            self._config.sensor_bundle, self._task_obs_keys(), data, info
        )
        return {
            "state": state,
            "privileged_state": self._obs_privileged(data, info),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def sync_mj_model_meshes(self) -> None:
        """Make mj_model's cube box + visual mesh match the current mjx_model's
        cube geom size, so renders (incl. W&B training videos) show the actual
        per-env (DR'd) cube size. The visible cube is a MESH whose vertices are
        unaffected by geom_size, hence the spec recompile. No-op when sizes match.
        The cube is a free body, so its pose comes from the trajectory qpos and
        body_pos is left untouched (unlike the welded pinch cube)."""
        mjx_geom_size = np.array(self._mjx_model.geom_size)
        if np.allclose(self._mj_model.geom_size, mjx_geom_size):
            return
        cube_bid = self._mj_model.body("cube").id
        mesh_gid = next(
            g for g in range(self._mj_model.ngeom)
            if self._mj_model.geom_bodyid[g] == cube_bid
            and self._mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        )
        for body in self._mj_spec.bodies:
            if body.name != "cube":
                continue
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_BOX and geom.name:
                    geom.size = mjx_geom_size[self._mj_model.geom(geom.name).id]
                elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    for mesh in self._mj_spec.meshes:
                        if mesh.name == "cube_mesh":
                            mesh.scale = mjx_geom_size[mesh_gid]
                            break
            break
        m_new = self._mj_spec.compile()
        self._mj_model.geom_size[:] = m_new.geom_size
        self._mj_model.mesh_vert[:] = m_new.mesh_vert

    def render(self, trajectory, height=240, width=320, camera=None,
               scene_option=None, modify_scene_fns=None):
        self.sync_mj_model_meshes()
        return mjx_env.render_array(
            self._mj_model, trajectory, height, width, camera,
            scene_option=scene_option, modify_scene_fns=modify_scene_fns,
        )

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, p_rng = jax.random.split(rng)
        start_pos = self._cube_init[:2] + jax.random.uniform(
            p_rng, (2,), minval=-0.01, maxval=0.01
        )
        # Rest on the floor for this env's (possibly DR-scaled) cube size.
        cube_z = self._cube_init[2] * (
            self._mjx_model.geom_size[self._cube_geom_id, 2] / self._cube_half_z
        )
        start_pos = jp.array([start_pos[0], start_pos[1], cube_z])

        rng, sign_rng, frac_rng = jax.random.split(rng, 3)
        goal_sign = jp.where(jax.random.bernoulli(sign_rng), 1.0, -1.0)
        goal_frac = jax.random.uniform(frac_rng, minval=0.0, maxval=1.0)
        task = self._config.task
        lo, hi = task.min_target_angle, task.max_target_angle
        angle = lo + goal_frac * (hi - lo)
        if task.symmetric_target:
            angle = goal_sign * angle
        goal_quat = jp.array([jp.cos(angle / 2), 0.0, 0.0, jp.sin(angle / 2)])

        qpos = self._init_q.at[self._cube_qids[:3]].set(start_pos)
        qvel = jp.zeros(self.mj_model.nv)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=self._default_pose,
            qvel=qvel,
            mocap_pos=self._init_mpos,
            mocap_quat=goal_quat,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
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
        # them affinely correlated); other reset keys are unchanged.
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

        info = {
            "rng": rng,
            "step": 0,
            "success_count": jp.zeros((), dtype=jp.int32),
            "at_target_step_counter": jp.zeros((), dtype=jp.int32),
            "motor_targets": data.ctrl,
            "last_action": jp.zeros(self.mj_model.nu),
            "goal_quat": goal_quat,
            "goal_sign": goal_sign,
            "goal_frac": goal_frac,
            "pert_wait_steps": pert_wait_steps,
            "pert_duration_steps": pert_duration_steps,
            "pert_vel": jp.array([pert_lin] * 3 + [pert_ang] * 3),
            "pert_dir": jp.zeros(6, dtype=float),
            "last_pert_step": jp.array([-jp.inf], dtype=float),
            "ori_error": jp.zeros(()),
            "lifted_once": jp.array(False),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}_per_step"] = jp.zeros(())
        metrics["reward/success_per_step"] = jp.zeros((), dtype=float)
        metrics["floor_support_fraction_per_step"] = jp.zeros((), dtype=float)
        metrics["goal_angle_per_step"] = jp.rad2deg(
            jp.abs(2.0 * jp.arctan2(goal_quat[3], goal_quat[0]))
        )

        obs = self._get_obs(data, info)
        rew, done = jp.zeros(2)
        return mjx_env.State(data, obs, rew, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        if self._config.action_mode == "absolute":
            # action in [-1, 1] maps linearly onto the actuator ctrl range:
            # direct joint-position targets rather than incremental deltas.
            target = self._lowers + 0.5 * (action + 1.0) * (self._uppers - self._lowers)
            target = jp.clip(target, self._lowers, self._uppers)
        else:
            target = jp.clip(
                state.data.ctrl + action * self._config.action_scale,
                self._lowers,
                self._uppers,
            )
        motor_targets = (
            self._config.ema_alpha * target
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        goal_quat = state.info["goal_quat"]

        cube_lifted = self._cube_floor_support_fraction(data) == 0.0
        ori_error = self._cube_orientation_error(data, goal_quat)
        at_goal = cube_lifted & (ori_error < self._config.task.ori_tolerance_rad)

        state.info["ori_error"] = ori_error

        # round() before the cast: 0.3/0.05 is 5.999... and int() truncates.
        hold_steps = jp.round(
            self._config.task.target_hold_time / self.dt
        ).astype(jp.int32)
        at_target_counter = jp.where(
            at_goal, state.info["at_target_step_counter"] + 1, 0
        ).astype(jp.int32)
        state.info["at_target_step_counter"] = at_target_counter
        success = at_target_counter >= hold_steps


        success_count = jp.where(
            success, state.info["success_count"] + 1, state.info["success_count"]
        ).astype(jp.int32)
        state.info["success_count"] = success_count
        state.metrics["goal_angle_per_step"] = jp.rad2deg(
            jp.abs(2.0 * jp.arctan2(goal_quat[3], goal_quat[0]))
        )

        done = self._get_termination(data)
        if self._config.terminate_on_redrop:
            support = self._cube_floor_support_fraction(data)
            lifted_once = state.info["lifted_once"] | (support < 0.05)
            state.info["lifted_once"] = lifted_once
            done = done | (lifted_once & (support > 0.5))
        obs = self._get_obs(data, state.info)
        raw_rewards = self._get_reward(data, state.info, action)
        state.info["last_action"] = action
        scaled_rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}

        state.info["step"] += 1
        state.metrics["reward/success_per_step"] = success.astype(float)
        for k, v in raw_rewards.items():
            state.metrics[f"reward/{k}_per_step"] = v
        state.metrics["floor_support_fraction_per_step"] = self._cube_floor_support_fraction(data)

        rew = sum(scaled_rewards.values()) * self.dt
        rew += self._config.reward_config.success_reward * success.astype(float) * self.dt
        done = done.astype(rew.dtype)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        done = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        if self._config.cube_oob_radius > 0.0:
            done = done | (
                jp.linalg.norm(self.get_cube_position(data)[:2])
                > self._config.cube_oob_radius
            )
        return done

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @staticmethod
    def r_fingertip_pos_per_tip(fingertip_dist: jax.Array, cube_half_size: float) -> jax.Array:
        return reward.tolerance(
            fingertip_dist, (0, cube_half_size), margin=0.1, sigmoid="reciprocal",
        )

    @staticmethod
    def r_joint_vel(
        hand_qvel: jax.Array, max_velocity: jax.Array, vel_tolerance: float
    ) -> jax.Array:
        active = max_velocity > vel_tolerance
        excess = jp.maximum(0.0, jp.abs(hand_qvel) - vel_tolerance)
        denom = jp.where(active, max_velocity - vel_tolerance, 1.0)
        return jp.mean(jp.where(active, (excess / denom) ** 2, 0.0))

    @staticmethod
    def r_wrist_vel(wrist_qvel: jax.Array, max_velocity: jax.Array) -> jax.Array:
        return jp.mean((wrist_qvel / jp.maximum(max_velocity, 1e-6)) ** 2)

    @staticmethod
    def r_cube_orientation(ori_error: jax.Array, margin_rad: float) -> jax.Array:
        return reward.tolerance(ori_error, (0.0, 0.0), margin=margin_rad, sigmoid="reciprocal")

    @staticmethod
    def r_cube_spin_speed(spin_z: jax.Array, target_speed: float) -> jax.Array:
        return jp.clip(spin_z / target_speed, -1.0, 1.0)

    @staticmethod
    def r_cube_on_floor(floor_support_fraction: jax.Array) -> jax.Array:
        return floor_support_fraction

    @staticmethod
    def r_action_rate(action: jax.Array, prev_action: jax.Array) -> jax.Array:
        return jp.mean(jp.square(action - prev_action))

    def _get_reward(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        action: jax.Array,
    ) -> dict[str, jax.Array]:
        cube_pos = self.get_cube_position(data)
        cube_ori_error = info["ori_error"]
        floor_support_fraction = self._cube_floor_support_fraction(data)
        # Lift gate in [0, 1]: 0 while the floor bears the cube, 1 once the hand
        # does. Continuous, so the orientation reward fades in without a jump.
        lift_gate = 1.0 - floor_support_fraction

        fingertip_distances = jp.linalg.norm(
            self.get_fingertip_positions(data).reshape(-1, 3) - cube_pos, axis=1
        )
        # Distance beyond this env's DR'd cube surface: tolerance() rejects
        # traced bounds, and for dist >= 0 this equals bounds=(0, half).
        half = self.mjx_model.geom_size[self._cube_geom_id, 0]
        per_tip = reward.tolerance(
            jp.maximum(fingertip_distances - half, 0.0),
            (0.0, 0.0), margin=0.1, sigmoid="reciprocal",
        )
        fingertip_reward = jp.mean(per_tip)

        if self._config.reward_config.cube_ori_mode == "speed":
            spin_z = self.get_cube_angvel(data)[2]
            cube_task_reward = self.r_cube_spin_speed(
                spin_z, self._config.reward_config.target_spin_speed
            )
        else:
            cube_task_reward = self.r_cube_orientation(
                cube_ori_error, self._config.reward_config.ori_reward_margin_rad
            )

        return {
            "fingertip_pos": fingertip_reward,
            "cube_ori": lift_gate * cube_task_reward,
            "joint_vel": self.r_joint_vel(
                data.qvel[self._hand_dqids], self._joint_max_vel, self._VEL_TOLERANCE
            ),
            "wrist_vel": self.r_wrist_vel(
                data.qvel[self._wrist_dqids], self._joint_max_vel[:6]
            ),
            "action_rate": self.r_action_rate(action, info["last_action"]),
            "cube_on_floor": self.r_cube_on_floor(floor_support_fraction),
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _cube_orientation_error(
        self, data: mjx.Data, goal_quat: jax.Array
    ) -> jax.Array:
        cube_ori = self.get_cube_orientation(data)
        quat_diff = math.quat_mul(cube_ori, math.quat_inv(goal_quat))
        quat_diff = math.normalize(quat_diff)
        return 2.0 * jp.asin(jp.clip(jp.linalg.norm(quat_diff[1:]), max=1.0))

    def _obs_goal_quat(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return info["goal_quat"]

    def _obs_privileged(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        parts = [
            data.qpos[self._hand_qids],        # 26: true joint angles
            data.qvel[self._hand_dqids],        # 26: true joint velocities
            self.get_cube_position(data),       # 3:  true cube position
            self.get_cube_linvel(data),         # 3:  cube linear velocity
            self.get_cube_angvel(data),         # 3:  cube angular velocity
            self.get_fingertip_positions(data), # 15: fingertip positions
            self.get_palm_position(data),       # 3:  palm position
            info["motor_targets"],              # 26: motor targets
        ]
        if self._config.reward_config.cube_ori_mode != "speed":
            parts.append(info["goal_quat"])     # 4:  goal orientation
        parts.append(self._obs_fingertip_forces(data, info))  # 5: per-tip forces
        return jp.concatenate(parts)

    # ------------------------------------------------------------------
    # Contact helpers
    # ------------------------------------------------------------------

    def _cube_floor_support_fraction(self, data: mjx.Data) -> jax.Array:
        """Fraction (in [0, 1]) of the cube's weight borne by the floor.

        Reads the net cube-floor contact force (world frame) from the
        ``cube_floor_force`` sensor and divides its vertical component by the
        cube's weight, so it is 0 when the hand fully supports the cube and 1
        once the floor takes its entire weight (clipped, since the hand can
        briefly press the cube into the floor with more than its own weight).
        """
        net_force = mjx_env.get_sensor_data(self.mj_model, data, "cube_floor_force")
        vertical_support = jp.abs(net_force[2])
        return jp.minimum(vertical_support / self._cube_weight, 1.0)

    def _maybe_apply_perturbation(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            d = jax.random.normal(rng, (6,))
            return d / jp.linalg.norm(d)

        def get_xfrc(
            state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
        ) -> jax.Array:
            u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
            force = (
                u_t * self._cube_mass * state.info["pert_vel"]
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




# Per-env cube-size scale range for domain randomization. Verified collision-free
# in the home pose up to ~1.20x (see experimentation check_cube_home_collision.py).
_CUBE_SIZE_RANGE = (0.85, 1.15)


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Cube-size-ONLY domain randomization: each env's cube geom is scaled by a
    factor in _CUBE_SIZE_RANGE and nothing else is randomized, so cube size is the
    single variable vs the no-DR control. The previous full DR suite (friction,
    mass, kp, damping, armature, joint jitter) is preserved, unused, as
    _domain_randomize_full below."""
    # Compile the scene directly for ID lookup: constructing DownwardsRotateZ()
    # here built an entire env (mjx.put_model included) just to read geom ids.
    mj_model = mujoco.MjSpec.from_file(
        consts.SCENE_XML.as_posix(), assets=tesollo_hand_base.get_assets()
    ).compile()
    cube_body_id = mj_model.body("cube").id
    cube_geom_ids = np.array(
        [g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == cube_body_id]
    )

    @jax.vmap
    def rand(rng):
        s = jax.random.uniform(rng, (), minval=_CUBE_SIZE_RANGE[0], maxval=_CUBE_SIZE_RANGE[1])
        return model.geom_size.at[cube_geom_ids].set(model.geom_size[cube_geom_ids] * s)

    geom_size = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"geom_size": 0})
    model = model.tree_replace({"geom_size": geom_size})
    return model, in_axes


def _domain_randomize_full(model: mjx.Model, rng: jax.Array):
    mj_model = mujoco.MjSpec.from_file(
        consts.SCENE_XML.as_posix(), assets=tesollo_hand_base.get_assets()
    ).compile()
    cube_body_id = mj_model.body("cube").id
    cube_geom_ids = np.array(
        [g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == cube_body_id]
    )
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
        silicone_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[silicone_geom_ids, 0].set(silicone_friction)

        rng, key = jax.random.split(rng)
        cube_scale = jax.random.uniform(
            key, (), minval=_CUBE_SIZE_RANGE[0], maxval=_CUBE_SIZE_RANGE[1]
        )
        geom_size = model.geom_size.at[cube_geom_ids].set(
            model.geom_size[cube_geom_ids] * cube_scale
        )

        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
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
            + jax.random.uniform(key, shape=(consts.NQ,), minval=-0.05, maxval=0.05)
        )

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=1.0, maxval=1.05
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
            key, (consts.NQ,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction, geom_size, body_mass, body_inertia, body_ipos, qpos0,
            dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
        )

    (
        geom_friction, geom_size, body_mass, body_inertia, body_ipos, qpos0,
        dof_frictionloss, dof_armature, dof_damping, actuator_gainprm, actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "geom_size": 0, "body_mass": 0, "body_inertia": 0,
        "body_ipos": 0, "qpos0": 0, "dof_frictionloss": 0, "dof_armature": 0,
        "dof_damping": 0, "actuator_gainprm": 0, "actuator_biasprm": 0,
    })
    model = model.tree_replace({
        "geom_friction": geom_friction, "geom_size": geom_size, "body_mass": body_mass,
        "body_inertia": body_inertia, "body_ipos": body_ipos, "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss, "dof_armature": dof_armature,
        "dof_damping": dof_damping, "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })
    return model, in_axes
