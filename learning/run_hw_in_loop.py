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
"""Train a PPO agent using JAX on the specified environment."""

import matplotlib.pyplot as plt

import zmq
import numpy as np

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX
import wandb

from scipy.spatial.transform import Rotation as R

import hw_scripts.tesollo_receiver as tesollo_state_receiver
import hw_scripts.ur5_state_receiver as ur5_state_receiver

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../../"))

from src.robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator


class PoseReceiver:
    def __init__(self, config):
        self.estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        self.estimator.start()

    def get_pose(self, id=None):
        # Receive pose data
        ids = self.estimator.get_tracked_board_ids()

        for box_id in ids:
            box_pose = self.estimator.get_pose(box_id)

            if id is None or box_id == id:
                return box_pose
            
        return None

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

#### HW FLAGS
_LOGGING = flags.DEFINE_boolean(
    "logging", False, "With data logging?"
)

_PLOTTING = flags.DEFINE_boolean(
    "plotting", False, "With data logging?"
)

_REAL = flags.DEFINE_boolean(
    "real_sys", False, "closed loop?"
)

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Number of timesteps")
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string("policy_obs_key", "state", "Policy obs key")
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")

_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in mujoco_playground.manipulation._envs:
        if _VISION.value:
            return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
        return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        if _VISION.value:
            return dm_control_suite_params.brax_vision_ppo_config(env_name, _IMPL.value)
        return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value

    ppo_params = get_rl_config(_ENV_NAME.value)

    if _NUM_TIMESTEPS.present:
        ppo_params.num_timesteps = _NUM_TIMESTEPS.value
    if True:
        ppo_params.num_timesteps = 0
    if _NUM_EVALS.present:
        ppo_params.num_evals = _NUM_EVALS.value
    if _REWARD_SCALING.present:
        ppo_params.reward_scaling = _REWARD_SCALING.value
    if _EPISODE_LENGTH.present:
        ppo_params.episode_length = _EPISODE_LENGTH.value
    if _NORMALIZE_OBSERVATIONS.present:
        ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _ACTION_REPEAT.present:
        ppo_params.action_repeat = _ACTION_REPEAT.value
    if _UNROLL_LENGTH.present:
        ppo_params.unroll_length = _UNROLL_LENGTH.value
    if _NUM_MINIBATCHES.present:
        ppo_params.num_minibatches = _NUM_MINIBATCHES.value
    if _NUM_UPDATES_PER_BATCH.present:
        ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
    if _DISCOUNTING.present:
        ppo_params.discounting = _DISCOUNTING.value
    if _LEARNING_RATE.present:
        ppo_params.learning_rate = _LEARNING_RATE.value
    if _ENTROPY_COST.present:
        ppo_params.entropy_cost = _ENTROPY_COST.value
    if _NUM_ENVS.present:
        ppo_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _BATCH_SIZE.present:
        ppo_params.batch_size = _BATCH_SIZE.value
    if _MAX_GRAD_NORM.present:
        ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
    if _CLIPPING_EPSILON.present:
        ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
    if _POLICY_HIDDEN_LAYER_SIZES.present:
        ppo_params.network_factory.policy_hidden_layer_sizes = list(
            map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
        )
    if _VALUE_HIDDEN_LAYER_SIZES.present:
        ppo_params.network_factory.value_hidden_layer_sizes = list(
            map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
        )
    if _POLICY_OBS_KEY.present:
        ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
    if _VALUE_OBS_KEY.present:
        ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
    if _VISION.value:
        env_cfg.vision = True
        env_cfg.vision_config.render_batch_size = ppo_params.num_envs
    env = registry.load(_ENV_NAME.value, config=env_cfg)
    if _RUN_EVALS.present:
        ppo_params.run_evals = _RUN_EVALS.value
    if _LOG_TRAINING_METRICS.present:
        ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
    if _TRAINING_METRICS_STEPS.present:
        ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

    print(f"Environment Config:\n{env_cfg}")
    print(f"PPO Training Parameters:\n{ppo_params}")

    # Generate unique experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"

    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Handle checkpoint loading
    if _LOAD_CHECKPOINT_PATH.value is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
        if ckpt_path.is_dir():
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None
        assert False

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    network_fn = (
        ppo_networks_vision.make_ppo_networks_vision
        if _VISION.value
        else ppo_networks.make_ppo_networks
    )
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(network_fn, **ppo_params.network_factory)
    else:
        network_factory = network_fn

    if _DOMAIN_RANDOMIZATION.value:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            _ENV_NAME.value
        )

    if _VISION.value:
        env = wrapper.wrap_for_brax_training(
            env,
            vision=True,
            num_vision_envs=env_cfg.vision_config.render_batch_size,
            episode_length=ppo_params.episode_length,
            action_repeat=ppo_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

    num_eval_envs = (
        ppo_params.num_envs if _VISION.value else ppo_params.get("num_eval_envs", 128)
    )

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_checkpoint_path,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

    times = [time.monotonic()]

    # Progress function for logging
    def progress(num_steps, metrics):
        times.append(time.monotonic())

    # Load evaluation environment.
    eval_env = None
    if not _VISION.value:
        eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
    num_envs = 1
    if _VISION.value:
        num_envs = env_cfg.vision_config.render_batch_size

    policy_params_fn = lambda *args: None

    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )

    print("Done training.")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    print("Starting inference...")

    # Create inference function.
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(rng)

    model = eval_env.mj_model
    data = mujoco.MjData(model)

    hand_context = zmq.Context()
    hand_conf_socket = hand_context.socket(zmq.PUB)
    hand_port = 8000
    hand_conf_socket.bind(f"tcp://localhost:{hand_port}")

    wrist_context = zmq.Context()
    wrist_conf_socket = wrist_context.socket(zmq.PUB)
    wrist_port = 8001
    wrist_conf_socket.bind(f"tcp://localhost:{wrist_port}")

    if _REAL.value:
        pose_estimator_config = {"port": 5557}
        mocap = PoseReceiver(pose_estimator_config)

        qpos_err_hist = jp.zeros(eval_env._config.history_len * 23)
        cube_pos_err_hist = jp.zeros(eval_env._config.history_len * 3)
        cube_ori_err_hist = jp.zeros(eval_env._config.history_len * 6)

    start_time = time.time()

    if _LOGGING.value or _PLOTTING.value:
        timestamps = []

        actual_wrist_poses = []
        desired_wrist_poses = []

        actual_hand_poses = []
        desired_hand_poses = []

        hand_joint_listener = tesollo_state_receiver.TesolloJointStateReceiver()
        robot_joint_listener = ur5_state_receiver.Ur5StateReceiver()

        sim_obs = []
        actual_obs = []

        all_actions = []
        all_targets = []

    act = np.zeros(23)
    targets = np.zeros(23)

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            while viewer.is_running():
                act_rng, rng = jax.random.split(rng)
                # ctrl, _ = jit_inference_fn(state.obs, act_rng)
                # state = jit_step(state, ctrl)
                # rollout.append(state)

                # make observation from data
                if _REAL.value:
                    # observation
                    # noisy_joint_angles,  # 23
                    # qpos_error_history,  # 23 * history_len
                    # cube_pos_error_history,  # 3 * history_len
                    # cube_ori_error_history,  # 6 * history_len
                    # info["last_act"],  # 23
                    def wrist_q_from_quaternion(quat):
                        """
                        quat: iterable [x, y, z, w]  (scalar last)
                        returns: (qx, qy, qz) rotations about X, Y, Z in radians
                        """

                        quat = np.asarray(quat, dtype=float)

                        # Normalize to be safe
                        quat = quat / np.linalg.norm(quat)

                        r_robot = R.from_quat(quat, scalar_first=False)
                        
                        R_x = R.from_euler("x", -90, degrees=True)
                        r_hand = R_x * r_robot * R.from_euler("z", -30, degrees=True)
                        # r_hand = r_robot

                        # intrinsic XYZ (matches X->Y->Z joint chain)
                        # qx, qy, qz = r_hand.as_euler('yxz', degrees=False)
                        qx, qy, qz = r_hand.as_euler('yxz', degrees=False)
                        qy *= -1

                        return np.array([qx, qy, qz])

                    wrist_q = wrist_q_from_quaternion(robot_joint_listener.get()[3:]) #scalar last
                    hand_q = hand_joint_listener.get()

                    qpos = np.concat([wrist_q, hand_q])

                    cube_pose = mocap.get_pose()

                    if cube_pose is None:
                        print("NO CUBE POSE MEASUREMENT")
                        cube_pose = np.array([0, 0, 0, 1, 0, 0, 0])

                    camera_to_mujoco_frame = R.from_euler("z", 0, degrees=True)

                    world_to_mujoco_offset = np.array([0.05, 0.01, -0.33]) # likely load from somewhere (guesstimate)

                    cube_pos = cube_pose[:3] + world_to_mujoco_offset
                    cube_ori_rot_mat = camera_to_mujoco_frame * R.from_quat(cube_pose[3:], scalar_first=True) # quat with scalar first
                    cube_ori = cube_ori_rot_mat.as_quat(scalar_first=True)

                    # cube_goal_pos = None
                    cube_goal_ori = eval_env.get_cube_goal_orientation(data)

                    cube_pos_error = cube_pos

                    quat_diff = mujoco.mjx._src.math.quat_mul(
                        cube_ori, mujoco.mjx._src.math.quat_inv(cube_goal_ori)
                    )
                    print(2.0 * jp.asin(jp.clip(mujoco.mjx._src.math.norm(quat_diff[1:]), a_max=1.0)))
                    xmat_diff = mujoco.mjx._src.math.quat_to_mat(quat_diff).ravel()[3:]

                    cube_ori_error = xmat_diff

                    qpos_err_hist = (
                        jp.roll(qpos_err_hist, 23)
                        .at[:23]
                        .set(qpos - targets)
                    )
                    cube_pos_err_hist = (
                        jp.roll(cube_pos_err_hist, 3).at[:3].set(cube_pos_error)
                    )
                    cube_ori_err_hist = (
                        jp.roll(cube_ori_err_hist, 6).at[:6].set(cube_ori_error)
                    )
                    # prev_act = state.info["motor_targets"]
                    prev_act = act

                    # print(cube_ori_error)

                    # qpos: 23
                    # qpos_err_hist: eval_env._config.history_len * 23
                    # cube_pos_err_hist: eval_env._config.history_len * 3
                    # cube_ori_err_hist: eval_env._config.history_len * 6
                    # prev_act: 23
                    obs = jp.concatenate([
                        qpos,
                        qpos_err_hist,
                        0 * cube_pos_err_hist,
                        cube_ori_err_hist,
                        prev_act
                    ])

                    state.obs["state"] = obs

                    # print(cube_pose)
                    # print(cube_pos)
                    # print(eval_env.get_palm_position(data) - eval_env.get_cube_position(data))

                    full_q = state.data.qpos
                    full_q = full_q.at[eval_env._hand_qids].set(qpos)
                    full_q = full_q.at[eval_env._cube_qids[3:]].set(cube_ori)
                    full_q = full_q.at[eval_env._cube_qids[:3]].set(cube_pos[:3])

                    # tmp = state.data.replace(
                    #     qpos=full_q,
                    #     # qvel=state.data.qvel * 0
                    # )

                    # state = state.replace(data=tmp)
                    # state = state.replace(data=tmp,obs=state.obs)
                    state = state.replace(obs=state.obs)

                    # print(qpos)
                    # print(state.obs["state"][:23])

                    if _LOGGING.value or _PLOTTING.value:
                        sim_obs.append(state.obs["state"])
                        actual_obs.append(obs)

                print("running inference")
                inf_start = time.time()
                act = jit_inference_fn(state.obs, act_rng)[0]
                print("inf time", time.time() - inf_start)

                step_start = time.time()
                # state = eval_env.step(state, act)
                state = jit_step(state, act)
                print("step time", time.time() - step_start)

                # data = state.data
                data.qpos = state.data.qpos
                data.mocap_pos = state.data.mocap_pos
                data.mocap_quat = state.data.mocap_quat
                mujoco.mj_forward(model, data)

                delta = act * eval_env._config.action_scale
                
                prev_targets = 1. * targets
                
                targets = qpos + delta
                targets = jp.clip(targets, eval_env._lowers, eval_env._uppers)

                alpha = eval_env._config.ema_alpha
                targets = (
                    alpha * targets
                    + (1 - alpha) * prev_targets
                )

                # hand_cmd = hand_pose + delta[3:]
                # wrist_cmd = wrist_q + delta[:3]
                hand_cmd = targets[3:]
                wrist_cmd = targets[:3]

                # hand_cmd = state.info["motor_targets"][3:]
                # hand_cmd = state.data.qpos[eval_env._finger_qids]

                # hand_pose = hand_joint_listener.get()
                # hand_joint_diff = hand_cmd - np.array(hand_pose)
                # max_size = 10.
                # hand_cmd_clip = np.clip(hand_joint_diff, -max_size * np.ones_like(hand_joint_diff), max_size * np.ones_like(hand_joint_diff))

                # hand_cmd = np.array(hand_pose) + hand_cmd_clip

                desired_hand_pose = state.data.qpos[eval_env._finger_qids]
                desired_wrist_pose = np.concatenate([eval_env.get_wrist_position(data), 
                                                     eval_env.get_wrist_orientation(data)])

                # controllers
                hand_conf_socket.send_json(
                    {"poses": hand_cmd.tolist()}
                )
                wrist_conf_socket.send_json(
                    {
                        "poses": eval_env.get_wrist_position(data).tolist()
                        + eval_env.get_wrist_orientation(data).tolist(),
                        # "config": state.info["motor_targets"][:3].tolist()
                        "config": wrist_cmd.tolist()
                    }
                )

                # listeners/loggers
                if _LOGGING.value or _PLOTTING.value:
                    timestamps.append(time.time() - start_time)

                    # actualy log -> should be async ideally, but whatever for now
                    hand_pose = hand_joint_listener.get()
                    desired_hand_poses.append(desired_hand_pose)
                    actual_hand_poses.append(hand_pose)

                    robot_ee_pose = robot_joint_listener.get()
                    desired_wrist_poses.append(desired_wrist_pose)
                    actual_wrist_poses.append(robot_ee_pose)

                    all_actions.append(act)
                    all_targets.append(targets)

                viewer.sync()
                time.sleep(env_cfg["ctrl_dt"])
    except KeyboardInterrupt:
        print('stopped!')

    # export logs
    if _LOGGING.value:
        now = datetime.datetime.now()
        folder_name = "./hw_logs/" + now.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the folder
        os.makedirs(folder_name, exist_ok=True)
        
        # put everythign in one folder
        np.save(os.path.join(folder_name, "times.npy"), np.array(timestamps))

        np.save(os.path.join(folder_name, "actual_wrist_poses.npy"), np.array(actual_wrist_poses))
        np.save(os.path.join(folder_name, "desired_wrist_poses.npy"), np.array(desired_wrist_poses))
        np.save(os.path.join(folder_name, "actual_hand_poses.npy"), np.array(actual_hand_poses))
        np.save(os.path.join(folder_name, "desired_hand_poses.npy"), np.array(desired_hand_poses))

        # Observations
        np.save(os.path.join(folder_name, "actual_observations.npy"), np.array(actual_obs))
        np.save(os.path.join(folder_name, "simulated_observations.npy"), np.array(sim_obs))

        # Observations
        np.save(os.path.join(folder_name, "actions.npy"), np.array(all_actions))
        np.save(os.path.join(folder_name, "targets.npy"), np.array(all_targets))

        print(f"Exported data to {folder_name}")

    if _PLOTTING.value:
        plt.figure("Observations wrist")
        plt.plot(np.array(actual_obs)[:, :3])
        plt.plot(np.array(sim_obs)[:, :3], '--')

        plt.figure("Observations hand")
        plt.plot(np.array(actual_obs)[:, 3:23])
        plt.plot(np.array(sim_obs)[:, 3:23], '--')

        fig = plt.figure("Rest of obs")
        # qpos: 23
        # qpos_err_hist: eval_env._config.history_len * 23
        # cube_pos_err_hist: eval_env._config.history_len * 3
        # cube_ori_err_hist: eval_env._config.history_len * 6
        # prev_act: 23
        ax = fig.add_subplot(4, 1, 1)
        start_idx = 23
        num_obs = eval_env._config.history_len*23
        ax.plot(np.array(actual_obs)[:, start_idx:start_idx+num_obs])
        
        ax = fig.add_subplot(4, 1, 2)
        start_idx = start_idx + num_obs
        num_obs = eval_env._config.history_len*3
        ax.plot(np.array(actual_obs)[:, start_idx:start_idx+num_obs])

        ax = fig.add_subplot(4, 1, 3)
        start_idx = start_idx + num_obs
        num_obs = eval_env._config.history_len*6
        ax.plot(np.array(actual_obs)[:, start_idx:start_idx+num_obs])

        ax = fig.add_subplot(4, 1, 4)
        start_idx = start_idx + num_obs
        num_obs = 23
        ax.plot(np.array(actual_obs)[:, start_idx:start_idx+num_obs])

        plt.figure("pred. actions")
        plt.plot(np.array(all_actions))

        plt.figure("comp. targets")
        plt.plot(np.array(all_targets))

        plt.show()

if __name__ == "__main__":
    app.run(main)
