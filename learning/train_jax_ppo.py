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

import datetime
import functools
import json
import os
import sys
import tempfile
import time
import warnings
import yaml

from absl import app
from absl import flags
from absl import logging
import brax_compat  # noqa: F401  -- must precede brax imports
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
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

import early_stop


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Use CUDA's stream-ordered allocator; avoids BFC fragmentation when JAX and
# warp share the same GPU (the BFC allocator OOMs even with free memory).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
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
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_LOG_VIDEO_TO_WANDB = flags.DEFINE_boolean(
    "log_video_to_wandb",
    False,
    "Upload rollout videos to Weights & Biases after training. Requires"
    " --use_wandb.",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
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
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
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
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
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
_ENV_OVERRIDES_FILE = flags.DEFINE_string(
    "env_overrides_file",
    None,
    "Path to a YAML file of flat-dotted env-config overrides "
    "(e.g. 'obs_noise.level: 2.0'). Applied on top of the env's default_config.",
)
_EVAL_ENV_OVERRIDES_FILE = flags.DEFINE_string(
    "eval_env_overrides_file",
    None,
    "Path to a YAML file of flat-dotted overrides applied to the EVAL env on top of"
    " the train overrides. Use to evaluate the real task while training on a proxy"
    " (e.g. 'curriculum.enable: false' to eval at full difficulty under a curriculum).",
)
_EARLY_STOP = flags.DEFINE_boolean(
    "early_stop",
    True,
    "Abort runs that diverge unrecoverably (NaN/Inf, KL collapse, reward collapse)."
    " Requires --log_training_metrics.",
)
_NUM_RESETS_PER_EVAL = flags.DEFINE_integer(
    "num_resets_per_eval",
    None,
    "Override ppo num_resets_per_eval. Set 0 to keep training envs (and any per-env"
    " curriculum state) across evals instead of resetting them to the initial state.",
)
_FULL_RESET = flags.DEFINE_boolean(
    "full_reset",
    False,
    "Re-run env.reset() on every episode end (instead of reverting to the cached"
    " first state), so per-episode quantities (e.g. a curriculum target) are"
    " resampled. Curriculum state is preserved via the wrapper's preserve_info.",
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
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def _collect_flag_overrides() -> dict:
  """Return {flag_name: value} for every CLI flag explicitly set by the user."""
  return {
      name: fh.value
      for name, fh in [
          ("env_name", _ENV_NAME),
          ("impl", _IMPL),
          ("vision", _VISION),
          ("load_checkpoint_path", _LOAD_CHECKPOINT_PATH),
          ("suffix", _SUFFIX),
          ("play_only", _PLAY_ONLY),
          ("use_wandb", _USE_WANDB),
          ("log_video_to_wandb", _LOG_VIDEO_TO_WANDB),
          ("use_tb", _USE_TB),
          ("domain_randomization", _DOMAIN_RANDOMIZATION),
          ("seed", _SEED),
          ("num_timesteps", _NUM_TIMESTEPS),
          ("num_videos", _NUM_VIDEOS),
          ("num_evals", _NUM_EVALS),
          ("reward_scaling", _REWARD_SCALING),
          ("episode_length", _EPISODE_LENGTH),
          ("normalize_observations", _NORMALIZE_OBSERVATIONS),
          ("action_repeat", _ACTION_REPEAT),
          ("unroll_length", _UNROLL_LENGTH),
          ("num_minibatches", _NUM_MINIBATCHES),
          ("num_updates_per_batch", _NUM_UPDATES_PER_BATCH),
          ("discounting", _DISCOUNTING),
          ("learning_rate", _LEARNING_RATE),
          ("entropy_cost", _ENTROPY_COST),
          ("num_envs", _NUM_ENVS),
          ("num_eval_envs", _NUM_EVAL_ENVS),
          ("batch_size", _BATCH_SIZE),
          ("max_grad_norm", _MAX_GRAD_NORM),
          ("clipping_epsilon", _CLIPPING_EPSILON),
          ("policy_hidden_layer_sizes", _POLICY_HIDDEN_LAYER_SIZES),
          ("value_hidden_layer_sizes", _VALUE_HIDDEN_LAYER_SIZES),
          ("policy_obs_key", _POLICY_OBS_KEY),
          ("value_obs_key", _VALUE_OBS_KEY),
          ("rscope_envs", _RSCOPE_ENVS),
          ("deterministic_rscope", _DETERMINISTIC_RSCOPE),
          ("run_evals", _RUN_EVALS),
          ("log_training_metrics", _LOG_TRAINING_METRICS),
          ("training_metrics_steps", _TRAINING_METRICS_STEPS),
          ("env_overrides_file", _ENV_OVERRIDES_FILE),
      ]
      if fh.present
  }


def _reward_scales(env_cfg) -> dict:
  """Return the env's reward-term scales, or {} if the env has none.

  Generic across environments: only reads the standard
  ``reward_config.scales`` location and tolerates its absence.
  """
  reward_config = env_cfg.get("reward_config", None)
  if reward_config is None:
    return {}
  scales = reward_config.get("scales", None)
  if scales is None:
    return {}
  return {k: float(v) for k, v in scales.items()}


def log_reward_scale_composition(env_cfg) -> None:
  """Log a one-shot stacked-bar plot of the reward-term scales to W&B.

  Each term is a coloured segment whose height is proportional to its scale;
  rewards stack upward from 0 and penalties stack downward, so the figure shows
  the composition of the reward budget at a glance. Static: logged once at
  init and never updated during the run.
  """
  scales = _reward_scales(env_cfg)
  if not scales:
    return

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  terms = list(scales.keys())
  cmap = plt.get_cmap("tab20")
  colors = [cmap(i % 20) for i in range(len(terms))]

  fig, ax = plt.subplots(figsize=(8, 2))
  pos_base, neg_base = 0.0, 0.0
  for term, color in zip(terms, colors):
    width = scales[term]
    base = pos_base if width >= 0 else neg_base
    ax.barh(0, width, left=base, height=0.5, color=color,
            label=f"{term} ({width:+g})")
    if width >= 0:
      pos_base += width
    else:
      neg_base += width

  ax.axvline(0, color="black", linewidth=0.8)
  ax.set_ylim(-0.5, 0.5)
  ax.set_yticks([])
  ax.set_xlabel("reward scale")
  ax.set_title("Reward-term scale composition")
  ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
  fig.tight_layout()

  wandb.log({"reward_scales/composition": wandb.Image(fig)}, step=0)
  plt.close(fig)


def make_eval_video_logger(env, episode_length: int, seed: int, render_every: int = 2):
  """Returns a policy_params_fn callback that logs a rollout video to W&B on every eval."""
  fps = 1.0 / env.dt / render_every
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  rng = jax.random.PRNGKey(seed)

  def log_video(current_step: int, make_policy, params) -> None:
    nonlocal rng
    rng, rollout_rng = jax.random.split(rng)
    reset_rng, step_rng = jax.random.split(rollout_rng)

    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    state = jax.jit(env.reset)(reset_rng)
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def step_fn(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      return (state, key), traj_data

    _, traj = jax.lax.scan(step_fn, (state, step_rng), None, length=episode_length)
    traj_list = [
        jax.tree.map(lambda x, j=j: x[j], traj) for j in range(episode_length)
    ]
    frames = env.render(
        traj_list[::render_every], height=480, width=640, scene_option=scene_option
    )
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
      tmp_path = f.name
    try:
      media.write_video(tmp_path, np.array(frames), fps=fps)
      wandb.log(
          {"eval/rollout": wandb.Video(tmp_path, fps=fps, format="mp4")},
          step=current_step,
      )
    finally:
      os.unlink(tmp_path)

  return log_video


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  if _IMPL.present:
    env_cfg["impl"] = _IMPL.value

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
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

  env_overrides = None
  if _ENV_OVERRIDES_FILE.value:
    with open(_ENV_OVERRIDES_FILE.value, "r", encoding="utf-8") as f:
      env_overrides = yaml.safe_load(f)

  env = registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=env_overrides)
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value
  if _NUM_RESETS_PER_EVAL.present:
    ppo_params.num_resets_per_eval = _NUM_RESETS_PER_EVAL.value

  flag_overrides = _collect_flag_overrides()

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")
  print(f"CLI overrides: {json.dumps(flag_overrides, indent=2)}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})
    if flag_overrides:
      wandb.config.update({"run_overrides": flag_overrides})
    # Static, one-shot reward-budget overview (not updated during the run).
    log_reward_scale_composition(env_cfg)

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

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
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
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
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
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
      wrap_env_fn=None if _VISION.value else functools.partial(
          wrapper.wrap_for_brax_training, full_reset=_FULL_RESET.value
      ),
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]

  # Abort runs that diverge unrecoverably so the next seed can start sooner.
  # The reward/KL/v_loss heuristics rely on episode/* metrics, which brax only
  # emits when log_training_metrics is enabled; the NaN/Inf guard works either
  # way. See learning/early_stop.py.
  early_stopper = early_stop.EarlyStopper(ppo_params.num_timesteps)
  if not _LOG_TRAINING_METRICS.value:
    print(
        "[early-stop] log_training_metrics is off; divergence detection is"
        " limited to the NaN/Inf guard. Pass --log_training_metrics for full"
        " KL / v_loss / reward-collapse detection."
    )

  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value and 'eval/episode_reward' in metrics:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

    # Abort the run if it has diverged unrecoverably (checked after logging so
    # the triggering metrics still land in W&B / TensorBoard).
    if not _PLAY_ONLY.value and _EARLY_STOP.value:
      reason = early_stopper.update(num_steps, metrics)
      if reason is not None:
        raise early_stop.EarlyStopException(reason)

  # Load evaluation environment. Build it from a FRESH default config (env_cfg was
  # mutated in place by the train env's config_overrides) so eval overrides can
  # differ from training -- e.g. disabling a training-only curriculum so eval
  # measures the real, full-difficulty task instead of the easiest level.
  eval_env = None
  if not _VISION.value:
    eval_overrides = dict(env_overrides or {})
    if _EVAL_ENV_OVERRIDES_FILE.value:
      with open(_EVAL_ENV_OVERRIDES_FILE.value, "r", encoding="utf-8") as f:
        eval_overrides.update(yaml.safe_load(f) or {})
    eval_cfg = registry.get_default_config(_ENV_NAME.value)
    if _IMPL.present:
      eval_cfg["impl"] = _IMPL.value
    eval_env = registry.load(
        _ENV_NAME.value, config=eval_cfg, config_overrides=eval_overrides
    )
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          # randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  if _LOG_VIDEO_TO_WANDB.value and _USE_WANDB.value and not _PLAY_ONLY.value and not _VISION.value:
    _video_logger = make_eval_video_logger(
        eval_env,
        episode_length=ppo_params.episode_length,
        seed=_SEED.value,
    )
    _base_policy_params_fn = policy_params_fn
    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=function-redefined
      _base_policy_params_fn(current_step, make_policy, params)
      _video_logger(current_step, make_policy, params)

  # Train or load the model
  try:
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )
  except early_stop.EarlyStopException as e:
    print(f"\n[early-stop] Aborting run: {e}")
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.run.summary["early_stop_reason"] = str(e)
      wandb.finish(exit_code=1)
    # Non-zero exit so a sweep runner treats this seed as failed and moves on.
    sys.exit(1)

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  if _VISION.value:
    reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'rollout{i}.mp4'.")


if __name__ == "__main__":
  app.run(main)
