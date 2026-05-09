# Training Parameter Reference

All parameters for `learning/train_jax_ppo.py`. Defaults shown are script defaults; env-specific configs (loaded via `manipulation_params.brax_ppo_config`) may override them.

---

## Generic Training Parameters

These control experiment management, logging, and evaluation — not the algorithm or environment.

| Flag | Default | Description |
|---|---|---|
| `--env_name` | `LeapCubeReorient` | Which registered environment to train on. |
| `--impl` | `jax` | MJX physics backend: `jax` (pure JAX) or `warp` (CUDA-accelerated, faster on GPU). |
| `--vision` | `False` | Use pixel observations instead of state. Switches to a vision-specific network and wraps the env differently. |
| `--seed` | `1` | Global random seed for JAX and environment resets. |
| `--num_timesteps` | `1_000_000` | Total environment steps to train for. |
| `--load_checkpoint_path` | `None` | Path to a checkpoint directory or specific checkpoint to resume from. If a directory, automatically picks the latest checkpoint inside. |
| `--suffix` | `None` | Appended to the auto-generated experiment name (`<EnvName>-<timestamp>[-suffix]`). Useful for tagging sweep variants. |
| `--play_only` | `False` | Skip training entirely; just load the checkpoint and record rollouts. Sets `num_timesteps=0`. |
| `--use_wandb` | `False` | Enable Weights & Biases logging. Logs all metrics and env config at `wandb.log()` per eval. |
| `--use_tb` | `False` | Enable TensorBoard logging to the run's log directory. |
| `--domain_randomization` | `False` | Apply the env-specific domain randomization function during training (e.g. random mass, friction). |
| `--num_videos` | `1` | Number of post-training rollout videos to render and save as `rollout{i}.mp4`. |
| `--num_evals` | `5` | How many evaluation checkpoints to run during training (evenly spaced over `num_timesteps`). |
| `--run_evals` | `True` | Whether to actually execute eval rollouts at eval checkpoints. Disable to skip evals and speed up training. |
| `--log_training_metrics` | `False` | Log per-step training metrics (episode reward, KL, losses) in addition to eval metrics. Significantly slows training if `training_metrics_steps` is too small. |
| `--training_metrics_steps` | `1_000_000` | Steps between training metric logs. Increase to reduce overhead when `--log_training_metrics` is on. |
| `--rscope_envs` | `None` | If set, saves live rollouts for the rscope interactive viewer. Number of parallel envs to roll out. |
| `--deterministic_rscope` | `True` | Whether rscope rollouts use a deterministic policy (no action noise). |

---

## PPO Parameters

These control the PPO algorithm: data collection, gradient updates, and network architecture.

### Data Collection

| Flag | Default | Description |
|---|---|---|
| `--num_envs` | `1024` | Number of parallel environments for data collection. Higher = larger effective batch size and more stable gradients, but more VRAM. |
| `--num_eval_envs` | `128` | Parallel environments used during evaluation. Does not affect training. |
| `--episode_length` | `1000` | Maximum steps per episode. Also controls the rollout horizon combined with `unroll_length`. |
| `--action_repeat` | `1` | How many physics steps to repeat each action. Effectively multiplies `ctrl_dt` from the env's perspective. Higher values reduce the policy's control frequency. |
| `--unroll_length` | `10` | Number of steps collected per environment before a gradient update (rollout chunk length). Total data per update = `num_envs × unroll_length`. |

### Optimization

| Flag | Default | Description |
|---|---|---|
| `--learning_rate` | `5e-4` | Adam learning rate. Typical range: `1e-4`–`5e-4`. Too high causes divergence; too low slows learning. |
| `--num_minibatches` | `8` | Number of minibatches to split each rollout batch into for gradient updates. Effective minibatch size = `(num_envs × unroll_length) / num_minibatches`. |
| `--num_updates_per_batch` | `8` | Number of gradient update passes over the same batch of data (PPO epochs). More epochs reuse data but risk overfitting / high KL divergence. |
| `--batch_size` | `256` | Minibatch size for gradient computation. Should divide `(num_envs × unroll_length) / num_minibatches` evenly. |
| `--max_grad_norm` | `1.0` | Gradient clipping threshold (L2 norm). Prevents exploding gradients. Reduce if training is unstable. |
| `--discounting` | `0.97` | Discount factor γ for future rewards. For 1000-step episodes: `0.97`–`0.99`. Lower = more myopic; higher = longer credit assignment horizon. |
| `--reward_scaling` | `0.1` | Scalar multiplied onto rewards before computing advantages. Acts as an effective learning rate on the value function. If raw rewards are large (>100), increase this or expect slow value learning. |
| `--clipping_epsilon` | `0.2` | PPO surrogate clip ratio. Limits how much the policy can change per update. Typical range: `0.1`–`0.3`. |
| `--entropy_cost` | `5e-3` | Coefficient on the entropy bonus added to the policy loss. Higher = more exploration, slower convergence. Too low causes premature collapse to a deterministic policy. |
| `--normalize_observations` | `True` | Online running mean/std normalization of observations. Almost always beneficial; disable only for debugging. |

### Network Architecture

Set via flags; also reflected in `checkpoints/<step>/ppo_network_config.json`.

| Flag / Field | Default | Description |
|---|---|---|
| `--policy_hidden_layer_sizes` | `[64, 64, 64]` | MLP layer widths for the policy network. Env-specific configs typically use `[256, 128, 64]`. |
| `--value_hidden_layer_sizes` | `[64, 64, 64]` | MLP layer widths for the value network. Should be at least as wide as the policy given privileged observations. |
| `--policy_obs_key` | `state` | Which observation dict key the policy reads. For asymmetric actor-critic, policy uses `state` (no privileged info). |
| `--value_obs_key` | `state` | Which observation dict key the value function reads. Set to `privileged_state` to give the critic extra info not available at deployment. |
| `activation` | `silu` | Activation function in both networks (`silu`, `relu`, `tanh`, etc.). |
| `distribution_type` | `tanh_normal` | Policy output distribution. `tanh_normal` squashes actions to `[-1, 1]` before `action_scale`. |
| `init_noise_std` | `1.0` | Initial standard deviation of the action distribution. Higher = more exploration at the start. Too low → premature convergence. |
| `noise_std_type` | `scalar` | Whether the std is a single shared scalar or per-dimension (`diagonal`). |
| `state_dependent_std` | `False` | If `True`, the network outputs both mean and std as functions of the observation. Increases expressivity but can be harder to train. |
| `use_distributional_critic` | `False` | Use a distributional value function (IQN-style quantile regression) instead of a scalar. Useful for multi-modal return distributions. |
| `num_quantiles` | `32` | Number of quantiles for the distributional critic (only relevant if `use_distributional_critic=True`). |
| `policy_network_kernel_init_fn` | `lecun_uniform` | Weight initializer for the policy MLP (`lecun_uniform`, `glorot_uniform`, `orthogonal`, etc.). |
| `value_network_kernel_init_fn` | `lecun_uniform` | Weight initializer for the value MLP. |

---

## Env Parameters

Stored in `logs/<run>/checkpoints/config.json` and logged to WandB config. Set by modifying the env config before calling `registry.load()`.

### Simulation

| Parameter | Example | Description |
|---|---|---|
| `impl` | `warp` | Physics backend for this env (`jax` or `warp`). |
| `sim_dt` | `0.01` | Physics simulation timestep in seconds. Smaller = more accurate contacts but slower simulation. |
| `ctrl_dt` | `0.05` | Control timestep (policy frequency). Must be a multiple of `sim_dt`. `ctrl_dt / sim_dt` gives sub-steps per action. |
| `nconmax` | `1638400` | Maximum number of contacts across all parallel envs. Increase if getting contact buffer overflow errors. |
| `njmax` | `1024` | Maximum number of constraint rows per env. Warp backend uses this for constraint solver allocation. |
| `episode_length` | `1000` | Steps per episode (in control steps, not physics steps). |
| `action_repeat` | `1` | Physics steps per policy action (duplicate of the PPO flag; the env config value wins). |
| `action_scale` | `0.5` | Scalar multiplied onto the policy's raw action output before sending to the controller. |
| `ema_alpha` | `1.0` | EMA smoothing on actions sent to actuators. `1.0` = no smoothing (raw actions). Lower = smoother but more lagged control. |

### Success Criterion

| Parameter | Example | Description |
|---|---|---|
| `force_target` | `10.0` | Target grasp force in Newtons that must be maintained for success. |
| `force_tolerance` | `0.5` | Tolerance around `force_target`; grasp is considered successful if `|force - force_target| < force_tolerance`. |
| `success_hold_time` | `3.0` | Seconds the success condition must be continuously satisfied before the episode is marked as a success. At `ctrl_dt=0.05`, this is 60 consecutive steps. |

### Observation Noise

Applied to the robot's sensed state before feeding to the policy. Simulates real-world sensor noise for sim-to-real transfer.

| Parameter | Example | Description |
|---|---|---|
| `obs_noise.level` | `1.0` | Global multiplier on all noise scales. Set to `0.0` to disable all noise (useful for debugging). |
| `obs_noise.scales.joint_pos` | `0.001` | Noise std on joint position observations (radians). |
| `obs_noise.scales.joint_vel` | `0.01` | Noise std on joint velocity observations (rad/s). |
| `obs_noise.scales.cube_pos` | `0.005` | Noise std on object position (meters). |
| `obs_noise.scales.cube_quat` | `0.02` | Noise std on object orientation (quaternion components). |

### Perturbations

Random external forces applied to the object mid-episode for robustness training. Disabled by default.

| Parameter | Example | Description |
|---|---|---|
| `pert_config.enable` | `false` | Whether to apply random perturbations during training. Enable for robustness / curriculum stage 2+. |
| `pert_config.linear_velocity_pert` | `[0.0, 3.0]` | Min/max magnitude of random linear impulse applied to the object (m/s). |
| `pert_config.angular_velocity_pert` | `[0.0, 0.5]` | Min/max magnitude of random angular impulse applied to the object (rad/s). |
| `pert_config.pert_duration_steps` | `[1, 100]` | Duration range of each perturbation event (in control steps). |
| `pert_config.pert_wait_steps` | `[60, 150]` | Steps between perturbation events (in control steps). |

### Reward Scales

Defined under `reward_config.scales`. Each component's raw value is multiplied by its scale and summed to produce the step reward.

| Component | Example Scale | Sign | What it rewards |
|---|---|---|---|
| `cube_force` | `5.0` | + | Contact force on the cube — primary task signal. Large positive scale drives the grasp. |
| `success` | `10.0` (via `success_reward`) | + | Binary bonus given each step the success condition is met. |
| `fingertip_pos` | `0.1` | + | Reward for fingertips being close to the cube surface. Shaping signal to encourage pre-grasp approach. |
| `termination` | `-100.0` | − | Large penalty on episode termination (drop or constraint violation). Discourages risky behavior. |
| `wrist_pose` | `-0.5` | − | Penalizes wrist deviation from a reference pose. Keeps the wrist in a natural configuration. |
| `hand_pose` | `-0.2` | − | Penalizes overall hand joint deviation from a rest pose. Regularizes to a natural-looking grasp. |
| `wrist_vel` | `-0.1` | − | Penalizes wrist velocity. Encourages smooth, stable wrist motion. |
| `cube_lin_vel` | `-0.1` | − | Penalizes linear velocity of the cube. Encourages holding the cube still rather than spinning it. |
| `cube_ang_vel` | `-0.1` | − | Penalizes angular velocity of the cube. Same intent as `cube_lin_vel`. |
| `action_rate` | `-0.005` | − | Penalizes change in action between steps (`|a_t - a_{t-1}|`). Encourages smooth control signals. |
| `joint_vel` | `-0.01` | − | Penalizes overall joint velocity. Discourages fast, jerky motions. |
| `energy` | `-0.001` | − | Penalizes joint torques × velocities (mechanical power). Encourages energy-efficient grasping. |
