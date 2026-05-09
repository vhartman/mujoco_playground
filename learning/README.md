# Learning RL Agents

In this directory, we demonstrate learning RL agents from MuJoCo Playground environments using [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We provide two entrypoints from the command line: `python train_jax_ppo.py` and `python train_rsl_rl.py`.

For more detailed tutorials on using MuJoCo Playground for RL, see:

1. Intro. to the Playground with DM Control Suite [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. Locomotion Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. Manipulation Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. Training CartPole from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. Robotic Manipulation from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## Training with brax PPO

To train with brax PPO, you can use the `train_jax_ppo.py` script. This script uses the brax PPO algorithm to train an agent on a given environment.

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

To train a vision-based policy using pixel observations:
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

Use `python train_jax_ppo.py --help` to see possible options and usage. Logs and checkpoints are saved in `logs` directory.

## Training with RSL-RL

To train with RSL-RL, you can use the `train_rsl_rl.py` script. This script uses the RSL-RL algorithm to train an agent on a given environment.

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

To render the behaviour from the resulting policy:
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

where `run_name` is the name of the run you want to load (will be printed in the console when the training run is started).

Logs and checkpoints are saved in `logs` directory.

## Training queues

`run_queue.py` reads a YAML queue file and executes each run as a subprocess
of `train_jax_ppo.py`, sequentially. Runs are isolated (full process teardown
between them). Failed runs are logged and skipped; the queue continues.

```bash
python learning/run_queue.py --queue learning/queues/example.yaml
```

Flags:

| Flag | Description |
|---|---|
| `--queue PATH` | Queue YAML to execute (required). |
| `--dry-run` | Print each command without running it. |
| `--start-from N` | Skip the first N entries (resume a partial queue). |
| `--yes` / `-y` | Skip the confirmation prompt. |

Each run's stdout/stderr is tee'd to `logs/_queue/<queue>-<timestamp>/run-NN-*.log`.
A `status.json` (updated after each run) records exit codes, timing, and the
path to the experiment directory produced by the trainer.

### Queue YAML format

```yaml
defaults:            # applied to every run; per-run values override these
  flags:
    use_wandb: true
  env_overrides:
    obs_noise.level: 1.0

runs:
  - flags:
      env_name: TesolloPinch
      suffix: baseline
      seed: 1
      num_timesteps: 50_000_000

  - flags:
      env_name: TesolloPinch
      suffix: kp5
      seed: 1
      num_timesteps: 50_000_000
    env_overrides:
      pid_gains.enable: true
      pid_gains.finger_kp: 5.0
```

`flags` maps 1-to-1 to `train_jax_ppo.py` CLI flags.  
`env_overrides` uses dotted keys matching `pinch.default_config()` fields and
is forwarded to the env via the existing `config_overrides` mechanism.

### Controllable env parameters (TesolloPinch)

| Key | Default | Description |
|---|---|---|
| `obs_noise.level` | `1.0` | Global noise multiplier (0 = clean obs). |
| `obs_noise.scales.joint_pos` | `0.001` | Noise scale for joint positions. |
| `obs_noise.scales.joint_vel` | `0.01` | Noise scale for joint velocities. |
| `obs_noise.scales.cube_pos` | `0.005` | Noise scale for cube position. |
| `obs_noise.scales.cube_quat` | `0.02` | Noise scale for cube quaternion. |
| `pid_gains.enable` | `false` | Must be `true` to apply any gain overrides. |
| `pid_gains.finger_kp` | `3.0` | kp broadcast to all 20 finger actuators. |
| `pid_gains.finger_kv` | `0.0` | kv broadcast to all 20 finger actuators. |
| `pid_gains.wrist_kp` | `[10, 75, 10]` | kp per wrist joint (3 values). |
| `pid_gains.wrist_kv` | `[2, 10, 2]` | kv per wrist joint (3 values). |
| `pid_gains.kp_per_actuator` | `[]` | Full length-23 kp override (takes priority). |
| `pid_gains.kv_per_actuator` | `[]` | Full length-23 kv override (takes priority). |
| `force_target` | `10.0` | Target pinch force in Newtons. |
