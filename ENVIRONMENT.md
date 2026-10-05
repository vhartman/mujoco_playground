# Environment

The exact software environment used for the Tesollo hand experiments (in-hand rotation
`TesolloDownwardsRotateZ`, force tracking `TesolloCubePinch`). Results in RL are sensitive
to simulator and JAX versions, so install from the pinned versions below, not the floating
git sources in `pyproject.toml`.

## Tested setup

| | |
|---|---|
| OS | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA RTX 3090 (24 GB), CUDA 12 via pip wheels (no system CUDA needed) |
| Python | 3.12 |
| Package manager | [uv](https://docs.astral.sh/uv/) |

Key versions (complete list: [`requirements-lock.txt`](requirements-lock.txt)):

| Package | Version |
|---|---|
| mujoco | 3.11.1.dev962197295 (nightly, from https://py.mujoco.org) |
| mujoco-mjx | 3.8.0 |
| mujoco-warp | 3.7.0.1 |
| warp-lang | 1.12.1 |
| jax / jaxlib | 0.10.1 (+ jax-cuda12-plugin 0.10.1) |
| brax | 0.14.2 |
| flax | 0.12.0 |
| orbax-checkpoint | 0.11.36 |

## Install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

# 1. The package with pinned dependencies. --no-sources ignores the floating git
#    sources in pyproject.toml; requirements-lock.txt pins every transitive dependency.
uv pip install --no-sources --override requirements-lock.txt \
  --extra-index-url https://py.mujoco.org --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  -e . wandb matplotlib pyyaml imageio-ffmpeg

# 2. Package extras. This must be a separate step: --override drops extras, so
#    [with-cuda] (the nvidia-* CUDA wheels for JAX) and wandb[media] (video/plot
#    logging) would be silently skipped above.
uv pip install "jax-cuda12-plugin[with-cuda]==0.10.1" "jax-cuda12-pjrt==0.10.1" "wandb[media]==0.30.0"

# 3. ffmpeg for the post-training rollout video, if no system ffmpeg is installed.
ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg
```

Check that JAX sees the GPU. If this prints `CpuDevice`, step 2 did not install the CUDA libraries,
and training will fail with `Unknown backend cuda`:

```bash
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

The `mujoco` nightly wheels on py.mujoco.org are pruned over time. If
`3.11.1.dev962197295` is no longer available, the 3.8.0 release compiles an identical
model for these scenes (checked on `TesolloDownwardsRotateZ`: all physical parameters match).

## Runtime settings

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab the whole GPU up front
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # make CUDA device indices match nvidia-smi
```

Weights & Biases logging (`--use_wandb`) needs `WANDB_ENTITY` / `WANDB_PROJECT`, or a logged-in default.

## Training

Single run:

```bash
python learning/train_jax_ppo.py --env_name=TesolloDownwardsRotateZ --num_timesteps=1000000000 \
  --nodomain_randomization --env_overrides_file=overrides.yaml
```

Sweeps are described as queue YAMLs and run sequentially with `learning/run_queue.py --queue <file> --yes`.
A 1B-step `TesolloDownwardsRotateZ` run takes about 3 hours on one RTX 3090 at 8192 environments
and uses about 17 GB of GPU memory.
