"""Find the largest num_envs that fits in GPU memory for TesolloPinch.

Usage:
    python learning/find_max_envs.py
    python learning/find_max_envs.py --env_name TesolloPinch --candidates 64,256,512,1024,2048

Sets TF_GPU_ALLOCATOR=cuda_malloc_async and unsets the Triton autotuner flag
automatically — these are required for the BFC-fragmentation workaround.
"""
import argparse
import gc
import os
import subprocess
import sys

# Must come before any JAX/brax import.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
# Strip the Triton flag that causes autotuner OOM on low-VRAM GPUs.
xla_flags = os.environ.get("XLA_FLAGS", "").replace("--xla_gpu_triton_gemm_any=True", "").strip()
os.environ["XLA_FLAGS"] = xla_flags

sys.path.insert(0, os.path.dirname(__file__))
import brax_compat  # noqa: F401, E402 — patches jax.device_put_replicated

import functools  # noqa: E402

import jax  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import registry, wrapper  # noqa: E402
from mujoco_playground.config import manipulation_params  # noqa: E402


def gpu_mb() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip())


def try_train(env_name: str, num_envs: int) -> tuple[bool, int]:
    """Return (success, peak_gpu_mb)."""
    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg)
    eval_env = registry.load(env_name, config=env_cfg)

    ppo_params = manipulation_params.brax_ppo_config(env_name, "jax")
    ppo_params.num_envs = num_envs
    ppo_params.num_eval_envs = num_envs
    ppo_params.batch_size = min(num_envs, 256)
    ppo_params.unroll_length = 10
    ppo_params.num_minibatches = max(4, num_envs // 64)
    ppo_params.num_timesteps = 500
    ppo_params.run_evals = False

    training_params = dict(ppo_params)
    del training_params["network_factory"]

    try:
        functools.partial(
            ppo.train,
            **training_params,
            network_factory=ppo_networks.make_ppo_networks,
            seed=1,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )(environment=env, progress_fn=lambda *_: None, eval_env=eval_env)
        return True, gpu_mb()
    except Exception as e:
        return False, gpu_mb()
    finally:
        del env, eval_env
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="TesolloPinch")
    parser.add_argument(
        "--candidates",
        default="64,128,256,512,1024,2048,4096",
        help="Comma-separated list of num_envs values to test (ascending order)",
    )
    args = parser.parse_args()

    candidates = [int(x) for x in args.candidates.split(",")]

    print(f"GPU baseline: {gpu_mb()} MB")
    print(f"Env: {args.env_name}")
    print(f"Candidates: {candidates}\n")
    print(f"{'num_envs':>10}  {'result':>8}  {'peak GPU':>10}")
    print("-" * 34)

    last_ok = None
    for n in candidates:
        ok, peak = try_train(args.env_name, n)
        status = "OK" if ok else "OOM"
        print(f"{n:>10}  {status:>8}  {peak:>8} MB")
        if ok:
            last_ok = n
        else:
            break

    print()
    if last_ok is not None:
        print(f"Recommended num_envs for this GPU: {last_ok}")
    else:
        print("No candidate succeeded — check GPU memory / env config.")


if __name__ == "__main__":
    main()
