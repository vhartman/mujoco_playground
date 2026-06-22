"""Shared engine for the pinch experimentation scripts.

Centralises what every script here needs: JAX/MuJoCo env setup, repo paths, env-
config loading from a checkpoint, policy restore (brax PPO with num_timesteps=0),
and queue-run resolution from status.json.

Import this module FIRST in each script -- it sets MUJOCO_GL / XLA env vars and the
brax_compat path before brax is imported.
"""

import os
import sys
import json
import functools
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# experimentation/pinch_plotting/rollout_lib.py -> repo root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "rollout_data"

sys.path.insert(0, str(PROJECT_ROOT / "learning"))
import brax_compat  # noqa: F401  -- must precede brax imports

import jax  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import registry, wrapper  # noqa: E402
from mujoco_playground.config import manipulation_params  # noqa: E402

# info channels that are bookkeeping / nested, not analysis signal.
DROP_INFO = {
    "rng", "pert_dir", "pert_vel", "last_pert_step", "pert_duration_steps",
    "pert_wait_steps", "obs_bias",
}


def load_env_cfg(run_dir: Path, **overrides):
    """(env_name, cfg) from a run's saved checkpoints/config.json, plus optional
    config overrides (e.g. remove_cube=True, cube_size_scale=1.1)."""
    env_name = run_dir.name.split("-")[0]
    cfg = registry.get_default_config(env_name)
    for k, v in json.load(open(run_dir / "checkpoints" / "config.json")).items():
        try:
            cfg[k] = v
        except Exception:
            pass  # ConfigDict rejects keys not in the schema
    for k, v in overrides.items():
        cfg[k] = v
    return env_name, cfg


def latest_ckpt(run_dir: Path):
    ckpts = sorted([d for d in (run_dir / "checkpoints").iterdir() if d.is_dir()],
                   key=lambda d: int(d.name))
    return ckpts[-1] if ckpts else None


def restore_policy(env_name, cfg, restore_path, deterministic=True):
    """Restore a trained policy. Returns (env, jitted inference_fn)."""
    ppo_params = manipulation_params.brax_ppo_config(env_name, cfg.get("impl", "warp"))
    ppo_params.num_timesteps = 0
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    training_params = {k: v for k, v in dict(ppo_params).items()
                       if k not in ("network_factory", "num_eval_envs")}
    env = registry.load(env_name, config=cfg)
    make_inf, params, _ = ppo.train(
        environment=env, **training_params, network_factory=network_factory, seed=1,
        restore_checkpoint_path=restore_path, save_checkpoint_path=None,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.get("num_eval_envs", 128),
        progress_fn=lambda *_: None, policy_params_fn=lambda *_: None,
        eval_env=registry.load(env_name, config=cfg),
    )
    return env, jax.jit(make_inf(params, deterministic=deterministic))


def queue_runs(queue_name: str, min_ckpts: int = 1):
    """Healthy run-dir names recorded in a queue's status.json (>= min_ckpts
    checkpoints; raises a diverged/early-stopped run above min_ckpts=1)."""
    status = json.load(open(LOGS_DIR / "_queue" / queue_name / "status.json"))
    out = []
    for r in status:
        name = r["exp_name"]
        ckpt_dir = LOGS_DIR / name / "checkpoints"
        n = len(list(ckpt_dir.glob("0*"))) if ckpt_dir.is_dir() else 0
        if n >= min_ckpts:
            out.append(name)
    return out
