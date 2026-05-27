"""
profile_training.py — measure the key stages of a mujoco_playground training
loop and predict total wall-clock time.

Stages measured
───────────────
  reset            vmapped env.reset over num_envs
  step             vmapped env.step with zero actions (pure physics + reward + obs cost)
  rollout          lax.scan over unroll_length with zero actions (physics throughput)
  inference        policy network forward pass over num_envs
  rollout+policy   lax.scan with real policy inference (actual PPO collection cost)

Proposed stages (see --proposed flag or bottom of output)
─────────────────────────────────────────────────────────
  ppo_update, gae, obs_norm, eval_rollout, contact_detect, host_sync

Usage
─────
  uv run python experimentation/profile_training.py
  uv run python experimentation/profile_training.py --env_name TesolloPickAndPlaceProprio
  uv run python experimentation/profile_training.py \\
      --env TesolloPickAndPlaceProprio --num_envs 1024 --num_timesteps 50_000_000
"""

import argparse
import functools
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "learning"))
import brax_compat  # noqa: F401

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)

# ─── PPO param lookup ─────────────────────────────────────────────────────────

def _get_ppo_params(env_name: str, impl: str):
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name, impl)
    raise ValueError(f"Unknown env: {env_name}")


# ─── timing helper ────────────────────────────────────────────────────────────

def _time_fn(fn, n_warmup: int = 3, n_iters: int = 20):
    """
    Returns (jit_s, mean_ms, std_ms).
    fn() must return a JAX pytree for block_until_ready to work correctly.
    """
    t0 = time.monotonic()
    jax.block_until_ready(fn())
    jit_s = time.monotonic() - t0

    for _ in range(n_warmup):
        jax.block_until_ready(fn())

    elapsed = []
    for _ in range(n_iters):
        t0 = time.monotonic()
        jax.block_until_ready(fn())
        elapsed.append(time.monotonic() - t0)

    return jit_s, float(np.mean(elapsed)) * 1e3, float(np.std(elapsed)) * 1e3


# ─── obs helpers ──────────────────────────────────────────────────────────────

def _obs_size_for_key(env, key: str) -> int:
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        return int(obs_size[key])
    return int(obs_size)


def _extract_obs(obs: Any, key: str):
    if isinstance(obs, dict):
        return obs[key]
    return obs


# ─── stage profilers ──────────────────────────────────────────────────────────

def profile_reset(env, num_envs: int, rng: jax.Array, n_iters: int = 20):
    keys = jax.random.split(rng, num_envs)
    fn = jax.jit(jax.vmap(env.reset))
    jit_s, mean_ms, std_ms = _time_fn(lambda: fn(keys), n_iters=n_iters)
    throughput = num_envs / (mean_ms * 1e-3)
    return jit_s, mean_ms, std_ms, f"{throughput:.0f} resets/s"


def profile_step(env, num_envs: int, rng: jax.Array, n_iters: int = 20):
    keys = jax.random.split(rng, num_envs)
    states = jax.block_until_ready(jax.jit(jax.vmap(env.reset))(keys))
    actions = jp.zeros((num_envs, env.action_size))
    fn = jax.jit(jax.vmap(env.step))
    jit_s, mean_ms, std_ms = _time_fn(lambda: fn(states, actions), n_iters=n_iters)
    throughput = num_envs / (mean_ms * 1e-3)
    return jit_s, mean_ms, std_ms, f"{throughput / 1e3:.1f}K steps/s"


def profile_rollout(env, num_envs: int, unroll_length: int, rng: jax.Array, n_iters: int = 10):
    """lax.scan rollout with zero actions — isolates physics throughput."""
    keys = jax.random.split(rng, num_envs)
    init_states = jax.block_until_ready(jax.jit(jax.vmap(env.reset))(keys))

    @jax.jit
    def rollout(init_states):
        def step(states, _):
            actions = jp.zeros((num_envs, env.action_size))
            return jax.vmap(env.step)(states, actions), None
        final, _ = jax.lax.scan(step, init_states, None, length=unroll_length)
        return final.reward

    jit_s, mean_ms, std_ms = _time_fn(lambda: rollout(init_states), n_iters=n_iters)
    total_steps = num_envs * unroll_length
    throughput = total_steps / (mean_ms * 1e-3)
    return jit_s, mean_ms, std_ms, f"{throughput / 1e6:.2f}M steps/s"


def profile_inference(env, ppo_params, num_envs: int, rng: jax.Array, n_iters: int = 20):
    """Policy network forward pass over num_envs observations."""
    policy_obs_key = ppo_params.network_factory.get("policy_obs_key", "state")
    obs_size = _obs_size_for_key(env, policy_obs_key)

    net_kwargs = {k: v for k, v in dict(ppo_params.network_factory).items()
                  if k not in ("policy_obs_key", "value_obs_key")}
    nets = ppo_networks.make_ppo_networks(obs_size, env.action_size, **net_kwargs)

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jp.zeros((num_envs, obs_size))
    params = nets.policy_network.init(init_rng, dummy_obs)
    fn = jax.jit(nets.policy_network.apply)

    jit_s, mean_ms, std_ms = _time_fn(lambda: fn(params, dummy_obs), n_iters=n_iters)
    throughput = num_envs / (mean_ms * 1e-3)
    return jit_s, mean_ms, std_ms, f"{throughput / 1e3:.1f}K infer/s"


def profile_rollout_with_policy(
    env, ppo_params, num_envs: int, unroll_length: int, rng: jax.Array, n_iters: int = 10
):
    """lax.scan rollout with real policy — the actual PPO data collection cost."""
    policy_obs_key = ppo_params.network_factory.get("policy_obs_key", "state")
    obs_size = _obs_size_for_key(env, policy_obs_key)

    net_kwargs = {k: v for k, v in dict(ppo_params.network_factory).items()
                  if k not in ("policy_obs_key", "value_obs_key")}
    nets = ppo_networks.make_ppo_networks(obs_size, env.action_size, **net_kwargs)

    rng, k1, k2 = jax.random.split(rng, 3)
    dummy_obs = jp.zeros((num_envs, obs_size))
    params = nets.policy_network.init(k1, dummy_obs)

    keys = jax.random.split(k2, num_envs)
    init_states = jax.block_until_ready(jax.jit(jax.vmap(env.reset))(keys))

    @jax.jit
    def rollout(init_states, params):
        def step(states, _):
            obs = _extract_obs(states.obs, policy_obs_key)
            # policy_network.apply returns raw output (mean ++ log_std); take mean as action
            raw = nets.policy_network.apply(params, obs)
            actions = raw[..., :env.action_size]
            return jax.vmap(env.step)(states, actions), states.reward
        _, rewards = jax.lax.scan(step, init_states, None, length=unroll_length)
        return rewards

    jit_s, mean_ms, std_ms = _time_fn(lambda: rollout(init_states, params), n_iters=n_iters)
    total_steps = num_envs * unroll_length
    throughput = total_steps / (mean_ms * 1e-3)
    return jit_s, mean_ms, std_ms, f"{throughput / 1e6:.2f}M steps/s"


# ─── formatting ───────────────────────────────────────────────────────────────

def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    elif s < 3600:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def _print_table(headers, rows):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "┼".join("─" * (w + 2) for w in widths)
    row_fmt = "│".join(f" {{:<{w}}} " for w in widths)
    print(row_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(row_fmt.format(*[str(x) for x in row]))


# ─── proposed stages description ─────────────────────────────────────────────

_PROPOSED_STAGES = """\
Proposed additional monitoring stages (require instrumentation inside brax or env):

  ppo_update      Minibatch gradient sweep: num_minibatches × num_updates_per_batch
                  optimizer steps. Typically 10–30% of rollout cost for physics-
                  heavy envs. Estimate by running a single full train step with
                  num_timesteps = num_envs × unroll_length and timing it.

  gae             Generalized advantage estimation over the collected buffer. Pure
                  lax.scan over unroll_length; usually cheap but worth verifying
                  when unroll_length is large.

  obs_norm        RunningMeanVariance update applied to every observation in the
                  batch. Negligible for flat obs (<200 dims); can matter for
                  image observations.

  eval_rollout    Full-episode rollout on num_eval_envs, run num_evals times.
                  Estimate: scale rollout+policy result by
                  (episode_length / unroll_length) × (num_eval_envs / num_envs).

  contact_detect  Broad/narrow-phase contact detection inside MJX physics step.
                  Dominant cost for dense-contact grasping envs. Benchmark by
                  comparing step time with nconmax=1 vs production value.

  host_sync       Blocking block_until_ready() called during progress_fn and eval
                  callbacks. Measure by timing the callback itself in train_jax_ppo.
                  Large when eval or video logging is frequent."""


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile mujoco_playground training stages.")
    parser.add_argument("--env", default="TesolloPickAndPlaceProprio",
                        help="Registered environment name")
    parser.add_argument("--impl", default="warp", choices=["warp", "jax"])
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override num_envs from ppo_params")
    parser.add_argument("--unroll_length", type=int, default=None,
                        help="Override unroll_length from ppo_params")
    parser.add_argument("--num_timesteps", type=int, default=None,
                        help="Override num_timesteps for prediction")
    parser.add_argument("--n_iters", type=int, default=20,
                        help="Timing iterations per stage (fewer = faster profiling)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_policy", action="store_true",
                        help="Skip inference and rollout+policy stages (faster)")
    args = parser.parse_args()

    # ── load env and params ──────────────────────────────────────────────────
    print(f"\nLoading env: {args.env} (impl={args.impl}) …", flush=True)
    env_cfg = registry.get_default_config(args.env)
    env_cfg["impl"] = args.impl
    env = registry.load(args.env, config=env_cfg)

    ppo_params = _get_ppo_params(args.env, args.impl)
    num_envs = args.num_envs or ppo_params.num_envs
    unroll_length = args.unroll_length or ppo_params.unroll_length
    num_timesteps = args.num_timesteps or ppo_params.num_timesteps
    episode_length = ppo_params.episode_length
    num_minibatches = ppo_params.num_minibatches
    num_updates_per_batch = ppo_params.num_updates_per_batch
    n_iters_rollout = max(3, args.n_iters // 2)

    print(f"  obs_size:        {env.observation_size}")
    print(f"  action_size:     {env.action_size}")
    print(f"  num_envs:        {num_envs:,}")
    print(f"  unroll_length:   {unroll_length}")
    print(f"  episode_length:  {episode_length}")
    print(f"  num_timesteps:   {num_timesteps:,}")
    print(f"  num_minibatches: {num_minibatches}")
    print(f"  num_updates/batch: {num_updates_per_batch}")

    rng = jax.random.PRNGKey(args.seed)
    n = args.n_iters

    # ── run stages ───────────────────────────────────────────────────────────
    results = {}

    def _run_stage(label, fn, *stage_args, n_iters, **stage_kwargs):
        """Run a stage, halving num_envs on OOM until it fits."""
        nonlocal num_envs
        attempt_envs = num_envs
        while True:
            try:
                print(f"\n{label}", end=" ", flush=True)
                result = fn(env, attempt_envs, *stage_args, n_iters=n_iters, **stage_kwargs)
                if attempt_envs != num_envs:
                    print(f"(reduced to {attempt_envs:,} envs due to OOM) ", end="")
                    num_envs = attempt_envs
                print(f"{result[1]:.1f} ms/call")
                return result
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                    attempt_envs //= 2
                    if attempt_envs < 64:
                        print(f"\n  OOM even at {attempt_envs * 2} envs — skipping stage")
                        return None
                    print(f"\n  OOM at {attempt_envs * 2} envs, retrying with {attempt_envs} …", end=" ", flush=True)
                else:
                    raise

    results["reset"] = _run_stage("[1/5] reset …", profile_reset, rng, n_iters=n)
    rng, _ = jax.random.split(rng)

    results["step"] = _run_stage("[2/5] step (no policy) …", profile_step, rng, n_iters=n)
    rng, _ = jax.random.split(rng)

    def _profile_rollout(env, ne, rng, n_iters):
        return profile_rollout(env, ne, unroll_length, rng, n_iters=n_iters)

    results["rollout"] = _run_stage(
        f"[3/5] rollout (scan {unroll_length} steps, no policy) …",
        _profile_rollout, rng, n_iters=n_iters_rollout,
    )
    rng, _ = jax.random.split(rng)

    if not args.skip_policy:
        def _profile_inference(env, ne, rng, n_iters):
            return profile_inference(env, ppo_params, ne, rng, n_iters=n_iters)

        results["inference"] = _run_stage(
            "[4/5] inference (policy forward pass) …",
            _profile_inference, rng, n_iters=n,
        )
        rng, _ = jax.random.split(rng)

        def _profile_rollout_policy(env, ne, rng, n_iters):
            return profile_rollout_with_policy(env, ppo_params, ne, unroll_length, rng, n_iters=n_iters)

        results["rollout+policy"] = _run_stage(
            f"[5/5] rollout+policy (scan {unroll_length} steps) …",
            _profile_rollout_policy, rng, n_iters=n_iters_rollout,
        )
        rng, _ = jax.random.split(rng)
    else:
        print("\n[4/5] inference — skipped")
        print("[5/5] rollout+policy — skipped")

    # drop stages that were skipped or OOM'd
    results = {k: v for k, v in results.items() if v is not None}

    # ── timing table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"  TIMING  ·  {args.env}  ·  {num_envs:,} envs  ·  unroll={unroll_length}")
    print("═" * 78)
    headers = ["Stage", "JIT (s)", "Mean (ms)", "±Std (ms)", "Throughput"]
    rows = [
        [name, f"{jit:.1f}", f"{mean:.1f}", f"{std:.1f}", tp]
        for name, (jit, mean, std, tp) in results.items()
    ]
    _print_table(headers, rows)

    # ── policy overhead breakdown ─────────────────────────────────────────────
    if "rollout" in results and "rollout+policy" in results:
        physics_ms = results["rollout"][1]
        total_ms = results["rollout+policy"][1]
        policy_ms = max(0.0, total_ms - physics_ms)
        print(f"\n  Physics share:  {100 * physics_ms / total_ms:.1f}%  ({physics_ms:.1f} ms)")
        print(f"  Policy share:   {100 * policy_ms / total_ms:.1f}%  ({policy_ms:.1f} ms)")

    # ── training time prediction ──────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  TRAINING TIME PREDICTION")
    print("─" * 78)

    n_updates = num_timesteps / (num_envs * unroll_length)
    print(f"  Gradient updates:       {n_updates:,.0f}  "
          f"= {num_timesteps:,} / ({num_envs:,} × {unroll_length})")

    if "rollout+policy" in results:
        rollout_s = results["rollout+policy"][1] * 1e-3
    elif "rollout" in results:
        rollout_s = results["rollout"][1] * 1e-3
        print("  (policy timings skipped — using physics-only rollout as lower bound)")
    else:
        rollout_s = None

    if rollout_s is not None:
        total_rollout_s = n_updates * rollout_s
        print(f"  Rollout time:           {_fmt_duration(total_rollout_s)}")

        num_eval_envs = ppo_params.get("num_eval_envs", 128)
        num_evals = ppo_params.num_evals
        eval_steps_ratio = episode_length / unroll_length
        eval_env_ratio = num_eval_envs / num_envs
        eval_total_s = num_evals * rollout_s * eval_steps_ratio * eval_env_ratio
        print(f"  Eval rollout time:      {_fmt_duration(eval_total_s)}  "
              f"({num_evals} evals × {episode_length} steps × {num_eval_envs} envs)")

        # PPO update is typically 10–25% of rollout for physics-heavy envs
        ppo_overhead = 0.15
        overhead_s = total_rollout_s * ppo_overhead
        total_s = total_rollout_s + eval_total_s + overhead_s
        print(f"  PPO update estimate:    {_fmt_duration(overhead_s)}  (~{ppo_overhead*100:.0f}% of rollout)")
        print(f"  ┌─────────────────────────────────────────────────────────")
        print(f"  │ Estimated total:  {_fmt_duration(total_s)}")
        print(f"  └─────────────────────────────────────────────────────────")
        print(f"  ({results.get('rollout+policy', results.get('rollout', ('',0,'','')))[1]:.1f} ms/update "
              f"× {n_updates:,.0f} updates + eval + ppo_update estimate)")

    # ── JIT compile cost ─────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  JIT COMPILE TIMES  (one-time cost at training start)")
    print("─" * 78)
    jit_rows = [[name, f"{jit:.1f}s"] for name, (jit, *_) in results.items()]
    _print_table(["Stage", "JIT (s)"], jit_rows)
    total_jit = sum(jit for jit, *_ in results.values())
    print(f"\n  Total (sequential upper bound): {total_jit:.1f}s")
    print("  Note: brax compiles rollout+policy+update as one XLA program —")
    print("        actual startup cost may differ from stage sum above.")

    # ── proposed stages ───────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print(_PROPOSED_STAGES)


if __name__ == "__main__":
    main()
