"""Measure achievable success-step fraction for a trained baseline pinch policy
across a grid of (force_tolerance, success_hold_time), WITHOUT retraining.

Rolls out a competent policy, records per-step effective_force vs force_target,
then replays the env's success logic (consecutive in-tolerance >= hold_steps) for
each candidate (tol, hold). Gives a conservative guide for lightening the task so
base_clean clears >30% successful steps.
"""
import os
import numpy as np
import jax
import jax.numpy as jp
from brax.training import networks as brax_networks
# Older checkpoints saved null kernel-init fn names; the loader maps name->init via
# KERNEL_INITIALIZER and KeyErrors on None. Init is irrelevant for inference (params
# are loaded), so alias None to the default.
if None not in brax_networks.KERNEL_INITIALIZER:
    brax_networks.KERNEL_INITIALIZER[None] = brax_networks.KERNEL_INITIALIZER["lecun_uniform"]
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from mujoco_playground._src.manipulation.tesollo_hand import pinch

import sys
CKPT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
    "logs/TesolloCubePinch-20260613-180551-cubepinch_sb_baseline_joint_pos0.01/"
    "checkpoints/000401080320"
)
N_ENVS = 256
CTRL_DT = 0.05

inference_fn = ppo_checkpoint.load_policy(CKPT, deterministic=True)
jit_inf = jax.jit(jax.vmap(inference_fn))

# Roll out under a CLEAN baseline config (no obs noise) -> represents base_clean.
env = pinch.CubePinch(config_overrides={
    "sensor_bundle": "baseline",
    "obs_noise.level": 0.0,
})
EP_LEN = int(env._config.episode_length)
jit_reset = jax.jit(jax.vmap(env.reset))
jit_step = jax.jit(jax.vmap(env.step))

rng = jax.random.PRNGKey(0)
rng, kreset = jax.random.split(rng)
state = jit_reset(jax.random.split(kreset, N_ENVS))

eff, tgt = [], []
for t in range(EP_LEN):
    rng, kact = jax.random.split(rng)
    act, _ = jit_inf(state.obs, jax.random.split(kact, N_ENVS))
    state = jit_step(state, act)
    eff.append(np.asarray(state.metrics["effective_force"]))
    tgt.append(np.asarray(state.metrics["force_target"]))

eff = np.stack(eff, axis=1)   # (N_ENVS, EP_LEN)
tgt = np.stack(tgt, axis=1)
err = eff - tgt

print(f"checkpoint: {os.path.basename(CKPT)}")
print(f"rollout: {N_ENVS} envs x {EP_LEN} steps, CLEAN baseline")
print(f"force_target  mean={tgt.mean():.2f}  range=[{tgt.min():.2f},{tgt.max():.2f}]")
print(f"effective_force mean={eff.mean():.2f}  std={eff.std():.2f}")
print(f"|err| mean={np.abs(err).mean():.3f}  median={np.median(np.abs(err)):.3f}  "
      f"p90={np.percentile(np.abs(err),90):.3f}")
print()


def success_step_frac(err, tol, hold_time):
    """Replay env success logic: success at a step iff the run of consecutive
    in-tolerance steps ending there is >= hold_steps. Return mean over all steps."""
    hold_steps = round(hold_time / CTRL_DT)
    in_tol = np.abs(err) <= tol            # (N, T)
    succ = np.zeros_like(in_tol)
    consec = np.zeros(in_tol.shape[0], dtype=int)
    for t in range(in_tol.shape[1]):
        consec = np.where(in_tol[:, t], consec + 1, 0)
        succ[:, t] = consec >= hold_steps
    return succ.mean()


tols = [0.3, 0.5, 0.75, 1.0, 1.5]
holds = [0.25, 0.5, 1.0]
print("success-step fraction  (rows=force_tolerance N, cols=hold_time s)")
hdr = "  tol \\ hold | " + " ".join(f"{h:>6.2f}s" for h in holds)
print(hdr); print("  " + "-" * (len(hdr) - 2))
for tol in tols:
    row = " ".join(f"{success_step_frac(err, tol, h)*100:>6.1f}%" for h in holds)
    print(f"  {tol:>9.2f} | {row}")
print()
print("frac of STEPS in-tolerance (hold=0): " +
      " ".join(f"tol{t}={ (np.abs(err)<=t).mean()*100:.1f}%" for t in tols))
