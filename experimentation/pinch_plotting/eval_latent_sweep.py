"""Open-loop diagnostic: sweep an unobserved cube latent (size or position) at EVAL
and measure steady-state force-tracking error per policy. Restore each checkpoint
once, then for each latent value rebuild the env, run N deterministic rollouts, and
record mean |info.force_error| over the hold window. Records action_mode so delta vs
target-joint policies can be compared.

    python experimentation/pinch_plotting/eval_latent_sweep.py --group=e2_pos_rand --latent=pos

Output: rollout_data/cubepinch_<latent>sweep_<group>.parquet
"""

import sys
import json

import numpy as np
import pandas as pd
import jax

import rollout_lib as rl

GROUP_QUEUES = {
    # E1 size DR (floor-fixed, both action modes)
    "e1_size_fixed": ["pinch_e1_size_fixed-20260619-095550"],
    "e1_size_rand": ["pinch_e1_size_rand-20260619-095600"],
    # E2 position DR
    "e2_pos_fixed": ["pinch_e2_pos_fixed-20260618-212306"],
    "e2_pos_rand": ["pinch_e2_pos_rand-20260619-032823"],
    # E3 delta-pose (apply delta to current q, not previous target)
    "e3_dp_fixed": ["pinch_dp_size_fixed-20260620-070732"],
    "e3_dp_rand": ["pinch_dp_size_rand-20260620-100744"],
}

# latent name -> (config field, sweep values, setter(cfg, value))
LATENTS = {
    "size": ("cube_size_scale", [0.85, 0.925, 1.0, 1.075, 1.15],
             lambda cfg, v: cfg.__setattr__("cube_size_scale", float(v))),
    "pos":  ("cube_pos_offset", [-0.015, -0.0075, 0.0, 0.0075, 0.015],
             lambda cfg, v: cfg.__setattr__("cube_pos_offset", [float(v), 0.0])),
}

GROUP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--group=")), "e2_pos_rand")
LATENT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--latent=")), "pos")
FIELD, SWEEP, SET = LATENTS[LATENT]
N_ROLLOUTS = 12
HOLD_FROM = 50
MIN_CKPTS = 5


def resolve_runs(group):
    """(run_dir, bundle, action_mode, seed) per healthy run, deduped."""
    seen, out = set(), []
    for q in GROUP_QUEUES[group]:
        for name in rl.queue_runs(q, min_ckpts=MIN_CKPTS):
            cfg = json.load(open(rl.LOGS_DIR / name / "checkpoints" / "config.json"))
            bundle = cfg.get("sensor_bundle", "?")
            amode = cfg.get("action_mode", "delta")
            seed = name.rsplit("_s", 1)[-1]
            key = (bundle, amode, seed)
            if key in seen:
                continue
            seen.add(key)
            out.append((name, bundle, amode, seed))
    return out


def make_rollout(env, inf_fn, episode_length):
    def step_fn(carry, _):
        state, rng = carry
        rng, ak = jax.random.split(rng)
        state = env.step(state, inf_fn(state.obs, ak)[0])
        return (state, rng), state.info["force_error"]

    def rollout(rng):
        state = env.reset(rng)
        _, ferr = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)
        return ferr
    return jax.jit(rollout)


def main():
    runs = resolve_runs(GROUP)
    print(f"group={GROUP} latent={LATENT}: {len(runs)} runs x {len(SWEEP)} vals x {N_ROLLOUTS} rollouts", flush=True)
    rows = []
    seeds = jax.random.split(jax.random.PRNGKey(0), N_ROLLOUTS)
    for run_name, bundle, amode, seed in runs:
        run_dir = rl.LOGS_DIR / run_name
        env_name, base_cfg = rl.load_env_cfg(run_dir)
        ckpt = rl.latest_ckpt(run_dir)
        # Restore the policy once (params are latent-independent; obs size is too,
        # since the cube isn't observed); rebuild only the env per latent value.
        _, inf = rl.restore_policy(env_name, base_cfg, ckpt, deterministic=True)
        for v in SWEEP:
            cfg = base_cfg.copy_and_resolve_references()
            SET(cfg, v)
            env = rl.registry.load(env_name, config=cfg)
            rollout = make_rollout(env, inf, int(cfg.episode_length))
            for i in range(N_ROLLOUTS):
                ferr = np.asarray(rollout(seeds[i]))
                rows.append({
                    "policy": run_name, "sensor_bundle": bundle, "action_mode": amode,
                    "seed": seed, "latent": float(v), "rollout": i,
                    "mae_force_error": float(np.mean(np.abs(ferr[HOLD_FROM:]))),
                })
        print(f"  done {bundle} {amode} s{seed}", flush=True)

    rl.DATA_DIR.mkdir(exist_ok=True)
    out = rl.DATA_DIR / f"cubepinch_{LATENT}sweep_{GROUP}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
