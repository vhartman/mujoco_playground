"""Collect per-step rollout datasets (state.metrics + state.info) for CubePinch
policy groups. Vector info fields are expanded into indexed columns; rng/pert/
obs_bias are dropped. One tidy parquet per (group, det|sto).

    python experimentation/pinch_plotting/collect_rollouts.py --group=pinch_perception [--stochastic]
"""

import sys

import numpy as np
import pandas as pd
import jax

import rollout_lib as rl

# group name -> (run dir names, output-file stem)
GROUPS = {
    # CubePinch perception ablation (06-14): full perception vs q-only vs qdot-only
    # vs no proprioception, 3 seeds each.
    "pinch_perception": ([
        "TesolloCubePinch-20260614-215629-cubepinch_sb_baseline_s1",
        "TesolloCubePinch-20260614-221211-cubepinch_sb_baseline_s2",
        "TesolloCubePinch-20260614-222758-cubepinch_sb_baseline_s3",
        "TesolloCubePinch-20260614-210930-cubepinch_sb_pos_only_s1",
        "TesolloCubePinch-20260614-212507-cubepinch_sb_pos_only_s2",
        "TesolloCubePinch-20260614-214044-cubepinch_sb_pos_only_s3",
        "TesolloCubePinch-20260614-202224-cubepinch_sb_vel_only_s1",
        "TesolloCubePinch-20260614-203807-cubepinch_sb_vel_only_s2",
        "TesolloCubePinch-20260614-205348-cubepinch_sb_vel_only_s3",
        "TesolloCubePinch-20260614-193555-cubepinch_sb_none_s1",
        "TesolloCubePinch-20260614-195127-cubepinch_sb_none_s2",
        "TesolloCubePinch-20260614-200653-cubepinch_sb_none_s3",
    ], "cubepinch_perception_rollouts"),
    # CubePinch bundle-ceiling queue. forcemagnitude_s2 diverged -> excluded.
    "pinch_ceiling": ([
        "TesolloCubePinch-20260614-191709-cubepinch_sb_baseline_s1",
        "TesolloCubePinch-20260614-192915-cubepinch_sb_baseline_s2",
        "TesolloCubePinch-20260614-194121-cubepinch_sb_baseline_s3",
        "TesolloCubePinch-20260614-195328-cubepinch_sb_propriotarget_s1",
        "TesolloCubePinch-20260614-200533-cubepinch_sb_propriotarget_s2",
        "TesolloCubePinch-20260614-201741-cubepinch_sb_propriotarget_s3",
        "TesolloCubePinch-20260614-202944-cubepinch_sb_propriodelta_s1",
        "TesolloCubePinch-20260614-204152-cubepinch_sb_propriodelta_s2",
        "TesolloCubePinch-20260614-205359-cubepinch_sb_propriodelta_s3",
        "TesolloCubePinch-20260614-210602-cubepinch_sb_forcemagnitude_s1",
        "TesolloCubePinch-20260614-212235-cubepinch_sb_forcemagnitude_s3",
    ], "cubepinch_ceiling_rollouts"),
}

GROUP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--group=")), "pinch_perception")
RUNS, OUT_STEM = GROUPS[GROUP]
N_ROLLOUTS = 50
DETERMINISTIC = "--stochastic" not in sys.argv
MODE = "det" if DETERMINISTIC else "sto"


def make_rollout_fn(env, inference_fn, episode_length):
    def step_fn(carry, _):
        state, rng = carry
        rng, ak = jax.random.split(rng)
        state = env.step(state, inference_fn(state.obs, ak)[0])
        return (state, rng), (state.reward, state.done, state.metrics, state.info)

    def rollout(rng):
        state = env.reset(rng)
        _, traj = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)
        return traj
    return jax.jit(rollout)


def traj_to_rows(reward, done, metrics, info, *, policy, sensor_bundle, rollout_idx):
    reward, done = np.asarray(reward), np.asarray(done)
    rows = []
    for t in range(reward.shape[0]):
        row = {"policy": policy, "sensor_bundle": sensor_bundle, "rollout": rollout_idx,
               "step": t, "reward": float(reward[t]), "done": bool(done[t])}
        for k, v in metrics.items():
            row[k] = float(np.asarray(v)[t])
        for k, v in info.items():
            if k in rl.DROP_INFO:
                continue
            full = np.asarray(v)
            if full.dtype == object:
                continue
            arr = full[t]
            if arr.ndim == 0:
                row[f"info.{k}"] = arr.item()
            else:
                for j, x in enumerate(arr.reshape(-1)):
                    row[f"info.{k}_{j:02d}"] = float(x)
        rows.append(row)
    return rows


def main():
    rl.DATA_DIR.mkdir(exist_ok=True)
    all_rows = []
    seeds = jax.random.split(jax.random.PRNGKey(0), N_ROLLOUTS)
    for run_name in RUNS:
        run_dir = rl.LOGS_DIR / run_name
        env_name, cfg = rl.load_env_cfg(run_dir)
        bundle = cfg.get("sensor_bundle", "")
        print(f"\n=== {run_name} (bundle={bundle}, mode={MODE})", flush=True)
        env, inf = rl.restore_policy(env_name, cfg, rl.latest_ckpt(run_dir), deterministic=DETERMINISTIC)
        rollout = make_rollout_fn(env, inf, int(cfg.episode_length))
        for i in range(N_ROLLOUTS):
            reward, done, metrics, info = rollout(seeds[i])
            all_rows.extend(traj_to_rows(reward, done, metrics, info,
                                         policy=run_name, sensor_bundle=bundle, rollout_idx=i))
        print(f"    {N_ROLLOUTS} rollouts done", flush=True)

    out = rl.DATA_DIR / f"{OUT_STEM}_{MODE}.parquet"
    pd.DataFrame(all_rows).to_parquet(out, index=False)
    print(f"\nWrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
