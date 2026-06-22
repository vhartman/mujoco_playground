"""Render free-space (cube transparent + non-colliding) videos of the latest
checkpoint of every policy in one or more queues. The policy runs with no contact
feedback (force sensors read 0); a faint ghost cube marks where the fingertips reach.

    python experimentation/pinch_plotting/render_cube_removed.py

Videos -> videos/pinch_cube_removed_<ts>/ (bind-mounted, gitignored).
"""

from datetime import datetime

import jax
import mediapy as media
import mujoco

import rollout_lib as rl

QUEUES = [
    "pinch_single_channel_ablation-20260614-193547",
    "pinch_bundle_ceiling-20260614-191701",
]
RENDER_EVERY = 2


def make_empty_traj(sample_state):
    empty_data = sample_state.data.__class__(**{k: None for k in sample_state.data.__annotations__})
    empty_traj = sample_state.__class__(**{k: None for k in sample_state.__annotations__})
    return empty_traj.replace(data=empty_data)


def rollout_states(env, inference_fn, episode_length, rng):
    sample = jax.jit(env.reset)(rng)
    empty = make_empty_traj(sample)

    def step_fn(carry, _):
        state, key = carry
        key, ak = jax.random.split(key)
        state = env.step(state, inference_fn(state.obs, ak)[0])
        td = empty.tree_replace({
            "data.qpos": state.data.qpos, "data.qvel": state.data.qvel,
            "data.time": state.data.time, "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos, "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, key), td

    _, traj = jax.lax.scan(step_fn, (sample, rng), None, length=episode_length)
    return [jax.tree.map(lambda x, i=i: x[i], traj) for i in range(episode_length)]


def main():
    out_dir = rl.PROJECT_ROOT / "videos" / f"pinch_cube_removed_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_option = mujoco.MjvOption()
    runs = [(q.split("-")[0].replace("pinch_", ""), name)
            for q in QUEUES for name in rl.queue_runs(q)]
    print(f"Rendering {len(runs)} policies (cube transparent) -> {out_dir}", flush=True)

    done, failed = [], []
    for short, run_name in runs:
        run_dir = rl.LOGS_DIR / run_name
        ckpt = rl.latest_ckpt(run_dir)
        suffix = run_name.split("-", 2)[-1]
        out_path = out_dir / f"{short}__{suffix}.mp4"
        try:
            env_name, cfg = rl.load_env_cfg(run_dir, remove_cube=True)
            env, inf = rl.restore_policy(env_name, cfg, ckpt, deterministic=True)
            states = rollout_states(env, inf, int(cfg.episode_length), jax.random.PRNGKey(1))
            fps = 1.0 / env.dt / RENDER_EVERY
            frames = env.render(states[::RENDER_EVERY], height=480, width=640, scene_option=scene_option)
            media.write_video(out_path, frames, fps=fps)
            done.append(out_path.name)
            print(f"  ok  {out_path.name}  (ckpt {ckpt.name})", flush=True)
        except Exception as e:
            failed.append(f"{run_name}: {type(e).__name__}: {e}")
            print(f"  FAIL {run_name}: {type(e).__name__}: {e}", flush=True)

    print(f"\nDone: {len(done)} videos, {len(failed)} failed.")
    for f in failed:
        print("  failed:", f)
    print(f"\nVIDEO_DIR: {out_dir}")


if __name__ == "__main__":
    main()
