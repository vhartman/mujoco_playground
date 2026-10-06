"""Build a grasp cache for in-hand rotation: stable lifted hand+cube states from an existing policy.

usage: make_grasp_cache.py <checkpoint_step_dir> <out.npz> [--n 4096] [--max_goal_deg 60]
Rolls the deterministic policy out from the floor start (full gravity) toward random goals within
+-max_goal_deg, records one state per rollout at a random time in [1, 3] s, and keeps it if the cube's
lowest point is > 5 mm above the floor and its tilt is < 5 deg. Saves qpos, qvel, ctrl (= motor targets).
"""
import argparse, json, pathlib, tempfile
import jax, jax.numpy as jp, numpy as np
from ml_collections import config_dict
from brax.training import checkpoint as bckpt
from brax.training.agents.ppo import checkpoint as ppo_checkpoint, networks as ppo_networks
from mujoco_playground import registry

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--max_goal_deg", type=float, default=60.0)
a = ap.parse_args()
ckpt = pathlib.Path(a.ckpt)

cfg = registry.get_default_config("TesolloDownwardsRotateZ")
def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], config_dict.ConfigDict):
            _merge(dst[k], v)
        else:
            dst[k] = v
_merge(cfg, json.loads((ckpt.parent / "config.json").read_text()))
cfg.nconmax = 1024 * a.n // 8
env = registry.load("TesolloDownwardsRotateZ", config=cfg)

raw = json.loads((ckpt / "ppo_network_config.json").read_text())
raw["network_factory_kwargs"] = {k: v for k, v in raw["network_factory_kwargs"].items() if v is not None}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(raw, f)
net = ppo_checkpoint._get_ppo_network(bckpt.load_config(f.name), ppo_networks.make_ppo_networks)
policy = ppo_networks.make_inference_fn(net)(ppo_checkpoint.load(ckpt), deterministic=True)
T = int(3.0 / env.dt)

def rollout(key):
    k_reset, k_goal, k_t, k_pol = jax.random.split(key, 4)
    s = env.reset(k_reset)
    ang = jax.random.uniform(k_goal, minval=-1.0, maxval=1.0) * jp.deg2rad(a.max_goal_deg)
    q = jp.array([jp.cos(ang / 2), 0.0, 0.0, jp.sin(ang / 2)])
    info = dict(s.info); info["goal_quat"] = q
    s = s.replace(info=info, obs=env._get_obs(s.data, info))
    t_pick = jax.random.randint(k_t, (), int(1.0 / env.dt), T)
    def body(carry, t):
        s, k, saved = carry
        k, ka = jax.random.split(k)
        s = env.step(s, policy(s.obs, ka)[0])
        take = t == t_pick
        saved = jax.tree.map(lambda new, old: jp.where(take, new, old), (s.data.qpos, s.data.qvel, s.data.ctrl), saved)
        return (s, k, saved), None
    init = (s.data.qpos, s.data.qvel, s.data.ctrl)
    (s, _, saved), _ = jax.lax.scan(body, (s, k_pol, init), jp.arange(T))
    return saved

qpos, qvel, ctrl = map(np.asarray, jax.jit(jax.vmap(rollout))(jax.random.split(jax.random.PRNGKey(0), a.n)))

# Filter: cube lifted (lowest point > 5 mm) and nearly upright (tilt < 5 deg), checked with mj_forward per state.
import mujoco
m = env.mj_model; d = mujoco.MjData(m)
cid = env._cube_qids
keep = []
for i in range(a.n):
    q = qpos[i, cid[3]:cid[3] + 4]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
    w, x, y, z = q
    r2 = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    z_min = qpos[i, cid[2]] - np.sum(np.asarray(m.geom_size[env._cube_geom_id]) * np.abs(r2))
    keep.append(z_min > 0.005 and tilt < 5.0 and np.all(np.isfinite(qpos[i])))
keep = np.array(keep)
yaw = np.degrees(2 * np.arctan2(qpos[keep, cid[3] + 3], qpos[keep, cid[3]]))
np.savez(a.out, qpos=qpos[keep], qvel=qvel[keep], ctrl=ctrl[keep], source=str(ckpt))
print(f"kept {keep.sum()}/{a.n} states -> {a.out}; cube yaw range {yaw.min():.0f}..{yaw.max():.0f} deg, "
      f"median |qvel| {np.median(np.abs(qvel[keep])):.3f}")
