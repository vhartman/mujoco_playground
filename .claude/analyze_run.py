"""Compact training-run inspector for /analyze-run.

Resolves logs/<name> ↔ wandb/run-<ts>-<id> pairing, extracts the metrics the
analyze-run skill always wants, and prints them in a fixed format. Caches the
extracted dict under .claude/cache/analyze_run/ so repeated invocations skip
re-parsing.

Usage:
    python learning/analyze_run.py                # most recent run
    python learning/analyze_run.py --last 6       # 6 most recent (table)
    python learning/analyze_run.py PREFIX [...]   # one or more name prefixes
    python learning/analyze_run.py --raw NAME     # dump the cached dict as JSON
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Optional

REPO = pathlib.Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
WANDB = REPO / "wandb"
CACHE = REPO / ".claude" / "cache" / "analyze_run"

TS_RE = re.compile(r"(\d{8})-(\d{6})")


# ---------- locate ----------

def list_log_runs() -> list[pathlib.Path]:
    """All logs/<run> dirs, newest first, excluding the _queue umbrella."""
    return sorted(
        (p for p in LOGS.iterdir() if p.is_dir() and p.name != "_queue"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_wandb_dir(log_run: pathlib.Path) -> Optional[pathlib.Path]:
    """Map logs/<env>-YYYYMMDD-HHMMSS-... → wandb/run-YYYYMMDD_HHMMSS-<id>.

    The wandb timestamp can be off by 0–2 seconds from the log dir; we accept
    the closest match within a 5-second window.
    """
    m = TS_RE.search(log_run.name)
    if not m:
        return None
    date, time = m.group(1), m.group(2)
    log_secs = int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:])
    best, best_delta = None, 999
    for d in WANDB.glob(f"run-{date}_*"):
        m2 = re.match(rf"run-{date}_(\d{{6}})-", d.name)
        if not m2:
            continue
        t2 = m2.group(1)
        s = int(t2[:2]) * 3600 + int(t2[2:4]) * 60 + int(t2[4:])
        delta = abs(s - log_secs)
        if delta < best_delta:
            best, best_delta = d, delta
    return best if best_delta <= 5 else None


def resolve(token: str) -> Optional[pathlib.Path]:
    """Match a token against logs/<run> dirs: exact, then prefix, then substring."""
    runs = list_log_runs()
    for p in runs:
        if p.name == token:
            return p
    for p in runs:
        if p.name.startswith(token):
            return p
    for p in runs:
        if token in p.name:
            return p
    return None


# ---------- extract ----------

def _yaml_value(c: dict, key: str) -> Any:
    v = c.get(key)
    return v.get("value") if isinstance(v, dict) and "value" in v else v


def extract(log_run: pathlib.Path) -> dict:
    """Pull everything analyze-run needs from one run. Cached by wandb run id."""
    wb = find_wandb_dir(log_run)
    if wb is None:
        return {"name": log_run.name, "error": "no matching wandb dir"}

    cache_file = CACHE / f"{wb.name}.json"
    summary_file = wb / "files" / "wandb-summary.json"
    if cache_file.exists() and summary_file.exists():
        if cache_file.stat().st_mtime >= summary_file.stat().st_mtime:
            return json.loads(cache_file.read_text())

    out: dict[str, Any] = {"name": log_run.name, "wandb": wb.name}
    files = wb / "files"

    # config.yaml
    cfg_path = files / "config.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        out["env_name"] = _yaml_value(cfg, "env_name")
        out["env_config"] = {
            k: _yaml_value(cfg, k)
            for k in (
                "action_scale", "ema_alpha", "action_repeat", "episode_length",
                "ctrl_dt", "sim_dt", "target_radius", "target_hold_time",
                "reward_config", "obs_noise", "pert_config", "kp_scale",
            )
            if _yaml_value(cfg, k) is not None
        }

    # output.log → reward curve
    log_path = files / "output.log"
    curve: list[tuple[int, float]] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = re.match(r"^(\d+):.*?=\s*([0-9.]+)\s*$", line)
            if m:
                try:
                    curve.append((int(m.group(1)), float(m.group(2))))
                except ValueError:
                    pass
    out["curve"] = curve
    out["n_evals"] = len(curve)
    out["final_step"] = curve[-1][0] if curve else None
    out["completed"] = "Done training." in (log_path.read_text() if log_path.exists() else "")

    # wandb-summary.json
    if summary_file.exists():
        d = json.loads(summary_file.read_text())
        keys = [
            "_step", "eval/episode_reward", "eval/episode_success_count",
            "training/kl_mean", "episode/kl_mean",
            "training/v_loss", "training/entropy_loss",
            "training/policy_dist_mean_std", "training/policy_dist_max_std",
            "training/policy_dist_max_loc", "training/policy_dist_min_loc",
            "training/learning_rate", "training/sps", "eval/sps",
            "eval/episode_reward/cube_pos", "eval/episode_reward/cube_ori",
            "eval/episode_reward/cube_height", "eval/episode_reward/fingertip_pos",
            "eval/episode_reward/joint_vel", "eval/episode_reward/wrist_vel",
            "eval/episode_reward/action_rate", "eval/episode_reward/cube_dropped",
            "eval/episode_reward/success",
        ]
        out["summary"] = {k: d.get(k) for k in keys if k in d}

    # logs/<run>/checkpoints/config.json (env config snapshot, sometimes the
    # ground truth when env_overrides_file was used)
    ckpt_cfg = log_run / "checkpoints" / "config.json"
    if ckpt_cfg.exists():
        try:
            out["checkpoint_config"] = json.loads(ckpt_cfg.read_text())
        except json.JSONDecodeError:
            pass

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out))
    return out


# ---------- format ----------

def _fmt(v, spec=".3f"):
    if v is None or isinstance(v, str):
        return "N/A"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _sample_curve(curve: list[tuple[int, float]], n: int = 12) -> str:
    if not curve:
        return "(no eval logs)"
    if len(curve) <= n:
        idx = range(len(curve))
    else:
        step = len(curve) / n
        idx = sorted(set([int(i * step) for i in range(n)] + [len(curve) - 1]))
    return "  ".join(f"{curve[i][0]/1e6:.0f}M:{curve[i][1]:.2f}" for i in idx)


def _instability(curve: list[tuple[int, float]]) -> dict:
    if len(curve) < 10:
        return {"max_drop": None, "low_reward_frac": None}
    peak = 0.0
    max_drop = 0.0
    lows = 0
    for i, (_, r) in enumerate(curve):
        peak = max(peak, r)
        if i > 5:
            max_drop = max(max_drop, peak - r)
            if r < 15:
                lows += 1
    return {
        "max_drop": max_drop,
        "low_reward_frac": lows / max(1, len(curve) - 5),
    }


def print_one(d: dict) -> None:
    if "error" in d:
        print(f"=== {d['name']} === ERROR: {d['error']}")
        return
    s = d.get("summary", {})
    ec = d.get("env_config", {})
    rc = ec.get("reward_config", {}).get("scales", {}) if isinstance(ec.get("reward_config"), dict) else {}
    inst = _instability(d.get("curve", []))

    print(f"=== {d['name']} ===")
    print(f"  env={d.get('env_name','?')}  steps={d.get('final_step','?')}  "
          f"evals={d['n_evals']}  completed={d['completed']}")
    print(f"  reward={_fmt(s.get('eval/episode_reward'), '.2f')}  "
          f"successes/ep={_fmt(s.get('eval/episode_success_count'), '.1f')}  "
          f"max_drop={_fmt(inst['max_drop'], '.1f')}  "
          f"low_frac={_fmt(inst['low_reward_frac'], '.2f')}")
    print(f"  train_kl={_fmt(s.get('training/kl_mean'))}  "
          f"ep_kl={_fmt(s.get('episode/kl_mean'))}  "
          f"v_loss={_fmt(s.get('training/v_loss'), '.4f')}  "
          f"ent_loss={_fmt(s.get('training/entropy_loss'))}  "
          f"mean_std={_fmt(s.get('training/policy_dist_mean_std'))}  "
          f"max_std={_fmt(s.get('training/policy_dist_max_std'), '.2f')}")
    print(f"  env: action_scale={ec.get('action_scale')}  ema_alpha={ec.get('ema_alpha')}  "
          f"target_radius={ec.get('target_radius')}  hold={ec.get('target_hold_time')}")
    print(f"  reward_scales: {rc}")
    print(f"  reward_breakdown: "
          f"cube_pos={_fmt(s.get('eval/episode_reward/cube_pos'), '.0f')}  "
          f"cube_ori={_fmt(s.get('eval/episode_reward/cube_ori'), '.1f')}  "
          f"cube_height={_fmt(s.get('eval/episode_reward/cube_height'), '.1f')}  "
          f"fingertip={_fmt(s.get('eval/episode_reward/fingertip_pos'), '.1f')}  "
          f"joint_vel={_fmt(s.get('eval/episode_reward/joint_vel'), '.1f')}  "
          f"wrist_vel={_fmt(s.get('eval/episode_reward/wrist_vel'), '.1f')}  "
          f"action_rate={_fmt(s.get('eval/episode_reward/action_rate'), '.2f')}  "
          f"dropped={_fmt(s.get('eval/episode_reward/cube_dropped'), '.2f')}")
    print(f"  curve: {_sample_curve(d.get('curve', []))}")


def print_table(runs: list[dict]) -> None:
    """One-line-per-run compact comparison."""
    print(f"{'name':<60} {'reward':>7} {'succ':>6} {'kl':>6} {'v_loss':>8} {'max_drop':>9} {'low':>5}")
    for d in runs:
        if "error" in d:
            print(f"{d['name']:<60} ERROR")
            continue
        s = d.get("summary", {})
        inst = _instability(d.get("curve", []))
        print(f"{d['name']:<60} "
              f"{_fmt(s.get('eval/episode_reward'), '.2f'):>7} "
              f"{_fmt(s.get('eval/episode_success_count'), '.1f'):>6} "
              f"{_fmt(s.get('training/kl_mean'), '.3f'):>6} "
              f"{_fmt(s.get('training/v_loss'), '.4f'):>8} "
              f"{_fmt(inst['max_drop'], '.1f'):>9} "
              f"{_fmt(inst['low_reward_frac'], '.2f'):>5}")


# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", nargs="*", help="Run name prefixes / substrings.")
    ap.add_argument("--last", type=int, help="Use the N most recent runs.")
    ap.add_argument("--raw", action="store_true", help="Print cached JSON for each run.")
    ap.add_argument("--table", action="store_true", help="Force compact table output.")
    args = ap.parse_args()

    if args.last:
        runs = list_log_runs()[: args.last]
    elif args.tokens:
        runs = [resolve(t) for t in args.tokens]
        if any(r is None for r in runs):
            missing = [t for t, r in zip(args.tokens, runs) if r is None]
            print(f"Could not resolve: {missing}", file=sys.stderr)
            sys.exit(2)
    else:
        runs = list_log_runs()[:1]

    data = [extract(r) for r in runs]

    if args.raw:
        print(json.dumps(data, indent=2, default=str))
        return

    if args.table:
        print_table(data)
        return
    if len(data) > 4:
        print_table(data)
        print()
    for d in data:
        print_one(d)
        print()


if __name__ == "__main__":
    main()
