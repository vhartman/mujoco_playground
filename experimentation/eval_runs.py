"""Evaluate one or more training checkpoints and save videos named by run suffix."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def find_runs_with_checkpoints(logs_dir: Path) -> list[Path]:
    runs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name != "_queue"],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [
        r for r in runs
        if (r / "checkpoints").exists()
        and any(p.is_dir() for p in (r / "checkpoints").iterdir())
    ]


def extract_suffix(run_name: str) -> str:
    # Match YYYYMMDD-HHMMSS timestamp and take everything after it
    m = re.search(r"\d{8}-\d{6}-(.+)$", run_name)
    return m.group(1) if m else run_name


def extract_env_name(run_name: str) -> str:
    return run_name.split("-")[0]


def eval_run(log_dir: Path, output_path: Path) -> bool:
    env_name = extract_env_name(log_dir.name)
    env = {**os.environ, "MUJOCO_GL": "egl"}
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "learning" / "train_jax_ppo.py"),
            f"--env_name={env_name}",
            "--play_only",
            f"--load_checkpoint_path={log_dir / 'checkpoints'}",
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        return False
    rollout = PROJECT_ROOT / "rollout0.mp4"
    if rollout.exists():
        shutil.move(str(rollout), str(output_path))
    return True


def resolve_runs(logs_dir: Path, args: list[str]) -> list[Path]:
    all_runs = find_runs_with_checkpoints(logs_dir)

    if not args:
        return all_runs[:1]

    if len(args) == 1 and args[0].isdigit():
        return all_runs[: int(args[0])]

    matched = []
    for token in args:
        for run in all_runs:
            if token in run.name and run not in matched:
                matched.append(run)
    return matched


def main():
    args = sys.argv[1:]
    logs_dir = PROJECT_ROOT / "logs"
    runs = resolve_runs(logs_dir, args)

    if not runs:
        print("No matching runs with checkpoints found.")
        sys.exit(1)

    results = []
    for run in runs:
        suffix = extract_suffix(run.name)
        output = PROJECT_ROOT / "videos" / f"{suffix}.mp4"
        print(f"\n=== {run.name} -> {output.name} ===", flush=True)
        ok = eval_run(run, output)
        results.append((run.name, suffix, output.name, ok))

    print("\n=== Summary ===")
    print(f"{'Run':<55} {'Video':<25} {'Status'}")
    print("-" * 90)
    for run_name, suffix, video, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"{run_name:<55} {video:<25} {status}")


if __name__ == "__main__":
    main()
