"""Sequential training-run queue orchestrator.

Reads a queue YAML file and executes each run as a subprocess, one after the
other. Logs stdout/stderr per run and writes a status.json on completion.

Usage:
    python learning/run_queue.py --queue learning/queues/overnight.yaml
    python learning/run_queue.py --queue learning/queues/overnight.yaml --dry-run
    python learning/run_queue.py --queue learning/queues/overnight.yaml --start-from 3
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

import yaml


# ---------------------------------------------------------------------------
# Queue parsing
# ---------------------------------------------------------------------------

def _flatten(d, prefix=""):
    """Recursively flatten a nested dict to dotted-key flat dict."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def parse_queue(path: pathlib.Path) -> list[dict]:
    """Load queue YAML and return a list of fully-resolved run specs."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    default_script = defaults.get("script", "learning/train_jax_ppo.py")
    default_flags = defaults.get("flags", {})
    default_overrides = defaults.get("env_overrides", {})

    runs = []
    for i, entry in enumerate(raw.get("runs", [])):
        flags = {**default_flags, **entry.get("flags", {})}
        env_overrides = {**default_overrides, **entry.get("env_overrides", {})}
        script = entry.get("script", default_script)
        if "env_name" not in flags:
            raise ValueError(f"Run {i} is missing required 'env_name' flag.")
        runs.append({"idx": i, "script": script, "flags": flags, "env_overrides": env_overrides})

    if not runs:
        raise ValueError("Queue file has no runs.")
    return runs


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _build_argv(script: str, flags: dict, overrides_file=None) -> list[str]:
    argv = [sys.executable, script]
    for k, v in flags.items():
        if isinstance(v, bool):
            argv.append(f"--{k}" if v else f"--no{k}")
        elif isinstance(v, list):
            argv.append(f"--{k}={','.join(str(x) for x in v)}")
        else:
            argv.append(f"--{k}={v}")
    if overrides_file:
        argv.append(f"--env_overrides_file={overrides_file}")
    return argv


def run_one(run: dict, log_dir: pathlib.Path) -> dict:
    """Execute one training run as a subprocess, tee output to a log file.

    Returns a status dict with timing, exit code, and paths.
    Raises KeyboardInterrupt if the user presses Ctrl-C (process is killed
    first so nothing is left orphaned).
    """
    idx = run["idx"]
    env_name = run["flags"]["env_name"]
    suffix = run["flags"].get("suffix", "")
    label = f"run-{idx:02d}-{env_name}" + (f"-{suffix}" if suffix else "")

    log_path = log_dir / f"{label}.log"

    overrides_path = None
    if run["env_overrides"]:
        overrides_path = log_dir / f"{label}-overrides.yaml"
        with open(overrides_path, "w") as f:
            yaml.dump(_flatten(run["env_overrides"]), f, default_flow_style=False)

    argv = _build_argv(run["script"], run["flags"], overrides_path)

    print(f"\n{'='*70}")
    print(f"[{idx:02d}] {label}")
    print(f"     log     : {log_path}")
    if overrides_path:
        print(f"     overrides: {overrides_path}")
    print(f"     command  : {' '.join(str(a) for a in argv)}")
    print(f"{'='*70}\n")

    started_at = datetime.datetime.now().isoformat()
    exp_name = None
    proc = None
    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                if m := re.search(r"Experiment name: (\S+)", line):
                    exp_name = m.group(1)
            proc.wait()
    except KeyboardInterrupt:
        if proc is not None:
            proc.kill()
            proc.wait()
        raise

    ended_at = datetime.datetime.now().isoformat()
    return {
        "idx": idx,
        "env_name": env_name,
        "suffix": suffix or None,
        "returncode": proc.returncode,
        "result": "ok" if proc.returncode == 0 else "failed",
        "started_at": started_at,
        "ended_at": ended_at,
        "exp_name": exp_name,
        "log": str(log_path),
        "overrides_file": str(overrides_path) if overrides_path else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run training runs sequentially from a queue YAML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--queue", required=True, help="Path to the queue YAML file.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command for each run without executing.",
    )
    parser.add_argument(
        "--start-from", type=int, default=0, metavar="N",
        help="Skip the first N entries (use to resume a partially-done queue).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    queue_path = pathlib.Path(args.queue)
    runs = parse_queue(queue_path)
    active = [r for r in runs if r["idx"] >= args.start_from]

    # Print the execution plan
    print(f"\nQueue    : {queue_path}")
    print(f"Runs     : {len(runs)} total, {len(active)} will execute")
    print()
    for r in runs:
        skip_tag = " (skip)" if r["idx"] < args.start_from else "      "
        f = r["flags"]
        ov = f"  overrides={sorted(r['env_overrides'])}" if r["env_overrides"] else ""
        print(
            f"  [{r['idx']:02d}]{skip_tag}  env={f['env_name']}"
            f"  suffix={f.get('suffix', '-')}"
            f"  seed={f.get('seed', '-')}{ov}"
        )
    print()

    if args.dry_run:
        print("--- DRY RUN ---")
        for r in active:
            argv = _build_argv(
                r["script"], r["flags"],
                "/tmp/overrides.yaml" if r["env_overrides"] else None,
            )
            print(f"  [{r['idx']:02d}]  {' '.join(str(a) for a in argv)}")
        return

    if not args.yes:
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if ans != "y":
            print("Aborted.")
            return

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = pathlib.Path("logs") / "_queue" / f"{queue_path.stem}-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    print(f"Queue log dir: {log_dir}\n")

    statuses = []
    try:
        for r in active:
            status = run_one(r, log_dir)
            statuses.append(status)
            status_path.write_text(json.dumps(statuses, indent=2))
            tag = "OK" if status["returncode"] == 0 else f"FAILED (exit {status['returncode']})"
            print(f"\n[{r['idx']:02d}] {tag}")
    except KeyboardInterrupt:
        print("\n\nInterrupted — queue stopped early.")

    passed = sum(1 for s in statuses if s["result"] == "ok")
    failed = [s for s in statuses if s["result"] == "failed"]
    print(f"\n{'='*70}")
    print(f"Result   : {passed}/{len(statuses)} passed")
    if failed:
        print("Failed   :")
        for s in failed:
            print(f"  [{s['idx']:02d}] {s['env_name']}  suffix={s['suffix']}  →  {s['log']}")
    print(f"Status   : {status_path}")


if __name__ == "__main__":
    main()
