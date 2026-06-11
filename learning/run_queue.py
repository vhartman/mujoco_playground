"""Sequential training-run queue orchestrator.

Reads a queue YAML file and executes each run as a subprocess, one after the
other. Logs stdout/stderr per run and writes a status.json on completion.

Usage:
    python learning/run_queue.py --queue learning/queues/overnight.yaml
    python learning/run_queue.py --queue learning/queues/overnight.yaml --dry-run
    python learning/run_queue.py --queue learning/queues/overnight.yaml --start-from 3

    # Continue the last 2 runs of the most recent queue for 50M more steps each,
    # warm-starting from their checkpoints. Env, overrides and flags are inferred
    # from the recorded run; only the count and step budget are given.
    python learning/run_queue.py --resume 2 --resume-steps 50_000_000
"""

import argparse
import datetime
import itertools
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


def _linspace(lo, hi, n):
    if n == 1:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _expand_param(spec):
    """Expand a single param spec to a list of values."""
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        lo, hi = spec["range"]
        return _linspace(lo, hi, spec.get("steps", 2))
    raise ValueError(f"Param spec must have 'values' or 'range': {spec}")


_KEY_ALIASES = {
    "sensor_bundle": "sb",
}

def _param_label(key, value):
    """Short label fragment for a param.

    Examples:
        'pid_gains.finger_kp', 3.0                   -> 'kp3'
        'sensor_bundle', 'force.full'                -> 'sb_forcefull'
        'sensor_bundle', 'proprio.target+force.magnitude' -> 'sb_propriotarget_forcemagnitude'
        'obs_noise.level', 2.0                       -> 'level2'
    """
    short = _KEY_ALIASES.get(key, key.split(".")[-1].replace("finger_", ""))
    if isinstance(value, str):
        # Sanitize the sensor-bundle DSL ('.'/'+') so it is filename- and
        # wandb-safe: '+' (group separator) -> '_', '.' (group.type) dropped.
        clean = value.replace("+", "_").replace(".", "")
        return f"{short}_{clean}"
    return f"{short}{value:g}"


def _env_prefix(env_name):
    """'TesolloPinchForce' -> 'pinchforce', 'TesolloPickAndPlaceProprio' -> 'pickandplaceproprio'."""
    for marker in ("Tesollo",):
        if env_name.startswith(marker):
            return env_name[len(marker):].lower()
    return env_name.lower()


def _expand_sweep(sweep: dict, defaults: dict, start_idx: int) -> list[dict]:
    """Expand a sweep block into a flat list of run specs.

    sweep:
      env_names: [TesolloPinchForce, ...]
      params:
        pid_gains.finger_kp:
          range: [1.5, 8.0]
          steps: 3          # → linspace [1.5, 4.75, 8.0]
        pid_gains.finger_kv:
          values: [0.0, 1.0]
    """
    env_names = sweep["env_names"]
    default_flags = defaults.get("flags", {})
    default_overrides = defaults.get("env_overrides", {})
    default_script = defaults.get("script", "learning/train_jax_ppo.py")

    params = sweep.get("params", {})
    param_keys = list(params.keys())
    param_value_lists = [_expand_param(params[k]) for k in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in itertools.product(*param_value_lists)]

    runs = []
    idx = start_idx
    for env_name in env_names:
        prefix = _env_prefix(env_name)
        for param_dict in combos:
            param_suffix = "_".join(_param_label(k, v) for k, v in param_dict.items())
            suffix = f"{prefix}_{param_suffix}"
            flags = {**default_flags, "env_name": env_name, "suffix": suffix}
            env_overrides = {**default_overrides, **param_dict}
            runs.append({
                "idx": idx,
                "script": default_script,
                "flags": flags,
                "env_overrides": env_overrides,
            })
            idx += 1
    return runs


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

    if "sweep" in raw:
        runs.extend(_expand_sweep(raw["sweep"], defaults, start_idx=len(runs)))

    if not runs:
        raise ValueError("Queue file has no runs.")
    return runs


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

_QUEUE_TS_RE = re.compile(r"-\d{8}-\d{6}$")


def _queue_stem(dir_name: str) -> str:
    """'downwards_rotate_z_force-20260609-155858' -> 'downwards_rotate_z_force'."""
    return _QUEUE_TS_RE.sub("", dir_name)


def _most_recent_queue_dir(queue_root: pathlib.Path) -> pathlib.Path:
    """Return the queue-log dir whose status.json was written most recently."""
    candidates = [p.parent for p in queue_root.glob("*/status.json")]
    if not candidates:
        raise FileNotFoundError(f"No previous queue runs found under {queue_root}.")
    return max(candidates, key=lambda d: (d / "status.json").stat().st_mtime)


def _load_overrides(path) -> dict:
    """Load a flat-dotted overrides YAML, tolerating a missing/None path."""
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _recover_run_spec(entry: dict, source_dir: pathlib.Path) -> dict | None:
    """Recover the script + flags + env_overrides that produced a status entry.

    Prefers the spec recorded inline in status.json (newer runs store it). Falls
    back to re-parsing the original queue YAML (matched by dir-name stem + idx)
    for older status files. Returns None if neither source is available.
    """
    if entry.get("flags") is not None and entry.get("script") is not None:
        return {
            "script": entry["script"],
            "flags": dict(entry["flags"]),
            "env_overrides": _load_overrides(entry.get("overrides_file")),
        }
    queue_file = pathlib.Path("learning/queues") / f"{_queue_stem(source_dir.name)}.yaml"
    if not queue_file.exists():
        return None
    for r in parse_queue(queue_file):
        if r["idx"] == entry["idx"]:
            return {
                "script": r["script"],
                "flags": dict(r["flags"]),
                "env_overrides": r["env_overrides"],
            }
    return None


def build_resume_runs(
    source_dir: pathlib.Path, last_n: int, resume_steps: int
) -> list[dict]:
    """Build run specs that continue the last `last_n` runs from `source_dir`.

    Each resumed run warm-starts from its predecessor's latest checkpoint
    (``logs/<exp_name>/checkpoints``) and trains for `resume_steps` more steps.
    Everything else — env, overrides, seed, logging flags — is recovered from
    the source run. Runs without a usable checkpoint are skipped with a warning.
    """
    statuses = json.loads((source_dir / "status.json").read_text())
    if last_n > len(statuses):
        print(f"  note: only {len(statuses)} run(s) recorded; resuming all of them.")
    selected = statuses[-last_n:]

    runs = []
    for entry in selected:
        src_label = f"run-{entry['idx']:02d}-{entry['env_name']}"
        exp_name = entry.get("exp_name")
        if not exp_name:
            print(f"  skip {src_label}: no experiment name recorded (no checkpoint).")
            continue
        ckpt_dir = (pathlib.Path("logs") / exp_name / "checkpoints").resolve()
        has_ckpt = ckpt_dir.is_dir() and any(
            p.is_dir() and p.name.isdigit() for p in ckpt_dir.glob("*")
        )
        if not has_ckpt:
            print(f"  skip {src_label}: no checkpoints under {ckpt_dir}.")
            continue
        spec = _recover_run_spec(entry, source_dir)
        if spec is None:
            print(f"  skip {src_label}: could not recover original run flags.")
            continue

        flags = spec["flags"]
        flags["load_checkpoint_path"] = str(ckpt_dir)
        flags["num_timesteps"] = resume_steps
        orig_suffix = flags.get("suffix")
        flags["suffix"] = f"{orig_suffix}_cont" if orig_suffix else "cont"

        runs.append({
            "idx": len(runs),
            "script": spec["script"],
            "flags": flags,
            "env_overrides": spec["env_overrides"],
            "resumed_from": exp_name,
        })

    if not runs:
        raise ValueError(f"No resumable runs found in {source_dir}.")
    return runs


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _build_argv(script: str, flags: dict, overrides_file=None) -> list[str]:
    argv = [sys.executable, "-u", script]
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
        # Full run spec, so a later --resume is self-contained (independent of
        # whether the source queue YAML has since changed).
        "script": run["script"],
        "flags": run["flags"],
        "resumed_from": run.get("resumed_from"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run training runs sequentially from a queue YAML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--queue", help="Path to the queue YAML file.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command for each run without executing.",
    )
    parser.add_argument(
        "--start-from", type=int, default=0, metavar="N",
        help="Skip the first N entries (use to resume a partially-done queue).",
    )
    parser.add_argument(
        "--resume", type=int, default=None, metavar="N",
        help="Continue the last N runs from a previous queue, warm-starting each "
             "from its latest checkpoint. Requires --resume-steps.",
    )
    parser.add_argument(
        "--resume-steps", type=int, default=None, metavar="STEPS",
        help="Additional timesteps to train each resumed run (used with --resume).",
    )
    parser.add_argument(
        "--resume-from", default=None, metavar="DIR",
        help="Queue-log dir to resume from (default: most recent under logs/_queue).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if args.resume is not None:
        if args.queue:
            parser.error("--queue and --resume are mutually exclusive.")
        if args.resume_steps is None:
            parser.error("--resume requires --resume-steps.")
        if args.resume < 1:
            parser.error("--resume must be >= 1.")
        queue_root = pathlib.Path("logs") / "_queue"
        source_dir = (
            pathlib.Path(args.resume_from)
            if args.resume_from
            else _most_recent_queue_dir(queue_root)
        )
        print(f"\nResuming last {args.resume} run(s) from: {source_dir}")
        print(f"Each continues for {args.resume_steps:,} more timesteps.\n")
        runs = build_resume_runs(source_dir, args.resume, args.resume_steps)
        run_label = f"{_queue_stem(source_dir.name)}-resume"
        source_desc = f"{source_dir}  (resume)"
    else:
        if not args.queue:
            parser.error("one of --queue or --resume is required.")
        queue_path = pathlib.Path(args.queue)
        runs = parse_queue(queue_path)
        run_label = queue_path.stem
        source_desc = str(queue_path)

    active = [r for r in runs if r["idx"] >= args.start_from]

    # Print the execution plan
    print(f"\nSource   : {source_desc}")
    print(f"Runs     : {len(runs)} total, {len(active)} will execute")
    print()
    for r in runs:
        skip_tag = " (skip)" if r["idx"] < args.start_from else "      "
        f = r["flags"]
        print(
            f"  [{r['idx']:02d}]{skip_tag}  env={f['env_name']}"
            f"  suffix={f.get('suffix', '-')}"
            f"  seed={f.get('seed', '-')}"
        )
        if r.get("resumed_from"):
            print(
                f"            resume_from={r['resumed_from']}"
                f"  steps={f.get('num_timesteps')}"
            )
        if r["env_overrides"]:
            flat_ov = _flatten(r["env_overrides"])
            print("            overrides={")
            for k, v in sorted(flat_ov.items()):
                print(f"              {k}: {v},")
            print("            }")
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
    log_dir = pathlib.Path("logs") / "_queue" / f"{run_label}-{timestamp}"
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
