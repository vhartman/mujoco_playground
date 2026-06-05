"""Policy analyzer pipeline: checkpoint -> rollout.npz -> plots [-> frontend].

Usage:
    python -m experimentation.policy_analyzer RUN [RUN ...]
    python -m experimentation.policy_analyzer --stochastic --seed 3 RUN
    python -m experimentation.policy_analyzer --video RUN
    python -m experimentation.policy_analyzer --frontend --serve 8000 RUN
    python -m experimentation.policy_analyzer --serve 8000   # just start server

RUN is a logs/<run> name or substring (resolved like eval_runs). Output goes to
analysis/<run_suffix>/ — named by checkpoint identity, not by when you ran this.
Re-running the same checkpoint overwrites plots.

--serve PORT starts an HTTP server on PORT serving the entire analysis/ directory.
analysis/index.html is auto-generated as a landing page listing all available runs.
Access via SSH port forwarding: ssh -L <local>:localhost:PORT host
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path

from experimentation.policy_analyzer import collect, frontend, visualize


def _run_plots(run_dir: Path, schema: dict | None = None, video: bool = False) -> None:
    visualize.visualize_input_distributions(run_dir, schema=schema)
    visualize.visualize_dof_evolution(run_dir, schema=schema)
    if video:
        visualize.visualize_rollout_video(run_dir, schema=schema)


def _serve(analysis_dir: Path, port: int) -> None:
    frontend.update_root_index(analysis_dir)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(analysis_dir),
    )
    runs = [d.name for d in sorted(analysis_dir.iterdir())
            if d.is_dir() and (d / "index.html").exists()] if analysis_dir.exists() else []
    with http.server.HTTPServer(("", port), handler) as httpd:
        print(f"\nServing {analysis_dir}  ({len(runs)} run(s))")
        print(f"  Landing page:  http://localhost:{port}/")
        for name in runs:
            print(f"  {name}:  http://localhost:{port}/{name}/")
        print(f"\n  SSH tunnel:  ssh -L <local>:localhost:{port} <host>")
        print("  Ctrl-C to stop.\n")
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(prog="policy_analyzer")
    ap.add_argument("runs", nargs="*",
                    help="logs/<run> names/substrings, or dirs with --from-npz; "
                         "omit to just start the server with --serve")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of deterministic mean")
    ap.add_argument("--video", action="store_true",
                    help="also generate the synchronized rollout video")
    ap.add_argument("--frontend", action="store_true",
                    help="generate interactive HTML frontend (frames + data.json)")
    ap.add_argument("--serve", type=int, metavar="PORT",
                    help="start HTTP server on PORT serving analysis/ (implies --frontend)")
    ap.add_argument("--from-npz", action="store_true",
                    help="skip collection; redraw plots from existing dirs")
    args = ap.parse_args()

    if args.serve:
        args.frontend = True

    analysis_dir = collect.REPO_ROOT / "analysis"

    if not args.runs:
        if args.serve:
            _serve(analysis_dir, args.serve)
        else:
            ap.print_help()
        return

    if args.from_npz:
        for r in args.runs:
            run_dir = Path(r)
            _run_plots(run_dir, video=args.video)
            if args.frontend:
                frontend.export_frontend(run_dir)
    else:
        logs_dir = collect.REPO_ROOT / "logs"
        resolved = collect.eval_runs.resolve_runs(logs_dir, args.runs)
        if not resolved:
            print("No matching runs with checkpoints found.", file=sys.stderr)
            sys.exit(1)

        for log_dir in resolved:
            suffix = collect.eval_runs.extract_suffix(log_dir.name)
            run_out = analysis_dir / suffix
            print(f"\n=== {log_dir.name} -> {run_out} ===", flush=True)
            rollout = collect.collect_rollout(
                log_dir, seed=args.seed, deterministic=not args.stochastic
            )
            collect.write_artifacts(run_out, rollout)
            _run_plots(run_out, schema=rollout["schema"], video=args.video)
            if args.frontend:
                frontend.export_frontend(run_out, schema=rollout["schema"])

    if args.serve:
        _serve(analysis_dir, args.serve)


if __name__ == "__main__":
    main()
