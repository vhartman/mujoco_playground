"""Policy Analyzer — multi-rollout session server.

Usage:
    python -m experimentation.policy_analyzer [--port 8000]

A single background thread runs rollouts sequentially and retries on VRAM
exhaustion (polling until enough memory is free). The HTTP server runs in the
main thread and provides a JSON API for the web UI.

API:
    GET  /api/policies              list available training runs
    GET  /api/checkpoints?run=NAME  list checkpoint steps for a run
    POST /api/sessions              start a session {run, checkpoint_step, n_det, n_sto}
    GET  /api/sessions              list all known sessions (newest first)
    GET  /api/sessions/{sid}        get session status (live)
    GET  /*                         static files from analysis/
"""

from __future__ import annotations

import argparse
import http.server
import json
import queue
import re
import shutil
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

from experimentation.policy_analyzer import collect, frontend, visualize
from experimentation.policy_analyzer.session import RolloutInfo, Session
from experimentation.policy_analyzer.worker import _is_oom, _free_vram_gb, MIN_FREE_VRAM_GB, VRAM_POLL_SECS

REPO_ROOT = collect.REPO_ROOT
_APP_TEMPLATE = Path(__file__).parent / "analyzer_template.html"


# ── policy listing ────────────────────────────────────────────────────────────

def _list_policies(logs_dir: Path) -> list[dict]:
    if not logs_dir.exists():
        return []
    date_re = re.compile(r"-\d{8}-")
    result = []
    for d in logs_dir.iterdir():
        if not d.is_dir():
            continue
        ckpt_dir = d / "checkpoints"
        if not ckpt_dir.exists():
            continue
        steps = sorted(
            [dd.name for dd in ckpt_dir.iterdir() if dd.is_dir()],
            key=lambda s: int(s),
        )
        if not steps:
            continue
        m = date_re.search(d.name)
        env = d.name[: m.start()] if m else d.name
        result.append({"name": d.name, "env": env, "n_checkpoints": len(steps)})
    return sorted(result, key=lambda r: r["name"], reverse=True)


# ── HTTP handler ──────────────────────────────────────────────────────────────

def _make_handler(server: "AnalysisServer", analysis_dir: Path):
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(analysis_dir), **kw)

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            if path == "/api/policies":
                self._json(_list_policies(server.logs_dir))
            elif path == "/api/checkpoints":
                run = params.get("run", "")
                steps = collect.list_checkpoints(server.logs_dir / run) if run else []
                self._json(steps)
            elif path == "/api/sessions" and not path[len("/api/sessions"):]:
                self._json(server.list_sessions())
            elif path.startswith("/api/sessions/"):
                sid = path[len("/api/sessions/"):]
                data = server.get_session(sid)
                if data is not None:
                    self._json(data)
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                super().do_GET()

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/sessions":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                sid = server.start_session(
                    run=body["run"],
                    checkpoint_step=body.get("checkpoint_step", "latest"),
                    n_det=int(body.get("n_det", 0)),
                    n_sto=int(body.get("n_sto", 0)),
                )
                self._json({"session_id": sid})
            else:
                self.send_response(405)
                self.end_headers()

        def do_DELETE(self):
            path = urllib.parse.urlsplit(self.path).path
            if path.startswith("/api/sessions/"):
                sid = path[len("/api/sessions/"):]
                ok = server.delete_session(sid)
                self.send_response(200 if ok else 404)
                self.end_headers()
            else:
                self.send_response(405)
                self.end_headers()

        def _json(self, data: object) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return _Handler


# ── analysis server ───────────────────────────────────────────────────────────

_MAX_OOM_RETRIES = 10


class AnalysisServer:
    def __init__(self, analysis_dir: Path, logs_dir: Path):
        self.analysis_dir = analysis_dir
        self.logs_dir = logs_dir
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._task_q: queue.Queue = queue.Queue()

        analysis_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_sessions()
        (analysis_dir / "index.html").write_bytes(_APP_TEMPLATE.read_bytes())

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="rollout-worker"
        )
        self._worker.start()
        print("Policy Analyzer worker ready.", flush=True)

    # ── session loading ───────────────────────────────────────────────────────

    def _load_existing_sessions(self) -> None:
        sessions_dir = self.analysis_dir / "sessions"
        if not sessions_dir.exists():
            return
        for d in sorted(sessions_dir.iterdir()):
            if not d.is_dir():
                continue
            sj = d / "session.json"
            if not sj.exists():
                continue
            try:
                data = json.loads(sj.read_text(encoding="utf-8"))
                rollouts = [
                    RolloutInfo(
                        name=r["name"],
                        deterministic=r["deterministic"],
                        seed=r["seed"],
                        status=(
                            "error" if r["status"] in ("running", "pending") else r["status"]
                        ),
                        error=(
                            "Server restarted"
                            if r["status"] in ("running", "pending")
                            else r.get("error")
                        ),
                    )
                    for r in data["rollouts"]
                ]
                sess = Session(
                    session_id=data["session_id"],
                    session_dir=d,
                    run=data["run"],
                    checkpoint_step=data["checkpoint_step"],
                    rollouts=rollouts,
                )
                self._sessions[data["session_id"]] = sess
            except Exception:
                pass

    # ── background worker ─────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        current_sid: str | None = None
        handles = None

        while True:
            sid, rollout_name = self._task_q.get()
            with self._lock:
                sess = self._sessions.get(sid)
            if sess is None:
                continue

            try:
                if sid != current_sid:
                    log_dir = self.logs_dir / sess.run
                    ckpt = None if sess.checkpoint_step == "latest" else sess.checkpoint_step
                    handles = collect.restore_policy(log_dir, checkpoint_step=ckpt)
                    current_sid = sid

                sess.update_rollout(rollout_name, "running")
                ri = next(r for r in sess.rollouts if r.name == rollout_name)
                rollout_dir = sess.session_dir / rollout_name

                self._run_with_vram_retry(sess, rollout_name, handles, ri, rollout_dir)
                print(f"[session {sid}] {rollout_name} done", flush=True)
            except Exception as exc:
                traceback.print_exc()
                sess.update_rollout(rollout_name, "error", str(exc))

    def _run_with_vram_retry(self, sess, rollout_name, handles, ri, rollout_dir):
        for attempt in range(_MAX_OOM_RETRIES + 1):
            try:
                rollout = collect.run_single_rollout(
                    handles, seed=ri.seed, deterministic=ri.deterministic
                )
                collect.write_artifacts(rollout_dir, rollout)
                visualize.visualize_input_distributions(rollout_dir, schema=rollout["schema"])
                visualize.visualize_dof_evolution(rollout_dir, schema=rollout["schema"])
                frontend.export_frontend(
                    rollout_dir, schema=rollout["schema"], update_index=False
                )
                sess.update_rollout(rollout_name, "done")
                return
            except Exception as exc:
                if not _is_oom(exc) or attempt >= _MAX_OOM_RETRIES:
                    raise
                free = _free_vram_gb(0)
                print(
                    f"[{rollout_name}] OOM (attempt {attempt + 1}), "
                    f"{free:.1f} GB free — waiting for {MIN_FREE_VRAM_GB} GB",
                    flush=True,
                )
                while True:
                    free = _free_vram_gb(0)
                    detail = f"waiting for VRAM — {free:.1f} / {MIN_FREE_VRAM_GB} GB"
                    sess.update_rollout_detail(rollout_name, detail)
                    if free >= MIN_FREE_VRAM_GB:
                        print(f"[{rollout_name}] VRAM OK ({free:.1f} GB), retrying", flush=True)
                        break
                    time.sleep(VRAM_POLL_SECS)

    # ── public API ────────────────────────────────────────────────────────────

    def start_session(self, run: str, checkpoint_step: str, n_det: int, n_sto: int) -> str:
        run_short = run.split("-")[-1] if "-" in run else run
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{run_short}"
        session_dir = self.analysis_dir / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        rollouts: list[RolloutInfo] = []
        for i in range(1, n_det + 1):
            rollouts.append(RolloutInfo(name=f"det-{i}", deterministic=True, seed=i))
        for i in range(1, n_sto + 1):
            rollouts.append(RolloutInfo(name=f"sto-{i}", deterministic=False, seed=i))

        sess = Session(
            session_id=sid, session_dir=session_dir,
            run=run, checkpoint_step=checkpoint_step, rollouts=rollouts,
        )
        sess._save()

        with self._lock:
            self._sessions[sid] = sess

        for r in rollouts:
            self._task_q.put((sid, r.name))

        return sid

    def list_sessions(self) -> list[dict]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(
            [s.to_dict() for s in sessions],
            key=lambda d: d["session_id"],
            reverse=True,
        )

    def get_session(self, sid: str) -> dict | None:
        with self._lock:
            sess = self._sessions.get(sid)
        return sess.to_dict() if sess else None

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(sid, None)
        if sess is None:
            return False
        if sess.session_dir.exists():
            shutil.rmtree(sess.session_dir)
        return True

    def serve(self, port: int) -> None:
        HandlerClass = _make_handler(self, self.analysis_dir)
        with http.server.ThreadingHTTPServer(("", port), HandlerClass) as httpd:
            print(f"\nPolicy Analyzer  →  http://localhost:{port}/")
            print(f"  SSH tunnel: ssh -L <local>:localhost:{port} <host>")
            print("  Ctrl-C to stop.\n", flush=True)
            httpd.serve_forever()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(prog="policy_analyzer")
    ap.add_argument("--port", type=int, default=8000, metavar="PORT")
    ap.add_argument("--serve", type=int, metavar="PORT", dest="port",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    server = AnalysisServer(
        analysis_dir=REPO_ROOT / "analysis",
        logs_dir=REPO_ROOT / "logs",
    )
    server.serve(args.port)


if __name__ == "__main__":
    main()
