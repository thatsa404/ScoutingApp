"""Stage 4 -- the always-on half of remote control. Supervises rtrack.watch.

    uv run -m rtrack.agent                      # run it
    uv run -m rtrack.agent --install-task       # register it to start at logon

Polls ``control/<agentId>`` on the relay for DESIRED STATE and makes reality match it by
starting or stopping a ``rtrack.watch`` subprocess. Posts ``status/<agentId>`` so the app
can show a lead scout that this machine is alive and what it is doing.

WHY THIS SUPERVISES rtrack.watch RATHER THAN ABSORBING IT. watch.py hands each match to
rtrack.pipeline, which takes a per-event lock, so two answers landing together are
processed one after the other -- and that ordering matters because every run updates the
shared appearance gallery. Re-implementing the watch loop here would mean re-deriving that
guarantee. Instead this process does no pipeline work at all: it starts a watcher, notices
if it dies, and reports. A crash here costs remote control, not data.

DESIRED STATE, NOT A COMMAND QUEUE. The relay holds one document saying what should be
true, not a list of things to do. Re-posting it is a no-op, so a flaky phone can retry
freely; and an agent that reboots mid-event reads the document and resumes without anyone
re-issuing anything. The cost is that "start, then stop, then start again" collapses if it
happens inside one poll interval, which is the right trade for a control surface a human
presses buttons on.

THE NONCE IS THE RECEIPT. status.appliedNonce echoes control.nonce once this process has
acted on that document. The app waits for the echo before claiming the machine is armed,
which is the difference between "we posted a command" and "the home machine has it" -- the
one piece of feedback that was missing when both watchers died silently and nothing picked
up curation answers until a human noticed hours later.

HEARTBEAT COST IS A REAL CONSTRAINT. Free-tier KV allows 1,000 writes a day and each
status post costs two of them (the value, plus the index manifest). So the interval is slow
by default -- 10 min idle, 2 min while running -- and responsiveness comes from posting
IMMEDIATELY on every state change instead of from polling fast. Reading /control is a read
(100k/day), so that stays quick and arming still feels instant.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from . import relay as R

PY = sys.executable

# Reading desired state is a KV READ, and reads are effectively free next to the 1k/day
# write cap. This is what makes arming feel immediate.
CONTROL_POLL_S = 30.0

# Writes, so slow on purpose. See the note in the module docstring.
HEARTBEAT_IDLE_S = 600.0
HEARTBEAT_RUNNING_S = 120.0

STATE_FILE = C.OUT_DIR / "agent_state.json"
TASK_NAME = "RTrackAgent"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def default_agent_id() -> str:
    """Stable per-machine id, so a reinstall does not orphan the relay document."""
    state = _load_state()
    if state.get("agentId"):
        return str(state["agentId"])
    # Hostname is readable in the UI and stable in practice; the uuid suffix only exists
    # to keep two machines with the same name from fighting over one document.
    agent_id = f"{socket.gethostname().lower().replace(' ', '-')}-{uuid.uuid4().hex[:4]}"
    state["agentId"] = agent_id
    _save_state(state)
    return agent_id


class Watcher:
    """The rtrack.watch subprocess, and nothing else."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.event: str | None = None
        self.calib_from: str | None = None
        self.started_at: str | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, event: str, calib_from: str | None) -> None:
        a = [PY, "-m", "rtrack.watch", "--event", event]
        if calib_from:
            a += ["--calib-from", calib_from]
        print(f"[agent] starting watcher: {' '.join(a[2:])}", flush=True)
        # cwd is the tracker root so relative paths behave as they do when a human runs
        # the same command by hand.
        self.proc = subprocess.Popen(a, cwd=C.TRACKER_ROOT)
        self.event = event
        self.calib_from = calib_from
        self.started_at = _now_iso()

    def stop(self) -> None:
        if not self.alive:
            self.proc = None
            self.event = None
            return
        print("[agent] stopping watcher", flush=True)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # A watcher mid-pipeline can ignore terminate for a while. Killing it is safe:
            # pipeline work is resumable and `pending()` recomputes from published-route
            # mtimes rather than a processed-set, so an interrupted match is simply picked
            # up again on the next poll.
            print("[agent] watcher ignored terminate; killing", flush=True)
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None
        self.event = None
        self.calib_from = None
        self.started_at = None


def queue_depth(event: str | None) -> dict:
    """Pending answer / gallery counts, for the app's 'is there work waiting' readout.

    Best-effort: a relay hiccup here must not stop the heartbeat, because the heartbeat is
    the thing that proves this machine is alive.
    """
    out: dict = {}
    try:
        from .watch import pending, pending_gallery
        out["pendingAnswers"] = len(pending(event))
        out["pendingGallery"] = len(pending_gallery(None))
    except Exception as exc:                      # noqa: BLE001 - see docstring
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def post_status(agent_id: str, watcher: Watcher, desired: dict | None,
                applied_nonce: str | None, state: str,
                error: str | None = None) -> None:
    doc = {
        "schemaVersion": 1,
        "agentId": agent_id,
        "at": _now_iso(),
        "state": state,
        "event": watcher.event or (desired or {}).get("event"),
        "appliedNonce": applied_nonce,
        "host": {"name": socket.gethostname(), "platform": platform.platform()},
        "watcher": {
            "alive": watcher.alive,
            "pid": watcher.proc.pid if watcher.proc else None,
            "startedAt": watcher.started_at,
            "calibFrom": watcher.calib_from,
        },
        "queue": queue_depth(watcher.event) if watcher.alive else {},
        "stream": (desired or {}).get("stream"),
        "error": error,
    }
    try:
        R.put("status", agent_id, doc)
    except Exception as exc:                      # noqa: BLE001
        # Losing a heartbeat is not fatal and must not end the loop -- the app will show
        # the agent as stale, which is the honest reading of "cannot reach the relay".
        print(f"[agent] could not post status ({type(exc).__name__}: {exc})", flush=True)


def reconcile(desired: dict | None, watcher: Watcher) -> tuple[bool, str | None]:
    """Make the watcher match `desired`. Returns (changed, error)."""
    want_running = bool(desired) and desired.get("desired") == "running"
    event = (desired or {}).get("event") or None
    calib_from = (desired or {}).get("calibFrom") or None

    if want_running and not event:
        return False, "control document asks for 'running' but names no event"

    if not want_running:
        if watcher.alive:
            watcher.stop()
            return True, None
        return False, None

    # Changing event (or calibration source) means a different watcher, not a reconfigured
    # one -- watch.py takes both as start-up arguments.
    if watcher.alive and (watcher.event != event or watcher.calib_from != calib_from):
        watcher.stop()

    if not watcher.alive:
        try:
            watcher.start(event, calib_from)
        except OSError as exc:
            return True, f"could not start watcher: {exc}"
        return True, None
    return False, None


def install_task(agent_id: str, extra: list[str] | None = None) -> int:
    """Register a logon scheduled task, so remote control survives a reboot.

    IN THE USER SESSION on purpose. The watcher this supervises runs CUDA work, and a task
    configured to run whether or not the user is logged on gets session 0, where GPU access
    is unreliable. 'no physical access to the host machine' is the requirement; surviving
    a reboot without a login is not.
    """
    if platform.system() != "Windows":
        print("[agent] --install-task is Windows-only; on other platforms use systemd "
              "--user or launchd", file=sys.stderr)
        return 2
    cmd = f'"{PY}" -m rtrack.agent --agent-id {agent_id}'
    if extra:
        cmd += " " + " ".join(extra)
    a = ["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "ONLOGON", "/F",
         "/TR", f'cmd /c cd /d "{C.TRACKER_ROOT}" && {cmd}']
    print(f"[agent] {' '.join(a)}")
    rc = subprocess.run(a).returncode
    if rc == 0:
        print(f"[agent] registered '{TASK_NAME}' to start at logon as {agent_id}.\n"
              f"        Remove it with:  schtasks /Delete /TN {TASK_NAME} /F")
    return rc


def post_control(agent_id: str, desired: str, event: str | None,
                 calib_from: str | None, stream_url: str | None) -> int:
    """Write the desired-state document, the way the app will.

    Exists so the whole loop is testable from a terminal before any UI is built, and so
    there is a way to disarm a machine from the machine itself. Uses the full home-machine
    token from .env rather than RTRACK_CONTROL_TOKEN -- the token split exists to keep a
    high-privilege secret out of a PUBLIC web bundle, and a local CLI is not that.
    """
    doc = {
        "schemaVersion": 1,
        "desired": desired,
        "event": event,
        "calibFrom": calib_from,
        "stream": {"url": stream_url} if stream_url else None,
        "issuedBy": f"cli:{socket.gethostname()}",
        "issuedAt": _now_iso(),
        "nonce": uuid.uuid4().hex[:12],
    }
    R.put("control", agent_id, doc)
    print(f"[agent] {agent_id}: desired={desired}"
          + (f" event={event}" if event else "")
          + f" nonce={doc['nonce']}")
    print("[agent] the agent echoes this nonce back as status.appliedNonce once it has "
          "acted; check with --show")
    return 0


def show(agent_id: str) -> int:
    for kind in ("control", "status"):
        try:
            doc = R.get(kind, agent_id)
        except Exception as exc:                  # noqa: BLE001
            print(f"[agent] {kind}: unreachable ({type(exc).__name__}: {exc})")
            continue
        if doc is None:
            print(f"[agent] {kind}/{agent_id}: nothing posted")
            continue
        print(f"[agent] {kind}/{agent_id}:")
        print(json.dumps(doc, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Relay-controlled supervisor for rtrack.watch.")
    ap.add_argument("--agent-id", default=None,
                    help="relay document id; defaults to a stable per-machine value")
    ap.add_argument("--poll", type=float, default=CONTROL_POLL_S,
                    help="seconds between /control reads (a KV read, so cheap)")
    ap.add_argument("--heartbeat-s", type=float, default=None,
                    help="override the status post interval; see the note about the "
                         "1k/day KV write cap before lowering it")
    ap.add_argument("--once", action="store_true",
                    help="reconcile once, post status, exit. NOTE: a watcher started this "
                         "way OUTLIVES the agent and nothing will stop it on disarm, so "
                         "this is a diagnostic rather than a way to run from cron.")
    ap.add_argument("--install-task", action="store_true",
                    help="register a Windows logon task and exit")

    g = ap.add_argument_group("operator actions (one-shot, then exit)")
    g.add_argument("--arm", metavar="EVENT", default=None,
                   help="post desired=running for EVENT, as the app will")
    g.add_argument("--disarm", action="store_true",
                   help="post desired=stopped")
    g.add_argument("--show", action="store_true",
                   help="print the current control and status documents")
    g.add_argument("--calib-from", default=None, metavar="VIDEO",
                   help="calibration stem to pass the watcher when arming")
    g.add_argument("--stream-url", default=None,
                   help="stream URL to record in the control document when arming")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    agent_id = args.agent_id or default_agent_id()

    if args.install_task:
        return install_task(agent_id)
    if args.show:
        return show(agent_id)
    if args.arm:
        return post_control(agent_id, "running", args.arm,
                            args.calib_from or args.arm, args.stream_url)
    if args.disarm:
        return post_control(agent_id, "stopped", None, None, None)

    R._env()          # fail now, loudly, if the relay is not configured

    state = _load_state()
    applied_nonce = state.get("appliedNonce")
    watcher = Watcher()
    print(f"[agent] {agent_id}: polling /control every {args.poll:.0f}s "
          f"-- Ctrl-C to stop", flush=True)

    last_beat = 0.0
    last_error: str | None = None
    desired: dict | None = None

    try:
        while True:
            try:
                desired = R.get("control", agent_id)
            except Exception as exc:              # noqa: BLE001
                print(f"[agent] relay unreachable ({type(exc).__name__}: {exc}); "
                      "retrying", flush=True)
                desired = None
                last_error = f"relay unreachable: {type(exc).__name__}"

            changed = False
            if desired is not None:
                nonce = desired.get("nonce")
                changed, err = reconcile(desired, watcher)
                last_error = err
                if err:
                    print(f"[agent] {err}", flush=True)
                # The nonce is recorded even when reconcile made no change: "already in the
                # requested state" is still having applied the document, and the app is
                # waiting on that echo to stop saying 'waiting for the home machine'.
                if nonce and nonce != applied_nonce and not err:
                    applied_nonce = nonce
                    state["appliedNonce"] = nonce
                    _save_state(state)
                    changed = True

            # A watcher that exited on its own is a state change worth reporting promptly:
            # this is the exact failure that used to go unnoticed.
            if watcher.proc is not None and not watcher.alive:
                rc = watcher.proc.poll()
                print(f"[agent] watcher exited ({rc})", flush=True)
                last_error = f"watcher exited with {rc}"
                watcher.proc = None
                watcher.started_at = None
                changed = True

            run_state = "running" if watcher.alive else ("error" if last_error else "idle")
            beat_every = args.heartbeat_s or (
                HEARTBEAT_RUNNING_S if watcher.alive else HEARTBEAT_IDLE_S)
            if changed or time.time() - last_beat >= beat_every:
                post_status(agent_id, watcher, desired, applied_nonce,
                            run_state, last_error)
                last_beat = time.time()

            if args.once:
                return 0
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[agent] stopping", flush=True)
        watcher.stop()
        post_status(agent_id, watcher, desired, applied_nonce, "idle",
                    "agent stopped by operator")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
