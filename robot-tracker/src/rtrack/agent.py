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

JOBS ARE SEPARATE FROM DESIRED STATE, and deliberately so. "Be watching event X" is a
state -- idempotent, re-postable, resumable. "Detect qm26 through qm100" is a unit of work
with a beginning and an end, and re-posting it must NOT re-run it. So jobs are their own
relay documents and completion is tracked in a local ledger, exactly the way watch.py
decides an answer is pending by comparing the relay against what is on disk.

JOBS RUN --prep-only, WHICH TAKES NO EVENT LOCK. track/stitch/appear are the expensive
GPU steps and none of them read the appearance gallery, so a 75-match backfill can grind
away without blocking a freshly curated match from resolving and publishing. That is the
whole reason detection stops there rather than carrying through to a bundle: the
gallery-dependent minute of work belongs immediately before curation, when the gallery is
as good as it is going to get.

HEARTBEAT COST IS A REAL CONSTRAINT. Free-tier KV allows 1,000 writes a day and each
status post costs two of them (the value, plus the index manifest). So the interval is slow
by default -- 10 min idle, 2 min while running -- and responsiveness comes from posting
IMMEDIATELY on every state change instead of from polling fast. Reading /control is a read
(100k/day), so that stays quick and arming still feels instant.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import os
import platform
import re
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
JOBS_FILE = C.OUT_DIR / "agent_jobs.json"

# While a job runs, status is posted at most this often. A 75-match backfill posting after
# every match would be 150 KV writes against a 1,000/day budget; a job that reports
# nothing for an hour is indistinguishable from a hung one. This is the compromise.
JOB_REPORT_S = 60.0

# How many matches one "next N uncurated" bundle request may cover, whatever the app asks
# for. Each bundle is 2-7 MB in a store that caps values at 25 MiB and expires them in 24
# hours, so an unbounded request would push bundles nobody can reach before they expire.
MAX_BUNDLE_BATCH = 8
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


def _load_jobs() -> dict:
    try:
        return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_jobs(jobs: dict) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=1), encoding="utf-8")


def pending_jobs(agent_id: str) -> list[dict]:
    """Job documents on the relay this agent has not finished.

    Mirrors watch.pending: the relay says what was asked for, local state says what has
    been done, and the difference is the work. A job with no agentId is addressed to
    whoever picks it up, which keeps a single-machine setup from having to know its own id.
    """
    url, _ = R._env()
    import requests
    r = requests.get(f"{url}/index", timeout=60)
    r.raise_for_status()
    ledger = _load_jobs()
    out = []
    for it in r.json().get("items", []):
        if it.get("kind") != "job":
            continue
        if it.get("agentId") not in (None, "", agent_id):
            continue
        if it.get("cancelled"):
            continue
        rec = ledger.get(it.get("id") or "")
        if rec and rec.get("state") in ("done", "failed"):
            continue
        out.append(it)
    return sorted(out, key=lambda it: it.get("at") or 0)


def _detected(match_key: str) -> bool:
    return (C.STAGE1_DIR / f"{match_key}_tracks_stitched.jsonl").exists()


def _curated(match_key: str) -> bool:
    return (C.TRACKER_ROOT / "corrections" / f"{match_key}_corrections.json").exists()


def _bundle_path(match_key: str):
    return C.STAGE3_DIR / f"{match_key}_curate_frames.json"


def run_detect(job: dict, report) -> tuple[int, int, list[str]]:
    """Fetch clips and run track/stitch/appear for each match. Returns (done, total, failed).

    replay runs ONCE for the whole range rather than per match: it makes one TBA call and,
    with --per-match, resolves each match's own upload. --per-match matters for backfill
    beyond correctness of convenience -- 2026necmp1's TBA actual_time is attributed to the
    wrong match from qm5 on, so slicing that event out of the day archive silently clips
    the neighbouring match. A per-match video cannot be off by one.
    """
    from .replay import parse_matches
    event = job["event"]
    wants = parse_matches(job.get("matches") or "")
    keys = [f"{event}_{suf}" for suf in wants]
    todo = [k for k in keys if not _detected(k)]
    report(f"detect {event}: {len(todo)} of {len(keys)} match(es) need work", 0, len(todo))
    if not todo:
        return 0, 0, []

    missing = [k.split("_", 1)[1] for k in todo
               if not (C.RAW_DIR / f"{k}.mp4").exists()]
    if missing:
        ra = [PY, "-m", "rtrack.replay", event, "--matches", ",".join(missing)]
        if job.get("options", {}).get("perMatch", True):
            ra.append("--per-match")
        report(f"fetching {len(missing)} clip(s)", 0, len(todo))
        subprocess.run(ra, cwd=C.TRACKER_ROOT)

    done, failed = 0, []
    for i, key in enumerate(todo):
        if not (C.RAW_DIR / f"{key}.mp4").exists():
            failed.append(key)
            continue
        a = [PY, "-m", "rtrack.pipeline", key, "--match", key,
             "--event", event, "--prep-only"]
        if job.get("calibFrom"):
            a += ["--calib-from", job["calibFrom"]]
        report(f"detecting {key}", i, len(todo))
        if subprocess.run(a, cwd=C.TRACKER_ROOT).returncode == 0:
            done += 1
        else:
            failed.append(key)
    return done, len(todo), failed


def run_bundle(job: dict, report) -> tuple[int, int, list[str]]:
    """Build and push curation bundles for the next N detected-but-uncurated matches.

    Ordered by match number so a curator works forward through the event, which is also
    the order the appearance gallery improves in -- bundling qm40 before qm30 would ask a
    human to label a match with a worse gallery than it needed to have.
    """
    event = job["event"]
    want = min(int(job.get("count") or 3), MAX_BUNDLE_BATCH)

    def num(key: str) -> tuple:
        m = re.search(r"_([a-z]+)(\d+)$", key)
        return (m.group(1), int(m.group(2))) if m else (key, 0)

    candidates = sorted(
        {p.name.split("_tracks_stitched.jsonl")[0]
         for p in C.STAGE1_DIR.glob(f"{event}_*_tracks_stitched.jsonl")},
        key=num)
    todo = [k for k in candidates if not _curated(k)][:want]
    report(f"bundle {event}: {len(todo)} match(es)", 0, len(todo))
    done, failed = 0, []
    for i, key in enumerate(todo):
        report(f"bundling {key}", i, len(todo))
        a = [PY, "-m", "rtrack.pipeline", key, "--match", key, "--event", event]
        if job.get("calibFrom"):
            a += ["--calib-from", job["calibFrom"]]
        # No --relay: that mode blocks for its whole --wait on ONE curator finishing, which
        # is the opposite of a batch. Build locally, push, move on.
        rc = subprocess.run(a, cwd=C.TRACKER_ROOT).returncode
        bundle = _bundle_path(key)
        if rc != 0 or not bundle.exists():
            failed.append(key)
            continue
        push = [PY, "-m", "rtrack.relay", "push-bundle", key, "--file", str(bundle)]
        if subprocess.run(push, cwd=C.TRACKER_ROOT).returncode == 0:
            done += 1
        else:
            failed.append(key)
    return done, len(todo), failed


JOB_RUNNERS = {"detect": run_detect, "bundle": run_bundle}


def run_job(job_item: dict, agent_id: str, on_report) -> None:
    """Fetch one job document, run it, and record the outcome in the ledger."""
    job_id = job_item.get("id") or ""
    doc = R.get("job", job_id)
    if doc is None:
        print(f"[agent] job {job_id}: vanished before it could be fetched", flush=True)
        return
    runner = JOB_RUNNERS.get(doc.get("type") or "")
    jobs = _load_jobs()
    if runner is None:
        jobs[job_id] = {"state": "failed", "at": _now_iso(),
                        "error": f"unknown job type {doc.get('type')!r}"}
        _save_jobs(jobs)
        return

    jobs[job_id] = {"state": "running", "type": doc.get("type"),
                    "event": doc.get("event"), "startedAt": _now_iso()}
    _save_jobs(jobs)
    last = [0.0]

    def report(note: str, i: int, total: int) -> None:
        print(f"[agent] job {job_id[:8]}: {note} ({i}/{total})", flush=True)
        rec = _load_jobs()
        cur = rec.get(job_id, {})
        cur.update({"state": "running", "note": note, "done": i, "total": total})
        rec[job_id] = cur
        _save_jobs(rec)
        # Throttled, because a per-match post would spend the day's write budget.
        if time.time() - last[0] >= JOB_REPORT_S:
            last[0] = time.time()
            on_report()

    try:
        done, total, failed = runner(doc, report)
        jobs = _load_jobs()
        jobs[job_id] = {"state": "done", "type": doc.get("type"), "event": doc.get("event"),
                        "done": done, "total": total, "failed": failed,
                        "finishedAt": _now_iso()}
        _save_jobs(jobs)
        print(f"[agent] job {job_id[:8]}: finished {done}/{total}"
              + (f", {len(failed)} failed" if failed else ""), flush=True)
    except Exception as exc:                          # noqa: BLE001
        # A job that raises must be recorded as failed rather than retried forever: the
        # usual cause is a missing video or a bad calibration, and neither fixes itself.
        jobs = _load_jobs()
        jobs[job_id] = {"state": "failed", "type": doc.get("type"),
                        "event": doc.get("event"),
                        "error": f"{type(exc).__name__}: {exc}", "finishedAt": _now_iso()}
        _save_jobs(jobs)
        print(f"[agent] job {job_id[:8]}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        on_report()


def job_summary(agent_id: str) -> dict:
    """The job ledger, trimmed to what the app needs to render progress."""
    jobs = _load_jobs()
    running = [{"id": k, **v} for k, v in jobs.items() if v.get("state") == "running"]
    recent = sorted(((k, v) for k, v in jobs.items() if v.get("state") != "running"),
                    key=lambda kv: kv[1].get("finishedAt") or "", reverse=True)[:5]
    return {"running": running,
            "recent": [{"id": k, **v} for k, v in recent]}


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
        "jobs": job_summary(agent_id),
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


def startup_cmd_path():
    import os
    return Path(os.environ["APPDATA"]) / ("Microsoft/Windows/Start Menu/Programs/Startup"
                                          "/rtrack-agent.cmd")


def install_task(agent_id: str, extra: list[str] | None = None) -> int:
    """Start the agent at logon, via the Startup folder.

    NOT a scheduled task, and that is a measured decision rather than a preference.
    `schtasks /Create /SC ONLOGON` fails with "ERROR: Access is denied." for a
    non-administrator: Windows treats a LOGON TRIGGER as privileged because it can affect
    other users' sessions. The same command with /SC ONCE succeeds unelevated, which
    isolates it to the trigger rather than to task creation or to the command being built
    wrong. Requiring an elevated shell to set this up would undercut the point -- this
    exists so nobody has to be at the machine.

    The Startup folder needs no elevation, runs in the interactive user session (which the
    supervised watcher needs, since a session-0 task has unreliable GPU access), and is
    trivially reversible: it is one file, and deleting it is the uninstall.
    """
    if platform.system() != "Windows":
        print("[agent] --install-task is Windows-only; on other platforms use a systemd "
              "--user unit or a launchd agent", file=sys.stderr)
        return 2
    target = startup_cmd_path()
    args = f"--agent-id {agent_id}" + (" " + " ".join(extra) if extra else "")
    # `start /min` so the console does not take focus at every logon, and a titled window
    # so it is identifiable in the taskbar rather than being an anonymous python.exe.
    # Joined, not escaped: a .cmd wants CRLF, and newline="" on the write means these are
    # the exact bytes that land on disk.
    # Joined rather than escaped: a .cmd file wants CRLF, and newline="" on the write
    # below means these are the exact bytes that land on disk.
    body = "\r\n".join([
        "@echo off",
        "rem rtrack agent -- listens for relay control and supervises rtrack.watch.",
        "rem Installed by: rtrack.agent --install-task",
        "rem Remove it by deleting THIS FILE, or: rtrack.agent --uninstall-task",
        f'cd /d "{C.TRACKER_ROOT}"',
        f'start "rtrack agent" /min "{PY}" -m rtrack.agent {args}',
        "",
    ])
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with io.open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(body)
    except OSError as exc:
        print(f"[agent] could not write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"[agent] will start at logon as {agent_id}\n"
          f"        {target}\n"
          f"        Remove it with:  uv run python -m rtrack.agent --uninstall-task\n"
          "        This does NOT start it now -- a running agent keeps running, "
          "and starting a second one is refused (see claim_singleton).")
    return 0


def uninstall_task() -> int:
    target = startup_cmd_path() if platform.system() == "Windows" else None
    if target is None:
        print("[agent] --uninstall-task is Windows-only", file=sys.stderr)
        return 2
    if target.exists():
        target.unlink()
        print(f"[agent] removed {target}")
    else:
        print(f"[agent] nothing installed at {target}")
    # Tidy up after the scheduled-task version this replaced, so an earlier install that
    # DID have elevation does not keep launching a second agent.
    subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                   capture_output=True)
    return 0


def _pid_alive(pid: int) -> bool:
    """Whether a pid is running. PermissionError means it exists and is not ours."""
    import os
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def claim_singleton(agent_id: str) -> bool:
    """Refuse to run if another agent for this id already is.

    Two agents sharing an id is not a harmless duplicate: both poll the same control
    document, both reconcile against their OWN watcher handle, and so both spawn a watcher
    for the same event. Two watchers then race for the per-event pipeline lock and take
    turns failing. This became reachable the moment installing at logon was possible while
    an agent was already running by hand.
    """
    lock = C.OUT_DIR / f"agent.{agent_id}.lock"
    try:
        prev = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prev = None
    if prev and _pid_alive(int(prev.get("pid", -1))):
        print(f"[agent] REFUSING: agent {agent_id} is already running as pid "
              f"{prev.get('pid')} (since {prev.get('at')}).\n"
              f"         Stop that one first, or use a different --agent-id.\n"
              f"         If it is dead, delete {lock}", file=sys.stderr)
        return False
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "at": _now_iso()}), encoding="utf-8")
    return True


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
                    help="start the agent at every logon (Startup folder; no admin "
                         "rights needed -- see install_task for why not a scheduled task)")
    ap.add_argument("--uninstall-task", action="store_true",
                    help="undo --install-task")

    g = ap.add_argument_group("operator actions (one-shot, then exit)")
    g.add_argument("--arm", metavar="EVENT", default=None,
                   help="post desired=running for EVENT, as the app will")
    g.add_argument("--disarm", action="store_true",
                   help="post desired=stopped")
    g.add_argument("--show", action="store_true",
                   help="print the current control and status documents")
    g.add_argument("--request", choices=sorted(JOB_RUNNERS), default=None,
                   metavar="TYPE",
                   help="post a job document (detect | bundle), as the app will")
    g.add_argument("--matches", default=None, metavar="SPEC",
                   help="for --request detect: qm26-qm100 or qm1,qm7")
    g.add_argument("--count", type=int, default=3,
                   help="for --request bundle: how many uncurated matches to bundle")
    g.add_argument("--jobs", action="store_true",
                   help="list this machine's job ledger")
    g.add_argument("--calib-from", default=None, metavar="VIDEO",
                   help="calibration stem to pass the watcher when arming")
    g.add_argument("--stream-url", default=None,
                   help="stream URL to record in the control document when arming")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    agent_id = args.agent_id or default_agent_id()

    if args.install_task:
        return install_task(agent_id)
    if args.uninstall_task:
        return uninstall_task()
    if args.show:
        return show(agent_id)
    if args.jobs:
        print(json.dumps(job_summary(agent_id), indent=2))
        return 0
    if args.request:
        if args.request == "detect" and not args.matches:
            print("[agent] --request detect needs --matches, e.g. qm26-qm100",
                  file=sys.stderr)
            return 2
        job_id = uuid.uuid4().hex[:16]
        doc = {"schemaVersion": 1, "jobId": job_id, "agentId": agent_id,
               "type": args.request,
               "event": args.arm or (args.calib_from or None),
               "matches": args.matches, "count": args.count,
               "calibFrom": args.calib_from,
               "requestedBy": f"cli:{socket.gethostname()}", "requestedAt": _now_iso()}
        if not doc["event"]:
            print("[agent] --request needs an event; pass it with --arm EVENT",
                  file=sys.stderr)
            return 2
        R.put("job", job_id, doc)
        print(f"[agent] queued {args.request} job {job_id} for {doc['event']}"
              + (f" {args.matches}" if args.matches else ""))
        return 0
    if args.arm:
        return post_control(agent_id, "running", args.arm,
                            args.calib_from or args.arm, args.stream_url)
    if args.disarm:
        return post_control(agent_id, "stopped", None, None, None)

    R._env()          # fail now, loudly, if the relay is not configured
    if not claim_singleton(agent_id):
        return 4

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

            # AFTER reconcile, so an arm/disarm is never stuck behind a long backfill,
            # and one job at a time: they compete for one GPU, and two detections in
            # flight make both slower rather than finishing either sooner.
            try:
                jobs_todo = pending_jobs(agent_id)
            except Exception as exc:                  # noqa: BLE001
                print(f"[agent] job check failed ({type(exc).__name__}: {exc})", flush=True)
                jobs_todo = []
            if jobs_todo:
                beat = lambda: post_status(agent_id, watcher, desired, applied_nonce,
                                           "running" if watcher.alive else "working",
                                           last_error)
                run_job(jobs_todo[0], agent_id, beat)
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
