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
from collections import defaultdict, deque
import math
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
CONTROL_POLL_S = 10.0             # a READ; puts an arm/disarm receipt within ~12 s

# STATUS POST BUDGET, and the arithmetic that was got wrong once.
#
# Free-tier KV allows 1,000 WRITES A DAY, and every put here costs TWO of them -- the value
# plus the index manifest that touchIndex maintains. So the real budget is ~500 puts/day
# across everything: curation bundles, answers, exported routes, gallery bundles AND status.
#
# The first cut of this set a 2-minute heartbeat while running, reasoning that ~300-400
# writes a day was comfortable. It counted one channel and forgot the doubling: 2 minutes is
# 720 posts = 1,440 writes from the heartbeat alone, over the cap before any real work. The
# quota duly ran out mid-event, the worker threw 1101 on every write, and a curator could
# not send answers back -- status chatter had crowded out the actual deliverable.
#
# So status is now RATIONED against an explicit daily budget, with real work given priority
# by simply leaving it out of the accounting: bundles, answers and routes are never
# throttled, they just spend from the same pool. The budget is deliberately well under
# 500 puts to leave that room.
#
# NOW ON THE WORKERS PAID PLAN: 1,000,000 writes a MONTH (~33,000/day), confirmed live by a
# write succeeding at 18:38 UTC on the same UTC day the free quota ran out at 17:15. The
# doubling above still applies, so the arithmetic is: a 30 s heartbeat for a whole day is
# 2,880 posts = 5,760 writes, ~173k a month. Real work sits on top of that comfortably.
#
# The ration is KEPT, raised to a runaway guard rather than a squeeze. Overage on paid is
# $5 per million writes -- cheap, but a loop posting every iteration would still find it,
# and a cap that never binds in normal use costs nothing. 6,000 posts/day is 12,000 writes,
# 360k a month at worst, leaving well over half the included million for everything else.
STATUS_POSTS_PER_DAY = 6000
HEARTBEAT_IDLE_S = 300.0            # 5 min
HEARTBEAT_RUNNING_S = 30.0          # the lag a lead scout sees in the relay panel
HEARTBEAT_FLOOR_S = 10.0            # a burst of state changes still cannot flood

STATE_FILE = C.OUT_DIR / "agent_state.json"
JOBS_FILE = C.OUT_DIR / "agent_jobs.json"

# While a job runs, status is posted at most this often. A 75-match backfill posting after
# every match would be 150 KV writes against a 1,000/day budget; a job that reports
# nothing for an hour is indistinguishable from a hung one. This is the compromise.
# Job progress shares the heartbeat's ration rather than having a cadence of its own.
# Two independent timers on one quota is how the budget was overrun: each looked modest
# alone and together they were double.
JOB_REPORT_S = HEARTBEAT_RUNNING_S

# How many matches one "next N uncurated" bundle request may cover, whatever the app asks
# for. Each bundle is 2-7 MB in a store that caps values at 25 MiB and expires them in 24
# hours, so an unbounded request would push bundles nobody can reach before they expire.
MAX_BUNDLE_BATCH = 8

# How many uncurated bundles a `process` job will leave on the relay at once. This is the
# throttle that lets ONE request cover a whole range: bundles are 2-7 MB and die after 24
# hours, so pushing 72 of them would expire most unread. Instead the job keeps a few in
# flight and pushes the next as each is answered, which also means the curator always has
# work without ever having a backlog they cannot reach.
MAX_OUTSTANDING_BUNDLES = 4

# Detection is minutes per match and blocks this pass, so a pass does a bounded number and
# returns. That keeps arm/disarm responsive -- a 72-match request must not make the control
# poll wait hours.
DETECT_PER_PASS = 3

# Strikes before a match is set aside. Three passes is enough to ride out a transient
# (a flaky download, a momentarily locked event) without grinding on a real fault.
MAX_MATCH_STRIKES = 3
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
        # --any-event: a curated bundle is work a human already did, so leaving it
        # unprocessed because the machine happens to be armed on another event just
        # loses it. --event still scopes which calibration --calib-from applies to.
        a = [PY, "-m", "rtrack.watch", "--event", event, "--any-event"]
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
        if it.get("jobType") and it["jobType"] not in JOB_RUNNERS:
            # A job type this agent's code does not know yet -- the app was updated before
            # the home machine was restarted. Leave it on the relay for an agent that
            # does, rather than failing it for good, which is what happened to the first
            # `followup` request.
            continue
        rec = ledger.get(it.get("id") or "") or None
        if rec and rec.get("state") in ("done", "failed", "superseded"):
            continue
        out.append(it)
    return sorted(out, key=lambda it: it.get("at") or 0)


def relay_bundle_state() -> tuple[dict, dict]:
    """(bundle timestamp by match, answer timestamp by match) from one /index read."""
    url, _ = R._env()
    import requests
    r = requests.get(f"{url}/index", timeout=60)
    r.raise_for_status()
    bundles, answers = {}, {}
    for it in r.json().get("items", []):
        kind, ident, at = it.get("kind"), it.get("id") or "", (it.get("at") or 0) / 1000.0
        if kind == "bundle":
            bundles[ident] = at
        elif kind == "answer":
            answers[ident] = at
    return bundles, answers


def outstanding_count(bundles: dict, answers: dict) -> int:
    """Bundles nobody has answered yet -- uncurated work already sitting on the relay.

    Counted across EVERY event on purpose. The cap exists to protect a person's attention
    and a 24-hour TTL, and neither is per-event: ten stale mawor bundles are exactly as
    much unreachable work as ten necmp1 ones.
    """
    return sum(1 for key, at in bundles.items()
               if answers.get(key, 0.0) <= at and not _is_followup(key, at))


def _is_followup(key: str, relay_at: float) -> bool:
    """Is the bundle on the relay for this match a FOLLOW-UP (rtrack.agent run_followup)?

    Follow-ups are asked for by a person, one match at a time, so they go around the
    outstanding-bundle cap rather than queueing behind it or displacing a first-round
    bundle from it. Recognised by the local follow-up file being at least as new as the
    relay copy (the push happens straight after it is written).
    """
    p = C.STAGE3_DIR / f"{key}_followup_frames.json"
    return p.exists() and p.stat().st_mtime >= relay_at - 120


def next_job(pending: list[dict]) -> dict | None:
    """Which pending job gets this loop's turn.

    NOT simply the oldest, which is what it used to be and which starved everything: a
    `process` job deliberately stays unfinished until every match in its range is
    curated -- days, for a 100-match event -- so as the oldest job it won every loop,
    and a calibration request queued behind it never ran at all. Now the job that has
    waited longest since its LAST turn goes next. A job that has never run has waited
    forever, so a new request always goes next, and long-running jobs take turns.

    Identical `process` requests are also collapsed here. Pressing Process again -- to
    change the bundle cap, say -- queued a second copy rather than replacing the first;
    2026necmp1 had four for qm1-qm100. The NEWEST is kept, since its settings are the
    ones the person meant, and the older copies are retired as superseded. Nothing is
    lost by switching: progress is read from disk and the relay, not from the job id.
    """
    if not pending:
        return None
    ledger = _load_jobs()
    groups: dict[tuple, list[dict]] = {}
    for it in pending:
        if it.get("jobType") == "process":
            groups.setdefault((it.get("event"), it.get("matches")), []).append(it)
    retired = set()
    for same in groups.values():
        if len(same) < 2:
            continue
        same.sort(key=lambda it: it.get("at") or 0)
        keep = same[-1]
        for old in same[:-1]:
            rec = ledger.get(old["id"], {})
            rec.update({"state": "superseded", "supersededBy": keep["id"],
                        "finishedAt": _now_iso(), "type": "process",
                        "event": old.get("event")})
            ledger[old["id"]] = rec
            retired.add(old["id"])
            print(f"[agent] job {old['id'][:8]} superseded by {keep['id'][:8]} "
                  f"(same range, newer request)", flush=True)
    if retired:
        _save_jobs(ledger)
    live = [it for it in pending if it["id"] not in retired]
    return min(live, key=lambda it: (float(ledger.get(it["id"], {}).get("lastRunAt") or 0),
                                     it.get("at") or 0)) if live else None


def _detected(match_key: str) -> bool:
    return (C.STAGE1_DIR / f"{match_key}_tracks_stitched.jsonl").exists()


def _corrections_exist(match_key: str) -> bool:
    """A human has answered this match, whatever became of the route afterwards."""
    return (C.TRACKER_ROOT / "corrections" / f"{match_key}_corrections.json").exists()


def _curated(match_key: str) -> bool:
    """Has this match reached a PUBLISHED ROUTE at least as new as its corrections?

    NOT "does a corrections file exist", which is what this used to ask and which
    stranded matches. 2026necmp1_qm24 had corrections from Sep 20 and no route: the
    process job read the file, called the match done and skipped it on every pass, while
    its relay answer had long since expired so the watcher could not see it either. Both
    halves believed the other owned it and nothing ever published it.

    Same test rtrack.watch.pending uses, for the same reason: the deliverable is the
    route, so only the route finishing counts as finished.
    """
    corr = C.TRACKER_ROOT / "corrections" / f"{match_key}_corrections.json"
    if not corr.exists():
        return False
    published = C.REPO_ROOT / "public" / "tracks" / f"{match_key}.json"
    if not published.exists():
        return False
    try:
        return published.stat().st_mtime + 1 >= corr.stat().st_mtime
    except OSError:
        return False


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
        t_start = time.time()
        rc, btail = _run_logged(a, cwd=C.TRACKER_ROOT)
        if rc == 3:
            report("event busy (lock held); will retry", i, len(todo))
            break
        bundle = _bundle_path(key)
        # Curated meanwhile: the pipeline published, and the file on disk is an OLD
        # bundle the curator already answered (see run_process). Never push it.
        if rc == 0 and _corrections_exist(key):
            continue
        if rc != 0 or not bundle.exists() or bundle.stat().st_mtime < t_start - 5:
            print(f"[agent] {key} bundle failed: "
                  + (failure_reason(btail, rc) if rc != 0
                     else "pipeline succeeded but produced no curation bundle"), flush=True)
            failed.append(key)
            continue
        push = [PY, "-m", "rtrack.relay", "push-bundle", key, "--file", str(bundle)]
        if subprocess.run(push, cwd=C.TRACKER_ROOT).returncode == 0:
            done += 1
        else:
            failed.append(key)
    return done, len(todo), failed


def process_plan(keys: list[str], event: str, bundles: dict, answers: dict,
                 cap: int, out: int, failed: list[str] | None = None) -> dict:
    """Per-match categories for one process job. Mutually exclusive, suffixes only.

    Built BEFORE the pass as well as after it. Attaching the breakdown only to a pass's
    final report meant the app had nothing to show for the whole pass -- and a pass that
    publishes a backlog runs for many minutes, so `plan: ABSENT` was the normal state
    rather than a rare one.
    """
    failed = failed or []
    strikes = _load_jobs().get("_strikes", {})
    plan = {"cap": cap, "outstanding": out,
            "failed": [k.split("_", 1)[1] for k in failed],
            "setAside": sorted(k.split("_", 1)[1] for k, n in strikes.items()
                               if n >= MAX_MATCH_STRIKES and k.startswith(event + "_")),
            "curated": [], "awaitingCuration": [], "awaitingPublish": [],
            "readyBlocked": [], "queued": []}
    stored = _load_jobs().get("_reasons", {})
    plan["reasons"] = {k.split("_", 1)[1]: (v.get("stage", "") + ": " + v.get("reason", ""))
                       for k, v in stored.items() if k.startswith(event + "_")
                       and k.split("_", 1)[1] in set(plan["failed"]) | set(plan["setAside"])}
    for key in keys:
        suf = key.split("_", 1)[1]
        if suf in plan["failed"] or suf in plan["setAside"]:
            continue
        if _curated(key):
            plan["curated"].append(suf)
        elif _corrections_exist(key) or answers.get(key, 0.0) > bundles.get(key, 0.0) > 0.0:
            # answered on the relay but not yet pulled down: the watcher's to publish
            plan["awaitingPublish"].append(suf)
        elif bundles.get(key, 0.0) > answers.get(key, 0.0):
            plan["awaitingCuration"].append(suf)
        elif _detected(key):
            plan["readyBlocked"].append(suf)
        else:
            plan["queued"].append(suf)
    return plan


def _run_logged(args: list, **kw) -> tuple[int, list[str]]:
    """Run a step, streaming its output to the agent log and keeping the last lines.

    Steps used to run with inherited output, so the agent saw only an exit code: a match
    failed and the ledger said so with nothing about WHY. The cause was in the log, often
    far from the failure and never attached to the match. Keeping the tail lets
    failure_reason() pull the cause out and store it against the match.
    """
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
    tail: deque[str] = deque(maxlen=60)
    for line in proc.stdout:
        line = line.rstrip("\n")
        if re.match(r"\s*\[download\]\s+[\d.]+%", line) and "ERROR:" not in line:
            continue
        print(line, flush=True)
        tail.append(line)
    return proc.wait(), list(tail)


# Most specific first. The pipeline's own verdicts beat a generic last line, and a refusal
# names the actual problem ("title says 40, 0/6 teams agree") where an exit code cannot.
_REASON_PATTERNS = (
    r"REFUSING[^:]*:\s*(.+)",
    r"!!\s*(.+)",
    r"per-match download failed:\s*(.+)",
    r"(ERROR:.+)",
    r"(HOLDING THE WRONG MATCH.*)",
)


def failure_reason(tail: list[str], rc: int | None = None) -> str:
    for pat in _REASON_PATTERNS:
        for line in reversed(tail):
            m = re.search(pat, line)
            if m:
                return m.group(1).strip()[:200]
    last = next((l.strip() for l in reversed(tail) if l.strip()), "")
    return (last or f"exited {rc}")[:200]


def control_pending(agent_id: str | None) -> bool:
    """Is there a control command this agent has not applied yet?

    The main loop reads /control only BETWEEN job passes, and a pass blocks for minutes --
    up to three fetch-and-detect cycles plus full pipeline runs for bundling and publishing.
    A lead scout pressed Stop, then Start, and the panel reported "asked to run 2026necmp1
    but the machine reports working" for the rest of the pass: both commands sat unread.

    So a pass checks this between matches and gives way. One GET per match, a read; it
    bounds the delay to whatever single step is already running rather than a whole pass.
    """
    if not agent_id:
        return False
    try:
        doc = R.get("control", agent_id)
    except Exception:                             # noqa: BLE001
        return False                              # an unreachable relay is not a command
    if not doc or not doc.get("nonce"):
        return False
    return doc["nonce"] != _load_state().get("appliedNonce")


def run_process(job: dict, report) -> tuple[int, int, list[str], bool]:
    """Take a range of matches from nothing to a curation bundle, then keep it flowing.

    ONE REQUEST PER MATCH, which is the whole point: asking for qm29-qm100 once should
    eventually produce a curated route for every one of them without anybody pressing
    anything again. So this job does not run to completion and exit -- it does a bounded
    pass and reports itself unfinished, and the agent re-enters it each poll until every
    requested match is curated.

    The loop per match is: detect if needed (prep-only, no event lock), then push a
    curation bundle if fewer than the cap are already waiting. A human answers one, the
    watcher turns that answer into a published route and writes the corrections file, and
    the next pass sees that match as curated and pushes the next bundle in its place. The
    backpressure is the curator's own pace, which is the only rate that matters.

    Returns (curated, total, failed, complete).
    """
    from .replay import parse_matches
    event = job["event"]
    keys = [f"{event}_{suf}" for suf in parse_matches(job.get("matches") or "")]
    if not keys:
        return 0, 0, [], True
    cap = max(1, min(int(job.get("maxOutstanding") or MAX_OUTSTANDING_BUNDLES), 12))

    # A match that fails repeatedly is skipped rather than retried forever. Every
    # 2026necmp1 bundle failed at the votes step for days, and each pass spent a full
    # pipeline run per match re-discovering that -- hours of GPU producing nothing, with
    # only a bare "4 failed" to show for it. After this many strikes the match is set
    # aside and NAMED, so the cause gets looked at instead of ground against.
    strikes = _load_jobs().get("_strikes", {})
    keys = [k for k in keys if strikes.get(k, 0) < MAX_MATCH_STRIKES]

    curated = [k for k in keys if _curated(k)]
    if len(curated) == len(keys):
        report(f"all {len(keys)} match(es) curated", len(keys), len(keys))
        return len(keys), len(keys), [], True

    try:
        bundles, answers = relay_bundle_state()
    except Exception as exc:                      # noqa: BLE001
        report(f"relay unreachable ({type(exc).__name__}); will retry", len(curated), len(keys))
        return len(curated), len(keys), [], False

    out = outstanding_count(bundles, answers)
    report(f"{len(curated)}/{len(keys)} curated", len(curated), len(keys),
           process_plan(keys, event, bundles, answers, cap, out))
    failed, detected_now, pushed, attempts = [], 0, 0, 0
    why: dict[str, dict] = {}         # key -> {stage, reason}, persisted to the ledger below
    fetch_note: dict[str, str] = {}   # key -> why the per-match download failed, if it did

    def fail(key: str, stage: str, reason: str) -> None:
        # If the right video could not be fetched and a fallback was used, that is part of
        # the story of any later failure -- it is how qm39 came to hold qm40's footage.
        if key in fetch_note and stage != "fetch":
            reason = f"{reason} [per-match download had failed: {fetch_note[key]}]"
        failed.append(key)
        why[key] = {"stage": stage, "reason": reason[:300], "at": _now_iso()}
    # ATTEMPTS, not successes. A failed push does not consume the cap, so bounding
    # only successes let one systematically broken match carry the pass through every
    # remaining bundle in the range -- a full pipeline run each, for nothing. A dry
    # run measured 11 attempts against a cap of 4.
    attempt_budget = max(1, cap - out) + 2

    for key in keys:
        if _curated(key):
            continue
        # Give way to a command the moment one is waiting, rather than at the end of a pass.
        # The job is not lost: it stays `running` in the ledger and resumes next iteration,
        # after the main loop has applied the command.
        if control_pending(job.get("agentId")):
            report("pausing: a new control command is waiting", len(curated), len(keys))
            break
        if not _detected(key):
            if detected_now >= DETECT_PER_PASS:
                break                             # yield to the control poll
            suffix = key.split("_", 1)[1]
            if not (C.RAW_DIR / f"{key}.mp4").exists():
                report(f"fetching {suffix}", len(curated), len(keys))
                ra = [PY, "-m", "rtrack.replay", event, "--matches", suffix]
                if job.get("options", {}).get("perMatch", True):
                    ra.append("--per-match")
                rrc, rtail = _run_logged(ra, cwd=C.TRACKER_ROOT)
                note = next((re.search(r"per-match download failed:\s*(.+)", l).group(1)
                             for l in rtail if "per-match download failed:" in l), None)
                if note:
                    fetch_note[key] = note.strip()[:200]
            if not (C.RAW_DIR / f"{key}.mp4").exists():
                fail(key, "fetch", fetch_note.get(key) or "no clip could be fetched")
                continue
            report(f"detecting {suffix}", len(curated), len(keys))
            a = [PY, "-m", "rtrack.pipeline", key, "--match", key,
                 "--event", event, "--prep-only"]
            if job.get("calibFrom"):
                a += ["--calib-from", job["calibFrom"]]
            detected_now += 1
            drc, dtail = _run_logged(a, cwd=C.TRACKER_ROOT)
            if drc != 0:
                reason = failure_reason(dtail, drc)
                # A clip the identity check REJECTS is moved aside, so the next pass
                # downloads afresh instead of re-failing on the same bytes forever. That is
                # how 2026necmp1_qm39 got stuck: the fetch step saw a file already present
                # and never retried, so qm40's footage was re-checked and re-rejected every
                # pass until the match was set aside. Renamed, never deleted.
                if "teams agree" in reason:
                    bad = C.RAW_DIR / f"{key}.mp4"
                    aside = bad.with_name(f"{bad.name}.rejected-{int(time.time())}")
                    try:
                        bad.rename(aside)
                        reason += f" [clip moved aside to {aside.name} for a fresh download]"
                    except OSError:
                        pass
                fail(key, "detect", reason)
                continue

        # ALREADY CURATED BUT NOT PUBLISHED: solve and publish, never re-bundle.
        #
        # This is the state that the old _curated test hid and the new one exposes: 19
        # 2026necmp1 matches have a corrections file and no route. The work a human did
        # still exists on disk, so asking them to curate the same match a second time
        # would be wasting it. Running the pipeline with corrections present re-solves and
        # publishes instead of building a bundle, which is exactly what is wanted -- and it
        # consumes no bundle slot, because it produces no bundle.
        if _corrections_exist(key):
            # WHOSE JOB IS THIS? If the answer is still on the relay, the watcher already
            # owns this match and will publish it -- both of us running rtrack.pipeline on
            # the same event just fights over the per-event lock, and the loser's run is
            # wasted. Every publish in the first pass after this path was added failed
            # exactly that way: "REFUSING: 2026necmp1_qm32 has been running on this event".
            #
            # So this path handles only what the watcher CANNOT see: a match whose answer
            # has expired off the relay, which is precisely the stranded case it exists for.
            if answers.get(key, 0.0) > 0.0:
                continue
            report(f"publishing {key.split('_', 1)[1]} from existing corrections",
                   len(curated), len(keys))
            a = [PY, "-m", "rtrack.pipeline", key, "--match", key, "--event", event]
            if job.get("calibFrom"):
                a += ["--calib-from", job["calibFrom"]]
            rc, ptail = _run_logged(a, cwd=C.TRACKER_ROOT)
            if rc == 3:
                # The event lock is held. Transient by definition, so NOT a failure and
                # NOT a strike -- benching a match for losing a race would eventually set
                # aside the whole event. Stop taking the lock this pass and try next time.
                report("event busy (lock held); will retry", len(curated), len(keys))
                break
            if rc != 0 or not _curated(key):
                fail(key, "publish", failure_reason(ptail, rc) if rc != 0
                     else "pipeline succeeded but no current route was published")
            continue

        # A bundle already waiting for this match is not re-pushed: that would reset a
        # curator's 24-hour window and churn the relay for no gain.
        if bundles.get(key, 0.0) > answers.get(key, 0.0):
            continue
        # AN ANSWER ON THE RELAY means the match is curated and the watcher owns it, even
        # when no corrections file exists yet (the watcher may be waiting on the event
        # lock). Bundling it anyway re-pushed qm49's and qm50's already-answered bundles
        # a minute after the curator sent them, so both read "bundle waiting · re-curate".
        if answers.get(key, 0.0) > 0.0:
            continue
        t_start = time.time()
        if out >= cap or attempts >= attempt_budget:
            continue
        attempts += 1
        suffix = key.split("_", 1)[1]
        report(f"bundling {suffix}", len(curated), len(keys))
        a = [PY, "-m", "rtrack.pipeline", key, "--match", key, "--event", event]
        if job.get("calibFrom"):
            a += ["--calib-from", job["calibFrom"]]
        # Logged like every other step. This one used to run bare and strike with no
        # reason, which is how 2026necmp1 qm45-qm65 were set aside with nothing to say why.
        rc, btail = _run_logged(a, cwd=C.TRACKER_ROOT)
        if rc == 3:
            # Event lock held (the watcher publishing a curated answer, usually).
            # Transient: not a strike, same as the publish path above.
            report("event busy (lock held); will retry", len(curated), len(keys))
            break
        bundle = _bundle_path(key)
        # Curated while this ran: the pipeline published instead of bundling, and the
        # bundle file on disk is the OLD one. Never push it.
        if rc == 0 and _corrections_exist(key):
            continue
        if rc != 0 or not bundle.exists() or bundle.stat().st_mtime < t_start - 5:
            fail(key, "bundle", failure_reason(btail, rc) if rc != 0
                 else "pipeline succeeded but wrote no new curation bundle")
            continue
        push = [PY, "-m", "rtrack.relay", "push-bundle", key, "--file", str(bundle)]
        prc, ptail = _run_logged(push, cwd=C.TRACKER_ROOT)
        if prc == 4:
            # The relay's daily write quota is spent. The bundle is built and on disk, so
            # nothing is lost -- stop pushing until the quota resets rather than burning
            # strikes on matches that are fine.
            report("relay write quota spent; pausing pushes until 00:00 UTC",
                   len(curated), len(keys))
            break
        if prc != 0:
            fail(key, "push", failure_reason(ptail, prc))
            continue
        out += 1
        pushed += 1

    # Re-read the relay after the pass: bundles pushed a moment ago are part of the
    # picture, and reusing the pre-pass snapshot would report them as still queued.
    try:
        bundles, answers = relay_bundle_state()
        out = outstanding_count(bundles, answers)
    except Exception:                             # noqa: BLE001
        pass

    # REASONS ARE KEPT WITH THE STRIKES. A match set aside after three failures used to be
    # named with no explanation, which left the next step -- "why?" -- to a log search.
    rec = _load_jobs()
    reasons = dict(rec.get("_reasons", {}))
    for k in keys:
        if k not in why and (_curated(k) or (_detected(k) and k not in failed)):
            reasons.pop(k, None)          # it has since got further: the old reason is stale
    reasons.update(why)
    rec["_reasons"] = reasons
    _save_jobs(rec)
    for k, w in why.items():
        print(f"[agent] {k} failed at {w['stage']}: {w['reason']}", flush=True)
    if failed:
        rec = _load_jobs()
        st = dict(rec.get("_strikes", {}))
        for k in failed:
            st[k] = int(st.get(k, 0)) + 1
        rec["_strikes"] = st
        _save_jobs(rec)
        benched = [k for k, n in st.items() if n >= MAX_MATCH_STRIKES]
        if benched:
            print(f"[agent] set aside after {MAX_MATCH_STRIKES} failures: "
                  f"{', '.join(sorted(benched))}", flush=True)

    plan = process_plan(keys, event, bundles, answers, cap, out, failed)

    report(f"{len(curated)}/{len(keys)} curated \u00b7 {out} bundle(s) awaiting curation"
           + (f" \u00b7 pushed {pushed}" if pushed else "")
           + (f" \u00b7 {len(failed)} failed" if failed else ""),
           len(curated), len(keys), plan)
    return len(curated), len(keys), failed, False


# ── CALIBRATION REVIEW ──────────────────────────────────────────────────────
#
# A calibration can be pulled from the app, corrected or extended on a phone, and sent
# back -- without anyone at this machine. The phone side (public/rtrack/calibrate.html)
# already loads the existing points and lets AprilTags be added to them; what was missing
# was any way to ASK for the bundle, and anything that applied the points that came back.
#
# Requesting is a job (calib). Applying is standing behaviour, like the watcher and
# curation answers: whenever the relay holds points newer than this camera's calibration,
# they are applied, the floor is checked, and the refreshed bundle goes back up so the
# person who sent them sees the result and can refine again.


def _camera_footage(camera: str, hint: str | None = None) -> str | None:
    """A clip to grab the plate from. The calibration is named for the camera; the
    footage is some match shot on it. 2026necmp1's doc even says video=2026necmp1 since
    its rename, and no clip has that name -- so try, in order: an explicit hint, what the
    agent used last time, the doc's own field, then the earliest clip for the camera."""
    for cand in (hint, _load_state().get("calibFootage", {}).get(camera)):
        if cand and (C.RAW_DIR / f"{cand}.mp4").exists():
            return cand
    doc_p = C.CALIB_DIR / f"{camera}.json"
    if doc_p.exists():
        try:
            v = json.loads(doc_p.read_text(encoding="utf-8")).get("video")
            if v and (C.RAW_DIR / f"{v}.mp4").exists():
                return v
        except (OSError, json.JSONDecodeError):
            pass
    def num(p):
        m = re.search(r"_qm(\d+)$", p.stem)
        return int(m.group(1)) if m else 10**6
    clips = sorted(C.RAW_DIR.glob(f"{camera}_qm*.mp4"), key=num)
    return clips[0].stem if clips else None


def _calib_state(camera: str, **fields) -> None:
    st = _load_state()
    cal = dict(st.get("calibration", {}))
    cal[camera] = {**cal.get(camera, {}), **fields}
    st["calibration"] = cal
    _save_state(st)


def publish_calib_bundle(camera: str, footage: str) -> tuple[bool, str]:
    """Build the review bundle (plate + existing points + last fit + overlay) and push it."""
    rc, tail = _run_logged([PY, "-m", "rtrack.calibrate", footage, "--camera", camera,
                            "--export-frame"], cwd=C.TRACKER_ROOT)
    if rc != 0:
        return False, failure_reason(tail, rc)
    rc, tail = _run_logged([PY, "-m", "rtrack.relay", "push-calib", camera],
                           cwd=C.TRACKER_ROOT)
    if rc != 0:
        return False, failure_reason(tail, rc)
    return True, ""


def run_calib(job: dict, report) -> tuple[int, int, list[str], bool]:
    """Put a camera's calibration on the relay for review. The job the app requests."""
    camera = job.get("camera") or job.get("event")
    if not camera:
        raise RuntimeError("calib job names no camera")
    footage = _camera_footage(camera, job.get("video"))
    if not footage:
        raise RuntimeError(f"no footage for {camera}: need a clip in data/raw to take "
                           f"the plate from")
    report(f"building the review bundle for {camera} from {footage}", 0, 1)
    ok, why = publish_calib_bundle(camera, footage)
    if not ok:
        raise RuntimeError(why)
    _calib_state(camera, footage=footage, bundlePostedAt=_now_iso(), state="awaiting points")
    st = _load_state(); st.setdefault("calibFootage", {})[camera] = footage; _save_state(st)
    report(f"{camera} is on the relay: open it from Cameras to review", 1, 1)
    return 1, 1, [], True


# ── FOLLOW-UP CURATION ──────────────────────────────────────────────────────
#
# A small second bundle for ONE already-curated match, at moments a person chose while
# watching its routes -- plus, optionally, the middle of the longest stretches of each
# route since that team was last labelled. Measured on 2026necmp1, curation fixes the
# moments it labels and little else (held-out per-image accuracy 89.7% uncurated against
# 90.9% curated), and no automatic signal located the remaining errors well (solver
# uncertainty covered 5% of wrong route time; robot contact is near 86% of ALL route
# time). A person watching the route is the detector; this makes asking cheap.
#
# The answer comes back through the normal relay path and rtrack.relay MERGES it into the
# match's corrections, so the first round's labels survive; the watcher then re-solves.
FOLLOWUP_MAX_FRAMES = 15
FOLLOWUP_SNAP_S = 1.0        # a requested moment snaps to the busiest frame this close
# EACH FLAG ANCHORS A SPAN, not an instant. A person flags where a route LOOKS wrong,
# which is rarely the exact moment it went wrong -- and a swap is fixed by labels on
# BOTH sides of it: pin inheritance (rtrack.robots.inherit_sandwich_pins) then carries
# the team across the pieces in between. So a flag asks about the start, middle and end
# of +-5 s around it -- wider than the 5 s leading up to the slider that the route
# inspector draws, because the labels must also land AFTER whatever went wrong.
FOLLOWUP_SPAN_S = 10.0
FOLLOWUP_PER_FLAG = 3
JUMP_MIN_MS = 4.5            # the inspector's "fast" line; slower steps are not a jump
JUMP_PAIR_S = 0.5            # ends closer than this are one moment: show one frame
JUMP_RUN_GAP_S = 0.5         # a pause this long ends the run of samples at a jump end


def _followup_frames(key: str, times: list, gaps: int) -> tuple[list[int], list[str], dict]:
    lab = C.STAGE3_DIR / f"{key}_labeled.jsonl"
    rj = C.STAGE3_DIR / f"{key}_robots.json"
    if not (lab.exists() and rj.exists()):
        raise RuntimeError(f"{key} has no solved routes to follow up")
    win = json.loads(rj.read_text(encoding="utf-8")).get("custodyWindow") or [0.0]
    off = float(win[0])                       # match t = 0 in video seconds
    rows = [json.loads(l) for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
    busy = [(r["t"], r["f"], sum(1 for d in r["dets"] if d["tid"] >= 0)) for r in rows]
    by_f = {r["f"]: r for r in rows}
    chosen: list[tuple[float, int, str]] = []
    focus_boxes: dict[int, list] = {}          # frame -> the flagged robots' boxes in it

    def team_boxes(f: int, teams: set) -> list:
        return [d["xyxy"] for d in by_f[f]["dets"] if str(d.get("team")) in teams]

    def take(vt: float, why: str, teams: set | None = None) -> None:
        # WITH A ROBOT OF INTEREST, only frames where its route claims a detection are
        # worth asking about: a frame without that robot cannot confirm or falsify its
        # route. Largest box first -- the most legible look at the claim. Widen once
        # before giving up, and say so when the robot is not in view at all.
        if teams:
            for radius in (FOLLOWUP_SNAP_S, 2 * FOLLOWUP_SNAP_S):
                cands = []
                for t, f, _n in busy:
                    if abs(t - vt) <= radius:
                        bx = team_boxes(f, teams)
                        if bx:
                            area = max((b[2] - b[0]) * (b[3] - b[1]) for b in bx)
                            cands.append((area, -abs(t - vt), t, f))
                if cands:
                    _a, _d, t, f = max(cands)
                    if all(abs(t - c[0]) > FOLLOWUP_SNAP_S for c in chosen):
                        chosen.append((t, f, why))
                        focus_boxes[f] = team_boxes(f, teams)
                    return
            why += f" -- {'/'.join(sorted(teams))} not in view"
        near = [b for b in busy if abs(b[0] - vt) <= FOLLOWUP_SNAP_S and b[2] >= 1]
        if not near:
            return
        t, f, _n = max(near, key=lambda b: (b[2], -abs(b[0] - vt)))
        if all(abs(t - c[0]) > FOLLOWUP_SNAP_S for c in chosen):
            chosen.append((t, f, why))

    # THE JUMP ITSELF, not just the span around it. 2026necmp1_qm6: 5813's route jumps
    # 9.6 m at 135.4 s because ONE stray box (frame 5172) was pinned to it. The anchors
    # each took 5813's largest box and so showed only its correct detections; the one
    # wrong box was never asked about, and the re-curation changed nothing. So for each
    # flagged team, the fastest step of its route within the span contributes the frames
    # at both ends -- the frame holding the suspect box is the one a label can falsify.
    pp = C.STAGE2_DIR / f"{key}_positions.json"
    track_pos: dict[str, list] = defaultdict(list)
    if pp.exists():
        seen_f = set()
        for s in json.loads(pp.read_text(encoding="utf-8")).get("samples") or []:
            if s.get("team") and (s["team"], s["f"]) not in seen_f:
                seen_f.add((s["team"], s["f"]))
                track_pos[str(s["team"])].append((s["t"], s["f"], s["x"], s["y"], s["tid"]))
        for v in track_pos.values():
            v.sort()

    def jump_frames(vt: float, team: str) -> list[tuple[float, int, float, float]]:
        v = [p for p in track_pos.get(team, []) if abs(p[0] - vt) <= FOLLOWUP_SPAN_S / 2]
        best = None
        for a, b in zip(v, v[1:]):
            dt = b[0] - a[0]
            if dt <= 0 or dt > FOLLOWUP_SPAN_S / 2:
                continue
            d = math.hypot(b[2] - a[2], b[3] - a[3])
            if d / max(dt, 0.2) >= JUMP_MIN_MS and (best is None or d / dt > best[0]):
                best = (d / dt, d, a, b)
        if best is None:
            return []
        _v, d, a, b = best
        if b[0] - a[0] > JUMP_PAIR_S:
            ends = [a, b]
        else:
            # Both ends in one moment: show the end on the SHORTER RUN -- the stretch of
            # samples reaching it with no pause over JUMP_RUN_GAP_S -- which is the
            # likelier stray. Not the shorter track: on qm6 the stray is the tail of a
            # long piece, after a 3.4 s gap; on qm10 it is two samples at the head of
            # one. Its frame usually carries the other end's robot as well.
            full = track_pos[team]
            i = full.index(a)
            ra = i
            while ra > 0 and full[ra][0] - full[ra - 1][0] <= JUMP_RUN_GAP_S:
                ra -= 1
            rb = i + 1
            while rb + 1 < len(full) and full[rb + 1][0] - full[rb][0] <= JUMP_RUN_GAP_S:
                rb += 1
            ends = [b if (rb - i) < (i + 1 - ra) else a]
        return [(p[0], p[1], d, b[0] - a[0]) for p in ends]

    for flag in times:
        # a flag is {t, teams} from the app, or a bare time from an older app
        t0 = float(flag["t"]) if isinstance(flag, dict) else float(flag)
        teams = ({str(x) for x in flag.get("teams") or []} if isinstance(flag, dict) else set())
        who = f" {'/'.join(sorted(teams))}" if teams else ""
        if isinstance(flag, dict) and flag.get("exact"):
            # EXACTLY THIS FRAME: the curator picked it in the route inspector, often to
            # re-review the very frame an old label sits on. No anchors, no jump search
            # -- the nearest processed frame, and nothing else for this flag.
            # A frame NUMBER (re-curating a labelled frame) is taken as given: snapping a
            # time to the nearest frame picked f1922 for a label on f1920, two frames off
            # the label it was meant to replace (2026necmp1_qm40).
            if flag.get("f") is not None and int(flag["f"]) in by_f:
                f = int(flag["f"]); t = by_f[f]["t"]
            else:
                t, f, _n = min(busy, key=lambda b: abs(b[0] - (off + t0)))
            if all(c[1] != f for c in chosen):
                chosen.append((t, f, f"requested exact frame{who} at {t - off:.2f}s"))
                if teams:
                    focus_boxes[f] = team_boxes(f, teams)
            continue
        for team in sorted(teams):
            for t, f, d, dt in jump_frames(off + t0, team):
                if all(c[1] != f for c in chosen):
                    chosen.append((t, f, f"requested {team} jump {d:.1f} m in {dt:.1f}s "
                                         f"at {t - off:.1f}s"))
                    focus_boxes[f] = team_boxes(f, teams)
        for k in range(FOLLOWUP_PER_FLAG):
            # evenly across the span, ends inset by the snap radius so both stay inside it
            frac = k / max(FOLLOWUP_PER_FLAG - 1, 1)
            dt = -FOLLOWUP_SPAN_S / 2 + FOLLOWUP_SNAP_S + frac * (FOLLOWUP_SPAN_S - 2 * FOLLOWUP_SNAP_S)
            take(off + t0 + dt, f"requested{who} at {t0:.1f}s ({dt:+.0f}s)", teams or None)

    if gaps > 0:
        corr = C.TRACKER_ROOT / "corrections" / f"{key}_corrections.json"
        labels = (json.loads(corr.read_text(encoding="utf-8")).get("labels") or []
                  if corr.exists() else [])
        t_of = {r["f"]: r["t"] for r in rows}
        spans, seen = [], defaultdict(list)
        for r in rows:
            for d in r["dets"]:
                if d.get("team"):
                    seen[str(d["team"])].append(r["t"])
        for team, ts in seen.items():
            lt = sorted(t_of[int(l["f"])] for l in labels
                        if str(l.get("team")) == team and int(l["f"]) in t_of)
            edges = [min(ts)] + lt + [max(ts)]
            spans += [(b - a, (a + b) / 2, team) for a, b in zip(edges, edges[1:])]
        for length, mid, team in sorted(spans, reverse=True):
            if sum(1 for c in chosen if c[2].startswith("gap")) >= gaps:
                break
            take(mid, f"gap {team} {length:.0f}s")

    # Exact frames are never trimmed -- the curator chose each one. Everything else fills
    # the cap behind them, requested before gap frames, then in time order for the curator.
    exact = [c for c in chosen if c[2].startswith("requested exact")]
    rest = [c for c in chosen if not c[2].startswith("requested exact")]
    rest.sort(key=lambda c: (not c[2].startswith("requested"), c[0]))
    chosen = sorted(exact + rest[:max(FOLLOWUP_MAX_FRAMES - len(exact), 0)])
    keep = {f for _t, f, _w in chosen}
    return ([f for _t, f, _w in chosen], [w for _t, _f, w in chosen],
            {f: b for f, b in focus_boxes.items() if f in keep})


def _focus_tids(doc: dict, focus_boxes: dict) -> dict:
    """Map the flagged robots' boxes (solver track space) onto the bundle's own track ids,
    so the curation screen highlights and preselects them (its `focus`, as the clash
    follow-up uses). Nearest box centre, in original video pixels."""
    out = {}
    for fr in doc.get("frames", []):
        want = focus_boxes.get(fr["f"])
        if not want:
            continue
        tids = []
        for b in want:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best = min(fr["dets"], key=lambda d: (d["xy"][0] - cx) ** 2 + (d["xy"][1] - cy) ** 2,
                       default=None)
            if best and ((best["xy"][0] - cx) ** 2 + (best["xy"][1] - cy) ** 2) ** 0.5 <= 60:
                tids.append(int(best["tid"]))
        if tids:
            out[str(fr["f"])] = tids
    return out


def _waiting_followup(key: str) -> dict | None:
    """The follow-up bundle for `key` still on the relay and not yet answered, as built
    locally -- or None. Any doubt (relay unreachable, file unreadable) reads as none, so
    a request is never blocked by this check, only at worst not merged."""
    p = C.STAGE3_DIR / f"{key}_followup_frames.json"
    if not p.exists():
        return None
    try:
        bundles, answers = relay_bundle_state()
        at = bundles.get(key)
        if at is None or answers.get(key, 0.0) > at or not _is_followup(key, at):
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return None


def run_followup(job: dict, report) -> tuple[int, int, list[str]]:
    """Build and push a follow-up bundle for one curated match."""
    key = job.get("match")
    if not key:
        raise RuntimeError("followup job names no match")
    # {t, teams} flags from the app; bare `times` from an app older than flags-with-teams
    flags = job.get("flags") or [float(t) for t in (job.get("times") or [])]
    gaps = max(0, min(int(job.get("gaps") or 0), FOLLOWUP_MAX_FRAMES))
    if not flags and not gaps:
        raise RuntimeError("followup job has no moments and no gap frames to ask about")
    report(f"follow-up {key}: choosing frames", 0, 2)
    frames, why, focus_boxes = _followup_frames(key, flags, gaps)
    if not frames:
        raise RuntimeError(f"none of the requested moments in {key} shows a tracked robot")
    # ADD TO A WAITING FOLLOW-UP rather than replacing it. A match has one bundle slot on
    # the relay, so a second request before the first was answered used to overwrite it:
    # 2026necmp1_qm40 lost a 5-frame request to a 3-frame one sent three minutes later.
    prev = _waiting_followup(key)
    if prev:
        old_f = [int(fr["f"]) for fr in prev.get("frames") or []]
        old_why = (prev.get("followup") or {}).get("reasons") or [""] * len(old_f)
        old_boxes = {int(k): v for k, v in
                     ((prev.get("followup") or {}).get("focusBoxes") or {}).items()}
        merged = {f: w for f, w in zip(old_f, old_why)}
        for f, w in zip(frames, why):
            merged.setdefault(f, w)
        frames = sorted(merged)
        why = [merged[f] for f in frames]
        focus_boxes = {**old_boxes, **focus_boxes}
        flags = list((prev.get("followup") or {}).get("flags") or []) + list(flags)
        report(f"follow-up {key}: adding to the unanswered follow-up already waiting "
               f"({len(old_f)} frame(s))", 0, 2)
    from . import curate as CU
    event = key.split("_")[0]
    st = C.STAGE1_DIR / f"{key}_tracks_stitched.jsonl"
    n_req = sum(1 for w in why if w.startswith("requested"))
    note = (f"Follow-up: {len(frames)} frame(s) -- {n_req} flagged from the route view"
            + (f", {len(frames) - n_req} from the longest unlabelled stretches" if len(frames) > n_req else "")
            + ". Boxes are pre-filled from the current routes; fix any that are wrong.")
    report(f"follow-up {key}: building {len(frames)} frame(s)", 1, 2)
    doc = CU.build_frames(key, st, key, 0, 0, frames=frames, note=note,
                          frame_w=800, frame_q=45, calib_stem=job.get("calibFrom") or event,
                          legible=False, guess_p=C.STAGE3_DIR / f"{key}_labeled.jsonl",
                          window=False)
    # The flagged robot's box in each frame is highlighted and selected first in the
    # curation screen, so the curator starts from the claim being tested.
    doc["focus"] = _focus_tids(doc, focus_boxes)
    doc["followup"] = {"round": "followup", "reasons": why,
                       "requestedAt": job.get("requestedAt"), "flags": flags, "gaps": gaps,
                       # solver-space boxes, so a later request can merge into this one
                       "focusBoxes": {str(f): b for f, b in focus_boxes.items()}}
    out = C.STAGE3_DIR / f"{key}_followup_frames.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    rc, tail = _run_logged([PY, "-m", "rtrack.relay", "push-bundle", key, "--file", str(out)],
                           cwd=C.TRACKER_ROOT)
    if rc != 0:
        raise RuntimeError(f"push failed: {failure_reason(tail, rc)}")
    report(f"follow-up {key}: {len(frames)} frame(s) on the relay for curation", 2, 2)
    return 1, 1, []


JOB_RUNNERS = {"process": run_process, "detect": run_detect, "bundle": run_bundle,
               "calib": run_calib, "followup": run_followup}


def pending_points() -> list[tuple[str, float]]:
    """Cameras whose points on the relay have not been applied yet.

    Two guards, both needed. NEWER THAN THE CALIBRATION FILE, so a points document that
    was already applied by hand is never re-applied -- the relay keeps them for a day.
    NEWER THAN THE LAST ATTEMPT, so a set of points that FAILS to fit is not retried on
    every loop forever; sending new points is what retries.
    """
    url, _ = R._env()
    import requests
    r = requests.get(f"{url}/index", timeout=60)
    r.raise_for_status()
    tried = _load_state().get("pointsTried", {})
    out = []
    for it in r.json().get("items", []):
        if it.get("kind") != "points":
            continue
        cam, at = it.get("id") or "", (it.get("at") or 0) / 1000.0
        cal = C.CALIB_DIR / f"{cam}.json"
        if cal.exists() and at <= cal.stat().st_mtime + 1:
            continue
        if at <= float(tried.get(cam, 0)) + 1:
            continue
        out.append((cam, at))
    return out


def apply_points(camera: str, at: float) -> None:
    """Fit the points a phone sent, measure the floor, and send the result back."""
    st = _load_state(); st.setdefault("pointsTried", {})[camera] = at; _save_state(st)
    footage = _camera_footage(camera)
    if not footage:
        _calib_state(camera, state="failed", error="no footage to fit against",
                     appliedAt=_now_iso())
        return
    doc = R.get("points", camera)
    if doc is None:
        return
    pts = C.STAGE3_DIR / f"{camera}_points.json"
    pts.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    cal_p = C.CALIB_DIR / f"{camera}.json"
    old = json.loads(cal_p.read_text(encoding="utf-8")) if cal_p.exists() else None
    n_tags = len(doc.get("aprilTags") or [])
    print(f"[agent] applying {len(doc.get('points') or [])} point(s) for {camera} "
          f"({n_tags} AprilTag) against {footage}", flush=True)
    # Frame 200, NOT --plate. The refit also traces the near barrier to fit the lens, and
    # measured: re-fitting 2026necmp1 from its own 15 points this way reproduced it
    # EXACTLY (floor shift 0.0 m). A different image could move the lens, and so the floor,
    # with no change to the points at all.
    rc, tail = _run_logged([PY, "-m", "rtrack.calibrate", footage, "--camera", camera,
                            "--points", str(pts)], cwd=C.TRACKER_ROOT)
    if rc != 0:
        _calib_state(camera, state="failed", error=failure_reason(tail, rc),
                     appliedAt=_now_iso())
        print(f"[agent] calibration for {camera} FAILED: {failure_reason(tail, rc)}",
              flush=True)
        return
    new = json.loads(cal_p.read_text(encoding="utf-8"))
    from .calibrate import floor_shift
    shift = floor_shift(old, new) if old else None
    moved = bool(shift and shift["maxM"] > 0.02)
    _calib_state(camera, state="applied", appliedAt=_now_iso(), pointsAt=at, error=None,
                 points=len(new.get("points") or []), aprilTags=len(new.get("aprilTags") or []),
                 reprojErrorM=new.get("reprojErrorM"), floorShift=shift,
                 # If the FLOOR moved, every route already projected through the old fit
                 # is now offset from anything projected through the new one. Not acted on
                 # automatically -- re-projecting an event is a decision -- but said plainly.
                 floorMoved=moved)
    print(f"[agent] {camera} calibration applied: {len(new.get('points') or [])} points, "
          f"{len(new.get('aprilTags') or [])} tags, floor shift {shift}"
          + ("  -- FLOOR MOVED: existing routes used the previous fit" if moved else ""),
          flush=True)
    # Close the loop: the refreshed bundle carries the new fit and its overlay, so the
    # person holding the phone sees what their points did and can refine again.
    ok, why = publish_calib_bundle(camera, footage)
    if not ok:
        print(f"[agent] could not re-publish {camera} for review: {why}", flush=True)


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

    prev_entry = jobs.get(job_id) or {}
    jobs[job_id] = {"state": "running", "type": doc.get("type"), "lastRunAt": time.time(),
                    "event": doc.get("event"), "startedAt": _now_iso(),
                    # Carried across the pass boundary so the app keeps a breakdown to
                    # show while the next pass is still working.
                    "plan": prev_entry.get("plan"), "note": prev_entry.get("note"),
                    "done": prev_entry.get("done"), "total": prev_entry.get("total")}
    _save_jobs(jobs)
    last = [0.0]

    def report(note: str, i: int, total: int, plan: dict | None = None) -> None:
        """Progress for a human, plus an optional STRUCTURED breakdown for the app.

        `note` alone could not answer "which matches are in progress toward the cap" --
        it is one line of prose, and only the latest one survives. `plan` carries the
        per-match categories so the app can name them instead of implying them.
        """
        print(f"[agent] job {job_id[:8]}: {note} ({i}/{total})", flush=True)
        rec = _load_jobs()
        cur = rec.get(job_id, {})
        cur.update({"state": "running", "note": note, "done": i, "total": total})
        if plan is not None:
            cur["plan"] = plan
        rec[job_id] = cur
        _save_jobs(rec)
        # Throttled, because a per-match post would spend the day's write budget.
        if time.time() - last[0] >= JOB_REPORT_S:
            last[0] = time.time()
            on_report()

    try:
        result = runner(doc, report)
        # A runner may report itself UNFINISHED (process does, since it waits on a human
        # between bundles). Such a job stays out of the done/failed set, so pending_jobs
        # hands it back on the next poll and it resumes where it stopped.
        done, total, failed, complete = (result if len(result) == 4
                                         else (*result, True))
        jobs = _load_jobs()
        entry = {"state": "done" if complete else "running",
                 "type": doc.get("type"), "event": doc.get("event"),
                 "done": done, "total": total, "failed": failed}
        entry["finishedAt" if complete else "updatedAt"] = _now_iso()
        if not complete:
            prev = _load_jobs().get(job_id, {})
            entry["note"] = prev.get("note")
            entry["plan"] = prev.get("plan")
            # Kept across the rewrite, or a long-running job would look like it had never
            # had a turn and win every tie again -- the starvation next_job() exists to end.
            entry["lastRunAt"] = prev.get("lastRunAt")
        jobs[job_id] = entry
        _save_jobs(jobs)
        if complete:
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
    jobs = {k: v for k, v in _load_jobs().items() if not k.startswith("_")}
    running = [{"id": k, **v} for k, v in jobs.items() if v.get("state") == "running"]
    recent = sorted(((k, v) for k, v in jobs.items() if v.get("state") != "running"),
                    key=lambda kv: kv[1].get("finishedAt") or "", reverse=True)[:5]
    return {"running": running,
            "recent": [{"id": k, **v} for k, v in recent]}


# Stage1 artifacts older than this are from an event nobody is working on, and listing
# them would grow the heartbeat for no reader. Two months covers a season's worth of
# "the event before this one".
DETECTED_MAX_AGE_S = 60 * 86400


def detected_summary() -> dict:
    """Which matches have been DETECTED, per event, for the app to render.

    Detection is otherwise invisible. rtrack.pipeline --prep-only writes stage1 tracks and
    nothing else -- no curation bundle, no routes -- so a match that has just been detected
    looks identical to one that has never been touched from the app's side: it has no
    relay bundle and no published route, and the Tracks tab falls through to "no tracks".
    Requesting detection and seeing nothing change is indistinguishable from the request
    having failed, which is the report this exists to answer.

    Suffixes rather than full keys, because the event is already the dict key and a
    hundred "2026necmp1_" prefixes is a kilobyte of nothing in a document posted on a
    timer.
    """
    cutoff = time.time() - DETECTED_MAX_AGE_S
    out: dict[str, list[str]] = {}
    for path in C.STAGE1_DIR.glob("*_tracks_stitched.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        stem = path.name[: -len("_tracks_stitched.jsonl")]
        event, _, suffix = stem.partition("_")
        # An event key is a year plus a code. Requiring that skips artifacts still named
        # after a raw video id -- "WFj_FsFQRkM" otherwise parses as event "WFj", which no
        # reader will ever ask about and which only makes the heartbeat noisier.
        if not suffix or not re.fullmatch(r"\d{4}[a-z0-9]+", event):
            continue
        out.setdefault(event, []).append(suffix)

    def num(suffix: str) -> tuple:
        m = re.match(r"([a-z]+)(\d+)", suffix)
        return (m.group(1), int(m.group(2))) if m else (suffix, 0)

    return {event: sorted(v, key=num) for event, v in sorted(out.items())}


def _budget_take(kind: str = "status") -> bool:
    """Spend one status post from today's ration. False means skip this post.

    Rationed rather than merely slowed, because an interval alone does not bound a day:
    every state change posts immediately too, and a busy hour of them is what actually
    overran the quota. The counter resets on the UTC day, matching when KV's limit resets.
    """
    state = _load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("budgetDay") != today:
        state["budgetDay"] = today
        state["statusPosts"] = 0
        state.pop("budgetWarned", None)
    used = int(state.get("statusPosts", 0))
    if used >= STATUS_POSTS_PER_DAY:
        if not state.get("budgetWarned"):
            state["budgetWarned"] = True
            _save_state(state)
            print(f"[agent] status budget for {today} spent ({used} posts). Heartbeats "
                  f"pause until 00:00 UTC so curation answers and route uploads keep "
                  f"their share of the daily KV writes.", flush=True)
        return False
    state["statusPosts"] = used + 1
    _save_state(state)
    return True


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
        "detected": detected_summary(),
        # Per camera: where its review round trip stands, and whether the last applied
        # points moved the floor. What the app's Cameras section reads.
        "calibration": _load_state().get("calibration", {}),
        "stream": (desired or {}).get("stream"),
        "error": error,
    }
    if not _budget_take():
        return
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
    # Output is APPENDED to out/agent.log: a console window keeps nothing, and when ten
    # 2026necmp1 matches were set aside there was no record of why.
    # Joined rather than escaped: a .cmd file wants CRLF, and newline="" on the write
    # below means these are the exact bytes that land on disk.
    body = "\r\n".join([
        "@echo off",
        "rem rtrack agent -- listens for relay control and supervises rtrack.watch.",
        "rem Installed by: rtrack.agent --install-task",
        "rem Remove it by deleting THIS FILE, or: rtrack.agent --uninstall-task",
        f'cd /d "{C.TRACKER_ROOT}"',
        f'start "rtrack agent" /min cmd /c ""{PY}" -m rtrack.agent {args} '
        f'>> "{C.TRACKER_ROOT / "out" / "agent.log"}" 2>&1"',
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
                   help="post a job document. `process` takes a match range from nothing "
                        "to a curated route and keeps it flowing; detect and "
                        "bundle are the older single-stage requests.")
    g.add_argument("--matches", default=None, metavar="SPEC",
                   help="for --request process/detect: qm26-qm100 or qm1,qm7")
    g.add_argument("--count", type=int, default=3,
                   help="for --request bundle: how many uncurated matches to bundle")
    g.add_argument("--max-outstanding", type=int, default=MAX_OUTSTANDING_BUNDLES,
                   metavar="N",
                   help="for --request process: how many uncurated bundles to leave on "
                        "the relay at once")
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
        if args.request in ("detect", "process") and not args.matches:
            print("[agent] --request detect needs --matches, e.g. qm26-qm100",
                  file=sys.stderr)
            return 2
        job_id = uuid.uuid4().hex[:16]
        doc = {"schemaVersion": 1, "jobId": job_id, "agentId": agent_id,
               "type": args.request,
               "event": args.arm or (args.calib_from or None),
               "matches": args.matches, "count": args.count,
               "maxOutstanding": args.max_outstanding,
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

            # Calibration points BEFORE jobs: someone is holding a phone waiting to see
            # what their points did, and a fit takes seconds where a job pass takes
            # minutes.
            try:
                for cam, at in pending_points():
                    apply_points(cam, at)
                    changed = True
            except Exception as exc:                  # noqa: BLE001
                print(f"[agent] points check failed ({type(exc).__name__}: {exc})",
                      flush=True)

            # AFTER reconcile, so an arm/disarm is never stuck behind a long backfill,
            # and one job at a time: they compete for one GPU, and two detections in
            # flight make both slower rather than finishing either sooner.
            try:
                jobs_todo = pending_jobs(agent_id)
            except Exception as exc:                  # noqa: BLE001
                print(f"[agent] job check failed ({type(exc).__name__}: {exc})", flush=True)
                jobs_todo = []
            chosen = next_job(jobs_todo)
            if chosen:
                beat = lambda: post_status(agent_id, watcher, desired, applied_nonce,
                                           "running" if watcher.alive else "working",
                                           last_error)
                run_job(chosen, agent_id, beat)
                changed = True

            run_state = "running" if watcher.alive else ("error" if last_error else "idle")
            beat_every = args.heartbeat_s or (
                HEARTBEAT_RUNNING_S if watcher.alive else HEARTBEAT_IDLE_S)
            due = time.time() - last_beat
            if (changed and due >= HEARTBEAT_FLOOR_S) or due >= beat_every:
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
