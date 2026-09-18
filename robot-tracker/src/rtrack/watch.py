"""Stage 4 -- notice curator answers as they arrive and finish those matches.

    uv run -m rtrack.watch --event 2026mawor --calib-from WFj_FsFQRkM

Polls the relay. When an answer turns up that is newer than the corrections already on
disk, it pulls it, re-solves, projects, exports, publishes and folds the match into the
appearance gallery -- the whole back half of rtrack.pipeline, unattended.

WHY A WATCHER RATHER THAN `pipeline --relay --wait`. That mode blocks on ONE match, which
is wrong for how curation actually happens: a curator works through several matches in
whatever order suits them, and the machine should pick up each as it lands rather than
waiting on a particular one. A watcher also survives the curator going away and coming
back, which a 30-minute timeout does not.

NEWNESS IS `answer.at` VS THE CORRECTIONS FILE MTIME, not a processed-set, because
re-curation has to work: a curator who reopens a match and sends again produces a newer
answer for a key already handled, and that must re-run rather than be skipped as seen.

SEQUENCING IS THE PIPELINE'S PROBLEM, NOT THIS ONE. Each match is handed to
rtrack.pipeline, which takes the per-event lock, so two answers arriving together are
processed one after the other. That matters because each run updates the shared gallery.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from . import relay as R

# 60 s, not 20. /index used to be a KV list(), which the free plan caps at 1000 calls PER
# DAY -- separately from the 100k reads -- so a 20 s poll spent the whole day's quota in
# about five hours and every /index after that returned a 500, taking the app's Tracks
# listing down with it while every other endpoint stayed healthy. The worker now serves
# /index from a manifest key (an ordinary get), which removes the cliff; this is the belt
# to that braces. A curator takes minutes per match, so polling faster buys nothing.
POLL_S = 60.0
PY = sys.executable


def pending(event: str | None) -> list[tuple[str, float]]:
    """(match key, answer timestamp) for answers newer than their corrections file."""
    url, _ = R._env()
    import requests
    r = requests.get(f"{url}/index", timeout=60)
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        if it.get("kind") != "answer":
            continue
        key = it.get("id") or ""
        if event and not key.startswith(event + "_"):
            continue
        at = (it.get("at") or 0) / 1000.0
        corr = C.TRACKER_ROOT / "corrections" / f"{key}_corrections.json"
        if not corr.exists() or at > corr.stat().st_mtime + 1:
            out.append((key, at))
    return sorted(out)


def finish(key: str, calib_from: str | None, event: str | None) -> bool:
    """Pull the answer, then run the match to publication."""
    print(f"\n[watch] {key}: answers are newer than what is on disk -- processing")
    a = [PY, "-m", "rtrack.relay", "wait-answer", key, "--timeout", "30"]
    if subprocess.run(a).returncode != 0:
        print(f"[watch] {key}: could not fetch the answer")
        return False
    b = [PY, "-m", "rtrack.pipeline", key, "--match", key]
    if calib_from:
        b += ["--calib-from", calib_from]
    if event:
        b += ["--event", event]
    rc = subprocess.run(b).returncode
    if rc == 3:
        print(f"[watch] {key}: another match holds the event lock; will retry")
        return False
    if rc != 0:
        print(f"[watch] {key}: pipeline failed ({rc})")
        return False
    print(f"[watch] {key}: published")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Finish matches as curators answer them.")
    ap.add_argument("--event", default=None, help="only watch this event key")
    ap.add_argument("--calib-from", default=None, metavar="VIDEO")
    ap.add_argument("--poll", type=float, default=POLL_S)
    ap.add_argument("--once", action="store_true",
                    help="process whatever is already waiting, then exit")
    args = ap.parse_args(argv)
    C.ensure_dirs()

    R._env()          # fail now, loudly, if the relay is not configured
    print(f"[watch] polling every {args.poll:.0f}s"
          + (f" for {args.event}" if args.event else "")
          + ("  (once)" if args.once else "  -- Ctrl-C to stop"))

    seen_quiet = False
    while True:
        try:
            todo = pending(args.event)
        except Exception as e:
            print(f"[watch] relay unreachable ({e}); retrying")
            todo = []
        if todo:
            seen_quiet = False
            for key, _ in todo:
                finish(key, args.calib_from, args.event)
        elif not seen_quiet:
            print("[watch] nothing waiting", flush=True)
            seen_quiet = True        # say it once, not every poll
        if args.once:
            return 0
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            print("\n[watch] stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
