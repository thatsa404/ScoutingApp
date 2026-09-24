"""Stage 4 -- notice curator answers as they arrive and finish those matches.

    uv run -m rtrack.watch --event 2026mawor --calib-from 2026mawor

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

Gallery answers are a separate queue.  They are applied and published after live route
answers, but they never block route publication and they never auto-publish a changed
historical route.

SEQUENCING IS THE PIPELINE'S PROBLEM, NOT THIS ONE. Each match is handed to
rtrack.pipeline, which takes the per-event lock, so two answers arriving together are
processed one after the other. That matters because each run updates the shared gallery.
"""

from __future__ import annotations

import argparse
import json
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
    """Answers that have not made it all the way to a current published route.

    Downloading an answer is not completion.  A prior watcher could fetch several
    answers, lose the event lock while invoking pipeline, and then suppress every
    retry because the corrections file existed.  Require the published route to be
    at least as new as both the relay answer and local corrections instead.
    """
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
        published = C.REPO_ROOT / "public" / "tracks" / f"{key}.json"
        required_mtime = max(at, corr.stat().st_mtime if corr.exists() else 0.0)
        if not published.exists() or published.stat().st_mtime + 1 < required_mtime:
            out.append((key, at))
    return sorted(out)


def pending_gallery(season: int | None = None) -> list[tuple[str, float]]:
    """(review id, relay timestamp) for gallery answers not applied locally."""
    url, _ = R._env()
    import requests
    r = requests.get(f"{url}/index", timeout=60)
    r.raise_for_status()
    out = []
    answer_dir = C.OUT_DIR / "gallery" / "review"
    for it in r.json().get("items", []):
        if it.get("kind") != "gallery-answer":
            continue
        if season is not None and it.get("season") not in (None, season):
            continue
        rid = it.get("id") or ""
        at = (it.get("at") or 0) / 1000.0
        local = answer_dir / f"{rid}_answer.json"
        if not local.exists() or at > local.stat().st_mtime + 1:
            out.append((rid, at))
    return sorted(out)


def finish_gallery(review_id: str) -> bool:
    """Pull, validate, and publish one gallery answer without replaying routes."""
    print(f"\n[watch] gallery {review_id}: answer is ready -- applying")
    answer_dir = C.OUT_DIR / "gallery" / "review"
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_path = answer_dir / f"{review_id}_answer.json"
    answer = R.get("gallery-answer", review_id)
    if answer is None:
        print(f"[watch] gallery {review_id}: answer disappeared before fetch")
        return False
    answer_path.write_text(json.dumps(answer, indent=1), encoding="utf-8")
    bundle_path = answer_dir / f"{review_id}.json"
    if not bundle_path.exists():
        bundle = R.get("gallery-bundle", review_id)
        if bundle is None:
            print(f"[watch] gallery {review_id}: source bundle is unavailable")
            return False
        bundle_path.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    season = None
    try:
        season = int(json.loads(bundle_path.read_text(encoding="utf-8"))["season"])
        a = [PY, "-m", "rtrack.gallery_review", "apply", str(answer_path),
             "--bundle", str(bundle_path)]
        if subprocess.run(a).returncode != 0:
            return False
        b = [PY, "-m", "rtrack.gallery_review", "rebuild", "--season", str(season)]
        if subprocess.run(b).returncode != 0:
            return False
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[watch] gallery {review_id}: {exc}")
        return False
    print(f"[watch] gallery {review_id}: published season {season}; replay is queued "
          "for vote-diff review")
    return True


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

    # The watcher runs pipeline without --relay because route answers have already
    # made the relay round trip.  That also means pipeline prepares a gallery-review
    # bundle locally but does not upload it.  Find the bundle produced for this match
    # and publish it explicitly so the Tracks tab can continue the review workflow.
    review_dir = C.OUT_DIR / "gallery" / "review"
    candidates = []
    for path in review_dir.glob("*.json"):
        if path.name.endswith("_answer.json"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("kind") != "galleryReviewBundle":
            continue
        source_matches = {
            str(candidate.get("source", {}).get("match", ""))
            for team in doc.get("teams", [])
            for candidate in team.get("candidates", [])
        }
        if key in source_matches:
            candidates.append(path)
    if candidates:
        bundle = max(candidates, key=lambda path: path.stat().st_mtime)
        review_id = bundle.stem
        push = [PY, "-m", "rtrack.relay", "push-gallery-bundle", review_id,
                "--file", str(bundle)]
        if subprocess.run(push).returncode != 0:
            print(f"[watch] {key}: route published but gallery bundle upload failed")
            return False
        print(f"[watch] {key}: gallery review {review_id} uploaded")
    else:
        print(f"[watch] {key}: no gallery-review bundle was produced")
    print(f"[watch] {key}: published")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Finish matches as curators answer them.")
    ap.add_argument("--event", default=None, help="only watch this event key")
    ap.add_argument("--season", type=int, default=None,
                    help="only apply gallery answers for this season")
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
        # Route answers have priority: a gallery answer can wait, while a newly
        # curated live match is the latency-sensitive deliverable.
        try:
            gallery_todo = pending_gallery(args.season)
        except Exception as e:
            print(f"[watch] gallery relay check failed ({e}); retrying")
            gallery_todo = []
        for review_id, _ in gallery_todo:
            finish_gallery(review_id)
        if not todo and not gallery_todo and not seen_quiet:
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
