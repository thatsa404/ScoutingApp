"""Stage 4 -- run one match end to end, optionally round-tripping a curator.

    uv run -m rtrack.pipeline 2026mawor_qm1 --match 2026mawor_qm1 \\
        --calib-from WFj_FsFQRkM --relay --wait 1800

Sequences what already exists rather than reimplementing any of it:

    track -> stitch -> appear -> reid votes -> robots -> curate
      -> [push bundle, wait for answers] -> robots again -> project -> export
      -> gallery

RESUMABLE BY DEFAULT. Every step is skipped when its output is already present and
newer than its input, because these steps cost minutes and the loop gets re-run after a
curator answers. `--force` or `--from <step>` overrides. That is not a convenience --
it is what makes the answer round trip practical, since the second pass should redo the
solve and everything downstream of it, and nothing above it.

THE GALLERY STEP IS WHY THE ORDER MATTERS. Curating match N updates the per-event
appearance gallery, which is what lets match N+1 be labelled ~84% correctly before
anyone looks at it (measured; 13% without). So matches should be run IN ORDER, and the
gallery update must happen after curation, not before.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .acquire import video_id

PY = sys.executable


def tba_mod_teams(match_key: str):
    """The six teams TBA lists for this match, or () if it cannot be asked."""
    try:
        from . import tba as tba_mod
        m = tba_mod.match_by_key(match_key)
        return [str(t) for t in m["red"]] + [str(t) for t in m["blue"]]
    except Exception:
        return ()


STEPS = ["track", "stitch", "appear", "votes", "robots", "curate",
         "send", "resolve", "viewcheck", "project", "export", "gallery"]
# A lock older than this is assumed to be from a crashed run. Generous,
# because a match with --relay legitimately blocks for its whole --wait
# while a human curates.
STALE_LOCK_S = 3 * 3600.0


def run(mod: str, *args: str, quiet: bool = False) -> bool:
    cmd = [PY, "-m", f"rtrack.{mod}", *[str(a) for a in args]]
    print(f"    $ {' '.join(cmd[2:])}", flush=True)
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if r.returncode != 0:
        if quiet and r.stdout:
            print(r.stdout[-1500:])
        if quiet and r.stderr:
            print(r.stderr[-1500:])
        print(f"    !! rtrack.{mod} failed ({r.returncode})")
        return False
    return True


def newer(out: Path, *ins: Path) -> bool:
    """True when `out` exists and is at least as new as every input that exists."""
    if not out.exists():
        return False
    t = out.stat().st_mtime
    return all(t >= i.stat().st_mtime - 1 for i in ins if i.exists())


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one match end to end.")
    ap.add_argument("video")
    ap.add_argument("--match", required=True)
    ap.add_argument("--calib-from", default=None)
    ap.add_argument("--event", default=None,
                    help="gallery/event key; default: the match key's prefix")
    ap.add_argument("--relay", action="store_true",
                    help="push the bundle and wait for a curator instead of stopping")
    ap.add_argument("--wait", type=float, default=1800.0)
    ap.add_argument("--model", type=Path,
                    default=Path("runs/detect/runs/r2026_s/weights/best.pt"))
    ap.add_argument("--from", dest="start", choices=STEPS, default=None,
                    help="redo from this step onward, ignoring existing outputs")
    ap.add_argument("--force", action="store_true", help="redo everything")
    ap.add_argument("--no-curate", action="store_true",
                    help="stop after solving; do not build a curation bundle")
    ap.add_argument("--no-votes", action="store_true",
                    help="solve identity WITHOUT appearance votes even though a "
                         "gallery exists. Normally a votes failure is fatal; this is "
                         "the deliberate override, not a routine flag.")
    ap.add_argument("--appearance", choices=("hist", "cnn"), default="hist",
                    help="descriptor backend for identity votes. 'hist' is the tuned "
                         "48-d histogram; 'cnn' is the learned embedding, which scores "
                         "90.6%% within-alliance against 55.1%% leave-one-match-out "
                         "(see rtrack.embed). 'cnn' costs one GPU pass on the same "
                         "decode and still writes the histogram, which "
                         "robots.split_on_appearance needs either way.")
    ap.add_argument("--no-identity-check", action="store_true",
                    help="process even if the clip cannot be confirmed to hold this "
                         "match. See the gate in _main(); this is the override, not a "
                         "routine flag.")
    ap.add_argument("--prep-only", action="store_true",
                    help="run only track/stitch/appear and stop. These are the "
                         "expensive steps and NONE of them depend on the appearance "
                         "gallery, so a whole event can be prepped unattended and each "
                         "match then needs only the ~1 min of gallery-dependent work "
                         "(votes, solve, bundle) immediately before it is curated. "
                         "Takes no lock: it cannot touch the gallery.")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    event = args.event or args.match.split("_")[0]
    t_all = time.time()

    # ONE RUN PER EVENT AT A TIME, and this is a correctness constraint rather than a
    # politeness one. Match N+1's identity votes come from the appearance gallery that
    # match N's CURATION produced, so two runs in flight would both read a stale
    # gallery -- defeating the improvement the whole design rests on -- and then race
    # to rewrite the same <event>_gallery.npz, where the loser's crops vanish.
    #
    # Contention matters too: a solve can take minutes, and two competing for one GPU
    # makes both slower, which is how a backlog compounds across a match cycle rather
    # than recovering.
    lock = C.STAGE3_DIR / f".{event}.pipeline.lock"
    if args.prep_only:
        lock = None
    if lock is not None and lock.exists():
        try:
            prev = json.loads(lock.read_text(encoding="utf-8"))
            age = time.time() - prev.get("at", 0)
        except Exception:
            prev, age = {}, 1e9
        if age < STALE_LOCK_S:
            print(f"[pipeline] REFUSING: {prev.get('match', '?')} has been running for "
                  f"{age / 60:.1f} min on this event (pid {prev.get('pid')}).\n"
                  f"           Matches must run one at a time -- see the note in main().\n"
                  f"           If that run is dead, delete {lock}")
            return 3
        print(f"[pipeline] clearing a stale lock ({age / 60:.0f} min old)")
    if lock is not None:
        lock.write_text(json.dumps({"match": args.match, "pid": os.getpid(),
                                    "at": time.time()}), encoding="utf-8")
        _LOCK_SINK.append(lock)

    raw = C.RAW_DIR / f"{stem}.mp4"
    if not raw.exists():
        raise SystemExit(f"[pipeline] {raw} not found -- slice it with rtrack.replay")

    # IDENTITY GATE, and it fails CLOSED. Everything below compounds: tracking, a solve
    # against six teams that are not on screen, a human labelling them, and a gallery
    # learning from it. On 2026necmp1 a one-match offset in TBA's actual_time ran
    # twenty-one matches deep and was caught by a person's eye, not by any check here.
    #
    # Two independent reads, so one obscured title does not stop the line: the broadcast
    # names the match, and the roster strip lists its six teams. UNKNOWN is allowed
    # through with a warning -- a check that cannot run must not become a check that
    # blocks -- but a POSITIVE disagreement stops the match dead.
    if not args.no_identity_check:
        try:
            from .scoreboard import identify
            mm = re.search(r"qm(\d+)$", args.match or "")
            if mm:
                want = tba_mod_teams(args.match)
                v = identify(stem, int(mm.group(1)), want)
                if v["ok"] is False:
                    print(f"[pipeline] REFUSING {args.match}: " + v["why"] + ".")
                    print(f"           Re-slice it (rtrack.replay --index-shift) "
                          f"or pass --no-identity-check to override.")
                    return 4
                if v["ok"] is None:
                    print(f"[pipeline] identity UNVERIFIED ({v['why']}) -- continuing")
                else:
                    print(f"[pipeline] identity confirmed: {v['why']}")
        except Exception as e:
            print(f"[pipeline] identity check skipped ({type(e).__name__}: {e})")

    tracks = C.STAGE1_DIR / f"{stem}_tracks.jsonl"
    st = C.STAGE1_DIR / f"{stem}_tracks_stitched.jsonl"
    appear = C.STAGE3_DIR / f"{stem}_appearance.npz"
    # The cnn backend keeps its gallery, votes and npz under separate names; the two
    # descriptors are not comparable and one shared path would mix 48-d histograms
    # with 512-d embeddings. See reid.gallery_path.
    _sfx = "_cnn" if args.appearance == "cnn" else ""
    votes = C.STAGE3_DIR / f"{stem}_reid{_sfx}.json"
    labeled = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
    bundle = C.STAGE3_DIR / f"{stem}_curate_frames.json"
    corr = C.TRACKER_ROOT / "corrections" / f"{args.match}_corrections.json"
    positions = C.STAGE2_DIR / f"{stem}_positions.json"
    out = C.STAGE3_DIR / f"{args.match}.json"
    gallery = C.STAGE3_DIR / f"{event}_gallery{_sfx}.npz"

    si = STEPS.index(args.start) if args.start else -1
    def do(step: str, fresh: bool) -> bool:
        if args.force:
            return True
        if si >= 0 and STEPS.index(step) >= si:
            return True
        return not fresh

    print(f"[pipeline] {stem} -> {args.match}  (event {event})")

    if do("track", newer(tracks, raw)):
        if not run("track", stem, "--model", args.model, "--imgsz", 640, "--square",
                   "--conf", 0.20, "--lost-decay", 0.7, "--nms-iou", 0.55):
            return 1
    else:
        print("    track: up to date")

    if do("stitch", newer(st, tracks)):
        if not run("stitch", tracks):
            return 1
    else:
        print("    stitch: up to date")

    # NO MATCH IN THE CLIP is as disqualifying as the WRONG match, and it is a separate
    # question -- 2026necmp1 qm19 passed both identity signals (title said 19, 6/6 teams
    # agreed) while containing 240 s of pre-match staging and no match at all. The
    # scoreboard advertises the UPCOMING match during staging, so identity confirms what
    # the clip is ABOUT; only motion confirms the match is in it.
    #
    # Runs after tracking because that is what it reads, but before the solve, the
    # bundle, and a curator's time. The tell is unmistakable: qm19 held exactly 3 robots
    # in all 3600 frames -- three staged robots sitting still.
    if not args.no_identity_check and not args.prep_only:
        try:
            from .curate import load_pipeline_rows, match_window
            _rows = load_pipeline_rows(stem, st, args.calib_from)
            if match_window(_rows) is None:
                print(f"[pipeline] REFUSING {args.match}: no sustained robot motion "
                      f"anywhere in this clip -- it holds staging or a field reset, not "
                      f"a match.")
                print(f"           Re-slice it wider (rtrack.replay --pad-after) or pass "
                      f"--no-identity-check to override.")
                return 5
        except Exception as e:
            print(f"[pipeline] motion check skipped ({type(e).__name__}: {e})")

    if do("appear", newer(appear, st)):
        if not run("appear", stem, "--tracks", st,
                   "--backend", "both" if args.appearance == "cnn" else "hist"):
            return 1
    else:
        print("    appear: up to date")

    if args.prep_only:
        print(f"[pipeline] prepped in {time.time() - t_all:.0f}s "
              f"(track/stitch/appear); gallery-dependent steps still to run")
        return 0

    # Votes need a gallery. Without one this is the event's first match and the solver
    # falls back to geometry plus bumper hue, which is ~13% correct -- expected, not an
    # error. Curating this match is what creates the gallery for the next.
    have_votes = False
    if args.no_votes:
        print("    votes: SKIPPED by --no-votes; identity from geometry and hue only")
    elif gallery.exists():
        if do("votes", newer(votes, appear, gallery)):
            have_votes = run("reid", "votes", stem, "--event", event,
                             "--match", args.match, "--tracks", st,
                             "--backend", args.appearance)
            if not have_votes:
                # FATAL, deliberately. A gallery exists, so appearance evidence was
                # available and something broke -- that is not the same state as an
                # event's first match having nothing to vote with, and it must not be
                # allowed to look like it. reid.vote_tracks raised NameError on every
                # call for the whole 2026necmp1 event; the pipeline carried on to the
                # 'identity will come from the curator' path and reported success, so
                # 20 matches were solved on geometry and bumper hue alone and the
                # resulting 23-39% auto-ID was read as the descriptor's performance.
                # A wrong number that looks like a right one costs more than a stop.
                print(f"    !! votes FAILED but {gallery.name} exists -- appearance "
                      f"evidence is available and was not used. Refusing to solve "
                      f"identity without it; fix the error above and re-run. "
                      f"(--no-votes to proceed deliberately without appearance.)")
                return 6
        else:
            have_votes = True
            print("    votes: up to date")
    else:
        print(f"    votes: no {gallery.name} yet -- first match of the event, "
              f"identity will come from the curator")

    def solve() -> bool:
        a = ["robots", stem, "--tracks", st, "--match", args.match, "--deconflict", 3]
        if args.calib_from:
            a += ["--calib-from", args.calib_from]
        if have_votes and votes.exists():
            a += ["--identity", votes]
        if corr.exists():
            a += ["--corrections", corr]
        return run(*a)

    if do("robots", newer(labeled, st, appear)) or corr.exists():
        if not solve():
            return 1
    else:
        print("    robots: up to date")

    if args.no_curate and not corr.exists():
        print(f"[pipeline] stopping before curation ({time.time() - t_all:.0f}s)")
        return 0

    # ---- curator round trip -------------------------------------------------
    if not corr.exists():
        if do("curate", newer(bundle, labeled)):
            # --tracks is the STITCHED file (the id space rtrack.robots rebuilds) and
            # --guess is the labelled one (where the solver's answer lives). Both are
            # needed and they are not interchangeable: passing only `st` shipped every
            # bundle with 0% pre-fill while the solver was getting 95% of detections
            # right, and passing `labeled` to --tracks instead re-segments an already
            # segmented file -- 47 tracks became 85 on qm8 -- thinning the per-track
            # anchoring that catches identity switches. See curate.overlay_guess.
            #
            # 18 frames, not 16: legibility-weighted selection spends some of the budget
            # on readable views rather than purely maximal coverage, and a couple of
            # extra frames are cheap to curate when they are the easy ones to read.
            ca = ["curate", stem, "--tracks", st, "--guess", labeled,
                  "--match", args.match,
                  "--frames", 18, "--anchors", 2, "--mobile"]
            if args.calib_from:
                ca += ["--calib-from", args.calib_from]
            if not run(*ca):
                return 1
        else:
            print("    curate: bundle up to date")

        if args.relay:
            if not run("relay", "push-bundle", args.match, "--file", bundle):
                return 1
            print(f"    waiting up to {args.wait:.0f}s for a curator...")
            if not run("relay", "wait-answer", args.match, "--timeout", args.wait):
                print("[pipeline] no answers came back; the bundle is still on the relay")
                return 2
        else:
            print(f"[pipeline] bundle ready: {bundle}")
            print(f"[pipeline] push it with:  rtrack.relay push-bundle {args.match} "
                  f"--file {bundle.name}")
            return 0

    if corr.exists():
        print(f"    re-solving with {corr.name}")
        if not solve():
            return 1

    # ---- downstream ---------------------------------------------------------
    # Camera-pose check BEFORE project, because project reads its output. Never fatal:
    # a clip where the check cannot form an opinion is still worth projecting, and
    # is_valid_at keeps everything when the file is missing. A failure here must not
    # cost a match its routes.
    va = ["viewcheck", stem]
    if args.calib_from:
        va += ["--calib-from", args.calib_from]
    if not run(*va):
        print("[pipeline] viewcheck failed; projecting without a camera-pose check")

    pa = ["project", stem, "--tracks", labeled]
    if args.calib_from:
        pa += ["--calib-from", args.calib_from]
    if not run(*pa):
        return 1

    ea = ["export", stem, "--match", args.match, "--hz", 5, "--publish"]
    if args.calib_from:
        ea += ["--calib-from", args.calib_from]
    if not run(*ea):
        return 1

    # Gallery LAST: it learns from the curated labelling, and only helps the NEXT match.
    if not run("reid", "gallery", stem, "--event", event, "--labeled", labeled,
               "--backend", args.appearance):
        return 1

    # Rebuild the manifest AFTER the gallery update. export already wrote one, but that
    # happens a step too early: the gallery does not yet contain THIS match's teams, so
    # the match would publish reading 0/6 models -- its own six teams uncounted -- and
    # stay that way until some later publish happened to rebuild the file.
    try:
        from .export import write_manifest
        n = write_manifest(C.REPO_ROOT / "public" / "tracks")
        print(f"[pipeline] manifest refreshed after gallery update ({n} matches)")
    except Exception as e:
        print(f"[pipeline] manifest refresh failed ({e}) -- model counts may lag")

    print(f"[pipeline] {args.match} done in {time.time() - t_all:.0f}s -> {out.name}")
    return 0


_LOCK_SINK: list = []


def main(argv=None) -> int:
    """Wrapper so every exit path releases the lock, including a traceback or Ctrl-C.

    A lock left behind by a crash blocks the next match until someone notices, which at
    an event means noticing during the next match. STALE_LOCK_S is the backstop; this
    is the part that should normally do the work.
    """
    lock_holder: list[Path] = []
    global _LOCK_SINK
    _LOCK_SINK = lock_holder
    try:
        return _main(argv)
    finally:
        for p in lock_holder:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
