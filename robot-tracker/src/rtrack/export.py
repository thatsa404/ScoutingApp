"""Stage 3b -- emit `rtrack-tracks v1`, the one file the scouting app consumes.

    uv run -m rtrack.export GSxbsE42o5o --match 2026necmp_f1m3 --publish

This is the handoff point between the Python pipeline and the Vite app, so the file is
deliberately self-contained: given only this JSON and the field PNG that ships in the
app's `public/`, a consumer can draw every robot's route without reading anything from
`calib/`, `out/`, or TBA. Nothing under robot-tracker/ is deployed.

Four decisions worth defending, because each one was wrong in an earlier draft:

1. FIELD SIZE COMES FROM calib/field_ref_2026.json AND IS NEVER RE-DERIVED. The
   authoritative WPILib k2026RebuiltWelded layout is 16.541 x 8.069 m, aspect 2.0499.
   The commonly quoted 54x27 ft gives 16.4592 x 8.2296 and aspect 2.0000 -- a 2% width
   error. An earlier version of the reference file forced the nominal figure and then
   "verified" it against field structures, which was circular and wrong.

2. THE METRE -> IMAGE-PIXEL MAPPING TRAVELS IN THE FILE. The app has no access to the
   calibration, so `fieldRectPx` and `pxPerMeter` ship here. Consumers map with

       px_x = fieldRectPx.x0 + X * pxPerMeter
       px_y = fieldRectPx.y1 - Y * pxPerMeter        (image +y is field -y)

   which is exact: 8.069 * 174.74 = 1410 = y1-y0, and 16.541 * 174.74 = 2890.6 = x1-x0.

3. `t` IS MATCH-RELATIVE, 0 = AUTO START, derived by routes.detect_auto_window rather
   than assumed. Scouting data in this app is keyed by match phase (auto / shift1-4 /
   endgame), not by video timestamp, so match-relative time is what makes the two
   joinable at all. `source.matchStartVideoSec` maps back to the video for seeking.

4. OUTPUT IS DECIMATED, THE TRACKER IS NOT. Measured on f1m3: 15 Hz native is 359 KB
   raw / 72 KB gzipped, 5 Hz is 120 KB / 26 KB. Route plots do not need 15 Hz. The
   inverse -- sampling the TRACKER coarsely -- breaks IoU association and is refused in
   rtrack.track for that reason; see its `--sample-hz` note.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .acquire import video_id
from . import tba as tba_mod

# Seconds either side of the match kept in the export. Enough for the run-up to the
# starting line and the moment after the buzzer; not enough for field reset.
EXPORT_PAD_S = 5.0

SCHEMA_VERSION = 1
GENERATOR_VERSION = "0.1.0"
DEFAULT_HZ = 5.0
TRACK_HZ = 15.0
# A gap longer than this is reported in `gaps` so a consumer can break the polyline
# rather than drawing a straight line through a period nobody observed.
GAP_S = 1.0


def _load(p: Path, what: str) -> dict:
    if not p.exists():
        raise SystemExit(f"[export] {p} not found -- {what}")
    return json.loads(p.read_text(encoding="utf-8"))


def build(stem: str, match_key: str, hz: float = DEFAULT_HZ,
          calib_stem: str | None = None, allow_stale: bool = False) -> dict:
    positions = _load(C.STAGE2_DIR / f"{stem}_positions.json",
                      "run rtrack.project against the LABELLED track file first")
    robots_doc = _load(C.STAGE3_DIR / f"{stem}_robots.json",
                       "run rtrack.robots first")
    ref = _load(C.CALIB_DIR / "field_ref_2026.json", "field reference missing")
    calib = _load(C.CALIB_DIR / f"{calib_stem or stem}.json",
                  "this camera has no calibration -- run rtrack.calibrate")

    # STALENESS IS THE FAILURE MODE THIS PIPELINE ACTUALLY HAS. positions.json is
    # derived from labeled.jsonl by rtrack.project, and nothing re-derives it when the
    # labelling changes. Measured: the first published f1m3 export was built on a
    # positions.json 19 HOURS older than its labelling, so it predated the grayscale
    # descriptor, the CP-SAT determinism fix, the field filter and the alliance-
    # confidence fix -- and still showed 6329 parked in the centre fuel pile, an
    # artifact that had already been fixed upstream. It looked like a tracker defect
    # and was a freshness bug. Refuse rather than warn: a warning scrolls past.
    lab = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
    pos = C.STAGE2_DIR / f"{stem}_positions.json"
    if (not allow_stale and lab.exists()
            and pos.stat().st_mtime < lab.stat().st_mtime - 1):
        raise SystemExit(
            f"[export] STALE: {pos.name} is older than {lab.name}.\n"
            f"         The labelling changed after positions were computed, so this "
            f"export would publish outdated routes.\n"
            f"         Fix: uv run -m rtrack.project {stem} --tracks {lab}"
            + (f" --calib-from <video>" if calib_stem else "")
            + "\n         Override with --allow-stale if you really mean it.")

    samples = positions["samples"]
    if not any(s.get("team") for s in samples):
        raise SystemExit(
            "[export] no sample carries a team. rtrack.project was run against the "
            "raw tracks; re-run it with --tracks out/stage3/<stem>_labeled.jsonl")

    # Match-relative time. Fall back to the custody window's start, which robots.py
    # already derived, rather than silently exporting video time as if it were match
    # time -- that would look right and join wrongly.
    # Auto START is measured (it moves with the match); auto DURATION comes from the
    # rulebook via C.AUTO_S. detect_auto_window reports an end too, but it is not
    # trustworthy -- see the note at C.AUTO_S. Since sample times are relative to t0,
    # auto is simply t in [0, AUTO_S].
    auto_end = C.AUTO_S
    try:
        from .routes import detect_auto_window
        t0, _t1 = detect_auto_window(positions)
    except Exception:
        t0 = (robots_doc.get("custodyWindow") or [0.0])[0]
        print(f"[export] auto-window detection failed; using custody start t0={t0}")

    m = tba_mod.match_by_key(match_key)
    alliance_of, station_of = {}, {}
    for side in ("red", "blue"):
        for i, t in enumerate(m[side], start=1):
            alliance_of[str(t)] = side
            station_of[str(t)] = i

    # CLIP TO THE MATCH. A slice is cut with generous padding and can run for minutes
    # either side -- 2026necmp1_qm13 is 678 s around a ~166 s match, so 48% of its
    # exported samples were staging and post-match milling about. Harmless to the
    # numbers and ruinous to the UI: the route slider's range is set by the data, so
    # dragging it spent most of its travel outside the match.
    #
    # Bounds are match-relative because `t` already is: auto starts at 0 and the match
    # is over by C.MATCH_SPAN_S. The pad keeps the approach to the starting line and
    # the moment after the buzzer, both of which are worth seeing.
    lo, hi = -EXPORT_PAD_S, C.MATCH_SPAN_S + EXPORT_PAD_S

    step = max(1.0 / max(hz, 0.1), 1e-6)
    by_team: dict[str, list] = defaultdict(list)
    n_clipped = 0
    for s in samples:
        team = s.get("team")
        fl = s.get("flags", [])
        if not (lo <= s["t"] - t0 <= hi):
            n_clipped += 1
            continue
        # `viewmoved` is excluded for the same reason as `offfield`: these are metres
        # the homography could not have produced correctly. The gap it leaves is
        # honest, and `gaps` below already describes gaps, so a route that stops while
        # the broadcast was on another camera reads as missing rather than as wrong.
        if not team or "offfield" in fl or "viewmoved" in fl:
            continue
        by_team[str(team)].append(s)

    if n_clipped:
        print(f"[export] clipped {n_clipped} sample(s) outside the match "
              f"(t {lo:.0f}..{hi:.0f} s relative to auto start)")

    custody = robots_doc.get("custody") or {}
    out_robots = []
    for team in sorted(by_team, key=lambda t: (alliance_of.get(t, "z"),
                                               station_of.get(t, 9))):
        ss = sorted(by_team[team], key=lambda z: z["t"])
        picked, last_t = [], None
        for s in ss:
            t = round(s["t"] - t0, 3)
            if last_t is not None and t - last_t < step:
                continue
            last_t = t
            picked.append({"t": t, "x": round(s["x"], 3), "y": round(s["y"], 3),
                           "conf": round(float(s.get("conf") or 0.0), 3)})
        gaps = [{"tStart": a["t"], "tEnd": b["t"]}
                for a, b in zip(picked, picked[1:]) if b["t"] - a["t"] > GAP_S]
        out_robots.append({
            "team": team,
            "alliance": alliance_of.get(team) or (ss[0].get("alliance") or "unknown"),
            "station": station_of.get(team),
            "custody": round((custody.get(team, {}).get("pct") or 0.0) / 100.0, 4),
            "samples": picked,
            "gaps": gaps,
        })

    q = positions.get("quality") or {}
    cur = robots_doc.get("curator") or {}
    n_labels = len(cur.get("pinned") or {}) + len(cur.get("flags") or {})
    # Visibility depends only on the camera pose, so it is a property of the
    # CALIBRATION and identical for every match sharing one. Computed here rather than
    # stored in calib/ so an older calibration file gains it without being re-fitted.
    vis_poly = vis_frac = cam_side = None
    try:
        from .project import load_calib, visible_region
        _H, _lens = load_calib(calib_stem or stem)
        vis_poly, vis_frac, cam_side = visible_region(_H, ref, _lens)
    except Exception as e:
        print(f"[export] visibility not computed ({type(e).__name__}); "
              f"the app will draw no visibility overlay")

    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "generator": {"name": "rtrack", "version": GENERATOR_VERSION,
                      "createdAt": datetime.now(timezone.utc)
                      .isoformat(timespec="seconds").replace("+00:00", "Z")},
        "match": {"key": match_key, "eventKey": match_key.split("_")[0],
                  "compLevel": m.get("compLevel"), "setNumber": m.get("setNumber"),
                  "matchNumber": m.get("matchNumber")},
        "source": {"provider": "youtube", "videoId": stem,
                   "matchStartVideoSec": round(float(t0), 3)},
        "field": {"year": C.YEAR, "units": "meters", "convention": "wpilib-2026",
                  "sizeM": ref["fieldSizeM"],          # authoritative; see docstring
                  "imageRef": "field/2026-field.png",
                  "imageSize": ref["imageSize"],
                  "fieldRectPx": ref["fieldRectPx"],
                  "pxPerMeter": ref["pxPerMeter"],
                  # What this camera can actually SEE, so a route that stops at a far
                  # corner reads as unobservable rather than as lost tracking. Null
                  # when it cannot be computed; consumers must treat that as "unknown
                  # visibility" and draw nothing, never as "all visible".
                  "visiblePolyM": vis_poly,
                  "visibleFrac": vis_frac,
                  # "low-y" or "high-y": which touchline the camera sits behind. A
                  # renderer should put that side at the BOTTOM, so the plot matches
                  # what someone watching the video saw. null = unknown, draw as-is.
                  "cameraSide": cam_side},
        "calibration": {"mode": calib.get("mode", "static-homography"),
                        "pointCount": calib.get("pointCount"),
                        "reprojErrorM": calib.get("reprojErrorM")},
        "sampling": {"trackHz": TRACK_HZ, "outputHz": hz},
        # Sample times are relative to auto start, so auto is t in [0, autoEndT].
        # null when motion detection could not find the window -- consumers should then
        # not draw a phase split rather than guess one.
        "phases": {"autoEndT": auto_end},
        "robots": out_robots,
        "quality": {
            "samplesIn": q.get("samples"),
            "samplesOut": sum(len(r["samples"]) for r in out_robots),
            "tracks": q.get("tracks"),
            "offField": q.get("offField"),
            # Whether the camera-pose check ran at all is reported alongside its result,
            # so a zero here can be read as "none found" rather than "never looked".
            "viewChecked": q.get("viewChecked", False),
            "viewMoved": q.get("viewMoved", 0),
            "kinematicViolations": q.get("kinematicViolations"),
            "meanCustody": round(
                sum(r["custody"] for r in out_robots) / max(len(out_robots), 1), 4),
            "custodyConflicts": len(robots_doc.get("custodyConflicts") or []),
            "curated": bool(n_labels),
            "curatorLabels": n_labels,
        },
    }
    return doc


def validate(doc: dict) -> list[str]:
    """Cheap invariants. A consumer that trusts this file deserves them checked."""
    bad = []
    FL, FW = doc["field"]["sizeM"]
    slack = 1.0
    n = len(doc["robots"])
    if n != 6:
        bad.append(f"{n} robots, expected 6")
    for r in doc["robots"]:
        if not r["samples"]:
            bad.append(f"{r['team']}: no samples")
            continue
        ts = [s["t"] for s in r["samples"]]
        if ts != sorted(ts):
            bad.append(f"{r['team']}: samples not sorted by t")
        off = sum(1 for s in r["samples"]
                  if not (-slack <= s["x"] <= FL + slack
                          and -slack <= s["y"] <= FW + slack))
        if off:
            bad.append(f"{r['team']}: {off} sample(s) outside the field +-{slack} m")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3b: emit rtrack-tracks v1.")
    ap.add_argument("video")
    ap.add_argument("--match", required=True, help="TBA match key, e.g. 2026necmp_f1m3")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ,
                    help=f"output sample rate (default {DEFAULT_HZ}); the TRACKER rate "
                         f"is unaffected and must not be lowered -- see rtrack.track")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-stale", action="store_true",
                    help="export even when positions.json predates the labelling "
                         "(see build(); this published wrong routes once already)")
    ap.add_argument("--calib-from", default=None, metavar="VIDEO",
                    help="reuse another video's calibration (same camera)")
    ap.add_argument("--no-relay", action="store_true",
                    help="publish to public/tracks/ only, skipping the relay push. The "
                         "relay is how routes reach the app without a commit; this is "
                         "the escape hatch for rebuilding a back catalogue without "
                         "filling it.")
    ap.add_argument("--publish", action="store_true",
                    help="also copy into ../public/tracks/ so the app can fetch it")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    doc = build(stem, args.match, args.hz,
                calib_stem=video_id(args.calib_from) if args.calib_from else None,
                allow_stale=args.allow_stale)

    problems = validate(doc)
    for p in problems:
        print(f"[export] WARNING: {p}")

    out = args.out or (C.STAGE3_DIR / f"{args.match}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[export] {len(doc['robots'])} robots, "
          f"{doc['quality']['samplesOut']} samples at {args.hz} Hz, "
          f"{kb:.0f} KB -> {out}")

    if args.publish:
        dest = C.REPO_ROOT / "public" / "tracks" / f"{args.match}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, dest)
        print(f"[export] published -> {dest}")
        n = write_manifest(dest.parent)
        print(f"[export] manifest lists {n} match(es)")

        # AND TO THE RELAY, because writing public/tracks/ only reaches an app someone
        # has since committed and deployed. The watcher runs unattended: it curated,
        # solved, projected and exported ten 2026mawor matches that stayed invisible in
        # the app for exactly that reason. Pushing here makes the live path automatic
        # and leaves git as the durable one for past events and their archives.
        #
        # Never fatal. A relay that is unreachable, unconfigured or full must not fail
        # an export whose real output is already safely on disk.
        if not args.no_relay:
            try:
                from . import relay as _relay
                res = _relay.put("tracks", args.match, doc)
                print(f"[export] relay -> tracks/{args.match} "
                      f"({res.get('bytes', 0) / 1024:.0f} KB)")
            except Exception as e:
                print(f"[export] relay push skipped ({type(e).__name__}: {e}); "
                      f"public/tracks/ still needs committing for this one")
    return 1 if problems else 0


def write_manifest(tracks_dir: Path) -> int:
    """Rebuild public/tracks/index.json by scanning the directory.

    The app needs this because it CANNOT discover tracks any other way. Probing
    <matchKey>.json for every match of an event would be dozens of requests for files
    that usually do not exist, and a miss is indistinguishable from a hit without
    inspecting content-type (this host serves 200 + SPA HTML for absent files).

    More importantly it decouples the routes UI from `db.matches`, which holds
    QUALIFICATION MATCHES ONLY -- main.js:4173 skips anything with comp_level != 'qm'.
    Every playoff match, including the two finals exported so far, is invisible to the
    app through the normal schedule path. The manifest carries its own match metadata
    and team list so a consumer can find and load a match the schedule never lists, and
    can filter by team without downloading anything.

    Rebuilt by scanning rather than appended to, so deleting a file cleans up after
    itself.
    """
    rows = []
    for p in sorted(tracks_dir.glob("*.json")):
        if p.name == "index.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("schemaVersion") != 1:
                continue
            m, q = d.get("match") or {}, d.get("quality") or {}
            rows.append({
                "key": m.get("key") or p.stem,
                "eventKey": m.get("eventKey") or p.stem.split("_")[0],
                "compLevel": m.get("compLevel"),
                "setNumber": m.get("setNumber"),
                "matchNumber": m.get("matchNumber"),
                "videoId": (d.get("source") or {}).get("videoId"),
                "teams": [str(r.get("team")) for r in d.get("robots") or []],
                "curated": bool(q.get("curated")),
                "meanCustody": q.get("meanCustody"),
                "bytes": p.stat().st_size,
                # When this export was produced, so a consumer can tell a FINISHED
                # match from one whose curator answers arrived after it was last built.
                # Without it, "an answer exists on the relay" is indistinguishable from
                # "an answer is still waiting to be applied" -- answers are not deleted
                # when consumed, so a published match reads as pending forever.
                "exportedAt": (d.get("generator") or {}).get("createdAt"),
            })
        except Exception as e:
            print(f"[export] manifest: skipping {p.name} ({e})")
    # Per-event appearance-gallery coverage, so the app can show whether a match can be
    # auto-identified before anyone curates it. `reid` needs a team in the gallery to
    # vote for it at all; a match where all six are known should come back ~84% correct
    # (measured), one where none are is back to geometry alone (~13%).
    gallery = {}
    for ev in sorted({r["eventKey"] for r in rows if r.get("eventKey")}):
        gp = C.STAGE3_DIR / f"{ev}_gallery.npz"
        if not gp.exists():
            continue
        try:
            import numpy as np
            z = np.load(gp, allow_pickle=False)
            gallery[ev] = {str(t): int(c) for t, c in zip(z["teams"], z["count"])}
        except Exception as e:
            print(f"[export] manifest: could not read {gp.name} ({e})")
    for r in rows:
        known = gallery.get(r.get("eventKey"), {})
        teams = r.get("teams") or []
        r["galleryKnown"] = sum(1 for t in teams if t in known)
        r["galleryTotal"] = len(teams)

    doc = {"schemaVersion": 1,
           "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")
           .replace("+00:00", "Z"),
           "gallery": gallery,
           "matches": rows}
    (tracks_dir / "index.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(main())
