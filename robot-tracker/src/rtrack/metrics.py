"""Identity accuracy and data-retention baseline, measured against curator labels.

    uv run -m rtrack.metrics --event 2026necmp1 --stage published auto final \\
        --out metrics/2026necmp1_baseline.json

WHAT IS MEASURED, AND AGAINST WHAT. The only ground truth is a curator's label: a team,
or a flag (unknown / notrobot / mixed), attached to one detection in one bundle frame.
Labels are resolved onto the STITCHED tracks, the tracker's own segmentation. For route
time a label vouches for its track only within TRUTH_RADIUS_S of it: the tracker can
switch robots inside a stitched track without changing its id (2026necmp1_qm17 track 9
carried red 1768, then blue 7127), so a label cannot speak for a minute of track away
from it. `unknown` ("cannot tell") is never counted as wrong; `notrobot` rejects a track.

FOUR ACCURACIES, because they fail differently:
  image   one labelled detection: did the solve name the right team?
  track   one labelled stitched track: is the team the solve gave most of it right?
  frame   one bundle frame: were ALL of its labelled robots right at once -- what a
          curator sees, and what a frame of route overlay shows
  route   per team and match: seconds of its route on detections known to be ANOTHER
          robot, against seconds verified correct and seconds that cannot be checked

STAGES, because a solve pinned to the labels it is scored on proves nothing:
  published  the live labelled tracks. Accuracy is IN-SAMPLE (the solve was pinned to
             these very labels) and reported only as a ceiling; retention is real.
  auto       re-solved with no curation at all, and with this match's own curated and
             reviewed prototypes left out of the gallery: what the automatic pipeline
             knows unaided, and what a curator is shown as the pre-fill.
  final      re-solved with the labels of all but one fold of bundle frames, scored on
             the held-out fold, for every fold: what curation delivers on robots the
             curator did not directly label.

RETENTION. Valid data is every stitched detection inside the match window, on the field
(not flagged off-field), inside the calibrated camera view, and not on a track the
curator rejected as unknown. Anything valid that reaches no team's route is DISCARDED,
and is reported in seconds and as a share so it can be driven down.

Every solve is written to a scratch directory; nothing live is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import corrections as CO

PY = sys.executable
# A label vouches for its track only this far either side in time. Propagating a label
# along a whole stitched track scored the solver WRONG for splitting tracks the tracker
# had itself switched between robots -- 2026necmp1_qm19 track 6 was labelled 10063 eight
# times and 6324 twice, and qm15's worst "errors" sat 48-100 s from the only labels.
TRUTH_RADIUS_S = 5.0


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _match_window(robots_json: Path) -> tuple[float, float] | None:
    try:
        w = json.loads(robots_json.read_text(encoding="utf-8")).get("custodyWindow")
    except (OSError, json.JSONDecodeError):
        return None
    if not w:
        return None
    return float(w[0]), float(w[0]) + C.MATCH_SPAN_S


def truth(stem: str) -> dict | None:
    """Curator truth for one match: labels (resolved to stitched tracks) and track truth."""
    corr = C.TRACKER_ROOT / "corrections" / f"{stem}_corrections.json"
    st = C.STAGE1_DIR / f"{stem}_tracks_stitched.jsonl"
    if not (corr.exists() and st.exists()):
        return None
    labels = [l for l in json.loads(corr.read_text(encoding="utf-8")).get("labels") or []
              if l.get("src", "human") == "human" and (l.get("team") or CO.flag_of(l))]
    if not labels:
        return None
    resolved = [l for l in CO.resolve(_rows(st), labels) if l["ok"]]
    # Per stitched track, every answer in time order: a team, or a curator flag.
    # `unknown` means "I looked and cannot tell" (see corrections.FLAGS) -- NOT evidence
    # of anything, so it never makes time wrong; `notrobot` rejects the track.
    timeline: dict[int, list] = defaultdict(list)
    for l in resolved:
        timeline[int(l["tid"])].append((float(l["t"]), str(l["team"]) if l.get("team")
                                        else CO.flag_of(l)))
    tracks, rejected = {}, set()
    for tid, tl in timeline.items():
        tl.sort()
        vals = [v for _t, v in tl]
        teams = {v for v in vals if v not in CO.FLAGS}
        if "notrobot" in vals and not teams:
            rejected.add(tid)
        # track-level truth only where every team answer agrees and none says "mixed"
        if len(teams) == 1 and "mixed" not in vals:
            tracks[tid] = next(iter(teams))
    return {"labels": resolved, "tracks": tracks, "timeline": dict(timeline),
            "rejected": rejected}


def _truth_at(tr: dict, sid, t: float) -> str | None:
    """The curator's answer for this track at this time, if a label is close enough."""
    tl = tr["timeline"].get(sid)
    if not tl:
        return None
    lt, v = min(tl, key=lambda x: abs(x[0] - t))
    if abs(lt - t) > TRUTH_RADIUS_S or v in CO.FLAGS:
        return None
    return v


def score(stem: str, labeled: Path, robots_json: Path, tr: dict,
          only_frames: set[int] | None = None) -> dict:
    """All four accuracies plus retention for one solve output of one match."""
    rows = _rows(labeled)
    # output detection -> (team, stitched id), keyed for label lookup
    by_frame = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            x1, y1, x2, y2 = d["xyxy"]
            by_frame[r["f"]].append(((x1 + x2) / 2, (y1 + y2) / 2, d.get("team"),
                                     d.get("source_tid", d.get("orig_tid", d["tid"]))))
    labels = [l for l in tr["labels"] if l.get("team")
              and (only_frames is None or int(l["f"]) in only_frames)]
    if only_frames is not None:
        # Held-out scoring: route time may only be judged against HELD-OUT labels. The
        # solve was pinned to the rest, so counting them would score the pins.
        tl = defaultdict(list)
        for l in tr["labels"]:
            if int(l["f"]) in only_frames:
                tl[int(l["tid"])].append((float(l["t"]), str(l["team"]) if l.get("team")
                                          else CO.flag_of(l)))
        tr = dict(tr, timeline={k: sorted(v) for k, v in tl.items()})

    # image
    img = Counter(); frame_out = defaultdict(Counter)
    for l in labels:
        cand = by_frame.get(int(l["f"]), [])
        best = min(cand, key=lambda c: (c[0] - l["xy"][0]) ** 2 + (c[1] - l["xy"][1]) ** 2,
                   default=None)
        if best is None or math.dist(best[:2], l["xy"]) > CO.MATCH_PX:
            outcome = "missing"
        elif best[2] is None:
            outcome = "unassigned"
        else:
            outcome = "correct" if str(best[2]) == str(l["team"]) else "wrong"
        img[outcome] += 1
        frame_out[int(l["f"])][outcome] += 1
    frame_rows = [[sum(c.values()), c["wrong"], c["unassigned"] + c["missing"]]
                  for c in frame_out.values()]

    # track: majority team the solve gave each truth track
    held_tracks = ({int(l["tid"]) for l in labels if int(l["tid"]) in tr["tracks"]}
                   if only_frames is not None else set(tr["tracks"]))
    got: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        for d in r["dets"]:
            sid = d.get("source_tid", d.get("orig_tid", d["tid"]))
            if sid in held_tracks and d.get("team"):
                got[sid][str(d["team"])] += 1
    trk = Counter()
    for tid in held_tracks:
        want = tr["tracks"].get(tid)
        if want is None:
            continue
        if not got[tid]:
            trk["unassigned"] += 1
        else:
            trk["correct" if got[tid].most_common(1)[0][0] == want else "wrong"] += 1

    # route seconds, within the match window
    win = _match_window(robots_json)
    ts = [r["t"] for r in rows]
    dt = float(np.median(np.diff(ts))) if len(ts) > 2 else 1 / 15
    route = defaultdict(Counter)
    for r in rows:
        if win and not (win[0] <= r["t"] <= win[1]):
            continue
        for d in r["dets"]:
            team = d.get("team")
            if not team:
                continue
            sid = d.get("source_tid", d.get("orig_tid", d["tid"]))
            want = _truth_at(tr, sid, r["t"])
            if want is not None:
                route[str(team)]["wrong" if want != str(team) else "correct"] += dt
            else:
                route[str(team)]["unverified"] += dt

    return {"image": dict(img), "track": dict(trk),
            # per bundle frame: [robots labelled, wrong team, no team] -- a missing robot
            # and a misidentified one are different failures and must not be pooled
            "frames": frame_rows,
            "route": {t: {k: round(v, 2) for k, v in c.items()} for t, c in route.items()},
            "retention": retention(stem, rows, win, tr)}


def retention(stem: str, rows: list[dict], win, tr: dict) -> dict:
    """Seconds of valid detection time that reach, or fail to reach, a team's route."""
    pp = C.STAGE2_DIR / f"{stem}_prepos.json"
    if not pp.exists() or not win:
        return {}
    samples = json.loads(pp.read_text(encoding="utf-8"))["samples"]
    vdoc = None
    try:
        from . import viewcheck as VC
        vdoc = VC.load(stem)
    except Exception:
        VC = None
    ts = sorted({s["t"] for s in samples})
    dt = float(np.median(np.diff(ts))) if len(ts) > 2 else 1 / 15
    teamed = {(r["f"], d.get("source_tid", d.get("orig_tid", d["tid"])))
              for r in rows for d in r["dets"] if d.get("team")}
    present = {(r["f"], d.get("source_tid", d.get("orig_tid", d["tid"])))
               for r in rows for d in r["dets"]}
    c = Counter()
    for s in samples:
        if not (win[0] <= s["t"] <= win[1]):
            continue
        if "offfield" in s["flags"]:
            c["offField"] += dt; continue
        if vdoc and VC and not VC.is_valid_at(vdoc, s["t"]):
            c["offView"] += dt; continue
        tid = s["tid"]
        if tid in tr["rejected"]:
            c["curatorRejected"] += dt; continue
        c["valid"] += dt
        key = (s["f"], tid)
        if key in teamed:
            c["kept"] += dt
        elif key in present:
            c["discardedNoTeam"] += dt
        else:
            c["discardedBeforeSolve"] += dt      # filtered out upstream of assignment
    out = {k: round(v, 1) for k, v in c.items()}
    if c["valid"]:
        out["discardedShare"] = round((c["discardedNoTeam"] + c["discardedBeforeSolve"])
                                      / c["valid"], 4)
    return out


# ── solves ────────────────────────────────────────────────────────────────

REUSE = False


class _nullctx:
    def __init__(self, p: Path): self.p = p
    def __enter__(self): self.p.mkdir(parents=True, exist_ok=True); return str(self.p)
    def __exit__(self, *a): return False


def _solve(stem: str, event: str, calib: str | None, out_dir: Path,
           votes: Path | None, corrections: Path | None) -> bool:
    if REUSE and (out_dir / f"{stem}_labeled.jsonl").exists():
        return True
    st = C.STAGE1_DIR / f"{stem}_tracks_stitched.jsonl"
    a = [PY, "-m", "rtrack.robots", stem, "--tracks", str(st), "--match", stem,
         "--deconflict", "3", "--output-dir", str(out_dir)]
    if calib:
        a += ["--calib-from", calib]
    pp = C.STAGE2_DIR / f"{stem}_prepos.json"
    if pp.exists():
        a += ["--positions", str(pp)]
    if votes and votes.exists():
        a += ["--identity", str(votes)]
    if corrections:
        a += ["--corrections", str(corrections)]
    r = subprocess.run(a, cwd=C.TRACKER_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    solve failed ({r.returncode}): {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
    return r.returncode == 0 and (out_dir / f"{stem}_labeled.jsonl").exists()


def _votes(stem: str, event: str, out: Path) -> Path | None:
    if REUSE and out.exists():
        return out
    st = C.STAGE1_DIR / f"{stem}_tracks_stitched.jsonl"
    a = [PY, "-m", "rtrack.reid", "votes", stem, "--event", event, "--match", stem,
         "--tracks", str(st), "--backend", "cnn", "--out", str(out),
         "--exclude-gallery-match", stem]
    subprocess.run(a, cwd=C.TRACKER_ROOT, capture_output=True, text=True)
    return out if out.exists() else None


def run_match(stem: str, event: str, stages: list[str], folds: int, work: Path) -> dict:
    tr = truth(stem)
    if tr is None:
        return {}
    calib = event
    res = {}
    if "published" in stages:
        lab = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
        if lab.exists():
            res["published"] = score(stem, lab, C.STAGE3_DIR / f"{stem}_robots.json", tr)
    votes = None
    if {"auto", "final"} & set(stages):
        votes = _votes(stem, event, work / f"{stem}_votes.json")
    if "auto" in stages:
        d = work / "auto"
        if _solve(stem, event, calib, d, votes, None):
            res["auto"] = score(stem, d / f"{stem}_labeled.jsonl", d / f"{stem}_robots.json", tr)
    if "final" in stages:
        corr = json.loads((C.TRACKER_ROOT / "corrections" / f"{stem}_corrections.json")
                          .read_text(encoding="utf-8"))
        frames = sorted({int(l["f"]) for l in corr.get("labels") or []})
        parts = []
        for k in range(folds):
            held = set(frames[k::folds])
            sub = dict(corr, labels=[l for l in corr["labels"] if int(l["f"]) not in held])
            cp = work / f"{stem}_fold{k}_corrections.json"
            cp.write_text(json.dumps(sub), encoding="utf-8")
            d = work / f"final{k}"
            if _solve(stem, event, calib, d, votes, cp):
                parts.append(score(stem, d / f"{stem}_labeled.jsonl",
                                   d / f"{stem}_robots.json", tr, only_frames=held))
        if parts:
            res["final"] = _merge_folds(parts)
    return res


def _merge_folds(parts: list[dict]) -> dict:
    out = {"image": Counter(), "track": Counter(), "frames": [], "route": {}, "retention": {}}
    for p in parts:
        out["image"].update(p["image"]); out["track"].update(p["track"])
        out["frames"].extend(p["frames"])
    # route and retention depend on the whole solve, not the held-out labels: average folds
    teams = {t for p in parts for t in p["route"]}
    for t in teams:
        acc = Counter()
        for p in parts:
            acc.update(p["route"].get(t, {}))
        out["route"][t] = {k: round(v / len(parts), 2) for k, v in acc.items()}
    ret = Counter()
    for p in parts:
        ret.update({k: v for k, v in p["retention"].items() if k != "discardedShare"})
    out["retention"] = {k: round(v / len(parts), 1) for k, v in ret.items()}
    if out["retention"].get("valid"):
        r = out["retention"]
        r["discardedShare"] = round((r.get("discardedNoTeam", 0) + r.get("discardedBeforeSolve", 0))
                                    / r["valid"], 4)
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in out.items()}


# ── summary ───────────────────────────────────────────────────────────────

def _dist(v) -> dict:
    a = np.array(v, float) if len(v) else np.zeros(1)
    return {"median": round(float(np.median(a)), 1), "mean": round(float(a.mean()), 1),
            "p90": round(float(np.percentile(a, 90)), 1)}


def summarise(per_match: dict[str, dict], stage: str) -> dict:
    img, trk, ret = Counter(), Counter(), Counter()
    frames = []
    known, scaled, lengths, ver_tot, all_tot, wrong_tot = [], [], [], 0.0, 0.0, 0.0
    for m in per_match.values():
        s = m.get(stage)
        if not s:
            continue
        img.update(s["image"]); trk.update(s["track"]); frames.extend(s.get("frames", []))
        ret.update({k: v for k, v in s["retention"].items() if k != "discardedShare"})
        for t, r in s["route"].items():
            w, c, u = r.get("wrong", 0.0), r.get("correct", 0.0), r.get("unverified", 0.0)
            total = w + c + u
            known.append(w); lengths.append(total)
            if w + c:
                scaled.append(w / (w + c) * total)
            ver_tot += w + c; all_tot += total; wrong_tot += w
    rate = lambda c, k: round(c[k] / max(sum(c.values()), 1), 4)
    F = [f for f in frames if f[0] >= 2]
    robots = sum(f[0] for f in F)
    by_n = defaultdict(list)
    for f in F:
        by_n[f[0]].append(f[1] == 0 and f[2] == 0)
    return {
        "matches": sum(1 for m in per_match.values() if m.get(stage)),
        "image": {"n": sum(img.values()), "correct": rate(img, "correct"),
                  "wrong": rate(img, "wrong"), "unassigned": rate(img, "unassigned"),
                  "missing": rate(img, "missing")},
        "track": {"n": sum(trk.values()), "correct": rate(trk, "correct"),
                  "wrong": rate(trk, "wrong"), "unassigned": rate(trk, "unassigned")},
        # A frame fails for two different reasons: a robot credited to the WRONG team,
        # or a robot credited to NO team (a gap in its route, not an identity error).
        # With ~6 labelled robots a frame, even 6% per-robot failure fails a third of
        # frames, so the per-robot rate is reported alongside.
        "frame": {"n": len(F), "meanRobots": round(robots / max(len(F), 1), 2),
                  "allPresentAndCorrect": round(sum(f[1] == 0 and f[2] == 0 for f in F) / max(len(F), 1), 4),
                  "noWrongIdentity": round(sum(f[1] == 0 for f in F) / max(len(F), 1), 4),
                  "noMissingRobot": round(sum(f[2] == 0 for f in F) / max(len(F), 1), 4),
                  "perRobotCorrect": round(1 - sum(f[1] + f[2] for f in F) / max(robots, 1), 4),
                  "allCorrectByRobotCount": {str(n): round(float(np.mean(v)), 3)
                                             for n, v in sorted(by_n.items())}},
        # Route time is checkable only on tracks a curator labelled somewhere. KNOWN wrong
        # seconds count only that part and are a LOWER BOUND; SCALED applies each
        # route's verified error rate to its whole length. Both are OPTIMISTIC: curators
        # label legible moments, and a tracker switch inside one stitched track is
        # credited as correct. The image and frame figures do not share that blind spot.
        "route": {"routes": len(known), "meanLengthS": round(float(np.mean(lengths)) if lengths else 0.0, 1),
                  "verifiedShare": round(ver_tot / max(all_tot, 1e-9), 4),
                  "wrongShareOfVerified": round(wrong_tot / max(ver_tot, 1e-9), 4),
                  "knownWrongSeconds": _dist(known),
                  "scaledWrongSeconds": _dist(scaled)},
        "retention": {**{k: round(v, 1) for k, v in ret.items()},
                      "discardedShare": round((ret["discardedNoTeam"] + ret["discardedBeforeSolve"])
                                              / max(ret["valid"], 1e-9), 4)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event", required=True)
    ap.add_argument("--stage", nargs="+", default=["published", "auto", "final"],
                    choices=["published", "auto", "final"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--matches", nargs="*", default=None, help="limit to these match keys")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="keep every solve here instead of a temporary directory, so a "
                         "change to SCORING can be re-measured with --reuse in seconds")
    ap.add_argument("--reuse", action="store_true",
                    help="with --work-dir: score solves already there instead of re-solving")
    args = ap.parse_args(argv)

    stems = sorted(p.name[: -len("_corrections.json")]
                   for p in (C.TRACKER_ROOT / "corrections").glob(f"{args.event}_*_corrections.json"))
    if args.matches:
        stems = [s for s in stems if s in set(args.matches)]
    per = {}
    global REUSE
    REUSE = args.reuse
    with (tempfile.TemporaryDirectory(prefix="rtrack-metrics-") if args.work_dir is None
          else _nullctx(args.work_dir)) as tmp:
        for i, stem in enumerate(stems, 1):
            w = Path(tmp) / stem
            w.mkdir(parents=True, exist_ok=True)
            print(f"[metrics] {i}/{len(stems)} {stem}", flush=True)
            try:
                per[stem] = run_match(stem, args.event, args.stage, args.folds, w)
            except Exception as exc:                  # noqa: BLE001 - one match must not sink the run
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
    doc = {"schemaVersion": 1, "kind": "rtrack-metrics", "event": args.event,
           "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "stages": args.stage, "folds": args.folds,
           "summary": {s: summarise(per, s) for s in args.stage},
           "perMatch": per}
    out = args.out or (C.TRACKER_ROOT / "metrics" /
                       f"{args.event}_{datetime.now():%Y%m%d-%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc["summary"], indent=1))
    print(f"[metrics] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
