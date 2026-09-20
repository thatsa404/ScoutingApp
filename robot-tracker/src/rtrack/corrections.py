"""Curator corrections -- ground truth, anchored to detections rather than tracks.

    robot-tracker/corrections/<matchkey>.json      (tracked in git; tiny and precious)

WHY NOT ANCHOR TO TRACK IDS. Track ids are manufactured by split_on_alliance,
split_on_appearance and split_chimeras, every one of which renumbers, and we retune
those thresholds constantly. A correction reading "track 22 is 9644" is void the
moment any of them changes. A correction reading "the robot at (812, 604) in frame
1234 is 9644" survives resplitting, retracking and solver changes alike -- it only
breaks if the DETECTOR re-runs, and even then it degrades to a near-miss rather than
a silent mislabel.

That anchoring is also what makes these worth keeping. Every metric in this project so
far has been a proxy: vote share, alliance consistency, whether a chicklet row looks
mixed. None is ground truth. These labels are, so they are simultaneously the input to
the solver and the eval set we have never had.

TWO LABELS ON ONE TRACK IS A CUT INSTRUCTION. If a curator marks frame 100 as 6329 and
frame 500 as 9644 and both land on the same track, they have said something stronger
than either label alone: that track contains two robots. We cut it between them rather
than making them argue. That is a better signal than any of our heuristics, because it
comes from someone who looked.

Schema (v1):

    {"schemaVersion": 1, "match": "2026necmp_f1m3", "video": "GSxbsE42o5o",
     "by": "james", "createdAt": "2026-09-13T...",
     "labels": [{"f": 1234, "xy": [812, 604], "team": "9644"},
                {"f": 2100, "xy": [455, 700], "mixed": true}]}

Four non-team answers, and they are deliberately NOT interchangeable, because they
ask the pipeline for different things:

    mixed      more than one robot here and I cannot say where the join is
               -> drop from grouping. Prefer labelling individual crops instead: two
                  differing labels on one track is a CUT, which recovers both robots.
    notrobot   this is not a robot at all -- field element, ball pile, a person
               -> drop from grouping AND record it, because it is a detector false
                  positive and we want to be able to count them.
    unknown    I looked and cannot tell which team this is
               -> change nothing. The solver still decides; this only records that a
                  human could not, which is what separates "hard" from "not yet asked".
    (skipped)  not answered at all -- never written to the file.

The difference between `unknown` and skipping matters for the loop: a skipped track
comes back next pass, an `unknown` one does not, and the count of `unknown` answers is
the honest ceiling on what curation can achieve on this footage.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import config as C

SCHEMA = 1
MATCH_PX = 90.0        # a label must land this close to a detection centre


def load(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schemaVersion") != SCHEMA:
        raise SystemExit(f"{path}: schemaVersion {doc.get('schemaVersion')} "
                         f"is not supported (expected {SCHEMA})")
    if not isinstance(doc.get("labels"), list):
        raise SystemExit(f"{path}: no labels array")
    return doc


def resolve(rows, labels, max_px: float = MATCH_PX) -> list[dict]:
    """Attach each label to the track that owns the detection it points at.

    Nearest box-centre within `max_px`, which tolerates the detector jittering a few
    pixels between runs but refuses to snap a label onto a different robot. Anything
    unresolved is returned with tid None and reported, never silently dropped -- a
    correction that quietly does nothing is worse than one that errors.
    """
    by_frame = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                x1, y1, x2, y2 = d["xyxy"]
                by_frame[r["f"]].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                         d["tid"], r["t"]))

    out = []
    for lab in labels:
        f = int(lab["f"])
        lx, ly = float(lab["xy"][0]), float(lab["xy"][1])
        best, best_d = None, None
        for cx, cy, tid, t in by_frame.get(f, ()):
            d2 = (cx - lx) ** 2 + (cy - ly) ** 2
            if best_d is None or d2 < best_d:
                best, best_d = (tid, t), d2
        ok = best is not None and best_d ** 0.5 <= max_px
        out.append({**lab,
                    "tid": best[0] if ok else None,
                    "t": best[1] if ok else None,
                    "dist": (best_d ** 0.5) if best_d is not None else None,
                    "ok": ok})
    return out


FLAGS = ("mixed", "notrobot", "unknown")


def flag_of(label: dict) -> str | None:
    for f in FLAGS:
        if label.get(f):
            return f
    return None


def cuts_from(resolved: list[dict]) -> dict[int, list[float]]:
    """Where a curator's own labels disagree inside one track, cut between them.

    An `unknown` placed on a specific crop counts as a value here, not as a missing
    one. "6329 at t50, unknown at t90" is a real statement -- the track is 6329 for a
    while and then becomes something the curator cannot name -- so it earns a cut just
    as "6329 then 9644" does. Without this, the only expressible answers on a long
    track were "all of it is 6329" (which a single label silently asserts over the
    whole track) or "discard all of it", and neither is what a partial reading means.

    Cutting on the boundary leaves the identified piece pinnable and the unidentified
    piece free for the solver, which is the whole point: partial knowledge should
    contribute the part that is known without claiming the part that is not.
    """
    by_tid: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for r in resolved:
        if not r["ok"]:
            continue
        f = flag_of(r)
        if f == "unknown":
            by_tid[r["tid"]].append((r["t"], "?"))
        elif r.get("team") and not f:
            by_tid[r["tid"]].append((r["t"], str(r["team"])))

    cuts: dict[int, list[float]] = {}
    for tid, items in by_tid.items():
        items.sort()
        for (t0, a), (t1, b) in zip(items, items[1:]):
            if a != b:
                cuts.setdefault(tid, []).append((t0 + t1) / 2.0)
    return cuts


def pins_from(resolved: list[dict]) -> tuple[dict[int, str], dict[int, str]]:
    """(track -> team) pins, and (track -> flag) for the non-team answers.

    A track carrying contradictory labels after cutting is left UNPINNED rather than
    resolved by majority: it means the cut did not separate what the curator saw, and
    inventing an answer there would launder a known-bad label into a confident one.
    """
    votes: dict[int, set[str]] = defaultdict(set)
    flags: dict[int, str] = {}
    for r in resolved:
        if not r["ok"]:
            continue
        f = flag_of(r)
        if f == "unknown" and any(
                o["ok"] and o["tid"] == r["tid"] and o.get("team") and not flag_of(o)
                for o in resolved):
            # A segment holding BOTH a team label and an unknown is one the cut did
            # not fully separate. The team label is real evidence, so keep it and drop
            # the unknown rather than letting the weaker answer veto the stronger.
            continue
        if f:
            # mixed/notrobot are stronger statements than unknown, so they win: an
            # `unknown` never overwrites one, but either may overwrite an `unknown`.
            if f == "unknown" and flags.get(r["tid"]) in ("mixed", "notrobot"):
                continue
            flags[r["tid"]] = f
        elif r.get("team"):
            votes[r["tid"]].add(str(r["team"]))

    blocked = {t for t, f in flags.items() if f in ("mixed", "notrobot")}
    pins = {tid: next(iter(teams)) for tid, teams in votes.items()
            if len(teams) == 1 and tid not in blocked}
    return pins, flags


def dropped(flags: dict[int, str]) -> set[int]:
    """Tracks that must not be grouped into a robot. `unknown` is NOT one of them --
    the curator failing to identify a robot is not evidence it is absent."""
    return {t for t, f in flags.items() if f in ("mixed", "notrobot")}


CROSS_WIDTHS = 1.5      # "close" means centres within this many box widths


def crossing_cuts(pins: dict[int, str], rows, label_t: dict[int, list[float]],
                  max_widths: float = CROSS_WIDTHS) -> tuple[dict, list]:
    """Where two tracks pinned to one team physically CROSS, cut both at the crossing.

    A tracker does not swap identities at random; it swaps when two objects occlude or
    pass each other, because that is exactly when IoU matching and a Kalman prediction
    both become ambiguous. So a crossing is not merely suspicious, it is the specific
    hypothesis that explains a clash.

    If a curator names track A "6201" early and track B "6201" late, and A and B cross
    in between, the likely truth is that the tracker exchanged them at the crossing:
    A-before + B-after is one robot, B-before + A-after the other. Cutting BOTH at the
    crossing makes that expressible -- the two pins then land on pieces that no longer
    overlap in time, so the clash dissolves with no guess about which label to discard.

    MEASURED on the clashes this produced: 6 of 12 clashing pairs have such a crossing,
    several with centres closer than half a box width. The other 6 never come near each
    other and are genuine contradictions -- worth a human's attention, which is exactly
    the split this makes possible. Demoting all twelve treated both kinds the same.

    Only cuts when the crossing SEPARATES the two labels in time; a crossing before or
    after both of them does not explain anything.
    """
    pos: dict[int, dict] = defaultdict(dict)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                x1, y1, x2, y2 = d["xyxy"]
                pos[d["tid"]][round(r["t"], 3)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                                   x2 - x1)

    by_team: dict[str, list[int]] = defaultdict(list)
    for tid, team in pins.items():
        by_team[team].append(tid)

    cuts: dict[int, list[float]] = defaultdict(list)
    found = []
    for team, tids in by_team.items():
        for i, a in enumerate(sorted(tids)):
            for b in sorted(tids)[i + 1:]:
                shared = sorted(set(pos[a]) & set(pos[b]))
                if not shared:
                    continue
                close = []
                for t in shared:
                    ax, ay, aw = pos[a][t]
                    bx, by, bw = pos[b][t]
                    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < \
                            max_widths * (aw + bw) / 2.0:
                        close.append(t)
                if not close:
                    continue
                ta = label_t.get(a, [])
                tb = label_t.get(b, [])
                if not ta or not tb:
                    continue
                lo, hi = min(min(ta), min(tb)), max(max(ta), max(tb))
                mid = [t for t in close if lo < t < hi]
                if not mid:
                    continue
                tc = mid[len(mid) // 2]
                cuts[a].append(tc)
                cuts[b].append(tc)
                found.append({"team": team, "tracks": [a, b], "atS": round(tc, 2),
                              "closeFrames": len(close)})
    return {k: sorted(set(v)) for k, v in cuts.items()}, found


# Two boxes on ONE robot is a different problem from two robots, and the separation
# between them is not subtle. Measured across 20 curated 2026necmp1 matches, every pair
# of co-detected tracks the curator gave the SAME team sits 39-118 px apart -- 0.20 to
# 0.71 box widths, with box IoU up to 0.56. The codebase's own measurements of pairs
# that really are two robots: 218-1206 px in custody_conflicts, 471 px at IoU ~0 in
# split_conflicts. A wide empty gap in between.
#
# 1.0 box width is the same constant custody_conflicts already uses to make this exact
# distinction, kept deliberately rather than re-tuned to the observed 0.71 -- the point
# is that one threshold serves both places, not that it is fitted tightly here.
MERGE_SEP_WIDTHS = 1.0
MERGE_MIN_FRAMES = 2


def merge_duplicates(pins: dict[int, str], rows, sep_widths: float = MERGE_SEP_WIDTHS):
    """Fuse tracks the curator labelled the same team while both were on screen.

    WHY THIS RUNS BEFORE split_conflicts. That function assumes a same-team
    co-detection means the CURATOR WAS WRONG -- two robots on screen, only one can hold
    the label -- and demotes the weaker pin to a preference. For genuinely distant pairs
    that is right. For a robot the detector fired on twice, low and high, it throws away
    a correct answer and leaves the duplicate free to take a different team, which is
    strictly worse than either merging or ignoring it.
    
    The curator is the authority here, not the geometry. solve.py's hard constraint says
    "co-detected tracks are different robots. This is geometry, not inference" -- true of
    two boxes on two robots, false of two boxes on one. A human looking at both crops and
    calling them the same robot is evidence the constraint's PREMISE failed, so the fix
    is to remove the second box, not to overrule the human.

    Returns (rows, merges). Detections of the absorbed track are re-tagged to the keeper;
    where both have one in the same frame the keeper's is kept and the other dropped, so
    the merged track never carries two boxes in a frame and the solver's constraint is
    satisfied by construction rather than suspended.
    """
    import numpy as np
    from collections import defaultdict as _dd

    close: dict[tuple[int, int], list] = _dd(list)
    for r in rows:
        here = [d for d in r["dets"] if d["tid"] in pins]
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                if pins[a["tid"]] != pins[b["tid"]]:
                    continue
                ax1, ay1, ax2, ay2 = a["xyxy"]
                bx1, by1, bx2, by2 = b["xyxy"]
                w = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
                d = float(np.hypot((ax1 + ax2) / 2 - (bx1 + bx2) / 2,
                                   (ay1 + ay2) / 2 - (by1 + by2) / 2))
                key = (min(a["tid"], b["tid"]), max(a["tid"], b["tid"]))
                close[key].append(d / w)

    merges = []
    absorb: dict[int, int] = {}
    for (a, b), seps in close.items():
        if len(seps) < MERGE_MIN_FRAMES:
            continue
        med = float(np.median(seps))
        if med > sep_widths:
            continue            # genuinely two robots -- split_conflicts handles it
        # Keep the track with more detections; it carries more of the robot's timeline.
        na = sum(1 for r in rows for d in r["dets"] if d["tid"] == a)
        nb = sum(1 for r in rows for d in r["dets"] if d["tid"] == b)
        keep, gone = (a, b) if na >= nb else (b, a)
        while keep in absorb:          # chains: c->b->a collapses to a
            keep = absorb[keep]
        if gone == keep:
            continue
        absorb[gone] = keep
        merges.append({"team": pins[a], "keep": keep, "absorbed": gone,
                       "frames": len(seps), "sepWidths": round(med, 2),
                       "dets": max(na, nb) + min(na, nb)})
    if not absorb:
        return rows, []

    out = []
    for r in rows:
        dets, seen = [], set()
        for d in r["dets"]:
            tid = d["tid"]
            while tid in absorb:
                tid = absorb[tid]
            if tid in seen and tid != d["tid"]:
                continue        # the keeper already has a box here; drop the duplicate
            if tid in seen:
                continue
            e = dict(d); e["tid"] = tid
            seen.add(tid)
            dets.append(e)
        out.append({**r, "dets": dets})
    return out, merges


def split_conflicts(pins: dict[int, str], rows, ident: dict
                    ) -> tuple[dict[int, str], dict[int, str], list[dict]]:
    """Separate pins a solver can satisfy from pins that contradict each other.

    A curator working from 8 crops of a bumper cannot see that two of the robots they
    labelled are on screen at the same moment. On the first real pass, 12 pairs of
    pinned tracks were co-detected while pinned to the SAME team -- one of them in 231
    frames together, with box IoU ~0 and centres 471 px apart, so they are plainly two
    different robots. At most one of each pair can carry that team.

    Hard-constraining all of them makes the model INFEASIBLE, which is the least useful
    thing we could tell anyone. Instead: keep the best-supported pin of each clashing
    set as a hard constraint, demote the rest to strong PREFERENCES the solver may
    overrule, and hand back the clash list.

    "Best supported" is the track whose own bumper reads agree with the label, then
    size. A curator's guess backed by OCR beats the same guess unbacked, and both beat
    a 40-detection fragment.

    The returned clashes are worth more than the repair: they are exactly the questions
    the curator got wrong, found for free, and they are what the next pass should ask
    about first.
    """
    co: dict[tuple[int, int], int] = {}
    for r in rows:
        ts = sorted(d["tid"] for d in r["dets"] if d["tid"] in pins)
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                if pins[a] == pins[b]:
                    co[(a, b)] = co.get((a, b), 0) + 1
    if not co:
        return dict(pins), {}, []

    # Union clashing tracks into per-team clusters.
    parent = {t: t for t in pins}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in co:
        parent[find(a)] = find(b)
    clusters: dict[int, list[int]] = {}
    for t in pins:
        clusters.setdefault(find(t), []).append(t)

    ndets: dict[int, int] = {}
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                ndets[d["tid"]] = ndets.get(d["tid"], 0) + 1

    def support(tid: int) -> tuple[int, int]:
        tally = ident["tracks"].get(str(tid), {}).get("tally", {})
        return int(tally.get(pins[tid], 0)), ndets.get(tid, 0)

    hard, soft, clashes = {}, {}, []
    for members in clusters.values():
        if len(members) == 1:
            hard[members[0]] = pins[members[0]]
            continue
        members.sort(key=support, reverse=True)
        keep = members[0]
        hard[keep] = pins[keep]
        for m in members[1:]:
            soft[m] = pins[m]
        clashes.append({
            "team": pins[keep], "kept": keep, "demoted": members[1:],
            "frames": max((c for (a, b), c in co.items()
                           if a in members and b in members), default=0),
            "support": {str(m): support(m) for m in members},
        })
    return hard, soft, clashes


def report(resolved: list[dict], pins: dict[int, str], flags: dict[int, str],
           n_cuts: int) -> None:
    good = [r for r in resolved if r["ok"]]
    bad = [r for r in resolved if not r["ok"]]
    print(f"[corrections] {len(good)}/{len(resolved)} labels resolved to a detection")
    if n_cuts:
        print(f"[corrections] {n_cuts} cut(s) induced where a curator's labels "
              f"disagreed inside one track")
    tally = Counter(flags.values())
    extra = "".join(f", {n} {k}" for k, n in sorted(tally.items()))
    print(f"[corrections] {len(pins)} track(s) pinned{extra}")
    for r in bad:
        print(f"[corrections]   UNRESOLVED f{r['f']} at {r['xy']} "
              f"(nearest detection {r['dist']:.0f} px away)"
              if r["dist"] is not None else
              f"[corrections]   UNRESOLVED f{r['f']} at {r['xy']} "
              f"(no tracked detection in that frame)")


# ---- timing ------------------------------------------------------------------
# "Curation takes about five minutes a match" was the working figure for every
# decision about frame counts, pre-fill and whether scouts could run this. It was a
# recollection. The curator now ships a `session` block (see TIMER in curate.html) and
# every label carries the `ms` it took, so the question has an answer.
#
# TWO CLOCKS, DELIBERATELY. `session.activeSec` is wall time minus long gaps -- what a
# curator would call "how long that took me". Summed `ms` is time with a box actually
# selected. The gap between them is the overhead: reading the frame, hunting the next
# box, waiting for a crop. Reporting only one of them would hide whichever half is
# worth fixing.

def timing_rows(event: str | None = None, root: Path | None = None) -> list[dict]:
    """One row per curated match that carries timing, newest schema only."""
    d = (root or (C.REPO_ROOT / "robot-tracker" / "corrections"))
    out = []
    for p in sorted(d.glob("*_corrections.json")):
        stem = p.name[: -len("_corrections.json")]
        if event and not stem.startswith(event):
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        labels = doc.get("labels") or []
        human = [l for l in labels if l.get("src") == "human"]
        ms = [l["ms"] for l in human if isinstance(l.get("ms"), (int, float))]
        sess = doc.get("session") or {}
        out.append({
            "match": stem,
            "labels": len(labels),
            "human": len(human),
            "activeSec": sess.get("activeSec"),
            "wallSec": sess.get("wallSec"),
            "onBoxSec": round(sum(ms) / 1000.0, 1) if ms else None,
            "msN": len(ms),
        })
    return out


def timing_report(event: str | None = None, root: Path | None = None) -> int:
    rows = timing_rows(event, root)
    if not rows:
        print("[timing] no corrections files found")
        return 1
    print(f"{'match':<26}{'labels':>7}{'yours':>7}{'active':>9}{'wall':>8}"
          f"{'on-box':>8}{'s/label':>9}")
    timed = []
    for r in rows:
        per = (r["activeSec"] / r["human"]) if r["activeSec"] and r["human"] else None
        if r["activeSec"]:
            timed.append(r)
        def f(v, suf=""):
            return "-" if v is None else f"{v:.0f}{suf}" if isinstance(v, float) else f"{v}{suf}"
        print(f"{r['match']:<26}{r['labels']:>7}{r['human']:>7}"
              f"{f(r['activeSec']):>9}{f(r['wallSec']):>8}{f(r['onBoxSec']):>8}"
              f"{('-' if per is None else f'{per:.1f}'):>9}")
    if not timed:
        print("\n[timing] no match carries a session block yet -- that field is written "
              "by curate.html from this version on, so only matches curated from now "
              "will appear. Per-label `ms` above is the older instrumentation.")
        return 0
    tot_a = sum(r["activeSec"] for r in timed)
    tot_h = sum(r["human"] for r in timed)
    tot_b = sum(r["onBoxSec"] or 0 for r in timed)
    print(f"\n[timing] {len(timed)} timed match(es): "
          f"{tot_a / len(timed) / 60:.1f} min/match average, "
          f"{tot_a / max(tot_h, 1):.1f} s/label")
    if tot_b:
        print(f"[timing] {100 * tot_b / tot_a:.0f}% of active time had a box selected; "
              f"the rest is reading the frame and moving between boxes")
    return 0




# ---- auto-ID agreement -------------------------------------------------------
# The question this answers is item 125 of the feedback list: "automatic ID still
# appears pretty suspect, need rigorous testing based on a larger set of curated
# matches." Pre-fill percentage does NOT answer it -- a bundle can arrive 98% pre-filled
# and be 98% wrong. What matters is how often the curator changes the guess.
#
# `was` is written by curate.html beside every human answer (see the note there). Labels
# without it predate that change and are reported separately rather than assumed to
# agree, because assuming agreement is exactly the bias this report exists to avoid.

def agreement_rows(event: str | None = None, root: Path | None = None) -> list[dict]:
    d = (root or (C.REPO_ROOT / "robot-tracker" / "corrections"))
    out = []
    for p in sorted(d.glob("*_corrections.json")):
        stem = p.name[: -len("_corrections.json")]
        if event and not stem.startswith(event):
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        human = [l for l in (doc.get("labels") or []) if l.get("src") == "human"]
        scored = [l for l in human if "was" in l]
        # A guess the curator kept, against one they replaced. Flag answers
        # (notrobot/unknown/mixed) count as a change when the machine named a team:
        # the machine asserting "this is 1768" where a human says "not a robot" is a
        # disagreement, and a bad one.
        def named(l):
            return l.get("team")
        agree = sum(1 for l in scored
                    if l.get("was") and named(l) and str(named(l)) == str(l["was"]))
        had_guess = sum(1 for l in scored if l.get("was"))
        no_guess = len(scored) - had_guess
        out.append({
            "match": stem, "human": len(human), "scored": len(scored),
            "legacy": len(human) - len(scored),
            "hadGuess": had_guess, "agree": agree,
            "pct": (round(100 * agree / had_guess, 1) if had_guess else None),
            "noGuess": no_guess,
        })
    return out


def agreement_report(event: str | None = None, root: Path | None = None) -> int:
    rows = agreement_rows(event, root)
    if not rows:
        print("[agreement] no corrections files found")
        return 1
    print(f"{'match':<26}{'yours':>7}{'scored':>8}{'guessed':>9}{'kept':>7}{'agree':>8}")
    for r in rows:
        pct = "-" if r["pct"] is None else f"{r['pct']:.0f}%"
        print(f"{r['match']:<26}{r['human']:>7}{r['scored']:>8}"
              f"{r['hadGuess']:>9}{r['agree']:>7}{pct:>8}")
    tg = sum(r["hadGuess"] for r in rows)
    ta = sum(r["agree"] for r in rows)
    legacy = sum(r["legacy"] for r in rows)
    if tg:
        print(f"\n[agreement] {ta}/{tg} guesses kept ({100 * ta / tg:.1f}%) across "
              f"{sum(1 for r in rows if r['hadGuess'])} match(es)")
    if legacy:
        print(f"[agreement] {legacy} label(s) carry no `was` field -- curated before it "
              f"was recorded, so their agreement is NOT measurable and is excluded "
              f"rather than counted as agreement")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Curation corrections: reports.")
    ap.add_argument("--timing", action="store_true",
                    help="minutes per match and seconds per label, from the session "
                         "blocks curators ship with their answers")
    ap.add_argument("--agreement", action="store_true",
                    help="how often the curator KEPT the machine's guess -- the only "
                         "honest measure of whether the auto-ID can be trusted")
    ap.add_argument("--event", default=None, help="limit to one event key prefix")
    args = ap.parse_args(argv)
    if args.timing:
        return timing_report(args.event)
    if args.agreement:
        return agreement_report(args.event)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
