"""Stage 3 -- resolve fragmented tracks into 6 identified robots.

Replaces the greedy pairwise stitch + per-track assign with one global step, because
measurement showed correlation is not an optimisation on top of identification -- it
is the PRECONDITION for it. There are 172 confident bumper reads in the test match,
ample to name six robots, but scattered across 26 tallies with a median of 5, which
is below any usable threshold. Pooled into 6 they are ~28 each, and every track that
reached double digits produced a clean answer.

Three ideas do the work:

1. CO-DETECTION IS A HARD CONSTRAINT. Two tracks seen in the same frame cannot be the
   same robot. That makes this interval-graph colouring rather than pairwise merging,
   and interval graphs are perfect -- greedy by start time uses the minimum number of
   colours. The previous greedy merge could not back out of an early mistake; a
   colouring chooses among all available groups at each step.

2. ONCE THERE ARE 6 GROUPS, NAMING THEM IS A 6x6 ASSIGNMENT. Pool each group's votes,
   build a cost matrix against the six teams TBA gives us, and solve it optimally with
   the alliance split enforced. No thresholds, no greedy first-come-first-served.

3. MERGE, THEN VERIFY. Aggressive grouping risks chimeras -- one track already mixes
   5687 and 1768. Pooled votes make that visible: a group whose votes disagree across
   time windows is flagged, not silently averaged into a confident wrong answer.

    uv run -m rtrack.robots GSxbsE42o5o --tracks out/stage1/MATCH4_st.jsonl \\
        --match 2026necmp_f1m3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config as C
from .acquire import video_id
from . import tba as tba_mod

N_ROBOTS = 6
ALLIANCE_MISMATCH = 1e6     # effectively forbidden, but still orderable
VOTE_DISAGREE = 60.0        # cost of joining a group whose votes say otherwise
GAP_PENALTY = 2.0           # per second unobserved, mild
MIN_CONFLICT_DETS = 5       # a custody conflict needs real presence on BOTH sides
# Two detections this far apart in time count as "at the same moment". Wider than one
# frame because tracks alternate; much wider and a moving robot's own travel swamps
# the separation being measured.
MATCH_DT_S = 0.25


def _local_speed(pts) -> float:
    """px/s from consecutive samples of one track; 0 when it cannot be measured."""
    q = sorted(pts)
    v = [np.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])
         for a, b in zip(q, q[1:]) if 0 < b[0] - a[0] < 0.5]
    return float(np.median(v)) if v else 0.0


def load_tracks(p: Path):
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])
    return rows


def conflicts(rows) -> dict[int, set[int]]:
    """Tracks co-detected in any frame. The hard 'cannot be the same robot' relation."""
    con: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        tids = [d["tid"] for d in r["dets"] if d["tid"] >= 0]
        for i, a in enumerate(tids):
            for b in tids[i + 1:]:
                con[a].add(b)
                con[b].add(a)
    return con


def positions_by_det(positions: dict | None, tracks_p) -> dict | None:
    """Re-key positions.json from ITS track ids onto (frame, box), which survives splits.

    rtrack.project runs on the STITCHED tracks, but the solver runs on the SPLIT ones,
    and splitting keeps the original id for the first fragment while minting new ids for
    the rest. Joining those two id spaces by number silently pairs a fragment with the
    whole track it came from. Measured on 2026necmp1_qm24: positions held 23 tids, the
    solver 172, and of the 23 that matched numerically 22 spanned a DIFFERENT time range
    -- solver track 1 ends at 20.9 s while positions track 1 ends at 108.9 s, most of a
    match later and usually elsewhere on the field.

    The consequence was not a missing penalty but a wrong one: _pair_cost compared the
    end of a whole stitched track against the start of some fragment, so it both missed
    real teleports and could invent penalties between tracks that are actually adjacent.
    87% of solver tracks had no position at all and scored kin = 0 regardless.

    Splitting never touches `xyxy`, so (frame, box) is stable across it -- the same
    anchoring rtrack.corrections uses, for the same reason. kinematic_conflicts already
    guards this hazard by refusing to run on a stale id space; this fixes it instead.
    """
    if not positions:
        return None
    box_of = {}
    for line in Path(tracks_p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for d in r["dets"]:
            if d["tid"] >= 0:
                box_of[(r["f"], d["tid"])] = tuple(round(float(v), 1) for v in d["xyxy"])
    out = {}
    for sm in positions.get("samples", ()):
        if sm["tid"] < 0 or "offfield" in sm.get("flags", ()):
            continue
        b = box_of.get((sm["f"], sm["tid"]))
        if b:
            out[(sm["f"], b)] = (sm["t"], sm["x"], sm["y"])
    return out


JOIN_GAP_MIN_S = 0.30   # a pause this long is a stitch join, not a dropped frame
JOIN_DIFF_COS = 0.89    # solve.APP_DIFF: different-robot p10 over curated pairs
JOIN_SIDE_S = 6.0       # how much of each side to describe


def split_chimeric_joins(rows, npz_path, head=None, thresh: float = JOIN_DIFF_COS):
    """Cut a track back open where STITCH joined two different robots.

    WHY HERE AND NOT IN STITCH. Stitch runs before rtrack.appear, so when it decides
    whether a reappearance continues a track it has geometry and nothing else. Measured
    on 2026necmp1_qm24 it joined across a 3.73 s gap and 678 px -- legal against its
    5.5 m/s bound -- and the embeddings either side sit 0.964 apart, as different as two
    robots get. 9 of 23 stitched tracks on that match carry a jump over 300 px. The
    resulting chimera then propagates: split_on_appearance cut it back apart, and the
    solver reassembled it by putting team 195 on both halves.

    WHY THIS THRESHOLD IS VALID HERE AND WAS NOT IN THE OBJECTIVE. 0.89 is the
    different-robot p10 measured over 1662 curated WHOLE-TRACK pairs. The two sides of
    a stitch join are long segments, so that calibration transfers. An earlier attempt
    applied the same number between adjacent FRAGMENTS and cost 10-12 accuracy points,
    because split_on_appearance cuts precisely where appearance changes most -- its
    cut-point pairs sit at a median of 0.81 by construction, so the test fired on
    legitimate continuations. Same number, wrong population.

    Only INTERNAL gaps are examined -- a pause inside one track, which is where stitch
    made a decision. Continuous stretches are the tracker's work and are left alone.
    """
    import numpy as _np
    if not Path(npz_path).exists():
        return rows, 0
    z = _np.load(npz_path)
    tid_a, t_a, feat = z["tid"], z["t"], z["feat"]
    if head is not None:
        feat = head(feat)

    per = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                per[d["tid"]].append(r["t"])
    cuts = defaultdict(list)
    for tid, ts in per.items():
        ts.sort()
        if len(ts) < 10:
            continue
        step = float(_np.median(_np.diff(ts))) if len(ts) > 2 else 0.07
        for a, b in zip(ts, ts[1:]):
            gap = b - a
            if gap < max(JOIN_GAP_MIN_S, 3 * step):
                continue
            m0 = (tid_a == tid) & (t_a > a - JOIN_SIDE_S) & (t_a <= a + 1e-6)
            m1 = (tid_a == tid) & (t_a >= b - 1e-6) & (t_a < b + JOIN_SIDE_S)
            if int(m0.sum()) < 4 or int(m1.sum()) < 4:
                continue
            v0, v1 = feat[m0].mean(0), feat[m1].mean(0)
            n0 = float(_np.linalg.norm(v0)) or 1.0
            n1 = float(_np.linalg.norm(v1)) or 1.0
            d = 1.0 - float((v0 / n0) @ (v1 / n1))
            if d >= thresh:
                cuts[tid].append((b, round(d, 3), round(gap, 2)))
    if not cuts:
        return rows, 0
    nxt = max((d["tid"] for r in rows for d in r["dets"]), default=0) + 1
    remap = {}
    for tid, cl in cuts.items():
        for b, _d, _g in sorted(cl):
            remap.setdefault(tid, []).append((b, nxt))
            nxt += 1
    out, n = [], 0
    for r in rows:
        dets = []
        for d in r["dets"]:
            t = d["tid"]
            if t in remap:
                for b, new in remap[t]:
                    if r["t"] >= b:
                        t = new
            dets.append(dict(d, tid=t))
        out.append({**r, "dets": dets})
    n = sum(len(v) for v in cuts.values())
    for tid, cl in sorted(cuts.items()):
        for b, d, g in cl:
            print(f"[robots]   #{tid} cut at t{b:.2f} -- {g}s pause, appearance "
                  f"{d} across it (different robots above {thresh})")
    return out, n


def fragment_embeddings(pre_split_rows, rows, npz_path, head=None):
    """One whitened, unit-length embedding per FRAGMENT.

    The cached descriptors are keyed by the STITCHED track id, but the solver works on
    fragments, so a fragment's crops are found by mapping each of its detections back
    through (frame, box) -- the one key that survives every split -- and then taking
    the cached rows for that stitched id inside the fragment's own time span.
    """
    import numpy as _np
    if not Path(npz_path).exists():
        return {}
    z = _np.load(npz_path)
    tid_a, t_a, feat = z["tid"], z["t"], z["feat"]
    if head is not None:
        feat = head(feat)
    parent = {}
    for r in pre_split_rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                parent[(r["f"], tuple(round(float(v), 1) for v in d["xyxy"]))] = d["tid"]
    span = defaultdict(lambda: [1e9, -1e9, None])
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            p = parent.get((r["f"], tuple(round(float(v), 1) for v in d["xyxy"])))
            if p is None:
                continue
            a = span[d["tid"]]
            a[0] = min(a[0], r["t"]); a[1] = max(a[1], r["t"]); a[2] = p
    out = {}
    for frag, (t0, t1, p) in span.items():
        if p is None:
            continue
        m = (tid_a == p) & (t_a >= t0 - 1e-6) & (t_a <= t1 + 1e-6)
        if int(m.sum()) < 4:
            continue
        v = feat[m].mean(0)
        n = float(_np.linalg.norm(v))
        if n > 0:
            out[frag] = v / n
    return out


HOLD_S = 20.0          # cap when nothing is ever seen to come back out
HOLD_TOL_PX = 40.0     # how close a death must be to count as INTO the structure


def hold_cells(st_rows, rows, regions, tol_px: float = HOLD_TOL_PX,
               max_hold_s: float = HOLD_S, require_exit: bool = False):
    """Occluders as HOLDING CELLS: a team that vanished into one is in it, not loose.

    THE MODEL. When a robot disappears into a marked structure it is still on the field
    -- it is behind that structure until something comes back out. So a structure holds
    a team for an interval, and two things follow. The team cannot also be a robot
    visible somewhere else meanwhile (exclusion), and the track that emerges is probably
    the team being held (emergence). The first is what pays: one disappearance then
    constrains every other track on the field, so a vanishing robot stops destabilising
    everyone else's labels.

    EVENTS COME FROM THE STITCHED TRACKS, NOT THE FRAGMENTS. A first version keyed on
    fragment deaths and was a clear regression at every weight -- custody 78% -> 69%,
    unlabelled detections 180 -> 1840. The reason is that a fragment ending almost never
    means a robot disappeared: 95 of ~128 fragment boundaries are split_on_appearance
    cuts with the robot still plainly visible, and measured on 2026necmp1_qm24 exactly
    ONE fragment death of 134 was inside a structure, the median being 193 px away. A
    stitched track ending IS a disappearance -- stitch has already rejoined everything
    it could. Measured at that level: 4 hold events on qm24, 13 on qm22, 10 on qm21.

    THE INTERVAL ENDS AT THE ACTUAL EMERGENCE, not a fixed window, so a robot that
    ducks behind and comes straight out frees its team immediately instead of staying
    excluded for the rest of a timeout.

    Returns (exclude, emerge) as fragment-id pairs, ready for the solver.
    """
    from .occluders import region_at
    if not regions:
        return [], []

    # (frame, box) survives every split, so it is how a stitched detection is found
    # again among the fragments. Track ids do not survive -- the splits rebuild them.
    frag_of = {}
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                frag_of[(r["f"], tuple(round(float(v), 1) for v in d["xyxy"]))] = d["tid"]

    st = defaultdict(list)
    for r in st_rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                x1, y1, x2, y2 = d["xyxy"]
                st[d["tid"]].append((r["t"], r["f"],
                                     tuple(round(float(v), 1) for v in d["xyxy"]),
                                     (x1 + x2) / 2.0, y2))
    for v in st.values():
        v.sort()
    if not st:
        return [], []
    t_lo = min(v[0][0] for v in st.values())
    t_hi = max(v[-1][0] for v in st.values())

    # Who vanished into what, and what came back out of it.
    deaths, births = [], []
    for tid, v in st.items():
        t, f, bx, x, y = v[-1]
        if t < t_hi - 2:
            S = region_at(x, y, regions, tol_px)
            if S:
                deaths.append((t, S, frag_of.get((f, bx))))
        t, f, bx, x, y = v[0]
        if t > t_lo + 2:
            S = region_at(x, y, regions, tol_px)
            if S:
                births.append((t, S, frag_of.get((f, bx))))
    births.sort()

    # Fragment spans and where each sat, for the exclusion test.
    fr = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                x1, y1, x2, y2 = d["xyxy"]
                fr[d["tid"]].append((r["t"], (x1 + x2) / 2.0, y2))
    for v in fr.values():
        v.sort()

    # How many robots are visibly tracked at each sampled instant, from the PRE-SPLIT
    # state so a split does not read as an extra robot.
    times = sorted({t for v in st.values() for t, _f, _b, _x, _y in v})
    live = {}
    for t in times:
        live[t] = sum(1 for v in st.values() if v[0][0] <= t <= v[-1][0])

    def release_at(t_start):
        """When conservation says this cell must let go.

        There are six robots. A team behind a structure is one of them, so
        `live + holds` cannot exceed six -- and when it does, the robot we thought was
        hidden is evidently back in view. This releases the hold instead of asserting
        it for a fixed 20 s, which is what made the ungated version harmful: measured
        on qm24, 7 of 11 vanished robots reappeared ELSEWHERE within seconds, and the
        cell kept excluding their team anyway.

        A RULE, NOT A CONSTRAINT, deliberately. Duplicate boxes still survive the merge
        on ~1% of detections, and >6 tracks are concurrent in 1-3% of frames, so a hard
        count would be violated on real data. Overcounting here only RELEASES a hold
        early -- less exclusion, never a false assertion -- so the error direction is
        'do nothing'.
        """
        for t in times:
            if t <= t_start:
                continue
            if live.get(t, 0) >= 6:
                return t
        return None

    exclude, emerge, cells = [], [], []
    for t1, S, tail in sorted(deaths):
        if tail is None:
            continue
        nxt = next((b for b in births if b[1] == S and b[0] > t1), None)
        t2 = min(nxt[0], t1 + max_hold_s) if nxt else t1 + max_hold_s
        rel = release_at(t1)
        if rel is not None and rel < t2:
            t2 = rel
        head = nxt[2] if nxt and nxt[0] <= t2 else None
        cells.append((S, round(t1, 1), round(t2, 1), tail, head))
        if head is not None and head != tail:
            emerge.append((tail, head))
        if require_exit and head is None:
            # Nothing was ever seen to come back out. Either the robot is still behind
            # the structure, or it emerged undetected, or -- the case that costs -- it
            # never went in. A cell with no observed exit is an assertion with no
            # evidence closing it, so under this flag it holds nothing.
            continue
        for c, v in fr.items():
            if c in (tail, head):
                continue
            inwin = [(t, x, y) for t, x, y in v if t1 < t < t2]
            if not inwin:
                continue
            # A fragment that was itself at the structure during the hold might BE the
            # robot coming out; one anywhere else cannot be the team being held.
            if any(region_at(x, y, regions, tol_px) == S for _t, x, y in inwin):
                continue
            exclude.append((tail, c))
    return exclude, emerge, cells


REBIND_MAX_COS = 0.70
"""Whitened-embedding cosine distance a rebind may not exceed.

CALIBRATED, not guessed -- whitening rescales distances, so an intuited threshold means
nothing. Measured over 20 curated 2026necmp1 matches, track-mean distances in the
stitched id space this runs on:

                               p10     p25   median     p75     p90
                 same team    0.25    0.31    0.38    0.45    0.53
  diff team, same alliance    0.89    0.94    1.00    1.06    1.12

Same-team p90 is 0.53 and different-team p10 is 0.89, with nothing in between:

    threshold   same kept   different admitted
      0.55         92%            0%
      0.70        100%            0%     <- here
      0.80        100%            1%
      0.90        100%           13%

0.55 was the first guess and threw away 8% of genuine matches for nothing. Past 0.80
impostors start arriving, and an impostor here is a track spanning two robots -- the
failure this whole path exists to avoid."""

REBIND_MARGIN = 1.25    # best must beat the runner-up by this factor


def occluder_rebind(rows, regions, npz_path, head=None, ident=None,
                    max_cos: float = REBIND_MAX_COS,
                    margin: float = REBIND_MARGIN):
    """Rejoin tracks across a structure a human marked, using APPEARANCE to choose.

    WHY THIS LIVES HERE AND NOT IN rtrack.stitch. Stitch has the geometry to find these
    -- measured on 2026necmp1_qm24, 33 of the 72 handoffs it refuses are hub -> the same
    hub -- but not the evidence to resolve them. Several tracks die behind one structure
    over a match, so when a new one appears there, two or three predecessors are equally
    plausible by position and stitch's uniqueness gate correctly refuses to guess:

        AMBIGUOUS #116: candidates ['#115(280px)', '#82(30px)', '#10(50px)']

    Enabling the occluder path in stitch therefore made things WORSE -- 35 tracks against
    32 -- because extra candidates trip that gate. The pipeline runs track -> stitch ->
    appear, so no descriptor exists yet at stitch time. Here one does, and the choice
    between #82 and #10 is exactly what it is good at: 0.788 AUC within an alliance,
    against a 2-3 way choice.

    Runs BEFORE any split, for the same reason the duplicate merge does: after 95
    appearance splits a track's evidence is shattered into fragments too short to
    describe, and the tids no longer match the cached npz.

    Refuses rather than guesses when appearance is not decisive either -- an unmerged
    track is an honest gap, a wrong merge silently rewrites a robot's history.
    """
    import numpy as _np
    from .occluders import region_at, transit_budget_s, parked_at, PARKED_MAX_S
    from .stitch import PX_PER_M
    if not regions or not Path(npz_path).exists():
        return rows, []

    z = _np.load(npz_path)
    tid_a, feat = z["tid"], z["feat"]
    if head is not None:
        feat = head(feat)
    mean, span, ends = {}, {}, {}
    for tid in sorted(set(tid_a.tolist())):
        m = tid_a == tid
        if int(m.sum()) < 4:
            continue
        v = feat[m].mean(0)
        n = float(_np.linalg.norm(v)) or 1.0
        mean[int(tid)] = v / n
    pts = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                x1, y1, x2, y2 = d["xyxy"]
                pts[d["tid"]].append((r["t"], (x1 + x2) / 2.0, y2))
    for tid, v in pts.items():
        v.sort()
        span[tid] = (v[0][0], v[-1][0])
        ends[tid] = ((v[0][1], v[0][2]), (v[-1][1], v[-1][2]))

    order = sorted(span, key=lambda t: span[t][0])
    absorb, merges = {}, []
    for b in order:
        if b not in mean:
            continue
        b_start, (t0b, _t1b) = ends[b][0], span[b]
        rb = region_at(b_start[0], b_start[1], regions)
        if rb is None:
            continue
        cands = []
        for a in order:
            if a == b or a not in mean or a in absorb:
                continue
            t0a, t1a = span[a]
            if t1a >= t0b:
                continue                     # coexisting -> different robots
            gap_s = t0b - t1a
            if gap_s > PARKED_MAX_S:
                continue
            a_end = ends[a][1]
            if region_at(a_end[0], a_end[1], regions) != rb:
                continue
            budget = transit_budget_s(regions, rb,
                                      C.ROBOT_MAX_SPEED_MS * PX_PER_M)
            if gap_s > budget and not parked_at(a_end, b_start, rb, regions, gap_s):
                continue
            cands.append((float(1.0 - mean[a] @ mean[b]), a, gap_s))
        if not cands:
            continue
        cands.sort()
        best_d, best, gap_s = cands[0]
        if best_d > max_cos:
            continue                         # looks like a different robot
        if len(cands) > 1 and cands[1][0] < best_d * margin:
            continue                         # appearance cannot separate them either
        root = best
        while root in absorb:
            root = absorb[root]
        if root == b:
            continue
        absorb[b] = root
        merges.append({"keep": root, "absorbed": b, "region": rb,
                       "gapS": round(gap_s, 1), "cos": round(best_d, 3),
                       "runnerUp": round(cands[1][0], 3) if len(cands) > 1 else None})
    if not absorb:
        return rows, []
    out = []
    for r in rows:
        dets = []
        for d in r["dets"]:
            t = d["tid"]
            while t in absorb:
                t = absorb[t]
            dets.append(dict(d, tid=t))
        out.append({**r, "dets": dets})
    if ident is not None:
        tks = ident.setdefault("tracks", {})
        for mg in merges:
            src = tks.pop(str(mg["absorbed"]), None)
            if not src:
                continue
            dst = tks.setdefault(str(mg["keep"]), {"tally": {}, "voteList": []})
            tal = dst.setdefault("tally", {})
            for team, n in (src.get("tally") or {}).items():
                tal[team] = tal.get(team, 0) + n
            dst.setdefault("voteList", []).extend(src.get("voteList") or [])
    return out, merges


DUP_SEP_M = 0.8
"""Metres between floor-contact points. NO LONGER the criterion -- kept only as a
corroborating signal, because the projection cannot be trusted for this question.

The homography assumes every box bottom sits on the GROUND. A box drawn around a
robot's superstructure has its bottom edge partway up the robot, so it projects away
along the camera ray and the reported separation is mostly an artefact of height.
Measured against this camera's calibration, from a vertical offset alone, same image x:

    box bottom at y=   dy=20px   40px   60px   80px
              500 px     0.49m  1.03m  1.60m  2.23m
              800 px     0.21m  0.42m  0.65m  0.88m
             1050 px     0.13m  0.27m  0.41m  0.56m

A 40 px offset at the far end already exceeds this threshold, and the error is
POSITION-DEPENDENT -- far-side duplicates read as conflicts while near-side ones do
not, a bias that looks deceptively like a real spatial pattern. On qm24 it disagreed
with the horizontal test on 4 of 12 co-detected pairs, every one of them a ~100 px
vertical offset in the same image column: a robot split low/high."""

DUP_DX_WIDTHS = 0.6
"""Horizontal separation in BOX WIDTHS -- the real criterion. Two boxes on one robot
share an image column and differ in height, so the artefact above lives entirely in
the projection and not in the pixels. Normalising by box width absorbs perspective,
since a distant robot's box shrinks in proportion."""

DUP_V_OVERLAP = 0.25
"""...and the boxes must overlap VERTICALLY, which is what a low/high split looks like.
Without this, one robot directly behind another -- same column, large dy, no overlap --
would merge. Measured on qm24, genuine pairs scored -0.67 and -0.45 (no shared rows)
while duplicates scored 0.30 to 1.00.

Appearance was tried as the discriminator here and does NOT work: a low/high split
crops the robot's TOP in one box and its MIDDLE in the other, so 6 of 9 known
duplicates read as different robots (median 0.753 against a same-robot p90 of 0.53).
The descriptor is answering correctly; the question was wrong for it."""

DUP_IOU = 0.35
"""Median box overlap -- a GUARD against a bad projection, not the discriminator.

Set to 0.15 first, which vetoed qm24's (19,20): 0.64 m apart over 118 frames, a
sustained physical impossibility rejected because its overlap missed the bar by 0.03.
The measured distribution leaves plenty of room -- pairs at 0.9-1.5 m sit at IoU 0.01
and everything beyond at 0.00 -- so a low threshold still excludes every genuine pair
while letting separation do the work it is actually qualified to do.

RAISED TO 0.35 after 2026mawor. That reasoning held while the OTHER criteria did the
excluding, and on a second camera they stopped: qm6 merged four pairs at IoU 0.09-0.29,
two of them co-detected for 137 and 140 frames -- nine seconds of a pair sitting
0.37 box widths apart. That is two robots driving alongside each other, not one robot
boxed twice, and merging them destroys a team. Genuine low/high duplicates are not
marginal on overlap: necmp1's sit at 0.27-0.53.

Measured, curator alignment at each threshold:

    match              0.05        0.20        0.35
    2026mawor_qm2      97%         --          97%
    2026mawor_qm6      94%         95%         97%
    2026mawor_qm9      94%         --          96%
    2026mawor_qm10     94%         --          94%
    2026necmp1_qm21    98%         97%         98%   (103 agree vs 101)

No match is worse and two are materially better. The old value was safe only because
necmp1's geometry never produced a sustained low-IoU pair to be wrong about."""

DUP_MIN_FRAMES = 5
"""Was 8, which missed qm24's (21,23): 0.60 m at IoU 0.38 over 5 frames. A brief
duplicate is still a duplicate, and the separation bound does not weaken with duration."""


def geometric_duplicates(rows, pos_at, sep_m: float = DUP_SEP_M,
                         min_iou: float = DUP_IOU, min_frames: int = DUP_MIN_FRAMES):
    """Fuse tracks that are ONE robot the detector boxed twice, using geometry alone.

    corrections.merge_duplicates does this from curator pins, and states why the curator
    is the authority: solve.py's hard constraint asserts that co-detected tracks are
    different robots, and a human calling both crops the same robot is evidence that
    PREMISE failed. That gate means it never runs on an uncurated match -- exactly where
    the damage is worst, because a duplicate does not merely confuse the solver, it
    COMPELS it to spend a second team identity on one robot.

    Geometry can now make the same argument, which it could not before per-detection
    field positions were joined correctly (see positions_by_det). FRC bumpers are ~0.9 m
    across, so two distinct robots cannot have floor-contact points much closer without
    colliding. Measured on 2026necmp1_qm24, 114 co-detected pairs split cleanly:

        median separation   pairs   median box IoU
        0.0 - 0.9 m            5      0.24 - 0.50
        0.9 - 1.5 m            8         0.01
        above 1.5 m          101         0.00

    Inspected in the video, all five below 0.9 m are one robot with a box on its
    superstructure and another on its lower body -- the low/high split. Their separation
    is not noise: the upper box's bottom edge sits partway up the robot, so it projects
    further along the camera ray. That also means a duplicate has been feeding a WRONG
    field position into the kinematic term, on top of stealing a team.

    KEEPER IS THE LOWER BOX, not the longer track. Only the box whose bottom edge is on
    the floor projects to the right place, and positions now feed a hard constraint.
    Box geometry is never rewritten -- a union box would be a better box, but
    positions_by_det keys on (frame, box) and reshaping would silently break that join.
    """
    import numpy as _np
    if not pos_at:
        return rows, []
    seps = defaultdict(list)
    ious = defaultdict(list)
    dxs = defaultdict(list)
    vovs = defaultdict(list)
    bottom = defaultdict(list)
    for r in rows:
        here = []
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            v = pos_at.get((r["f"], tuple(round(float(c), 1) for c in d["xyxy"])))
            if v:
                here.append((d, v))
                bottom[d["tid"]].append(float(d["xyxy"][3]))
        for i, (da, va) in enumerate(here):
            for db, vb in here[i + 1:]:
                k = (min(da["tid"], db["tid"]), max(da["tid"], db["tid"]))
                seps[k].append(float(_np.hypot(va[1] - vb[1], va[2] - vb[2])))
                ax1, ay1, ax2, ay2 = da["xyxy"]
                bx1, by1, bx2, by2 = db["xyxy"]
                iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                un = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - iw * ih
                ious[k].append(iw * ih / un if un > 0 else 0.0)
                w = max(ax2 - ax1, bx2 - bx1, 1.0)
                dxs[k].append(abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2) / w)
                vovs[k].append(ih / max(min(ay2 - ay1, by2 - by1), 1.0))

    absorb = {}
    merges = []
    cand = []
    for k, v in seps.items():
        if len(v) < min_frames:
            continue
        md, mi = float(_np.median(v)), float(_np.median(ious[k]))
        mdx = float(_np.median(dxs[k])) if dxs[k] else 9.9
        mvo = float(_np.median(vovs[k])) if vovs[k] else 0.0
        if mdx < DUP_DX_WIDTHS and mvo > DUP_V_OVERLAP and mi > min_iou:
            cand.append((mdx, mi, len(v), k))
    for md, mi, n, (a, b) in sorted(cand):
        ba = float(_np.median(bottom[a])) if bottom[a] else 0.0
        bb = float(_np.median(bottom[b])) if bottom[b] else 0.0
        keep, gone = (a, b) if ba >= bb else (b, a)   # larger y2 == nearer the floor
        while keep in absorb:
            keep = absorb[keep]
        if gone == keep or gone in absorb:
            continue
        absorb[gone] = keep
        merges.append({"keep": keep, "absorbed": gone, "frames": n,
                       "dxWidths": round(md, 2), "iou": round(mi, 2)})
    if not absorb:
        return rows, []
    out = []
    for r in rows:
        dets, seen = [], set()
        for d in r["dets"]:
            t = d["tid"]
            while t in absorb:
                t = absorb[t]
            if t >= 0 and t in seen:
                continue          # keeper already has a box this frame; drop the copy
            if t >= 0:
                seen.add(t)
            dets.append(dict(d, tid=t))
        out.append({**r, "dets": dets})
    return out, merges


def track_info(rows, positions: dict | None, pos_at: dict | None = None):
    """Per-track span, alliance and endpoint positions (metres where available)."""
    info: dict[int, dict] = {}
    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid < 0:
                continue
            it = info.setdefault(tid, {"t0": r["t"], "t1": r["t"], "n": 0,
                                       "alli": Counter()})
            it["t0"] = min(it["t0"], r["t"])
            it["t1"] = max(it["t1"], r["t"])
            it["n"] += 1
            if d.get("alliance"):
                it["alli"][d["alliance"]] += 1
    for tid, it in info.items():
        c = it["alli"]
        it["alliance"] = c.most_common(1)[0][0] if c else None
        # How much evidence actually backs that call: what share of this track's
        # detections got a hue decision at all, times how one-sided those decisions
        # were. The mode alone hides the difference between 500/500 detections
        # agreeing and 3 of 500 agreeing, and the solver was treating both as certain.
        #
        # MEASURED, on 2026mawor: team 190 runs dark non-standard bumpers, and only
        # 30% of its detections get any hue call -- of which 75% are right, so ~22% of
        # its detections carry a correct alliance. Every other robot in that match
        # scores 75-97%. A flat penalty asserts the same confidence for both, and when
        # 190's mode comes out wrong it charges the CORRECT team 250 (25 votes' worth).
        dec = sum(c.values())
        it["alliConf"] = ((dec / it["n"]) * (c.most_common(1)[0][1] / dec)
                          if dec and it["n"] else 0.0)
        del it["alli"]
    if pos_at is not None:
        # Keyed by (frame, box), so it follows a detection through every split.
        pts = defaultdict(list)
        for r in rows:
            for d in r["dets"]:
                if d["tid"] < 0:
                    continue
                v = pos_at.get((r["f"], tuple(round(float(c), 1) for c in d["xyxy"])))
                if v:
                    pts[d["tid"]].append(v)
        for tid, ps in pts.items():
            if tid in info and ps:
                ps.sort()
                info[tid]["start"] = (ps[0][1], ps[0][2])
                info[tid]["end"] = (ps[-1][1], ps[-1][2])
                # THE WHOLE PATH, not just its ends. solve.pair_forbidden compares
                # a["end"] to b["start"], which is the right question only when the two
                # tracks are SEQUENTIAL. They frequently are not: a 3-detection fragment
                # sitting inside a 232-detection track's span shares no frame with it
                # (so the co-detection constraint never sees the pair) yet yields
                # gap = 0, and the distance measured runs between two points hundreds of
                # frames apart. Measured on 2026mawor: 92 cross-field jumps survived,
                # and every interleaved pair inspected shared zero frames.
                info[tid]["path"] = ps
        n_have = sum(1 for t in info if "start" in info[t])
        print(f"[robots] kinematic endpoints on {n_have}/{len(info)} track(s)")
    elif positions:
        pts = defaultdict(list)
        for s in positions["samples"]:
            if s["tid"] >= 0 and "offfield" not in s["flags"]:
                pts[s["tid"]].append((s["t"], s["x"], s["y"]))
        for tid, ps in pts.items():
            if tid in info and ps:
                ps.sort()
                info[tid]["start"] = (ps[0][1], ps[0][2])
                info[tid]["end"] = (ps[-1][1], ps[-1][2])
                info[tid]["path"] = ps
    return info


ALLI_WIN = 3.0        # seconds per bin when reading a track's alliance over time
ALLI_MIN_N = 5        # decided detections before a bin gets a vote
ALLI_PURE = 0.80      # ...and this much one-sidedness before we believe it
ALLI_MIN_RUN = 2      # consecutive agreeing bins before we believe a CHANGE


def split_on_alliance(rows, window: float = ALLI_WIN, min_run: int = ALLI_MIN_RUN):
    """Cut tracks where their own bumper HUE changes alliance. Run this FIRST.

    MEASURED, and it reframes the identity problem: 13 of 31 tracks change alliance
    mid-life, and 72% of all detections sit in one that does. Track 8 reads
    RRRRRRRRRbbbbb.bbbb over time; track 22 reads bbbbbbbbbbRRRRRRR. Those are not
    noise -- the per-track chicklet sheet shows it directly, red "6329" crops turning
    into blue "9644" crops halfway along the same track id.

    Nothing downstream survives that. A chimeric track poisons its own alliance vote
    (so the solver's alliance term goes toothless), pools two robots' bumper votes into
    one tally, and drags two robots' positions into one route.

    Why hue rather than the OCR votes split_chimeras already uses: there are 319
    confident bumper reads in this match against ~12,000 detections carrying a hue
    call. 38x the density is the difference between catching a switch and catching one
    switch in ten.

    The cost of being wrong is asymmetric and we lean accordingly: a wrong split costs
    a fragment, which the grouping exists to reassemble, while a missed switch is
    permanent. But single-bin hue blips are common (a robot crossing a coloured field
    element), so a change must hold for `min_run` consecutive bins before we believe
    it -- without that, tracks 3, 16, 18 and 26 would each be cut on one stray bin.
    """
    by_tid = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0 and d.get("alliance") in ("red", "blue"):
                by_tid[d["tid"]].append((r["t"], d["alliance"]))

    cuts: dict[int, list[float]] = {}
    for tid, pts in by_tid.items():
        bins: dict[int, Counter] = defaultdict(Counter)
        for t, a in pts:
            bins[int(t // window)][a] += 1
        decided: list[tuple[int, str]] = []
        for k in sorted(bins):
            c = bins[k]
            tot = sum(c.values())
            if tot < ALLI_MIN_N:
                continue
            if c["red"] / tot >= ALLI_PURE:
                decided.append((k, "red"))
            elif c["blue"] / tot >= ALLI_PURE:
                decided.append((k, "blue"))

        runs: list[list] = []          # [colour, first_bin, last_bin]
        for k, a in decided:
            if runs and runs[-1][0] == a:
                runs[-1][2] = k
            else:
                runs.append([a, k, k])
        solid = [r for r in runs if (r[2] - r[1]) + 1 >= min_run]
        for a, b in zip(solid, solid[1:]):
            if a[0] != b[0]:
                cuts.setdefault(tid, []).append((a[2] + 1 + b[1]) / 2.0 * window)

    if not cuts:
        return rows, 0, {}

    next_id = max((d["tid"] for r in rows for d in r["dets"]), default=0) + 1
    remap: dict[tuple[int, int], int] = {}
    orig_of: dict[int, int] = {}
    n_split = 0
    for tid, ts in cuts.items():
        orig_of[tid] = tid
        for k in range(len(ts)):
            remap[(tid, k + 1)] = next_id
            orig_of[next_id] = tid
            next_id += 1
            n_split += 1

    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid in cuts:
                seg = sum(1 for c in cuts[tid] if r["t"] >= c)
                if seg:
                    d.setdefault("orig_tid", tid)
                    d["tid"] = remap[(tid, seg)]
    return rows, n_split, orig_of


def split_impossible_steps(rows, pos_at, mult: float = 1.5):
    """Cut a track where IT MOVES further between two of its own detections than a
    robot can.

    Every kinematic check until now compared two TRACKS. That cannot see a track which
    is itself wrong: a bad detection arrives already wearing the right identity, so
    there is no pair to forbid and the route simply draws a line to it.

    2026mawor_qm13 track 275, the largest surviving jump in the event: at f4584 the box
    is a robot beside the red hub (60x50 px, conf 0.537); at f4596 it is a PERSON in an
    FTA vest at the near barrier (98x84 px, conf 0.613). The detector scored the person
    higher than the robot it had been following, BoT-SORT associated it into the track,
    and the export drew 7.30 m in 0.20 s. Nothing about the homography is involved --
    the box moved 335 px and grew 60%. The alliance check cannot help either, because a
    navy vest reads as blue bumper colour absorbed into a blue robot's track.

    The bound that already knows this is impossible is EMPIRICAL_P999_M; it just never
    ran within a track. Applied here it is surgical: across 13 curated matches, 10 of
    159,830 consecutive within-track steps exceed 1.5x budget -- 0.006%, about one cut
    per match. Nothing exceeds 3x, so the setting is not near a cliff in either
    direction.

    Cutting rather than dropping the detection: which SIDE of the step is the impostor
    is not knowable here, and a cut lets the solver decide by treating the two pieces as
    separate candidates. Returns (rows, n_cuts).
    """
    from .solve import distance_budget
    if not pos_at:
        return rows, 0
    seq = defaultdict(list)
    for ri, r in enumerate(rows):
        for di, d in enumerate(r["dets"]):
            if d["tid"] < 0:
                continue
            v = pos_at.get((r["f"], tuple(round(float(c), 1) for c in d["xyxy"])))
            if v:
                seq[d["tid"]].append((r["t"], v[1], v[2], ri, di))
    nxt = max((d["tid"] for r in rows for d in r["dets"]), default=-1) + 1
    cuts = 0
    for tid, v in seq.items():
        v.sort()
        new = None
        for a, b in zip(v, v[1:]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            d = float(np.hypot(b[1] - a[1], b[2] - a[2]))
            if d > mult * distance_budget(dt):
                new = nxt
                nxt += 1
                cuts += 1
            if new is not None:
                rows[b[3]]["dets"][b[4]]["tid"] = new
    return rows, cuts


def split_on_appearance(rows, npz_path: Path, thresh: float, win: int = 12,
                        metric: str = "hellinger", head=None):
    """Cut tracks where the robot's APPEARANCE changes discontinuously.

    Hue splitting only separates red from blue. Within one alliance the three robots
    share a bumper colour, so what is left has to come from the superstructure, which
    rtrack.appear describes as a coarse grayscale histogram per band per detection.

    The change-point metric below is the Hellinger distance sqrt(1 - sum(sqrt(a*b))),
    which is ONLY a distance when the descriptor sums to 1. Any change to
    appear.descriptor must preserve that: a descriptor summing to 3 makes the
    Bhattacharyya coefficient ~3, so 1-3 clips to 0 and every distance silently
    becomes zero -- no cuts, no error, one point of coverage gone.

    THE THRESHOLD IS DESCRIPTOR-SPECIFIC. Grayscale compresses the distance scale
    relative to the old HSV descriptor, so the previous 0.55 default fires NOTHING on
    it. Re-tuned end to end on f1m3 (curated), which is the only calibration that has
    ever been trustworthy here:

        threshold   appearance cuts   coverage   mean custody
          0.55              0             96%         76%
          0.35              8             96%         76%
          0.30             25             96%         76%
          0.25             78             98%         77%      <- default
          0.22            121             97%         76%
          0.20            160             97%         76%
          0.15            268             96%         76%

    The shape is the same one the HSV descriptor showed and the reasoning is unchanged:
    a cut in the middle of one robot's track costs a fragment the grouping reassembles,
    while a missed switch is permanent -- so lean toward cutting. But past the optimum
    the fragments carry too little evidence to be grouped confidently and coverage
    falls again. 0.25 is the peak on both metrics.

    An earlier calibration tried to tune this against hue-confirmed switches directly
    and concluded the discriminator was poor (background distances overlapped real
    switches almost completely). That was true and still is; it is also not the right
    target, because the asymmetry above means the best threshold for END-TO-END
    coverage is far more aggressive than the best threshold for switch precision. Tune
    on coverage and custody, not on switch recall.
    """
    if thresh <= 0 or not npz_path.exists():
        return rows, 0, {}
    z = np.load(npz_path)
    tid_a, t_a, feat = z["tid"], z["t"], z["feat"]
    # COSINE for the learned embedding, Hellinger for the histogram. The two are not
    # interchangeable: Hellinger is only a distance for normalised distributions, and
    # an embedding is not one -- feeding it here would clip to zero and silently cut
    # nothing (the same failure the docstring above warns about from the other
    # direction). The whitening head is applied first when present, because splitting
    # asks the same question re-identification does -- is this the same robot -- and
    # the head is what makes that question answerable within an alliance.
    if metric == "cosine" and head is not None:
        feat = head(feat)

    cuts: dict[int, list[float]] = {}
    for tid in sorted(set(tid_a.tolist())):
        m = tid_a == tid
        if m.sum() < 2 * win + 4:
            continue
        order = np.argsort(t_a[m])
        ts, F = t_a[m][order], feat[m][order]
        n = len(F)
        cs = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F]), axis=0)
        d = np.zeros(n)
        for i in range(win, n - win):
            a = (cs[i] - cs[i - win]) / win
            b = (cs[i + win] - cs[i]) / win
            if metric == "cosine":
                na = float(np.linalg.norm(a)) or 1.0
                nb = float(np.linalg.norm(b)) or 1.0
                d[i] = 1.0 - float(a @ b) / (na * nb)
            else:
                d[i] = np.sqrt(max(0.0,
                                   1.0 - np.sum(np.sqrt(np.clip(a * b, 0, None)))))
        chosen: list[int] = []
        for i in np.argsort(-d):
            if d[i] < thresh:
                break
            if all(abs(int(i) - j) >= win for j in chosen):
                chosen.append(int(i))
        for i in sorted(chosen):
            cuts.setdefault(tid, []).append(float(ts[i]))

    if not cuts:
        return rows, 0, {}
    return _apply_cuts(rows, cuts)


def _apply_cuts(rows, cuts: dict[int, list[float]]):
    """Shared tail of every splitter: renumber later segments and report the mapping."""
    next_id = max((d["tid"] for r in rows for d in r["dets"]), default=0) + 1
    remap: dict[tuple[int, int], int] = {}
    orig_of: dict[int, int] = {}
    n_split = 0
    for tid, ts in cuts.items():
        orig_of[tid] = tid
        for k in range(len(sorted(ts))):
            remap[(tid, k + 1)] = next_id
            orig_of[next_id] = tid
            next_id += 1
            n_split += 1
    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid in cuts:
                seg = sum(1 for c in cuts[tid] if r["t"] >= c)
                if seg:
                    d.setdefault("orig_tid", tid)
                    d["tid"] = remap[(tid, seg)]
    return rows, n_split, orig_of


def split_chimeras(rows, ident, window=6.0):
    """Cut tracks where their own bumper votes change identity.

    A chimeric track is the one error the colouring CANNOT undo: once two robots
    share a track id their votes are permanently mixed, and no grouping or assignment
    can separate them. A fragment, by contrast, costs nothing -- the colouring exists
    to reassemble fragments.

    That asymmetry means we should prefer MORE, purer tracks over fewer, longer ones,
    which is the opposite of what track-count minimisation optimises for. This splits
    at the midpoint between the last window voting A and the first voting B.

    Returns (rows, n_splits, orig_of) where orig_of maps every resulting track id back
    to the id it was cut from -- retally() needs that to hand each segment the votes
    actually cast during its own lifetime. New pieces get fresh ids above the maximum.
    """
    cuts: dict[int, list[float]] = {}
    for tid_s, t in ident["tracks"].items():
        tl = [(w, team, n) for w, team, n in t.get("timeline", []) if n >= 2]
        for (w0, a, _), (w1, b, _) in zip(tl, tl[1:]):
            if a != b:
                cuts.setdefault(int(tid_s), []).append((w0 + w1) / 2 + window / 2)
    if not cuts:
        return rows, 0, {}

    next_id = max((d["tid"] for r in rows for d in r["dets"]), default=0) + 1
    remap: dict[tuple[int, int], int] = {}
    orig_of: dict[int, int] = {}
    n_split = 0
    for tid, ts in cuts.items():
        orig_of[tid] = tid          # segment 0 keeps the original id
        for k in range(len(ts)):
            remap[(tid, k + 1)] = next_id
            orig_of[next_id] = tid
            next_id += 1
            n_split += 1

    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid in cuts:
                seg = sum(1 for c in cuts[tid] if r["t"] >= c)
                if seg:
                    d["orig_tid"] = tid
                    d["tid"] = remap[(tid, seg)]
    return rows, n_split, orig_of


def retally(ident: dict, rows, orig_of: dict[int, int]) -> dict:
    """Re-attribute bumper votes to track SEGMENTS after chimera splitting.

    Without this, a cut track keeps its whole tally on segment 0 -- including votes
    cast during the windows that were split off. Observed directly: a 15-detection
    segment reported 24 votes, which is impossible. The split then separates the
    detections but not the evidence, so the chimera it was built to break stays intact
    in the votes, and segment 0 looks far better supported than it is.

    Returns a NEW ident; the original is left alone so a caller can compare.
    """
    if not orig_of:
        return ident

    spans: dict[int, list[float]] = {}
    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid < 0:
                continue
            s = spans.setdefault(tid, [r["t"], r["t"]])
            s[0], s[1] = min(s[0], r["t"]), max(s[1], r["t"])

    # segments of each original track, in time order
    segs: dict[int, list[int]] = defaultdict(list)
    for tid, orig in orig_of.items():
        if tid in spans:
            segs[orig].append(tid)
    for orig in segs:
        segs[orig].sort(key=lambda t: spans[t][0])

    tracks = {k: dict(v) for k, v in ident["tracks"].items()}
    for orig, members in segs.items():
        src = ident["tracks"].get(str(orig))
        if not src:
            continue
        vl = src.get("voteList")
        if vl is None:
            continue                       # older identity.json: nothing we can do
        fresh: dict[int, list] = {m: [] for m in members}
        for vt, team in vl:
            # the segment whose span contains this vote, else the nearest one
            hit = next((m for m in members
                        if spans[m][0] <= vt <= spans[m][1]), None)
            if hit is None:
                hit = min(members, key=lambda m: min(abs(vt - spans[m][0]),
                                                     abs(vt - spans[m][1])))
            fresh[hit].append((vt, team))
        for m, vs in fresh.items():
            tally = Counter(team for _t, team in vs)
            wins = defaultdict(Counter)
            for vt, team in vs:
                wins[int(vt // 6.0)][team] += 1
            timeline = [(int(k * 6.0), c.most_common(1)[0][0], sum(c.values()))
                        for k, c in sorted(wins.items())]
            winner = tally.most_common(1)[0] if tally else (None, 0)
            tracks[str(m)] = {
                "detections": sum(1 for r in rows for d in r["dets"]
                                  if d["tid"] == m),
                "votes": len(vs), "tally": dict(tally), "team": winner[0],
                "share": round(winner[1] / max(len(vs), 1), 3),
                "timeline": timeline, "switches": [],
                "voteList": [[vt, team] for vt, team in vs],
            }
    return {**ident, "tracks": tracks}


def colour(info, con, ident, n_colours=N_ROBOTS):
    """Greedy interval colouring, choosing among available groups by cost.

    Greedy by start time is optimal in colour COUNT for interval graphs; the cost
    function decides WHICH valid colouring we get.

    THIS ORDERING IS LOAD-BEARING. Ordering by
    vote confidence instead -- so tracks that know who they are anchor the groups --
    is intuitively better and is not: it breaks the interval structure that makes six
    colours sufficient, taking parked tracks 4 -> 7 and coverage 83% -> 80%.

    Be aware of what this function can and cannot fix. Measured on the test match,
    17 of 44 tracks had exactly ONE legal group by the time they were placed and 10
    had none; only 5 ever saw three or more options. The colouring is therefore
    decided almost entirely by the co-detection structure, not by any cost we assign
    -- which is why raising the vote-disagreement penalty below changed the output not
    at all, and why a track carrying 30 of 30 votes for 9644 still ended up merged
    with tracks reading 6329. Improving identity from here needs a different
    formulation, not a better weight; see OCR_PLAN.md.
    """
    order = sorted(info, key=lambda t: (info[t]["t0"], -info[t]["n"]))
    groups: list[list[int]] = []
    gvotes: list[Counter] = []
    assign: dict[int, int] = {}
    extra = []

    for tid in order:
        it = info[tid]
        my_votes = Counter(ident["tracks"].get(str(tid), {}).get("tally", {}))
        best, best_cost = None, None
        for gi, members in enumerate(groups):
            if any(m in con[tid] for m in members):
                continue                      # hard conflict: co-detected
            last = members[-1]
            li = info[last]
            cost = 0.0
            # Compare against the GROUP's majority alliance, not just its most recent
            # member. Using the last member let an undecided track join, become the
            # "last", and then admit a track of the opposite alliance -- which is how
            # a group ended up holding 30 votes for a blue team while reading red.
            galli = Counter(info[m]["alliance"] for m in members
                            if info[m]["alliance"]).most_common(1)
            galli = galli[0][0] if galli else None
            if it["alliance"] and galli and it["alliance"] != galli:
                cost += ALLIANCE_MISMATCH
            gap = max(0.0, it["t0"] - li["t1"])
            cost += GAP_PENALTY * gap
            if "start" in it and "end" in li:
                d = float(np.hypot(it["start"][0] - li["end"][0],
                                   it["start"][1] - li["end"][1]))
                budget = C.ROBOT_MAX_SPEED_MS * max(gap, 0.1) + 1.0
                cost += 12.0 * max(0.0, d - budget) / budget
            # Left as a flat penalty deliberately. Scaling it by how much evidence
            # each side has -- so 30 votes at 100% outweighs 2 votes at 50% -- is
            # clearly the more principled rule and changed the output by NOTHING,
            # because the tracks it would affect had no alternative group to move to.
            # See the note in colour()'s docstring before tuning this.
            if my_votes and gvotes[gi]:
                mine = my_votes.most_common(1)[0][0]
                theirs = gvotes[gi].most_common(1)[0][0]
                if mine != theirs:
                    cost += VOTE_DISAGREE
            if best_cost is None or cost < best_cost:
                best, best_cost = gi, cost

        if best is None:
            if len(groups) < n_colours:
                groups.append([tid]); gvotes.append(Counter(my_votes))
                assign[tid] = len(groups) - 1
            else:
                # more than n_colours mutually co-detected: a real over-detection,
                # not a colouring failure. Park it rather than corrupt a group.
                extra.append(tid)
            continue
        if best_cost >= ALLIANCE_MISMATCH and len(groups) < n_colours:
            groups.append([tid]); gvotes.append(Counter(my_votes))
            assign[tid] = len(groups) - 1
            continue
        groups[best].append(tid)
        gvotes[best].update(my_votes)
        assign[tid] = best
    return groups, gvotes, assign, extra


def plan_deconflict_cuts(rows, groups: list[list[int]], targets: list[int],
                         min_run: int = 25) -> dict[int, list[float]]:
    """Cut a track that no group will accept, at the boundary of a stretch one will.

    The co-detection constraint is evaluated per FRAME but disqualifies a track for its
    ENTIRE life: if track A shares one frame with any member of group G, A can never be
    G. For a long track that is fatal -- over 85 seconds it brushes past a member of
    every group at some point and becomes unplaceable everywhere.

    MEASURED, and it is the whole problem: track 35 (1077 detections) was blocked from
    all six groups, yet across its 1077 frames the number of groups actually on screen
    was never more than five. A slot was free in 44-67% of its frames. Nothing was
    competing for its place; the constraint was just quantified over the wrong thing.

    So find, for each group, the longest run of the track's own frames during which
    that group has nobody on screen, and cut the track at that run's edges. The middle
    piece is then legally assignable to that group and the remainder is re-examined on
    the next round. This does not decide WHO the track is -- the solver still does that
    on the evidence -- it only stops a whole track being discarded because of an
    overlap that happened somewhere else in the match.
    """
    present: dict[float, set[int]] = {}
    times_of: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        ids = {d["tid"] for d in r["dets"] if d["tid"] >= 0}
        present[r["t"]] = ids
        for t in ids:
            times_of[t].append(r["t"])

    cuts: dict[int, list[float]] = {}
    for tid in targets:
        ts = sorted(times_of.get(tid, ()))
        if len(ts) < min_run:
            continue
        best = None
        for members in groups:
            ms = set(members)
            run: list[int] = []
            for i, tt in enumerate(ts):
                if present[tt] & ms:
                    if run and (best is None or len(run) > best[0]):
                        best = (len(run), run[0], run[-1])
                    run = []
                else:
                    run.append(i)
            if run and (best is None or len(run) > best[0]):
                best = (len(run), run[0], run[-1])
        # A run covering the whole track means it was never really blocked; and a run
        # shorter than min_run is not worth fragmenting the track for.
        if not best or best[0] < min_run or (best[1] == 0 and best[2] == len(ts) - 1):
            continue
        _, a, b = best
        at = []
        if a > 0:
            at.append((ts[a - 1] + ts[a]) / 2.0)
        if b < len(ts) - 1:
            at.append((ts[b] + ts[b + 1]) / 2.0)
        if at:
            cuts[tid] = at
    return cuts


FIELD_SLACK_M = 0.6     # matches project.SLACK_M; a robot's foot point can sit just
                        # outside the rectangle through projection error alone


def drop_offview(rows, stem: str):
    """Remove detections from spans where the camera was not in its calibrated pose.

    Returns (rows, n_dropped, n_total), and leaves rows untouched when no view check
    has been run -- absence of a finding is not a finding.

    WHY THIS DROPS RATHER THAN FLAGS, WHICH IS THE OPPOSITE OF rtrack.project.

    project attaches `viewmoved` and keeps the row, on the reasoning that a camera move
    invalidates the GEOMETRY while leaving the IDENTITY work perfectly good: the robot
    was still recognised, it just cannot be placed. That reasoning is sound for a cut to
    a close-angle camera covering this same match.

    It is wrong, and dangerously so, for what 2026necmp1_qm4 actually does. At t=120 the
    broadcast cuts to ANOTHER DIVISION'S FIELD and never comes back. The FMS overlay is
    unchanged throughout -- same title, same roster, a clock that keeps counting -- so
    every identity check we have says this is still qualification 4. The robots beneath
    it are strangers. The tracker labelled four of them 10910, 6324, 4905 and 4925 with
    full confidence; their bumpers read 839, ~2223, ~8389 and ~95.

    The damage is not confined to those samples. 45% of qm4's detections and 70 of its
    141 track ids exist ONLY in that span, so CP-SAT was distributing six teams across a
    population half of which belongs to another match, and rtrack.reid would have taught
    the event gallery what those six teams "look like" from the wrong robots -- poisoning
    the pre-fill of every later match. (It had not yet: no 2026necmp1 gallery existed.)

    So the policy here is deliberately more conservative than project's: a span we cannot
    verify the camera on is a span we cannot verify the ROBOTS in either. Losing the
    close-up crops from a legitimate cut costs some good evidence. Accepting a foreign
    robot as a team's exemplar costs correctness, quietly, everywhere downstream.
    """
    from . import viewcheck as VC
    n_tot = sum(len(r.get("dets", ())) for r in rows)
    doc = VC.load(stem)
    if not doc or not (doc.get("validIntervals") or []):
        return rows, 0, n_tot
    out, kept = [], 0
    for r in rows:
        if VC.is_valid_at(doc, r["t"]):
            out.append(r)
            kept += len(r.get("dets", ()))
    return out, n_tot - kept, n_tot


def drop_offfield(rows, stem: str, slack: float = FIELD_SLACK_M,
                  calib_stem: str | None = None,
                  allow_unsafe_calibration: bool = False):
    """Remove detections whose foot point projects outside the field. Returns
    (rows, n_dropped, n_total) and leaves rows untouched when there is no calibration.

    THE TEST ALREADY EXISTED; IT JUST RAN TOO LATE. rtrack.project has flagged
    `offfield` per sample since Stage 2, but projection happens after grouping, so a
    detection on a scoring tower or in the crowd has already spawned a track, competed
    for a team in CP-SAT, and taken up a slot in the curator's frame budget by the time
    anything notices it is not on the field. Flagging it at the end reports the problem;
    dropping it here removes it.

    Measured motivation (JG's observation, see README): on the 2026mawor camera 66% of
    curator-confirmed false detections and 52 of 83 false TRACKS lie outside the region
    robots actually occupy, while the same test removes 0% on the championship camera --
    it is worth most exactly where the false-positive rate is worst.

    What it cannot do: sf11m1's false positives are fuel piles ON the carpet, and only
    7% of them fall outside. This is a partial remedy, not a substitute for a detector
    that does not fire on field furniture.

    The slack is in METRES on purpose. An earlier attempt at the same idea dilated a
    hull in image pixels, which is wildly non-uniform under perspective -- an 8% pixel
    dilation swallowed almost the whole gain. Metres are uniform; pixels are not.
    """
    from . import project as PJ
    # A calibration describes a CAMERA, and rtrack.replay names clips after the MATCH,
    # so a whole event's slices share one calibration under a different name. Without
    # this override the filter silently does nothing on every sliced match -- measured
    # on 2026mawor_qm1, that means solving with 8.45 detections/frame instead of ~4.
    cal = calib_stem or stem
    if not (C.CALIB_DIR / f"{cal}.json").exists():
        return rows, 0, sum(len(r["dets"]) for r in rows)
    if (not PJ.calibration_status_for(cal)["usable"]
            and not allow_unsafe_calibration):
        return rows, 0, sum(len(r["dets"]) for r in rows)
    H, lens = PJ.load_calib(cal)
    ref = json.loads((C.CALIB_DIR / "field_ref_2026.json").read_text(encoding="utf-8"))
    FL, FW = ref["fieldSizeM"]

    idx, pts = [], []
    for ri, r in enumerate(rows):
        for di, d in enumerate(r["dets"]):
            x1, _, x2, y2 = d["xyxy"][0], d["xyxy"][1], d["xyxy"][2], d["xyxy"][3]
            idx.append((ri, di))
            pts.append(((x1 + x2) / 2.0, y2))
    if not pts:
        return rows, 0, 0
    XY = PJ.project_points(np.array(pts, np.float32), H, ref, lens)

    kill = defaultdict(set)
    for (ri, di), (X, Y) in zip(idx, XY):
        if not (-slack <= X <= FL + slack and -slack <= Y <= FW + slack):
            kill[ri].add(di)
    if not kill:
        return rows, 0, len(pts)
    out = []
    for ri, r in enumerate(rows):
        if ri in kill:
            r = {**r, "dets": [d for di, d in enumerate(r["dets"])
                               if di not in kill[ri]]}
        out.append(r)
    return out, sum(len(v) for v in kill.values()), len(pts)


def prepare_tracks(stem: str, tracks_p: Path, ident: dict,
                   alliance_split: bool = True, appear_thresh: float = 0.25,
                   chimera_split: bool = True, quiet: bool = True,
                   field_filter: bool = True, calib_stem: str | None = None,
                   view_filter: bool = True):
    """Apply the whole split chain, so callers share ONE track segmentation.

    This exists because two parts of the system disagreed about what a "track" is.
    rtrack.curate built its bundle from labeled.jsonl -- the OUTPUT of a previous run,
    already cut -- while this module starts from the stitched tracks and re-derives its
    own cuts. Measured: of 96 detections present in both, 31 carried a DIFFERENT track
    id, and the bundle's 16 frames held 67 ids against the pipeline's 101.

    Correction anchors are (frame, xy) so labels still landed on the right detections.
    But anything reasoning about TRACKS across the boundary was comparing two id
    spaces: the co-occurrence used to choose conflict-settling frames was measured in
    one grouping and the clashes evaluated in the other, so the frames chosen to
    resolve a confusable pair did not resolve the pairs that actually clashed.

    Returns (rows, ident) with the same segmentation this module will use.
    """
    rows = load_tracks(tracks_p)
    # BEFORE the field filter and before any split. A foreign robot must not reach the
    # segmentation at all: once it has a track id it competes for a team in CP-SAT and
    # takes a slot in the curator's frame budget.
    if view_filter:
        rows, n_v, n_t = drop_offview(rows, stem)
        if n_v and not quiet:
            print(f"[prepare] dropped {n_v}/{n_t} detection(s) outside the "
                  f"calibrated camera view")
    if field_filter:
        # Must match main() exactly. When these two disagreed about what a track is,
        # 31 of 96 shared detections carried different ids and every cross-boundary
        # inference was comparing two id spaces -- see the note above.
        rows, n_off, n_tot = drop_offfield(rows, stem, calib_stem=calib_stem)
        if n_off and not quiet:
            print(f"[prepare] dropped {n_off}/{n_tot} off-field detection(s)")
    if alliance_split:
        rows, n, orig = split_on_alliance(rows)
        if n:
            ident = retally(ident, rows, orig)
            if not quiet:
                print(f"[prepare] {n} alliance split(s)")
    if appear_thresh > 0:
        ap = C.STAGE3_DIR / f"{stem}_appearance.npz"
        rows, n, orig = split_on_appearance(rows, ap, appear_thresh)
        if n:
            ident = retally(ident, rows, orig)
            if not quiet:
                print(f"[prepare] {n} appearance split(s)")
    if chimera_split:
        rows, n, orig = split_chimeras(rows, ident)
        if n:
            ident = retally(ident, rows, orig)
            if not quiet:
                print(f"[prepare] {n} vote-change split(s)")
    return rows, ident


def custody_conflicts(rows, min_overlap_s: float = 1.0, min_frac: float = 0.20,
                      sep_widths: float = 1.0) -> list[dict]:
    """Two tracks holding the same team at overlapping times, far enough apart to be
    two different robots.

    The solver already forbids two co-detected tracks from sharing a team, and that
    holds exactly -- measured, 0 of 2654 frames carry a team on two boxes. But the
    constraint is per FRAME, and two tracks that alternate detections over the same
    stretch never co-occur in any single frame, so nothing stops them sharing a label.

    Most such overlaps are benign and worth keeping: one track filling another's gap,
    or a single robot whose detections flicker between two ids. Measured on the test
    match, three of four overlapping pairs had ZERO detections from one member inside
    the window (a gap-fill), and the fourth alternated densely with its box centres
    73 px apart -- well inside one robot's box width, so one robot, correctly labelled.

    What separates benign from broken is therefore not the overlap but the DISTANCE.
    Two genuinely different robots sat 218-1206 px apart in the clash analysis; one
    robot's flicker sat at 73. So flag only when both tracks are densely present in
    the window AND their centres are more than `sep_widths` box widths apart.
    """
    # All processed timestamps, so "densely present" is measured against the frames
    # that EXIST in the window. Measuring against the two tracks' own timestamps makes
    # the test vacuous when both are sparse: one detection each scores 1 of 2 = 50%.
    all_t = sorted({r["t"] for r in rows})

    det: dict[int, list] = defaultdict(list)
    team: dict[int, str] = {}
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0 or not d.get("team"):
                continue
            x1, y1, x2, y2 = d["xyxy"]
            det[d["tid"]].append((r["t"], (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1))
            team[d["tid"]] = str(d["team"])

    by_team: dict[str, list[int]] = defaultdict(list)
    for tid, t in team.items():
        by_team[t].append(tid)

    out = []
    for t, tids in by_team.items():
        for i, a in enumerate(sorted(tids)):
            for b in sorted(tids)[i + 1:]:
                A, B = det[a], det[b]
                lo = max(min(p[0] for p in A), min(p[0] for p in B))
                hi = min(max(p[0] for p in A), max(p[0] for p in B))
                if hi - lo < min_overlap_s:
                    continue
                wa = [p for p in A if lo <= p[0] <= hi]
                wb = [p for p in B if lo <= p[0] <= hi]
                if not wa or not wb:
                    continue                      # gap-fill, not a conflict
                n = sum(1 for t_ in all_t if lo <= t_ <= hi)
                if min(len(wa), len(wb)) < MIN_CONFLICT_DETS:
                    continue                      # too sparse to mean anything
                if len(wa) < min_frac * n or len(wb) < min_frac * n:
                    continue

                # Compare positions at MATCHED TIMES, never mean against mean.
                # Overlapping spans do not imply coexistence: a span can have holes,
                # and two tracks routinely alternate, one filling the other's gap --
                # which is a handover of ONE robot, not two robots at once. Measured,
                # all three "conflicts" this reported were of that kind: track 94 ran
                # 83.9-84.7 s and track 14 ran 84.9-85.7 s, never once together, and
                # the 356 px "separation" was just where the robot averaged early
                # versus late while travelling 624 px.
                pairs = []
                tb_sorted = sorted(p[0] for p in wb)
                for pa in wa:
                    j = int(np.argmin([abs(tb - pa[0]) for tb in tb_sorted]))
                    dt = abs(tb_sorted[j] - pa[0])
                    if dt > MATCH_DT_S:
                        continue
                    pb = next(p for p in wb if p[0] == tb_sorted[j])
                    pairs.append((np.hypot(pa[1] - pb[1], pa[2] - pb[2]), dt,
                                  (pa[3] + pb[3]) / 2.0))
                if len(pairs) < MIN_CONFLICT_DETS:
                    continue                      # never really on screen together

                # A robot keeps moving between two samples taken dt apart, so allow
                # for that before calling the gap a contradiction.
                speed = _local_speed(wa) + _local_speed(wb)
                bad = [p for p in pairs
                       if p[0] > sep_widths * p[2] + speed * p[1]]
                if len(bad) < 0.5 * len(pairs):
                    continue                      # one robot flickering between ids
                sep = float(np.median([p[0] for p in bad]))
                width = float(np.median([p[2] for p in bad]))
                out.append({"team": t, "tracks": [a, b],
                            "window": [round(lo, 1), round(hi, 1)],
                            "dets": [len(wa), len(wb)],
                            "matchedPairs": len(pairs),
                            "sepPx": round(sep), "boxPx": round(width)})
    return out


MATCH_SECONDS = 150.0


def auto_start_px(rows, bin_s: float = 0.5, moving: float = 0.30,
                  hold_s: float = 2.0) -> float | None:
    """Auto start from PIXEL motion, for cameras with no calibration.

    Same shape as routes.detect_auto_window -- robots sit still, everything moves at
    once, so the first sustained burst of motion is auto start -- but it needs no
    homography, which is the whole point.

    Speed is measured in BOX WIDTHS PER SECOND rather than pixels per second. A box
    width is about one robot width, so that is a scale-free proxy for robot-lengths per
    second and the same threshold works whether the camera is close or far. Raw px/s
    would need retuning per camera, which is exactly the kind of hidden per-camera
    constant this project keeps getting bitten by.
    """
    per = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            per[d["tid"]].append((r["t"], (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1))
    bins = defaultdict(list)
    for pts in per.values():
        pts.sort()
        for a, b in zip(pts, pts[1:]):
            dt = b[0] - a[0]
            w = max((a[3] + b[3]) / 2.0, 1.0)
            if 0 < dt < 0.5:
                v = np.hypot(b[1] - a[1], b[2] - a[2]) / dt / w
                if v < 12.0:                      # discard teleports (id switches)
                    bins[round(b[0] / bin_s) * bin_s].append(v)
    if not bins:
        return None
    ts = np.array(sorted(bins))
    spd = np.array([np.mean(bins[t]) for t in ts])
    need = int(round(hold_s / bin_s))
    mv = spd > moving
    for i in range(len(ts)):
        if i + need <= len(mv) and mv[i:i + need].all():
            return float(ts[i])
    return None


def match_window(stem: str, rows=None) -> tuple[float, float] | None:
    """[auto start, auto start + 150 s], from the motion profile. None if unavailable.

    Custody MUST be measured against this and not against the tracked span. The clip
    runs 3-180 s while the match runs 8.5-158.5 s, so 406 of 2654 frames sit outside
    it -- 15% of the denominator, during which robots are staged, celebrating, or off
    the field entirely. Counting those as "lost custody" understates every robot.

    TWO SOURCES, because the metric one is not always available. The preferred path
    reads positions.json, which needs a calibration. When there is none this used to
    return None and custody silently fell back to the whole clip -- which made the
    number INCOMPARABLE between cameras while still printing as a percentage in the
    same table. Measured: 2026necmp1_sf11m1 read 61% custody that way and 73% over the
    same [8,158] window the other matches used, so it looked like the worst camera in
    the fleet when it was mid-pack. The pixel fallback removes that trap.
    """
    try:
        from .routes import detect_auto_window
        p = C.STAGE2_DIR / f"{stem}_positions.json"
        if p.exists():
            t0, _ = detect_auto_window(json.loads(p.read_text(encoding="utf-8")))
            return t0, t0 + MATCH_SECONDS
    except Exception:
        pass
    if rows is not None:
        t0 = auto_start_px(rows)
        if t0 is not None:
            return t0, t0 + MATCH_SECONDS
    return None


def custody(rows, window: tuple[float, float] | None = None) -> dict[str, dict]:
    """How much of the match we actually hold each robot.

    The headline accuracy numbers say how often a label is RIGHT. This says how often
    there is a label at all, which is the other half of whether a route is usable --
    a 94%-correct track covering 60% of the match still leaves 40% of the route
    missing.
    """
    lo, hi = window if window else (-1e9, 1e9)
    rows = [r for r in rows if lo <= r["t"] <= hi]
    frames = sorted({r["t"] for r in rows})
    if not frames:
        return {}
    step = float(np.median(np.diff(frames))) if len(frames) > 1 else 0.0
    seen: dict[str, set] = defaultdict(set)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0 and d.get("team"):
                seen[str(d["team"])].add(round(r["t"], 3))

    out = {}
    for t, ts in seen.items():
        s = sorted(ts)
        gaps = [(b - a) for a, b in zip(s, s[1:]) if (b - a) > max(1.5 * step, 0.3)]
        out[t] = {"frames": len(s), "pct": 100.0 * len(s) / len(frames),
                  "gaps": len(gaps), "longestGapS": round(max(gaps), 1) if gaps else 0.0,
                  "lostS": round(sum(gaps), 1),
                  "firstS": round(s[0], 1), "lastS": round(s[-1], 1)}
    return out


def name_groups(groups, gvotes, info, red, blue):
    """Assign 6 groups to 6 teams optimally, with the alliance split enforced."""
    from scipy.optimize import linear_sum_assignment
    teams = red + blue
    team_alli = {t: "red" for t in red} | {t: "blue" for t in blue}
    n = max(len(groups), len(teams))
    cost = np.full((n, n), 50.0)
    for gi, votes in enumerate(gvotes):
        members = groups[gi]
        alli = Counter(info[m]["alliance"] for m in members if info[m]["alliance"])
        g_alli = alli.most_common(1)[0][0] if alli else None
        total = sum(votes.values())
        for ti, team in enumerate(teams):
            c = 50.0 - (100.0 * votes.get(team, 0) / total if total else 0.0)
            if g_alli and team_alli[team] != g_alli:
                c += 500.0
            cost[gi, ti] = c
    r, c = linear_sum_assignment(cost)
    return {int(gi): teams[ti] for gi, ti in zip(r, c)
            if gi < len(groups) and ti < len(teams)}


def verify(groups, ident, window=20.0):
    """Flag a group whose pooled votes disagree across time -- a likely chimera."""
    out = {}
    for gi, members in enumerate(groups):
        wins = defaultdict(Counter)
        for m in members:
            for t, team, _n in ident["tracks"].get(str(m), {}).get("timeline", []):
                wins[int(t // window)][team] += _n
        seq = [c.most_common(1)[0][0] for _, c in sorted(wins.items())
               if sum(c.values()) >= 3]
        out[gi] = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
    return out



# A robot cannot cross the field between two consecutive samples. When a team's route
# does exactly that, two tracks have been put on one robot that cannot both be it.
KIN_MAX_GAP_S = 1.5      # beyond this the route already draws an honest gap
KIN_MIN_DETS = 4         # a 2-detection fragment is not worth asking a human about
KIN_SLACK = 1.30         # allow 30% over the limit for projection noise at range
# SPEED ALONE IS NOT ENOUGH, because dt at a handover is often a single frame. At 15 Hz
# a 0.53 m step reads as 7.9 m/s and is smaller than a robot -- that is the box-bottom
# jitter project.py documents, not a teleport, and no curator can adjudicate it because
# the two positions are not far enough apart to be different robots in the first place.
# The question being asked is "are these two different machines", so the jump must be
# at least a robot-and-a-half before it is worth a person's time.
KIN_MIN_DIST_M = 1.5


def kinematic_conflicts(rows, positions: dict | None,
                        max_speed: float = None) -> list[dict]:
    """Team routes that teleport where one track hands over to another.

    THE GAP EVERY OTHER GUARD LEAVES. There are three kinematic checks and this case
    falls between all of them:

      project.py's `fast` flag groups BY TRACK, so a discontinuity BETWEEN two tracks
        of one team is invisible to it -- each track is internally smooth.
      custody_conflicts only examines tracks that OVERLAP in time, and wants a second
        of it; two tracks handing over sequentially never qualify.
      solve.py's _pair_cost does penalise an implausible pairing, but it is SOFT and a
        curator pin is HARD, so it cannot win against a mistaken label.

    Measured on the 20 published 2026necmp1 matches: 212 impossible transitions, in
    every single match, the worst implying 61 m/s -- about twelve times a drivetrain's
    top speed. They are drawn as straight lines across the field and read as real
    movement.

    Returns records shaped like custody_conflicts', so _write_clash_bundle can ask
    about both kinds without caring which it is holding.
    """
    if not positions:
        return []          # metres are required; a pixel proxy is not comparable
    # POSITIONS MAY BE FROM AN OLDER SEGMENTATION, in which case its track ids name
    # different things and every question built from them would be nonsense. The whole
    # id-space hazard prepare_tracks documents, one layer down. Cheap to test: the two
    # should be talking about mostly the same tracks.
    live = {d["tid"] for r in rows for d in r["dets"] if d["tid"] >= 0}
    have = {s_["tid"] for s_ in positions.get("samples", ()) if s_["tid"] >= 0}
    if live and len(live & have) < 0.5 * len(live):
        print(f"[clash] positions.json covers {len(live & have)}/{len(live)} of the "
              f"current tracks -- it predates this segmentation, so the kinematic "
              f"check is SKIPPED rather than asked about the wrong tracks")
        return []
    max_speed = max_speed or (C.ROBOT_MAX_SPEED_MS * KIN_SLACK)
    pts: dict[int, list] = defaultdict(list)
    team_of: dict[int, str] = {}
    for sm in positions["samples"]:
        if sm["tid"] < 0 or not sm.get("team"):
            continue
        if "offfield" in sm["flags"] or "viewmoved" in sm["flags"]:
            continue
        pts[sm["tid"]].append((sm["t"], sm["x"], sm["y"]))
        team_of[sm["tid"]] = str(sm["team"])
    for v in pts.values():
        v.sort()

    by_team: dict[str, list] = defaultdict(list)
    for tid, t in team_of.items():
        if len(pts[tid]) >= KIN_MIN_DETS:
            by_team[t].append(tid)

    out = []
    for team, tids in by_team.items():
        # EVERY ORDERED PAIR, not just start-order-adjacent ones. Sorting by start and
        # zipping neighbours looks right and silently misses the common case: a short
        # track NESTED inside a longer one's span. On 2026necmp1_qm16 team 78, track 17
        # (129.73-130.13) sits inside track 111 (123.67-130.73), so start-order put it
        # between 111 and 18 and the real handover -- 111 ending at (7.72,7.00), 18
        # starting 0.8 s later at (2.98,3.07), 7.7 m/s -- was never examined at all.
        # That was the exact jump visible in the published route.
        for a in tids:
            for b in tids:
                if a == b:
                    continue
                ea = pts[a][-1]
                sb = pts[b][0]
                dt = sb[0] - ea[0]
                if dt <= 0 or dt > KIN_MAX_GAP_S:
                    continue
                # Only the FIRST track to resume counts as the handover partner;
                # without this one long gap generates a conflict per later track.
                if any(ea[0] < pts[c][0][0] < sb[0] for c in tids if c not in (a, b)):
                    continue
                dist = float(np.hypot(sb[1] - ea[1], sb[2] - ea[2]))
                v = dist / dt
                if v <= max_speed or dist < KIN_MIN_DIST_M:
                    continue
                out.append({"team": team, "tracks": [a, b],
                            # The window brackets the handover so the bundle builder
                            # finds frames of BOTH tracks near it -- the last of one
                            # and the first of the other.
                            "window": [round(ea[0] - 0.5, 1), round(sb[0] + 0.5, 1)],
                            "dets": [len(pts[a]), len(pts[b])],
                            "kind": "kinematic",
                            "gapS": round(dt, 2), "distM": round(dist, 2),
                            "impliedMs": round(v, 1)})
    out.sort(key=lambda z: -z["impliedMs"])
    return out


def _write_clash_bundle(stem, args, rows, clashes, red, blue,
                        positions: dict | None = None) -> None:
    """Ask about the clashes that physics could not explain, and nothing else.

    A clash whose two tracks CROSS is resolved automatically -- the tracker swapped
    them and cutting at the crossing recovers both. What reaches here is the other
    kind: two robots visible at the same moment, far apart, both carrying one team
    name. No cut fixes that; only a person can say which is which.

    Frames are chosen to show BOTH tracks of a clash at once, which is the one thing
    the main bundle could not guarantee -- measured, 0 of 12 clashing pairs had ever
    been put in front of the curator together.
    """
    # Ask about CUSTODY CONFLICTS, not the clash list. Clashes are recomputed after
    # the deconfliction rounds and the cuts there dissolve most of them; what actually
    # survives to the output is a custody conflict -- one team held by two tracks that
    # are far apart at the same instant. That is the same question in its final form,
    # and unlike the clash list it is measured on the tracks that were really emitted.
    # TWO KINDS OF THE SAME QUESTION. A custody conflict is "one team, two tracks, at
    # the same instant, far apart". A kinematic conflict is "one team, two tracks, one
    # after the other, too far apart to be the same robot". Both mean at most one of
    # the pair really is that team, and both are answerable only by looking. They are
    # asked together because the curator does not care which detector found it.
    issues = custody_conflicts(rows) + kinematic_conflicts(rows, positions)
    # ONLY ASK ABOUT THE MATCH. Conflicts in staging or post-match footage are real but
    # worthless: nothing downstream uses those seconds, and every question spent there
    # is one a curator did not spend on the match. qm18 otherwise led with a clash at
    # t=496 s, four minutes after its own match window closed.
    try:
        from .curate import match_window
        win = match_window(rows)
    except Exception:
        win = None
    if win:
        before = len(issues)
        issues = [i for i in issues
                  if win[0] <= (i["window"][0] + i["window"][1]) / 2 <= win[1]]
        if before != len(issues):
            print(f"[clash] {before - len(issues)} conflict(s) outside the match window "
                  f"{win[0]:.0f}-{win[1]:.0f}s -- not asked about")
    if not issues:
        print("[clash] no custody or kinematic conflicts -- nothing left to ask about")
        return
    n_kin = sum(1 for i in issues if i.get("kind") == "kinematic")
    if n_kin:
        print(f"[clash] {n_kin} kinematic conflict(s): a team's route jumps further "
              f"than a robot can travel where one track hands over to another")
        for i in [z for z in issues if z.get("kind") == "kinematic"][:8]:
            print(f"[clash]   {i['team']}: tracks {i['tracks']} -- {i['distM']} m in "
                  f"{i['gapS']} s = {i['impliedMs']} m/s")
    from . import curate as CU

    # A custody conflict CANNOT be shown in one frame -- that is what defines it. If
    # the two tracks ever appeared together the hard co-detection constraint would
    # already have stopped them sharing a team. They alternate: A is detected, then B,
    # within the same window, metres apart. So the question takes two frames, one of
    # each, as close together in time as possible -- "these are moments apart and both
    # are labelled 6201; which one is it?"
    by_f = {r["f"]: r for r in rows}
    want, focus = [], {}
    for c in issues:
        a, b = c["tracks"]
        lo, hi = c["window"]
        inwin = [f for f in by_f if lo <= by_f[f]["t"] <= hi]
        fa = [f for f in inwin if a in {d["tid"] for d in by_f[f]["dets"]}]
        fb = [f for f in inwin if b in {d["tid"] for d in by_f[f]["dets"]}]
        if not fa or not fb:
            continue
        # the closest-in-time pair, so the two pictures are as comparable as possible
        best = min(((abs(by_f[x]["t"] - by_f[y]["t"]), x, y) for x in fa for y in fb),
                   default=None)
        if best is None:
            continue
        _, x, y = best
        want += [x, y]
        focus.setdefault(str(x), []).append(a)
        focus.setdefault(str(y), []).append(b)
    want = sorted(set(want))
    if not want:
        print("[clash] the conflicting tracks never share a frame -- cannot ask visually")
        return

    # CARRY IS KEYED BY TRACK, NOT BY FRAME. The obvious form -- ship the curator's
    # existing labels and let the viewer match them by (f, xy) -- cannot work here and
    # measurably did not: a clash bundle picks frames that show the conflicting pair,
    # which are deliberately NOT the frames the main bundle chose, so 0 of 106 labels
    # ever matched. Resolving them to track ids first makes them apply to whatever
    # frame this bundle happens to show.
    #
    # The focus tracks are EXCLUDED on purpose. Pre-filling them would answer the very
    # question the bundle exists to ask, and a curator confirming a pre-filled box is
    # not the same evidence as one naming it unprompted.
    carry = []
    if args.corrections and args.corrections.exists():
        from . import corrections as CO
        labels = json.loads(args.corrections.read_text(encoding="utf-8"))["labels"]
        asked = {t for tids in focus.values() for t in tids}
        by_tid: dict[int, str] = {}
        for r in CO.resolve(rows, labels):
            if r["ok"] and r.get("team") and not CO.flag_of(r):
                if r["tid"] not in asked:
                    by_tid[r["tid"]] = str(r["team"])
        carry = [{"tid": t, "team": tm} for t, tm in sorted(by_tid.items())]

    doc = CU.build_frames(stem, args.tracks, args.match, 0, 2, rows=rows,
                          frames=want, focus=focus, carry=carry,
                          note=("These frames come in pairs, moments apart. In each "
                                "pair the pipeline gave ONE team name to two different "
                                "robots -- the amber box in each frame. They cannot "
                                "both be that team: either they are on screen together, "
                                "or the robot would have had to cross the field between "
                                "them. Name the amber-ringed box in each frame -- some "
                                "frames have two. Everything else is already filled in "
                                "from your earlier answers."))
    dest = (C.STAGE3_DIR / f"{stem}_curate_clash.json"
            if str(args.clash_bundle) == "AUTO" else args.clash_bundle)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc), encoding="utf-8")
    print(f"[clash] {len(want)} frame(s) covering {len(issues)} conflict(s) "
          f"({n_kin} kinematic, {len(issues) - n_kin} custody), carrying "
          f"{len(carry)} existing label(s) -> {dest}  "
          f"({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: tracks -> 6 named robots.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--identity", type=Path, default=None)
    ap.add_argument("--step-cut", type=float, default=1.5, metavar="MULT",
                    help="cut a track where it moves further between two of its own "
                         "detections than MULT x the measured displacement bound. 0 "
                         "disables. See split_impossible_steps -- this is the only "
                         "check that can see a bad detection INSIDE a track.")
    ap.add_argument("--positions", type=Path, default=None)
    ap.add_argument("--match", required=True)
    ap.add_argument("--no-split", action="store_true",
                    help="do not cut tracks whose own votes change identity")
    ap.add_argument("--no-alliance-split", action="store_true",
                    help="do not cut tracks whose bumper hue changes alliance")
    ap.add_argument("--appear-thresh", type=float, default=0.25,
                    help="cut tracks where appearance changes by more than this "
                         "(0 = off; needs rtrack.appear; see split_on_appearance)")
    ap.add_argument("--solver", choices=("greedy", "cpsat"), default="cpsat",
                    help="one global CP-SAT assignment of tracks to teams (default), "
                         "or the legacy greedy interval colouring + Hungarian naming. "
                         "greedy names groups from OCR vote pools and cannot honour "
                         "curator pins, so it is unusable without an identity.json")
    ap.add_argument("--calib-from", default=None, metavar="VIDEO",
                    help="use another video's calibration for the field filter -- "
                         "required for clips sliced by rtrack.replay, which are named "
                         "for the match rather than the camera")
    ap.add_argument("--no-field-filter", action="store_true",
                    help="keep detections that project outside the field. The filter "
                         "needs a calibration and is a no-op without one; see "
                         "drop_offfield for what it does and does not remove")
    ap.add_argument("--no-view-filter", action="store_true",
                    help="keep detections recorded while the camera was not in its "
                         "calibrated pose. A no-op without a viewcheck run; see "
                         "drop_offview for why those detections are not merely "
                         "unprojectable but of possibly the wrong robots entirely")
    ap.add_argument("--clash-bundle", type=Path, nargs="?", const=Path("AUTO"),
                    default=None, metavar="PATH",
                    help="after solving, write a small curation bundle showing the "
                         "tracks that still CLASH -- two robots on screen together "
                         "both labelled the same team, with no crossing to explain "
                         "it. These are the only questions left that a human can "
                         "answer and the pipeline cannot.")
    ap.add_argument("--corrections", type=Path, default=None,
                    help="curator corrections JSON (see rtrack.corrections). Applied "
                         "after all heuristic splits, and overrides them.")
    ap.add_argument("--pin", action="append", default=[], metavar="TID=TEAM",
                    help="hard-constrain a track to a team, e.g. --pin 22=9644. "
                         "cpsat only; this is what a human seed becomes.")
    # WAS 60. Measured on 2026mawor_qm1 -- 61 tracks, no gallery, the hardest instance
    # seen so far -- sweeping this budget while holding everything else fixed:
    #
    #     --time-limit    wall    coverage   custody
    #           15         50 s      98%       87%
    #           25        223 s      99%       88%
    #           40        364 s      96%       85%
    #           60        383-513 s 100%       88%
    #
    # QUALITY IS NOT MONOTONIC IN EFFORT, and that is the finding that sets the
    # default. 40 is worse than 15 on both metrics while costing 7x the time. The cause
    # is the deconfliction loop: it cuts tracks that the CURRENT solution leaves
    # blocked, so a different intermediate answer produces different cuts, which
    # cascade into a different final grouping. More search buys a different answer, not
    # a better one. 15 gets within a point of the best for an eighth of the time.
    #
    # Wall time varies with machine load (the same config measured 383 s and 513 s);
    # the quality column does not, because the solver is deterministic given a budget.
    ap.add_argument("--time-limit", type=float, default=15.0, metavar="UNITS",
                    help="CP-SAT budget PER SOLVE, in DETERMINISTIC time units rather "
                         "than seconds -- it counts search work, so wall time varies "
                         "with how hard the instance is. Multiplies up: the solver runs "
                         "2 x (1 + --deconflict) times. See the measured sweep above "
                         "before raising it; more is not reliably better.")
    ap.add_argument("--appear-backend", choices=("hist", "cnn"), default="hist",
                    help="descriptor used to CUT tracks where appearance changes. "
                         "hist is the tuned 48-d histogram (0.665 single-crop AUC); "
                         "cnn is the learned embedding (0.788), which should need far "
                         "fewer cuts to catch the same switches. Threshold scales are "
                         "NOT comparable between them -- see --appear-thresh.")
    ap.add_argument("--occluders", default=None, metavar="CAMERA",
                    help="camera stem whose drawn occluders to use. Defaults to the "
                         "calibration stem when calib/<stem>_occluders.json exists, so "
                         "a camera someone has drawn is used without being asked for. "
                         "Also rejoins tracks that stop and restart at a structure "
                         "public/rtrack/occluders.html, choosing the predecessor by "
                         "APPEARANCE. Geometry finds these pairs but cannot resolve "
                         "them -- several robots use one structure over a match; see "
                         "occluder_rebind.")
    ap.add_argument("--join-check", action="store_true",
                    help="re-open stitch joins whose two sides look like different "
                         "robots. OFF by default: measured on curated qm21 it cost 5 "
                         "accuracy points, because the 0.89 threshold is calibrated on "
                         "whole-track pairs and these are cut points. See "
                         "split_chimeric_joins.")
    ap.add_argument("--app-weight", type=int, default=0, metavar="PTS",
                    help="reward two adjacent tracks sharing a team when they LOOK "
                         "like one robot, and penalise it when they do not. 0 = off. "
                         "The solver has never had an appearance term; see "
                         "solve.APP_SAME for the calibration and what it costs.")
    ap.add_argument("--hold-weight", type=int, default=600, metavar="PTS",
                    help="penalty for giving a team to a visible track while that team "
                         "is presumed behind a structure it vanished into. 0 = off. "
                         "Needs --occluders. See occlusion_holds.")
    ap.add_argument("--hold-require-exit", action="store_true",
                    help="only hold a team when something was actually seen to come "
                         "back out of that structure. A cell with no observed exit is "
                         "an assertion nothing closes.")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, metavar="S",
                    help="how long a structure keeps a team after one vanishes into it")
    ap.add_argument("--no-dup-merge", action="store_true",
                    help="do NOT fuse tracks that geometry says are one robot boxed "
                         "twice. See geometric_duplicates.")
    ap.add_argument("--dup-sep", type=float, default=DUP_SEP_M, metavar="M",
                    help="floor-contact separation below which two co-detected tracks "
                         "cannot be two robots (bumpers are ~0.9 m across).")
    ap.add_argument("--dup-iou", type=float, default=DUP_IOU, metavar="FRAC",
                    help="median box overlap required alongside --dup-sep. Genuine "
                         "pairs measured 0.00-0.01; duplicates 0.24-0.50.")
    ap.add_argument("--kin-weight", type=int, default=120, metavar="PTS",
                    help="penalty per unit of kinematic impossibility when pairing two "
                         "tracks onto one robot.")
    ap.add_argument("--kin-cap", type=float, default=30.0, metavar="MULT",
                    help="the penalty stops growing past this multiple of the distance "
                         "budget. At the default, ANY violation costs at most "
                         "kin-weight*3 = 360, which vote evidence outweighs 13:1 at "
                         "--vote-weight 200.")
    ap.add_argument("--kin-hard", type=float, default=1.5, metavar="MULT",
                    help="FORBID pairing two tracks whose separation exceeds this "
                         "multiple of the physical distance budget, rather than pricing "
                         "it. 0 = off. A robot cannot be in two places; past some "
                         "multiple that is geometry, not evidence -- the same argument "
                         "the co-detection constraint already rests on.")
    ap.add_argument("--seed", type=int, default=0,
                    help="CP-SAT random seed. Changes only search ORDER, not the model "
                         "or its optimum -- so sweeping it samples DIFFERENT solutions "
                         "of the same problem, which is how to measure whether the "
                         "objective actually discriminates a correct labelling.")
    ap.add_argument("--no-hint", action="store_true",
                    help="do NOT seed CP-SAT with the vote-implied assignment. The hint "
                         "changes neither the feasible set nor the optimum, only where "
                         "the search starts; this flag exists to measure it.")
    ap.add_argument("--opt-gap", type=float, default=0.0, metavar="FRAC",
                    help="stop each solve once proven within this fraction of optimal "
                         "(0 = prove exact optimality, the default). Unlike a wall-clock "
                         "cap this keeps the solver deterministic, because it stops on "
                         "a property of the objective rather than on a timer. Aimed at "
                         "instances carrying identity votes, which are far harder to "
                         "PROVE than to solve.")
    ap.add_argument("--deconflict", type=int, default=3,
                    help="rounds of cutting tracks that no group can accept, then "
                         "re-solving (cpsat only; 0 disables)")
    ap.add_argument("--min-piece", type=int, default=25,
                    help="a deconfliction cut must isolate at least this many "
                         "detections, else the track is left whole")
    # Coverage/purity dial, and the tradeoff is real. MEASURED, all proven optimal,
    # against the greedy baseline of cov 83% / 2 robots >=70% / mean 48% / 1 alliance
    # contradiction:
    #   park    coverage  robots >=70%  worst  mean  alliance-bad  uncertain
    #     40        66%             2    17%   53%             0          6
    #    300        80%             2    18%   56%             0         12
    #    700        86%             1     3%   36%             1          8
    # 300 matches greedy's coverage while adding 8 points of mean share and removing
    # the alliance contradiction. Past ~700 the extra coverage is bought by forcing
    # evidence-free tracks into some robot, which is not coverage worth having.
    # MEASURED: leave this soft. Once split_on_alliance makes tracks alliance-pure it
    # is tempting to make alliance a hard constraint, and it is worse -- at 2000 and
    # at 20000 the result is identical and both are worse than 250: coverage 86% ->
    # 79%, mean vote share 59% -> 52%. Bumper hue is only ~85-90% reliable per track,
    # so forcing it throws away good tracks whose colour was misread.
    ap.add_argument("--alli-weight", type=float, default=250.0,
                    help="cost of assigning a track to a team on the other alliance "
                         "(cpsat only); see the note above before raising it")
    ap.add_argument("--vote-weight", type=int, default=200, metavar="PTS",
                    help="points per identity vote. Measured on qm21: at the default "
                         "10 the vote terms are 8.7%% of the objective magnitude "
                         "against 87%% for pair+park, and even a PERFECT descriptor "
                         "caps identity at 20%%. Parity with the structural terms "
                         "needs ~41. Raise this rather than MAX_VOTES: the vote cap "
                         "counts EVIDENCE (reid saturates ~40 crops), so inflating it "
                         "double-counts the same observation, whereas this is honestly "
                         "a weight.")
    ap.add_argument("--park", type=float, default=300.0,
                    help="cost of leaving a track unassigned; raise for coverage, "
                         "lower for purity (cpsat only)")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    rows = load_tracks(args.tracks)
    # Kept in lockstep with prepare_tracks, in this order. The two must segment
    # identically or every cross-boundary inference compares two id spaces -- see the
    # note in prepare_tracks.
    if not args.no_view_filter:
        rows, n_v, n_t = drop_offview(rows, stem)
        if n_v:
            print(f"[robots] dropped {n_v}/{n_t} detection(s) recorded while the "
                  f"camera was not in its calibrated view (see rtrack.viewcheck)")
    if not args.no_field_filter:
        rows, n_off, n_tot = drop_offfield(rows, stem,
                                           calib_stem=args.calib_from)
        if n_off:
            print(f"[robots] dropped {n_off}/{n_tot} detection(s) projecting "
                  f"outside the field (+{FIELD_SLACK_M} m)")
        elif not (C.CALIB_DIR / f"{args.calib_from or stem}.json").exists():
            print(f"[robots] no calibration for {args.calib_from or stem} -- "
                  f"field filter INACTIVE")
    ip = args.identity or (C.STAGE3_DIR / f"{stem}_identity.json")
    # OCR is off the live path. identity.json is honoured when a previous run left one
    # behind, but its absence is the normal case now, not an error: measured, blanking
    # it changed nothing once curator corrections were loaded (97% coverage, 76% mean
    # custody, 0 conflicts, 11 cuts either way). See OCR_PLAN.md for why.
    ident = (json.loads(ip.read_text(encoding="utf-8")) if ip.exists()
             else {"video": stem, "tracks": {}})
    pp = args.positions or (C.STAGE2_DIR / f"{stem}_positions.json")
    positions = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else None

    # Use the camera's drawn occluders when they exist. The watcher and the pipeline
    # pass neither this nor the weights, so anything left opt-in is effectively off in
    # the one path that matters -- a curator's answer coming back and being re-solved
    # unattended. Everything defaulted here was measured against curator labels.
    if args.occluders is None:
        from .occluders import path_for as _occ_path
        _cam = args.calib_from or stem
        if _occ_path(_cam).exists():
            args.occluders = _cam
            print(f"[robots] using drawn occluders for {_cam}")

    m = tba_mod.match_by_key(args.match)
    red, blue = [str(t) for t in m["red"]], [str(t) for t in m["blue"]]
    print(f"[robots] red {red}  blue {blue}")

    # BEFORE ANY SPLIT, and that position is the whole point. On the stitched tracks a
    # duplicate is one long-lived pair -- qm24's worst shared 304 frames. After the 95
    # appearance splits it is shattered into a dozen short-lived fragment pairs, each
    # individually below any sane frame threshold and indistinguishable from noise.
    # Run late, this merged 9 pairs while 57 pairs still came within 1.0 m of each other
    # in the output; run early, it sees the evidence intact.
    #
    # It also has to precede the splits for a second reason: split_on_appearance cuts a
    # track when its descriptor jumps, and a box that slides between a robot's
    # superstructure and its lower body IS such a jump. Feeding the splitter duplicates
    # manufactures fragments it then has to have cut.
    pos_at = positions_by_det(positions, args.tracks)
    if not args.no_dup_merge:
        rows, dups = geometric_duplicates(rows, pos_at, args.dup_sep, args.dup_iou)
        if dups:
            print(f"[robots] {len(dups)} duplicate track(s) merged -- one robot boxed "
                  f"twice (low/high), which would otherwise force a second team on it:")
            for mg in dups:
                print(f"[robots]   #{mg['absorbed']} -> #{mg['keep']}: "
                      f"{mg['dxWidths']} box widths apart horizontally, "
                      f"IoU {mg['iou']}, {mg['frames']} shared frame(s)")
            tks = ident.setdefault("tracks", {})
            for mg in dups:
                src = tks.pop(str(mg["absorbed"]), None)
                if not src:
                    continue
                dst = tks.setdefault(str(mg["keep"]), {"tally": {}, "voteList": []})
                tal = dst.setdefault("tally", {})
                for team, n in (src.get("tally") or {}).items():
                    tal[team] = tal.get(team, 0) + n
                dst.setdefault("voteList", []).extend(src.get("voteList") or [])

    # Rejoin across the structures a human marked, BEFORE any split -- same reason as
    # the duplicate merge above: afterwards the evidence is shattered and the tids no
    # longer match the cached descriptors.
    if args.occluders:
        from .occluders import load as _occ_load, to_pixels as _occ_px
        from .embed import load_head as _lh
        _doc = _occ_load(args.occluders)
        if _doc is None:
            print(f"[robots] no occluder file for {args.occluders} -- nothing to rebind")
        else:
            _regs = _occ_px(_doc, (1920, 1080))
            rows, rb = occluder_rebind(
                rows, _regs, C.STAGE3_DIR / f"{stem}_appearance_cnn.npz",
                head=_lh(str(args.match).split("_")[0]), ident=ident)
            if rb:
                print(f"[robots] {len(rb)} track(s) rejoined across a marked structure "
                      f"using appearance:")
                for mg in rb:
                    ru = f", runner-up {mg['runnerUp']}" if mg["runnerUp"] else ""
                    print(f"[robots]   #{mg['absorbed']} -> #{mg['keep']} at "
                          f"{mg['region']}: gap {mg['gapS']}s, cos {mg['cos']}{ru}")
            else:
                print(f"[robots] no track pairs met the rebind test at "
                      f"{len(_regs)} marked structure(s)")

    # The state the holding-cell events are derived from: after the duplicate merge
    # (so two boxes on one robot are already one track) but before ANY split (so a
    # track ending still means a robot disappeared). Re-reading the stitched file
    # instead was wrong -- the merge has since absorbed tracks, so two stitched ids
    # could map to one fragment and a cell reported itself as its own emergence.
    pre_split_rows = [{"f": r["f"], "t": r["t"],
                       "dets": [dict(d) for d in r["dets"]]} for r in rows]

    # Undo bad stitch joins BEFORE anything else reasons about these tracks: a chimera
    # left whole poisons the descriptors, the votes and the assignment alike.
    if args.join_check:
        from .embed import load_head as _lh3
        rows, n_j = split_chimeric_joins(
            rows, C.STAGE3_DIR / f"{stem}_appearance_cnn.npz",
            head=_lh3(str(args.match).split("_")[0]))
        if n_j:
            print(f"[robots] {n_j} stitch join(s) cut -- two robots had been joined "
                  f"into one track")

    if not args.no_alliance_split:
        rows, n_a, orig_a = split_on_alliance(rows)
        if n_a:
            ident = retally(ident, rows, orig_a)
            print(f"[robots] split {n_a} segment(s) where a track's bumper HUE "
                  f"changed alliance ({len(orig_a) - n_a} tracks affected)")

    if args.appear_thresh > 0:
        cnn_split = args.appear_backend == "cnn"
        ap_p = C.STAGE3_DIR / (f"{stem}_appearance_cnn.npz" if cnn_split
                               else f"{stem}_appearance.npz")
        _hd = None
        if cnn_split:
            from .embed import load_head as _lh
            # "2026necmp1_qm24" -> "2026necmp1"; the head is per event.
            _hd = _lh(str(args.match).split("_")[0])
        rows, n_p, orig_p = split_on_appearance(
            rows, ap_p, args.appear_thresh, metric=("cosine" if cnn_split
                                                    else "hellinger"), head=_hd)
        if n_p:
            ident = retally(ident, rows, orig_p)
            print(f"[robots] split {n_p} segment(s) on APPEARANCE change "
                  f"(threshold {args.appear_thresh})")
        elif not ap_p.exists():
            print(f"[robots] {ap_p} missing -- run rtrack.appear first")

    if not args.no_split:
        rows, n_split, orig_of = split_chimeras(rows, ident)
        if n_split:
            before = sum(len(v.get("voteList", [])) for v in ident["tracks"].values())
            ident = retally(ident, rows, orig_of)
            after = sum(len(v.get("voteList", [])) for v in ident["tracks"].values())
            print(f"[robots] split {n_split} chimeric segment(s) where a track's own "
                  f"votes changed identity")
            if before:
                print(f"[robots] re-attributed votes to segments by timestamp "
                      f"({before} -> {after} votes across {len(orig_of)} segments)")
            else:
                print("[robots] identity.json has no voteList -- votes stay on "
                      "segment 0 (legacy file; OCR is no longer on the live path)")
    # WITHIN-TRACK kinematic cuts, after the appearance and vote splits and before the
    # curator's. A track that teleports inside itself is not a robot for its whole
    # length, and every constraint downstream assumes it is.
    if args.step_cut > 0 and pos_at:
        rows, n_step = split_impossible_steps(rows, pos_at, args.step_cut)
        if n_step:
            print(f"[robots] cut {n_step} track(s) where the track's OWN motion was "
                  f"impossible (> {args.step_cut}x the measured displacement bound)")

    # Curator corrections LAST, so they override every heuristic above rather than
    # being argued with by one. A cut a person asked for is better evidence than any
    # threshold we tuned.
    pinned: dict[int, str] = {}
    preferred: dict[int, str] = {}
    clashes: list[dict] = []
    mixed: set[int] = set()
    # NOT `flags` -- that name is taken below by the per-group chimera flags, and
    # letting them collide silently wrote the wrong object into the output file.
    cflags: dict[int, str] = {}
    if args.corrections:
        from . import corrections as CO
        doc = CO.load(args.corrections)
        resolved = CO.resolve(rows, doc["labels"])
        cuts = CO.cuts_from(resolved)
        n_cuts = sum(len(v) for v in cuts.values())
        if cuts:
            rows, _n, orig_c2 = _apply_cuts(rows, cuts)
            ident = retally(ident, rows, orig_c2)
            resolved = CO.resolve(rows, doc["labels"])   # ids moved; re-resolve
        pinned, cflags = CO.pins_from(resolved)
        mixed = CO.dropped(cflags)
        CO.report(resolved, pinned, cflags, n_cuts)

        # A clash between two tracks that physically CROSS is almost certainly the
        # tracker having swapped them there, not a bad label. Cut both at the crossing
        # so each pin lands on a piece that no longer overlaps the other, and the
        # conflict dissolves without discarding anybody's answer.
        for _ in range(2):
            lt: dict[int, list[float]] = defaultdict(list)
            for r_ in resolved:
                if r_["ok"] and r_.get("team") and not CO.flag_of(r_):
                    lt[r_["tid"]].append(r_["t"])
            xc, found = CO.crossing_cuts(pinned, rows, lt)
            if not xc:
                break
            rows, n_x, orig_x = _apply_cuts(rows, xc)
            ident = retally(ident, rows, orig_x)
            resolved = CO.resolve(rows, doc["labels"])
            pinned, cflags = CO.pins_from(resolved)
            mixed = CO.dropped(cflags)
            print(f"[corrections] {len(found)} clash(es) explained by a track CROSSING "
                  f"-- cut both tracks there instead of discarding a label:")
            for fnd in found:
                print(f"[corrections]   {fnd['team']}: tracks {fnd['tracks']} cross at "
                      f"t{fnd['atS']}s ({fnd['closeFrames']} close frames)")

        # BEFORE split_conflicts, deliberately: a same-team co-detection whose boxes sit
        # on top of each other is ONE robot the detector fired on twice, not a curator
        # error. Fusing it here means split_conflicts only ever sees the clashes that
        # really are two robots. The deconflict loop below re-derives pins from these
        # already-merged rows, so it needs no second pass.
        rows, merges = CO.merge_duplicates(pinned, rows)
        if merges:
            resolved = CO.resolve(rows, doc["labels"])
            pinned, cflags = CO.pins_from(resolved)
            mixed = CO.dropped(cflags)
            print(f"[corrections] {len(merges)} duplicate detection(s) merged -- one "
                  f"robot boxed twice, both boxes given the same team by the curator:")
            for mg in merges:
                print(f"[corrections]   {mg['team']}: track {mg['absorbed']} folded "
                      f"into {mg['keep']} ({mg['frames']} shared frame(s), centres "
                      f"{mg['sepWidths']} box widths apart)")

        pinned, preferred, clashes = CO.split_conflicts(pinned, rows, ident)
        if clashes:
            print(f"[corrections] {len(clashes)} label clash(es) -- tracks pinned to "
                  f"the SAME team that are on screen together. At most one of each "
                  f"can be right; the rest are demoted to preferences:")
            for c in clashes:
                sup = ", ".join(f"#{k} ({v[0]} votes, {v[1]} dets)"
                                for k, v in c["support"].items())
                print(f"[corrections]   {c['team']}: kept #{c['kept']}, demoted "
                      f"{c['demoted']} -- together in {c['frames']} frames  [{sup}]")

    # Built once from the tracks file on disk, then reused by every rebuild below.
    # The deconflict loop recomputes `info` after each round of cuts, and the FINAL
    # answer comes from the last of those -- so a fix applied only to the first call
    # would leave the result untouched. That is exactly what happened on the first
    # attempt here.
    info = track_info(rows, positions, pos_at=pos_at)
    con = conflicts(rows)

    # SAME-TEAM PINS SEPARATED BY AN IMPOSSIBLE TRANSITION.
    #
    # split_conflicts above resolves same-team pins that are CO-DETECTED -- two tracks
    # on screen together, so at most one can carry the team. It cannot see a pair that
    # never shares a frame yet still cannot be one robot, because the transition between
    # them is physically impossible.
    #
    # Measured on 2026mawor_qm5: the curator named 9644 at t=76.7 s (inside track 217)
    # and again at t=83.8 s (inside track 29). Both labels are defensible on their own
    # crop. But 217 ends at (5.5, 1.3) and 29 begins at (12.1, 7.9) one sample later --
    # 9.3 m in 0.067 s. Neither track has an internal discontinuity worth the name
    # (worst internal step 0.39 m and 0.52 m), so they are two clean tracks of two
    # DIFFERENT robots, and one of the two labels is on the wrong one.
    #
    # Cutting is the wrong remedy here precisely because neither track is chimeric;
    # there is nothing to cut. The right one is the same as split_conflicts': keep the
    # better-supported pin, demote the other to a preference, and say which. Without
    # this the solve reaches the kinematic exclusion, finds both tracks pinned to one
    # team, and applies the curator override -- the human wins, and the route teleports.
    if args.kin_hard > 0:
        from .solve import pair_forbidden as _pf, paths_forbidden as _pathsf
        _byteam = defaultdict(list)
        for _t, _tm in pinned.items():
            if _t in info and "start" in info[_t]:
                _byteam[_tm].append(_t)
        _imp = []
        for _tm, _ts in _byteam.items():
            _ts.sort(key=lambda t: info[t]["t0"])
            for _i, _a in enumerate(_ts):
                for _b in _ts[_i + 1:]:
                    if info[_a]["t1"] <= info[_b]["t0"]:
                        _bad = _pf(info[_a], info[_b], args.kin_hard)
                    elif info[_b]["t1"] <= info[_a]["t0"]:
                        _bad = _pf(info[_b], info[_a], args.kin_hard)
                    elif _b in con.get(_a, ()):
                        continue          # co-detected: split_conflicts owns this one
                    else:
                        _bad = _pathsf(info[_a], info[_b], args.kin_hard)
                    if not _bad:
                        continue
                    # Keep the one with more evidence behind it. Detection count is the
                    # honest proxy here: a curator's label on a 250-detection track has
                    # far more of the match standing behind it than the same label on a
                    # fragment, and both pins are otherwise identical in kind.
                    _keep, _drop = ((_a, _b) if info[_a]["n"] >= info[_b]["n"]
                                    else (_b, _a))
                    if _drop not in pinned:
                        continue
                    del pinned[_drop]
                    preferred[_drop] = _tm
                    _imp.append((_tm, _keep, _drop, info[_keep]["n"], info[_drop]["n"]))
        if _imp:
            print(f"[corrections] {len(_imp)} same-team pin pair(s) separated by a "
                  f"KINEMATICALLY IMPOSSIBLE transition -- not co-detected, so "
                  f"split_conflicts could not see them. Weaker pin demoted to a "
                  f"preference:")
            for _tm, _k, _d, _nk, _nd in _imp:
                print(f"[corrections]   {_tm}: kept #{_k} ({_nk} dets), demoted "
                      f"#{_d} ({_nd} dets)")
    frag_emb = {}
    if args.app_weight > 0:
        from .embed import load_head as _lh2
        frag_emb = fragment_embeddings(
            pre_split_rows, rows, C.STAGE3_DIR / f"{stem}_appearance_cnn.npz",
            head=_lh2(str(args.match).split("_")[0]))
        print(f"[robots] appearance term on {len(frag_emb)}/{len(info)} fragment(s)")
    holds_x, holds_e = [], []
    if args.occluders and args.hold_weight > 0:
        from .occluders import load as _ol, to_pixels as _op
        _d = _ol(args.occluders)
        if _d:
            _rg = _op(_d, (1920, 1080))
            # The STITCHED tracks, re-read from disk: `rows` has been split since, and a
            # fragment ending is not a disappearance. See hold_cells.
            holds_x, holds_e, _cells = hold_cells(
                pre_split_rows, rows, _rg, max_hold_s=args.hold_s,
                require_exit=args.hold_require_exit)
            print(f"[robots] {len(_cells)} holding-cell event(s) over {len(_rg)} "
                  f"structure(s): {len(holds_x)} exclusion(s), "
                  f"{len(holds_e)} emergence pair(s)")
            for S, t1, t2, tail, head in _cells:
                print(f"[robots]   {S}: held {t1}-{t2}s ({t2 - t1:.1f}s) from #{tail}"
                      + (f", out as #{head}" if head is not None else ""))
    # Denominator BEFORE dropping anything, so coverage stays comparable run to run.
    # Excluding curator-rejected tracks from both halves of the fraction would make
    # every rejection look like an improvement.
    all_dets = sum(info[t]["n"] for t in info)
    # A track the curator called mixed or not-a-robot is dropped from grouping. It
    # keeps its detections in the output, it just stops claiming to be one robot.
    # `unknown` is deliberately NOT dropped: a human failing to identify a robot is
    # not evidence that it is absent, and the solver may still place it from geometry.
    for tid in mixed:
        info.pop(tid, None)

    unstable: set[int] = set()
    if args.solver == "cpsat":
        from . import solve as S
        for spec in args.pin:
            tid_s, _, team = spec.partition("=")
            pinned[int(tid_s)] = team
        S.KIN_CAP = args.kin_cap
        W = S.Weights(park=args.park, alli=args.alli_weight, vote=args.vote_weight,
                      kin=args.kin_weight, hold=args.hold_weight, app=args.app_weight)

        def run_solve(limit):
            a, t, m, u = S.solve(info, con, ident, red, blue, pinned or None,
                                 limit, W, preferred=preferred or None,
                                 opt_gap=args.opt_gap, hint=not args.no_hint, seed=args.seed,
                                 kin_hard=args.kin_hard, holds=(holds_x, holds_e),
                                 emb=frag_emb or None)
            # UNKNOWN and INFEASIBLE are opposite problems and must not share a fate.
            # INFEASIBLE is proved: the constraints cannot all hold, and more time will
            # only prove it again -- fail now. UNKNOWN means the budget ran out before
            # ANY solution was found, which more time genuinely fixes.
            #
            # These sit right on the edge rather than being structurally hard: on
            # 2026necmp1, qm16 solved at 131 tracks / 587 conflict pairs while qm23
            # failed at 118 / 589. Four matches of a fourteen-match rebuild died at the
            # default limit, which is too brittle for a difference that small.
            if a is None and (m.get("status") or "").upper() != "INFEASIBLE":
                longer = limit * 4
                print(f"[robots] CP-SAT {m.get('status')} at {limit}s "
                      f"-- retrying once at {longer}s")
                a, t, m, u = S.solve(info, con, ident, red, blue, pinned or None,
                                     longer, W, preferred=preferred or None,
                                     opt_gap=args.opt_gap, hint=not args.no_hint, seed=args.seed,
                                 kin_hard=args.kin_hard, holds=(holds_x, holds_e),
                                 emb=frag_emb or None)
            if a is None:
                raise SystemExit(f"[robots] CP-SAT found no solution ({m['status']})")
            return a, t, m, u

        assign_k, teams_order, meta, unstable = run_solve(args.time_limit)
        print(f"[robots] CP-SAT {meta['status']} in {meta['wall']:.2f}s "
              f"({meta['vars']} vars, {meta['pairs']} pair terms); "
              f"objective {meta['objective']:.0f}, bound {meta['bound']:.0f}")

        # Deconfliction rounds. A track co-detected with a member of every group is
        # unplaceable AS A WHOLE even when a slot is free at every one of its own
        # frames -- see plan_deconflict_cuts. Cut those at the boundary of a stretch
        # some group can take, then solve again with the pieces.
        for rnd in range(args.deconflict):
            groups = S.groups_from(assign_k, len(teams_order))
            blocked = [t for t in meta["parked"]
                       if t in info and info[t]["n"] >= args.min_piece
                       and all(con.get(t, set()) & set(ms) for ms in groups if ms)]
            if not blocked:
                break
            cuts = plan_deconflict_cuts(rows, groups, blocked, args.min_piece)
            if not cuts:
                print(f"[robots] deconflict round {rnd + 1}: "
                      f"{len(blocked)} blocked track(s), none has a placeable stretch")
                break
            n_before = sum(info[t]["n"] for t in cuts)
            rows, n_new, orig_d = _apply_cuts(rows, cuts)
            ident = retally(ident, rows, orig_d)
            # Track ids moved, so every curator pin must be re-derived against them.
            if args.corrections:
                resolved = CO.resolve(rows, doc["labels"])
                pinned, cflags = CO.pins_from(resolved)
                mixed = CO.dropped(cflags)
                pinned, preferred, clashes = CO.split_conflicts(pinned, rows, ident)
            info = track_info(rows, positions, pos_at=pos_at)
            con = conflicts(rows)
            for tid in mixed:
                info.pop(tid, None)
            assign_k, teams_order, meta, unstable = run_solve(args.time_limit)
            still = sum(info[t]["n"] for t in meta["parked"] if t in info)
            print(f"[robots] deconflict round {rnd + 1}: cut {len(cuts)} blocked "
                  f"track(s) ({n_before} dets) into {n_new} extra piece(s); "
                  f"{still} dets still parked")

        groups = S.groups_from(assign_k, len(teams_order))
        gvotes = S.pooled_votes(groups, ident)
        names = {gi: teams_order[gi] for gi in range(len(groups))}
        flags = S.timeline_flags(groups, ident)
        extra = meta["parked"]
        if "altGap" in meta:
            print(f"[robots] second-best solution is {meta['altGap']:.0f} worse; "
                  f"{len(unstable)} track(s) differ between them")
    else:
        if not any(v.get("tally") for v in ident.get("tracks", {}).values()):
            raise SystemExit(
                "[robots] --solver greedy names groups from OCR vote pools, and this "
                "run has no votes (OCR is off the live path). Use --solver cpsat.")
        if pinned:
            print(f"[robots] WARNING: {len(pinned)} pin(s) IGNORED -- the greedy "
                  f"colouring cannot honour them. Use --solver cpsat.")
        groups, gvotes, assign, extra = colour(info, con, ident)
        names = name_groups(groups, gvotes, info, red, blue)
        flags = verify(groups, ident)

    print(f"\n[robots] {len(info)} tracks -> {len(groups)} groups"
          f"{f', {len(extra)} parked' if extra else ''}\n")
    total_n = all_dets
    named_n = 0
    for gi, members in enumerate(groups):
        team = names.get(gi, "?")
        votes = gvotes[gi]
        tot = sum(votes.values())
        share = (votes.get(team, 0) / tot) if tot else 0.0
        n = sum(info[t]["n"] for t in members)
        named_n += n
        alli = Counter(info[m_]["alliance"] for m_ in members if info[m_]["alliance"])
        print(f"  robot {gi}: {team:>5} ({alli.most_common(1)[0][0] if alli else '?'})"
              f"  {n:>5} dets, {tot:>3} votes, {share:.0%} for {team}")
        shown = ", ".join(f"{t}*" if t in unstable else str(t)
                          for t in sorted(members))
        print(f"      tracks [{shown}]")
        print(f"      votes  {dict(votes)}")
        if flags.get(gi):
            print(f"      *** VOTES CHANGE OVER TIME {flags[gi]} -- possible chimera")
    if extra:
        print(f"\n  parked (more than {N_ROBOTS} co-detected): {extra}")
    if unstable:
        # These are the ONLY decisions worth a human's attention: everything else is
        # settled by the evidence, and re-deciding it by hand changes nothing.
        by_size = sorted(unstable, key=lambda t: -info[t]["n"])
        print(f"\n  * = differs in the second-best solution ({len(unstable)} tracks, "
              f"{sum(info[t]['n'] for t in unstable)} detections). Ask about these:")
        for t in by_size[:10]:
            v = ident["tracks"].get(str(t), {}).get("tally", {})
            print(f"      track {t:>3}: {info[t]['n']:>5} dets, "
                  f"t {info[t]['t0']:.0f}-{info[t]['t1']:.0f}s, votes {v or '{}'}")

    print(f"\n[robots] {named_n}/{total_n} detections in named robots "
          f"({100*named_n/total_n:.0f}%)")
    missing = [t for t in red + blue if t not in names.values()]
    if missing:
        print(f"[robots] teams not matched to any group: {missing}")

    # Also emit the (possibly split) tracks with their team attached, so the overlay
    # can render team numbers directly. The split creates ids that exist nowhere else,
    # so a bare tid->team map would not be resolvable downstream.
    tid_team = {}
    for gi, members in enumerate(groups):
        for m in members:
            tid_team[m] = names.get(gi)
    # CURATOR ALIGNMENT -- the one statistic on this pipeline that cannot be gamed.
    # Every other number (custody, coverage, label churn, impossible transitions) is
    # computed over the solver's OWN output, so a run that labels less can score better
    # on several at once. Measured repeatedly, those proxies pointed the OPPOSITE way to
    # truth: an appearance term that cut label churn 46% cost 10-12 points here, and a
    # stitch change that cut fragments 19% cost 5. Reported whenever corrections exist,
    # so it sits in the log beside the numbers it should be trusted over.
    if args.corrections and args.corrections.exists():
        try:
            _hl = [l for l in json.loads(
                       args.corrections.read_text(encoding="utf-8"))["labels"]
                   if l.get("src") == "human" and l.get("team")]
            _byf = defaultdict(list)
            for _r in rows:
                _byf[_r["f"]].append(_r)
            _hit = _miss = _unres = 0
            for _lab in _hl:
                _b, _bd = None, 1e9
                for _r in _byf.get(_lab["f"], ()):
                    for _d in _r["dets"]:
                        _x1, _y1, _x2, _y2 = _d["xyxy"]
                        _dd = float(np.hypot((_x1 + _x2) / 2 - _lab["xy"][0],
                                             (_y1 + _y2) / 2 - _lab["xy"][1]))
                        if _dd < _bd:
                            _b, _bd = _d, _dd
                if _b is None or _bd > 90:
                    _unres += 1
                # tid_team, NOT d["team"] -- the teams are written onto the
                # detections in the loop below this, so reading them here compares
                # every label against None and reports 0%.
                elif str(tid_team.get(_b["tid"]) or "") == str(_lab["team"]):
                    _hit += 1
                else:
                    _miss += 1
            if _hit + _miss:
                print(f"[robots] CURATOR ALIGNMENT "
                      f"{100 * _hit / (_hit + _miss):.0f}%  "
                      f"({_hit} agree, {_miss} differ"
                      + (f", {_unres} unresolved" if _unres else "") + ")")
        except Exception as _e:
            print(f"[robots] curator alignment not computed ({type(_e).__name__})")

    lab = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
    with lab.open("w", encoding="utf-8") as fh:
        for r in rows:
            for d in r["dets"]:
                d["team"] = tid_team.get(d["tid"])
            fh.write(json.dumps(r) + "\n")
    print(f"-> {lab}")

    issues = custody_conflicts(rows)
    if issues:
        print(f"\n[robots] *** {len(issues)} CUSTODY CONFLICT(S): one team on two "
              f"tracks that are far apart at the same time")
        for c in issues:
            print(f"[robots]     {c['team']}: tracks {c['tracks']} over "
                  f"{c['window'][0]}-{c['window'][1]}s, {c['dets'][0]}/{c['dets'][1]} "
                  f"detections, centres {c['sepPx']} px apart "
                  f"(box {c['boxPx']} px) -- these are two robots")
    else:
        print("\n[robots] custody check: no team held by two separated tracks at once")

    win = match_window(stem, rows)
    cust = custody(rows, win)
    n_frames = len({r["t"] for r in rows
                    if not win or win[0] <= r["t"] <= win[1]})
    scope = (f"the {MATCH_SECONDS:.0f}s match ({win[0]:.1f}-{win[1]:.1f}s)" if win
             else "the TRACKED SPAN -- *** NOT COMPARABLE to other matches, the "
                  "denominator includes pre/post-match dead time ***")
    print(f"\n[robots] CUSTODY over {scope} -- share of its {n_frames} processed "
          f"frames in which each robot is tracked at all:")
    print(f"    {'team':>6}{'held':>8}{'frames':>8}{'gaps':>6}{'lost':>8}"
          f"{'longest':>9}   span")
    for t in sorted(cust, key=lambda k: -cust[k]["pct"]):
        c = cust[t]
        print(f"    {t:>6}{c['pct']:7.0f}%{c['frames']:>8}{c['gaps']:>6}"
              f"{c['lostS']:>7.0f}s{c['longestGapS']:>8.1f}s   "
              f"{c['firstS']:.0f}-{c['lastS']:.0f}s")
    if cust:
        mean = sum(c["pct"] for c in cust.values()) / len(cust)
        print(f"    {'mean':>6}{mean:7.0f}%   "
              f"({len(cust)}/6 robots present at all)")

    if args.clash_bundle is not None:
        _write_clash_bundle(stem, args, rows, clashes, red, blue, positions)

    out = C.STAGE3_DIR / f"{stem}_robots.json"
    out.write_text(json.dumps(
        {"video": stem, "teams": {"red": red, "blue": blue},
         "solver": args.solver,
         "groups": {str(gi): {"team": names.get(gi), "tracks": sorted(ms),
                              "votes": dict(gvotes[gi]),
                              "voteChanges": flags.get(gi, []),
                              "uncertain": sorted(t for t in ms if t in unstable)}
                    for gi, ms in enumerate(groups)},
         "uncertain": sorted(unstable),
         # Recorded so detector false positives can be COUNTED rather than merely
         # excluded, and so a later pass knows which tracks a human has already
         # failed to identify instead of asking again.
         "curator": {"pinned": {str(k): v for k, v in pinned.items()},
                     "preferred": {str(k): v for k, v in preferred.items()},
                     "flags": {str(k): v for k, v in cflags.items()},
                     "clashes": clashes},
         "custodyWindow": list(win or []),
         "custodyWindowSource": ("positions" if (C.STAGE2_DIR / f"{stem}_positions.json").exists()
                                 else ("pixel-motion" if win else "NONE -- tracked span")),
         "custody": cust,
         "custodyConflicts": custody_conflicts(rows),
         "parked": extra}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
