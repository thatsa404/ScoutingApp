"""Merge track fragments that the online tracker split.

Tuning BoT-SORT bottoms out around 21 ids for 6 robots. The residual failure is
structural: while a robot is occluded, the Kalman filter extrapolates its velocity,
so the predicted box walks off the robot. When the robot reappears -- sometimes only
5 px from where it vanished -- IoU with the prediction is zero and no threshold can
rescue the match. The tracker spawns a new id instead.

Offline we do not have to guess in real time. We can look at a fragment's death and
a later fragment's birth together and ask whether one robot could plausibly have
done both. That is the plan's Stage 3 identity propagation, applied one level down
to raw tracks, and it uses the same gates:

  1. alliance gate   -- classes must agree (undecided is a wildcard)
  2. spatial gate    -- reachable at <= ROBOT_MAX_SPEED, given the gap
  3. uniqueness      -- merge only when exactly ONE candidate survives
  4. no overlap      -- fragments that coexist are different robots, never merged

Ambiguity is left unmerged on purpose. A wrong merge silently teleports a robot's
history; an unmerged fragment is an honest gap that Stage 3 can surface for a human.

    uv run -m rtrack.stitch out/stage1/G_loose.jsonl --out out/stage1/G_stitched.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C

# Field is 16.46 m wide across roughly 1800 px of a 1920-wide broadcast frame.
# Refine per-video once Stage 2 gives a real homography; until then this is the
# honest conversion and it is only used for a reachability bound.
PX_PER_M = 109.0
MAX_GAP_FRAMES = 90     # processed frames (~6 s)
MARGIN_PX = 60.0        # slack for box-centre jitter between fragments

# The budget still grows linearly with the gap, so cap it: past roughly a field width
# the gate stops discriminating and a "merge" is a guess. Measured consequence of
# leaving it unconstrained on the full match -- merging INTRODUCED 3 teleports and a
# 125 px/frame jump the raw tracks did not have. A wrong merge silently rewrites a
# robot's history, which is worse than an honest gap.
HARD_CAP_PX = 900.0

# STATIONARY REAPPEARANCE. The gap cap above exists because the distance budget grows
# with the gap, so at long gaps the reachability test stops discriminating and a merge
# becomes a guess. That argument is about the BUDGET growing -- it says nothing about a
# robot that did not move.
#
# Measured on 2026necmp1_qm24: of 17 fragment births the stitcher refused to join, 13
# were blocked by the 6 s cap alone, and their best candidate sat 2-79 px away against a
# median robot box of 128 px. #6 -> #88 reappeared 13 px from where it vanished after
# 137 s; #6 -> #8, 2 px after 8 s. Those are robots waiting out an occlusion, and the
# evidence gets STRONGER the longer they hold still, not weaker.
#
# So the long-gap path is gated on absolute stillness rather than on a grown budget: the
# reappearance must be within roughly one robot width of the disappearance. The alliance
# and uniqueness gates still apply, and the HARD_CAP is untouched for the normal path.
STILL_PX = 0.0            # DEFAULT OFF -- see below
"""Reappearance within this many pixels is treated as the same robot despite a gap
past MAX_GAP_FRAMES. DISABLED by default because it measured badly.

It was built on the observation that 13 of 17 blocked handoffs on qm24 had their best
candidate 2-79 px away, and it does join those. But "same pixels" is not "same robot":
it cannot represent a robot that transits an occluder and emerges elsewhere, and it
fuses two robots that reuse one spot. Measured end to end -- qm21 +11 accuracy points,
qm16 -2, qm20 -5, mean +1.3 with sd 8.5, i.e. nothing -- and on qm24 the giveaway was
reid vote agreement collapsing from 3 tracks at >=80% to ZERO, which is what a track
spanning two different robots looks like.

rtrack.occluders supersedes it: a drawn structure knows its own edges, so a track that
vanishes at one can be rebound to a birth at ANY of them. Kept and switchable rather
than deleted, because the pixel test is still the right fallback for a camera nobody
has drawn."""
STILL_MAX_GAP_S = 150.0   # beyond this even stillness is not evidence


@dataclass
class Frag:
    tid: int
    frames: list[int]
    xy: list[tuple[float, float]]
    alliance: str | None
    conf: float

    @property
    def f0(self) -> int: return self.frames[0]
    @property
    def f1(self) -> int: return self.frames[-1]
    @property
    def start(self) -> tuple[float, float]: return self.xy[0]
    @property
    def end(self) -> tuple[float, float]: return self.xy[-1]
    def __len__(self) -> int: return len(self.frames)


def load_frags(rows: list[dict]) -> tuple[dict[int, Frag], int]:
    step = (rows[1]["f"] - rows[0]["f"]) if len(rows) > 1 else 1
    acc: dict[int, list] = defaultdict(list)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            acc[d["tid"]].append((r["f"], (x1 + x2) / 2, y2,
                                  d.get("alliance"), d.get("conf", 0.0)))

    frags: dict[int, Frag] = {}
    for tid, pts in acc.items():
        pts.sort()
        votes = Counter(p[3] for p in pts if p[3])
        frags[tid] = Frag(
            tid=tid,
            frames=[p[0] for p in pts],
            xy=[(p[1], p[2]) for p in pts],
            alliance=votes.most_common(1)[0][0] if votes else None,
            conf=float(np.mean([p[4] for p in pts])),
        )
    return frags, step


def alliance_ok(a: Frag, b: Frag) -> bool:
    """Undecided is a wildcard; two decided fragments must agree."""
    return a.alliance is None or b.alliance is None or a.alliance == b.alliance


# EMPIRICAL DISPLACEMENT BOUND. The physical one -- ROBOT_MAX_SPEED_MS * gap -- is the
# worst case a drivetrain allows, and robots do not drive like that: they accelerate,
# turn, queue and stop. Measured INSIDE continuous track stretches across 8 matches of
# 2026necmp1, where the detector never lost the robot so it is certainly one robot:
#
#     window      p50    p90    p95    p99    max    physical bound
#      0.5 s        3    109    149    228    458              360
#      1.0 s        4    193    267    406    773              660
#      2.0 s        5    306    426    660   1354             1259
#      3.0 s        5    356    518    788   1558             1858
#      4.0 s        6    387    562    873   1526             2458
#      6.0 s        5    433    596    893   1589             3657
#
# The physical bound runs 2-4x loose and the gap widens with time, because a robot
# cannot hold top speed for six seconds but the formula assumes it can. Displacement
# saturates instead -- the field is only so big and robots double back.
#
# This cost a real error: on qm24 stitch joined two different robots across 3.73 s and
# 678 px, inside the physical bound but at the 97.7th percentile of real 3-4 s
# displacements. The chimera then propagated the whole way down -- see
# robots.split_chimeric_joins.
#
# Anchors are the measured p95, linearly interpolated, held flat past the last one.
# p95 rather than p99 because the tail is where wrong joins live; a legitimate pair that
# just misses stays split, which is an honest gap rather than a rewritten history.
EMPIRICAL_P95 = ((0.5, 149.0), (1.0, 267.0), (2.0, 426.0),
                 (3.0, 518.0), (4.0, 562.0), (6.0, 596.0))


def empirical_budget_px(gap_s: float) -> float:
    """How far a robot really travels in `gap_s`, at the 95th percentile."""
    if gap_s <= EMPIRICAL_P95[0][0]:
        return EMPIRICAL_P95[0][1]
    for (t0, d0), (t1, d1) in zip(EMPIRICAL_P95, EMPIRICAL_P95[1:]):
        if gap_s <= t1:
            f = (gap_s - t0) / (t1 - t0)
            return d0 + f * (d1 - d0)
    return EMPIRICAL_P95[-1][1]


def reachable(a: Frag, b: Frag, step: int, fps: float,
              empirical: bool = True) -> tuple[bool, float, float]:
    gap_frames = (b.f0 - a.f1) / step   # already in PROCESSED frames
    # `fps` is the processed rate (source fps / stride), so do NOT multiply by step
    # again -- doing so double-counted the stride and made every budget 2x too big.
    gap_s = gap_frames / fps
    dist = float(np.hypot(b.start[0] - a.end[0], b.start[1] - a.end[1]))
    if empirical:
        budget = empirical_budget_px(gap_s) + MARGIN_PX
    else:
        budget = min(C.ROBOT_MAX_SPEED_MS * PX_PER_M * gap_s + MARGIN_PX, HARD_CAP_PX)
    return dist <= budget, dist, budget


def stationary(a: Frag, b: Frag, step: int, fps: float, still_px: float) -> bool:
    """Reappeared essentially where it vanished, after a gap too long for `reachable`."""
    if still_px <= 0:
        return False
    gap_s = ((b.f0 - a.f1) / step) / fps
    if gap_s > STILL_MAX_GAP_S:
        return False
    d = float(np.hypot(b.start[0] - a.end[0], b.start[1] - a.end[1]))
    return d <= still_px


def occluded_pair(a: Frag, b: Frag, regions, step: int, fps: float) -> str | None:
    """Did `a` vanish behind a structure that `b` could have emerged from?

    Both endpoints must sit on the SAME structure -- its outline is the set of places a
    robot can go in or come out -- and the gap must fit the time it takes to get across
    it. That is the case pixel-proximity cannot express: a robot driving behind a tower
    leaves one edge and returns at another, tens of pixels apart or hundreds.

    Returns the structure's name, or None.
    """
    if not regions:
        return None
    from .occluders import region_at, transit_budget_s, parked_at
    ra = region_at(a.end[0], a.end[1], regions)
    if ra is None:
        return None
    rb = region_at(b.start[0], b.start[1], regions)
    if rb != ra:
        return None
    gap_s = ((b.f0 - a.f1) / step) / fps
    if gap_s < 0:
        return None
    # Two ways to be the same robot here, and the common one is not transit.
    budget = transit_budget_s(regions, ra, C.ROBOT_MAX_SPEED_MS * PX_PER_M)
    if gap_s <= budget:
        return ra
    return ra if parked_at(a.end, b.start, ra, regions, gap_s) else None


def stitch(frags: dict[int, Frag], step: int, fps: float, verbose: bool,
           still_px: float = STILL_PX, regions=None, empirical: bool = True
           ) -> tuple[dict[int, int], list[str]]:
    parent = {t: t for t in frags}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    log: list[str] = []
    # Chronological by birth, so a chain A->B->C merges in order.
    order = sorted(frags, key=lambda t: frags[t].f0)
    merged_into: dict[int, int] = {}

    for tid in order:
        b = frags[tid]
        cands = []
        for otid in order:
            if otid == tid:
                continue
            a = frags[find(otid)] if find(otid) in frags else frags[otid]
            a = frags[otid]
            # 4. no temporal overlap -- coexisting fragments are different robots
            if a.f1 >= b.f0:
                continue
            long_gap = (b.f0 - a.f1) / step > MAX_GAP_FRAMES
            occl = occluded_pair(a, b, regions, step, fps) if long_gap else None
            if long_gap and not occl and not stationary(a, b, step, fps, still_px):
                continue
            # already claimed by another fragment
            if a.tid in merged_into:
                continue
            if not alliance_ok(a, b):
                continue
            ok, dist, budget = reachable(a, b, step, fps, empirical)
            if occl:
                # Scored by DISPLACEMENT, on the same 0-1 scale the normal path uses,
                # so the two compete honestly. A flat score was tried and made things
                # worse -- 35 tracks against 32 -- because a constant ties with every
                # real candidate and trips the uniqueness gate, so adding a permissive
                # path REMOVED merges. A robot reappearing 13 px away has to outrank
                # one 689 px into its budget, and this makes it.
                from .occluders import PARKED_MOVE_PX as _PMP
                cands.append((min(dist / max(_PMP, 1e-9), 0.99), a.tid, dist, budget))
            elif long_gap:
                # Scored on stillness, not on a budget that has stopped meaning
                # anything at this gap. Ranks ahead of nothing else competing.
                cands.append((dist / max(still_px, 1e-9), a.tid, dist, still_px))
            elif ok:
                cands.append((dist / max(budget, 1e-9), a.tid, dist, budget))

        if not cands:
            continue
        cands.sort()
        # 3. uniqueness -- if two predecessors are plausible, refuse to guess.
        if len(cands) > 1 and cands[1][0] < cands[0][0] * 1.5:
            log.append(f"  AMBIGUOUS #{tid}: candidates "
                       f"{[f'#{c[1]}({c[2]:.0f}px)' for c in cands[:3]]} -- left split")
            continue

        _, best, dist, budget = cands[0]
        parent[find(tid)] = find(best)
        merged_into[best] = tid
        log.append(f"  merge #{best} -> #{tid}  gap {(b.f0-frags[best].f1)//step:>3}f  "
                   f"{dist:>5.0f}px of {budget:>5.0f}px budget")

    return {t: find(t) for t in frags}, log


def vote_alliance(rows: list[dict], mapping: dict[int, int]) -> dict[int, str | None]:
    """One alliance per track, by confidence-weighted vote over its whole life.

    A robot does not change alliance mid-match, so deciding this per frame is strictly
    worse than deciding it per track. Per-frame classification fails whenever a robot
    parks on a coloured field element -- measured: 104 frames where one alliance
    exceeded 3 despite six or fewer detections, i.e. a robot on the wrong side.

    Votes are weighted by the per-frame margin, so confident reads outvote marginal
    ones rather than every frame counting equally.
    """
    score: dict[int, dict[str, float]] = defaultdict(lambda: {"red": 0.0, "blue": 0.0})
    for r in rows:
        for d in r["dets"]:
            a = d.get("alliance")
            if d["tid"] < 0 or a not in ("red", "blue"):
                continue
            score[mapping[d["tid"]]][a] += max(float(d.get("aconf", 0.0)), 0.05)

    out: dict[int, str | None] = {}
    for root, s in score.items():
        if s["red"] == s["blue"] == 0:
            out[root] = None
        else:
            out[root] = "red" if s["red"] > s["blue"] else "blue"
    return out


def apply(rows: list[dict], mapping: dict[int, int],
          votes: dict[int, str | None] | None = None) -> tuple[list[dict], int]:
    # Renumber survivors to 1..N so the output reads like what it is.
    roots = sorted(r for r in set(mapping.values()) if r >= 0)
    renum = {r: i + 1 for i, r in enumerate(roots)}
    renum[-1] = -1
    changed = 0
    out = []
    for r in rows:
        dets = []
        for d in r["dets"]:
            d = dict(d)
            if d["tid"] >= 0:
                root = mapping[d["tid"]]
                d["orig_tid"] = d["tid"]
                d["tid"] = renum[root]
                if votes is not None:
                    decided = votes.get(root)
                    if decided is not None and d.get("alliance") != decided:
                        d["alliance_raw"] = d.get("alliance")
                        d["alliance"] = decided
                        changed += 1
            dets.append(d)
        out.append({**r, "dets": dets})
    return out, changed


# WITHIN-TRACK STEP BOUND, in BOX WIDTHS PER SECOND.
#
# EMPIRICAL_P95 above answers a different question -- how far apart two fragments may be
# and still be joined across a gap -- at p95, which is far too loose to cut on. This is
# the per-step question: between two CONSECUTIVE detections of one track, how far can the
# box move and still be the same robot?
#
# Box widths rather than raw pixels, because a robot near the camera covers several times
# the pixels of one at the far barrier for the same real motion. A box width is roughly a
# bumper perimeter, so the ratio is scale-invariant -- the same reason DUP_DX_WIDTHS and
# CROSS_WIDTHS are expressed this way.
#
# Measured over 665k consecutive within-track steps across both events, displacement of
# the floor-contact point in box widths:
#
#     dt            p50    p90    p99   p99.9    max      event
#     0.05-0.1s    0.01   0.10   0.21    0.41    2.3      2026mawor
#     0.05-0.1s    0.00   0.11   0.20    0.34    1.4      2026necmp1
#
# p99.9 at one sample interval is 0.41 widths in 0.067 s = 6.1 widths/s. An FRC bumper
# perimeter is ~0.9 m, so that is 5.5 m/s -- the drivetrain figure, recovered from the
# footage without being told. Good evidence the measure is the right one.
#
# 8.0 sits ~30% above that, so it cuts only what no drivetrain can do. Cut rates:
#
#     widths/s / floor    mawor            necmp1
#     6.0 / 0.5           241  (0.064%)     81  (0.028%)
#     8.0 / 1.2            27  (0.007%)      6  (0.002%)
#     12.0 / 2.0           13  (0.003%)      1  (0.000%)
#
# The floor covers very small dt, where box jitter rather than motion dominates.
STEP_WIDTHS_PER_S = 8.0
STEP_FLOOR_WIDTHS = 1.2


def cut_impossible_steps(rows, mapping, wps: float = STEP_WIDTHS_PER_S,
                         floor: float = STEP_FLOOR_WIDTHS) -> int:
    """Cut a track where its own box moves further than a robot can, in place.

    This is the only check in the pipeline that can see a bad detection INSIDE a track.
    Everything downstream compares tracks to each other, so a wrong detection arrives
    already wearing the right identity and there is no pair to forbid.

    2026mawor_qm13 track 275: at f4584 the box is a robot beside the red hub (60x50 px,
    conf 0.537); at f4596 it is a PERSON in an FTA vest at the near barrier (98x84 px,
    conf 0.613). The detector scored the person higher than the robot it was following
    and BoT-SORT took it. The box moved 4.2 box widths in 0.2 s against a bound of 1.6.

    HERE rather than in rtrack.robots, which has the same check in metres: three stages
    consume the stitched tracks before the solver ever sees them -- appear embeds every
    crop, reid votes computes identity per track, and rtrack.curate builds the bundle a
    human is asked to label. That last one matters most: cutting late means a curator can
    be shown a crop of a person and asked which robot it is.
    """
    import numpy as _np
    # GROUPED BY THE MERGED TRACK, not the fragment. rows still carry fragment ids at
    # this point and `mapping` is what turns them into tracks -- grouping by the raw tid
    # inspects contiguous tracker output, which almost never contains an impossible step,
    # and the check found 1 case in 13 matches instead of the 27 measured on the stitched
    # OUTPUT. The steps worth catching are the ones a JOIN introduced.
    seq = defaultdict(list)
    for ri, r in enumerate(rows):
        for di, d in enumerate(r["dets"]):
            if d["tid"] < 0:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            seq[mapping.get(d["tid"], d["tid"])].append(
                (r["t"], (x1 + x2) / 2.0, y2, max(x2 - x1, 1.0), ri, di))
    nxt = max(max((d["tid"] for r in rows for d in r["dets"]), default=-1),
              max(mapping.values(), default=-1)) + 1
    cuts = 0
    for v in seq.values():
        v.sort()
        new = None
        for a, b in zip(v, v[1:]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            dw = float(_np.hypot(b[1] - a[1], b[2] - a[2])) / ((a[3] + b[3]) / 2.0)
            if dw > max(floor, wps * dt):
                new = nxt
                nxt += 1
                # Its own root, so the piece after the cut is a separate TRACK rather
                # than rejoining the one it was severed from.
                mapping[new] = new
                cuts += 1
            if new is not None:
                rows[b[4]]["dets"][b[5]]["tid"] = new
    return cuts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge split track fragments.")
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="processed frame rate (source fps / stride)")
    ap.add_argument("--step-widths", type=float, default=STEP_WIDTHS_PER_S,
                    metavar="W_PER_S",
                    help="cut a track where its box moves faster than this many box "
                         "widths per second between consecutive detections. 0 disables. "
                         "See cut_impossible_steps.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--min-track", type=int, default=40,
                    help="final tracks with fewer detections than this are demoted "
                         "to untracked (tid -1) rather than counted as robots")
    # MEASURED: off by default. Voting assumes per-frame errors are random noise
    # around a correct majority. They are not -- a robot parked on a coloured field
    # element is misread for most of its visible life, so the vote entrenches the
    # error for the whole track instead of averaging it away. On the full match this
    # took frames-with-an-alliance-over-3 from 123 (4.6%) to 412 (15.5%).
    ap.add_argument("--physical-bound", action="store_true",
                    help="use ROBOT_MAX_SPEED_MS * gap for reachability instead of the "
                         "measured p95 displacement. The physical bound runs 2-4x loose "
                         "-- see EMPIRICAL_P95 -- and let two different robots be joined "
                         "on qm24.")
    ap.add_argument("--occluders", default=None, metavar="CAMERA",
                    help="camera stem whose drawn occluder regions should be used to "
                         "rebind tracks across a structure (calib/<CAMERA>_occluders"
                         ".json, drawn in public/rtrack/occluders.html). Without it, "
                         "long gaps are refused as before.")
    ap.add_argument("--still-px", type=float, default=STILL_PX,
                    help="fallback for a camera with no drawn occluders: rebind a "
                         "reappearance within this many pixels. 0 = off; see STILL_PX "
                         "for why it defaults off.")
    ap.add_argument("--vote", action="store_true",
                    help="assign one alliance per track by weighted vote. Only "
                         "helps once per-frame classification is right more often "
                         "than not; see the note in the source.")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.tracks.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])
    frags, step = load_frags(rows)
    regions = None
    if args.occluders:
        from .occluders import load as _load_occ, to_pixels as _occ_px
        doc = _load_occ(args.occluders)
        if doc is None:
            print(f"[stitch] no occluder file for {args.occluders} -- long gaps stay "
                  f"refused. Draw one in public/rtrack/occluders.html.")
        else:
            # Frame size from the tracks themselves, so a drawing made on a different
            # resolution is rescaled rather than silently misplaced.
            mx = max((d["xyxy"][2] for r in rows for d in r["dets"]), default=1920)
            my = max((d["xyxy"][3] for r in rows for d in r["dets"]), default=1080)
            wh = (1920 if mx <= 1920 else 3840, 1080 if my <= 1080 else 2160)
            regions = _occ_px(doc, wh)
            print(f"[stitch] {len(regions)} occluder region(s) for {args.occluders} "
                  f"at {wh[0]}x{wh[1]}: "
                  + ", ".join(r["name"] for r in regions))
    mapping, log = stitch(frags, step, args.fps, not args.quiet,
                          still_px=args.still_px, regions=regions,
                          empirical=not args.physical_bound)

    if not args.quiet:
        for line in log:
            print(line)

    before, after = len(frags), len(set(mapping.values()))
    print(f"\n[stitch] {before} fragments -> {after} tracks "
          f"({before - after} merges)")
    sizes = Counter(mapping.values())
    survivors = sorted(((sum(len(frags[t]) for t in frags if mapping[t] == root), root)
                        for root in set(mapping.values())), reverse=True)
    print("[stitch] merged track sizes (detections): "
          f"{[s for s, _ in survivors]}")

    # Before the alliance vote and the micro-fragment drop: a track that contains a
    # person should not contribute that crop to an alliance vote, and the pieces it
    # breaks into should face the same size test as everything else.
    if args.step_widths > 0:
        n_step = cut_impossible_steps(rows, mapping, args.step_widths)
        if n_step:
            print(f"[stitch] cut {n_step} impossible step(s) inside a track "
                  f"(> {args.step_widths} box widths/s) -- see cut_impossible_steps")

    votes = vote_alliance(rows, mapping) if args.vote else None
    if votes is not None:
        tally = Counter(v for v in votes.values() if v)
        print(f"[stitch] track alliances by weighted vote: "
              f"{dict(tally)} ({sum(1 for v in votes.values() if v is None)} undecided)")

    # Drop micro-fragments. They are almost always spurious, and they are what push
    # the co-detected track count above 6 -- which is the constraint that decides
    # whether the whole track set can be coloured into exactly 6 robots. Measured:
    # 6 of 32 tracks held <40 detections, 131 detections total (~1% of the data).
    # Detections are KEPT (tid -> -1) so nothing is lost, they just stop claiming to
    # be a distinct robot.
    sizes = Counter()
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                sizes[mapping[d["tid"]]] += 1
    tiny = {root for root, n in sizes.items() if n < args.min_track}
    if tiny:
        n_tiny = sum(sizes[t] for t in tiny)
        print(f"[stitch] dropping {len(tiny)} micro-track(s) under "
              f"{args.min_track} detections ({n_tiny} detections -> untracked)")
        mapping = {k: (-1 if v in tiny else v) for k, v in mapping.items()}
        if votes:
            votes = {k: v for k, v in votes.items() if k not in tiny}

    out_rows, changed = apply(rows, mapping, votes)
    if votes is not None:
        print(f"[stitch] alliance relabelled on {changed} detections "
              f"(original kept as alliance_raw)")

    out = args.out or args.tracks.with_name(args.tracks.stem + "_stitched.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[stitch] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
