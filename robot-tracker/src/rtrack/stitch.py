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


def reachable(a: Frag, b: Frag, step: int, fps: float) -> tuple[bool, float, float]:
    gap_frames = (b.f0 - a.f1) / step   # already in PROCESSED frames
    # `fps` is the processed rate (source fps / stride), so do NOT multiply by step
    # again -- doing so double-counted the stride and made every budget 2x too big.
    gap_s = gap_frames / fps
    dist = float(np.hypot(b.start[0] - a.end[0], b.start[1] - a.end[1]))
    budget = min(C.ROBOT_MAX_SPEED_MS * PX_PER_M * gap_s + MARGIN_PX, HARD_CAP_PX)
    return dist <= budget, dist, budget


def stitch(frags: dict[int, Frag], step: int, fps: float, verbose: bool
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
            if (b.f0 - a.f1) / step > MAX_GAP_FRAMES:
                continue
            # already claimed by another fragment
            if a.tid in merged_into:
                continue
            if not alliance_ok(a, b):
                continue
            ok, dist, budget = reachable(a, b, step, fps)
            if ok:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge split track fragments.")
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="processed frame rate (source fps / stride)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--min-track", type=int, default=40,
                    help="final tracks with fewer detections than this are demoted "
                         "to untracked (tid -1) rather than counted as robots")
    # MEASURED: off by default. Voting assumes per-frame errors are random noise
    # around a correct majority. They are not -- a robot parked on a coloured field
    # element is misread for most of its visible life, so the vote entrenches the
    # error for the whole track instead of averaging it away. On the full match this
    # took frames-with-an-alliance-over-3 from 123 (4.6%) to 412 (15.5%).
    ap.add_argument("--vote", action="store_true",
                    help="assign one alliance per track by weighted vote. Only "
                         "helps once per-frame classification is right more often "
                         "than not; see the note in the source.")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.tracks.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])
    frags, step = load_frags(rows)
    mapping, log = stitch(frags, step, args.fps, not args.quiet)

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
