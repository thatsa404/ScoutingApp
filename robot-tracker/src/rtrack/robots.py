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


def track_info(rows, positions: dict | None):
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
    if positions:
        pts = defaultdict(list)
        for s in positions["samples"]:
            if s["tid"] >= 0 and "offfield" not in s["flags"]:
                pts[s["tid"]].append((s["t"], s["x"], s["y"]))
        for tid, ps in pts.items():
            if tid in info and ps:
                ps.sort()
                info[tid]["start"] = (ps[0][1], ps[0][2])
                info[tid]["end"] = (ps[-1][1], ps[-1][2])
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


def split_on_appearance(rows, npz_path: Path, thresh: float, win: int = 12):
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
            d[i] = np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(np.clip(a * b, 0, None)))))
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


def drop_offfield(rows, stem: str, slack: float = FIELD_SLACK_M,
                  calib_stem: str | None = None):
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
                   field_filter: bool = True, calib_stem: str | None = None):
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



def _write_clash_bundle(stem, args, rows, clashes, red, blue) -> None:
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
    issues = custody_conflicts(rows)
    if not issues:
        print("[clash] no custody conflicts -- nothing left to ask about")
        return
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

    carry = []
    if args.corrections and args.corrections.exists():
        carry = json.loads(args.corrections.read_text(encoding="utf-8"))["labels"]

    doc = CU.build_frames(stem, args.tracks, args.match, 0, 2, rows=rows,
                          frames=want, focus=focus, carry=carry,
                          note=("These frames come in pairs, moments apart. In each "
                                "pair the pipeline gave ONE team name to two different "
                                "robots -- the amber box in each frame. They cannot "
                                "both be that team. Name the amber box in each."))
    dest = (C.STAGE3_DIR / f"{stem}_curate_clash.json"
            if str(args.clash_bundle) == "AUTO" else args.clash_bundle)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc), encoding="utf-8")
    print(f"[clash] {len(want)} frame(s) covering {len(issues)} custody "
          f"conflict(s), carrying {len(carry)} existing label(s) "
          f"-> {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: tracks -> 6 named robots.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--identity", type=Path, default=None)
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
    ap.add_argument("--park", type=float, default=300.0,
                    help="cost of leaving a track unassigned; raise for coverage, "
                         "lower for purity (cpsat only)")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    rows = load_tracks(args.tracks)
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

    m = tba_mod.match_by_key(args.match)
    red, blue = [str(t) for t in m["red"]], [str(t) for t in m["blue"]]
    print(f"[robots] red {red}  blue {blue}")

    if not args.no_alliance_split:
        rows, n_a, orig_a = split_on_alliance(rows)
        if n_a:
            ident = retally(ident, rows, orig_a)
            print(f"[robots] split {n_a} segment(s) where a track's bumper HUE "
                  f"changed alliance ({len(orig_a) - n_a} tracks affected)")

    if args.appear_thresh > 0:
        ap_p = C.STAGE3_DIR / f"{stem}_appearance.npz"
        rows, n_p, orig_p = split_on_appearance(rows, ap_p, args.appear_thresh)
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

    info = track_info(rows, positions)
    con = conflicts(rows)
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
        W = S.Weights(park=args.park, alli=args.alli_weight)

        def run_solve(limit):
            a, t, m, u = S.solve(info, con, ident, red, blue, pinned or None,
                                 limit, W, preferred=preferred or None)
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
                                     longer, W, preferred=preferred or None)
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
            info = track_info(rows, positions)
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
        _write_clash_bundle(stem, args, rows, clashes, red, blue)

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
