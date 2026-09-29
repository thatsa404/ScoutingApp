"""Route cleaning at export: fix what the solver's identity and the detector's boxes got
wrong, using evidence the solver did not weigh at the moment it decided.

    Called from rtrack.export.build() on the full-rate (15 Hz) samples, before they are
    thinned to the output rate. OFF by default; enable with `rtrack.export --clean`.

Three passes, in this order, each measured against video before it was written:

1. ALLIANCE CONFLICT. A tracker track's bumper colour is measured on every detection.
   When a team's route runs over a track whose colour is decisively the OTHER alliance,
   that stretch belongs to somebody else. Across 70 matches, 280 in-match handovers
   landed a team on such a track; two were confirmed in video -- 2026mawor_qm14 put blue
   8724 on red 10254 for about a minute, and 2026necmp1_qm17 put red 1768 on blue 7127
   for three seconds of auto. The stretch is handed to the one opposite-alliance robot
   whose own route has a hole there and whose ends it joins within the distance budget;
   with no such robot, or more than one, it is dropped. Never kept on the wrong team.

2. MOMENTUM AT HANDOVERS. Where the route passes from one track to another, the new
   track's first position is compared with where the old track was HEADED, not merely
   with how far away it is (rtrack.solve.distance_budget is direction-blind). Only at
   handovers: applied to every sample, the same test flagged box-geometry glitches and
   hard stops, and none of ten checked in video were identity swaps. At handovers it
   flagged colour-confirmed swaps at 7.2% against the distance gate's 4.0% (baseline
   2.5%) -- a better signal, but mostly still false alarms, so a flagged handover is
   only REASSIGNED when exactly one other robot fits it on both ends. Otherwise the
   route is broken there and the samples stay.

3. PARTIAL BOXES. A robot half-hidden behind a hub, a referee or the frame edge gets a
   box that covers only its top; the box bottom rises and its floor point lands a metre
   or more too far away (2026necmp1_qm11 4925, qm23 8709, 2026mawor_qm15 2168, all seen
   in video). Detected against the same team's own recent box height, not a global
   model, because robots differ in height far more than a partial box differs from a
   whole one. Such samples are dropped, leaving a gap rather than a wrong point.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .solve import distance_budget

# ── alliance conflict ──────────────────────────────────────────────────────
COLOUR_MIN_DETS = 8          # a track needs this many coloured detections to have a colour
COLOUR_DECISIVE = 0.75       # ...and this share of them one way
COLOUR_SMOOTH = 7            # rolling majority over this many coloured detections, so one
                             # misread bumper never opens a conflict stretch on its own
# ── momentum ───────────────────────────────────────────────────────────────
ACCEL_MS2 = 10.0             # traction-limited acceleration, FRC drivetrains
POS_NOISE_M = 0.30
VMAX_MS = 5.5
FIT_S = 0.5                  # velocity from the outgoing track's last half second
HANDOVER_MAX_GAP_S = 1.0     # past this the reachable set is the speed cap; no direction
MOMENTUM_RATIO = 2.0
# ── partial boxes ──────────────────────────────────────────────────────────
BOX_WINDOW_S = 1.5           # the team's own box height over the preceding window
BOX_SHRINK = 0.70            # shorter than this share of it...
BOX_RISE = 0.25              # ...with its bottom risen by this share of it


def _runs(ss: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for s in ss:
        if out and out[-1][-1]["tid"] == s["tid"]:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def _reachable(a: dict, b: dict, mult: float = 1.0) -> bool:
    dt = abs(b["t"] - a["t"])
    return math.hypot(b["x"] - a["x"], b["y"] - a["y"]) <= mult * distance_budget(max(dt, 1 / 15))


def _momentum_ratio(before: list[dict], after: list[dict]) -> float | None:
    end = before[-1]["t"]
    w = [s for s in before if s["t"] >= end - FIT_S]
    if len(w) < 4:
        w = before[-4:]
    if len(w) < 4:
        return None
    t = np.array([s["t"] - end for s in w])
    vx, x0 = np.polyfit(t, [s["x"] for s in w], 1)
    vy, y0 = np.polyfit(t, [s["y"] for s in w], 1)
    sp = math.hypot(vx, vy)
    if sp > VMAX_MS:
        vx, vy = vx * VMAX_MS / sp, vy * VMAX_MS / sp
    dt = max(after[0]["t"] - end, 1 / 15)
    bx = float(np.median([s["x"] for s in after[:3]]))
    by = float(np.median([s["y"] for s in after[:3]]))
    reach = min(0.5 * ACCEL_MS2 * dt * dt, 2 * VMAX_MS * dt) + POS_NOISE_M
    return math.hypot(bx - (x0 + vx * dt), by - (y0 + vy * dt)) / reach


def _track_colours(by_team: dict[str, list[dict]]) -> dict[int, str]:
    """Each track's bumper colour, from every detection on it -- whoever it is credited to."""
    c: dict[int, Counter] = defaultdict(Counter)
    for ss in by_team.values():
        for s in ss:
            if s.get("alliance") in ("red", "blue"):
                c[s["tid"]][s["alliance"]] += 1
    out = {}
    for tid, cc in c.items():
        n = sum(cc.values())
        col, k = cc.most_common(1)[0]
        if n >= COLOUR_MIN_DETS and k / n >= COLOUR_DECISIVE:
            out[tid] = col
    return out


def _conflict_stretches(ss: list[dict], mine: str) -> list[tuple[int, int]]:
    """Index ranges [i, j] of `ss` where the bumper colour is sustainedly the OTHER alliance.

    Judged along the team's route, not per tracker track, because the tracker can switch
    robots without switching track id: in 2026necmp1_qm17 track 9 carried red 1768 until
    3.9 s of auto and blue 7127 after it, so the track as a whole read 58% blue -- not
    decisive -- while the stretch after the switch read almost entirely blue.
    """
    col = [(i, s["alliance"] != mine) for i, s in enumerate(ss)
           if s.get("alliance") in ("red", "blue")]
    if len(col) < COLOUR_MIN_DETS:
        return []
    flags = [f for _, f in col]
    h = COLOUR_SMOOTH // 2
    smooth = [sum(flags[max(0, k - h):k + h + 1]) * 2 > len(flags[max(0, k - h):k + h + 1])
              for k in range(len(flags))]
    out, k = [], 0
    while k < len(col):
        if not smooth[k]:
            k += 1; continue
        m = k
        while m + 1 < len(col) and smooth[m + 1]:
            m += 1
        opp = flags[k:m + 1]
        if len(opp) >= COLOUR_MIN_DETS and sum(opp) / len(opp) >= COLOUR_DECISIVE:
            # trim to the first and last opposite reading, so the stretch starts where
            # the other robot's bumpers first appear rather than where smoothing does
            ks = next(q for q in range(k, m + 1) if flags[q])
            ms = next(q for q in range(m, k - 1, -1) if flags[q])
            out.append((col[ks][0], col[ms][0]))
        k = m + 1
    return out


def _unique_taker(seg: list[dict], donor: str, by_team: dict[str, list[dict]],
                  alliance_of: dict[str, str], want_alliance: str | None) -> str | None:
    """The one robot this stretch plausibly belongs to: silent throughout it, and joined
    to it within the distance budget at every end where it has a neighbouring sample."""
    t0, t1 = seg[0]["t"], seg[-1]["t"]
    fits = []
    for team, ss in by_team.items():
        if team == donor or (want_alliance and alliance_of.get(team) != want_alliance):
            continue
        if any(t0 - 0.1 <= s["t"] <= t1 + 0.1 for s in ss):
            continue                                   # it was somewhere else then
        before = [s for s in ss if s["t"] < t0]
        after = [s for s in ss if s["t"] > t1]
        ends = []
        if before:
            ends.append(_reachable(before[-1], seg[0]))
        if after:
            ends.append(_reachable(seg[-1], after[0]))
        if ends and all(ends):
            fits.append(team)
    return fits[0] if len(fits) == 1 else None


def load_boxes(labeled: Path) -> dict[tuple[int, int], tuple[float, float]]:
    """(frame, tid) -> (box height, box bottom) in pixels."""
    out = {}
    if not labeled.exists():
        return out
    with labeled.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            for d in o["dets"]:
                x0, y0, x1, y1 = d["xyxy"]
                out[(o["f"], d["tid"])] = (y1 - y0, y1)
    return out


def clean(by_team: dict[str, list[dict]], alliance_of: dict[str, str],
          boxes: dict | None = None) -> tuple[dict[str, list[dict]], dict, list[dict]]:
    """Return (cleaned samples per team, counts, route breaks to add as gaps).

    `by_team` holds positions.json samples (t, x, y, tid, alliance, f) sorted or not;
    times may be in any frame so long as they are consistent. Nothing is mutated.
    """
    teams = {t: sorted(ss, key=lambda s: s["t"]) for t, ss in by_team.items()}
    stats = Counter()
    breaks: list[dict] = []

    # 3 first: a partial box also distorts the positions passes 1 and 2 read.
    if boxes:
        for team, ss in teams.items():
            keep, hist = [], []
            for s in ss:
                hb = boxes.get((s.get("f"), s["tid"]))
                if hb is None:
                    keep.append(s); continue
                h, bot = hb
                recent = [(hh, bb) for tt, hh, bb in hist if s["t"] - BOX_WINDOW_S <= tt < s["t"]]
                if len(recent) >= 5:
                    mh = float(np.median([r[0] for r in recent]))
                    mb = float(np.median([r[1] for r in recent]))
                    if h < BOX_SHRINK * mh and (mb - bot) > BOX_RISE * mh:
                        stats["partialBoxDropped"] += 1
                        continue          # not added to hist either: it is not the robot's size
                hist.append((s["t"], h, bot))
                keep.append(s)
            teams[team] = keep

    # 1. alliance conflict
    for team in list(teams):
        mine = alliance_of.get(team)
        if mine not in ("red", "blue"):
            continue
        ss = teams[team]
        other = "blue" if mine == "red" else "red"
        drop = set()
        for i, j in _conflict_stretches(ss, mine):
            run = ss[i:j + 1]
            drop.update(range(i, j + 1))
            taker = _unique_taker(run, team, teams, alliance_of, other)
            if taker:
                teams[taker] = sorted(teams[taker] + run, key=lambda s: s["t"])
                stats["allianceReassigned"] += len(run)
                stats["allianceReassignedRuns"] += 1
            else:
                stats["allianceDropped"] += len(run)
                stats["allianceDroppedRuns"] += 1
            breaks.append({"team": team, "tStart": run[0]["t"], "tEnd": run[-1]["t"],
                           "reason": "alliance-conflict", "to": taker})
        teams[team] = [s for k, s in enumerate(ss) if k not in drop]

    colours = _track_colours(teams)

    # 2. momentum at handovers
    for team in list(teams):
        runs = _runs(teams[team])
        kept, i = [], 0
        while i < len(runs):
            run = runs[i]
            if kept and 0 < run[0]["t"] - kept[-1]["t"] <= HANDOVER_MAX_GAP_S and len(run) >= 2:
                r = _momentum_ratio(kept, run)
                if r is not None and r > MOMENTUM_RATIO:
                    stats["momentumFlagged"] += 1
                    # Without colour evidence the stretch may only move within the
                    # donor's alliance. Crossing on kinematics alone moved a 17 s stretch
                    # of red 10254 onto blue 8724 in 2026mawor_qm14.
                    col = colours.get(run[0]["tid"]) or alliance_of.get(team)
                    taker = _unique_taker(run, team, teams, alliance_of, col)
                    # A taker must also carry on the OLD track's motion less badly than
                    # this team does -- otherwise both are equally wrong and moving the
                    # stretch only relocates the error.
                    if taker:
                        tb = [s for s in teams[taker] if s["t"] < run[0]["t"]]
                        rt = _momentum_ratio(tb, run) if len(tb) >= 4 else None
                        if rt is not None and rt > MOMENTUM_RATIO:
                            taker = None
                    if taker:
                        teams[taker] = sorted(teams[taker] + run, key=lambda s: s["t"])
                        stats["momentumReassigned"] += 1
                        breaks.append({"team": team, "tStart": kept[-1]["t"], "tEnd": run[-1]["t"],
                                       "reason": "handover-reassigned", "to": taker})
                        i += 1
                        continue
                    breaks.append({"team": team, "tStart": kept[-1]["t"], "tEnd": run[0]["t"],
                                   "reason": "handover-jump"})
            kept.extend(run)
            i += 1
        teams[team] = kept

    return teams, dict(stats), breaks
