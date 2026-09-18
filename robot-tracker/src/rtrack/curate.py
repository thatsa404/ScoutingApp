"""Stage 3 -- build a self-contained bundle for a human curator, and nothing else.

    uv run -m rtrack.curate GSxbsE42o5o --match 2026necmp_f1m3

Writes out/stage3/<stem>_curate.json: one file holding every track worth asking about,
each with a strip of crops as base64 JPEGs. Open viewer/curate.html, drag the file in,
adjudicate, save a corrections file, feed it back with:

    uv run -m rtrack.robots <video> --tracks ... --match ... --solver cpsat \\
        --corrections corrections/<matchkey>.json

WHY A SINGLE FILE. The curator's machine needs no server, no CORS, no deployment and
no Python -- `<input type="file">` works from disk and works identically later on
GitHub Pages. The bundle is also exactly what a relay would carry if curation ever
leaves this machine; nexus-relay's KV limit is 25 MB and a 30-track bundle is ~2-3 MB.

WHAT IT ASKS ABOUT, AND IN WHAT ORDER. Tracks sorted by detection count, because that
is what a correction is worth: the top 10 tracks carry 44% of all detections, the top
30 carry 80%, and the tail of 6-detection fragments is not worth a human's attention
at any price. A curator who stops early has still fixed the most of the match that
could be fixed in that time.

The machine's current guess is included but flagged `hint`, and the viewer keeps it
hidden until asked. Showing it makes confirmation fast and makes anchoring certain,
and anchoring is precisely how a wrong label gets laundered into a verified one.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id
from . import tba as tba_mod
from .chicklets import plan, grab, choose, CROP_TOP, CROP_BOTTOM

MAX_TRACKS = 40
# 12, not 8. The viewer wraps the strip rather than scrolling it, so every crop is on
# screen at once -- and the whole point of showing a track over time is to catch the
# moment it changes robot. Eight crops over a 60 s track is one every 7 s, which is
# coarse enough to miss a switch entirely; that is how track 16 was labelled 9644 from
# its first half while its second half read 6329.
CROPS_PER_TRACK = 10
# Taller than before because the crop now spans the whole robot rather than the
# bumper band: at 132 px the number would have shrunk with the wider framing.
CROP_H = 200
JPEG_Q = 66
# Beyond this a single track id is more likely than not to have crossed robots, and a
# whole-track label loses the distinction. 40 s is a quarter of the match.
LONG_TRACK_S = 40.0


def encode(img: np.ndarray) -> str:
    f = CROP_H / img.shape[0]
    im = cv2.resize(img, (max(24, int(img.shape[1] * f)), CROP_H),
                    interpolation=cv2.INTER_CUBIC)
    im = im[:, :420]
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


# Frames are downscaled to this width before shipping. 1600/Q70 suits a desktop curator
# working from a local file; over venue wifi it is painful.
#
# --mobile ships 800/Q45, and that is only reasonable because each detection now
# carries its own crop taken from the FULL-resolution frame (see _crop). Once
# identification moved into the crop, the frame's remaining jobs are "show me where
# this box is" and "be big enough to tap" -- neither needs detail. Measured on a
# 16-frame bundle:
#
#     frame setting   frames   crops   total   robot width in-frame
#       1100 / Q58     2.8 MB  0.6 MB  3.4 MB        60 px
#        800 / Q45     1.4 MB  0.6 MB  2.0 MB        44 px
#        700 / Q40     1.1 MB  0.6 MB  1.7 MB        38 px
#
# 800 halves the bundle against 1100 and costs nothing that is looked at. Going lower
# starts to make boxes fiddly to hit, though the viewer's nearest-centre hit test
# tolerates a lot.
FRAME_W = 1600
FRAME_Q = 70
MIN_ROBOTS = 4          # a frame showing fewer is not worth a curator's attention


# Per-detection crop shipped alongside each frame. PAD gives context either side of the
# robot -- what it is next to and what it is doing both carry identity, and a box cut
# exactly to the bumpers is harder to judge than one with a little room.
CROP_PAD = 0.45
CROP_MAX_H = 260        # cap the tall dimension; native crops of near robots are huge
CROP_JPEG_Q = 78        # higher than the frame's: this is the image being judged


def _crop(img, x1: float, y1: float, x2: float, y2: float) -> str:
    """One robot, cut from the FULL-RESOLUTION frame and encoded as a data URI."""
    H, W = img.shape[:2]
    px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
    c, d = min(W, int(x2 + px)), min(H, int(y2 + py))
    if c - a < 8 or d - b < 8:
        return ""
    sub = img[b:d, a:c]
    if sub.shape[0] > CROP_MAX_H:
        f = CROP_MAX_H / sub.shape[0]
        sub = cv2.resize(sub, (max(8, int(sub.shape[1] * f)), CROP_MAX_H),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", sub, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_Q])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


MIN_FRAME_GAP_S = 4.0   # chosen frames must be this far apart in time

# ---- match window --------------------------------------------------------------
# A slice runs [actual_time - 30 s, +210 s] and the match does not fill it. On
# 2026necmp1_qm1 the robots are parked until t=28 and again from t=195, while the clip
# runs to 240 -- so the last curation frames showed a still field with PEOPLE WALKING ON
# IT, which is not a labelling question anyone can answer.
#
# Set cover walks into this because it maximises DETECTIONS COVERED and a parked robot
# is still a detection: post-match buckets held 4-5 detections per frame, as many as
# mid-teleop. Nothing in the objective knew the match was over.
#
# TBA's actual_time cannot fix it -- measured here it is ~30 s early relative to the
# broadcast, which is why the pad exists in the first place. Robot motion can: median
# inter-frame box travel is 0.05-0.07 px parked against 8-12 px in teleop, two orders of
# magnitude, and needs no calibration or projection to read.
#
# NOT shot detection, which was my first guess and was wrong: all five "shots" this
# detector found in that clip are the same static camera, split on the SCOREBOARD
# redrawing as auto rolled into teleop. The camera never moved.
# p75, NOT the median, and that distinction decides whether this works. Measured on
# 2026necmp1_qm1, the two statistics separate very differently:
#
#                     parked / post-match      playing
#     median                0.05 - 0.81      1.56 - 8.59     <- overlaps
#     p75                   0.10 - 2.35      4.41 - 19.99    <- clean gap
#
# The median overlaps because after the match robots get DRIVEN OFF the field: a couple
# move while the rest sit, which lifts the median just past any threshold low enough to
# catch slow auto. p75 asks "are several robots moving at once", which is true during
# play and false while a field is cleared one robot at a time. A first attempt at 0.50
# on the median let the window run to t=243 on a match that ended at 202.
MOTION_BIN_S = 2.0
MOTION_PCTL = 75        # percentile of per-detection travel within a bin
MOTION_MIN_PX = 3.0     # ...that counts as "playing"; the gap above is 2.35 to 4.41
MOTION_HOLD_S = 4.0     # ...sustained this long, so one jitter cannot open the window
WINDOW_PAD_S = 3.0      # keep a little either side: the first move is worth seeing


def match_window(rows, span_s: float | None = None) -> tuple[float, float] | None:
    """(t0, t1) over which robots are actually moving, or None if it cannot be read.

    THE TAIL IS CAPPED BY THE RULEBOOK, THE HEAD IS MEASURED. Sustained motion is a
    sound way to find where a match STARTS -- nothing else on the field moves like six
    robots being enabled. It is a poor way to find where one ENDS, because the things
    that happen next also move: drivers keep driving through the buzzer, field crew walk
    out to reset, and the broadcast cuts to a replay. Across 25 curated 2026necmp1
    bundles that put six matches past their real length, qm22 by 400 s. Where the end IS
    unambiguous -- sustained motion followed by a dead field -- it lands at 162-164 s in
    12 of the 14 matches that show one, which is what C.MATCH_SPAN_S is set from.

    So t1 is `min(last sustained motion, t0 + span)`. `span` defaults to C.MATCH_SPAN_S
    and is an argument only so a different game, or a measured per-event value, can be
    passed in without editing this.
    """
    prev: dict[int, tuple[float, tuple[float, float]]] = {}
    bins: dict[float, list[float]] = defaultdict(list)
    for r in sorted(rows, key=lambda z: z["f"]):
        b = round(r["t"] / MOTION_BIN_S) * MOTION_BIN_S
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            c = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            p = prev.get(d["tid"])
            if p and 0 < r["t"] - p[0] < 0.3:
                bins[b].append(float(np.hypot(c[0] - p[1][0], c[1] - p[1][1])))
            prev[d["tid"]] = (r["t"], c)
    if not bins:
        return None
    ts = sorted(bins)
    spd = [float(np.percentile(bins[t], MOTION_PCTL)) for t in ts]
    need = max(1, int(round(MOTION_HOLD_S / MOTION_BIN_S)))
    hot = [i for i in range(len(ts) - need + 1)
           if all(spd[j] >= MOTION_MIN_PX for j in range(i, i + need))]
    if not hot:
        return None
    t_start = ts[hot[0]]
    t_end = ts[hot[-1] + need - 1]
    # The cap is measured from the START OF MOTION, not from the padded window, so the
    # pad cannot smuggle extra seconds past the rulebook at either end.
    t_end = min(t_end, t_start + (C.MATCH_SPAN_S if span_s is None else span_s))
    return (max(0.0, t_start - WINDOW_PAD_S), t_end + WINDOW_PAD_S)
CONFUSABLE_FRAMES = 25  # two tracks sharing this many frames can be given one team
PAIR_BONUS = 400.0      # worth of settling one confusable pair, in 'detections'

# ---- legibility ---------------------------------------------------------------
# Set cover asks "which frames label the most of the match". It never asks whether a
# curator can READ those frames, and qm9 is what that costs: 95% of detections covered,
# 96% of boxes pre-filled, and a curator review that came back "an awful lot of
# mistakes". Coverage and legibility are different quantities and only one of them was
# in the objective.
#
# CALIBRATED AGAINST HUMAN JUDGEMENT, not invented. Curators mark a box "can't tell" when
# they cannot read it, which is ground truth for exactly this question: across seven
# curated 2026mawor matches that is 607 readable boxes against 34 unreadable ones. Every
# feature below was scored by how well it separates those two (AUC; 0.5 is a coin flip):
#
#     satw      bumper-blob width / box width       0.750     <- best single feature
#     sat       bumper-coloured fraction of band     0.748
#     whiteon   white ON the bumper blob             0.728
#     sep       separation from other boxes          0.711
#     boxh      box height                           0.533
#     blurn     log Laplacian variance               0.386     <- ANTI-predictive
#     white     white anywhere in the band           0.366     <- ANTI-predictive
#
# The first version of this scored 0.602 and weighted `white` highest at 0.45. It was the
# worst feature available and pointed the wrong way: unreadable boxes average MORE white
# than readable ones (0.079 vs 0.054), because a poorly-fitted box fills its band with
# bright background. Asking the same pixels a better question -- white ON the bumper,
# found by dilating the bumper mask -- turns 0.366 into 0.728 with no new information.
#
# Blur is a trap for the same reason: sharp background texture inflates Laplacian
# variance precisely when the bumper is NOT filling the band. Measured, not assumed.
#
# Detecting the DIGITS directly was tried and failed: connected components filtered to
# digit-like aspect and height scored 0.43-0.48, no better than chance. At ~60 px of robot
# height the bumper band is ~25 px and a digit ~15 px, which is not enough to segment.
# Whether a bumper FACE is turned toward the camera is answerable at this resolution;
# whether its digits are legible is not, so the score asks the first and infers the second.
SAT_S_MIN, SAT_V_MIN = 90, 60               # a real bumper colour, not a dim wall
RED_HUE_LO, RED_HUE_HI = 12, 168            # OpenCV hue is 0-179, red wraps
BLUE_HUE_LO, BLUE_HUE_HI = 95, 135
WHITE_S_MAX = 70        # saturation below this is "unsaturated"
WHITE_V_MIN = 165       # ...and value above this is "bright", so together: white-ish
BUMPER_TOP, BUMPER_BOTTOM = 0.55, 0.98      # just below appear.BODY_BOTTOM (0.58)

SATW_FULL = 0.85        # bumper blob spanning this much of the box reads as fully face-on
WHITEON_FULL = 0.030    # this much white on the bumper is as good as it needs to get

# Equal thirds, which measured better (0.870) than every weighting that favoured one
# term -- 0.40/0.30/0.30 gave 0.862, 0.50/0.25/0.25 gave 0.846. The three features are
# nearly independent and none deserves to dominate.
W_FACE, W_DIGITS, W_SEP = 1 / 3, 1 / 3, 1 / 3

# RELATIVE, NOT ABSOLUTE, and this was the mistake in the first version. Scored against
# a fixed scale, a robot parked at the far end of the field reads badly in every frame it
# appears in, so it only ever drags down whichever frame contains it -- the objective
# cannot tell "a bad view of this robot" from "this robot is never readable anywhere",
# and only the first is worth acting on. Measured on qm10, absolute weighting moved mean
# legibility 0.672 -> 0.703 while leaving MORE boxes below the hard-to-read line.
#
# So each track is scored against its OWN best view in the match: rel = L / max(L). A
# robot that is never legible contributes 1.0 at its best frames instead of dragging, and
# the question the objective asks becomes the right one -- is this the clearest look at
# this robot the match has to offer?
REL_FLOOR = 0.20        # coverage still matters; this is how much survives a bad view
REL_GAMMA = 2.0         # sharpen: near-best views should clearly beat merely-ok ones

# ---- best views ---------------------------------------------------------------
# The bumper number is the signal a curator actually reads, and there is no reason the
# frame they happen to be answering in has to be where they read it. So a box that is
# not near its own best gets that track's clearest crops attached, from anywhere in the
# match. The frame keeps doing the job only a frame can do -- six boxes in one frame are
# provably six different robots, which is what makes the answers mutually exclusive --
# while identification moves to the crops that can actually support it.
#
# Spread in time on purpose: three crops from one second are one view with extra steps,
# and cannot reveal the track changing robot partway, which is the failure that has cost
# the most here.
BEST_VIEWS = 3          # crops attached per box that needs help
BEST_SPREAD_S = 3.0     # minimum time between them
BEST_TRIGGER = 0.85     # attach when a box is below this fraction of its track's best
BEST_ABS_OK = 0.62      # ...or below this outright, however good its track's best is


def bumper_face(img, box: tuple) -> tuple[float, float]:
    """(face, digits) in [0,1] for one box: is a bumper turned toward the camera, and
    does it carry white where a number would be.

    Separate from _det_legibility because the two callers want different things.
    Curation needs separation folded in -- an unambiguous box matters as much as a
    readable one. Picking a REFERENCE crop of a team whose identity is already settled
    does not: there is nothing left to confuse, so overlap is irrelevant and including
    it would only bias the choice toward frames where the robot happened to be alone.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    if bw <= 1 or bh <= 1:
        return 0.0, 0.0
    by1 = int(round(y1 + bh * BUMPER_TOP))
    by2 = int(round(y1 + bh * BUMPER_BOTTOM))
    band = img[max(0, by1):min(H, by2), max(0, int(x1)):min(W, int(x2))]
    if not band.size or band.shape[0] < 3 or band.shape[1] < 3:
        return 0.0, 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strong = (sat > SAT_S_MIN) & (val > SAT_V_MIN)
    red = ((hue < RED_HUE_LO) | (hue > RED_HUE_HI)) & strong
    blue = (hue > BLUE_HUE_LO) & (hue < BLUE_HUE_HI) & strong
    # Whichever alliance colour is actually present. Never both: taking the larger keeps
    # a red bumper from being diluted by blue reflections off the field.
    mask = (red if red.mean() >= blue.mean() else blue).astype(np.uint8)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0.0, 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # WIDTH of the largest contiguous bumper blob, not its area: a bumper square to the
    # camera spans the box, one seen edge-on is a sliver. The "side-on view" question
    # asked directly.
    s_face = min(float(stats[k, cv2.CC_STAT_WIDTH]) / max(1.0, band.shape[1])
                 / SATW_FULL, 1.0)
    white = (sat < WHITE_S_MAX) & (val > WHITE_V_MIN)
    # White ON the bumper. Dilated so digits touching the blob's edge still count, and so
    # a number sitting just above a low-slung bumper is not missed.
    near = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
    s_digits = min(float(np.count_nonzero(white & near)) / float(white.size)
                   / WHITEON_FULL, 1.0)
    return s_face, s_digits


def _det_legibility(img, boxes: list[tuple]) -> list[float]:
    """Legibility in [0,1] for every box in ONE frame. See the constants above."""
    out: list[float] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            out.append(0.0)
            continue
        s_face, s_digits = bumper_face(img, (x1, y1, x2, y2))
        worst = 0.0
        for j, (a1, b1, a2, b2) in enumerate(boxes):
            if i == j:
                continue
            iw = min(x2, a2) - max(x1, a1)
            ih = min(y2, b2) - max(y1, b1)
            if iw <= 0 or ih <= 0:
                continue
            small = max(1.0, min(bw * bh, (a2 - a1) * (b2 - b1)))
            worst = max(worst, (iw * ih) / small)
        out.append(W_FACE * s_face + W_DIGITS * s_digits
                   + W_SEP * max(0.0, 1.0 - worst))
    return out


def legibility(stem: str, rows, min_robots: int = MIN_ROBOTS) -> dict:
    """{frame: {tid: legibility}} for every frame worth curating. ONE sequential pass.

    Sequential rather than seeking to each candidate: the candidates are spread over the
    whole match, so seeking to them costs more than walking, and random seeks are how
    this project once ended up describing the wrong frames entirely (see rtrack.appear).

    The walk stops at the last candidate, so it is bounded by the match, not the clip.
    """
    want = {r["f"]: r for r in rows
            if sum(1 for d in r["dets"] if d["tid"] >= 0) >= min_robots}
    if not want:
        return {}
    last = max(want)
    cap = cv2.VideoCapture(str(raw_path(stem)))
    out: dict[int, dict[int, float]] = {}
    idx = 0
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        r = want.get(idx)
        if r is not None:
            dets = [d for d in r["dets"] if d["tid"] >= 0]
            boxes = [tuple(float(v) for v in d["xyxy"]) for d in dets]
            out[idx] = {int(d["tid"]): s
                        for d, s in zip(dets, _det_legibility(img, boxes))}
        idx += 1
    cap.release()
    return out


def pick_frames(rows, n_frames: int, min_robots: int = MIN_ROBOTS,
                anchors: int = 2, min_gap_s: float = MIN_FRAME_GAP_S,
                leg: dict | None = None) -> list[int]:
    """Choose the frames that label the most of the match, by greedy set cover.

    A full-frame view is worth more per click than a per-track strip, because the
    detections in one frame are provably DIFFERENT robots: labelling six of them gives
    six mutually exclusive constraints at once, and the co-detection clashes that had
    to be detected and repaired afterwards become impossible to express.

    MEASURED on this match (101 tracks, 12165 detections), greedy set cover:

        frames   1 anchor/track   2 anchors/track
             8            76.8%             52.4%
            12            87.7%             69.3%
            20            95.9%             82.8%

    So ~12 frames x ~5 robots = 60 clicks covers 88% of the match -- comparable effort
    to the 209 labels of a per-track pass, for better-conditioned answers.

    `anchors` is the real dial. One anchor per track says who it is; two are needed to
    notice it CHANGES partway, which is the failure that has cost the most here. Two
    roughly doubles the frame count, which is affordable.

    MINIMUM SPACING IS NOT OPTIONAL. Pure set cover happily picks adjacent frames --
    it chose t9.04 and t9.11, 0.07 s apart, and counted them as two anchors on every
    track they shared. Two views of the same instant are one anchor with extra steps:
    they cannot disagree, so they cannot reveal a switch. Frames must be spread.

    `leg` ({frame: {tid: 0..1}}, from legibility()) weights each track's contribution by
    whether the curator can actually read that robot HERE. Without it the objective
    maximises coverage alone, which is how a bundle covering 95% of detections came back
    full of wrong labels: an unreadable frame scores exactly as well as a readable one.
    The weight is floored (LEGIBILITY_FLOOR) so coverage still dominates and a match with
    no good view of a robot still gets anchored rather than skipped.
    """
    ndets: dict[int, int] = defaultdict(int)
    at: dict[int, set] = {}
    t_of: dict[int, float] = {}
    co: dict[tuple, int] = defaultdict(int)
    for r in rows:
        ids = {d["tid"] for d in r["dets"] if d["tid"] >= 0}
        for t in ids:
            ndets[t] += 1
        srt = sorted(ids)
        for i, a in enumerate(srt):
            for b in srt[i + 1:]:
                co[(a, b)] += 1
        if len(ids) >= min_robots:
            at[r["f"]] = ids
            t_of[r["f"]] = r["t"]
    if not at:
        return []

    # CONFUSABLE PAIRS. Two tracks that share the screen for a long time are the ones
    # a curator can give the same team to -- not by carelessness, but because the two
    # labels are made in DIFFERENT frames and nothing on screen connects them.
    # Measured on the first frame pass: zero same-team-twice-in-one-frame mistakes,
    # and still 8 clashes, every one between tracks that co-occur elsewhere (one pair
    # for 101 frames). Showing a frame that contains BOTH turns that from an
    # after-the-fact repair into a question the curator can simply answer, because
    # the viewer will not let one team be used twice in a frame.
    pairs = {p for p, n in co.items() if n >= CONFUSABLE_FRAMES}
    unresolved = set(pairs)

    # Each track's best view anywhere in the match -- the denominator that makes
    # legibility relative rather than absolute. See REL_FLOOR.
    best_l: dict[int, float] = defaultdict(float)
    for _f, _m in (leg or {}).items():
        for _t, _s in _m.items():
            if _s > best_l[_t]:
                best_l[_t] = _s

    seen: dict[int, int] = defaultdict(int)
    chosen: list[int] = []
    for _ in range(n_frames):
        best, gain = None, -1.0
        for f, ids in at.items():
            if any(abs(t_of[f] - t_of[c]) < min_gap_s for c in chosen):
                continue
            if leg is None:
                g = float(sum(ndets[t] for t in ids if seen[t] < anchors))
            else:
                lf = leg.get(f, {})
                g = 0.0
                for t in ids:
                    if seen[t] >= anchors:
                        continue
                    bl = best_l.get(t, 0.0)
                    # A track with no legible view anywhere scores 1.0 at every frame
                    # rather than 0 -- it should not penalise the frames that carry it.
                    rel = (lf.get(t, 0.0) / bl) if bl > 1e-6 else 1.0
                    g += ndets[t] * (REL_FLOOR + (1.0 - REL_FLOOR) * rel ** REL_GAMMA)
            srt = sorted(ids)
            settles = sum(1 for i, a in enumerate(srt) for b in srt[i + 1:]
                          if (a, b) in unresolved)
            g += PAIR_BONUS * settles
            if g > gain:
                best, gain = f, g
        if best is None:
            break
        chosen.append(best)
        srt = sorted(at[best])
        for t in srt:
            seen[t] += 1
        for i, a in enumerate(srt):
            for b in srt[i + 1:]:
                unresolved.discard((a, b))
    return sorted(chosen)


def load_pipeline_rows(stem: str, tracks_p: Path, calib_stem: str | None = None):
    """Tracks segmented EXACTLY as rtrack.robots will segment them.

    Reading a previous run's labeled.jsonl instead put the bundle in a different track
    id space from the pipeline -- 31 of 96 shared detections carried different ids --
    which silently broke every piece of reasoning that crosses the boundary, including
    the conflict-aware frame choice. See robots.prepare_tracks.
    """
    from .robots import prepare_tracks
    ip = C.STAGE3_DIR / f"{stem}_identity.json"
    ident = (json.loads(ip.read_text(encoding="utf-8")) if ip.exists()
             else {"tracks": {}})
    # calib_stem MUST match what rtrack.robots was given. If robots applies the
    # off-field filter and this does not, the two disagree about what a track is,
    # and every correction anchored here lands on a different segment there.
    rows, _ = prepare_tracks(stem, tracks_p, ident, calib_stem=calib_stem)
    rows.sort(key=lambda r: r["f"])
    return rows


def plan_best_views(rows, want: list[int], leg: dict) -> tuple[dict, dict]:
    """Pick each hard-to-read box the clearest crops of that robot in the whole match.

    Returns (need, assign). `need` maps an EXTRA frame to the (tid, box) crops to cut
    while walking the video; `assign` maps a shown (frame, tid) to the frames backing it.

    Only boxes that are NOT already a good look at their robot get views attached --
    both in absolute terms and relative to what that track ever achieves. A box that is
    already the best the match offers needs no help, and attaching crops to every box
    would multiply the bundle for no gain on venue wifi.
    """
    by_f = {r["f"]: r for r in rows}
    t_of = {r["f"]: r["t"] for r in rows}
    per_track: dict[int, list] = defaultdict(list)
    for f, m in leg.items():
        for t, s in m.items():
            per_track[t].append((s, f))
    for t in per_track:
        per_track[t].sort(reverse=True)

    need: dict[int, list] = defaultdict(list)
    assign: dict[tuple, list] = {}
    for f in want:
        r = by_f.get(f)
        if r is None:
            continue
        for d in r["dets"]:
            t = d["tid"]
            if t < 0:
                continue
            here = leg.get(f, {}).get(t, 0.0)
            cand = per_track.get(t) or []
            best = cand[0][0] if cand else 0.0
            if best <= 1e-6 or (here >= BEST_ABS_OK and here >= BEST_TRIGGER * best):
                continue
            picked: list[tuple[float, int]] = []
            for s, ef in cand:
                if len(picked) >= BEST_VIEWS:
                    break
                if ef == f:
                    continue
                if any(abs(t_of.get(ef, 0.0) - t_of.get(pf, 0.0)) < BEST_SPREAD_S
                       for _s, pf in picked):
                    continue
                picked.append((s, ef))
            if not picked:
                continue
            assign[(f, t)] = [(ef, s) for s, ef in picked]
            for _s, ef in picked:
                rr = by_f.get(ef)
                if rr is None:
                    continue
                for dd in rr["dets"]:
                    if dd["tid"] == t:
                        need[ef].append((t, dd["xyxy"]))
                        break
    return need, assign


def overlay_guess(rows, guess_p: Path, max_px: float = 12.0) -> int:
    """Copy the solver's per-detection team onto rows segmented from the STITCHED file.

    Two requirements pull in opposite directions here. The curator wants the machine's
    answer, which only exists in labeled.jsonl. The pipeline needs the bundle to live in
    the same track id space rtrack.robots will rebuild, which only load_pipeline_rows
    gives (see its docstring). Handing labeled.jsonl straight to --tracks satisfies the
    first and breaks the second: prepare_tracks re-segments an already-segmented file,
    which on qm8 turned 47 tracks into 85 -- and since anchoring is per track, splitting
    tracks spreads the same frame budget thinner and weakens exactly the second anchor
    that catches an identity switch.

    So: segment from stitched, read the guess from labeled, join on frame + nearest box
    centre -- the same join rtrack.corrections.resolve uses to anchor a human label, so
    the guess and the correction that overrides it agree by construction.
    """
    idx: dict[int, list] = defaultdict(list)
    for line in guess_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for d in r["dets"]:
            if d.get("team"):
                x1, y1, x2, y2 = d["xyxy"]
                idx[int(r["f"])].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                         str(d["team"]), d.get("alliance")))
    n = 0
    for r in rows:
        cands = idx.get(int(r["f"]))
        if not cands:
            continue
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best, bd = None, None
            for gx, gy, team, alli in cands:
                d2 = (gx - cx) ** 2 + (gy - cy) ** 2
                if bd is None or d2 < bd:
                    best, bd = (team, alli), d2
            if best is not None and bd ** 0.5 <= max_px:
                d["team"] = best[0]
                if best[1] and not d.get("alliance"):
                    d["alliance"] = best[1]
                n += 1
    return n


def build_frames(stem: str, tracks_p: Path, match_key: str | None,
                 n_frames: int, anchors: int, rows=None,
                 frames: list[int] | None = None, note: str | None = None,
                 carry: list | None = None, focus: dict | None = None,
                 frame_w: int = FRAME_W, frame_q: int = FRAME_Q,
                 calib_stem: str | None = None, legible: bool = True,
                 guess_p: Path | None = None, window: bool = True) -> dict:
    """Bundle whole frames with their detections, for full-field curation.

    `frames` overrides the set-cover choice, which is what the clash follow-up uses:
    those frames are chosen to show two specific tracks together, not to cover the
    match. `carry` is any existing correction labels, passed through the viewer so a
    follow-up pass saves ONE corrections file containing both rounds rather than
    leaving the operator to merge two by hand.
    """
    if rows is None:
        rows = load_pipeline_rows(stem, tracks_p, calib_stem)
    # Clip to the match BEFORE anything else looks at the rows, so frame choice,
    # coverage and legibility all speak about the match rather than the slice. Explicit
    # `frames` are exempt: the clash follow-up names the frames it needs.
    if window and frames is None:
        w = match_window(rows)
        if w:
            t0, t1 = w
            keep = [r for r in rows if t0 <= r["t"] <= t1]
            drop = len(rows) - len(keep)
            if keep and drop:
                print(f"[curate] match window {t0:.0f}-{t1:.0f}s of "
                      f"{rows[0]['t']:.0f}-{rows[-1]['t']:.0f}s "
                      f"-- dropped {drop} frame(s) of staging and post-match")
                rows = keep
        else:
            print("[curate] match window: not readable from motion; using the whole clip")
    if guess_p is not None and guess_p.exists():
        n = overlay_guess(rows, guess_p)
        print(f"[curate] pre-fill: {n} detection(s) carry the solver's guess "
              f"from {guess_p.name}")
    by_f = {r["f"]: r for r in rows}
    # Explicit `frames` (the clash follow-up) are chosen to show two specific tracks
    # together, so legibility must not re-rank them -- it would defeat their whole point.
    leg = None
    if frames is None and legible:
        t0 = time.time()
        leg = legibility(stem, rows)
        print(f"[curate] legibility: scored {len(leg)} candidate frame(s) "
              f"in {time.time() - t0:.0f}s")
    want = frames if frames is not None else pick_frames(rows, n_frames,
                                                         anchors=anchors, leg=leg)
    want = sorted(f for f in want if f in by_f)
    if not want:
        raise SystemExit("no frame shows enough robots to be worth curating")

    ndets: dict[int, int] = defaultdict(int)
    for r in rows:
        for d in r["dets"]:
            if d["tid"] >= 0:
                ndets[d["tid"]] += 1
    total = sum(ndets.values())

    teams: dict[str, list[str]] = {}
    if match_key:
        m = tba_mod.match_by_key(match_key)
        teams = {"red": [str(t) for t in m["red"]],
                 "blue": [str(t) for t in m["blue"]]}

    # Confirmed crops of THIS match's six teams from previously curated matches, so a
    # curator can compare against known examples instead of recalling a bumper from
    # twenty minutes ago. Only these six ship: the event sheet holds every team seen so
    # far, and sending forty teams' references to identify six would be most of the
    # bundle. Written by rtrack.reid's gallery step, so it is absent for an event's first
    # match -- which the viewer handles by not offering the gesture.
    team_refs: dict[str, list] = {}
    if teams:
        rp = C.STAGE3_DIR / f"{(match_key or '').split('_')[0]}_refs.json"
        if rp.exists():
            try:
                have = json.loads(rp.read_text(encoding="utf-8")).get("teams", {})
                for t in teams["red"] + teams["blue"]:
                    if have.get(t):
                        team_refs[t] = have[t]
            except Exception as e:
                print(f"[curate] reference crops unreadable ({e}); continuing without")
        if team_refs:
            print(f"[curate] references: {sum(len(v) for v in team_refs.values())} crop(s) "
                  f"for {len(team_refs)}/6 team(s)")

    # Best-view crops come from frames scattered across the match, so they are gathered
    # in the SAME walk as the shown frames rather than a second pass -- the walk is what
    # costs, not the number of targets on it.
    need, assign = ({}, {})
    if leg:
        need, assign = plan_best_views(rows, want, leg)
    t_of = {r["f"]: r["t"] for r in rows}
    views: dict[tuple, str] = {}
    want_set = set(want)
    last = max(list(want_set) + list(need))

    out_frames: list[dict] = []
    cap = cv2.VideoCapture(str(raw_path(stem)))
    idx, i = 0, 0
    covered: set[int] = set()
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        for _t, _box in need.get(idx, ()):
            views[(idx, _t)] = _crop(img, *(float(v) for v in _box))
        if i < len(want) and idx == want[i]:
            r = by_f[idx]
            h, w = img.shape[:2]
            fw = min(frame_w, w)
            sc = fw / w
            small = cv2.resize(img, (fw, int(h * sc)),
                               interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", small,
                                    [cv2.IMWRITE_JPEG_QUALITY, frame_q])
            dets = []
            for d in r["dets"]:
                if d["tid"] < 0:
                    continue
                covered.add(d["tid"])
                x1, y1, x2, y2 = (float(v) for v in d["xyxy"])
                dets.append({
                    "tid": int(d["tid"]),
                    # Box in SHIPPED-image pixels, so the viewer needs no scale maths.
                    "box": [round(x1 * sc, 1), round(y1 * sc, 1),
                            round(x2 * sc, 1), round(y2 * sc, 1)],
                    # Anchor in ORIGINAL video pixels -- what rtrack.corrections wants.
                    "xy": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "n": ndets[d["tid"]],
                    "team": (str(d["team"]) if d.get("team") else None),
                    "alliance": d.get("alliance"),
                    # How readable this robot is HERE (white/size/separation, 0..1).
                    # Shipped so the selection can be measured after the fact rather
                    # than argued about, and so the viewer can warn on a hard box.
                    "leg": (round(leg.get(idx, {}).get(int(d["tid"]), 0.0), 3)
                            if leg else None),
                    # NATIVE-RESOLUTION crop of this robot, cut from the full frame
                    # before any downscale. The curator identifies robots from this,
                    # not from the frame -- and cropping the SHIPPED frame instead
                    # meant magnifying an already-downscaled image, so a robot ~60 px
                    # wide at 1100 px was blown up to fill a phone screen and looked
                    # like mush. Sending the crop separately buys real detail for a
                    # fraction of what raising the whole frame's resolution costs,
                    # because it is ~1% of the frame's area.
                    "crop": _crop(img, x1, y1, x2, y2),
                })
            out_frames.append({
                "f": idx, "t": round(r["t"], 2),
                # fw, NOT FRAME_W. The boxes below are scaled by `sc`, which is derived
                # from fw, so declaring the constant instead of the width actually used
                # puts every box in a different coordinate space from its own image the
                # moment --mobile (or --frame-width) changes it. At 1100 px shipped and
                # 1600 declared, boxes land 1.45x too far right and too wide: real
                # robots end up unboxed and boxes sit on background, which looks exactly
                # like the detector failing.
                "w": fw, "h": int(h * sc),
                "img": ("data:image/jpeg;base64,"
                        + base64.b64encode(buf).decode("ascii")) if ok2 else "",
                "dets": sorted(dets, key=lambda d: d["box"][0]),
            })
            i += 1
        idx += 1
    cap.release()

    # Attach the best views now the walk has cut them. Timestamps ride along because a
    # strip of crops with no times cannot show a track CHANGING robot, which is half of
    # what spreading them apart was for.
    nv = 0
    for fr in out_frames:
        for d in fr["dets"]:
            picks = assign.get((fr["f"], d["tid"]))
            if not picks:
                continue
            vs = [{"t": round(t_of.get(ef, 0.0), 1), "s": round(s, 2),
                   "crop": views[(ef, d["tid"])]}
                  for ef, s in picks if (ef, d["tid"]) in views]
            if vs:
                d["views"] = vs
                nv += len(vs)
    if leg:
        print(f"[curate] best views: {nv} extra crop(s) on "
              f"{sum(1 for fr in out_frames for d in fr['dets'] if d.get('views'))} "
              f"hard-to-read box(es)")

    cov = sum(ndets[t] for t in covered)
    shown = [d["leg"] for fr in out_frames for d in fr["dets"] if d.get("leg") is not None]
    mleg = round(sum(shown) / len(shown), 3) if shown else None
    return {"schemaVersion": 2, "mode": "frames", "video": stem, "match": match_key,
            "note": note, "carry": carry or [], "focus": focus or {},
            "teams": teams, "teamRefs": team_refs, "totalDetections": total,
            "framesShown": len(out_frames), "tracksTouched": len(covered),
            "tracksTotal": len(ndets), "coveragePct": round(100.0 * cov / total, 1),
            "meanLegibility": mleg,
            "anchors": anchors, "frames": out_frames}


def build(stem: str, tracks_p: Path, match_key: str | None,
          max_tracks: int, per_track: int, calib_stem: str | None = None) -> dict:
    rows = load_pipeline_rows(stem, tracks_p, calib_stem)

    ndets: dict[str, int] = defaultdict(int)
    span: dict[str, list] = {}
    hint: dict[str, str] = {}
    alli: dict[str, defaultdict] = {}
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            k = str(d["tid"])
            ndets[k] += 1
            s = span.setdefault(k, [r["t"], r["t"]])
            s[0], s[1] = min(s[0], r["t"]), max(s[1], r["t"])
            if d.get("team"):
                hint[k] = str(d["team"])
            a = alli.setdefault(k, defaultdict(int))
            if d.get("alliance"):
                a[d["alliance"]] += 1

    # Which tracks the last solve could not place, and why. A track that is co-detected
    # with a member of EVERY group can never be assigned as a whole, no matter how it
    # is labelled -- the co-detection constraint is per frame but disqualifies a track
    # for its entire life, so the longer the track the more certain it is to brush past
    # all six groups. Measured: track 35 (1077 detections, 85 s) was blocked from all
    # six, yet at no instant were more than five groups even present -- a slot was free
    # in 44-67% of its frames.
    #
    # Asking "which team is this whole track?" about one of these is the wrong
    # question. The useful answer is WHERE IT CHANGES, so the viewer prompts for
    # per-crop labels instead, which become cut instructions.
    parked: set[int] = set()
    blocked: set[int] = set()
    rp = C.STAGE3_DIR / f"{stem}_robots.json"
    if rp.exists():
        try:
            rdoc = json.loads(rp.read_text(encoding="utf-8"))
            parked = {int(t) for t in rdoc.get("parked", [])}
            groups = [set(g["tracks"]) for g in rdoc.get("groups", {}).values()]
            if groups:
                from .robots import conflicts
                con = conflicts(rows)
                blocked = {int(k) for k in ndets
                           if all(con.get(int(k), set()) & ms for ms in groups)}
        except Exception:
            pass

    keep = [k for k, _ in sorted(ndets.items(), key=lambda kv: -kv[1])][:max_tracks]
    keep_set = {int(k) for k in keep}
    t_lo = min(r["t"] for r in rows)
    t_hi = max(r["t"] for r in rows)

    # Reuse the chicklet pipeline verbatim -- same crops the operator has been
    # inspecting, so what the curator sees is what we have been judging by.
    picks = plan(rows, t_lo, t_hi, per_track, by_track=True)
    picks = {k: v for k, v in picks.items() if int(k) in keep_set}
    chosen = choose(grab(stem, picks, rows), per_track)

    teams: dict[str, list[str]] = {}
    if match_key:
        m = tba_mod.match_by_key(match_key)
        teams = {"red": [str(t) for t in m["red"]],
                 "blue": [str(t) for t in m["blue"]]}

    ident_p = C.STAGE3_DIR / f"{stem}_identity.json"
    tallies = (json.loads(ident_p.read_text(encoding="utf-8"))["tracks"]
               if ident_p.exists() else {})

    out_tracks = []
    total = sum(ndets.values())
    run = 0
    for k in keep:
        cands = chosen.get(k, [])
        if not cands:
            continue
        run += ndets[k]
        a = alli.get(k, {})
        tid = int(k)
        dur = span[k][1] - span[k][0]
        why = ("co-detected with every robot group, so it cannot be one robot"
               if tid in blocked else
               "could not be placed in the last solve" if tid in parked else
               f"{dur:.0f}s long -- long tracks often span more than one robot"
               if dur >= LONG_TRACK_S else None)
        out_tracks.append({
            "tid": tid,
            "n": ndets[k],
            "t0": round(span[k][0], 1),
            "t1": round(span[k][1], 1),
            "durS": round(dur, 1),
            "blocked": tid in blocked,
            "parked": tid in parked,
            "splitAdvised": bool(why),
            "splitWhy": why,
            "alliance": (max(a, key=a.get) if a else None),
            "hint": hint.get(k),
            "tally": tallies.get(k, {}).get("tally", {}),
            "cumPct": round(100.0 * run / total, 1),
            # anchor: the label a curator gives this track is stored against THIS
            # detection, not against the track id -- see rtrack.corrections.
            "crops": [{"f": int(c["f"]), "t": round(float(c["t"]), 1),
                       "xy": [int(c["cx"]), int(c["cy"])],
                       "img": encode(c["img"])} for c in cands],
        })

    return {"schemaVersion": 1, "video": stem, "match": match_key,
            "teams": teams, "totalDetections": total,
            "tracksShown": len(out_tracks), "tracksTotal": len(ndets),
            "splitAdvised": sum(1 for t in out_tracks if t["splitAdvised"]),
            "cropRegion": [CROP_TOP, CROP_BOTTOM],
            "tracks": out_tracks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a curator bundle.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--match", default=None)
    ap.add_argument("--max-tracks", type=int, default=MAX_TRACKS)
    ap.add_argument("--per-track", type=int, default=CROPS_PER_TRACK)
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="build a FULL-FRAME bundle of N frames instead of per-track "
                         "crop strips: one frame shows every robot at one instant, so "
                         "the labels are mutually exclusive by construction")
    ap.add_argument("--anchors", type=int, default=2,
                    help="with --frames, how many times each track should be seen. "
                         "1 says who it is; 2 is needed to notice it CHANGES.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--calib-from", default=None, metavar="VIDEO",
                    help="must match what rtrack.robots is given, or the bundle and the "
                         "solver end up in different track id spaces")
    ap.add_argument("--mobile", action="store_true",
                    help="smaller frames for a phone on venue wifi (800px/Q45). Safe "
                         "because identification happens in the per-detection crop, "
                         "which is cut from the FULL-resolution frame regardless -- "
                         "the frame only has to show where a box is and be tappable.")
    ap.add_argument("--no-window", dest="window", action="store_false",
                    help="do not clip to the match; curate the whole slice, staging and post-match included. See match_window().")
    ap.add_argument("--guess", type=Path, default=None, metavar="LABELED",
                    help="labeled.jsonl to pre-fill the curator with. Pass this rather "
                         "than feeding it to --tracks: see overlay_guess().")
    ap.add_argument("--no-legibility", dest="legible", action="store_false",
                    help="pick frames by coverage alone, ignoring whether the curator "
                         "can read them. Costs one extra sequential decode to score, "
                         "so this is the escape hatch if that ever matters.")
    ap.add_argument("--frame-width", type=int, default=None)
    ap.add_argument("--frame-quality", type=int, default=None)
    args = ap.parse_args(argv)
    fw = args.frame_width or (800 if args.mobile else FRAME_W)
    fq = args.frame_quality or (45 if args.mobile else FRAME_Q)

    C.ensure_dirs()
    stem = video_id(args.video)
    tp = args.tracks or (C.STAGE1_DIR / "MATCH3_st.jsonl")
    if not tp.exists():
        raise SystemExit(f"{tp} not found -- pass --tracks with the stitched "
                         f"Stage 1 track file (NOT a labeled.jsonl: the bundle must "
                         f"be segmented the same way rtrack.robots segments it)")

    if args.frames:
        doc = build_frames(stem, tp, args.match, args.frames, args.anchors,
                           frame_w=fw, frame_q=fq,
                           calib_stem=args.calib_from, legible=args.legible,
                           guess_p=args.guess, window=args.window)
        out = args.out or (C.STAGE3_DIR / f"{stem}_curate_frames.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc), encoding="utf-8")
        print(f"[curate] {doc['framesShown']} frames, "
              f"{doc['tracksTouched']}/{doc['tracksTotal']} tracks touched, "
              f"{doc['coveragePct']}% of detections, "
              + (f"legibility {doc['meanLegibility']:.2f}, "
                 if doc.get("meanLegibility") is not None else "") +
              f"{out.stat().st_size / 1e6:.1f} MB -> {out}")
        print("[curate] open viewer/curate.html and drag this file in")
        return 0

    doc = build(stem, tp, args.match, args.max_tracks, args.per_track,
                calib_stem=args.calib_from)
    out = args.out or (C.STAGE3_DIR / f"{stem}_curate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    mb = out.stat().st_size / 1e6
    covered = doc["tracks"][-1]["cumPct"] if doc["tracks"] else 0.0
    print(f"[curate] {doc['tracksShown']}/{doc['tracksTotal']} tracks, "
          f"{covered:.0f}% of detections, {mb:.1f} MB -> {out}")
    print("[curate] open viewer/curate.html and drag this file in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
