"""Stage 2a -- is the camera still in the pose the homography was fitted to?

    uv run -m rtrack.viewcheck 2026necmp1_qm1 --calib-from 2026necmp1

WHY THIS EXISTS. A static homography is a statement about ONE camera pose. The moment
the broadcast cuts to a corner camera, zooms, or pans to follow a robot, every position
derived from it is wrong -- and wrong quietly, by a metre or two, with no residual to
give it away. rtrack.calibrate can only report how well the fit matched the points it
was given; it cannot know the camera has since moved.

WHAT THIS IS NOT. rtrack.shots measures frame-to-frame motion, which finds TRANSITIONS.
That is a different question and it cannot answer this one: after a cut away and a cut
back, differential motion has seen two transitions and has no idea which side of them
is the calibrated view. This measures ABSOLUTE agreement with the calibration frame, so
it says "valid" again the moment the broadcast returns, and drift cannot accumulate.

HOW, IN TWO LAYERS. Patches of the reference frame ("anchors") are matched in every
sampled frame by normalised cross-correlation, which is illumination-invariant to first
order; if they are all found where they were, the camera has not moved. Alongside that,
a whole-frame thumbnail correlation asks the coarser question of whether this is even
the same scene.

BOTH ARE NEEDED, AND NEITHER IS ALLOWED TO DECIDE ALONE.

  Anchors alone are blind to a CUT. When the broadcast switches camera the scene is
  entirely different, so the few anchors still clearing MATCH_OK are matching noise --
  and noise looks exactly like partial occlusion. Measured on 2026necmp1:

      qm8 real cuts    3.0 of 16 anchors, median NCC 0.69, shift 39.0, spread 26.9
      true occlusion   3.0 of 16 anchors, median NCC 0.70, shift 21.0, spread 24.0

  Identical on every anchor feature. The first version called both "unknown" and let
  the valid span inherit through three real cuts in qm8 and six more in qm1.

  The thumbnail alone is blind to an OVERLAY. qm14 and qm15 are screen captures of the
  FMS Audience Display inside a browser window, so chrome and scoreboard cover a third
  of the frame; the thumbnail reads 0.35 while every field anchor sits at (0,-2). Judged
  on that alone, qm15 went to 0% valid -- a whole match discarded on evidence that was
  never about geometry.

So: strong anchor agreement is decisive and outranks the thumbnail, because it measures
the camera's pose directly. Only when the anchors do NOT agree does the thumbnail get to
say whether the scene changed at all.

ANCHOR CHOICE IS THE WHOLE PROBLEM, and the obvious choice is the wrong one. The first
version used the CALIBRATION POINTS, which seemed principled -- they are the very
features the homography rests on. Measured on 2026necmp1_qm1, a match with a locked-off
camera, it flagged 26 of 120 samples as "camera moved". Every one was a false positive,
because calibration points sit on the PLAYING SURFACE by construction and robots drive
over them. The tell was in the shape of the failures:

    static samples   median 7/11 anchors matched, 2.0 px spread
    flagged samples  median 2/11 anchors matched, 26.9 px spread

A real camera move is MANY anchors agreeing on ONE shift. That was few anchors
disagreeing wildly -- the signature of occlusion, not motion.

So anchors are chosen automatically for TEMPORAL STABILITY instead: grid the reference
frame, keep patches with real texture whose appearance at that exact spot barely changes
across probe frames spread through the clip. Anything robots drive over fails that test
by definition, so the survivors are field structure -- truss, perimeter, banners,
scoring table. Same match, same code, after the change: median 16/16 anchors matched,
0.0 px shift, 0.0 px spread, and ZERO false positives.

This also makes the "have the curator box some static regions" idea unnecessary. It was
a sound instinct -- it is exactly the right criterion -- and the reason it is not needed
is that the criterion can be measured directly. A curator's time is better spent on
identity, which nothing can measure for them.

STATES. Like rtrack.scoreboard this refuses to force a binary answer: 'ok', 'cut',
'moved', 'blank', or None for "cannot tell". Callers must not read None as either
answer -- valid_intervals lets it inherit rather than guess.

WHAT THIS FOUND. The Burns broadcast cuts to close-angle cameras several times per
match, which nobody had noticed: qm1 spends 25% of its clip on another camera, qm8 14%,
qm22 26%. Every one of those seconds was previously being projected through a homography
that did not apply, silently, and the positions were wrong by metres with nothing in the
output to say so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path

# Patch and search half-sizes in pixels, at source resolution. SEARCH bounds how far a
# move can be measured: beyond it the match saturates and the reported shift reads low.
# That is a safe direction to fail -- a move larger than 56 px is still far over every
# threshold below.
PATCH, SEARCH = 32, 56

# Anchor selection.
PROBE_FRAMES = 10           # spread through the clip, to judge stability
GRID_X, GRID_Y = 16, 9
KEEP = 24                   # upper bound; typically 12-20 survive both filters
MIN_TEXTURE = 60.0          # Laplacian variance. Flat sky or floor matches anywhere.
MIN_STABLE = 0.80           # median NCC at the SAME spot across probe frames

MATCH_OK = 0.55             # NCC above which an anchor counts as found

# THRESHOLDS, IN FIELD UNITS. 2026necmp1_qm1 calibrates at 1.2 cm/px median, and its own
# worst-case reprojection error is 7.5 cm. SHIFT_PX of 6 is 7.2 cm there -- i.e. "the
# camera has moved by more than the calibration was ever accurate to". Below that the
# move is inside the noise the positions already carry.
#
# Verified by injecting known moves into a real frame: 2 px read 2.0, 5 px read 5.0,
# 10,5 px read 11.2 (true 11.18), 25,10 read 26.9 (true 26.9). Exact, so the threshold
# separates cleanly rather than sitting in a smear.
SHIFT_PX = 6.0

# A ZOOM OR ROTATION MOVES ANCHORS RADIALLY, so the median shift stays near zero while
# they disagree with each other. Injected zooms confirm it: 1.02x gives 3.2 px shift but
# 8.6 px spread, 1.05x gives 6.7/20.8, 1.10x gives 13.4/33.0. Spread catches what the
# median cannot, and without it a zoomed broadcast reads as perfectly static.
SPREAD_PX = 6.0

# Fraction of anchors that must be found before a shift verdict is offered at all.
QUORUM = 0.40

# GLOBAL SIMILARITY, AND WHY IT IS THE FIRST TEST RATHER THAN A REFINEMENT.
#
# Anchors are precise and they are blind to the one case that matters most. When the
# broadcast CUTS to another camera the scene is entirely different, so the handful of
# anchors that still clear MATCH_OK are matching noise -- and noise is indistinguishable
# from partial occlusion. Measured on 2026necmp1, real cuts against genuine occlusion:
#
#                      anchors ok   median NCC   shift   spread
#     qm8 real cuts        3.0          0.69      39.0    26.9
#     occlusion            3.0          0.70      21.0    24.0
#
# Identical. No threshold on those features can separate them, and the first version of
# this module called both "unknown" and let the valid span inherit straight through
# three real camera cuts in qm8 alone.
#
# A whole-frame comparison separates them immediately, because occlusion changes a small
# fraction of the frame and a cut changes all of it:
#
#     qm8 cuts (t140,156,210)   thumbnail NCC 0.475 - 0.593
#     qm1 cut  (t172)                         0.570 - 0.574
#     genuine occlusion                       0.859 - 0.863
#     ordinary good frames                    0.792 - 0.835
#
# A gap from 0.60 to 0.79 with nothing in it. So: global similarity asks "is this the
# same scene at all", and only if it is do the anchors get asked "has it shifted".
SIM_MIN = 0.70
SIM_HOLD_S = 4.0            # a cut lasts; a one-frame dissolve does not

# BUT GLOBAL SIMILARITY IS A PROXY, AND ANCHORS OUTRANK IT WHEN THEY AGREE.
#
# The thumbnail compares the WHOLE frame, including everything the broadcast paints on
# top of the camera. 2026necmp1 qm14 and qm15 are screen captures of the FMS Audience
# Display inside a browser window: chrome, tabs and a large scoreboard occupy a third of
# the image, so the thumbnail reads 0.35 against qm1 -- far below SIM_MIN -- while the
# camera has not moved at all. Per-anchor on a qm15 frame:
#
#     every FIELD anchor       NCC 0.82-0.99 at (0,-2) to (0,-3)
#     the four OVERLAY anchors NCC 0.26-0.76, scattered
#
# The field is at identical pixel coordinates. Judging that frame by its thumbnail threw
# away a whole match -- qm15 went to 0% valid -- on evidence that was never about
# geometry. Anchor agreement is the specific measurement and it wins.
#
# It cannot be abused to rescue a real cut, because a cut does not produce agreement:
# qm8's and qm1's cut samples sit at 3-4 of 16 anchors with 22-39 px of shift, nowhere
# near this bar, and zero of them are rescued by it.
STRONG_FRAC = 0.60

# AT OR BELOW THIS FRACTION FOUND, the frame does not resemble the calibrated view in
# any part, which is different in kind from "some anchors are occluded". Partial loss is
# robots and people in front of the camera; total loss is a different picture -- a corner
# camera, a full-screen graphic, a results screen.
#
# It is not decisive on its own, because a replay wipe also blanks everything for a
# second or two, so it must be SUSTAINED (see BLANK_HOLD_S). Without this rule a cut to
# an entirely different camera reads as "unknown" forever and, because unknown inherits,
# the span before it stays valid straight through -- which is the one failure this
# module exists to prevent. Measured on 2026necmp1_qm14: 66 samples at zero anchors,
# all of them the post-match results screen, correctly excluded.
BLANK_FRAC = 0.10
BLANK_HOLD_S = 6.0

# Hysteresis. A single sample is not a camera move: a foreground robot crossing an
# anchor, one frame of a replay wipe, a flash of pyro. Requires this many consecutive
# samples to change state, in either direction.
HOLD = 2

SAMPLE_S = 2.0              # how often to test. The question changes on the scale of
                            # broadcast shots, not frames.


def _probe_frames(video: Path, n: int = PROBE_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, img = cap.read()
        if ok:
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def _thumb(gray: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance 64x36, so the dot product IS normalised correlation."""
    t = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def dominant(frames: list[np.ndarray]) -> list[np.ndarray]:
    """The probe frames that agree with each other -- i.e. the shot the camera mostly is.

    THE PROBE FRAMES ARE NOT ALL TRUSTWORTHY. They are taken at even intervals through
    the clip, and a clip with cuts in it will hand some of them to a different camera.
    Anchors picked from such a frame, or a reference thumbnail set that includes one,
    would then accept that other camera as valid -- the check would certify the very
    thing it exists to catch.

    So the reference is the largest mutually-similar group: every frame scored by how
    many others it resembles, and the best one's neighbourhood kept. On a locked-off
    broadcast that is all of them and this costs nothing.
    """
    if len(frames) < 3:
        return frames
    ths = [_thumb(f) for f in frames]
    n = len(ths)
    sim = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = float((ths[i] * ths[j]).mean())
    votes = (sim >= SIM_MIN).sum(axis=1)
    best = int(np.argmax(votes))
    keep = [i for i in range(n) if sim[best, i] >= SIM_MIN]
    return [frames[i] for i in keep] or frames


def pick_anchors(frames: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]]:
    """Textured patches that do not change over the clip -- i.e. not the playing field."""
    if not frames:
        return []
    ref = frames[0]
    H, W = ref.shape
    lo = PATCH + SEARCH
    cands = []
    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            x = int(W * (gx + 0.5) / GRID_X)
            y = int(H * (gy + 0.5) / GRID_Y)
            if not (lo <= x < W - lo and lo <= y < H - lo):
                continue
            p = ref[y - PATCH:y + PATCH, x - PATCH:x + PATCH]
            tex = float(cv2.Laplacian(p, cv2.CV_64F).var())
            if tex < MIN_TEXTURE:
                continue
            sc = [float(cv2.matchTemplate(f[y - PATCH:y + PATCH, x - PATCH:x + PATCH],
                                          p, cv2.TM_CCOEFF_NORMED)[0][0])
                  for f in frames[1:]]
            st = float(np.median(sc)) if sc else 0.0
            if st < MIN_STABLE:
                continue
            # Stability first, texture as a capped tie-break: past a point more texture
            # does not make a patch easier to find, and uncapped it drags selection
            # toward a few very busy spots that may sit together in one corner.
            cands.append((st * min(tex, 2000.0), x, y, p))
    cands.sort(key=lambda c: -c[0])
    return [(x, y, p) for _s, x, y, p in cands[:KEEP]]


def score(gray: np.ndarray, anchors, refs: list[np.ndarray] | None = None) -> dict:
    """Global similarity plus where the anchors are, relative to where they were.

    `refs` are reference thumbnails; similarity is the BEST match against any of them,
    because a static shot still changes legitimately over a match -- balls drain from
    the pile, robots move, the crowd shifts -- and one reference instant would penalise
    that as though the camera had moved.
    """
    sim = None
    if refs:
        t = _thumb(gray)
        sim = round(float(max((t * r).mean() for r in refs)), 3)
    ds, ss = [], []
    for (x, y, patch) in anchors:
        win = gray[y - PATCH - SEARCH:y + PATCH + SEARCH,
                   x - PATCH - SEARCH:x + PATCH + SEARCH]
        if win.shape[0] < patch.shape[0] or win.shape[1] < patch.shape[1]:
            continue
        r = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, loc = cv2.minMaxLoc(r)
        ds.append((loc[0] - SEARCH, loc[1] - SEARCH))
        ss.append(mx)
    n = len(ds)
    if not n:
        return {"n": 0, "nOk": 0, "shift": None, "spread": None,
                "score": None, "sim": sim}
    ds = np.asarray(ds, float)
    ss = np.asarray(ss, float)
    good = ss >= MATCH_OK
    n_ok = int(good.sum())
    if n_ok == 0:
        return {"n": n, "nOk": 0, "shift": None, "spread": None,
                "score": round(float(ss.max()), 3), "sim": sim}
    med = np.median(ds[good], axis=0)
    return {
        "n": n, "nOk": n_ok,
        "shift": round(float(np.hypot(*med)), 2),
        "spread": round(float(np.median(np.hypot(*(ds[good] - med).T))), 2),
        "score": round(float(np.median(ss[good])), 3),
        "sim": sim,
    }


def verdict(row: dict) -> str | None:
    """'ok' | 'moved' | 'blank', or None for 'cannot tell'. Never guesses.

    'blank' and None are both "no usable measurement", and they are kept apart because
    they mean opposite things about the evidence: None is a partly obscured view of the
    right scene, 'blank' is not the right scene at all.
    """
    # STRONG ANCHOR AGREEMENT SETTLES IT, whatever the thumbnail says. Most of the
    # anchors found, all within a few pixels of home, all agreeing with each other, is
    # direct evidence about the camera's pose -- which is the actual question. See
    # STRONG_FRAC for the overlay case this exists to survive.
    if (row["n"] and row["nOk"] >= STRONG_FRAC * row["n"]
            and row["shift"] is not None and row["shift"] <= SHIFT_PX
            and row["spread"] is not None and row["spread"] <= SPREAD_PX):
        return "ok"
    # Otherwise global similarity decides whether this is even the same scene. If it is
    # not, nothing the remaining anchors say means anything -- they are reporting where
    # noise happened to correlate.
    if row.get("sim") is not None and row["sim"] < SIM_MIN:
        return "cut"
    if not row["n"]:
        return None
    if row["nOk"] <= BLANK_FRAC * row["n"]:
        return "blank"
    if row["nOk"] < QUORUM * row["n"]:
        return None
    if row["spread"] is not None and row["spread"] > SPREAD_PX:
        return "moved"      # radial disagreement: a zoom or rotation, not occlusion
    if row["shift"] is not None and row["shift"] > SHIFT_PX:
        return "moved"
    return "ok"


def valid_intervals(rows: list[dict], hold: int = HOLD,
                    sample_s: float = SAMPLE_S) -> list[tuple[float, float]]:
    """Spans over which the homography may be trusted.

    THREE KINDS OF EVIDENCE, WEIGHTED DIFFERENTLY.

      ok      -> valid, after `hold` consecutive samples
      moved   -> invalid, after `hold` consecutive samples
      cut     -> invalid, after SIM_HOLD_S. The whole frame stopped resembling the
                 calibrated shot, so the broadcast is on another camera.
      blank   -> invalid, but only after BLANK_HOLD_S of it. A replay wipe or a score
                 graphic blanks the anchors for a second or two and means nothing; a
                 minute of it is a different camera or a different screen.
      None    -> INHERITS. A partly occluded view of the right scene is not evidence
                 that the scene changed. Treating it as such would punch holes in every
                 route for reasons that have nothing to do with geometry -- 2026necmp1
                 qm1/qm2/qm8 carry 12-25 such samples each, every one a robot crossing
                 an anchor.

    Hysteresis in BOTH directions, so a single bad sample neither opens nor closes a
    span. The run counter is only reset by a sample that actually disagrees, so an
    alternating ok/unknown stretch still latches.
    """
    if not rows:
        return []
    blank_hold = max(1, int(round(BLANK_HOLD_S / max(sample_s, 1e-6))))
    cut_hold = max(1, int(round(SIM_HOLD_S / max(sample_s, 1e-6))))
    state = "ok"            # assume valid until something says otherwise
    start = rows[0]["t"]
    out: list[tuple[float, float]] = []
    pend, run, run_start = None, 0, rows[0]["t"]
    last_ok = rows[0]["t"]
    for r in rows:
        v = r.get("verdict")
        if v is None:
            continue                      # inherit
        if v == "ok":
            last_ok = r["t"]
        want = "ok" if v == "ok" else "bad"
        need = {"blank": blank_hold, "cut": cut_hold}.get(v, hold)
        if want == state:
            pend, run = None, 0
            continue
        if pend != v:
            pend, run, run_start = v, 1, r["t"]
        else:
            run += 1
        if run < need:
            continue
        if want == "ok":
            # The FIRST sample of the confirming run, not the last. Every sample in it
            # was observed ok, so including them is not optimism -- and dropping them
            # made spans asymmetric once ends moved to the last observed-good sample,
            # which collapsed short valid islands between two cuts to zero length.
            start = run_start
        elif start is not None:
            # End at the LAST SAMPLE ACTUALLY OBSERVED GOOD, not at the first bad one.
            #
            # Closing on the first bad sample looks right -- the camera moved when the
            # run began -- but is_valid_at tests `a <= t <= b` inclusively, so that
            # endpoint then reads as valid and exactly one cut sample leaks through
            # every boundary. 2026necmp1_qm7 put such a frame straight into a curation
            # bundle at t=102, the first sample of a cut, and a curator was shown a
            # robot from another camera.
            #
            # Span STARTS were already conservative -- they latch on the last sample of
            # a good run, discarding the first `hold - 1` -- so this makes both ends
            # agree: a span covers only what was measured, never a sample either side.
            if last_ok > start:
                out.append((start, last_ok))
            start = None
        state, pend, run = want, None, 0
    if state == "ok" and start is not None:
        out.append((start, rows[-1]["t"]))
    # A span that opens on the last sample has no duration and no content; it appears
    # when the view returns right at the end of a clip. Reporting it would claim
    # validity for an instant nothing was measured across.
    return [(a, b) for a, b in out if b - a >= sample_s]


def view_path(stem: str) -> Path:
    return C.STAGE2_DIR / f"{stem}_view.json"


def run(stem: str, calib_stem: str | None = None, sample_s: float = SAMPLE_S) -> Path:
    """Anchors come from the CALIBRATION clip, the check runs on this one.

    They are different videos whenever a calibration is reused across matches, which is
    the normal case -- and they must be, because the question is whether THIS match's
    camera agrees with the pose the homography was fitted in, not with itself.
    """
    src = raw_path(stem)
    ref = raw_path(calib_stem or stem)
    if not Path(ref).exists():
        raise SystemExit(f"[viewcheck] no clip for the calibration source {calib_stem}")
    probes = _probe_frames(Path(ref))
    # Both the anchors and the reference thumbnails come from the DOMINANT shot, never
    # from whatever frame an even-interval probe happened to land on.
    dom = dominant(probes)
    if len(dom) < len(probes):
        print(f"[viewcheck] {len(dom)}/{len(probes)} probe frames agree; "
              f"the rest are other shots and are not used as reference")
    refs = [_thumb(f) for f in dom]
    anchors = pick_anchors(dom)
    if len(anchors) < 6:
        print(f"[viewcheck] only {len(anchors)} stable anchor(s) in {calib_stem or stem}"
              f" -- not enough to judge; every sample will read 'unknown'")
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * sample_s)))
    rows, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                r = score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), anchors, refs)
                r["t"] = round(i / fps, 2)
                r["verdict"] = verdict(r)
                rows.append(r)
        i += 1
    cap.release()

    spans = valid_intervals(rows, sample_s=sample_s)
    n_ok = sum(1 for r in rows if r["verdict"] == "ok")
    n_mv = sum(1 for r in rows if r["verdict"] == "moved")
    n_bl = sum(1 for r in rows if r["verdict"] == "blank")
    n_ct = sum(1 for r in rows if r["verdict"] == "cut")
    n_un = len(rows) - n_ok - n_mv - n_bl - n_ct
    doc = {
        "schemaVersion": 1, "video": stem, "calibFrom": calib_stem or stem,
        "anchors": len(anchors), "sampleS": sample_s,
        "thresholds": {"simMin": SIM_MIN, "simHoldS": SIM_HOLD_S,
                       "shiftPx": SHIFT_PX, "spreadPx": SPREAD_PX,
                       "quorum": QUORUM, "hold": HOLD,
                       "blankFrac": BLANK_FRAC, "blankHoldS": BLANK_HOLD_S},
        "counts": {"ok": n_ok, "moved": n_mv, "cut": n_ct,
                   "blank": n_bl, "unknown": n_un},
        "validIntervals": [[round(a, 2), round(b, 2)] for a, b in spans],
        "samples": rows,
    }
    out = view_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    total = rows[-1]["t"] if rows else 0.0
    covered = sum(b - a for a, b in spans)
    print(f"[viewcheck] {len(anchors)} anchors, {len(rows)} samples: "
          f"{n_ok} ok, {n_mv} moved, {n_ct} cut, {n_bl} blank, {n_un} unknown")
    print(f"[viewcheck] {len(spans)} valid span(s) covering "
          f"{covered:.0f}s of {total:.0f}s ({100 * covered / max(total, 1):.0f}%)")
    for a, b in spans:
        print(f"[viewcheck]   {a:7.1f} - {b:7.1f}s")
    print(f"[viewcheck] -> {out}")
    return out


def load(stem: str) -> dict | None:
    p = view_path(stem)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_valid_at(doc: dict | None, t: float) -> bool:
    """No document means no evidence of a problem, so positions are kept.

    A check that has not run must not silently delete data -- the same rule the
    scoreboard check follows. Absence of the file is absence of a finding.
    """
    if not doc:
        return True
    spans = doc.get("validIntervals") or []
    if not spans:
        return True
    return any(a <= t <= b for a, b in spans)


def match_coverage(stem: str, calib_stem: str | None = None) -> dict | None:
    """Coverage of the MATCH, which is the only coverage that means anything.

    Coverage of the whole clip mixes in staging, the post-match results screen and
    whatever the broadcast did before the slice started, and those vary by minutes
    between clips. On 2026necmp1 the two numbers disagree badly and in both directions:
    qm22 reads 74% of its clip and 100% of its match; qm8 reads 86% and 76%. Judging
    route quality by the clip figure would have condemned three clean matches and
    excused the one that actually lost data.
    """
    from .curate import load_pipeline_rows, match_window      # local: avoids a cycle
    doc = load(stem)
    labeled = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
    if not doc or not labeled.exists():
        return None
    try:
        win = match_window(load_pipeline_rows(stem, labeled, calib_stem))
    except Exception:
        return None
    if not win:
        return None
    spans = [tuple(s) for s in (doc.get("validIntervals") or [])]
    inside = sum(max(0.0, min(b, win[1]) - max(a, win[0])) for a, b in spans)
    cuts = [r["t"] for r in doc["samples"]
            if r.get("verdict") == "cut" and win[0] <= r["t"] <= win[1]]
    return {
        "match": stem, "window": [round(win[0], 1), round(win[1], 1)],
        "windowS": round(win[1] - win[0], 1),
        "validS": round(inside, 1),
        "pct": round(100 * inside / max(win[1] - win[0], 1e-6), 1),
        "cutSamplesInMatch": len(cuts),
    }


def report(event: str, calib_stem: str | None = None) -> int:
    import re
    stems = sorted(C.STAGE2_DIR.glob(f"{event}*_view.json"),
                   key=lambda p: int(re.search(r"(\d+)_view", p.name).group(1))
                   if re.search(r"(\d+)_view", p.name) else 0)
    rows = []
    print(f"{'match':<9}{'window':>16}{'len':>7}{'valid':>8}{'cov':>7}  cut samples in match")
    for p in stems:
        stem = p.name[: -len("_view.json")]
        c = match_coverage(stem, calib_stem)
        if not c:
            print(f"{stem.split('_')[-1]:<9}(no match window or no labelled tracks)")
            continue
        rows.append(c)
        print(f"{stem.split('_')[-1]:<9}{c['window'][0]:7.0f}-{c['window'][1]:<8.0f}"
              f"{c['windowS']:7.0f}{c['validS']:7.0f}s{c['pct']:6.0f}%"
              f"{c['cutSamplesInMatch']:>7}")
    if rows:
        tw = sum(r["windowS"] for r in rows)
        tv = sum(r["validS"] for r in rows)
        clean = sum(1 for r in rows if r["pct"] >= 99.5)
        print(f"\n[viewcheck] {len(rows)} match(es): {tv:.0f}s valid of {tw:.0f}s of "
              f"match time ({100 * tv / tw:.1f}%); {clean} with no cuts at all")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Check whether the camera stays in its calibrated pose.")
    ap.add_argument("stem", nargs="?",
                    help="clip to check; omit when using --report")
    ap.add_argument("--calib-from", default=None,
                    help="clip the calibration was fitted on; anchors come from there")
    ap.add_argument("--sample-s", type=float, default=SAMPLE_S)
    ap.add_argument("--report", metavar="EVENT", default=None,
                    help="summarise coverage OF THE MATCH for every checked clip of an "
                         "event, rather than coverage of the clip")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    if args.report:
        return report(args.report, args.calib_from)
    if not args.stem:
        ap.error("give a clip to check, or --report EVENT")
    run(args.stem, args.calib_from, args.sample_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
