"""Stage 3 -- per-detection appearance descriptors, cached to disk.

    uv run -m rtrack.appear GSxbsE42o5o --tracks out/stage1/MATCH3_st.jsonl

Exists to catch the identity switches that bumper hue cannot see. Splitting tracks on
hue fixed the cross-alliance chimeras -- 13 of 31 tracks were changing alliance
mid-life, 72% of detections -- but hue is blind to WHICH of the three blue robots it
is looking at, and the chicklet sheets show that within-alliance switches remain.

The descriptor deliberately excludes the bumper band. Inside one alliance the bumper
is the same colour on all three robots, so including it would add a large constant
term to every comparison and bury the differences we need. What actually distinguishes
FRC robots at this resolution is the superstructure: hopper, intake, elevator, the
colour and layout of the mechanisms above the bumper.

Written to an .npz because the descriptors cost one full video decode (~90 s) and the
change-point threshold needs tuning against them. Re-deriving them per experiment
would repeat the same mistake the project already made once with random frame seeking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id

# Superstructure only: from the top of the box down to just above the bumper.
BODY_TOP, BODY_BOTTOM = 0.02, 0.58
GRAY_BINS = 16      # luminance bins per band; coarse on purpose, see descriptor()
BANDS = 3           # horizontal bands; see descriptor()
STRIDE = 2          # every Nth detection per track; 15 fps source, so still ~7/s
MIN_BOX = 28        # px; below this the crop is too small to describe


def descriptor(crop: np.ndarray) -> np.ndarray:
    """L1-normalised GRAYSCALE histogram per horizontal band, concatenated.

    No colour, deliberately, and this is not a trade -- measured against curated
    labels from three matches, dropping colour won on every axis at once:

        descriptor        in-match 6-way   cross-match   across an alliance flip
        3-band HSV, 384d       77%           92% / 87%          21%
        3-band gray16, 48d     81%           99% / 94%          84%

    (detection-weighted; cross-match is f1m3<->f1m2, the flip is f1m3->sf11m1 where
    three teams change bumper colour.)

    Colour was not merely redundant, it was actively harmful. FRC teams switch red and
    blue between matches, and a colour histogram spends most of its dynamic range on
    the alliance and on the venue's white balance -- both noise for identity. Measured,
    alliance was predictable from the old bumper-EXCLUDED colour descriptor at 72%, and
    when three robots swapped to red bumpers 53% of their crops matched a *different*
    robot that had worn red, against 17% matching their true identity. Subtracting the
    alliance component out (projection, per-alliance centering, tighter crops) reached
    53% detection-weighted transfer at best. Not looking at colour reaches 84%.

    Luminance survives because it describes the robot's physical layout -- dark intake,
    bright polycarbonate, the vertical arrangement of mechanisms -- which is stable
    across venues and across bumper changes.

    GRAY_BINS is coarse on purpose. Transfer falls monotonically as bins get finer
    (16 bins 73%, 32 bins 69%, 256 bins 67% on the flip test), because coarse
    luminance bins are insensitive to the exposure and white-balance differences
    between one venue's camera and another's.

    The BANDS split makes it spatial: a single global histogram cannot tell 'dark over
    light' from 'light over dark'. Bands rather than a 2D grid because robots rotate --
    binning left-to-right scrambles a robot's own structure as it turns, while
    top-to-bottom is invariant to rotation about the vertical axis. Bands matter less
    for luminance than they did for colour (2, 3 and 4 are within a point; 1 band is
    ~6 worse), so 3 is kept but is not delicate.

    The result must sum to 1, not to BANDS. split_on_appearance compares descriptors
    with the Hellinger distance sqrt(1 - sum(sqrt(a*b))), which is only a distance for
    normalised distributions: concatenating three histograms that each sum to 1 makes
    the Bhattacharyya coefficient ~3, so 1-3 clips to 0 and EVERY distance becomes
    zero. That silently removed all 20 appearance cuts and cost a point of coverage.
    Hence the division by BANDS below.

    Changing this invalidates the cached .npz -- rerun rtrack.appear -- and moves the
    change-point distance scale, so APPEAR_THRESH in robots.py must be re-tuned with it.
    """
    gray = cv2.cvtColor(cv2.resize(crop, (40, 40), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    out = []
    for b in range(BANDS):
        band = gray[b * 40 // BANDS:(b + 1) * 40 // BANDS]
        h = cv2.calcHist([band], [0], None, [GRAY_BINS], [0, 256]).flatten()
        s = h.sum()
        out.append((h / s) if s > 0 else h)
    return np.concatenate(out) / BANDS


def cnn_path(out: Path) -> Path:
    """Sibling of the histogram npz, so one decode can fill both."""
    return out.with_name(out.name.replace("_appearance", "_appearance_cnn"))


def run(stem: str, tracks_p: Path, out: Path, backend: str = "hist") -> None:
    rows = [json.loads(l) for l in tracks_p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])

    # Plan first, then ONE sequential pass. Random seeking this AV1 file costs 747 ms
    # a frame against 10.5 ms sequential -- see rtrack.identify.
    seen: dict[int, int] = {}
    plan: dict[int, list] = {}
    for r in rows:
        for d in r["dets"]:
            tid = d["tid"]
            if tid < 0:
                continue
            n = seen.get(tid, 0)
            seen[tid] = n + 1
            if n % STRIDE:
                continue
            if (d["xyxy"][2] - d["xyxy"][0]) < MIN_BOX:
                continue
            plan.setdefault(r["f"], []).append((tid, r["t"], d))

    wanted = sorted(plan)
    print(f"[appear] {len(seen)} tracks, "
          f"{sum(len(v) for v in plan.values())} crops over {len(wanted)} frames")

    want_hist = backend in ("hist", "both")
    want_cnn = backend in ("cnn", "both")
    tids, ts, feats, crops = [], [], [], []
    cap = cv2.VideoCapture(str(raw_path(stem)))
    idx, i = 0, 0
    while i < len(wanted):
        ok, img = cap.read()
        if not ok:
            break
        if idx == wanted[i]:
            H, W = img.shape[:2]
            for tid, t, d in plan[idx]:
                x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
                bh = y2 - y1
                cy1 = max(0, y1 + int(bh * BODY_TOP))
                cy2 = min(H, y1 + int(bh * BODY_BOTTOM))
                cx1, cx2 = max(0, x1), min(W, x2)
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
                    continue
                tids.append(tid)
                ts.append(t)
                if want_hist:
                    feats.append(descriptor(crop))
                if want_cnn:
                    # Kept at source resolution here; embed() does its own resize, and
                    # downsampling twice would throw away detail for nothing.
                    crops.append(crop.copy())
            i += 1
            if i % 400 == 0:
                print(f"    {i}/{len(wanted)} frames", flush=True)
        idx += 1
    cap.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    tid_a, t_a = np.array(tids, np.int32), np.array(ts, np.float32)
    if want_hist:
        np.savez_compressed(out, tid=tid_a, t=t_a,
                            feat=np.array(feats, np.float32))
        print(f"[appear] {len(tids)} histogram descriptors -> {out}")
    if want_cnn:
        from .embed import embed
        cp = cnn_path(out)
        np.savez_compressed(cp, tid=tid_a, t=t_a, feat=embed(crops))
        print(f"[appear] {len(tids)} cnn embeddings -> {cp}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cache per-detection appearance.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backend", choices=("hist", "cnn", "both"), default="hist",
                    help="which descriptor(s) to cache from the single decode. "
                         "'hist' is the tuned default that robots.split_on_appearance "
                         "needs; 'cnn' is the learned embedding reid can vote with; "
                         "'both' fills each from one pass. See rtrack.embed.")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    stem = video_id(args.video)
    run(stem, args.tracks, args.out or (C.STAGE3_DIR / f"{stem}_appearance.npz"),
        backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
