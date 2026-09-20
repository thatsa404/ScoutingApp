"""Stage 2 -- propose OCCLUDER regions from accumulated tracking data.

    uv run -m rtrack.occlude --event 2026necmp1 --matches 3 \
        --overlay 2026necmp1_qm1 --out out/stage2/2026necmp1_occluders.png

WHY PROPOSE RATHER THAN ASK. Robots that vanish behind a field element and reappear
are the largest remaining source of identity churn: the track dies, a new one is born,
and the solver has to re-establish who it is from a cold start. Knowing WHERE the
occluders are would let a reappearance be bound back to the track that entered, which
is what pixel-proximity alone cannot do once the robot transits and emerges elsewhere.

The regions are static per camera, so a human could draw them once. This proposes them
instead, because drawing is fiddly, has to be redone per venue, and -- as it turns out
-- the data already contains the answer.

WHAT THE SIGNAL IS. Not emptiness. A region nobody drives through and a region that
hides robots both look empty in a density map. The discriminator is the ENDPOINT RATE:
track births and deaths per detection. An occluder is surrounded by tracks that stop
and start; a quiet corner is not. Measured over 23 matches of 2026necmp1, the top 3%
of endpoint-rate cells form bands at x = 5.0 and x = 11.0-11.5 m.

THE MIRROR TEST IS WHAT MAKES THIS TRUSTWORTHY ON THIN DATA. An FRC field is symmetric
about its centre line, so a real occluder has a twin at the mirrored x. Noise does not.
Measured, building from the first N matches of the event:

    matches   endpoints   bands found (x, m)          mirrored pair?
       1          222     2.6, 9.7                    no
       2          414     2.2, 14.1, 5.2, 11.2        YES (5.2 <-> 11.2)
       3          664     11.1, 5.2, 6.3              YES
       8         1724     11.2, 5.2, 5.1              YES
      23         6929     5.0, 11.0-11.5              YES

One match is not enough and says so; two are. 5.2 mirrored about the centre line is
11.34 against an observed 11.2 -- 14 cm from 414 endpoints. Bands WITHOUT a twin are
reported as low confidence rather than presented as fact.

PRECISION IS COARSE, DELIBERATELY. Cell-level agreement with the 23-match map only
reaches IoU 0.46 even at 12 matches, so this proposes bands a few cells wide, not
polygons. It is a starting point for a human to confirm or adjust, and it is honest
about that.

NO INVERSE HOMOGRAPHY. To draw a field-space region on the camera view, a dense grid
of IMAGE pixels is projected forward into field metres and tested against the region.
rtrack.project only ever maps video pixels -> field, and inverting a homography that
carries lens distortion would reintroduce exactly the error the undistortion removes.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .project import load_calib, load_field_ref, project_points, visible_region

CELL = 0.25          # metres per grid cell
TOP_PCT = 97.0       # endpoint-rate percentile that counts as "hot"
SMOOTH_DETS = 20.0   # added to the denominator so thin cells cannot spike the rate
MIN_CELLS = 3        # cells before a blob is a candidate at all
MIRROR_TOL_M = 0.40  # how close a band's twin must sit to the mirrored position
MIN_MATCHES = 2      # below this the proposal is not offered; see the table above


def positions_files(event: str, limit: int | None = None) -> list[Path]:
    """Chronological, because the question is what an event knows EARLY."""
    def num(p: Path) -> int:
        m = re.search(r"qm(\d+)_positions", p.name)
        return int(m.group(1)) if m else 10_000
    fs = sorted(C.STAGE2_DIR.glob(f"{event}_qm*_positions.json"), key=num)
    return fs[:limit] if limit else fs


def field_maps(paths: list[Path], size_m: tuple[float, float]):
    """(density, endpoints) per field cell, over every match given."""
    FW, FH = size_m
    nx, ny = int(FW / CELL) + 1, int(FH / CELL) + 1
    dens = np.zeros((ny, nx), np.float64)
    ends = np.zeros((ny, nx), np.float64)
    for p in paths:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        per: dict[int, list] = collections.defaultdict(list)
        for s in doc.get("samples", ()):
            if s["tid"] < 0 or "offfield" in s.get("flags", ()):
                continue
            x, y = s["x"], s["y"]
            if not (0 <= x < FW and 0 <= y < FH):
                continue
            dens[int(y / CELL), int(x / CELL)] += 1
            per[s["tid"]].append((s["t"], x, y))
        for _tid, v in per.items():
            if len(v) < 5:
                continue                      # too short to have a meaningful endpoint
            v.sort()
            for _t, x, y in (v[0], v[-1]):
                ends[int(y / CELL), int(x / CELL)] += 1
    return dens, ends


def visible_mask(H, ref, lens, shape, size_m) -> np.ndarray:
    """Cells inside the camera's view, eroded by ~a robot width.

    The frame edge sheds tracks for a reason that has nothing to do with occlusion --
    the robot simply left the picture -- so the border must not be allowed to vote.
    Measured on 2026necmp1: 11% of all track endpoints fall outside this region.
    """
    ny, nx = shape
    poly, _frac, _side = visible_region(H, ref, lens)
    vis = np.zeros((ny, nx), np.uint8)
    cv2.fillPoly(vis, [np.round(np.array(poly, np.float32) / CELL).astype(np.int32)], 1)
    return cv2.erode(vis, np.ones((5, 5), np.uint8)).astype(bool)


def propose(dens, ends, mask, size_m, top_pct: float = TOP_PCT,
            grow: int = 1) -> list[dict]:
    """Candidate occluder bands, each tagged with whether it has a mirror twin."""
    FW, _FH = size_m
    rate = np.where(mask, ends / (dens + SMOOTH_DETS), 0.0)
    vals = rate[mask]
    if not (vals > 0).any():
        return []
    hot = (rate >= np.percentile(vals, top_pct)) & mask
    if grow:
        # Endpoints scatter around an occluder's edge rather than stacking on one cell,
        # so close small gaps before labelling. Without this a single structure reports
        # as three or four separate slivers of 3-7 cells.
        k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
        hot = (cv2.morphologyEx(hot.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0) & mask
    n_lab, lab = cv2.connectedComponents(hot.astype(np.uint8))
    out = []
    for i in range(1, n_lab):
        ys, xs = np.where(lab == i)
        if len(ys) < MIN_CELLS:
            continue
        out.append({
            # The CELLS, not just their bounding box. A connected component that snakes
            # around a structure has a box covering half the field -- measured, one
            # candidate reported x 4.75-12.75 / y 0.00-8.25 from 446 scattered cells --
            # and shading that box would claim almost the whole field is occluded.
            "cellsXY": [[int(a), int(b)] for a, b in zip(xs.tolist(), ys.tolist())],
            "cells": int(len(ys)),
            "xM": [round(float(xs.min()) * CELL, 2), round(float(xs.max() + 1) * CELL, 2)],
            "yM": [round(float(ys.min()) * CELL, 2), round(float(ys.max() + 1) * CELL, 2)],
            "xCentreM": round(float(xs.mean()) * CELL, 2),
            "endpoints": int(ends[ys, xs].sum()),
            "detections": int(dens[ys, xs].sum()),
        })
    # Mirror test. A real field element has a twin at FW - x; noise does not.
    for a in out:
        want = FW - a["xCentreM"]
        twin = min((b for b in out if b is not a),
                   key=lambda b: abs(b["xCentreM"] - want), default=None)
        d = abs(twin["xCentreM"] - want) if twin else None
        a["mirrored"] = bool(twin and d is not None and d <= MIRROR_TOL_M)
        a["mirrorErrM"] = round(float(d), 2) if d is not None else None
    out.sort(key=lambda r: (-r["mirrored"], -r["endpoints"]))
    return out


def image_overlay(stem: str, regions: list[dict], H, ref, lens,
                  frame: np.ndarray, only_mirrored: bool = True) -> np.ndarray:
    """Shade the proposed regions on a real video frame.

    Forward projection only: every pixel of a coarse grid is mapped to field metres and
    tested, so the homography is never inverted.
    """
    h, w = frame.shape[:2]
    step = 4
    gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    XY = project_points(pts, H, ref, lens)
    keep = [r for r in regions if r["mirrored"] or not only_mirrored]
    cells = {(a, b) for r in keep for a, b in r.get("cellsXY", ())}
    ix = np.floor(XY[:, 0] / CELL).astype(np.int64)
    iy = np.floor(XY[:, 1] / CELL).astype(np.int64)
    hit = np.fromiter(((int(a), int(b)) in cells for a, b in zip(ix, iy)),
                      bool, count=len(ix))
    small = hit.reshape(gy.shape).astype(np.uint8) * 255
    mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    tint = np.zeros_like(frame)
    tint[:, :] = (40, 200, 255)
    out = np.where(mask[..., None] > 0, (0.55 * out + 0.45 * tint).astype(np.uint8), out)
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    out[edge > 0] = (40, 200, 255)
    cv2.putText(out, f"{len(keep)} proposed occluder region(s) -- {stem}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose occluder regions from tracking data.")
    ap.add_argument("--event", required=True)
    ap.add_argument("--matches", type=int, default=None,
                    help="use only the first N matches, to test what an event knows "
                         "EARLY. Below %d the proposal is withheld." % MIN_MATCHES)
    ap.add_argument("--calib-from", default=None,
                    help="calibration stem; defaults to the first match of the event")
    ap.add_argument("--overlay", default=None,
                    help="video stem to draw the proposal on")
    ap.add_argument("--frame", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT,
                    help="endpoint-rate percentile counted as hot. Lower proposes "
                         "larger, coarser regions; 97 on 3 matches yields slivers.")
    ap.add_argument("--grow", type=int, default=1,
                    help="morphological closing radius in cells, to knit the scatter "
                         "around one structure into a single region")
    ap.add_argument("--all-regions", action="store_true",
                    help="draw unmirrored candidates too (low confidence)")
    args = ap.parse_args(argv)
    C.ensure_dirs()

    paths = positions_files(args.event, args.matches)
    if len(paths) < MIN_MATCHES:
        raise SystemExit(f"[occlude] {len(paths)} match(es) with positions -- need "
                         f"{MIN_MATCHES}. One match produces bands with no mirror twin "
                         f"and should not be offered as a proposal.")
    ref = load_field_ref()
    size_m = tuple(ref["fieldSizeM"])
    calib = args.calib_from or paths[0].name.split("_positions")[0]
    H, lens = load_calib(calib)

    dens, ends = field_maps(paths, size_m)
    mask = visible_mask(H, ref, lens, dens.shape, size_m)
    regions = propose(dens, ends, mask, size_m, args.top_pct, args.grow)
    mir = [r for r in regions if r["mirrored"]]
    print(f"[occlude] {len(paths)} match(es), {ends.sum():.0f} track endpoints, "
          f"{len(regions)} candidate(s), {len(mir)} with a mirror twin")
    for r in regions:
        tag = f"mirror +-{r['mirrorErrM']}m" if r["mirrored"] else "NO TWIN (low conf)"
        print(f"    x {r['xM'][0]:5.2f}-{r['xM'][1]:<5.2f} y {r['yM'][0]:5.2f}-"
              f"{r['yM'][1]:<5.2f}  {r['cells']:>3} cells  "
              f"{r['endpoints']:>4} endpoints / {r['detections']:>5} dets   {tag}")

    doc = {"schemaVersion": 1, "event": args.event, "matches": len(paths),
           "cellM": CELL, "regions": regions}
    jp = C.STAGE2_DIR / f"{args.event}_occluders.json"
    jp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[occlude] -> {jp}")

    if args.overlay:
        from .acquire import raw_path
        cap = cv2.VideoCapture(str(raw_path(args.overlay)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"[occlude] could not read frame {args.frame}")
        img = image_overlay(args.overlay, regions, H, ref, lens, frame,
                            only_mirrored=not args.all_regions)
        out = args.out or (C.STAGE2_DIR / f"{args.event}_occluders.png")
        cv2.imwrite(str(out), img)
        print(f"[occlude] overlay -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
