"""Field structures that HIDE robots, drawn once per camera in public/rtrack/occluders.html.

    uv run -m rtrack.occluders --camera 2026necmp1_qm1 --check

WHAT THE FILE CONTAINS, and why it is shaped this way.

A polygon per structure, in NORMALISED IMAGE coordinates. Image space because an
occluder hides part of the picture: its silhouette in the frame IS the hidden region,
whereas in field metres the same structure becomes a wedge whose extent depends on
robot height and camera angle. Testing a field-space wedge would also need the
homography inverted, which would undo the lens undistortion rtrack.project applies
going the other way.

FULL REGIONS, NOT ENTRIES AND EXITS. The edges of the polygon ARE the entries and
exits, and deriving them beats asking for them: a person marking entry points has to
guess which side robots use, and would have to guess again for every camera. It also
covers the case that matters most -- a robot that drives behind a tower and emerges on
the OTHER side. An earlier attempt bound reappearances only to the same pixels a track
vanished at; measured on 2026necmp1_qm24 that caught 13 of 17 blocked handoffs, but it
could not represent transit at all, and it fused different robots that reused a spot
(vote agreement collapsed from 3 tracks at >=80% to none). Knowing the structure's
outline is what fixes that: any edge can be the exit.

WHAT IT IS NOT FOR. The camera's own field of view is handled by
project.visible_region -- 11% of track endpoints on this event fall outside it, and
counting those here would rebind robots that simply left the picture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C

# A track "ends at" a structure when its last box lands within this of the outline.
# One robot width: the box centre sits half a robot from the silhouette edge at the
# moment the robot slips behind it, and the detector usually drops it a frame or two
# before it is fully hidden.
EDGE_TOL_PX = 130.0

# Longest a robot may be hidden while CROSSING behind a structure and still be rebound.
MAX_HIDDEN_S = 20.0

# ...and the far longer allowance for a robot that never went anywhere.
#
# Transit is not the common case. Measured on 2026necmp1_qm24, 33 of the 72 handoffs
# the stitcher refuses are hub -> the SAME hub, and their displacements are 13-38 px
# across gaps of 78-137 s: track #6 dies at (359,652) and is reborn six times within
# 40 px of that spot. That is a robot parked at the hub scoring, dropping in and out of
# detection -- not one driving behind it. A transit budget sized from the structure's
# diagonal gives ~2.8 s here and never fires.
#
# A bare "reappeared at the same pixels" rule was tried first and measured badly: qm21
# +11 accuracy points, qm16 -2, qm20 -5, and vote agreement on qm24 collapsing to zero
# tracks at >=80%, because two different robots reusing one spot get fused. Requiring
# BOTH ends to sit on a structure a human marked is what makes it safe: it is no longer
# any spot on the field, it is this tower.
PARKED_MAX_S = 180.0
PARKED_MOVE_PX = 60.0   # "never went anywhere", well inside one robot width


def path_for(camera: str) -> Path:
    return C.CALIB_DIR / f"{camera}_occluders.json"


def load(camera: str) -> dict | None:
    """The drawn regions for this camera, or None. Absent is normal, not an error:
    a camera nobody has drawn yet simply gets no occlusion handling."""
    p = path_for(camera)
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("kind") != "occluders" or not isinstance(doc.get("regions"), list):
        raise SystemExit(f"[occluders] {p} is not an occluders file")
    return doc


def to_pixels(doc: dict, frame_wh: tuple[int, int] = (1920, 1080)) -> list[dict]:
    """Normalised polygons -> pixel polygons for THIS frame size.

    The drawing is stored normalised so a camera that changes resolution keeps its
    shapes; frameSize records what it was drawn on, and a mismatch is reported rather
    than silently rescaled into the wrong place.
    """
    W, H = frame_wh
    drawn = doc.get("frameSize") or [W, H]
    if drawn[0] and abs(drawn[0] / max(drawn[1], 1) - W / max(H, 1)) > 0.02:
        print(f"[occluders] WARNING drawn on {drawn[0]}x{drawn[1]}, applying to {W}x{H} "
              f"-- different aspect ratio, the shapes will not line up")
    out = []
    for r in doc["regions"]:
        pts = np.array([[p[0] * W, p[1] * H] for p in r["points"]], np.float64)
        # AREA (closed) or EDGE (open). An area is a structure a robot goes BEHIND, so
        # "inside" is meaningful. An edge is a line a robot goes ACROSS -- the lip of
        # the picture, a corner exit -- where there is no inside to be in, and the whole
        # outside of the frame cannot be enclosed by a polygon anyway.
        #
        # An edge is also deliberately a SEGMENT rather than the whole frame boundary:
        # a robot leaving near-left and returning far-right is not one event, and
        # treating the boundary as a single region would let a held team reappear
        # anywhere along it.
        kind = (r.get("type") or ("edge" if len(pts) < 3 else "area")).lower()
        if len(pts) < (2 if kind == "edge" else 3):
            continue
        out.append({"name": r.get("name") or f"region {len(out) + 1}",
                    "poly": pts, "closed": kind != "edge"})
    return out


def _dist_to_poly(pt: np.ndarray, poly: np.ndarray, closed: bool = True) -> float:
    """Distance from a point to a closed outline (0 inside) or to an open polyline."""
    if closed and _inside(pt, poly):
        return 0.0
    best = float("inf")
    n = len(poly)
    for i in range(n if closed else n - 1):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((pt - a) @ ab) / L2))
        best = min(best, float(np.hypot(*(pt - (a + t * ab)))))
    return best


def _inside(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    hit = False
    n = len(poly)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            hit = not hit
    return hit


def region_at(x: float, y: float, regions: list[dict],
              tol_px: float = EDGE_TOL_PX) -> str | None:
    """Name of the structure this point is at (inside, or within tol of the outline)."""
    pt = np.array([x, y], np.float64)
    best, bestd = None, float("inf")
    for r in regions:
        d = _dist_to_poly(pt, r["poly"], r.get("closed", True))
        if d <= tol_px and d < bestd:
            best, bestd = r["name"], d
    return best


def parked_at(a_end, b_start, name: str, regions: list[dict],
              gap_s: float) -> bool:
    """Same structure, barely moved, gap too long for transit -- a robot that sat there."""
    if gap_s > PARKED_MAX_S:
        return False
    d = float(np.hypot(b_start[0] - a_end[0], b_start[1] - a_end[1]))
    return d <= PARKED_MOVE_PX


def transit_budget_s(regions: list[dict], name: str,
                     speed_px_s: float) -> float:
    """How long a robot could plausibly stay hidden behind THIS structure.

    Sized by the structure rather than by a global constant: crossing behind a wide
    tower takes longer than slipping past a narrow post, and a robot may also pause.
    Capped by MAX_HIDDEN_S because beyond that the evidence that it is the SAME robot
    has decayed to almost nothing.
    """
    for r in regions:
        if r["name"] == name:
            if not r.get("closed", True):
                # Crossing a line is instantaneous; what takes time is whatever the
                # robot does out of shot, which the line cannot bound. Fall back to
                # the global cap rather than inventing a transit from its length.
                return MAX_HIDDEN_S
            w = float(r["poly"][:, 0].max() - r["poly"][:, 0].min())
            h = float(r["poly"][:, 1].max() - r["poly"][:, 1].min())
            span = float(np.hypot(w, h))
            return min(MAX_HIDDEN_S, span / max(speed_px_s, 1e-6) + 2.0)
    return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a camera's occluder file.")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--frame-size", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--overlay", default=None, metavar="IMAGE",
                    help="draw the regions on this frame and write --out. Worth doing "
                         "once: a file that loads cleanly can still be drawn on the "
                         "wrong frame, and only the picture shows that.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    doc = load(args.camera)
    if not doc:
        raise SystemExit(f"[occluders] no {path_for(args.camera).name} -- draw one in "
                         f"public/rtrack/occluders.html")
    regs = to_pixels(doc, tuple(args.frame_size))
    print(f"[occluders] {args.camera}: {len(regs)} region(s), drawn on "
          f"{doc.get('frameSize')}, applied at {args.frame_size}")
    for r in regs:
        xs, ys = r["poly"][:, 0], r["poly"][:, 1]
        print(f"    {r['name']:<22} {'area' if r.get('closed', True) else 'edge':>4} "
              f"{len(r['poly']):>2} pts   "
              f"x {xs.min():6.0f}-{xs.max():<6.0f} y {ys.min():6.0f}-{ys.max():<6.0f}")

    if args.overlay:
        import cv2
        img = cv2.imread(args.overlay)
        if img is None:
            raise SystemExit(f"[occluders] could not read {args.overlay}")
        h, w = img.shape[:2]
        regs = to_pixels(doc, (w, h))
        for i, r in enumerate(regs):
            col = [(60, 200, 255), (255, 170, 60), (170, 255, 120),
                   (255, 120, 200)][i % 4]
            pts = r["poly"].astype(np.int32).reshape(-1, 1, 2)
            closed = r.get("closed", True)
            if closed:
                ov = img.copy()
                cv2.fillPoly(ov, [pts], col)
                img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
            cv2.polylines(img, [pts], closed, col, 3 if closed else 5, cv2.LINE_AA)
            x, y = r["poly"][:, 0].min(), r["poly"][:, 1].min()
            cv2.putText(img, r["name"], (int(x), int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        out = args.out or "occluders_check.png"
        cv2.imwrite(out, img)
        print(f"[occluders] overlay -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
