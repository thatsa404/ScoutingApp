"""Colour-threshold bumper finder -- a LABELLING AID, not a detector.

Stage 0 showed a locked-off camera, one venue, constant arena lighting and 80-130 px
bumpers in saturated alliance colours. That is the narrow case where classical CV is
good enough to put draft boxes on a frame so a human corrects rather than draws --
roughly halving the eval-set labelling time.

It is deliberately NOT a candidate detector: it cannot survive a venue change, a
lighting change, or a robot whose bumpers are largely occluded, and it has no notion
of a robot as opposed to a red-ish blob. Do not let it leak into Stage 1.

    uv run -m rtrack.prelabel --frames eval/frames --debug
    uv run -m rtrack.prelabel --tune eval/frames/f001348.png   # hue histogram
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C

# OpenCV hue is 0-179. Red wraps the origin, hence two bands.
RED_BANDS = ((0, 8), (170, 179))
BLUE_BAND = (100, 128)
MIN_SAT, MIN_VAL = 90, 55

# Bumpers are wide, short slabs. Generous because perspective varies across the field.
MIN_AREA = 220
MIN_W, MAX_W = 22, 280
MIN_H, MAX_H = 7, 95
MIN_ASPECT = 0.9      # w/h -- rejects tall thin slivers of field structure
MIN_FILL = 0.32       # blob area / box area -- rejects scattered speckle


@dataclass
class Roi:
    """Playing-surface region plus static exclusions, in source pixels.

    Valid only for one camera setup, which is exactly what a locked-off broadcast
    gives us. Stored per video so a second clip cannot silently inherit it.
    """
    y0: int
    y1: int
    x0: int
    x1: int
    exclude: list[list[int]]  # [x0,y0,x1,y1] boxes -- ramps, alliance walls, score bug

    def mask(self, shape: tuple[int, ...]) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        m[self.y0:self.y1, self.x0:self.x1] = 255
        for x0, y0, x1, y1 in self.exclude:
            m[y0:y1, x0:x1] = 0
        return m

    def save(self, p: Path) -> None:
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(p: Path) -> "Roi":
        return Roi(**json.loads(p.read_text(encoding="utf-8")))


# Measured against GSxbsE42o5o: carpet spans roughly y 545-905. The exclusions are
# the two alliance ramps and the driver-station walls, which are large saturated
# red/blue field elements sitting at fixed positions -- the main false-positive
# source, and free to remove precisely because the camera never moves.
DEFAULT_ROI = Roi(
    y0=540, y1=910, x0=55, x1=1875,
    exclude=[
        [55, 540, 335, 910],      # left alliance wall / driver station
        [1585, 540, 1875, 910],   # right alliance wall / driver station
        [395, 540, 630, 650],     # red Hub upper structure (roof + tag panels)
        [1235, 540, 1465, 650],   # blue Hub upper structure
        [280, 685, 580, 840],     # red ramp
        [1325, 725, 1600, 890],   # blue ramp
    ],
)
# The Hub exclusions deliberately stop at y=650: the structures' coloured faces sit
# above that line, while the carpet robots actually drive on is below it. Excluding
# the whole Hub footprint would blind the aid to any robot parked in front of one.


def color_masks(bgr: np.ndarray, roi_mask: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_val = (hsv[:, :, 1] >= MIN_SAT) & (hsv[:, :, 2] >= MIN_VAL)
    h = hsv[:, :, 0]

    red = np.zeros(h.shape, bool)
    for lo, hi in RED_BANDS:
        red |= (h >= lo) & (h <= hi)
    blue = (h >= BLUE_BAND[0]) & (h <= BLUE_BAND[1])

    red = (red & sat_val).astype(np.uint8) * 255
    blue = (blue & sat_val).astype(np.uint8) * 255

    if roi_mask is not None:
        red = cv2.bitwise_and(red, roi_mask)
        blue = cv2.bitwise_and(blue, roi_mask)

    # Close first to bridge the gaps team numbers punch through a bumper, then open
    # to drop speckle. Order matters: opening first would erase thin far-field bumpers.
    # The kernel is wide and flat because the gap to bridge is the width of a white
    # digit -- roughly 20 px on a near bumper -- and bridging vertically would instead
    # weld a bumper to whatever sits above it.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    for m in (red, blue):
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, dst=m, iterations=2)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k2)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k2)
    return red, blue


def boxes_from_mask(mask: np.ndarray, cls: int) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or not (MIN_W <= w <= MAX_W) or not (MIN_H <= h <= MAX_H):
            continue
        if w / max(h, 1) < MIN_ASPECT or area / (w * h) < MIN_FILL:
            continue
        out.append((x, y, x + w, y + h, cls))
    return out


def merge_split(boxes, max_gap: int = 18, min_v_overlap: float = 0.5):
    """Weld same-class fragments back into one bumper.

    Morphological closing handles most digit-gaps, but a bumper crossed by a wide
    number or a strut can still arrive in pieces. Two fragments merge when they are
    the same alliance, nearly level with each other, and horizontally adjacent.
    """
    out: list[list[int]] = []
    for b in sorted(boxes, key=lambda b: (b[4], b[0])):
        x1, y1, x2, y2, cls = b
        for m in out:
            if m[4] != cls:
                continue
            ov = min(y2, m[3]) - max(y1, m[1])
            if ov <= 0:
                continue
            if ov / min(y2 - y1, m[3] - m[1]) < min_v_overlap:
                continue
            if max(x1, m[0]) - min(x2, m[2]) > max_gap:
                continue
            m[0], m[1] = min(m[0], x1), min(m[1], y1)
            m[2], m[3] = max(m[2], x2), max(m[3], y2)
            break
        else:
            out.append([x1, y1, x2, y2, cls])
    return [tuple(m) for m in out]


def detect(bgr: np.ndarray, roi: Roi = DEFAULT_ROI
           ) -> list[tuple[int, int, int, int, int]]:
    red, blue = color_masks(bgr, roi.mask(bgr.shape))
    boxes = (boxes_from_mask(red, C.CLASS_RED)
             + boxes_from_mask(blue, C.CLASS_BLUE))
    return merge_split(boxes)


def to_yolo(boxes, w: int, h: int) -> str:
    """YOLO format: cls cx cy bw bh, all normalised. What Roboflow imports."""
    lines = []
    for x1, y1, x2, y2, cls in boxes:
        lines.append(f"{cls} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                     f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def draw(bgr: np.ndarray, boxes, roi: Roi | None = None) -> np.ndarray:
    vis = bgr.copy()
    if roi is not None:
        cv2.rectangle(vis, (roi.x0, roi.y0), (roi.x1, roi.y1), (90, 90, 90), 1)
        for x0, y0, x1, y1 in roi.exclude:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 90), 1)
    for x1, y1, x2, y2, cls in boxes:
        col = C.COLOR_RED_BGR if cls == C.CLASS_RED else C.COLOR_BLUE_BGR
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, C.CLASS_NAMES[cls].split("_")[0], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return vis


def tune(path: Path, roi: Roi = DEFAULT_ROI) -> None:
    """Hue histogram of saturated in-ROI pixels -- pick bands from data, not memory."""
    bgr = cv2.imread(str(path))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = roi.mask(bgr.shape) > 0
    sel = m & (hsv[:, :, 1] >= MIN_SAT) & (hsv[:, :, 2] >= MIN_VAL)
    hues = hsv[:, :, 0][sel]
    print(f"{path.name}: {sel.sum()} saturated in-ROI px of {m.sum()} in ROI")
    if hues.size == 0:
        print("  none -- MIN_SAT/MIN_VAL are too high for this footage")
        return
    hist, _ = np.histogram(hues, bins=36, range=(0, 180))
    for i, c in enumerate(hist):
        if c:
            bar = "#" * min(60, int(60 * c / max(hist.max(), 1)))
            print(f"  H {i*5:3d}-{i*5+4:3d} {c:7d} {bar}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draft bumper boxes for hand correction.")
    ap.add_argument("--frames", type=Path, help="directory of .png frames")
    ap.add_argument("--tune", type=Path, help="print a hue histogram for one frame")
    ap.add_argument("--debug", action="store_true", help="also write overlay JPEGs")
    ap.add_argument("--roi", type=Path, help="ROI json (default: built-in)")
    args = ap.parse_args(argv)

    roi = Roi.load(args.roi) if args.roi else DEFAULT_ROI

    if args.tune:
        tune(args.tune, roi)
        return 0
    if not args.frames:
        ap.error("pass --frames or --tune")

    dbg = args.frames.parent / "prelabel_debug"
    if args.debug:
        dbg.mkdir(parents=True, exist_ok=True)

    n_box = 0
    frames = sorted(args.frames.glob("*.png"))
    for p in frames:
        bgr = cv2.imread(str(p))
        boxes = detect(bgr, roi)
        n_box += len(boxes)
        h, w = bgr.shape[:2]
        p.with_suffix(".txt").write_text(to_yolo(boxes, w, h), encoding="utf-8")
        if args.debug:
            cv2.imwrite(str(dbg / f"{p.stem}.jpg"), draw(bgr, boxes, roi))
        r = sum(1 for b in boxes if b[4] == C.CLASS_RED)
        print(f"{p.name:<22} {len(boxes):>2} boxes (red {r}, blue {len(boxes)-r})")

    print(f"\n{n_box} draft boxes over {len(frames)} frames "
          f"({n_box/max(len(frames),1):.1f}/frame; 6 is the target)")
    if args.debug:
        print(f"overlays: {dbg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
