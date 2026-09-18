"""Assign red/blue to a detected robot box by bumper hue.

The Stage 1b dataset has a single `robots` class, so alliance cannot come free from
the detector as the plan assumed. It has to be recovered from pixels.

This is a much easier problem than detection: the box already tells us where to look,
FRC rules mandate a solid, saturated alliance colour, and the hue bands here were
measured off the target footage (see rtrack.prelabel). The one real subtlety is that
the dataset's boxes enclose the WHOLE robot, and a 2025-style robot is mostly tall
superstructure -- so sampling the full box dilutes the bumper. We sample the lower
band, where FRC rules require the bumper to be.

Unlike rtrack.prelabel, this IS intended for the real pipeline: it runs on a region
the detector already localised, rather than trying to find robots in open field.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from .prelabel import BLUE_BAND, MIN_SAT, MIN_VAL, RED_BANDS

# Bumpers sit 1-7.5 in off the floor and are 5-7.5 in tall, so on a whole-robot box
# they occupy roughly the lower third.
#
# TUNED, not guessed. Swept against a constraint that needs no hand labels: FRC
# guarantees 3 red and 3 blue, so any frame where one alliance exceeds 3 is wrong by
# construction. Over 443 frames spanning the match, (0.65, 0.90, 0.25) halved the
# over-3 rate versus the original (0.55, 0.97, 0.00) -- 8.6% -> 4.3% -- at the cost
# of more undecided calls (9.6% -> 15.0%). That trade is deliberate: an undecided
# detection is honest, a wrong alliance silently corrupts a team's data.
#
# Why each direction helps:
#   BAND_TOP 0.65    - start lower; the upper box is superstructure and background
#   BAND_BOTTOM 0.90 - drop the last rows, which catch carpet, shadow and the
#                      coloured ramp a robot is parked on
#   BAND_INSET 0.25  - trim the left/right quarters, where field elements behind the
#                      robot bleed into the box
BAND_TOP, BAND_BOTTOM = 0.65, 0.90
BAND_INSET = 0.25

# Below this share of decisive pixels the crop is mostly carpet or shadow and the
# answer is a coin flip -- say so rather than guessing.
MIN_DECISIVE_FRAC = 0.02
MIN_MARGIN = 0.15


@dataclass(frozen=True)
class AllianceCall:
    alliance: str | None   # "red" | "blue" | None when undecidable
    confidence: float      # 0-1 margin between the two colours
    red_px: int
    blue_px: int

    @property
    def cls(self) -> int | None:
        if self.alliance == "red":
            return C.CLASS_RED
        if self.alliance == "blue":
            return C.CLASS_BLUE
        return None


def _band_crop(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    h = max(y2 - y1, 1.0)
    w = max(x2 - x1, 1.0)
    ty = int(y1 + h * BAND_TOP)
    by = int(y1 + h * BAND_BOTTOM)
    lx = int(x1 + w * BAND_INSET)
    rx = int(x2 - w * BAND_INSET)
    ty, by = max(0, ty), min(bgr.shape[0], by)
    lx, rx = max(0, lx), min(bgr.shape[1], rx)
    if by <= ty or rx <= lx:
        return np.empty((0, 0, 3), np.uint8)
    return bgr[ty:by, lx:rx]


def classify(bgr: np.ndarray, box: tuple[int, int, int, int]) -> AllianceCall:
    crop = _band_crop(bgr, box)
    if crop.size == 0:
        return AllianceCall(None, 0.0, 0, 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strong = (s >= MIN_SAT) & (v >= MIN_VAL)

    red = np.zeros(h.shape, bool)
    for lo, hi in RED_BANDS:
        red |= (h >= lo) & (h <= hi)
    blue = (h >= BLUE_BAND[0]) & (h <= BLUE_BAND[1])

    r = int((red & strong).sum())
    b = int((blue & strong).sum())
    total = crop.shape[0] * crop.shape[1]

    if total == 0 or (r + b) / total < MIN_DECISIVE_FRAC:
        return AllianceCall(None, 0.0, r, b)

    margin = abs(r - b) / (r + b)
    if margin < MIN_MARGIN:
        return AllianceCall(None, margin, r, b)
    return AllianceCall("red" if r > b else "blue", margin, r, b)


def classify_all(bgr: np.ndarray, boxes) -> list[AllianceCall]:
    return [classify(bgr, b[:4]) for b in boxes]


def enforce_three_three(calls: list[AllianceCall]) -> list[AllianceCall]:
    """Nudge an over-full alliance toward the 3-red/3-blue a match must have.

    Only ever REASSIGNS the least-confident calls, and only when one side has more
    than three. It cannot invent a robot that was never detected, and it is not a
    substitute for a correct call -- it is a cheap consistency prior for the common
    case where one bumper is half-occluded and reads as the wrong colour.

    Deliberately does nothing when fewer than 6 robots are visible, which Stage 0
    showed is frequent: guessing under occlusion is exactly how silent errors get in.
    """
    decided = [i for i, c in enumerate(calls) if c.alliance]
    if len(decided) != 6:
        return calls

    out = list(calls)
    for colour, other in (("red", "blue"), ("blue", "red")):
        idx = [i for i in decided if out[i].alliance == colour]
        if len(idx) <= 3:
            continue
        idx.sort(key=lambda i: out[i].confidence)
        for i in idx[:len(idx) - 3]:
            c = out[i]
            out[i] = AllianceCall(other, c.confidence, c.red_px, c.blue_px)
    return out


def draw(bgr: np.ndarray, boxes, calls: list[AllianceCall]) -> np.ndarray:
    vis = bgr.copy()
    for (x1, y1, x2, y2, *_), call in zip(boxes, calls):
        col = {"red": C.COLOR_RED_BGR, "blue": C.COLOR_BLUE_BGR}.get(
            call.alliance, (160, 160, 160))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        # Show the band we actually sampled -- makes a wrong call self-explanatory.
        h = max(float(y2) - float(y1), 1.0)
        w = max(float(x2) - float(x1), 1.0)
        cv2.rectangle(vis,
                      (int(float(x1) + w * BAND_INSET), int(float(y1) + h * BAND_TOP)),
                      (int(float(x2) - w * BAND_INSET), int(float(y1) + h * BAND_BOTTOM)),
                      col, 1)
        label = f"{call.alliance or '?'} {call.confidence:.2f}"
        cv2.putText(vis, label, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return vis
