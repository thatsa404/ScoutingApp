"""Radial lens distortion, solved from the calibration correspondences themselves.

WHY THIS EXISTS. A homography can only model a pinhole camera looking at a plane.
This broadcast camera has pronounced barrel distortion -- the straight near barrier
images as an obvious curve -- and no homography can represent that. The consequence
was measured rather than assumed: with 8 clicked points the fit gave 0.536 m mean
error, and DOUBLING to 16 points changed it to 0.517 m. That plateau is the tell.
Click noise shrinks with more points; model error does not.

THE PRINCIPAL POINT IS A FREE PARAMETER, AND THIS MATTERS MORE THAN THE DISTORTION.
An earlier version pinned it to the image centre, reasoning that planar
correspondences cannot identify it. That was wrong once the points spread wide, and
the cost was large -- the model absorbed the offset as excess curvature:

                            in-sample    leave-one-out   near-barrier straightness
    centre fixed (960,540)     0.192 m          0.250 m                    6.35 px
    principal point free       0.098 m          0.134 m                    1.88 px

The solved centre is (875, 477), 85 and 63 px off the image centre. Broadcast footage
is cropped and rescaled from the sensor, so there was never a reason for the optical
centre to land at the image centre.

HOW IT IS SOLVED. Not with a checkerboard, and not with extra clicking. The existing
correspondences already constrain it: undistort with candidate parameters, fit the
homography, measure the residual, optimise. f, k1 and the principal point trade off
against each other, so none is physically meaningful alone -- only the undistortion
map they produce together.

VALIDATED TWO WAYS, because adding parameters to ~20 points is exactly the setup
where a residual drop means nothing:

  1. Leave-one-out. Fit on n-1, predict the held-out point.
  2. An INDEPENDENT plumb line. The near barrier is a straight world line; tracing it
     and measuring how straight it becomes after undistortion uses no correspondence
     data at all. It reaches 1.88 px RMS over a 1650 px span.

A caution learned the hard way: two other traced "straight" structures appeared to
contradict the model, until a quadratic fit showed they were noise (residual 6.01 ->
5.84, i.e. no smooth curve at all). Only trust a plumb line whose curvature a
quadratic can actually absorb -- the near barrier goes 17.15 -> 0.61.

FUTURE IMPROVEMENT, NOT BUILT: calibrate from traced EDGES rather than points. Four
field edges give hundreds of constraints instead of ~40, straightness pins the lens
model with no metric knowledge, and the rectangle's known 16.541 x 8.069 m pins the
homography -- separating two things that currently both come from one set of points.
Tracing is also more accurate per unit of effort (the barrier trace is sub-pixel) and
degrades better, since a stray sample is one of hundreds instead of a silent outlier.
The catch: a rectangle alone admits a family of (focal, distortion, pose) solutions,
so it needs interior tape lines or a few retained point correspondences to break the
degeneracy. Build it as lines for the lens model, points for the metric anchor.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

W, H = 1920, 1080


def camera_matrix(f: float, cx: float | None = None, cy: float | None = None,
                  w: int = W, h: int = H) -> np.ndarray:
    return np.array([[f, 0.0, w / 2.0 if cx is None else cx],
                     [0.0, f, h / 2.0 if cy is None else cy],
                     [0.0, 0.0, 1.0]])


def _unpack(lens: dict, w: int = W, h: int = H):
    return (float(lens["f"]), float(lens.get("k1", 0.0)), float(lens.get("k2", 0.0)),
            float(lens.get("cx", w / 2.0)), float(lens.get("cy", h / 2.0)))


def fold_radius(k1: float, k2: float = 0.0) -> float:
    """Largest normalised radius at which the radial model is still monotonic.

    r_d = r(1 + k1 r^2 + k2 r^4) only describes a lens while d(r_d)/dr > 0. Past the
    first root of 1 + 3 k1 r^2 + 5 k2 r^4 the map FOLDS: greater radii come back
    inward, and a point beyond it gets a position that looks plausible and means
    nothing. The overlay once drew both near field corners inside the frame, complete
    with a hook, for corners that are not visible in the video at all.
    """
    a, b, c = 5.0 * k2, 3.0 * k1, 1.0
    if abs(a) < 1e-12:
        return float("inf") if b >= 0 else float(np.sqrt(-c / b))
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    pos = [r for r in roots if r > 0]
    return float(np.sqrt(min(pos))) if pos else float("inf")


def undistort_points(pts, lens: dict, w: int = W, h: int = H) -> np.ndarray:
    """Distorted image points -> ideal pinhole image. Identity when k1=k2=0."""
    f, k1, k2, cx, cy = _unpack(lens, w, h)
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not k1 and not k2:
        return p.reshape(-1, 2)
    K = camera_matrix(f, cx, cy, w, h)
    d = np.array([k1, k2, 0.0, 0.0, 0.0], np.float64)
    return cv2.undistortPoints(p, K, d, P=K).reshape(-1, 2)


def distort_points(pts, lens: dict, w: int = W, h: int = H,
                   mark_invalid: bool = True) -> np.ndarray:
    """Ideal pinhole pixels -> real image pixels. Inverse of undistort.

    This is what lets a diagnostic be drawn on the ORIGINAL frame instead of a
    rectified one: distorting the LINES leaves the video untouched and bends the
    overlay to match the lens. Straight world lines become curves here, so sample
    them densely -- two endpoints joined by a straight segment would hide exactly the
    curvature this is meant to show.
    """
    f, k1, k2, cx, cy = _unpack(lens, w, h)
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if not k1 and not k2:
        return p
    xn = (p[:, 0] - cx) / f
    yn = (p[:, 1] - cy) / f
    r2 = xn * xn + yn * yn
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    out = np.stack([f * xn * s + cx, f * yn * s + cy], axis=1)
    if mark_invalid:
        # NaN rather than a plausible-looking number, so a caller that forgets to
        # check draws nothing instead of drawing a lie.
        out[r2 > fold_radius(k1, k2) ** 2] = np.nan
    return out


def undistort_image(img: np.ndarray, lens: dict, grow: float = 1.4):
    """Undistort a frame at a wider canvas so nothing falls off. (image, its K).

    Only for viewing a rectified frame. The pipeline distorts overlays onto the raw
    frame instead, which resamples nothing and crops nothing.
    """
    h, w = img.shape[:2]
    f, k1, k2, cx, cy = _unpack(lens, w, h)
    K = camera_matrix(f, cx, cy, w, h)
    if not k1 and not k2:
        return img, K
    d = np.array([k1, k2, 0.0, 0.0, 0.0], np.float64)
    size = (int(w * grow), int(h * grow))
    newK, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1.0, size)
    mx, my = cv2.initUndistortRectifyMap(K, d, None, newK, size, cv2.CV_32FC1)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR), newK


def straightness(pts, lens: dict) -> float:
    """RMS deviation from a straight line, after undistortion, in pixels.

    A world line that is straight must image straight once the lens is removed. This
    is the ONE observable that constrains the lens independently of the homography,
    which is why it matters so much here: every point correspondence lies on a single
    plane, and any amount of radial bend can be absorbed by a compensating focal
    length and homography. Measured, that degeneracy is total -- sweeping f from 380
    to 1200 with k1 re-optimised at each step gave a point residual of 0.157 m at
    EVERY step, while barrier straightness moved between 1.9 and 6.3 px.

    So the points cannot pick a lens, and a straight line can.
    """
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    u = undistort_points(p, lens)
    # Fit along whichever axis the line actually runs, so a near-vertical line does
    # not blow up a y-on-x fit.
    span = u.max(axis=0) - u.min(axis=0)
    if span[0] >= span[1]:
        a = np.polyfit(u[:, 0], u[:, 1], 1)
        r = u[:, 1] - np.polyval(a, u[:, 0])
    else:
        a = np.polyfit(u[:, 1], u[:, 0], 1)
        r = u[:, 0] - np.polyval(a, u[:, 1])
    return float(np.sqrt(np.mean(r * r)))


def _fit(src, dst, ppm, lens: dict, train=None):
    und = undistort_points(src, lens)
    idx = np.arange(len(src)) if train is None else train
    Hm, _ = cv2.findHomography(und[idx].astype(np.float32),
                               np.asarray(dst, np.float32)[idx], 0)
    if Hm is None:
        return None, None
    proj = cv2.perspectiveTransform(und.reshape(-1, 1, 2).astype(np.float32),
                                    Hm).reshape(-1, 2)
    return np.linalg.norm(proj - dst, axis=1) / ppm, Hm


def solve(src_px, dst_px, ppm: float, free_centre: bool = True,
          plumb: list | None = None, plumb_weight: float = 0.002) -> dict:
    """Search (f, k1, k2[, cx, cy]) for the undistortion that best linearises things.

    `plumb` is a list of point arrays, each traced along a world-straight line. It
    enters the objective as a TIE-BREAKER, not a competitor: the default weight makes
    a 5 px straightness change worth 0.01 m of point error, which is negligible
    against a ~0.16 m residual but decisive along the flat direction where the point
    residual does not move at all. Without it the optimiser picks a point on that
    valley arbitrarily, and the one it picks decides how badly the model extrapolates
    outside the calibrated radius.

    Returns parameters, the resulting error, and the leave-one-out error with and
    without distortion, so a caller can refuse the model when it does not generalise.
    """
    from scipy.optimize import minimize

    src = np.asarray(src_px, np.float64)
    dst = np.asarray(dst_px, np.float64)
    plain = {"f": 1000.0, "k1": 0.0, "k2": 0.0}
    base, _ = _fit(src, dst, ppm, plain)

    def pack(p):
        d = {"f": p[0], "k1": p[1], "k2": p[2]}
        if free_centre:
            d["cx"], d["cy"] = p[3], p[4]
        return d

    def cost(p, train=None):
        if not (200.0 < p[0] < 6000.0):
            return 1e6
        ln = pack(p)
        e, _ = _fit(src, dst, ppm, ln, train)
        if e is None:
            return 1e6
        c = float(e.mean())
        for line in (plumb or ()):
            c += plumb_weight * straightness(line, ln)
        return c

    best = None
    for f0 in (500.0, 800.0, 1200.0, 1600.0, 2200.0):
        for k0 in (-0.02, -0.05, -0.10, -0.20, -0.40):
            x0 = [f0, k0, 0.0] + ([W / 2.0, H / 2.0] if free_centre else [])
            r = minimize(cost, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-3, "fatol": 1e-7})
            if best is None or r.fun < best.fun:
                best = r

    lens = pack(best.x)
    err, Hm = _fit(src, dst, ppm, lens)

    def loo(ln):
        out = []
        for i in range(len(src)):
            tr = np.array([j for j in range(len(src)) if j != i])
            e, _ = _fit(src, dst, ppm, ln, tr)
            if e is not None:
                out.append(float(e[i]))
        return np.array(out) if out else np.array([np.nan])

    l_plain, l_dist = loo(plain), loo(lens)
    return {
        **{k: float(v) for k, v in lens.items()},
        "errM": {"mean": float(err.mean()), "max": float(err.max())},
        "errPlainM": {"mean": float(base.mean()), "max": float(base.max())},
        "looM": {"mean": float(l_dist.mean()), "max": float(l_dist.max())},
        "looPlainM": {"mean": float(l_plain.mean()), "max": float(l_plain.max())},
        # The only honest gate: does it help on points the fit never saw?
        "generalises": bool(l_dist.mean() < l_plain.mean() * 0.85),
        "plumbPx": [round(straightness(l, lens), 2) for l in (plumb or ())],
        "plumbPxPlain": [round(straightness(l, plain), 2) for l in (plumb or ())],
        "H": Hm,
    }
