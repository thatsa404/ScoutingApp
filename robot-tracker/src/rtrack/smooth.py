"""Route smoothing: a Kalman (RTS) smoother over each team's full-rate positions.

Published routes zigzag more than robots do. Measured on 833 published routes: the drawn
5 Hz path ran 8% longer than a smoothed copy of itself (p90 14%), and 6.1% of moving
steps reversed direction by more than 120 degrees within a fifth of a second -- which a
driven robot essentially never does. Two causes, both fixed here:

  * Position noise is small but CORRELATED: about 5 cm sd about a 1.5 s smoothed path
    (p99 17 cm), with neighbouring 15 Hz samples correlated 0.73 at one step and 0.37 at
    two -- a box changing shape for a few frames, not independent jitter. A smoother
    that assumed white noise (the maximum-likelihood fit lands at sigma_z ~4 cm) would
    read the wobble as motion and keep it.
  * The export THINNED 15 Hz to 5 Hz by picking every third sample, keeping whatever
    wobble landed on it. Smoothing happens before thinning now.

MODEL. Constant velocity per axis, white-noise acceleration SIGMA_A -- set from the
measured spread of real acceleration (1 s window: median 2.3 m/s^2, p99 6.9) rather
than the white-noise fit, which inflates it to absorb the correlated wobble -- and an
effective measurement noise SIGMA_Z sized to that wobble. Forward Kalman filter, then a
Rauch-Tung-Striebel backward pass, so every point uses the samples on BOTH sides.

NEVER ACROSS A BREAK. A segment ends at a gap longer than SEG_GAP_S or at a step beyond
JUMP_MULT x the solver's movement envelope (the export's "teleport"): smoothing across
either would blur a swap into a plausible-looking curve and hide the evidence for it.

Each smoothed point carries `sd`, its position uncertainty in metres -- small inside a
long clean run, larger near a segment's ends and on short segments.
"""

from __future__ import annotations

import math

import numpy as np

from .solve import distance_budget

SIGMA_A = 3.0        # m/s^2, process noise: spread of real acceleration
SIGMA_Z = 0.10       # m, effective measurement noise (correlated box wobble)
SEG_GAP_S = 0.45     # a longer gap ends a segment
JUMP_MULT = 1.5      # a step beyond this x the envelope ends a segment


def _segments(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    cuts = [0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if dt > SEG_GAP_S or math.hypot(x[k] - x[k - 1], y[k] - y[k - 1]) > \
                JUMP_MULT * distance_budget(max(dt, 1 / 15)):
            cuts.append(k)
    cuts.append(len(t))
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _rts(t: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One axis: smoothed positions and their variances."""
    n = len(t)
    xs = np.zeros((n, 2)); Ps = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2)); Fs = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0]); P = np.diag([SIGMA_Z ** 2, 4.0])
    R = SIGMA_Z ** 2
    for k in range(n):
        if k:
            dt = t[k] - t[k - 1]
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = SIGMA_A ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
            x = F @ x; P = F @ P @ F.T + Q
            Fs[k] = F
        xp[k] = x; Pp[k] = P
        S = P[0, 0] + R
        K = P[:, 0] / S
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0, :])
        xs[k] = x; Ps[k] = P
    for k in range(n - 2, -1, -1):           # Rauch-Tung-Striebel backward pass
        F = Fs[k + 1]
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0], Ps[:, 0, 0]


def smooth(samples: list[dict]) -> list[dict]:
    """Return copies of time-ordered samples (dicts with t, x, y) with x, y smoothed and
    `sd` (metres) added. Order and count are unchanged."""
    if len(samples) < 3:
        return [dict(s, sd=SIGMA_Z) for s in samples]
    t = np.array([float(s["t"]) for s in samples])
    x = np.array([float(s["x"]) for s in samples])
    y = np.array([float(s["y"]) for s in samples])
    out = [dict(s) for s in samples]
    for a, b in _segments(t, x, y):
        if b - a < 3:
            for k in range(a, b):
                out[k]["sd"] = SIGMA_Z
            continue
        sx, vx = _rts(t[a:b], x[a:b])
        sy, vy = _rts(t[a:b], y[a:b])
        for i, k in enumerate(range(a, b)):
            out[k]["x"], out[k]["y"] = float(sx[i]), float(sy[i])
            out[k]["sd"] = float(math.sqrt((vx[i] + vy[i]) / 2))
    return out
