"""Decay Kalman velocity while a BoT-SORT track is lost.

WHY
---
Measured on this footage: when a robot goes behind a Hub structure, BoT-SORT keeps
extrapolating its last velocity. Over a 12-27 frame occlusion the predicted box walks
right off the robot, so on reappearance IoU is ~0 and no match threshold can rebind
it -- a new id is spawned instead. One observed handoff spanned 23 frames while the
robot moved 5 px.

Constant-velocity is not merely noisy here, it is wrong in a knowable direction: at
5 m/s a 2 s occlusion predicts 10 m of travel, more than half the field. The observed
handoff distances were 5-184 px (median ~80), i.e. robots reappear near where they
vanished, because they are usually manoeuvring around an obstacle rather than
barrelling through it.

Ultralytics ALREADY applies this reasoning to box size -- BOTrack.multi_predict zeroes
the width/height velocities (indices 6, 7) for any track that is not Tracked. It just
never extended it to position (indices 4, 5). This patch does, with a decay factor
rather than a hard zero so that a 1-2 frame gap still benefits from real motion while
a long one converges to "it is about where I last saw it".

Total extra drift is bounded by a geometric series: v0 * gamma / (1 - gamma).
    gamma 0.0 -> 0 frames of coasting (hard stop, the size treatment)
    gamma 0.7 -> ~2.3 frames
    gamma 0.8 -> ~4 frames
    gamma 0.9 -> ~9 frames  (probably too loose)

CAUTION
-------
This monkeypatches ultralytics internals and is therefore version-fragile. It is
opt-in (`--lost-decay`), it verifies the function it is replacing still looks the way
we expect, and it refuses to patch rather than silently doing nothing if the upstream
implementation has moved on. Re-check after any ultralytics upgrade.

Also note this helps the Stage 4 live path specifically: offline `rtrack.stitch` can
look ahead to repair fragments, but a livestream cannot.
"""

from __future__ import annotations

import numpy as np

_applied: float | None = None


def apply(gamma: float = 0.8, strict: bool = True) -> bool:
    """Patch BOTrack.multi_predict to decay lost-track position velocity.

    Returns True if the patch was applied. Idempotent per gamma.
    """
    global _applied
    if _applied == gamma:
        return True

    from ultralytics.trackers import bot_sort
    from ultralytics.trackers.basetrack import TrackState

    BOTrack = bot_sort.BOTrack

    # Guard: make sure upstream still does what we think before we replace it.
    import inspect
    src = inspect.getsource(BOTrack.multi_predict)
    expected = ("multi_mean[i][6] = 0", "multi_mean[i][7] = 0", "shared_kalman.multi_predict")
    missing = [e for e in expected if e not in src]
    if missing:
        msg = (f"botpatch: ultralytics BOTrack.multi_predict no longer matches the "
               f"implementation this patch was written against (missing {missing}). "
               f"Re-read it before trusting --lost-decay.")
        if strict:
            raise RuntimeError(msg)
        print(f"[botpatch] SKIPPED -- {msg}")
        return False

    def multi_predict(stracks) -> None:
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0          # width velocity  (upstream behaviour)
                multi_mean[i][7] = 0          # height velocity (upstream behaviour)
                multi_mean[i][4] *= gamma     # x velocity  <-- added
                multi_mean[i][5] *= gamma     # y velocity  <-- added
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    BOTrack.multi_predict = staticmethod(multi_predict)
    _applied = gamma
    print(f"[botpatch] lost-track velocity decay active, gamma={gamma} "
          f"(~{gamma/(1-gamma):.1f} frames of coasting)" if gamma < 1
          else "[botpatch] gamma>=1 is not a decay")
    return True
