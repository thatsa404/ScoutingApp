"""Stage 2 -- automatic calibration from field AprilTags. No user intervention.

Goal: point it at a stream and have a metric field homography within a minute, at any
venue, with nobody clicking anything.

How it works
------------
1. BLIND SWEEP. Detect tags across the frame at several upscales. A static camera
   means every frame re-measures the same geometry, so corner estimates are the
   median over many detections (a ~35 px tag is individually jittery; the median's
   standard error is what matters and is reported).

2. BOOTSTRAP. The blind sweep finds only the nearest few tags. Fit a rough pose from
   those, PREDICT where all 32 tags should land, and re-detect in small ROIs around
   each prediction at high zoom. Distant tags that are invisible to a whole-frame
   search are recoverable once you know where to look.

3. FIT. Solve pose + focal length + principal point by minimising reprojection error,
   with solvePnP inside the loop (pose is closed-form once the intrinsics are fixed,
   so only 3 parameters are actually searched). Corner-order and axis-sign
   conventions are searched, not assumed -- both fail silently if guessed wrong.

4. VALIDATE, independently of the fit. Leave-one-out cross-validation: refit without
   each tag and measure that tag's reprojection error. This is the check that
   generalises to any venue, and it is the one that matters -- an earlier version of
   this module reported 1.44 px reprojection error while placing the camera 15 cm off
   the floor and mis-projecting the field by 1680 px.

Known limitation
----------------
The tags a broadcast camera can see are often close to coplanar -- it sees one face
of each Hub, and those faces are parallel. Coplanar PnP with unknown focal length is
DEGENERATE (focal length trades against depth for identical projections). The
bootstrap exists largely to break that degeneracy by recovering a tag at a different
height or orientation. If it cannot, the fit is reported as degenerate rather than
returned as a confident wrong answer.

    uv run -m rtrack.autocal GSxbsE42o5o
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id

TAG_SIZE_M = 0.1651
LAYOUT = "2026-apriltag-layout.json"


def _params():
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax = 3, 23
    p.adaptiveThreshWinSizeStep = 5
    p.minMarkerPerimeterRate = 0.002
    p.polygonalApproxAccuracyRate = 0.06
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


DET = None


def detector():
    global DET
    if DET is None:
        DET = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _params())
    return DET


def load_layout():
    p = C.DATA_DIR / "field" / LAYOUT
    if not p.exists():
        raise SystemExit(f"{p} not found -- see README, extract via robotpy-apriltag")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {t["ID"]: t for t in doc["tags"]}, doc["field"]


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q["W"], q["X"], q["Y"], q["Z"]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tag_corners_3d(tag, order: int, flip: int) -> np.ndarray:
    t = tag["pose"]["translation"]
    c = np.array([t["x"], t["y"], t["z"]], float)
    R = quat_to_R(tag["pose"]["rotation"]["quaternion"])
    s = TAG_SIZE_M / 2.0
    base = [(+s, +s), (-s, +s), (-s, -s), (+s, -s)]
    if flip:
        base = [(-a, b) for a, b in base]
    base = base[order:] + base[:order]
    return np.array([c + R @ np.array([0.0, a, b]) for a, b in base])


def _merge(acc: dict[int, list], verbose: bool) -> dict[int, np.ndarray]:
    out = {}
    for tid, obs in acc.items():
        a = np.stack(obs)
        if len(a) < 6:
            continue
        med = np.median(a, axis=0)
        d = np.linalg.norm(a - med, axis=2).mean(axis=1)
        iqr = np.percentile(d, 75) - np.percentile(d, 25)
        keep = d < np.median(d) + 2.5 * (iqr + 1e-6)
        if keep.sum() >= 6:
            a = a[keep]
        med = np.median(a, axis=0)
        se = float(np.linalg.norm(a - med, axis=2).mean()) / np.sqrt(len(a))
        if se < 2.0:
            out[tid] = med
            if verbose:
                print(f"    tag {tid:>3}: {len(a):>4} obs, median SE {se:.2f} px")
    return out


def median_plate(stem: str, frames: list[int]) -> np.ndarray:
    """One clean static image from many frames.

    The camera is locked off (Stage 0 measured 0.04 px drift across the whole match),
    so every frame re-measures identical geometry. Median-stacking removes robots,
    people and fuel, leaving only static structure -- and lets detection run ONCE
    instead of once per frame. This is the difference between seconds and minutes.
    """
    cap = cv2.VideoCapture(str(raw_path(stem)))
    acc = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            acc.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not acc:
        raise SystemExit("could not read any frames")
    return np.median(np.stack(acc), axis=0).astype(np.uint8)


def detect_on(gray: np.ndarray, scales=(1, 2)) -> dict[int, np.ndarray]:
    out = {}
    for s in scales:
        g = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s,
                                           interpolation=cv2.INTER_CUBIC)
        cs, ids, _ = detector().detectMarkers(g)
        if ids is None:
            continue
        for c, i in zip(cs, ids.ravel()):
            # prefer the higher-zoom estimate when a tag is found at several scales
            out[int(i)] = c.reshape(4, 2) / s
    return out


def tiled_detect(gray: np.ndarray, tile: int = 320, overlap: int = 80,
                 zoom: int = 5) -> dict[int, np.ndarray]:
    """Systematic high-zoom search over overlapping tiles.

    A whole-frame sweep must upscale the entire image to reach the zoom a distant
    tag needs, which is why the 3x pass cost seconds per frame and still found the
    far tag only ~5% of the time. Tiling reaches 5x over the same area for
    comparable work, and needs no pose estimate -- unlike the bootstrap, which is
    useless when the seed pose is itself degenerate.
    """
    h, w = gray.shape[:2]
    step = tile - overlap
    out = {}
    for y0 in range(0, max(1, h - overlap), step):
        for x0 in range(0, max(1, w - overlap), step):
            x1, y1 = min(w, x0 + tile), min(h, y0 + tile)
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue
            big = cv2.resize(gray[y0:y1, x0:x1], None, fx=zoom, fy=zoom,
                             interpolation=cv2.INTER_CUBIC)
            cs, ids, _ = detector().detectMarkers(big)
            if ids is None:
                continue
            for c, i in zip(cs, ids.ravel()):
                out.setdefault(int(i), c.reshape(4, 2) / zoom + np.array([x0, y0]))
    return out


def targeted_detect(gray: np.ndarray, preds: dict[int, np.ndarray],
                    pad: int = 90, zoom: int = 6) -> dict[int, np.ndarray]:
    """Re-detect in small ROIs around predicted tag locations, heavily upscaled.

    A tag 12 px across is hopeless in a whole-frame search but decodable at 6x in a
    180 px crop. Cheap because the crops are tiny -- which is why the blind pass no
    longer needs an expensive full-frame 3x sweep.
    """
    h, w = gray.shape[:2]
    out = {}
    for tid, (x, y) in preds.items():
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(w, x + pad)), int(min(h, y + pad))
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        roi = gray[y0:y1, x0:x1]
        big = cv2.resize(roi, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        cs, ids, _ = detector().detectMarkers(big)
        if ids is None:
            continue
        for c, i in zip(cs, ids.ravel()):
            out[int(i)] = c.reshape(4, 2) / zoom + np.array([x0, y0])
    return out


def _reproj(obj, img_pts, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return 1e9, None, None, K
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    return err, rvec, tvec, K


def fit(obs, layout, w, h, solve_pp: bool, order=None, flip=None):
    ids = sorted(i for i in obs if i in layout)
    img_pts = np.concatenate([obs[i] for i in ids]).astype(np.float64)
    combos = ([(order, flip)] if order is not None
              else list(itertools.product(range(4), (0, 1))))
    best = None
    for o, fl in combos:
        obj = np.concatenate([tag_corners_3d(layout[i], o, fl)
                              for i in ids]).astype(np.float64)
        # coarse f sweep first -- cheap and avoids local minima
        for f0 in np.arange(800, 4200, 50.0):
            e, r, t, K = _reproj(obj, img_pts, f0, w / 2, h / 2)
            if best is None or e < best[0]:
                best = (e, f0, w / 2, h / 2, o, fl, r, t, K, obj)
    e, f0, cx0, cy0, o, fl, *_ = best
    obj = np.concatenate([tag_corners_3d(layout[i], o, fl) for i in ids]).astype(float)

    if solve_pp:
        from scipy.optimize import minimize
        def _cost(p):
            f, cx, cy = p
            # Nelder-Mead has no bounds; reject nonsense rather than let it wander
            # to f=0, which projects everything to a point and scores well.
            if not (400.0 < f < 6000.0) or abs(cx - w / 2) > w / 3                     or abs(cy - h / 2) > h / 3:
                return 1e9
            return _reproj(obj, img_pts, f, cx, cy)[0]

        res = minimize(_cost,
                       x0=[f0, cx0, cy0], method="Nelder-Mead",
                       options={"xatol": 0.5, "fatol": 1e-4, "maxiter": 4000})
        f0, cx0, cy0 = res.x
    else:
        from scipy.optimize import minimize_scalar
        # bounded, not bracketed: the coarse sweep's minimum can sit at an endpoint
        # or on a flat region, and Brent's bracketing then raises.
        r = minimize_scalar(lambda ff: _reproj(obj, img_pts, ff, cx0, cy0)[0],
                            bounds=(400.0, 6000.0), method="bounded",
                            options={"xatol": 0.5})
        f0 = float(r.x)

    e, rvec, tvec, K = _reproj(obj, img_pts, f0, cx0, cy0)
    return {"err": e, "f": f0, "cx": cx0, "cy": cy0, "order": o, "flip": fl,
            "rvec": rvec, "tvec": tvec, "K": K, "ids": ids}


def loo_validate(obs, layout, w, h, solve_pp, order=None, flip=None
                 ) -> list[tuple[int, float]]:
    """Leave-one-out: refit without each tag, measure that tag's error.

    Venue-agnostic, needs no known scene geometry, and is the check that catches a
    fit which merely interpolates its own inputs.
    """
    ids = sorted(i for i in obs if i in layout)
    out = []
    for held in ids:
        sub = {i: obs[i] for i in ids if i != held}
        if len(sub) < 3:
            continue
        r = fit(sub, layout, w, h, solve_pp, order=order, flip=flip)
        obj = tag_corners_3d(layout[held], r["order"], r["flip"]).astype(float)
        proj, _ = cv2.projectPoints(obj, r["rvec"], r["tvec"], r["K"], None)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - obs[held], axis=1).mean())
        out.append((held, err))
    return out


def floor_homography(K, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Hf = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])
    Hf = Hf / Hf[2, 2]
    return Hf, np.linalg.inv(Hf)


def plausible(cam, f, w) -> list[str]:
    """Venue-agnostic sanity checks. Cheap, and they catch gross failures."""
    bad = []
    if not (1.5 < cam[2] < 25):
        bad.append(f"camera height {cam[2]:.2f} m is implausible")
    fov = 2 * np.degrees(np.arctan(w / (2 * f)))
    if not (15 < fov < 95):
        bad.append(f"horizontal FOV {fov:.1f} deg is implausible")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: AprilTag auto-calibration.")
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--no-pp", action="store_true",
                    help="do not solve for the principal point")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    layout, field = load_layout()
    solve_pp = not args.no_pp

    cap = cv2.VideoCapture(str(raw_path(stem)))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frames = list(np.linspace(100, min(n - 5, 5300), args.frames, dtype=int))

    import time
    t0 = time.perf_counter()
    print(f"[1] median plate from {len(frames)} frames", flush=True)
    plate = median_plate(stem, frames)
    cv2.imwrite(str(C.STAGE2_DIR / f"{stem}_plate.png"), plate)
    print(f"    {time.perf_counter()-t0:.1f}s", flush=True)

    # The plate is clean but median-stacking softens tag edges, which costs
    # detections. Search it at higher zoom, then top up from a few RAW frames --
    # detection is stochastic, and an occasional favourable frame finds tags the
    # plate misses. Both are cheap; only the old per-frame full sweep was not.
    obs = detect_on(plate, scales=(1, 2, 3))
    print(f"[2] blind detect on plate -> {sorted(obs)} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    cap = cv2.VideoCapture(str(raw_path(stem)))
    for f in frames[::max(1, len(frames) // 4)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            for k, v in detect_on(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  scales=(1, 2)).items():
                obs.setdefault(k, v)
    cap.release()
    print(f"    + raw frames -> {sorted(obs)} ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    for k, v in tiled_detect(plate).items():
        obs.setdefault(k, v)
    print(f"    + tiled plate -> {sorted(obs)} ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # The plate is clean but soft; a marginal distant tag decodes only on sharp
    # source, and only in a fraction of frames. Tile a few raw frames too.
    cap = cv2.VideoCapture(str(raw_path(stem)))
    for f in frames[::max(1, len(frames) // 5)][:5]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for k, v in tiled_detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).items():
            obs.setdefault(k, v)
    cap.release()
    print(f"    + tiled raw -> {sorted(obs)} ({time.perf_counter()-t0:.1f}s)",
          flush=True)
    if len(obs) < 3:
        raise SystemExit("too few tags for any fit")

    if not args.no_bootstrap:
        rough = fit(obs, layout, w, h, solve_pp=False)
        preds = {}
        for tid, tag in layout.items():
            if tid in obs:
                continue
            obj = tag_corners_3d(tag, rough["order"], rough["flip"]).astype(float)
            proj, _ = cv2.projectPoints(obj, rough["rvec"], rough["tvec"],
                                        rough["K"], None)
            ctr = proj.reshape(-1, 2).mean(axis=0)
            if 0 < ctr[0] < w and 0 < ctr[1] < h:
                preds[tid] = ctr
        print(f"[3] bootstrap: rough pose predicts {len(preds)} more tags in frame "
              f"{sorted(preds)}", flush=True)
        if preds:
            extra = targeted_detect(plate, preds)
            new_ids = {k: v for k, v in extra.items() if k not in obs and k in layout}
            print(f"    recovered {sorted(new_ids)} ({time.perf_counter()-t0:.1f}s)",
                  flush=True)
            obs.update(new_ids)

    ids = sorted(i for i in obs if i in layout)
    P = np.array([[layout[i]["pose"]["translation"][k] for k in "xyz"] for i in ids])
    sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    cond = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    print(f"\n[3] fitting with {len(ids)} tags {ids}")
    print(f"    geometry conditioning (min/max singular value): {cond:.4f}")
    if cond < 0.01:
        print("    WARNING: tag centres are near-coplanar/collinear. Focal length "
              "trades against depth here; treat the result as unreliable.")

    best = fit(obs, layout, w, h, solve_pp=solve_pp)
    R, _ = cv2.Rodrigues(best["rvec"])
    cam = (-R.T @ best["tvec"]).ravel()
    fov = 2 * np.degrees(np.arctan(w / (2 * best["f"])))
    print(f"    f={best['f']:.0f}px  cx={best['cx']:.1f} cy={best['cy']:.1f}  "
          f"(image centre {w/2:.0f},{h/2:.0f})")
    print(f"    reprojection {best['err']:.2f} px over {len(ids)*4} points")
    print(f"    camera x={cam[0]:.2f} y={cam[1]:.2f} z={cam[2]:.2f} m, HFOV {fov:.1f} deg")
    for msg in plausible(cam, best["f"], w):
        print(f"    IMPLAUSIBLE: {msg}")

    print(f"\n[4] leave-one-out validation")
    loo = loo_validate(obs, layout, w, h, solve_pp,
                       order=best["order"], flip=best["flip"])
    for tid, e in loo:
        print(f"    hold out tag {tid:>3}: reprojection {e:8.1f} px")
    loo_med = float(np.median([e for _, e in loo])) if loo else float("nan")
    print(f"    median held-out error: {loo_med:.1f} px")
    verdict = ("USABLE" if loo_med < 15 and cond >= 0.01 and
               not plausible(cam, best["f"], w) else "NOT TRUSTWORTHY")
    print(f"    VERDICT: {verdict}")

    Hf, Hi = floor_homography(best["K"], best["rvec"], best["tvec"])
    doc = {"video": stem, "mode": "apriltag-pnp", "verdict": verdict,
           "focalPx": best["f"], "cx": best["cx"], "cy": best["cy"],
           "hfovDeg": round(float(fov), 2), "tagsUsed": ids,
           "geometryConditioning": round(cond, 5),
           "reprojErrorPx": round(best["err"], 3),
           "looMedianPx": round(loo_med, 2),
           "loo": [[t, round(e, 2)] for t, e in loo],
           "cameraPosM": cam.tolist(), "K": best["K"].tolist(),
           "rvec": best["rvec"].ravel().tolist(),
           "tvec": best["tvec"].ravel().tolist(),
           "H_fieldm_to_imagepx": Hf.tolist(), "H_imagepx_to_fieldm": Hi.tolist()}
    dest = C.CALIB_DIR / f"{stem}_autocal.json"
    dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
