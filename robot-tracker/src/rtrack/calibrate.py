"""Stage 2 -- homography from broadcast pixels to field metres.

Stage 0 established the ideal conditions for this: a locked-off camera with 0.04 px
median translation and 1.0000 scale across the entire match. One homography covers
the whole video.

Two ways to supply correspondences:

  interactive   click a point in the video, then the matching point on the field PNG
                uv run -m rtrack.calibrate GSxbsE42o5o --frame 200 --interactive

  headless      supply them as JSON, for scripted iteration
                uv run -m rtrack.calibrate GSxbsE42o5o --frame 200 --points pts.json

Either way it writes calib/<video_id>.json and a warped-overlay preview. LOOK AT THE
PREVIEW: a bad correspondence can still produce a low reprojection error while
visibly shearing the field into a trapezoid, and the number will not tell you.

Point selection rules that matter:
  * FLOOR PLANE ONLY. Tape line intersections, carpet seams, the base of a guardrail.
    Never a Hub top, a Trench rail, or anything with height -- a point 1 m off the
    floor silently wrecks the fit.
  * Spread across all four quadrants. Points clustered near the camera give a
    homography that is beautifully accurate exactly where you do not need it.
  * 8-10 points, not 4. RANSAC needs redundancy to tell you which click was bad.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import lens as _lens
from .acquire import raw_path, video_id


def load_field_ref() -> dict:
    p = C.CALIB_DIR / f"field_ref_{C.YEAR}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def field_px_to_m(ref: dict, x: float, y: float) -> tuple[float, float]:
    """Field-PNG pixel -> metres, origin bottom-left of the field rect, +y up."""
    r = ref["fieldRectPx"]
    ppm = ref["pxPerMeter"]
    return ((x - r["x0"]) / ppm, (r["y1"] - y) / ppm)


def field_m_to_px(ref: dict, X: float, Y: float) -> tuple[float, float]:
    r = ref["fieldRectPx"]
    ppm = ref["pxPerMeter"]
    return (r["x0"] + X * ppm, r["y1"] - Y * ppm)


PLATE_FROM_S = 12.0     # skip title cards and any opening reframe; see grab_plate
PLATE_TO_S = 175.0      # a 150 s match plus its lead-in, before post-match cutaways


def grab_plate(stem: str, n: int = 25,
               t_from: float = PLATE_FROM_S, t_to: float = PLATE_TO_S) -> np.ndarray:
    """Median-stack frames into a clean static image.

    Robots, people and fuel are transient and wash out; the carpet, tape lines and
    guardrails remain. Far easier to click accurately than a raw frame, where the
    features you need are usually behind a robot.

    THE WINDOW IS IN SECONDS, NOT FRAMES, and that matters. This used to stack
    linspace(100, 5300) -- frame numbers tuned for a 30 fps broadcast. On the 58 fps
    2026mawor stream those same numbers mean t = 1.7 s to 91 s, which swallows the
    full-screen title card (to t=3.4 s) AND the opening reframe (the camera settles at
    t=6.5 s, 227 px and 1.2% of zoom away from where it ends up), then covers only half
    the match. Stacking two different camera geometries produces a plate that looks
    plausible and is doubled, and every point clicked on it is wrong by some fraction of
    that offset.

    Median-stacking is robust enough to survive a couple of bad frames, which is exactly
    why this would not have announced itself.
    """
    cap = cv2.VideoCapture(str(raw_path(stem)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = max(0, int(t_from * fps))
    hi = min(total - 5, int(t_to * fps))
    if hi <= lo:                      # short clip: fall back to the middle half
        lo, hi = int(total * 0.25), int(total * 0.75)
    acc = []
    for f in np.linspace(lo, hi, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            acc.append(img)
    cap.release()
    if not acc:
        raise SystemExit("could not read frames for the plate")
    print(f"[calibrate] plate: {len(acc)} frames over t={lo / fps:.1f}-{hi / fps:.1f}s "
          f"({fps:.2f} fps source)")
    return np.median(np.stack(acc), axis=0).astype(np.uint8)


def grab_frame(stem: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(raw_path(stem)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame}")
    return img


def geometry_warnings(src_px: np.ndarray, h: int) -> list[str]:
    """Catch degenerate click patterns BEFORE fitting.

    In this camera view the field's full 8 m depth occupies only ~330 video pixels,
    so points crowded into a narrow horizontal band produce a near-singular
    homography: a tiny video-space spread mapping to a huge field-space spread gives
    an enormous gradient, and the warp smears into radial streaks. The reprojection
    error will not warn you -- 4 points always fit a homography exactly.
    """
    msgs = []
    ys, xs = src_px[:, 1], src_px[:, 0]
    yspan, xspan = float(ys.max() - ys.min()), float(xs.max() - xs.min())
    if yspan < 150:
        msgs.append(f"points span only {yspan:.0f} px vertically in the video. "
                    f"Spread them from the far rail to the near edge of the carpet.")
    if xspan < 600:
        msgs.append(f"points span only {xspan:.0f} px horizontally.")
    # quadrant coverage about the centroid of the clicked points
    cx, cy = xs.mean(), ys.mean()
    quads = {(x > cx, y > cy) for x, y in zip(xs, ys)}
    if len(quads) < 4:
        msgs.append(f"points occupy only {len(quads)} of 4 quadrants; "
                    "a homography needs spread in both axes.")
    return msgs


def trace_plumb(plate: np.ndarray, y0: int = 790, y1: int = 980,
                thr: int = 140, step: int = 8) -> np.ndarray | None:
    """Trace the bright near barrier -- a straight world line -- across the frame.

    Returns None unless the trace is a genuinely smooth curve. That check is not
    optional. Two other structures traced this way LOOKED like contradicting evidence
    about the lens until a quadratic fit showed they were noise: residual 6.01 -> 5.84
    for a line fit vs a quadratic, i.e. no smooth curve at all, just a gradient
    follower hopping between features. The real barrier goes 17.15 -> 0.61.

    A noisy trace cannot be straightened by any lens model, so feeding one to the
    solver would drag the fit toward nonsense while looking like a constraint.
    """
    g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    xs, ys = [], []
    for x in range(140, plate.shape[1] - 130, step):
        col = g[y0:y1, x].astype(float)
        if col.max() > thr:
            xs.append(float(x))
            ys.append(float(int(np.argmax(col)) + y0))
    if len(xs) < 40:
        return None
    xs, ys = np.array(xs), np.array(ys)
    for _ in range(4):
        c = np.polyfit(xs, ys, 2)
        r = ys - np.polyval(c, xs)
        keep = np.abs(r) < 2.0 * r.std()
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 40:
            return None
    lin = np.sqrt(np.mean((ys - np.polyval(np.polyfit(xs, ys, 1), xs)) ** 2))
    quad = np.sqrt(np.mean((ys - np.polyval(np.polyfit(xs, ys, 2), xs)) ** 2))
    if quad > 0.4 * lin:
        print(f"[plumb] trace rejected: a quadratic fits it no better than a line "
              f"({lin:.2f} -> {quad:.2f} px), so it is noise, not a curve")
        return None
    print(f"[plumb] traced {len(xs)} points along the near barrier; "
          f"line residual {lin:.2f} px, quadratic {quad:.2f} px -- a real curve")
    return np.stack([xs, ys], axis=1)


def compute(src_px: np.ndarray, dst_px: np.ndarray, ref: dict,
            thresh_m: float = 2.00, lens_params: dict | None = None
            ) -> tuple[np.ndarray, dict]:
    """Homography video-pixels -> field-PNG-pixels, plus an error report.

    The RANSAC threshold is specified in METRES and converted, because field pixels
    are an artefact of the render resolution. The previous default of 3.0 field px
    was 1.7 cm at 174.74 px/m -- far tighter than achievable click precision, so
    RANSAC rejected every honest point, fell back to a minimal 4-point set (which
    fits any homography exactly) and returned a singular warp with four 0.000 errors.
    """
    # Undistort FIRST if we have lens parameters. Everything downstream -- RANSAC,
    # the least-squares comparison, the residual report -- then operates on ideal
    # pinhole coordinates, which is the only space a homography is valid in.
    src_fit = src_px
    if lens_params:
        src_fit = _lens.undistort_points(src_px, lens_params).astype(np.float32)

    thresh = thresh_m * ref["pxPerMeter"]
    H, inliers = cv2.findHomography(src_fit, dst_px, cv2.RANSAC,
                                    ransacReprojThreshold=thresh)
    if H is None:
        raise SystemExit("findHomography failed -- check the correspondences")
    # Least-squares over ALL points, for comparison. If RANSAC kept only 4 it has
    # fitted a minimal set exactly and told you nothing; the LSQ error is honest.
    Hl, _ = cv2.findHomography(src_fit, dst_px, 0)
    if Hl is not None:
        pl = cv2.perspectiveTransform(src_fit.reshape(-1, 1, 2), Hl).reshape(-1, 2)
        el = np.linalg.norm(pl - dst_px, axis=1) / ref["pxPerMeter"]
        print(f"[lsq] all-point least-squares error (m): "
              f"mean {el.mean():.2f}, max {el.max():.2f}")

    proj = cv2.perspectiveTransform(src_fit.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_px = np.linalg.norm(proj - dst_px, axis=1)
    err_m = err_px / ref["pxPerMeter"]

    report = {
        "pointCount": int(len(src_px)),
        "inliers": int(inliers.sum()) if inliers is not None else None,
        "reprojErrorFieldPx": {"mean": float(err_px.mean()), "max": float(err_px.max())},
        # The metres number is the one that matters -- field px are an artefact of
        # whatever resolution the PNG happens to be rendered at.
        "reprojErrorM": {"mean": float(err_m.mean()), "max": float(err_m.max())},
        "perPointErrorM": [round(float(e), 3) for e in err_m],
        "lens": ({k: lens_params[k] for k in ("f", "k1", "k2", "cx", "cy")
                  if k in lens_params} if lens_params else None),
    }
    return H, report


def preview(frame: np.ndarray, field: np.ndarray, H: np.ndarray,
            dest: Path, alpha: float = 0.5, lens: dict | None = None) -> None:
    """Warp the video frame into field space and blend it over the field render.

    This is the check that actually catches mistakes.

    With lens parameters it samples the ORIGINAL frame directly: for every field
    pixel, H^-1 gives the ideal pixel and the distortion model gives the real one.
    Rectifying the frame first and then warping would resample twice, losing detail
    and cropping whatever fell outside the rectified canvas -- the near corners, on
    this footage. One remap, no intermediate image.
    """
    fh, fw = field.shape[:2]
    if lens:
        h, w = frame.shape[:2]
        gx, gy = np.meshgrid(np.arange(fw, dtype=np.float32),
                             np.arange(fh, dtype=np.float32))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
        dis = _lens.distort_points(und, lens, w, h)
        mx = dis[:, 0].reshape(fh, fw).astype(np.float32)
        my = dis[:, 1].reshape(fh, fw).astype(np.float32)
        warped = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        warped = cv2.warpPerspective(frame, H, (fw, fh))
    mask = (warped.sum(axis=2) > 0)[..., None]
    blend = np.where(mask, cv2.addWeighted(field, 1 - alpha, warped, alpha, 0), field)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), cv2.resize(blend, (1600, int(1600 * fh / fw))))


def reproject_overlay(frame: np.ndarray, H: np.ndarray, ref: dict,
                      dest: Path, lens: dict | None = None) -> dict:
    """Draw the predicted field grid back onto the video frame.

    This is the diagnostic that actually catches a bad fit. The warped overlay is
    hard to read because most of the target is empty, and a homography can look
    plausible there while placing the field boundary metres off. Here you are
    comparing predicted geometry against the real thing in the same image: if the
    cyan far edge does not sit on the far guardrail and the orange near edge does not
    sit on the near one, the calibration is wrong no matter what the residual says.

    Returns where the four field corners land, which is the quickest numeric tell --
    a corner you can plainly see in the video landing outside the frame means the fit
    is extrapolating into a region no calibration point constrained.
    """
    Hi = np.linalg.inv(H)
    r, ppm = ref["fieldRectPx"], ref["pxPerMeter"]
    FL, FW = ref["fieldSizeM"]
    img = frame.copy()
    h, w = img.shape[:2]

    def to_video(X, Y):
        """Field metres -> pixels in the ORIGINAL, un-rectified frame.

        H maps undistorted pixels, so the result has to be pushed back through the
        distortion model. Drawing on the raw frame instead of a rectified one keeps
        the video exactly as the detector sees it -- and straight field lines come out
        curved, which is the correct depiction of what this camera does.
        """
        fx = r["x0"] + np.asarray(X, float) * ppm
        fy = r["y1"] - np.asarray(Y, float) * ppm
        pts = np.stack([fx, fy], axis=-1).astype(np.float32).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, Hi).reshape(-1, 2)
        if lens:
            return _lens.distort_points(und, lens, w, h)
        return und

    def draw(pts, col, th=2):
        for a, b in zip(pts, pts[1:]):
            # NaN marks a point past the lens model's fold radius, where the radial
            # polynomial stops being monotonic and returns a meaningless position.
            # Skipping leaves an honest gap; drawing produced a hook that looked like
            # real geometry.
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            if all(abs(v) < 1e5 for v in (*a, *b)):
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, th,
                         cv2.LINE_AA)

    n = 60
    for X in np.arange(2, FL, 2):
        draw(to_video(np.full(n, X), np.linspace(0, FW, n)), (200, 200, 200), 1)
    for Y in np.arange(2, FW, 2):
        draw(to_video(np.linspace(0, FL, n), np.full(n, Y)), (200, 200, 200), 1)
    draw(to_video(np.linspace(0, FL, n), np.full(n, FW)), (255, 255, 0), 3)   # far
    draw(to_video(np.linspace(0, FL, n), np.zeros(n)), (0, 165, 255), 3)      # near
    draw(to_video(np.zeros(n), np.linspace(0, FW, n)), (255, 0, 255), 3)
    draw(to_video(np.full(n, FL), np.linspace(0, FW, n)), (255, 0, 255), 3)
    draw(to_video(np.full(n, FL / 2), np.linspace(0, FW, n)), (0, 255, 0), 3)

    corners, outside, beyond = {}, [], []
    for name, (X, Y) in {"far-L": (0, FW), "far-R": (FL, FW),
                         "near-L": (0, 0), "near-R": (FL, 0)}.items():
        pt = to_video([X], [Y])[0]
        if np.isnan(pt).any():
            # Outside the lens model's valid radius. Reporting a number here is worse
            # than reporting none: the old code claimed both near corners were in
            # frame at plausible pixel coordinates when they are not in the video.
            corners[name] = None
            beyond.append(name)
            continue
        corners[name] = [round(float(pt[0]), 1), round(float(pt[1]), 1)]
        if not (0 <= pt[0] <= w and 0 <= pt[1] <= h):
            outside.append(name)
        if abs(pt[0]) < 1e4 and abs(pt[1]) < 1e4:
            cv2.circle(img, tuple(np.int32(pt)), 9, (0, 0, 255), -1)
            cv2.putText(img, name, tuple(np.int32(pt) + np.array([12, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return {"cornersInVideoPx": corners, "cornersOutsideFrame": outside,
            "cornersBeyondLensModel": beyond}


def point_coverage(src_px: np.ndarray, w: int) -> list[str]:
    """Warn when a whole region of the image has no calibration point.

    A homography is only trustworthy where points constrained it. Dropping outliers
    can silently leave half the field unanchored -- which is exactly how a fit with
    a 0.22 m residual placed the field boundary metres out.
    """
    xs = src_px[:, 0]
    msgs = []
    for lo, hi, name in ((0, w / 3, "left"), (w / 3, 2 * w / 3, "middle"),
                         (2 * w / 3, w, "right")):
        if not ((xs >= lo) & (xs < hi)).any():
            msgs.append(f"NO calibration point in the {name} third of the image; "
                        f"the fit is extrapolating there and cannot be trusted.")
    return msgs


def error_jacobian(H: np.ndarray, ref: dict, frame_shape, dest: Path,
                   lens: dict | None = None) -> dict:
    """Metres of field error per pixel of image error, across the frame.

    Perspective means accuracy is wildly non-uniform. This map defines the usable
    region of the field and every downstream accuracy claim should reference it.
    """
    h, w = frame_shape[:2]
    ppm = ref["pxPerMeter"]
    xs = np.linspace(w * 0.05, w * 0.95, 40)
    ys = np.linspace(h * 0.35, h * 0.95, 30)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    # The grid is in RAW video pixels, because that is the space a detection
    # box jitters in. H expects undistorted pixels, so undistort first or the
    # map answers a question nobody asked.
    def _fwd(q):
        if lens:
            q = _lens.undistort_points(q, lens).astype(np.float32)
        return cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float32),
                                        H).reshape(-1, 2)

    base = _fwd(pts)
    d = 2.0
    sens = np.zeros(len(pts))
    for dx, dy in ((d, 0), (0, d)):
        moved = _fwd(pts + np.array([dx, dy], np.float32))
        sens = np.maximum(sens, np.linalg.norm(moved - base, axis=1) / ppm / d)

    # Only report where the projection lands on the actual field.
    r = ref["fieldRectPx"]
    on = ((base[:, 0] > r["x0"]) & (base[:, 0] < r["x1"]) &
          (base[:, 1] > r["y0"]) & (base[:, 1] < r["y1"]))
    valid = sens[on]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4), dpi=130)
    sc = ax.scatter(base[on, 0], base[on, 1], c=valid * 100, s=26,
                    cmap="turbo", vmin=0, vmax=np.percentile(valid, 95) * 100)
    ax.set_xlim(r["x0"], r["x1"])
    ax.set_ylim(r["y1"], r["y0"])
    ax.set_title("Field error per pixel of image error (cm/px) -- the usable region")
    fig.colorbar(sc, ax=ax, label="cm of field error per px")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)

    return {
        "cmPerPx": {
            "min": round(float(valid.min()) * 100, 1),
            "median": round(float(np.median(valid)) * 100, 1),
            "p90": round(float(np.percentile(valid, 90)) * 100, 1),
            "max": round(float(valid.max()) * 100, 1),
        }
    }


def interactive(stem: str, frame_idx: int, ref: dict,
                use_plate: bool = False,
                seed: list[list[float]] | None = None) -> list[list[float]]:
    """Click a video point, then its match on the field. u/r/s/q.

    `seed` pre-loads correspondences already saved for this video so a session can ADD
    to them rather than start over. Without it, improving an 8-point fit means
    re-clicking all 8 from memory, which risks replacing points that were fine with
    worse ones -- the opposite of the intent.
    """
    frame = grab_plate(stem) if use_plate else grab_frame(stem, frame_idx)
    field = cv2.imread(str(C.REPO_ROOT / ref["image"]))
    fh, fw = field.shape[:2]
    fscale = 1500 / fw
    vscale = 1500 / frame.shape[1]
    fsmall = cv2.resize(field, None, fx=fscale, fy=fscale)
    vsmall = cv2.resize(frame, None, fx=vscale, fy=vscale)

    pairs: list[list[float]] = [list(p) for p in (seed or [])]
    pending: list[float] | None = None
    n_seed = len(pairs)

    def on_video(ev, x, y, *_):
        nonlocal pending
        if ev == cv2.EVENT_LBUTTONDOWN:
            pending = [x / vscale, y / vscale]
            print(f"  video ({pending[0]:.0f}, {pending[1]:.0f}) -- now click the field")

    def on_field(ev, x, y, *_):
        nonlocal pending
        if ev == cv2.EVENT_LBUTTONDOWN and pending is not None:
            pairs.append(pending + [x / fscale, y / fscale])
            print(f"  pair {len(pairs)}: {pairs[-1]}")
            pending = None

    cv2.namedWindow("video"); cv2.setMouseCallback("video", on_video)
    cv2.namedWindow("field"); cv2.setMouseCallback("field", on_field)
    print("click video point then field point. keys: u undo, r reset, s save, q quit")

    while True:
        v, f = vsmall.copy(), fsmall.copy()
        for i, p in enumerate(pairs):
            # Pre-loaded points in grey, ones added this session in yellow, so it is
            # obvious what is new and what is inherited.
            col = (170, 170, 170) if i < n_seed else (0, 255, 255)
            cv2.circle(v, (int(p[0] * vscale), int(p[1] * vscale)), 5, col, -1)
            cv2.putText(v, str(i + 1), (int(p[0] * vscale) + 6, int(p[1] * vscale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.circle(f, (int(p[2] * fscale), int(p[3] * fscale)), 5, col, -1)
            cv2.putText(f, str(i + 1), (int(p[2] * fscale) + 6, int(p[3] * fscale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        hud = f"{len(pairs)} points ({n_seed} preloaded)"
        if pending is not None:
            hud += "  -- now click the MATCHING point on the field"
        cv2.rectangle(v, (0, 0), (v.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(v, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.imshow("video", v); cv2.imshow("field", f)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pairs:
            pairs.pop()
        elif k == ord("r"):
            pairs.clear(); pending = None
        elif k in (ord("s"), ord("q")):
            break
    cv2.destroyAllWindows()
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: video -> field homography.")
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--points", type=Path,
                    help='JSON list of [vx, vy, fx, fy] in video px and field-PNG px, '
                         'or the envelope posted by the remote calibrator')
    ap.add_argument("--export-frame", type=Path, nargs="?", const=Path("AUTO"),
                    metavar="PATH", default=None,
                    help="write a calibration frame bundle for the REMOTE calibrator "
                         "instead of opening a window, then exit. Push it with "
                         "rtrack.relay push-calib")
    ap.add_argument("--ransac", type=float, default=2.00, metavar="METRES",
                    help="RANSAC inlier threshold in METRES. Deliberately loose. "
                         "With only 8-10 points, rejecting any of them can strip a "
                         "whole region of its only constraint -- measured here, a "
                         "0.40 m threshold dropped 3 points, cut the residual from "
                         "0.44 m to 0.22 m, and produced a fit that placed the field "
                         "boundary metres out. The lower residual was the WORSE fit. "
                         "Only tighten this when you have many more points than DOF.")
    ap.add_argument("--plate", action="store_true",
                    help="calibrate against a median-stacked plate (no robots or "
                         "people obscuring the carpet) instead of one raw frame")
    ap.add_argument("--fresh", action="store_true",
                    help="start interactive calibration from zero instead of "
                         "pre-loading the points already saved for this video")
    ap.add_argument("--no-plumb", action="store_true",
                    help="do not use the traced near barrier as a straightness "
                         "constraint (see rtrack.lens: without it the lens "
                         "parameters are degenerate)")
    ap.add_argument("--no-lens", action="store_true",
                    help="skip radial-distortion estimation and fit a plain "
                         "homography (see rtrack.lens for why that is worse here)")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    ref = load_field_ref()

    if args.export_frame is not None:
        # The remote calibrator needs pixels, not a video. Send the median PLATE rather
        # than a raw frame: robots, people and fuel wash out, so the tape lines and
        # guardrail bases you actually want to click are not behind a robot.
        import base64
        plate = grab_plate(stem)
        ok, buf = cv2.imencode(".jpg", plate, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise SystemExit("[calibrate] could not encode the plate")
        doc = {
            "schemaVersion": 1,
            "videoId": stem,
            "frame": None,                       # a plate, not one frame
            "w": int(plate.shape[1]), "h": int(plate.shape[0]),
            "img": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
            "fieldImage": "../field/2026-field.png",
            "fieldRef": {k: ref[k] for k in ("imageSize", "fieldRectPx", "pxPerMeter",
                                             "fieldSizeM") if k in ref},
        }
        # Existing points ride along, so the remote calibrator can ADD to a calibration
        # instead of restarting it. Without this the only way to fix one bad or missing
        # point was to re-click all twelve -- and you cannot know a point is missing
        # until the reprojection comes back from the machine, which is exactly when
        # re-clicking everything is most annoying. --interactive already seeded this way
        # (see below); only the remote path was missing it.
        if not args.fresh:
            prev = C.CALIB_DIR / f"{stem}.json"
            if prev.exists():
                try:
                    fit = json.loads(prev.read_text(encoding="utf-8"))
                    pts = fit.get("points")
                    if pts:
                        doc["points"] = pts
                        print(f"[calibrate] seeding {len(pts)} existing point(s) into "
                              f"the bundle -- --fresh starts empty instead")
                    # The LAST FIT rides back too, so the person who clicked the points
                    # can see what they produced without walking to the machine. The
                    # numbers alone are not enough -- a low residual can be the worse fit
                    # (see --ransac) -- so the reprojection overlay goes with them, which
                    # is the artefact this project treats as the actual gate.
                    doc["fit"] = {k: fit[k] for k in
                                  ("reprojErrorM", "inliers", "pointCount",
                                   "cornersOutsideFrame", "perPointErrorM")
                                  if k in fit}
                    shot = C.STAGE2_DIR / f"{stem}_reproject.png"
                    if shot.exists():
                        img = cv2.imread(str(shot))
                        if img is not None:
                            # JPEG, not the source PNG: 2.3 MB of lossless overlay is
                            # pointless on venue wifi when the judgement being made is
                            # "does that line sit on that guardrail".
                            okj, bufj = cv2.imencode(
                                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                            if okj:
                                doc["reproject"] = ("data:image/jpeg;base64,"
                                                    + base64.b64encode(bufj).decode("ascii"))
                                print(f"[calibrate] including the reprojection overlay "
                                      f"({len(bufj) / 1e6:.1f} MB as JPEG)")
                except Exception as e:
                    print(f"[calibrate] could not read {prev.name} ({e}); starting empty")
        dest = (C.STAGE3_DIR / f"{stem}_calib_frame.json"
                if str(args.export_frame) == "AUTO" else args.export_frame)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        print(f"[calibrate] {dest} ({dest.stat().st_size / 1e6:.1f} MB)\n"
              f"[calibrate] push it:  uv run -m rtrack.relay push-calib {stem}")
        return 0

    if args.interactive:
        seed = None
        if not args.fresh:
            p = C.CALIB_DIR / f"{stem}.json"
            if p.exists():
                seed = json.loads(p.read_text(encoding="utf-8")).get("points") or None
                if seed:
                    print(f"[calibrate] pre-loaded {len(seed)} existing point(s) -- "
                          f"shown grey. Add to them; 'u' removes the last one.")
        pairs = interactive(stem, args.frame, ref, use_plate=args.plate, seed=seed)
    elif args.points:
        raw = json.loads(args.points.read_text(encoding="utf-8"))
        # Accept either a bare [[vx,vy,fx,fy], ...] or the envelope the remote
        # calibrator posts ({videoId, frame, createdAt, points: [...]}). Keeping the
        # envelope is worth it -- it records WHICH frame the points were clicked on,
        # which is the first thing you want when a calibration looks wrong.
        pairs = raw["points"] if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and raw.get("frame") is not None:
            print(f"[calibrate] points clicked on frame {raw['frame']}"
                  + (f" of {raw['videoId']}" if raw.get("videoId") else ""))
    else:
        ap.error("pass --interactive or --points")

    if len(pairs) < 4:
        raise SystemExit(f"need at least 4 correspondences, got {len(pairs)}")

    arr = np.array(pairs, np.float32)
    src, dst = arr[:, :2], arr[:, 2:]

    frame0 = grab_plate(stem) if args.plate else grab_frame(stem, args.frame)
    warns = geometry_warnings(src, frame0.shape[0])
    for wmsg in warns:
        print(f"[geometry] WARNING: {wmsg}")

    # Solve for lens distortion from these same correspondences, and accept it only
    # if leave-one-out says it helps on points the fit never saw. Adding parameters
    # to a handful of points otherwise buys a smaller residual and a worse model.
    lens_params = None
    if not args.no_lens and len(pairs) >= 8:
        plumb = None if args.no_plumb else trace_plumb(frame0)
        lp = _lens.solve(src.astype(np.float64), dst.astype(np.float64),
                         ref["pxPerMeter"],
                         plumb=[plumb] if plumb is not None else None)
        print(f"[lens] f={lp['f']:.0f}  k1={lp['k1']:+.4f}  k2={lp['k2']:+.4f}  "
              f"principal point ({lp.get('cx', 960):.0f}, {lp.get('cy', 540):.0f}) "
              f"[image centre is (960, 540)]")
        print(f"[lens]   in-sample   plain {lp['errPlainM']['mean']:.3f} m "
              f"-> corrected {lp['errM']['mean']:.3f} m")
        print(f"[lens]   leave-one-out plain {lp['looPlainM']['mean']:.3f} m "
              f"-> corrected {lp['looM']['mean']:.3f} m")
        if lp.get("plumbPx"):
            print(f"[lens]   straight-line check  uncorrected "
                  f"{lp['plumbPxPlain'][0]:.2f} px -> corrected {lp['plumbPx'][0]:.2f} px")
        if lp["generalises"]:
            lens_params = lp
            print("[lens] ACCEPTED -- the gain holds on held-out points")
        else:
            print("[lens] REJECTED -- no out-of-sample gain; keeping plain homography")
    elif not args.no_lens:
        print(f"[lens] skipped: {len(pairs)} points is too few to fit distortion "
              f"(need 8+)")

    H, report = compute(src, dst, ref, args.ransac, lens_params)
    if report["inliers"] is not None and report["inliers"] <= 4:
        print(f"[geometry] WARNING: RANSAC kept only {report['inliers']} points. "
              "Four points fit a homography EXACTLY, so their 0.000 errors mean "
              "nothing -- the other points are being called outliers.")

    for m in point_coverage(src, frame0.shape[1]):
        print(f"[coverage] WARNING: {m}")

    # Render the PREVIEWS from the undistorted frame. H now maps undistorted pixels,
    # so warping the raw frame with it would show a misalignment the pipeline does not
    # actually have -- the check that catches mistakes has to be run on what the
    # pipeline really consumes.
    # Both previews are drawn on the ORIGINAL frame. Rectifying the video to draw on
    # it warps the picture and pushes the near corners off the canvas; bending the
    # OVERLAY to match the lens instead leaves the video exactly as the detector sees
    # it, which is what a diagnostic should show.
    lens_cfg = ({k: lens_params[k] for k in ("f", "k1", "k2", "cx", "cy")
                 if k in lens_params} if lens_params else None)
    field = cv2.imread(str(C.REPO_ROOT / ref["image"]))
    preview(frame0, field, H, C.STAGE2_DIR / f"{stem}_warp.png", lens=lens_cfg)
    rep_corners = reproject_overlay(frame0, H, ref,
                                    C.STAGE2_DIR / f"{stem}_reproject.png",
                                    lens=lens_cfg)
    report.update(rep_corners)
    if rep_corners["cornersOutsideFrame"]:
        print(f"[reproject] corners predicted OUTSIDE the frame: "
              f"{rep_corners['cornersOutsideFrame']}")
    report.update(error_jacobian(H, ref, frame0.shape,
                                 C.STAGE2_DIR / f"{stem}_error_map.png",
                                 lens_params))

    doc = {
        "video": stem,
        "refFrame": args.frame,
        "mode": "static-homography",
        "H_video_to_fieldpx": H.tolist(),
        "fieldRef": f"field_ref_{C.YEAR}.json",
        "points": pairs,
        **report,
    }
    dest = C.CALIB_DIR / f"{stem}.json"
    dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "perPointErrorM"},
                     indent=2))
    print(f"per-point error (m): {report['perPointErrorM']}")
    print(f"\n-> {dest}")
    print(f"-> {C.STAGE2_DIR / f'{stem}_warp.png'}   <-- LOOK AT THIS")
    print(f"-> {C.STAGE2_DIR / f'{stem}_reproject.png'}   <-- AND THIS ONE")
    print(f"-> {C.STAGE2_DIR / f'{stem}_error_map.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
