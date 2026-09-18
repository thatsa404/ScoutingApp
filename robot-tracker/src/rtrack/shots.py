"""Stage 0b -- measure the broadcast before writing any tracking code.

A cutting or panning broadcast breaks homography AND tracking continuity at the
same time. This is the cheapest fatal test in the project (risks R1 and R5), so it
runs before anything else.

    uv run -m rtrack.shots GSxbsE42o5o

Outputs to out/stage0/:
    <id>_motion.csv        per-frame histogram correlation, MAD, translation, scale
    <id>_shots.csv         one row per shot: bounds, duration, motion class
    <id>_summary.json      the numbers that go in the README
    <id>_contact_sheet.png one representative frame per shot -- LOOK AT THIS
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import config as C
from .acquire import raw_path, video_id
from .source import VideoFileSource

# Contact-sheet tiles. Small on purpose: the sheet answers "what kind of shot is
# this", and a 6-across grid of 360px tiles reads fine at a glance.
TILE_W = 360
SHEET_COLS = 6


@dataclass
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int          # inclusive
    start_t: float
    end_t: float
    trans: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    rep_frame: np.ndarray | None = None
    rep_idx: int = 0

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def motion(self) -> tuple[str, float, float]:
        """(class, median translation px, median scale) over the shot."""
        if not self.trans:
            return "unknown", 0.0, 1.0
        mt = float(np.median(self.trans))
        ms = float(np.median(self.scales))
        if abs(ms - 1.0) > C.STATIC_SCALE_EPS * 3:
            return "zoom", mt, ms
        if mt > C.STATIC_TRANS_PX:
            return "pan", mt, ms
        return "static", mt, ms


def _hsv_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _thumb(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def _camera_motion(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float, int]:
    """Similarity-transform estimate between consecutive frames.

    Returns (translation px, scale, inlier count). Scale ~1.0 and translation ~0
    means a locked-off camera.
    """
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if pts is None or len(pts) < 12:
        return 0.0, 1.0, 0

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None:
        return 0.0, 1.0, 0
    ok = status.ravel() == 1
    src, dst = pts[ok], nxt[ok]
    if len(src) < 12:
        return 0.0, 1.0, 0

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if M is None:
        return 0.0, 1.0, 0

    a, b = M[0, 0], M[1, 0]
    scale = float(np.hypot(a, b))
    trans = float(np.hypot(M[0, 2], M[1, 2]))
    return trans, scale, int(inliers.sum()) if inliers is not None else 0


def analyze(path: Path, correl_cut: float, mad_cut: float) -> tuple[list[dict], list[Shot], dict]:
    src = VideoFileSource(path, height=C.ANALYSIS_HEIGHT)
    total = src.n_frames_expected
    scale_back = 1.0 / src.scale if src.scale else 1.0

    rows: list[dict] = []
    shots: list[Shot] = []
    cur: Shot | None = None

    prev_hist = prev_thumb = prev_gray = None

    for f in tqdm(src, total=total, desc="stage0", unit="f"):
        gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        hist, thumb = _hsv_hist(f.image), _thumb(gray)

        if prev_hist is None:
            correl, mad, is_cut = 1.0, 0.0, True  # first frame opens shot 0
            trans = 0.0
            sc = 1.0
            inl = 0
        else:
            correl = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            mad = float(np.mean(np.abs(thumb - prev_thumb)))
            # Both signals must fire. A flashbulb or a score-bug animation moves
            # the histogram hard but leaves the structure -- and therefore the
            # thumbnail MAD -- largely intact.
            is_cut = (correl < correl_cut) and (mad > mad_cut)
            trans, sc, inl = _camera_motion(prev_gray, gray)
            if is_cut:
                trans, sc, inl = 0.0, 1.0, 0  # flow across a cut is meaningless

        rows.append({
            "frame": f.idx, "t": round(f.t, 4),
            "histCorrel": round(correl, 4), "thumbMad": round(mad, 3),
            "transPx": round(trans, 3), "scale": round(sc, 5),
            "flowInliers": inl, "isCut": int(is_cut),
        })

        if is_cut:
            if cur is not None:
                cur.end_frame, cur.end_t = f.idx - 1, f.t
                shots.append(cur)
            cur = Shot(len(shots), f.idx, f.idx, f.t, f.t)
            cur.rep_idx = f.idx
        else:
            assert cur is not None
            cur.trans.append(trans)
            cur.scales.append(sc)

        if cur is not None:
            cur.end_frame, cur.end_t = f.idx, f.t
            # Representative frame = the shot's midpoint-so-far. Cheaper than a
            # second pass and avoids the transition frames at either edge.
            if cur.rep_frame is None or f.idx - cur.start_frame == cur.n_frames // 2:
                h = int(f.image.shape[0] * TILE_W / f.image.shape[1])
                cur.rep_frame = cv2.resize(f.image, (TILE_W, h), interpolation=cv2.INTER_AREA)
                cur.rep_idx = f.idx

        prev_hist, prev_thumb, prev_gray = hist, thumb, gray

    if cur is not None:
        shots.append(cur)

    summary = _summarize(src, shots, scale_back)
    return rows, shots, summary


def _summarize(src: VideoFileSource, shots: list[Shot], scale_back: float) -> dict:
    total_dur = sum(s.duration for s in shots) or 1e-9
    by_class: dict[str, float] = {}
    for s in shots:
        cls, _, _ = s.motion()
        by_class[cls] = by_class.get(cls, 0.0) + s.duration

    statics = [s for s in shots if s.motion()[0] == "static"]
    longest = max(statics, key=lambda s: s.duration) if statics else None

    return {
        "source": src.meta.to_dict(),
        "analysisHeight": C.ANALYSIS_HEIGHT,
        "motionScaleNote": (
            f"translation is px at {C.ANALYSIS_HEIGHT}p; multiply by "
            f"{scale_back:.2f} for source-resolution px"
        ),
        "shotCount": len(shots),
        "totalDurationSec": round(total_dur, 2),
        "durationByMotionClass": {k: round(v, 2) for k, v in sorted(by_class.items())},
        "pctTimeByMotionClass": {
            k: round(100 * v / total_dur, 1) for k, v in sorted(by_class.items())
        },
        "longestStaticShot": None if longest is None else {
            "shotId": longest.shot_id,
            "startSec": round(longest.start_t, 2),
            "endSec": round(longest.end_t, 2),
            "durationSec": round(longest.duration, 2),
            "pctOfVideo": round(100 * longest.duration / total_dur, 1),
        },
        "shotsOver20s": sum(1 for s in shots if s.duration > 20),
        "medianShotSec": round(float(np.median([s.duration for s in shots])), 2) if shots else 0,
    }


# 15 s auto + 135 s teleop. A static shot at least this long can hold a whole match.
MATCH_SECONDS = 150.0


def verdict(summary: dict) -> str:
    """Map the numbers onto the plan's Stage 0 decision tree.

    Measured against MATCH_SECONDS rather than against percent-of-video: a clip
    padded with a title card and a crowd reaction shot can be 100% usable for
    tracking while the longest shot is only ~80% of the runtime. What actually
    matters is whether one locked-off shot contains the whole match.
    """
    longest = summary.get("longestStaticShot")
    pct_static = summary["pctTimeByMotionClass"].get("static", 0.0)
    pct_moving = (summary["pctTimeByMotionClass"].get("pan", 0.0)
                  + summary["pctTimeByMotionClass"].get("zoom", 0.0))

    if longest and longest["durationSec"] >= MATCH_SECONDS:
        return (f"BEST CASE -- a single static shot runs {longest['durationSec']:.0f}s "
                f"({longest['startSec']:.0f}-{longest['endSec']:.0f}s), long enough to hold a "
                f"whole {MATCH_SECONDS:.0f}s match with no cuts. One homography, no "
                "re-association. Proceed as planned; Stage 2 ~1 day.")
    if pct_static >= 50:
        return (f"EXPECTED CASE -- {pct_static}% static, but the longest single static shot is "
                f"only {(longest or {}).get('durationSec', 0):.0f}s, short of a "
                f"{MATCH_SECONDS:.0f}s match. Process wide shots only, one homography per camera "
                "setup, re-associate identity across the gaps. +0.5 day.")
    if pct_moving >= 50:
        return (f"BAD CASE -- {pct_moving}% of time is pan/zoom. Stage 2 needs per-frame "
                "registration and BoT-SORT GMC becomes mandatory in Stage 1. +2-3 days.")
    return ("KILL CASE for this footage -- no sustained wide shot. The concept still "
            "works but needs your own fixed camera in the stands. Re-scope before Stage 1.")


def contact_sheet(shots: list[Shot], dest: Path) -> None:
    tiles = [s for s in shots if s.rep_frame is not None]
    if not tiles:
        return
    th, tw = tiles[0].rep_frame.shape[:2]
    label_h = 34
    cols = min(SHEET_COLS, len(tiles))
    rows_n = (len(tiles) + cols - 1) // cols

    sheet = np.full((rows_n * (th + label_h), cols * tw, 3), 24, np.uint8)
    for i, s in enumerate(tiles):
        r, c = divmod(i, cols)
        y0, x0 = r * (th + label_h), c * tw
        sheet[y0 + label_h:y0 + label_h + th, x0:x0 + tw] = s.rep_frame

        cls, mt, ms = s.motion()
        color = {"static": (120, 230, 120), "pan": (90, 190, 255),
                 "zoom": (120, 120, 255)}.get(cls, (200, 200, 200))
        cv2.putText(sheet, f"#{s.shot_id}  {s.start_t:6.1f}-{s.end_t:6.1f}s  ({s.duration:.1f}s)",
                    (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{cls}  trans={mt:.2f}px  scale={ms:.4f}",
                    (x0 + 6, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + tw - 1, y0 + label_h + th - 1), (60, 60, 60), 1)

    cv2.imwrite(str(dest), sheet)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: shot boundaries + camera motion.")
    ap.add_argument("video", help="YouTube id/URL (uses data/raw/<id>.mp4) or a file path")
    ap.add_argument("--correl-cut", type=float, default=C.HIST_CORREL_CUT)
    ap.add_argument("--mad-cut", type=float, default=C.THUMB_MAD_CUT)
    args = ap.parse_args(argv)

    C.ensure_dirs()
    p = Path(args.video)
    if p.exists():
        path, stem = p, p.stem
    else:
        stem = video_id(args.video)
        path = raw_path(stem)
        if not path.exists():
            raise SystemExit(f"{path} not found -- run: uv run -m rtrack.acquire {stem}")

    rows, shots, summary = analyze(path, args.correl_cut, args.mad_cut)
    summary["verdict"] = verdict(summary)

    _write_csv(C.STAGE0_DIR / f"{stem}_motion.csv", rows)
    _write_csv(C.STAGE0_DIR / f"{stem}_shots.csv", [
        {
            "shotId": s.shot_id, "startFrame": s.start_frame, "endFrame": s.end_frame,
            "startSec": round(s.start_t, 3), "endSec": round(s.end_t, 3),
            "durationSec": round(s.duration, 3), "nFrames": s.n_frames,
            "motionClass": s.motion()[0],
            "medianTransPx": round(s.motion()[1], 3),
            "medianScale": round(s.motion()[2], 5),
            "repFrame": s.rep_idx,
        }
        for s in shots
    ])
    (C.STAGE0_DIR / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    contact_sheet(shots, C.STAGE0_DIR / f"{stem}_contact_sheet.png")

    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print("VERDICT:", summary["verdict"])
    print("=" * 78)
    print(f"\nNow LOOK AT: {C.STAGE0_DIR / f'{stem}_contact_sheet.png'}")
    print("Five minutes of eyes on that sheet beats any algorithm at this stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
