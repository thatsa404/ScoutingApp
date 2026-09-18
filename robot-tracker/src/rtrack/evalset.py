"""Stage 1 eval set -- sample frames, extract them, and score a model against them.

The eval set exists so candidate detectors can be COMPARED rather than eyeballed.
It is built BEFORE any model is tried, on purpose: the Universe datasets are 2024
Crescendo footage being asked to transfer to 2026 Rebuilt, and that is not something
you can judge from an overlay.

    uv run -m rtrack.evalset sample GSxbsE42o5o
    uv run -m rtrack.evalset export GSxbsE42o5o --prelabel

Sampling is stratified into uniform coverage plus hard-mined frames. "Hard" is
scored with the colour pre-labeller (rtrack.prelabel): frames where it finds fewer
than six bumpers, or finds them crowded together, or finds them small. That is a
crude proxy for occlusion, scrums and distance -- which is exactly what we want
over-represented, since risk R4 (ID continuity through occlusion) is the live one.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import config as C
from . import prelabel
from .acquire import raw_path, video_id
from .source import VideoFileSource

N_UNIFORM = 56
N_HARD = 24
SCAN_STRIDE = 5       # frames between hardness probes
MIN_SEPARATION = 25   # frames -- keeps hard picks from being near-duplicates


def index_path() -> Path:
    return C.EVAL_DIR / "frames.index.json"


def match_shot(stem: str, shot_id: int | None) -> tuple[int, int]:
    """(start_frame, end_frame) of the shot to sample from.

    Defaults to the longest static shot, which Stage 0 established is the whole
    match for this footage.
    """
    p = C.STAGE0_DIR / f"{stem}_shots.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found -- run: uv run -m rtrack.shots {stem}")
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if shot_id is not None:
        r = next(r for r in rows if int(r["shotId"]) == shot_id)
    else:
        statics = [r for r in rows if r["motionClass"] == "static"]
        r = max(statics, key=lambda r: float(r["durationSec"]))
    return int(r["startFrame"]), int(r["endFrame"])


def hardness(boxes) -> float:
    """0 = easy, 1 = hard. Occlusion weighted highest; it is the live risk."""
    n = len(boxes)
    occl = min(max(6 - n, 0), 6) / 6.0

    if n >= 2:
        cx = sorted(((b[0] + b[2]) / 2) for b in boxes)
        gap = min(b - a for a, b in zip(cx, cx[1:]))
        crowd = 1.0 - min(gap, 300.0) / 300.0
    else:
        crowd = 0.0

    small = 1.0 - min(min((b[2] - b[0]) for b in boxes), 120.0) / 120.0 if n else 0.0
    return 0.5 * occl + 0.3 * crowd + 0.2 * small


def sample(stem: str, shot_id: int | None, n_uniform: int, n_hard: int) -> dict:
    path = raw_path(stem)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run: uv run -m rtrack.acquire {stem}")

    f0, f1 = match_shot(stem, shot_id)
    print(f"[evalset] sampling shot frames {f0}-{f1} ({f1 - f0 + 1} frames)")

    uniform = [int(round(x)) for x in np.linspace(f0, f1, n_uniform)]

    src = VideoFileSource(path, stride=SCAN_STRIDE, start_frame=f0, end_frame=f1 + 1)
    scored: list[tuple[float, int, int]] = []
    for fr in tqdm(src, total=src.n_frames_expected, desc="hard-mine", unit="f"):
        boxes = prelabel.detect(fr.image)
        scored.append((hardness(boxes), fr.idx, len(boxes)))

    scored.sort(key=lambda t: -t[0])
    chosen = list(uniform)
    hard: list[int] = []
    for score, idx, _ in scored:
        if len(hard) >= n_hard:
            break
        if all(abs(idx - c) >= MIN_SEPARATION for c in chosen):
            hard.append(idx)
            chosen.append(idx)

    by_frame = {idx: (s, n) for s, idx, n in scored}
    doc = {
        "video": stem,
        "shot": {"startFrame": f0, "endFrame": f1},
        "counts": {"uniform": len(uniform), "hard": len(hard), "total": len(chosen)},
        "note": (
            "All frames come from the single static match shot. This clip's other "
            "shots are a title card, a crowd reaction, a different-camera driver "
            "station view and a fade -- none show the playing field, so the plan's "
            "'20 from secondary shots' does not apply here."
        ),
        "uniform": sorted(uniform),
        "hard": sorted(hard),
        "hardStats": {
            str(i): {"score": round(by_frame[i][0], 3), "prelabelBoxes": by_frame[i][1]}
            for i in sorted(hard) if i in by_frame
        },
        "frames": sorted(set(chosen)),
    }
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    index_path().write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[evalset] {doc['counts']} -> {index_path()}")
    return doc


def export(stem: str, do_prelabel: bool) -> int:
    doc = json.loads(index_path().read_text(encoding="utf-8"))
    if doc["video"] != stem:
        raise SystemExit(f"index is for {doc['video']}, not {stem}")

    path = raw_path(stem)
    C.EVAL_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(doc["frames"])

    cap = cv2.VideoCapture(str(path))
    written = 0
    # Sequential decode rather than per-frame seek: seeking 80 times into a long
    # AV1 stream is slower and, on some builds, off by a frame.
    idx = 0
    with tqdm(total=max(wanted) + 1, desc="export", unit="f") as bar:
        while idx <= max(wanted):
            ok, img = cap.read()
            if not ok:
                break
            if idx in wanted:
                dest = C.EVAL_FRAMES_DIR / f"{stem}_f{idx:06d}.png"
                cv2.imwrite(str(dest), img)
                if do_prelabel:
                    boxes = prelabel.detect(img)
                    h, w = img.shape[:2]
                    dest.with_suffix(".txt").write_text(
                        prelabel.to_yolo(boxes, w, h), encoding="utf-8")
                written += 1
            idx += 1
            bar.update(1)
    cap.release()

    print(f"[evalset] wrote {written} frames to {C.EVAL_FRAMES_DIR}")
    if do_prelabel:
        print("[evalset] draft .txt labels alongside each frame (YOLO format).")
        print("          These are a colour-threshold AID: expect ~3 false positives")
        print("          per frame and some misses. Correct them, do not trust them.")
    (C.EVAL_DIR / "classes.txt").write_text(
        "\n".join(C.CLASS_NAMES) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Stage 1 eval set.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="choose frames and write the index")
    s.add_argument("video")
    s.add_argument("--shot", type=int, default=None)
    s.add_argument("--uniform", type=int, default=N_UNIFORM)
    s.add_argument("--hard", type=int, default=N_HARD)

    e = sub.add_parser("export", help="extract the chosen frames as PNGs")
    e.add_argument("video")
    e.add_argument("--prelabel", action="store_true",
                   help="also write draft YOLO labels for hand correction")

    args = ap.parse_args(argv)
    C.ensure_dirs()
    stem = video_id(args.video)

    if args.cmd == "sample":
        sample(stem, args.shot, args.uniform, args.hard)
    else:
        export(stem, args.prelabel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
