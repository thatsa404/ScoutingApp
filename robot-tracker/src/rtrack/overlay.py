"""Stage 1 -- render tracks.jsonl over the video. THE PROOF-GATE ARTEFACT.

    uv run -m rtrack.overlay GSxbsE42o5o

Writes:
    out/stage1/<id>_overlay.mp4   boxes, alliance colours, stable ids, motion trails
    out/stage1/<id>_counts.png    n_red / n_blue against time

The counts plot is the fastest read on detector health you will have: you want two
flat lines at 3. Every dip is a miss, every spike a phantom.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id

TRAIL = 45  # frames of history drawn behind each robot (3 s at 15 fps)
GREY = (150, 150, 150)


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def colour(alliance: str | None):
    return {"red": C.COLOR_RED_BGR, "blue": C.COLOR_BLUE_BGR}.get(alliance, GREY)


def render(stem: str, tracks_p: Path, out_mp4: Path, plot_p: Path,
           show_conf: bool) -> None:
    rows = load(tracks_p)
    if not rows:
        raise SystemExit(f"{tracks_p} is empty")
    by_frame = {r["f"]: r for r in rows}
    frames = sorted(by_frame)
    f0, f1 = frames[0], frames[-1]

    video = raw_path(stem)
    cap = cv2.VideoCapture(str(video))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, (frames[1] - frames[0]) if len(frames) > 1 else 1)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps_src / step, (w, h))

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL))
    tid_colour: dict[int, tuple] = {}
    switches = 0
    seen: set[int] = set()
    series_t, series_r, series_b = [], [], []

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    idx = f0
    while idx <= f1:
        ok, img = cap.read()
        if not ok:
            break
        row = by_frame.get(idx)
        if row is None:
            idx += 1
            continue

        n_r = n_b = 0
        for d in row["dets"]:
            x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
            tid = d["tid"]
            col = colour(d.get("alliance"))
            if d.get("alliance") == "red":
                n_r += 1
            elif d.get("alliance") == "blue":
                n_b += 1

            # A track id appearing for the first time after the opening frames is,
            # in a six-robot match, almost always a switch rather than a new robot.
            if tid >= 0:
                if tid not in seen:
                    seen.add(tid)
                    if idx > f0 + 10 * step:
                        switches += 1
                tid_colour[tid] = col
                trails[tid].append(((x1 + x2) // 2, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            team = d.get("team")
            label = (str(team) if team else (f"#{tid}" if tid >= 0 else "#?"))
            if show_conf:
                label += f" {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 1, cv2.LINE_AA)

        # Trails last, so they sit under nothing and read as motion at a glance.
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            p = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [p], False, tid_colour.get(tid, GREY), 2, cv2.LINE_AA)

        t_rel = (idx - f0) / fps_src
        series_t.append(t_rel)
        series_r.append(n_r)
        series_b.append(n_b)

        named = len({d.get("team") for d in row["dets"] if d.get("team")})
        hud = (f"f{idx}  t{t_rel:6.1f}s   red {n_r}  blue {n_b}   "
               f"tracks {len(seen)}  teams-shown {named}")
        cv2.rectangle(img, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(img, hud, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (235, 235, 235), 1, cv2.LINE_AA)

        vw.write(img)
        idx += 1

    cap.release()
    vw.release()

    _plot(series_t, series_r, series_b, plot_p)
    print(f"[overlay] {out_mp4}")
    print(f"[overlay] {plot_p}")
    print(f"[overlay] {len(seen)} distinct track ids; "
          f"{switches} appeared after the opening (likely ID switches)")
    if series_r:
        print(f"[overlay] mean visible: red {np.mean(series_r):.2f}, "
              f"blue {np.mean(series_b):.2f} (3.00 each is perfect)")
        both3 = sum(1 for r, b in zip(series_r, series_b) if r == 3 and b == 3)
        print(f"[overlay] frames with exactly 3+3: {both3}/{len(series_r)} "
              f"({100*both3/len(series_r):.1f}%)")


def _plot(t, r, b, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=130)
    ax.plot(t, r, lw=1.0, color="#ef4444", label="red")
    ax.plot(t, b, lw=1.0, color="#3b82f6", label="blue")
    ax.axhline(3, color="#888", lw=0.8, ls="--", label="expected (3)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("robots detected")
    ax.set_ylim(-0.2, max(7, (max(r + b) if r + b else 6) + 0.5))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Detections per alliance over time -- you want two flat lines at 3")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the Stage 1 overlay video.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-conf", action="store_true")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    tracks = args.tracks or (C.STAGE1_DIR / f"{stem}_tracks.jsonl")
    out = args.out or (C.STAGE1_DIR / f"{stem}_overlay.mp4")
    render(stem, tracks, out, C.STAGE1_DIR / f"{stem}_counts.png",
           show_conf=not args.no_conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
