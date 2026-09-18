"""Stage 1 -- detection + tracking over a video, written to tracks.jsonl.

This is the module the proof gate runs on. Its output is the single artefact every
later stage reads: Stage 2 (projection) and Stage 3 (identity) never re-run the
detector, so calibration and identity work iterate in seconds rather than minutes.

    uv run -m rtrack.track GSxbsE42o5o --model runs/detect/runs/robot_s/weights/best.pt
    uv run -m rtrack.track GSxbsE42o5o --model ... --tracker cfg/tracktrack_frc.yaml

One line of JSON per processed frame:
    {"f": 412, "t": 13.75, "dets": [{"tid": 3, "xyxy": [...], "conf": 0.91,
                                     "alliance": "red", "aconf": 0.83}]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from . import config as C
from . import alliance as al
from .acquire import raw_path, video_id


def _square_to_orig(xyxy: list[float], w: int, h: int, n: int) -> list[float]:
    """Map a box from square (stretched) inference space back to source pixels."""
    sx, sy = w / n, h / n
    return [xyxy[0] * sx, xyxy[1] * sy, xyxy[2] * sx, xyxy[3] * sy]


def default_model() -> Path:
    """Most recent best.pt under runs/, so the common case needs no --model."""
    cands = sorted(Path("runs").rglob("weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit("no trained weights found under runs/ -- pass --model")
    return cands[0]


def run(stem: str, model_path: Path, tracker: Path, start: int, end: int | None,
        stride: int | None, imgsz: int, conf: float, out: Path | None,
        three_three: bool, square: bool, lost_decay: float | None,
        nms_iou: float, sample_hz: float = 15.0, half: bool = True) -> Path:
    """Detect and track, sampling at a fixed rate in TIME rather than in frames.

    SAMPLE RATE IS IN HERTZ, NOT FRAMES, and that is load-bearing. A fixed --stride
    assumes the source is ~30 fps. A 58 fps broadcast (measured: the 2026mawor final)
    then runs at double the intended rate, and every frame-counted parameter
    downstream silently halves in real terms:

        cfg/botsort_frc.yaml  track_buffer: 60   4 s at 15 Hz -> 2 s at 30 Hz
        appear.STRIDE                            every 2 detections, so half the span
        robots.split_on_appearance win=12        a 12-DETECTION change-point window

    None of those announce themselves; the pipeline just gets quietly worse. Pinning
    the processed rate in Hz makes all of them correct again by construction, because
    they are all counted in PROCESSED frames and the processed rate is now constant.

    15 Hz is the default because that is what the tracker was tuned at, not because
    it is a universal optimum. Going much coarser breaks association rather than just
    costing resolution: an FRC robot tops out near 5 m/s and is about 0.75 m wide, so
    at 4 Hz it travels ~1.25 m between samples -- well over one box width -- and
    IoU-based matching sees no overlap at all. Roughly 10 Hz is the floor before that
    starts to bite. Output can be decimated afterwards as much as you like; the
    tracker cannot.
    """
    from ultralytics import YOLO

    if lost_decay is not None:
        from . import botpatch
        botpatch.apply(lost_decay)

    video = raw_path(stem)
    if not video.exists():
        raise SystemExit(f"{video} not found -- run: uv run -m rtrack.acquire {stem}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end = end if end is not None else total

    if stride is None:
        stride = max(1, int(round(fps / max(sample_hz, 0.1))))
        eff = fps / stride
        print(f"[track] source {fps:.2f} fps -> stride {stride} "
              f"= {eff:.1f} Hz processed (asked {sample_hz:.1f} Hz)")
        if eff < 10.0:
            print(f"[track] WARNING: {eff:.1f} Hz is below the ~10 Hz floor where "
                  f"IoU association starts to fail on fast robots")
    else:
        print(f"[track] stride {stride} forced -> {fps/stride:.1f} Hz processed "
              f"(source {fps:.2f} fps)")

    model = YOLO(str(model_path))
    out = out or (C.STAGE1_DIR / f"{stem}_tracks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[track] {video.name} frames {start}-{end} stride {stride} "
          f"({(end-start)//stride} to process)")
    print(f"[track] model {model_path}")
    print(f"[track] tracker {tracker}")

    # NOTE: ultralytics' own vid_stride would renumber frames, and we need indices in
    # ORIGINAL video coordinates so Stage 2/3 and the viewer all agree on frame 412.
    # So we iterate ourselves and feed frames one at a time with persist=True.
    cap = cv2.VideoCapture(str(video))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    n_frames = n_dets = n_undecided = 0
    seen_tids: set[int] = set()
    t0 = time.perf_counter()

    with out.open("w", encoding="utf-8") as fh:
        idx = start
        while idx < end:
            # DECODE ONLY WHAT WE PROCESS. At 15 Hz from a 30 fps source the stride is
            # 2, so half of every decode used to be colour-converted into a numpy
            # array and thrown away one line later. grab() demuxes and decodes without
            # that conversion, which is the expensive half. Decode is 14% of the loop
            # (1.67 s of 11.9 s, measured), so this is worth ~5-7% -- small, but it is
            # the whole of what is available on the decode side, and the profile is
            # why the ffmpeg/NVDEC rewrite is NOT here: it would be chasing the same
            # 14% at much greater cost. The 85% is inference, and `half` above is
            # where that was taken from.
            if (idx - start) % stride:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, img = cap.read()
            if not ok:
                break

            # The 2026 dataset was preprocessed "Resize to 640x640 (Stretch)", which
            # squeezes a 16:9 source horizontally by ~1.78x. Ultralytics letterboxes
            # instead, so feeding a 16:9 frame straight in presents the model with
            # proportions it never saw. --square reproduces the training distortion
            # and maps the boxes back afterwards.
            src_h, src_w = img.shape[:2]
            inp = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR) \
                if square else img

            # FP16. Measured on 2026necmp1_qm1 (300 frames, RTX 3060, imgsz 1280):
            # 57.7 fps at fp32 against 81.2 fps at fp16 -- 1.41x -- for 5.98 vs 6.00
            # detections per frame. The speed is bought from the tensor cores, not
            # from the detector: there are six robots on the field and both settings
            # find six. Anything that DID cost detections was rejected; shrinking
            # imgsz to 960 runs at 113 fps and finds 4.70/frame, and those missing
            # detections are identity evidence the whole pipeline rests on.
            # `quantize=16`, not `half=True`: ultralytics 8.4 renamed the flag and
            # the old one still works but prints a deprecation warning on EVERY call,
            # which buries the per-100-frame progress line this loop prints.
            res = model.track(inp, persist=True, tracker=str(tracker), conf=conf,
                              iou=nms_iou, imgsz=imgsz, device=0,
                              quantize=(16 if half else None), verbose=False)[0]

            # ONE device-to-host transfer per FRAME, not three per BOX. Iterating
            # `res.boxes` and reading `b.xyxy[0].tolist()`, `b.id.item()` and
            # `float(b.conf)` is three CUDA synchronisations per robot, ~18 per frame,
            # each of which stalls the pipeline until the GPU drains. That cost is
            # invisible in an inference benchmark -- which never touches the boxes --
            # and it is why FP16 alone moved the full loop by 5.7% while moving
            # inference by 43%: the time it freed was being spent here instead.
            boxes, dets = [], []
            bx = res.boxes
            if bx is not None and len(bx):
                a_xyxy = bx.xyxy.cpu().numpy()
                a_conf = bx.conf.cpu().numpy()
                a_id = bx.id.cpu().numpy() if bx.id is not None else None
                for i in range(len(a_xyxy)):
                    xyxy = [float(v) for v in a_xyxy[i]]
                    if square:
                        xyxy = _square_to_orig(xyxy, src_w, src_h, imgsz)
                    xyxy = [round(v, 1) for v in xyxy]
                    tid = int(a_id[i]) if a_id is not None else -1
                    boxes.append((*xyxy, tid))
                    dets.append({"tid": tid, "xyxy": xyxy,
                                 "conf": round(float(a_conf[i]), 3)})

            calls = al.classify_all(img, boxes)
            if three_three:
                calls = al.enforce_three_three(calls)
            for d, c in zip(dets, calls):
                d["alliance"] = c.alliance
                d["aconf"] = round(c.confidence, 3)
                if c.alliance is None:
                    n_undecided += 1
                if d["tid"] >= 0:
                    seen_tids.add(d["tid"])

            fh.write(json.dumps({"f": idx, "t": round(idx / fps, 4), "dets": dets}) + "\n")
            n_frames += 1
            n_dets += len(dets)
            if n_frames % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {n_frames} frames  {n_dets/max(n_frames,1):.1f} det/frame  "
                      f"{n_frames/el:.1f} fps", flush=True)
            idx += 1
    cap.release()

    el = time.perf_counter() - t0
    print(f"\n[track] {n_frames} frames in {el:.1f}s ({n_frames/max(el,1e-9):.1f} fps)")
    print(f"[track] {n_dets} detections, {n_dets/max(n_frames,1):.2f}/frame "
          f"(6 is the target)")
    print(f"[track] {len(seen_tids)} distinct track ids "
          f"(6 would be perfect; more means ID switches)")
    print(f"[track] alliance undecided on {n_undecided}/{n_dets} detections "
          f"({100*n_undecided/max(n_dets,1):.1f}%)")
    print(f"[track] -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: detect + track -> jsonl.")
    ap.add_argument("video")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--tracker", type=Path, default=C.CFG_DIR / "botsort_frc.yaml")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--sample-hz", type=float, default=15.0,
                    help="PROCESS this many frames per second of video, whatever the "
                         "source frame rate. Default 15 Hz, which is what the tracker "
                         "and every frame-counted parameter downstream were tuned at. "
                         "See the note in run() on why this is not --stride.")
    ap.add_argument("--stride", type=int, default=None,
                    help="override --sample-hz with a fixed frame stride. Only for "
                         "reproducing an old run; it silently changes the effective "
                         "sample rate when the source fps is not ~30.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-3v3", action="store_true",
                    help="disable the 3-red/3-blue consistency nudge")
    ap.add_argument("--square", action="store_true",
                    help="stretch frames to imgsz x imgsz to match a dataset "
                         "preprocessed with Roboflow 'Resize (Stretch)'")
    ap.add_argument("--nms-iou", type=float, default=0.55,
                    help="NMS IoU. Ultralytics defaults to 0.70, which leaves "
                         "duplicate boxes stacked on one robot in 6.9%% of frames "
                         "and inflates per-alliance counts past 3.")
    ap.add_argument("--fp32", action="store_true",
                    help="run the detector at full precision. FP16 is the default "
                         "and costs nothing measurable (6.00 vs 5.98 det/frame); "
                         "this exists to reproduce an older run or to rule half "
                         "precision out when a result looks wrong.")
    ap.add_argument("--lost-decay", type=float, default=None, metavar="GAMMA",
                    help="decay Kalman velocity by GAMMA per frame while a track is "
                         "lost, so the prediction converges to stationary instead of "
                         "coasting off the robot (see rtrack.botpatch). Try 0.8.")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    run(video_id(args.video), args.model or default_model(), args.tracker,
        args.start, args.end, args.stride, args.imgsz, args.conf, args.out,
        three_three=not args.no_3v3, square=args.square,
        lost_decay=args.lost_decay, nms_iou=args.nms_iou,
        sample_hz=args.sample_hz, half=not args.fp32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
