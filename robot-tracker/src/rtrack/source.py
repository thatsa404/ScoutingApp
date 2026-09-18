"""Frame sources.

THE load-bearing abstraction of this project. Every downstream stage -- detect,
track, project, export -- consumes `FrameSource` and never touches
`cv2.VideoCapture` directly. That is what makes the Stage 4 livestream path a new
source implementation and nothing else.

    for f in VideoFileSource(path, stride=2):
        ...  # f.idx, f.t, f.image
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    idx: int           # index in the ORIGINAL video, not in the strided iteration
    t: float           # seconds from video start
    image: np.ndarray  # BGR


@dataclass
class SourceMeta:
    width: int
    height: int
    fps: float
    frame_count: int | None  # None for live sources
    duration: float | None
    codec: str | None = None
    path: str | None = None
    is_live: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class FrameSource(Protocol):
    meta: SourceMeta

    def __iter__(self) -> Iterator[Frame]: ...


# --------------------------------------------------------------- ffprobe

def ffprobe(path: Path) -> dict:
    """Full container metadata. Raises if ffprobe is missing or the file is bad."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not on PATH -- winget install Gyan.FFmpeg")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt",
            "-show_entries", "format=duration,bit_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _parse_rate(rate: str | None) -> float | None:
    """'30000/1001' -> 29.97."""
    if not rate or "/" not in rate:
        return float(rate) if rate else None
    num, den = rate.split("/")
    return float(num) / float(den) if float(den) else None


def probe_meta(path: Path) -> tuple[SourceMeta, dict]:
    """SourceMeta plus the raw ffprobe dict, for the Stage 0 README table."""
    raw = ffprobe(path)
    st = raw["streams"][0]
    fmt = raw.get("format", {})
    r_rate = _parse_rate(st.get("r_frame_rate"))
    avg_rate = _parse_rate(st.get("avg_frame_rate"))

    meta = SourceMeta(
        width=int(st["width"]),
        height=int(st["height"]),
        fps=r_rate or avg_rate or 30.0,
        frame_count=int(st["nb_frames"]) if st.get("nb_frames") else None,
        duration=float(fmt["duration"]) if fmt.get("duration") else None,
        codec=st.get("codec_name"),
        path=str(path),
    )
    # VFR rips desync the whole t axis. Surface it loudly rather than silently
    # producing timestamps that drift against the match clock.
    if r_rate and avg_rate and abs(r_rate - avg_rate) > 0.01:
        raw["_vfr_warning"] = (
            f"r_frame_rate={r_rate:.4f} != avg_frame_rate={avg_rate:.4f} -- variable "
            f"frame rate. Normalize first: ffmpeg -i in.mp4 -vsync cfr -r {r_rate:.3f} out.mp4"
        )
    return meta, raw


# --------------------------------------------------------------- file source

class VideoFileSource:
    """Sequential reader over a local video file.

    `stride` subsamples (stride=2 on 30 fps -> 15 fps) while `Frame.idx` and
    `Frame.t` stay in ORIGINAL video units, so every stage agrees on what frame
    412 means regardless of the stride it was run at.

    `height` downscales for analysis passes (Stage 0 runs at 480p); detection
    leaves it at None and lets the model handle imgsz.
    """

    def __init__(
        self,
        path: Path | str,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: int | None = None,
        height: int | None = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.stride = max(1, int(stride))
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.height = height

        self.meta, self.probe_raw = probe_meta(self.path)
        self.scale = 1.0
        if height and self.meta.height:
            self.scale = height / self.meta.height

    @property
    def n_frames_expected(self) -> int | None:
        if self.meta.frame_count is None:
            return None
        end = self.end_frame if self.end_frame is not None else self.meta.frame_count
        return max(0, (end - self.start_frame + self.stride - 1) // self.stride)

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {self.path}")
        try:
            if self.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            idx = self.start_frame
            fps = self.meta.fps
            while True:
                if self.end_frame is not None and idx >= self.end_frame:
                    break
                ok, img = cap.read()
                if not ok:
                    break
                # Decode every frame but only yield every stride-th. grab()-only
                # skipping is faster but desyncs on some broadcast H.264 rips.
                if (idx - self.start_frame) % self.stride == 0:
                    if self.scale != 1.0:
                        img = cv2.resize(
                            img, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    yield Frame(idx=idx, t=idx / fps, image=img)
                idx += 1
        finally:
            cap.release()


# --------------------------------------------------------------- live source

class LivePipeSource:
    """Stage 4. streamlink -> ffmpeg -> raw BGR frames on a pipe.

    Deliberately unimplemented until Stage 3 passes. The signature is fixed here
    so the rest of the pipeline can be written against it today.

    Two things this must do when built, both non-obvious:
      - DROP frames under backpressure, never queue. A growing queue turns a 30 s
        stream lag into a 5 min lag over the course of a match.
      - Carry wall-clock timestamps, not frame counts, since the pipe can stall.
    """

    def __init__(self, url: str, quality: str = "720p", fps: int = 15):
        self.url = url
        self.quality = quality
        self.fps = fps
        raise NotImplementedError(
            "LivePipeSource is Stage 4 and gated behind Stage 3. See the plan's "
            "latency budget first: YouTube live delivery alone is 15-60 s."
        )

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover
        raise NotImplementedError
