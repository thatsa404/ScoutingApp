"""Stage 0a -- pull a match video and record what it actually is.

    uv run -m rtrack.acquire "https://www.youtube.com/watch?v=GSxbsE42o5o"
    uv run -m rtrack.acquire GSxbsE42o5o --list-formats
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from . import config as C
from .source import probe_meta

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def video_id(url_or_id: str) -> str:
    """A YouTube id, a YouTube URL, or the stem of a local clip already in data/raw.

    The local-stem case exists because rtrack.replay slices matches out of a full-event
    archive and names them for the MATCH ('2026mawor_qm1'), not for the YouTube video
    they came from -- one archive yields dozens of clips, so the video id stops being a
    useful identifier at that point. Every stage keys its outputs off this, so it has to
    accept those names or the whole Stage 4 path is unreachable.

    Still strict about what it invents: a non-YouTube name is only accepted when the
    file actually exists, so a typo fails here rather than several minutes later.
    """
    if _YT_ID.match(url_or_id):
        return url_or_id
    for pat in (r"[?&]v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})",
                r"/live/([A-Za-z0-9_-]{11})"):
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    stem = Path(url_or_id).stem if url_or_id.endswith(".mp4") else url_or_id
    if re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", stem or "") and raw_path(stem).exists():
        return stem
    raise ValueError(
        f"{url_or_id!r} is not a YouTube id/URL, and {raw_path(stem or url_or_id)} "
        f"does not exist")


def watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def raw_path(vid: str) -> Path:
    return C.RAW_DIR / f"{vid}.mp4"


def list_formats(vid: str) -> None:
    subprocess.run([sys.executable, "-m", "yt_dlp", "-F", watch_url(vid)], check=True)


def download(vid: str, max_height: int | None = None, force: bool = False) -> Path:
    """Fetch at the best available quality.

    No max_height by default, and that is deliberate: far-field robots are the
    binding constraint on this whole project (risk R2) and you cannot get pixels
    back later. Downscaling happens at inference time via imgsz, not here.
    """
    C.ensure_dirs()
    out = raw_path(vid)
    if out.exists() and not force:
        print(f"[acquire] already have {out}")
        return out

    height_filter = f"[height<={max_height}]" if max_height else ""
    fmt = (
        f"bv*{height_filter}[ext=mp4]+ba[ext=m4a]/"
        f"bv*{height_filter}+ba/"
        f"b{height_filter}"
    )
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", str(C.RAW_DIR / "%(id)s.%(ext)s"),
        watch_url(vid),
    ]
    print("[acquire]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not out.exists():
        raise RuntimeError(f"yt-dlp finished but {out} is missing")
    return out


def describe(path: Path) -> dict:
    """Everything Stage 0 needs to record about the footage."""
    meta, raw = probe_meta(path)
    info = {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "width": meta.width,
        "height": meta.height,
        "fps": round(meta.fps, 4),
        "frameCount": meta.frame_count,
        "durationSec": round(meta.duration, 2) if meta.duration else None,
        "codec": meta.codec,
    }
    if "_vfr_warning" in raw:
        info["vfrWarning"] = raw["_vfr_warning"]
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a match video and probe it.")
    ap.add_argument("url", help="YouTube URL or 11-char video id")
    ap.add_argument("--list-formats", action="store_true")
    ap.add_argument("--max-height", type=int, default=None,
                    help="cap resolution (default: best available -- see R2)")
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args(argv)

    vid = video_id(args.url)
    if args.list_formats:
        list_formats(vid)
        return 0

    path = download(vid, max_height=args.max_height, force=args.force)
    info = describe(path)

    C.STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    dest = C.STAGE0_DIR / f"{vid}_source.json"
    dest.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(json.dumps(info, indent=2))
    if "vfrWarning" in info:
        print(f"\n[acquire] WARNING: {info['vfrWarning']}", file=sys.stderr)
    print(f"\n[acquire] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
