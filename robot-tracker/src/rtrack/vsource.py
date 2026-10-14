"""Where a local clip came from on YouTube, so a route can link to the footage it was made from.

    uv run -m rtrack.vsource backfill 2026mawor          # every clip of an event
    uv run -m rtrack.vsource show 2026necmp1_qm13

A clip in data/raw/ is named for its MATCH, and is one of two things:

  match   TBA's per-match upload, downloaded whole. Clip time IS video time.
  stream  A window cut from the day's livestream archive. Video time = clipStartSec +
          clip time.

Each clip gets a sidecar, data/raw/<stem>.source.json:

    {"youtubeId": "9njBqpH4fhA", "kind": "stream", "clipStartSec": 18342.6,
     "method": "aligned", "score": 0.97}

rtrack.replay writes it as it downloads or cuts, from the numbers it actually used.
`backfill` recovers it for clips made before that, and does not trust arithmetic to do
so: TBA's match times were a whole match off on 2026necmp1 from qm5 on, and the padding
used to cut a clip has changed more than once. Instead it takes a short piece of the
stream at the predicted time and FINDS it inside the local clip, by correlating frame-
to-frame motion -- a broadcast copy of the same moment matches far better than any
other moment does, so a wrong prediction fails loudly rather than linking a strategist
to the wrong match.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config as C
from . import tba as tba_mod

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

SIG_FPS = 10              # motion signature rate; 0.1 s link precision
SIG_WH = (64, 36)
SNIPPET_S = 8.0           # stream piece fetched per attempt
MIN_SCORE = 0.6           # correlation of motion; the right moment scores ~0.9+
MIN_MARGIN = 0.25         # over the best moment more than 1 s away
DUR_TOL_S = 1.5           # a whole per-match upload matches its YouTube duration


def sidecar_path(stem: str) -> Path:
    return C.RAW_DIR / f"{stem}.source.json"


def load(stem: str) -> dict | None:
    """The clip's YouTube source, or None when unknown.

    A clip named by its own YouTube id (the original f1m2/f1m3 test clips) needs no
    sidecar: it is the whole video."""
    p = sidecar_path(stem)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    if _YT_ID.match(stem):
        return {"youtubeId": stem, "kind": "match", "clipStartSec": 0.0, "method": "named"}
    return None


def save(stem: str, youtube_id: str, kind: str, clip_start: float, method: str,
         **extra) -> dict:
    doc = {"youtubeId": youtube_id, "kind": kind,
           "clipStartSec": round(float(clip_start), 2), "method": method, **extra,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    sidecar_path(stem).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


# ── media helpers ────────────────────────────────────────────────────────────

def _ffmpeg() -> str:
    from .replay import ffmpeg_bin
    return ffmpeg_bin()


def _ffprobe() -> str:
    return str(Path(_ffmpeg()).with_name("ffprobe" + Path(_ffmpeg()).suffix))


def local_duration(path: Path) -> float | None:
    r = subprocess.run([_ffprobe(), "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                       timeout=60)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


@lru_cache(maxsize=64)
def _yt_info(vid: str) -> dict:
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "format": "bv*[height<=720]/bv*"}
    with yt_dlp.YoutubeDL(opts) as y:
        return y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)


_URLS: dict[str, tuple[str, float]] = {}


def _media_url(vid: str) -> str:
    hit = _URLS.get(vid)
    if hit and time.time() - hit[1] < 3000:
        return hit[0]
    _yt_info.cache_clear()
    info = _yt_info(vid)
    url = info.get("url") or next((f["url"] for f in info.get("requested_formats") or []
                                   if f.get("url")), None)
    if not url:
        raise RuntimeError(f"{vid}: no media URL")
    _URLS[vid] = (url, time.time())
    return url


def _signature(src: str, start: float | None = None, dur: float | None = None,
               timeout: float = 300) -> np.ndarray:
    """Grey thumbnails at SIG_FPS -> frame-to-frame differences, one row per step."""
    w, h = SIG_WH
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    if src.startswith("http"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-rw_timeout", "30000000"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-an", "-vf", f"fps={SIG_FPS},scale={w}:{h},format=gray",
            "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    f = np.frombuffer(r.stdout, np.uint8)
    n = f.size // (w * h)
    if n < 3:
        return np.zeros((0, w * h), np.float32)
    f = f[: n * w * h].reshape(n, w * h).astype(np.float32)
    return np.diff(f, axis=0)


def _locate(local: np.ndarray, snip: np.ndarray) -> tuple[float, float, float]:
    """Best position of `snip` inside `local` (seconds), its score, and the runner-up's."""
    m = len(snip)
    if m < 10 or len(local) < m:
        return 0.0, 0.0, 0.0
    s = snip - snip.mean(axis=1, keepdims=True)
    s = (s / (np.linalg.norm(s) + 1e-6)).ravel()
    L = local - local.mean(axis=1, keepdims=True)
    scores = np.empty(len(local) - m + 1, np.float32)
    for k in range(len(scores)):
        win = L[k:k + m].ravel()
        scores[k] = float(win @ s) / (np.linalg.norm(win) + 1e-6)
    k = int(np.argmax(scores))
    far = np.abs(np.arange(len(scores)) - k) > SIG_FPS
    second = float(scores[far].max()) if far.any() else 0.0
    return k / SIG_FPS, float(scores[k]), second


# ── backfill ─────────────────────────────────────────────────────────────────

def _streams(event: str) -> dict[str, float]:
    from .replay import archive_for, stream_start
    out = {}
    for a in archive_for(event):
        try:
            out[a["videoId"]] = stream_start(a["videoId"])
        except SystemExit as e:
            print(f"  [vsource] {e}")
    return out


def resolve(stem: str, tba_match: dict, all_m: dict, streams: dict[str, float],
            log_hint: str | None = None) -> dict | None:
    """Work out and save where `stem` came from. None if it cannot be shown."""
    clip = C.RAW_DIR / f"{stem}.mp4"
    dur = local_duration(clip)
    if not dur:
        print(f"  {stem}: unreadable clip")
        return None

    # 1. A per-match upload, downloaded whole: same length as the YouTube video.
    cands = [v["key"] for v in (tba_match.get("videos") or [])
             if v.get("type") == "youtube" and v.get("key")]
    if log_hint and log_hint not in cands:
        cands.insert(0, log_hint)
    for vid in cands:
        try:
            yd = float(_yt_info(vid).get("duration") or 0)
        except Exception as e:
            print(f"  {stem}: {vid} metadata failed ({type(e).__name__})")
            continue
        if yd and abs(yd - dur) <= DUR_TOL_S:
            return save(stem, vid, "match", 0.0, "duration", durationS=round(dur, 1))

    # 2. A window of the livestream: find a piece of the stream inside the clip.
    if not streams:
        print(f"  {stem}: no per-match video of its length and no stream archive")
        return None
    local = None
    suf = stem.split("_", 1)[1]
    mm = re.fullmatch(r"([a-z]+)(\d+)", suf)
    tried = []
    # Where TBA says each nearby match started in the stream. Neighbours are tried
    # because TBA's times can be a whole match off (2026necmp1 from qm5 on), and they are
    # tried BEFORE other offsets: a neighbour's time is the common miss, so checking
    # every lead on the wrong match first cost three stream reads per clip.
    times = []
    for shift in (0, -1, 1, -2, 2):
        if shift and not mm:
            break
        key = suf if not shift else f"{mm.group(1)}{int(mm.group(2)) + shift}"
        m = all_m.get(key)
        at = (m or {}).get("actual_time") or (m or {}).get("predicted_time")
        cand = [(at - s, v) for v, s in streams.items() if at and 0 < at - s < 16 * 3600]
        if cand:
            times.append((key, *min(cand)))
    for lead in (25.0, 70.0, -20.0):
        for key, off, vid in times:
            if local is None:
                local = _signature(str(clip))
            t_stream = off + lead
            try:
                snip = _signature(_media_url(vid), t_stream, SNIPPET_S, timeout=120)
            except Exception as e:
                print(f"  {stem}: stream read failed ({type(e).__name__}); retrying")
                _URLS.pop(vid, None)
                continue
            pos, score, second = _locate(local, snip)
            tried.append(f"{key}{lead:+.0f}s:{score:.2f}/{second:.2f}")
            if score >= MIN_SCORE and score - second >= MIN_MARGIN:
                return save(stem, vid, "stream", t_stream - pos, "aligned",
                            score=round(score, 3), runnerUp=round(second, 3),
                            timeFrom=key)
    print(f"  {stem}: NOT FOUND in the stream ({', '.join(tried) or 'no candidates'})")
    return None


def refine_cut(stem: str, vid: str, approx_start: float) -> dict | None:
    """Pin a fresh stream cut to the exact moment. A stream-copy cut starts at the
    keyframe BEFORE the requested time -- a second or two early -- so the requested
    offset alone would link that much early. One short read of the stream fixes it."""
    try:
        local = _signature(str(C.RAW_DIR / f"{stem}.mp4"))
        t_stream = approx_start + 60.0
        snip = _signature(_media_url(vid), t_stream, SNIPPET_S, timeout=120)
        pos, score, second = _locate(local, snip)
    except Exception as e:
        print(f"      (cut not refined: {type(e).__name__})")
        return None
    if score >= MIN_SCORE and score - second >= MIN_MARGIN:
        return save(stem, vid, "stream", t_stream - pos, "cut+aligned",
                    score=round(score, 3), runnerUp=round(second, 3))
    print(f"      (cut not refined: score {score:.2f} / {second:.2f})")
    return None


def _agent_log_hints() -> dict[str, str]:
    """per-match downloads rtrack.replay announced: {stem: youtubeId}."""
    p = C.TRACKER_ROOT / "out" / "agent.log"
    if not p.exists():
        return {}
    pat = re.compile(r"per-match video ([A-Za-z0-9_-]{11}) -> ([^ ]+)\.mp4")
    return {m.group(2): m.group(1)
            for m in pat.finditer(p.read_text(encoding="utf-8", errors="replace"))}


def backfill(event: str, only: list[str] | None = None, force: bool = False,
             published: bool = False) -> int:
    clips = sorted(p.stem for p in C.RAW_DIR.glob(f"{event}_*.mp4"))
    if published:
        pub = C.REPO_ROOT / "public" / "tracks"
        clips = [c for c in clips if (pub / f"{c}.json").exists()]
    if only:
        clips = [c for c in clips if c.split("_", 1)[1] in only or c in only]
    todo = [c for c in clips if force or not sidecar_path(c).exists()]
    print(f"[vsource] {event}: {len(clips)} clip(s), {len(todo)} to resolve")
    if not todo:
        return 0
    all_m = {m["key"].split("_", 1)[1]: m for m in tba_mod.fetch(f"/event/{event}/matches")}
    hints = _agent_log_hints()
    streams = None
    ok = bad = 0
    for stem in todo:
        m = all_m.get(stem.split("_", 1)[1])
        if not m:
            print(f"  {stem}: not a TBA match")
            bad += 1
            continue
        has_pm = any(v.get("type") == "youtube" for v in (m.get("videos") or []))
        if streams is None and not (has_pm or stem in hints):
            streams = _streams(event)
        doc = resolve(stem, m, all_m, streams or {}, hints.get(stem))
        if doc is None and streams is None:
            streams = _streams(event)
            doc = resolve(stem, m, all_m, streams, None)
        if doc:
            ok += 1
            print(f"  {stem}: {doc['kind']} {doc['youtubeId']} +{doc['clipStartSec']:.1f}s "
                  f"({doc['method']}{', %.2f' % doc['score'] if 'score' in doc else ''})",
                  flush=True)
        else:
            bad += 1
    print(f"[vsource] {ok} resolved, {bad} unresolved")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record which YouTube video each clip came from.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("event")
    b.add_argument("--matches", default=None, help="qm1,qm7 (default: all clips)")
    b.add_argument("--force", action="store_true")
    b.add_argument("--published", action="store_true",
                   help="only clips whose match has a route in public/tracks/")
    s = sub.add_parser("show")
    s.add_argument("stem")
    a = ap.parse_args(argv)
    if a.cmd == "show":
        print(json.dumps(load(a.stem), indent=1))
        return 0
    return backfill(a.event, a.matches.split(",") if a.matches else None, a.force,
                    a.published)


if __name__ == "__main__":
    raise SystemExit(main())
