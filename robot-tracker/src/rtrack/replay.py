"""Stage 4 -- slice matches out of a full-event stream archive.

    uv run -m rtrack.replay 2026mawor --matches qm1-qm25

This is the offline half of the livestream path, and it is deliberately the SAME
operation a live capture would perform: given a continuous recording and a match time,
cut the window and hand it to the existing pipeline. Validating it against a finished
event means the live version is a different trigger, not different machinery.

WHY NOT yt-dlp --download-sections: measured on this archive, it does not issue ranged
requests for a `was_live` VOD -- it downloads from the start at ~200 KiB/s, which for a
10.6 h / ~40 GB recording is days. Resolving the direct media URL and letting ffmpeg
range-seek to the offset takes 169 s for a 240 s window.

TIMING. offset = TBA actual_time - stream release_timestamp. Verified on 2026mawor qm1:
the computed offset landed within ~4 s of the true match start, with the score bug
reading "Qualification 1 of 75". That is the same arithmetic main.js already does in
findStreamForMatch(), so the live path can reuse it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config as C
from . import tba as tba_mod

# PADDING CANNOT FIX A WRONG INDEX, and a previous version of this comment claimed it
# could. Measured on 2026necmp1: clips looked increasingly misaligned (the match starting
# anywhere from 23 s to 185 s into a 240 s window, three containing no match at all), so
# the pad was widened to 60/210. Then the scoreboard was actually READ:
#
#     2026necmp1_qm4.mp4 -> "Qualification 4 of 100"   correct
#     2026necmp1_qm5.mp4 -> "Qualification 6 of 100"
#     2026necmp1_qm6.mp4 -> "Qualification 7 of 100"
#     2026necmp1_qm7.mp4 -> "Qualification 8 of 100"
#
# The clips are not drifting, they are ONE MATCH OFF from qm5 onward -- so every "drift"
# measurement above was timing the wrong match, and a wider window just captures the
# wrong match more completely.
#
# The cause is upstream: TBA's actual_time for match N is the real start of match N+1
# from qm5 on. Match 6 begins at stream 7706 s = 13:19:17 wall, and TBA gives 13:19:10
# as qm5's actual_time. The shift appears immediately after a 1323 s gap between qm4 and
# qm5 (typical is ~550 s), i.e. a field fault whose replay renumbered what followed. The
# stream itself is innocent: 29712 s of video across a 29781 s wall span, 69 s of loss in
# 8.25 hours.
#
# So this is back to the modest pad that genuinely covers start-time slop, and the real
# defence is verifying WHICH match a clip contains rather than assuming. See the note in
# main(). --per-match sidesteps it entirely for backfill, at the cost of not existing
# during a live event.
PAD_BEFORE_S = 30.0     # match start is approximate; also catches pre-auto staging
PAD_AFTER_S = 60.0      # 150 s match plus the settle, minus the padding we start with
MATCH_S = 150.0
URL_TTL_S = 3600.0      # re-resolve the media URL well before its `expire` param


def ffmpeg_bin() -> str:
    """ffmpeg, wherever it is. winget installs it off PATH on this machine."""
    hit = shutil.which("ffmpeg")
    if hit:
        return hit
    root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for p in root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"):
        return str(p)
    raise SystemExit("ffmpeg not found -- winget install Gyan.FFmpeg")


def archive_for(event_key: str, date: str | None = None) -> list[dict]:
    """The event's YouTube webcast archives, newest field first."""
    ev = tba_mod.fetch(f"/event/{event_key}")
    out = []
    for w in ev.get("webcasts") or []:
        if w.get("type") == "youtube" and w.get("channel"):
            if date and w.get("date") != date:
                continue
            out.append({"videoId": w["channel"], "date": w.get("date")})
    return out


def stream_start(video_id: str) -> float:
    """Unix seconds when the live broadcast began, from yt-dlp metadata."""
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    ts = info.get("release_timestamp") or info.get("timestamp")
    if not ts:
        raise SystemExit(f"{video_id}: no release_timestamp -- cannot align match times")
    return float(ts)


def media_url(video_id: str, max_height: int = 1080) -> str:
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "format": f"bv*[height<={max_height}]"}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    url = info.get("url")
    if not url:
        for f in info.get("requested_formats") or []:
            if f.get("url"):
                url = f["url"]
                break
    if not url:
        raise SystemExit(f"{video_id}: could not resolve a media URL")
    return url


def parse_matches(spec: str) -> list[str]:
    """'qm1-qm25' or 'qm1,qm7,sf3m1' -> a list of match suffixes."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"([a-z]+)(\d+)\s*-\s*([a-z]+)(\d+)", part)
        if m and m.group(1) == m.group(3):
            lo, hi = int(m.group(2)), int(m.group(4))
            out += [f"{m.group(1)}{i}" for i in range(lo, hi + 1)]
        elif part:
            out.append(part)
    return out


# A STALLED READ MUST FAIL, NOT HANG, and this is not a hypothetical robustness note:
# slicing 2026necmp1 got through qm1 and qm2 in five minutes, then sat on qm3 for FOUR
# HOURS. ffmpeg was alive and had used 3.6 seconds of CPU -- blocked on a googlevideo
# socket that never delivered another byte. The URL had not even expired. Because the
# slices are sequential, one dead socket halted the whole 25-match batch, silently,
# with the process still running so nothing looked wrong.
#
# Two independent guards, because either alone still hangs:
#   -rw_timeout   makes ffmpeg give up on a read that stalls (microseconds)
#   subprocess timeout  catches ffmpeg hanging anywhere -rw_timeout does not reach
# Plus -reconnect, so a recoverable drop is retried in-process rather than failing the
# slice outright -- which is the common case on a long CDN read.
FF_NET = ["-reconnect", "1", "-reconnect_streamed", "1",
          "-reconnect_on_network_error", "1", "-reconnect_delay_max", "30",
          "-rw_timeout", "30000000"]           # 30 s, in microseconds
SLICE_TIMEOUT_S = 420.0                        # a 150 s stream-copy takes well under 60 s


def fetch_match_video(yt: str, dest: Path) -> bool:
    """Download TBA's per-match video straight to this match's slice name.

    OPT-IN, NOT THE DEFAULT, and the reason is the whole point of this project: at a LIVE
    event these do not exist. Per-match uploads appear hours or days afterwards, so a
    pipeline that preferred them would work beautifully on last season's footage and have
    nothing to read during the event it was built for. The stream is the real input;
    these are for backfilling old events cheaply, which is exactly what the Burns and
    mawor test cases are.

    When they are available they are strictly better -- 215 s, 1920x1080, no offset to
    get wrong, smaller than the 240 s slice they replace -- which is why `--per-match`
    exists at all.
    """
    from .acquire import download, raw_path
    try:
        got = download(yt, max_height=None, force=False)
    except Exception as e:
        print(f"      per-match download failed: {str(e)[:140]}")
        return False
    got = Path(got)
    if got.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(dest))
    if not (dest.exists() and playable(dest)):
        dest.unlink(missing_ok=True)
        return False
    return True


def playable(dest: Path) -> bool:
    """Does this file actually have a readable container?

    SIZE IS NOT ENOUGH. A killed ffmpeg leaves a large, plausible-looking mp4 with no
    moov atom -- 2026necmp1_qm2 sat at 51 MB and passed the size check, then produced
    zero detections, an empty track file, and finally a CP-SAT INFEASIBLE four stages
    later, which is a long way to travel from the real cause. ffprobe answers it here.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
        for p in root.glob("Gyan.FFmpeg*/**/bin/ffprobe.exe"):
            exe = str(p)
            break
    if not exe:
        return True          # cannot check; do not fail a good slice over a missing tool
    r = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(dest)],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and (r.stdout or "").strip() not in ("", "N/A")


def slice_match(url: str, offset: float, dur: float, dest: Path, ff: str,
                tries: int = 2) -> bool:
    """Range-seek to `offset` and copy `dur` seconds. -ss BEFORE -i is the fast seek."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-hide_banner", "-loglevel", "error", *FF_NET,
           "-ss", f"{offset:.2f}", "-i", url, "-t", f"{dur:.2f}",
           "-c", "copy", "-y", str(dest)]
    for attempt in range(1, tries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=SLICE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"      ffmpeg exceeded {SLICE_TIMEOUT_S:.0f}s and was killed "
                  f"(attempt {attempt}/{tries})")
            dest.unlink(missing_ok=True)
            continue
        if r.returncode == 0 and dest.exists() and dest.stat().st_size >= 100_000 \
                and playable(dest):
            return True
        print(f"      ffmpeg failed (attempt {attempt}/{tries}): "
              f"{(r.stderr or '').strip()[:160]}")
        dest.unlink(missing_ok=True)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4: slice matches from an archive.")
    ap.add_argument("event", help="TBA event key, e.g. 2026mawor")
    ap.add_argument("--matches", required=True, help="qm1-qm25 or qm1,qm7,f1m2")
    ap.add_argument("--date", default=None, help="pick one webcast day (YYYY-MM-DD)")
    ap.add_argument("--out-prefix", default=None, help="default: <event>_<match>")
    ap.add_argument("--force", action="store_true", help="re-slice even if present")
    ap.add_argument("--pad-before", type=float, default=None, metavar="S",
                    help=f"override PAD_BEFORE_S (default {PAD_BEFORE_S:.0f})")
    ap.add_argument("--pad-after", type=float, default=None, metavar="S",
                    help=f"override PAD_AFTER_S (default {PAD_AFTER_S:.0f}). Widen this "
                         f"when a clip turns out to hold only pre-match staging -- the "
                         f"match is later than the window reached.")
    ap.add_argument("--index-shift", type=int, default=0, metavar="N",
                    help="take the slice TIME from match (this one + N), keeping the "
                         "output named for the match you asked for. For events where "
                         "TBA attributes actual_time to the wrong match -- 2026necmp1 "
                         "is off by one from qm5 on, so --index-shift -1 recovers it. "
                         "Always confirm the result with rtrack.scoreboard.")
    ap.add_argument("--no-verify", action="store_false", dest="no_verify",
                    default=False,
                    help="skip the scoreboard check on each slice")
    ap.add_argument("--per-match", action="store_true",
                    help="use TBA's per-match video when one exists, instead of "
                         "slicing the archive. Backfill only -- these are uploaded "
                         "after an event, so they do not exist during one. See "
                         "fetch_match_video.")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    ff = ffmpeg_bin()
    wants = parse_matches(args.matches)
    print(f"[replay] {args.event}: {len(wants)} match(es) requested")

    # /matches, not /matches/simple: `videos` is what carries the per-match uploads, and
    # simple omits it. See fetch_match_video for why those are preferred.
    all_m = {m["key"].split("_", 1)[1]: m
             for m in tba_mod.fetch(f"/event/{args.event}/matches")}
    have_per_match = sum(1 for m in all_m.values()
                         if any(v.get("type") == "youtube" and v.get("key")
                                for v in (m.get("videos") or [])))
    print(f"[replay] {have_per_match}/{len(all_m)} match(es) have a per-match video")
    archives = archive_for(args.event, args.date)
    if not archives and not have_per_match:
        raise SystemExit(f"[replay] {args.event} has neither per-match videos nor a "
                         f"YouTube webcast archive on TBA")

    # One archive per day; pick whichever day contains each match.
    starts = {}
    for a in archives:
        try:
            starts[a["videoId"]] = stream_start(a["videoId"])
            print(f"[replay] archive {a['videoId']} ({a['date']}) starts "
                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(starts[a['videoId']]))}")
        except SystemExit as e:
            print(f"[replay] {e}")

    urls, fetched_at = {}, {}
    ok = skip = fail = mismatch = 0
    for suf in wants:
        m = all_m.get(suf)
        if not m:
            print(f"  {suf}: not in {args.event}")
            fail += 1
            continue
        # The TIME may come from a different match than the one being named. Only the
        # time: `m` still supplies this match's videos and identity below.
        tm = m
        if args.index_shift:
            mm = re.fullmatch(r"([a-z]+)(\d+)", suf)
            alt = all_m.get(f"{mm.group(1)}{int(mm.group(2)) + args.index_shift}") if mm else None
            if alt is None:
                print(f"  {suf}: no match at shift {args.index_shift:+d}; using its own time")
            else:
                tm = alt
        at = tm.get("actual_time") or tm.get("predicted_time")
        if not at:
            print(f"  {suf}: no actual_time on TBA")
            fail += 1
            continue
        # The archive whose start precedes this match by the smallest positive gap.
        cand = [(at - s, v) for v, s in starts.items() if 0 < at - s < 16 * 3600]
        if not cand:
            print(f"  {suf}: no archive covers {time.strftime('%m-%d %H:%M', time.localtime(at))}")
            fail += 1
            continue
        offset, vid = min(cand)
        stem = args.out_prefix or f"{args.event}_{suf}"
        dest = C.RAW_DIR / f"{stem}.mp4"
        if dest.exists() and not args.force:
            print(f"  {suf}: already have {dest.name}")
            skip += 1
            continue
        # Backfill path only; the archive is what exists during a live event.
        if args.per_match:
            yt = next((v.get("key") for v in (m.get("videos") or [])
                       if v.get("type") == "youtube" and v.get("key")), None)
            if yt:
                t0 = time.time()
                if fetch_match_video(yt, dest):
                    print(f"  {suf}: per-match video {yt} -> {dest.name} "
                          f"({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
                    ok += 1
                    continue
                print(f"  {suf}: per-match video {yt} failed; slicing the archive instead")
        if vid not in urls or time.time() - fetched_at.get(vid, 0) > URL_TTL_S:
            urls[vid] = media_url(vid)
            fetched_at[vid] = time.time()
        t0 = time.time()
        pb = args.pad_before if args.pad_before is not None else PAD_BEFORE_S
        pa = args.pad_after if args.pad_after is not None else PAD_AFTER_S
        good = slice_match(urls[vid], offset - pb, MATCH_S + pb + pa, dest, ff)
        if good:
            mb = dest.stat().st_size / 1e6
            print(f"  {suf}: t+{offset:.0f}s -> {dest.name} "
                  f"({mb:.0f} MB, {time.time() - t0:.0f}s)")
            # Ask the broadcast which match this actually is. Checked HERE rather than
            # downstream because everything after this compounds the error: tracking,
            # solving against six wrong teams, a curator labelling them, and a gallery
            # learning from it. On 2026necmp1 that ran twenty-one matches deep.
            ok += 1
            if not args.no_verify:
                want = int(re.sub(r"\D", "", suf) or 0)
                if want:
                    # Both signals: the title names the match, the roster strip lists its
                    # six teams. The roster carries it when the title is behind a replay
                    # wipe, which is the only reason two 2026necmp1 clips were ever
                    # "unreadable" rather than confirmed.
                    from .scoreboard import identify
                    roster = [str(t) for t in (m.get("alliances", {})
                              .get("red", {}).get("team_keys", []))
                              + (m.get("alliances", {})
                                 .get("blue", {}).get("team_keys", []))]
                    roster = [t[3:] if t.startswith("frc") else t for t in roster]
                    v = identify(stem, want, roster)
                    if v["ok"] is False:
                        print(f"      !! MISMATCH: {v['why']}. "
                              f"The slice is NOT usable as {suf}.")
                        mismatch += 1
                    elif v["ok"] is None:
                        print(f"      (identity UNVERIFIED -- {v['why']})")
                    else:
                        print(f"      identity confirmed: {v['why']}")
        else:
            # Force a fresh media URL for the next match. The TTL alone is not enough:
            # a URL can go bad well before it expires (CDN node rotation, throttling),
            # and without this every remaining match reuses the same dead one until the
            # hour is up -- turning one failure into a whole failed batch.
            urls.pop(vid, None)
            fetched_at.pop(vid, None)
            fail += 1
    print(f"[replay] {ok} sliced, {skip} already present, {fail} failed"
          + (f", {mismatch} HOLDING THE WRONG MATCH" if mismatch else ""))
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
