"""Read the broadcast's own title bar: "Qualification 12 of 100".

WHY THIS EXISTS. A slice is cut at `TBA actual_time - PAD`, and that number can simply be
wrong. On 2026necmp1 it ran ~520 s late from qm5 onward -- the length of one match cycle
-- so every clip from qm5 to qm25 contained the NEXT match, and the whole set was
curated, solved and pushed before anyone noticed. Every diagnosis made from timing alone
was wrong in turn: a camera change, then drift, then stream pauses. The broadcast says
which match it is, in text, in the same place, every frame. Asking it settles in one read
what timing arithmetic could not settle at all.

AND WHY THIS IS NOT THE OCR THAT WAS REMOVED. Bumper OCR was dropped because it measured
57% against appearance re-id's 99%: ~15 px digits on a robot that is moving, rotating and
half-occluded. This is a fixed region of white text on black, ~30 px tall, that does not
move. Measured across 25 2026necmp1 clips: 23 read correctly, 0 wrong, 2 unreadable from
a single sample. Same technique, a task three orders of magnitude easier.

OPTIONAL DEPENDENCY. easyocr is not required to run the pipeline -- when it is absent
this returns None and callers carry on with a warning. A check that cannot run must not
become a check that blocks.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2

from .acquire import raw_path

# Fractions of frame size, not pixels: the overlay is laid out proportionally and the
# same numbers hold for 720p and 1080p sources.
# A GENEROUS BAND, SEARCHED -- not a tight crop assumed. 2026necmp1_qm15 is broadcast as
# a screen capture of the FMS Audience Display inside a BROWSER WINDOW, tabs and address
# bar included, so the whole overlay sits lower and smaller than a full-bleed feed. A
# 0-5% title crop missed it completely and the clip was reported as "a different match"
# when it was perfectly correct. OCR finds the text wherever it is inside the band; the
# only cost of looking wider is a little time.
TITLE_TOP, TITLE_BOTTOM = 0.0, 0.14
TITLE_LEFT, TITLE_RIGHT = 0.20, 0.80

# "Qualification 12 of 100". The WORD is not matched -- easyocr reliably reads its Q as an
# O ("Oualification"), and requiring the word was what made the first version of this
# report nothing while the numbers underneath were perfectly correct. `<n> of <total>` is
# the distinctive part and survives that.
NUM_RE = re.compile(r"(\d+)\s*of\s*(\d+)")

# The team-number strip, just under the title. A SECOND AND INDEPENDENT signal, and the
# reason this module is worth more than a title reader: the title is occasionally covered
# by a replay wipe or an award graphic (qm14 and qm15 of 2026necmp1 are unreadable at
# every sampled moment), but the roster is on screen whenever the scoreboard is. Six
# numbers agreeing with TBA's roster is very hard to produce by accident -- a different
# match shares at most a team or two.
TEAM_TOP, TEAM_BOTTOM = 0.02, 0.26
# Scores, fuel counts and the clock also land in this strip, so the read is INTERSECTED
# with the expected roster rather than compared to it. Extra numbers cost nothing;
# missing ones are what matters.
TEAM_MIN_HIT = 5        # of 6, to call identity confirmed on the roster alone
TEAM_MAX_MISS = 2       # at or below this many hits, the clip is a different match

# Several attempts, because the title is occasionally covered by a replay wipe or an
# award graphic. Spread across the match rather than clustered, and mid-match first since
# that is where a clean title is most likely.
SAMPLE_MS = (120_000, 100_000, 140_000, 90_000, 160_000, 60_000, 200_000)

_READER = None


def _reader():
    """easyocr is slow to construct (~5 s) and the pipeline reads many clips."""
    global _READER
    if _READER is None:
        import warnings
        warnings.filterwarnings("ignore")
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _READER


def read_title(img) -> tuple[int, int] | None:
    """(match number, total) from one frame, or None if the title is not legible."""
    h, w = img.shape[:2]
    crop = img[int(h * TITLE_TOP):int(h * TITLE_BOTTOM),
               int(w * TITLE_LEFT):int(w * TITLE_RIGHT)]
    if crop.size == 0:
        return None
    try:
        txt = " ".join(t for _box, t, _conf in _reader().readtext(crop))
    except Exception:
        return None
    m = NUM_RE.search(txt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def read_teams(img) -> set[str]:
    """Every number visible in the team strip. Filtered by the caller, not here."""
    h, w = img.shape[:2]
    band = img[int(h * TEAM_TOP):int(h * TEAM_BOTTOM), :]
    if band.size == 0:
        return set()
    try:
        toks = [t for _box, t, _conf in _reader().readtext(band)]
    except Exception:
        return set()
    out = set()
    for t in toks:
        d = re.sub(r"\D", "", t)
        if d and 1 <= len(d) <= 5:
            out.add(str(int(d)))
    return out


def identify(stem: str, expect_n: int, expect_teams=(), samples=SAMPLE_MS) -> dict:
    """Both signals at once, from the same frames. One decode, two independent checks.

    Returns {ok, title, teams_hit, teams_total, why}. `ok` is True (this is the match),
    False (it is demonstrably a different one) or None (could not tell) -- and the caller
    must treat None as unknown rather than as either answer.
    """
    want = {str(t) for t in expect_teams}
    res = {"ok": None, "title": None, "teams_hit": 0,
           "teams_total": len(want), "why": "no legible frame"}
    p = Path(raw_path(stem))
    if not p.exists():
        res["why"] = "no such clip"
        return res
    try:
        _reader()
    except Exception as e:
        res["why"] = f"OCR unavailable ({type(e).__name__})"
        return res
    cap = cv2.VideoCapture(str(p))
    try:
        for ms in samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, ms)
            ok, img = cap.read()
            if not ok:
                continue
            title = read_title(img)
            hit = len(read_teams(img) & want) if want else 0
            if title:
                res["title"] = title[0]
            res["teams_hit"] = max(res["teams_hit"], hit)
            # The title is decisive when it reads at all: it names the match outright.
            if title:
                res["ok"] = (title[0] == expect_n)
                res["why"] = (f"title says {title[0]}"
                              + (f", {hit}/{len(want)} teams agree" if want else ""))
                return res
            # Falling back to the roster only once the title has failed everywhere.
            if want and hit >= TEAM_MIN_HIT:
                res["ok"] = True
                res["why"] = f"title unreadable; {hit}/{len(want)} teams match"
                return res
    finally:
        cap.release()
    if want and res["teams_hit"] <= TEAM_MAX_MISS:
        res["ok"] = False
        res["why"] = (f"title unreadable and only {res['teams_hit']}/{len(want)} teams "
                      f"match -- this looks like a different match")
    return res


def match_number(stem: str, samples=SAMPLE_MS) -> int | None:
    """The match number this clip actually contains, or None if it cannot be read.

    None means "could not tell", never "wrong" -- easyocr may be absent, the title may be
    covered in every sample, the file may be short. Callers must treat it as unknown.
    """
    p = Path(raw_path(stem))
    if not p.exists():
        return None
    try:
        _reader()
    except Exception as e:
        print(f"[scoreboard] OCR unavailable ({type(e).__name__}); skipping the check. "
              f"Install the `ocr` extra to enable it.")
        return None
    cap = cv2.VideoCapture(str(p))
    try:
        for ms in samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, ms)
            ok, img = cap.read()
            if not ok:
                continue
            hit = read_title(img)
            if hit:
                return hit[0]
    finally:
        cap.release()
    return None


def verify(stem: str, expect: int) -> tuple[bool | None, int | None]:
    """(ok, seen). ok is None when the title could not be read at all."""
    seen = match_number(stem)
    if seen is None:
        return None, None
    return seen == expect, seen


def main(argv=None) -> int:
    import argparse
    from . import tba as tba_mod
    ap = argparse.ArgumentParser(description="Check which match a clip really contains.")
    ap.add_argument("event")
    ap.add_argument("--matches", required=True, help="qm1-qm25 or qm1,qm7")
    args = ap.parse_args(argv)
    from .replay import parse_matches
    good = bad = unknown = 0
    for suf in parse_matches(args.matches):
        n = int(re.sub(r"\D", "", suf) or 0)
        stem = f"{args.event}_{suf}"
        ok, seen = verify(stem, n)
        if ok is None:
            unknown += 1
            print(f"  {suf:<6} unreadable")
        elif ok:
            good += 1
            print(f"  {suf:<6} OK")
        else:
            bad += 1
            print(f"  {suf:<6} CONTAINS MATCH {seen}  <-- mismatch")
    print(f"\n{good} correct, {bad} mismatched, {unknown} unreadable")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
