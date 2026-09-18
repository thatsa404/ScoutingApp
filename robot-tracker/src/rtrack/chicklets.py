"""Stage 3 -- contact sheet of what each team label was actually put on.

    uv run -m rtrack.chicklets GSxbsE42o5o --auto

One row per team, a handful of crops of the robot that label was assigned to, spread
across the window so a chimera shows up as a row that changes robot halfway along.

This exists because a vote share cannot be inspected. "6329, 77%" does not say whether
the 23% is noise on one robot or a second robot spliced into the same identity, and
"6201, 40%" does not say whether the label is wrong or merely thinly evidenced. The
answer is visible in two seconds if you can see the bumpers side by side, and there is
no metric that substitutes for it.

Crops are chosen the way rtrack.identify chooses what to OCR -- biggest and sharpest
win -- because those are the ones where a human can read the number too. They are
stratified across the window for the same reason identify stratifies: concentrating on
whichever stretch the robot spent nearest the camera would hide exactly the mid-track
identity switch this is meant to reveal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id

TILE_H = 175            # px per crop in the sheet
HEAD_W = 210            # left-hand label column
PER_TEAM = 8            # crops shown per team
OVERSAMPLE = 4          # candidates per slot before the sharpness cut
# How much of the robot to show. There are two competing needs and the balance moved.
#
# A tight bumper band (0.52-1.06, 5% side pad) maximises the number's pixel size, which
# is right when the question is "read this number". But a curator does not identify a
# robot only by its number -- shape, intake, hopper, bumper wear and who it is next to
# all carry identity, and most crops have no legible number at all (only ~33% do).
# Cropping them away leaves the human strictly less to go on than the machine had.
#
# So: include most of the robot body and a wider margin. The number shrinks but stays
# readable because CROP_H rises to compensate.
# Slightly beyond the box on every side: detector boxes clip, and the surroundings
# (who it is next to, what it is doing) carry identity too.
CROP_TOP, CROP_BOTTOM = -0.08, 1.20  # of box height; boxes routinely clip the bottom
CROP_PAD_X = 0.32

RED_BGR, BLUE_BGR = (68, 68, 239), (246, 130, 59)
BG, FG, DIM = (26, 26, 26), (238, 238, 238), (150, 150, 150)


def plan(rows, t0: float, t1: float, per_team: int, by_track: bool = False):
    """Candidate crops per team (or per track), stratified in time, biggest first.

    `by_track` answers the question the per-team sheet raises but cannot settle: when
    a team's row changes robot halfway along, is that the GROUPING joining two robots,
    or was the underlying track already chimeric? Only the second is unfixable
    downstream, so it decides where the work goes.
    """
    by_team: dict[str, list] = defaultdict(list)
    for r in rows:
        if not (t0 <= r["t"] <= t1):
            continue
        for d in r["dets"]:
            if d["tid"] < 0:
                continue
            if by_track:
                by_team[str(d["tid"])].append((r["f"], r["t"], d))
            elif d.get("team"):
                by_team[str(d["team"])].append((r["f"], r["t"], d))

    picks: dict[str, list] = {}
    for team, items in by_team.items():
        n_slots = per_team
        lo = min(t for _f, t, _d in items)
        hi = max(t for _f, t, _d in items) + 1e-6
        out = []
        for s in range(n_slots):
            a = lo + (hi - lo) * s / n_slots
            b = lo + (hi - lo) * (s + 1) / n_slots
            chunk = [it for it in items if a <= it[1] < b]
            chunk.sort(key=lambda it: -(it[2]["xyxy"][2] - it[2]["xyxy"][0]))
            out += chunk[:OVERSAMPLE]
        picks[team] = out
    return picks


def mark_subject(crop: np.ndarray, rect: tuple[int, int, int, int],
                 subject: tuple[int, int, int, int],
                 others: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Outline any OTHER tracked robot whose box reaches into this crop, and mark the
    narrow band the alliance classifier actually samples.

    MEASURED, and it invalidates naive reading of these crops: 7 of 12 crops offered
    for 6201 contained another robot covering 10%+ of the area, averaging 53%, and the
    intruder was 9644 five times over. Two crops read unmistakably "blue 9644" while
    the track's own hue was 75-81% RED -- because rtrack.alliance samples a 25%-inset
    band at 0.65-0.90 of box height, whereas this crop is full box width plus padding
    at 0.52-1.06. The classifier never sees the neighbour; the crop is full of it.

    An earlier version tried to DIM the other robots and was a silent no-op: the
    subject's own box is the crop by construction, so "other box minus subject box" is
    always empty. Two adjacent robots have overlapping boxes, and no box arithmetic
    separates them. Pretending otherwise would have been worse than not trying.

    So: state the ambiguity instead of hiding it. Amber outlines say "another tracked
    robot is in here", and the green band says "this is the strip the pipeline judged
    the alliance from" -- which is where the curator should look too.
    """
    out = crop.copy()
    h, w = out.shape[:2]

    def local(b):
        return (int(b[0] - rect[0]), int(b[1] - rect[1]),
                int(b[2] - rect[0]), int(b[3] - rect[1]))

    for b in others:
        x1, y1, x2, y2 = local(b)
        if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
            continue
        cv2.rectangle(out, (max(0, x1), max(0, y1)),
                      (min(w - 1, x2), min(h - 1, y2)), (30, 170, 250), 2)

    # The alliance-sampling band, in the subject's own coordinates.
    from .alliance import BAND_TOP, BAND_BOTTOM, BAND_INSET
    sx1, sy1, sx2, sy2 = subject
    bw, bh = sx2 - sx1, sy2 - sy1
    bx1, bx2 = local((sx1 + bw * BAND_INSET, 0, sx2 - bw * BAND_INSET, 0))[0::2]
    by1 = int(sy1 + bh * BAND_TOP) - rect[1]
    by2 = int(sy1 + bh * BAND_BOTTOM) - rect[1]
    if bx2 > bx1 and by2 > by1:
        cv2.rectangle(out, (max(0, bx1), max(0, by1)),
                      (min(w - 1, bx2), min(h - 1, by2)), (70, 235, 70), 2)
    return out


def grab(stem: str, picks: dict[str, list], rows=None):
    """One sequential pass over the video. Random seeking this file costs 747 ms a
    frame against 10.5 ms sequential, so we never seek -- see rtrack.identify."""
    wanted: dict[int, list] = defaultdict(list)
    for team, items in picks.items():
        for f, t, d in items:
            wanted[f].append((team, t, d))
    order = sorted(wanted)
    if not order:
        return {}
    # Every tracked box per frame, so a crop can dim the robots it is NOT about.
    by_frame: dict[int, list] = {}
    for r in (rows or []):
        by_frame[r["f"]] = [d for d in r["dets"] if d["tid"] >= 0]

    got: dict[str, list] = defaultdict(list)
    cap = cv2.VideoCapture(str(raw_path(stem)))
    idx, i = 0, 0
    while i < len(order):
        ok, img = cap.read()
        if not ok:
            break
        if idx == order[i]:
            H, W = img.shape[:2]
            for team, t, d in wanted[idx]:
                x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
                bw, bh = x2 - x1, y2 - y1
                cx1 = max(0, x1 - int(bw * CROP_PAD_X))
                cx2 = min(W, x2 + int(bw * CROP_PAD_X))
                cy1 = max(0, y1 + int(bh * CROP_TOP))
                cy2 = min(H, y1 + int(bh * CROP_BOTTOM))
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 8:
                    continue
                # Sharpness BEFORE dimming: dimming lowers contrast, so measuring
                # after it would quietly rank crops by how crowded they are rather
                # than by how readable the subject is.
                sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                            cv2.CV_64F).var())
                others = [tuple(int(v) for v in o["xyxy"])
                          for o in by_frame.get(idx, ())
                          if o["tid"] != d["tid"]]
                crop = mark_subject(crop, (cx1, cy1, cx2, cy2),
                                    (x1, y1, x2, y2), others)
                # f/cx/cy are the correction ANCHOR: a curator's label on this crop is
                # stored against this detection, which survives track renumbering.
                got[team].append({"t": t, "tid": d["tid"], "w": bw, "f": idx,
                                  "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                                  "sharp": sharp, "img": crop})
            i += 1
        idx += 1
    cap.release()
    return got


def quality(c) -> float:
    """Bigger and sharper is more readable -- the same signals that predict OCR yield."""
    return c["w"] * float(np.sqrt(max(c["sharp"], 1.0)))


def choose(got: dict[str, list], per_team: int):
    """Best crop in each EQUAL TIME SLOT across the track's life.

    Ranking globally by quality and taking the top N looks right and is wrong: it
    concentrates every crop on whichever stretch the robot spent nearest the camera.
    Measured on a 1077-detection track spanning 39-124 s, global ranking returned 11
    of 12 crops from 44-65 s, several within the same second, and nothing at all
    between 70 and 100 s.

    That defeats the entire purpose. These sheets exist to show whether a track
    changes robot partway through, and a sheet that samples one 20-second window
    cannot show that -- it is how track 16 came to be labelled 9644 from its first
    half while its second half read 6329.

    So: slot first, quality second. An empty slot is left empty rather than back-filled
    from a neighbour, because a gap in the strip is itself information about when the
    robot was visible.
    """
    out = {}
    for team, cands in got.items():
        if not cands:
            out[team] = []
            continue
        lo = min(c["t"] for c in cands)
        hi = max(c["t"] for c in cands) + 1e-6
        best: dict[int, dict] = {}
        for c in cands:
            s = min(per_team - 1, int(per_team * (c["t"] - lo) / (hi - lo)))
            if s not in best or quality(c) > quality(best[s]):
                best[s] = c
        picked = [best[s] for s in sorted(best)]
        # Slots can come up empty when a track is briefly lost; top up from whatever
        # is left, best-first, so the strip still fills out.
        if len(picked) < per_team:
            chosen = {id(c) for c in picked}
            spare = sorted((c for c in cands if id(c) not in chosen),
                           key=lambda c: -quality(c))
            picked += spare[:per_team - len(picked)]
        out[team] = sorted(picked, key=lambda c: c["t"])
    return out


def sheet(chosen: dict[str, list], order: list[str], alliance: dict[str, str],
          shares: dict[str, dict], title: str) -> np.ndarray:
    rows_img = []
    for team in order:
        cands = chosen.get(team, [])
        col = RED_BGR if alliance.get(team) == "red" else BLUE_BGR
        head = np.full((TILE_H, HEAD_W, 3), BG, np.uint8)
        cv2.rectangle(head, (0, 0), (6, TILE_H), col, -1)
        cv2.putText(head, team, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.15, col, 2,
                    cv2.LINE_AA)
        info = shares.get(team, {})
        cv2.putText(head, f"{alliance.get(team, '?')}", (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
        if info:
            cv2.putText(head, f"{info.get('share', 0):.0%} of {info.get('votes', 0)}"
                        f" votes", (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, FG, 1,
                        cv2.LINE_AA)
            cv2.putText(head, f"tracks {info.get('tracks', '')}"[:26], (16, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
        tiles = [head]
        for c in cands:
            f = TILE_H / c["img"].shape[0]
            im = cv2.resize(c["img"], (max(24, int(c["img"].shape[1] * f)), TILE_H),
                            interpolation=cv2.INTER_CUBIC)
            im = im.copy()
            cv2.rectangle(im, (0, TILE_H - 20), (im.shape[1], TILE_H), BG, -1)
            cv2.putText(im, f"t{c['t']:.0f}s #{c['tid']}", (4, TILE_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, TILE_H - 1), col, 2)
            tiles.append(np.full((TILE_H, 3, 3), BG, np.uint8))
            tiles.append(im)
        rows_img.append(np.hstack(tiles))

    W = max(r.shape[1] for r in rows_img)
    padded = [np.hstack([r, np.full((TILE_H, W - r.shape[1], 3), BG, np.uint8)])
              if r.shape[1] < W else r[:, :W] for r in rows_img]
    gap = np.full((6, W, 3), BG, np.uint8)
    body = [padded[0]]
    for p in padded[1:]:
        body += [gap, p]
    bar = np.full((42, W, 3), BG, np.uint8)
    cv2.putText(bar, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 1,
                cv2.LINE_AA)
    return np.vstack([bar] + body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Contact sheet of the crops behind each team label.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, default=None,
                    help="labelled track file (default: Stage 3 output)")
    ap.add_argument("--auto", action="store_true",
                    help="use the detected auto period as the window")
    ap.add_argument("--start", type=float, default=None, help="seconds (absolute)")
    ap.add_argument("--end", type=float, default=None, help="seconds (absolute)")
    ap.add_argument("--per-team", type=int, default=PER_TEAM)
    ap.add_argument("--by-track", action="store_true",
                    help="one row per TRACK id instead of per team -- shows whether "
                         "a track is itself chimeric, which grouping cannot fix")
    ap.add_argument("--min-dets", type=int, default=120,
                    help="with --by-track, skip tracks smaller than this")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    tp = args.tracks or (C.STAGE3_DIR / f"{stem}_labeled.jsonl")
    if not tp.exists():
        raise SystemExit(f"{tp} not found -- run rtrack.robots first")
    rows = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])

    t0, t1 = args.start, args.end
    if args.auto or t0 is None:
        from .routes import detect_auto_window, load as load_pos
        t0, t1 = detect_auto_window(load_pos(stem, None))
        print(f"[chicklets] auto period {t0:.1f}-{t1:.1f}s from the motion profile")
    label = (f"AUTO {t0:.1f}-{t1:.1f}s" if args.auto else f"t {t0:.0f}-{t1:.0f}s")

    rp = C.STAGE3_DIR / f"{stem}_robots.json"
    alliance: dict[str, str] = {}
    shares: dict[str, dict] = {}
    order: list[str] = []
    if rp.exists():
        doc = json.loads(rp.read_text(encoding="utf-8"))
        t = doc.get("teams", {})
        alliance = ({k: "red" for k in t.get("red", [])}
                    | {k: "blue" for k in t.get("blue", [])})
        order = list(t.get("red", [])) + list(t.get("blue", []))
        for g in doc.get("groups", {}).values():
            team = g.get("team")
            if not team:
                continue
            v = g.get("votes", {})
            tot = sum(v.values())
            shares[str(team)] = {"votes": tot,
                                 "share": (v.get(team, 0) / tot) if tot else 0.0,
                                 "tracks": g.get("tracks", [])}

    picks = plan(rows, t0, t1, args.per_team, args.by_track)
    if args.by_track:
        ndets: dict[str, int] = defaultdict(int)
        for r in rows:
            if t0 <= r["t"] <= t1:
                for d in r["dets"]:
                    if d["tid"] >= 0:
                        ndets[str(d["tid"])] += 1
        picks = {k: v for k, v in picks.items() if ndets[k] >= args.min_dets}
        # Row header shows the track's own votes and the team it was assigned to.
        tid_team = {}
        for r in rows:
            for d in r["dets"]:
                if d["tid"] >= 0 and d.get("team"):
                    tid_team[str(d["tid"])] = str(d["team"])
        ip = C.STAGE3_DIR / f"{stem}_identity.json"
        tallies = (json.loads(ip.read_text(encoding="utf-8"))["tracks"]
                   if ip.exists() else {})
        alliance = {k: alliance.get(v, "?") for k, v in tid_team.items()}
        shares = {}
        for k in picks:
            tal = tallies.get(k, {}).get("tally", {})
            tot = sum(tal.values())
            top = max(tal, key=tal.get) if tal else None
            shares[k] = {"votes": tot,
                         "share": (tal[top] / tot) if tot else 0.0,
                         "tracks": f"-> {tid_team.get(k, 'unassigned')}"
                                   f"  reads {top or '?'}"}
        order = sorted(picks, key=lambda k: -len(picks[k]))
    if not order:
        order = sorted(picks)
    order = [t for t in order if t in picks] + [t for t in picks if t not in order]
    if not order:
        raise SystemExit("no labelled detections in that window")

    print(f"[chicklets] {len(order)} teams, "
          f"{sum(len(v) for v in picks.values())} candidate crops")
    chosen = choose(grab(stem, picks, rows), args.per_team)
    for team in order:
        n = len(chosen.get(team, []))
        tids = sorted({c["tid"] for c in chosen.get(team, [])})
        print(f"    {team:>6} {alliance.get(team, '?'):<5} {n} crops  tracks {tids}")

    img = sheet(chosen, order, alliance, shares,
                f"{stem}   team labels, {label}   "
                f"(one row per team; a row that changes robot is a chimera)")
    out = args.out or (C.STAGE3_DIR / f"{stem}_chicklets.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[chicklets] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
