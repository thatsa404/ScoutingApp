"""Stage 3 -- per-event appearance re-identification. Replaces rtrack.identify (OCR).

Two subcommands, and the split between them is the whole design:

    reid gallery <video> --event 2026necmp     # after a match is CURATED
    reid votes   <video> --event 2026necmp --match <key>   # before the NEXT one

`gallery` learns what each team looks like from a curated match and accumulates that
into one per-event file. `votes` reads the gallery and produces an opinion per track for
a new match.

WHY THIS SHAPE: the output of `votes` is deliberately identical to what rtrack.identify
used to emit -- {"tracks": {tid: {"tally": {team: n}, "voteList": [[t, team], ...]}}}.
That means rtrack.solve and rtrack.robots need NO changes: pass it with --identity and
the CP-SAT objective picks it up through the same term the bumper votes used, and
robots.retally redistributes it across track splits through the same timestamp
machinery. Matching an existing interface beat inventing a better one.

Measured against curated labels across three matches (detection-weighted, per track):

    train -> test                                            appearance      OCR
    f1m3 -> f1m2  (same camera, same alliances)                 99%          57%
    f1m2 -> f1m3                                                94%
    f1m3 -> sf11m1 (different field AND alliance flipped)       84%

See OCR_PLAN.md for the full evidence, including why the descriptor is grayscale.

VOTE BUDGET. OCR emitted 8-30 votes per track; appearance could emit one per detection,
which is hundreds. At w.vote=10 that would swamp every other term in the objective --
the alliance penalty is 250, i.e. 25 votes, and hundreds of appearance votes would make
bumper hue irrelevant. So votes are CAPPED per track (MAX_VOTES) and spread evenly in
time. The cap is not just a scaling trick: the learning curve saturated at ~40 crops per
robot, so beyond a couple of dozen consistent observations further ones carry almost no
information anyway.

The cap sits just under the alliance penalty on purpose. 24 votes = 240 < 250, so
appearance alone cannot overturn a confident bumper-hue reading, but combined with
geometry or a second signal it can. Curator pins (PIN_SOFT = 5000) still dominate
everything.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path, video_id
from . import tba as tba_mod
from .appear import descriptor, BODY_TOP, BODY_BOTTOM, MIN_BOX
# Imported for bumper_face only. curate does not import reid, so no cycle.
from . import curate as CU

STRIDE = 3            # every Nth labelled detection per track when building a gallery

# ---- reference crops ----------------------------------------------------------
# The gallery holds what a team looks like to the SOLVER: 48 grayscale numbers a person
# cannot inspect. These are the same knowledge in the form a person can use -- "here is
# what 1768 looked like the last four times it was identified" -- so a curator deciding
# whether the robot in front of them is 1768 can compare against confirmed examples
# instead of recalling a bumper from twenty minutes ago.
#
# Ranked by bumper_face (facing + white-on-bumper), NOT full legibility: separation is
# about whether a box is ambiguous, and a reference crop of an already-settled identity
# has no ambiguity left to worry about. See curate.bumper_face.
REF_PER_TEAM = 6      # kept per team, across the whole event
REF_PER_MATCH = 2     # ...of which at most this many from any ONE match, so a team's
                      # references show it in different lighting, positions and poses
                      # rather than four frames of a single drive down the field
REF_H = 150           # px; taller than the curator's inline crops because these are the
                      # reference being compared AGAINST and want to be legible
REF_Q = 72
MAX_VOTES = 24        # per track; see the module docstring on the vote budget
MIN_TRACK_DETS = 8    # below this a track's mean descriptor is too noisy to vote


def refs_path(event: str) -> Path:
    return C.STAGE3_DIR / f"{event}_refs.json"


def _ref_crop(img, box) -> str:
    """Whole-robot crop as a data URI, framed like chicklets rather than like appear.

    appear's BODY_TOP/BODY_BOTTOM deliberately CUT THE BUMPER OFF, because bumper colour
    is identical across an alliance and poisons a descriptor. A human reference needs the
    exact opposite -- the bumper is the number -- so this uses the chicklets framing that
    includes the whole robot and a margin around it.
    """
    from .chicklets import CROP_TOP, CROP_BOTTOM, CROP_PAD_X
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    px = bw * CROP_PAD_X
    cx1, cx2 = int(max(0, x1 - px)), int(min(W, x2 + px))
    cy1, cy2 = int(max(0, y1 + bh * CROP_TOP)), int(min(H, y1 + bh * CROP_BOTTOM))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return ""
    sc = REF_H / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), REF_H),
                      interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, REF_Q])
    import base64
    return ("data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")) if ok else ""


def merge_refs(event: str, found: dict[str, list]) -> int:
    """Fold this match's candidate reference crops into the event's reference sheet.

    `found` is {team: [{crop, t, match, s}]}. Kept: the best REF_PER_TEAM per team, with
    no more than REF_PER_MATCH from any single match -- diversity beats raw score here,
    because six crops of one drive down the field teach a curator less than three crops
    from three different matches.
    """
    p = refs_path(event)
    cur: dict[str, list] = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8")).get("teams", {})
        except Exception:
            cur = {}
    for team, cands in found.items():
        pool = cur.get(team, []) + cands
        pool.sort(key=lambda r: -r.get("s", 0.0))
        kept, per_match = [], Counter()
        for r in pool:
            m = r.get("match", "")
            if per_match[m] >= REF_PER_MATCH:
                continue
            kept.append(r)
            per_match[m] += 1
            if len(kept) >= REF_PER_TEAM:
                break
        cur[team] = kept
    p.write_text(json.dumps({"schemaVersion": 1, "event": event, "teams": cur}),
                 encoding="utf-8")
    return sum(len(v) for v in cur.values())


def gallery_path(event: str) -> Path:
    return C.STAGE3_DIR / f"{event}_gallery.npz"


# ---------------------------------------------------------------- gallery


def _load_gallery(p: Path) -> dict[str, tuple[np.ndarray, int]]:
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return {str(t): (z["cent"][i], int(z["count"][i]))
            for i, t in enumerate(z["teams"])}


def _save_gallery(p: Path, g: dict[str, tuple[np.ndarray, int]]) -> None:
    teams = sorted(g)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, teams=np.array(teams),
                        cent=np.stack([g[t][0] for t in teams]).astype(np.float32),
                        count=np.array([g[t][1] for t in teams], np.int32))


def build_gallery(stem: str, labeled_p: Path, event: str,
                  refs_only: bool = False) -> None:
    """Accumulate per-team mean descriptors from one CURATED match into the event file.

    Decodes the video once. This runs AFTER curation, so it is off the critical path for
    the match it reads -- the gallery it updates only helps subsequent matches.

    Merging is a count-weighted mean, so a team seen in four matches is not outvoted by
    its most recent one. Teams change during an event (repairs, a new intake, a swapped
    bumper), and the weighting means the gallery tracks that slowly rather than lurching.
    """
    rows = [json.loads(l) for l in labeled_p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])

    plan: dict[int, list] = defaultdict(list)
    seen: Counter = Counter()
    for r in rows:
        for d in r["dets"]:
            if d["tid"] < 0 or not d.get("team"):
                continue
            seen[d["tid"]] += 1
            if seen[d["tid"]] % STRIDE:
                continue
            if d["xyxy"][2] - d["xyxy"][0] < MIN_BOX:
                continue
            plan[r["f"]].append((str(d["team"]), d))

    want = sorted(plan)
    print(f"[reid] gallery from {labeled_p.name}: "
          f"{sum(len(v) for v in plan.values())} crops over {len(want)} frames")

    t_of = {r["f"]: r.get("t", 0.0) for r in rows}
    acc: dict[str, list] = defaultdict(list)
    refs: dict[str, list] = defaultdict(list)
    cap = cv2.VideoCapture(str(raw_path(stem)))
    idx = i = 0
    while i < len(want):
        ok, img = cap.read()
        if not ok:
            break
        if idx == want[i]:
            H, W = img.shape[:2]
            for team, d in plan[idx]:
                x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
                bh = y2 - y1
                cy1 = max(0, y1 + int(bh * BODY_TOP))
                cy2 = min(H, y1 + int(bh * BODY_BOTTOM))
                crop = img[cy1:cy2, max(0, x1):min(W, x2)]
                if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
                    continue
                acc[team].append(descriptor(crop))
                # Rolling top-K per team, so the JPEG is only encoded for a candidate
                # that actually displaces one. Encoding every candidate would be
                # thousands of encodes to keep two.
                face, digits = CU.bumper_face(img, d["xyxy"])
                s = round(0.5 * face + 0.5 * digits, 3)
                cur = refs[team]
                if len(cur) < REF_PER_MATCH or s > cur[-1]["s"]:
                    enc = _ref_crop(img, d["xyxy"])
                    if enc:
                        cur.append({"t": round(t_of.get(idx, 0.0), 1), "s": s,
                                    "match": stem, "crop": enc})
                        cur.sort(key=lambda r: -r["s"])
                        del cur[REF_PER_MATCH:]
            i += 1
        idx += 1
    cap.release()

    if refs_only:
        if refs:
            n = merge_refs(event, dict(refs))
            print(f"[reid] refs-only: {refs_path(event).name} now holds {n} crop(s); "
                  f"gallery untouched")
        return

    g = _load_gallery(gallery_path(event))
    for team, feats in sorted(acc.items()):
        F = np.stack(feats)
        new_c, new_n = F.mean(0), len(F)
        if team in g:
            old_c, old_n = g[team]
            tot = old_n + new_n
            g[team] = ((old_c * old_n + new_c * new_n) / tot, tot)
            print(f"    {team}: +{new_n} crops (now {tot})")
        else:
            g[team] = (new_c, new_n)
            print(f"    {team}: {new_n} crops (new)")
    _save_gallery(gallery_path(event), g)
    print(f"[reid] gallery -> {gallery_path(event)} ({len(g)} teams)")

    if refs:
        n = merge_refs(event, dict(refs))
        print(f"[reid] reference crops -> {refs_path(event).name} "
              f"({n} across {len(refs)} team(s) this match)")


# ---------------------------------------------------------------- voting


def vote_tracks(stem: str, tracks_p: Path, event: str, teams: list[str],
                max_votes: int = MAX_VOTES, npz_path: Path | None = None) -> dict:
    """Produce an identity-shaped opinion per track, from the CACHED descriptors.

    Reuses out/stage3/<stem>_appearance.npz rather than decoding again -- that cache is
    already built for split_on_appearance and holds exactly the descriptors needed, in
    the same (stitched) track id space this consumes. Zero extra decodes on the critical
    path between a match ending and a curator seeing frames.

    Votes are cast only among the six teams TBA lists for THIS match. That closed set is
    most of the accuracy: an event gallery may hold forty teams, but only six can be on
    the field, and restricting to them is free information.

    No alliance gating here, deliberately. The solver already carries an alliance term
    worth 250; gating the votes as well would count bumper hue twice and let a misread
    hue silently veto correct appearance evidence. Division of labour: appearance says
    who it looks like, CP-SAT arbitrates against colour and geometry.
    """
    npz_path = npz_path or (C.STAGE3_DIR / f"{stem}_appearance.npz")
    if not npz_path.exists():
        raise SystemExit(f"[reid] {npz_path} missing -- run rtrack.appear first")
    if refs_only:
        if refs:
            n = merge_refs(event, dict(refs))
            print(f"[reid] refs-only: {refs_path(event).name} now holds {n} crop(s); "
                  f"gallery untouched")
        return

    g = _load_gallery(gallery_path(event))
    if not g:
        raise SystemExit(f"[reid] {gallery_path(event)} missing or empty -- "
                         f"run 'reid gallery' on a curated match first")

    known = [t for t in teams if t in g]
    missing = [t for t in teams if t not in g]
    if not known:
        # NOT an error. FRC schedules spread teams out so nobody plays back-to-back --
        # measured on 2026mawor, matches 1-6 share NO teams at all and the gallery only
        # reaches full coverage at qm8. Early matches legitimately have nothing to vote
        # with, and the caller should carry on without votes rather than treat it as a
        # failure.
        print(f"[reid] none of this match's teams are in the {event} gallery yet "
              f"({', '.join(teams)}) -- no votes. Expected for an event's first "
              f"matches; coverage arrives once teams start repeating.")
        return None
    print(f"[reid] voting among {len(known)} known team(s): {known}")
    if missing:
        print(f"[reid] NOT in the gallery, cannot be voted for: {missing}")

    C_mat = np.stack([g[t][0] for t in known])
    C_mat = C_mat / (np.linalg.norm(C_mat, axis=1, keepdims=True) + 1e-9)

    z = np.load(npz_path)
    tid_a, t_a, feat = z["tid"], z["t"], z["feat"]
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)

    tracks: dict[str, dict] = {}
    for tid in sorted(set(tid_a.tolist())):
        m = tid_a == tid
        n = int(m.sum())
        if n < MIN_TRACK_DETS:
            continue
        ts, F = t_a[m], feat[m]
        order = np.argsort(ts)
        ts, F = ts[order], F[order]
        # Spread the vote budget evenly over the track's life, so retally can split it
        # correctly if the track is later cut. Voting on a contiguous block instead
        # would put every vote on one side of any cut.
        pick = np.linspace(0, n - 1, min(max_votes, n)).astype(int)
        sims = F[pick] @ C_mat.T
        vl = [[float(ts[p]), known[int(j)]]
              for p, j in zip(pick, sims.argmax(1))]
        tally = Counter(team for _t, team in vl)
        wins: dict[int, Counter] = defaultdict(Counter)
        for vt, team in vl:
            wins[int(vt // 6.0)][team] += 1
        top, share = tally.most_common(1)[0][0], 0.0
        if sum(tally.values()):
            share = tally[top] / sum(tally.values())
        tracks[str(tid)] = {
            "detections": n,
            "sampled": len(pick),
            "votes": len(vl),
            "tally": dict(tally),
            "team": top,
            "share": round(share, 3),
            "margin": round(float(np.median(sims.max(1) - np.sort(sims, 1)[:, -2]))
                            if len(known) > 1 else 1.0, 4),
            "voteList": vl,
            "timeline": [[int(k * 6.0), c.most_common(1)[0][0], sum(c.values())]
                         for k, c in sorted(wins.items())],
            "switches": [],
        }

    doc = {"video": stem, "event": event, "teams": teams,
           "source": "appearance", "gallery": str(gallery_path(event)),
           "maxVotes": max_votes, "tracks": tracks}
    agree = sum(1 for v in tracks.values() if v["share"] >= 0.8)
    print(f"[reid] {len(tracks)} track(s) voted; {agree} with >=80% agreement")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-event appearance re-identification.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("gallery", help="learn team appearance from a curated match")
    b.add_argument("video")
    b.add_argument("--event", required=True)
    b.add_argument("--labeled", type=Path, default=None)
    b.add_argument("--refs-only", action="store_true",
                   help="write reference crops WITHOUT merging descriptors into the "
                        "gallery. For backfilling matches whose gallery contribution "
                        "was already counted: build_gallery merges count-weighted, so "
                        "re-running it on the same match would count every crop twice "
                        "and make that team artificially hard to shift later.")

    v = sub.add_parser("votes", help="produce per-track identity votes for a match")
    v.add_argument("video")
    v.add_argument("--event", required=True)
    v.add_argument("--match", required=True)
    v.add_argument("--tracks", type=Path, default=None)
    v.add_argument("--max-votes", type=int, default=MAX_VOTES)
    v.add_argument("--out", type=Path, default=None)

    args = ap.parse_args(argv)
    C.ensure_dirs()
    stem = video_id(args.video)

    if args.cmd == "gallery":
        lp = args.labeled or (C.STAGE3_DIR / f"{stem}_labeled.jsonl")
        if not lp.exists():
            raise SystemExit(f"[reid] {lp} missing -- run rtrack.robots with "
                             f"--corrections first")
        build_gallery(stem, lp, args.event, refs_only=args.refs_only)
        return 0

    m = tba_mod.match_by_key(args.match)
    teams = [str(t) for t in m["red"]] + [str(t) for t in m["blue"]]
    doc = vote_tracks(stem, args.tracks, args.event, teams, args.max_votes)
    if doc is None:
        return 0                      # no coverage yet; not a failure, see vote_tracks
    out = args.out or (C.STAGE3_DIR / f"{stem}_reid.json")
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[reid] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
