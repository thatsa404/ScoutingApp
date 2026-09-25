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

VOTE BUDGET. Appearance keeps a dense, time-distributed opinion stream in ``voteList``
because a long stitched track can later be cut into many short fragments. Sampling only
24 times before that cut left real fragments with just 1-3 local opinions. The dense
stream is provenance, not objective weight: the initial track tally and every final
fragment tally are independently capped at TALLY_VOTE_CAP after redistribution.

That cap sits just under the alliance penalty on purpose. 24 votes = 240 < 250, so
appearance alone cannot overturn a confident bumper-hue reading, but combined with
geometry or a second signal it can. Curator pins (PIN_SOFT = 5000) still dominate
everything. MAX_VOTES only bounds retained temporal provenance for memory and file size.
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
from .embed import (EMBEDDING_SPACE, fit_head, save_head, load_head,
                    head_path, MIN_HEAD_TRACKS)
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
MAX_VOTES = 4096      # dense temporal provenance; solver tally remains capped below
TALLY_VOTE_CAP = 24   # objective evidence per final segment; dense voteList is provenance
MIN_TRACK_DETS = 8    # below this a track's mean descriptor is too noisy to vote
POOL_RADIUS_S = 1.0   # measured optimum; three seconds crossed pose/identity boundaries
TEAM_TOP_K = 3        # local-best prototypes beat one maximum or a whole-team centroid
# alliance.classify emits no colour below this measured margin.  Using the same gate
# reproduces the audited "known alliance, otherwise six-team fallback" condition.
ALLIANCE_GATE_MIN_CONF = 0.15
VOTE_POLICY_VERSION = "cnn-fusion-pool1-top3-alliance-dense-segment-cap-v2"


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
        kept, per_match, seen = [], Counter(), set()
        for r in pool:
            m = r.get("match", "")
            # Dedupe by (match, timestamp): that pair names one frame, so two entries
            # sharing it are the same crop. Without this, re-running gallery on a match
            # -- which --refs-only exists to allow -- stacks identical candidates and a
            # team's slots fill with copies of one frame. Measured: 22 of 240 slots
            # duplicated after one backend rebuild.
            sig = (m, round(float(r.get("t", 0.0)), 1))
            if sig in seen:
                continue
            seen.add(sig)
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


def gallery_path(event: str, backend: str = "hist") -> Path:
    """Separate file per backend. The two descriptors are not comparable, and sharing
    one path would silently mix 48-d histograms with learned embeddings."""
    sfx = "_cnn" if backend == "cnn" else ""
    return C.STAGE3_DIR / f"{event}_gallery{sfx}.npz"


def appear_npz(stem: str, backend: str = "hist") -> Path:
    sfx = "_cnn" if backend == "cnn" else ""
    return C.STAGE3_DIR / f"{stem}_appearance{sfx}.npz"


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


def _reviewed_gallery(season: int, teams: list[str],
                      exclude_match: str | None = None) -> dict | None:
    """Load the versioned reviewed prototype cache, if one has been published.

    A missing or malformed reviewed cache is a normal migration state: appearance votes
    fall back to the legacy event centroid rather than making the pipeline unusable.
    """
    latest = C.OUT_DIR / "gallery" / str(int(season)) / "latest.json"
    if not latest.exists():
        return None
    try:
        meta = json.loads(latest.read_text(encoding="utf-8"))
        path = Path(meta["path"])
        if not path.is_absolute():
            path = C.TRACKER_ROOT / path
        z = np.load(path, allow_pickle=False)
        if str(z["kind"].item()) != "reviewed-gallery-v1":
            return None
        spaces = str(z["embeddingSpace"].item())
        if spaces != EMBEDDING_SPACE:
            print(f"[reid] reviewed gallery embedding space {spaces!r} is incompatible")
            return None
        proto_team = z["seasonTeam"].astype(str)
        wanted = {f"{season}:{t}" for t in teams}
        source_match = z["sourceMatch"].astype(str)
        keep = np.array([x in wanted for x in proto_team], dtype=bool)
        if exclude_match:
            # Held-out evaluation (and prospective free-run simulation) must not let
            # a match identify itself from crops accepted while reviewing that same
            # match.  Keep this filtering at the prototype loader so every downstream
            # aggregation sees the same leakage-free gallery.
            keep &= source_match != str(exclude_match)
        if not keep.any():
            return None
        return {"teams": proto_team[keep], "embedding": z["embedding"][keep],
                "sourceMatch": source_match[keep],
                "sourceTrack": z["sourceTrack"][keep].astype(int),
                "version": str(meta.get("version", z["manifestVersion"].item())),
                "embeddingSpace": spaces}
    except (KeyError, OSError, ValueError, TypeError) as exc:
        print(f"[reid] reviewed gallery unavailable ({exc}); using legacy gallery")
        return None


def build_gallery(stem: str, labeled_p: Path, event: str,
                  refs_only: bool = False, backend: str = "hist") -> None:
    """Accumulate per-team mean descriptors from one CURATED match into the event file.

    Decodes the video once. This runs AFTER curation, so it is off the critical path for
    the match it reads -- the gallery it updates only helps subsequent matches.

    Merging is a count-weighted mean, so a team seen in four matches is not outvoted by
    its most recent one. Teams change during an event (repairs, a new intake, a swapped
    bumper), and the weighting means the gallery tracks that slowly rather than lurching.
    """
    if backend == "cnn" and not refs_only:
        raise SystemExit("[reid] legacy solver-labelled CNN galleries cannot be "
                         "compiled into the fused representation. Use the reviewed "
                         "gallery workflow, which retains authoritative full crops.")
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
                acc[team].append(crop.copy() if backend == "cnn"
                                 else descriptor(crop))
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

    g = _load_gallery(gallery_path(event, backend))
    for team, feats in sorted(acc.items()):
        # RAW embeddings are accumulated, never whitened ones. The head is linear, so
        # whitening the mean equals the mean of the whitened -- storing raw means the
        # head can be refit from new curation without rebuilding any gallery.
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
    _save_gallery(gallery_path(event, backend), g)
    print(f"[reid] gallery -> {gallery_path(event, backend)} ({len(g)} teams)")

    if refs:
        n = merge_refs(event, dict(refs))
        print(f"[reid] reference crops -> {refs_path(event).name} "
              f"({n} across {len(refs)} team(s) this match)")


# ---------------------------------------------------------------- head

def build_head(event: str, min_tracks: int = MIN_HEAD_TRACKS,
               exclude: tuple[str, ...] = ()) -> int:
    """Fit the whitening head from every CURATED match of this event.

    Truth is the curator's label and nothing else. Using the solver's own auto-assigned
    teams would fit the head to the solver's mistakes and then score it against them --
    the error would be invisible and self-confirming.

    A track contributes one mean embedding, not one per crop. Crops within a track are
    near-duplicates; counting each would let a long track dominate the within-team
    covariance that the whole method rests on.
    """
    from . import corrections as CO

    V, teams = [], []
    n_match = 0
    corr_dir = C.TRACKER_ROOT / "corrections"
    for corr_p in sorted(corr_dir.glob(f"{event}_*_corrections.json")):
        stem = corr_p.name[: -len("_corrections.json")]
        if stem in exclude:
            print(f"    {stem}: EXCLUDED")
            continue
        npz_p = appear_npz(stem, "cnn")
        # STITCHED, not _labeled. The npz is written by rtrack.appear against the
        # stitched tracks, and robots.prepare_tracks then SPLITS those into a new id
        # space -- qm10 has 25 stitched tracks against 117 labelled ones. The integers
        # overlap (24 of 25 on that match) so resolving against _labeled silently pairs
        # a curator label with a different robot's embedding and still looks fine.
        st_p = C.STAGE1_DIR / f"{stem}_tracks_stitched.jsonl"
        if not (npz_p.exists() and st_p.exists()):
            continue
        rows = [json.loads(l) for l in st_p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        doc = json.loads(corr_p.read_text(encoding="utf-8"))
        human = [l for l in (doc.get("labels") or [])
                 if l.get("src") == "human" and l.get("team")]
        if not human:
            continue
        votes: dict[int, Counter] = defaultdict(Counter)
        for lab in CO.resolve(rows, human):
            if lab["ok"] and lab["tid"] is not None:
                votes[int(lab["tid"])][str(lab["team"])] += 1
        z = np.load(npz_p, allow_pickle=False)
        try:
            cache_space = str(z["embeddingSpace"].item())
        except (KeyError, ValueError, TypeError):
            cache_space = None
        if cache_space != EMBEDDING_SPACE:
            print(f"    {stem}: SKIPPED stale CNN cache ({cache_space!r})")
            continue
        tid_a, feat = z["tid"], z["feat"]
        used = 0
        for tid, c in votes.items():
            team, n = c.most_common(1)[0]
            # A track the curator labelled two different ways is a chimera; its mean
            # embedding is a blend of two robots and would poison the covariance.
            if n < 0.8 * sum(c.values()):
                continue
            m = tid_a == tid
            if int(m.sum()) < 4:
                continue
            V.append(feat[m].mean(0))
            teams.append(team)
            used += 1
        if used:
            n_match += 1
            print(f"    {stem}: {used} curator-labelled track(s)")

    if len(V) < min_tracks:
        raise SystemExit(f"[reid] only {len(V)} labelled track(s) across {n_match} "
                         f"match(es) -- need {min_tracks} to fit a head. This is not a "
                         f"failure: below that a head scores WORSE than no head "
                         f"(57.1% against 61.7% correct, measured prequentially), so "
                         f"votes correctly fall back to raw embeddings meanwhile.")
    h = fit_head(np.stack(V), np.array(teams))
    out = save_head(event, h)
    print(f"[reid] head from {h['tracks']} tracks / {h['teams']} teams "
          f"across {n_match} match(es), {h['npc']} dims -> {out}")
    return h["tracks"]


# ---------------------------------------------------------------- voting


def _pooled_queries(features: np.ndarray, times: np.ndarray, picks: np.ndarray,
                    radius: float = POOL_RADIUS_S) -> np.ndarray:
    """Mean a short same-track time window at selected samples, then normalize."""
    prefix = np.vstack([np.zeros((1, features.shape[1]), np.float64),
                        np.cumsum(features, axis=0, dtype=np.float64)])
    left = np.searchsorted(times, times[picks] - radius, side="left")
    right = np.searchsorted(times, times[picks] + radius, side="right")
    pooled = (prefix[right] - prefix[left]) / (right - left)[:, None]
    pooled = pooled.astype(np.float32)
    return pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)


def _team_scores(similarities: np.ndarray, prototype_teams: list[str],
                 teams: list[str], top: int = TEAM_TOP_K) -> np.ndarray:
    """Score each team by its best few reviewed views, never by one lucky maximum."""
    proto = np.asarray(prototype_teams)
    result = np.full((len(similarities), len(teams)), -np.inf, np.float32)
    for j, team in enumerate(teams):
        values = similarities[:, proto == team]
        if not values.shape[1]:
            continue
        k = min(int(top), values.shape[1])
        result[:, j] = np.partition(values, values.shape[1] - k, axis=1)[:, -k:].mean(1)
    return result


def _gate_team_scores(scores: np.ndarray, sample_alliance: np.ndarray,
                      sample_confidence: np.ndarray, teams: list[str],
                      team_alliance: dict[str, str],
                      minimum_confidence: float = ALLIANCE_GATE_MIN_CONF) -> np.ndarray:
    """Restrict confident samples to their three-team alliance; unknown falls back."""
    gated = scores.copy()
    for i, (alliance, confidence) in enumerate(zip(sample_alliance, sample_confidence)):
        alliance = str(alliance)
        if alliance not in ("red", "blue") or float(confidence) < minimum_confidence:
            continue
        eligible = np.array([team_alliance.get(team) == alliance for team in teams])
        if eligible.any():
            gated[i, ~eligible] = -np.inf
    return gated


def vote_tracks(stem: str, tracks_p: Path, event: str, teams: list[str],
                max_votes: int = MAX_VOTES, npz_path: Path | None = None,
                backend: str = "hist",
                exclude_gallery_match: str | None = None,
                include_scores: bool = False) -> dict:
    """Produce an identity-shaped opinion per track, from the CACHED descriptors.

    Reuses out/stage3/<stem>_appearance.npz rather than decoding again -- that cache is
    already built for split_on_appearance and holds exactly the descriptors needed, in
    the same (stitched) track id space this consumes. Zero extra decodes on the critical
    path between a match ending and a curator seeing frames.

    Votes are cast only among the six teams TBA lists for THIS match. That closed set is
    most of the accuracy: an event gallery may hold forty teams, but only six can be on
    the field, and restricting to them is free information.

    CNN votes use confidence-aware alliance gating.  A confident bumper call restricts
    the appearance comparison to the three scheduled teams on that alliance; unknown
    or low-confidence calls retain the six-team fallback.  This measured as the largest
    cheap appearance gain and avoids making a weak colour read a hard veto.
    """
    npz_path = npz_path or appear_npz(stem, backend)
    if not npz_path.exists():
        raise SystemExit(f"[reid] {npz_path} missing -- run rtrack.appear "
                         f"--backend {'both' if backend == 'cnn' else 'hist'} first")
    season = int(str(event)[:4]) if str(event)[:4].isdigit() else C.YEAR
    reviewed = (_reviewed_gallery(season, teams, exclude_gallery_match)
                if backend == "cnn" else None)
    g = _load_gallery(gallery_path(event, backend))
    if not g and reviewed is None:
        raise SystemExit(f"[reid] {gallery_path(event, backend)} missing or empty -- "
                         f"run 'reid gallery' on a curated match first, or publish "
                         f"a reviewed gallery for season {season}")
    available = ({str(t).split(":", 1)[1] for t in reviewed["teams"]}
                 if reviewed is not None else set(g))
    known = [t for t in teams if t in available]
    missing = [t for t in teams if t not in known]
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

    z = np.load(npz_path, allow_pickle=False)
    tid_a, t_a, feat = z["tid"], z["t"], z["feat"]
    if backend == "cnn":
        try:
            cache_space = str(z["embeddingSpace"].item())
        except (KeyError, ValueError, TypeError) as exc:
            raise SystemExit(f"[reid] {npz_path.name} has no embedding-space metadata; "
                             "rerun rtrack.appear --backend both") from exc
        if cache_space != EMBEDDING_SPACE:
            raise SystemExit(f"[reid] {npz_path.name} uses {cache_space!r}, expected "
                             f"{EMBEDDING_SPACE!r}; rerun rtrack.appear --backend both")
        try:
            alliance_a = z["alliance"].astype(str)
            alliance_confidence_a = z["allianceConfidence"].astype(np.float32)
        except KeyError as exc:
            raise SystemExit(f"[reid] {npz_path.name} lacks alliance provenance; "
                             "rerun rtrack.appear --backend both") from exc
    else:
        alliance_a = np.full(len(tid_a), "", dtype="U1")
        alliance_confidence_a = np.zeros(len(tid_a), np.float32)

    head = load_head(event) if backend == "cnn" else None
    if reviewed is not None:
        C_mat = reviewed["embedding"]
        proto_teams = [str(t).split(":", 1)[1] for t in reviewed["teams"]]
        gallery_version = reviewed["version"]
    else:
        C_mat = np.stack([g[t][0] for t in known])
        proto_teams = known
        gallery_version = None
    if backend == "cnn":
        if head is None:
            # Not an error. An event's first matches are curated before any head can be
            # fit, and raw embeddings still beat the histogram (0.685 vs 0.604 AUC).
            print(f"[reid] no {head_path(event).name} yet -- voting on RAW embeddings. "
                  f"Run 'reid fit-head' once a few matches are curated.")
        else:
            C_mat, feat = head(C_mat), head(feat)

    if C_mat.shape[1] != feat.shape[1]:
        raise SystemExit(f"[reid] gallery/cache dimensions differ "
                         f"({C_mat.shape[1]} vs {feat.shape[1]}); rebuild the reviewed "
                         "gallery and rerun appearance")
    C_mat = C_mat / (np.linalg.norm(C_mat, axis=1, keepdims=True) + 1e-9)
    if backend != "cnn":
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    team_alliance = {team: ("red" if i < 3 else "blue")
                     for i, team in enumerate(teams)}

    tracks: dict[str, dict] = {}
    for tid in sorted(set(tid_a.tolist())):
        m = tid_a == tid
        n = int(m.sum())
        if n < MIN_TRACK_DETS:
            continue
        ts, F = t_a[m], feat[m]
        track_alliance = alliance_a[m]
        track_alliance_confidence = alliance_confidence_a[m]
        order = np.argsort(ts)
        ts, F = ts[order], F[order]
        track_alliance = track_alliance[order]
        track_alliance_confidence = track_alliance_confidence[order]
        # Spread the vote budget evenly over the track's life, so retally can split it
        # correctly if the track is later cut. Voting on a contiguous block instead
        # would put every vote on one side of any cut.
        pick = np.linspace(0, n - 1, min(max_votes, n)).astype(int)
        query = (_pooled_queries(F, ts, pick) if backend == "cnn" else F[pick])
        sims = query @ C_mat.T
        scores = _team_scores(sims, proto_teams, known)
        raw_scores = scores.copy()
        if backend == "cnn":
            scores = _gate_team_scores(
                scores, track_alliance[pick], track_alliance_confidence[pick],
                known, team_alliance)
        winners = scores.argmax(1)
        vl = [[float(ts[p]), known[int(j)]] for p, j in zip(pick, winners)]
        # Keep a dense temporal opinion stream when requested, but do not let a long
        # unsplit track gain unlimited objective weight. Later split passes redistribute
        # voteList by time and independently cap each resulting segment; this solves the
        # current starvation problem (1-3 votes after splitting) without letting crop
        # count masquerade as independent evidence.
        tally_vl = vl
        if len(tally_vl) > TALLY_VOTE_CAP:
            tally_pick = np.linspace(0, len(tally_vl) - 1, TALLY_VOTE_CAP).astype(int)
            tally_vl = [tally_vl[int(i)] for i in tally_pick]
        tally = Counter(team for _t, team in tally_vl)
        wins: dict[int, Counter] = defaultdict(Counter)
        for vt, team in vl:
            wins[int(vt // 6.0)][team] += 1
        top, share = tally.most_common(1)[0][0], 0.0
        if sum(tally.values()):
            share = tally[top] / sum(tally.values())
        if len(known) > 1:
            ordered_scores = np.sort(scores, axis=1)
            gaps = ordered_scores[:, -1] - ordered_scores[:, -2]
            gaps = gaps[np.isfinite(gaps)]
            margin = float(np.median(gaps)) if len(gaps) else 1.0
        else:
            margin = 1.0
        tracks[str(tid)] = {
            "detections": n,
            "sampled": len(pick),
            "votes": len(tally_vl),
            "tally": dict(tally),
            "team": top,
            "share": round(share, 3),
            "margin": round(margin, 4),
            "voteList": vl,
            "timeline": [[int(k * 6.0), c.most_common(1)[0][0], sum(c.values())]
                         for k, c in sorted(wins.items())],
            "switches": [],
        }
        if include_scores:
            # Investigation-only provenance. Production needs only winners; retaining
            # all six similarities measures whether confidence margins discarded by
            # one-hot votes can resolve global assignment cascades.
            tracks[str(tid)]["scoreList"] = [
                [float(ts[p]), [round(float(value), 6) for value in row]]
                for p, row in zip(pick, raw_scores)
            ]

    doc = {"video": stem, "event": event, "teams": teams,
           "source": "appearance", "backend": backend,
           "head": (head is not None), "gallery": str(gallery_path(event, backend)),
           "reviewedGallery": (gallery_version is not None),
           "galleryVersion": gallery_version,
           "excludedGalleryMatch": exclude_gallery_match,
           "embeddingSpace": EMBEDDING_SPACE if backend == "cnn" else None,
           "votePolicyVersion": VOTE_POLICY_VERSION if backend == "cnn" else None,
           "poolRadiusSeconds": POOL_RADIUS_S if backend == "cnn" else None,
           "prototypeAggregator": f"top{TEAM_TOP_K}-mean" if backend == "cnn" else "max",
           "allianceGateMinimumConfidence": (ALLIANCE_GATE_MIN_CONF
                                                if backend == "cnn" else None),
           "maxVotes": max_votes, "tracks": tracks}
    doc["tallyVoteCap"] = TALLY_VOTE_CAP
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

    fh = sub.add_parser("fit-head", help="fit the cnn whitening head from curation")
    fh.add_argument("--event", required=True)
    fh.add_argument("--min-tracks", type=int, default=MIN_HEAD_TRACKS,
                    help="below this a head is worse than no head; see "
                         "embed.MIN_HEAD_TRACKS for the measurement.")
    fh.add_argument("--exclude", nargs="*", default=[],
                    help="match stems to leave out. For held-out evaluation: a head "
                         "fit on the match being scored would report its own training "
                         "accuracy.")

    v = sub.add_parser("votes", help="produce per-track identity votes for a match")
    v.add_argument("video")
    v.add_argument("--event", required=True)
    v.add_argument("--match", required=True)
    v.add_argument("--tracks", type=Path, default=None)
    v.add_argument("--max-votes", type=int, default=MAX_VOTES)
    v.add_argument("--out", type=Path, default=None)

    for q in (b, v):
        q.add_argument("--backend", choices=("hist", "cnn"), default="hist",
                       help="hist = the tuned 48-d histogram; cnn = the learned "
                            "embedding (needs rtrack.appear --backend both, and "
                            "'reid fit-head' for full accuracy). See rtrack.embed.")

    args = ap.parse_args(argv)
    C.ensure_dirs()
    if args.cmd == "fit-head":
        build_head(args.event, args.min_tracks, tuple(args.exclude))
        return 0

    stem = video_id(args.video)

    if args.cmd == "gallery":
        lp = args.labeled or (C.STAGE3_DIR / f"{stem}_labeled.jsonl")
        if not lp.exists():
            raise SystemExit(f"[reid] {lp} missing -- run rtrack.robots with "
                             f"--corrections first")
        build_gallery(stem, lp, args.event, refs_only=args.refs_only,
                      backend=args.backend)
        return 0

    m = tba_mod.match_by_key(args.match)
    teams = [str(t) for t in m["red"]] + [str(t) for t in m["blue"]]
    doc = vote_tracks(stem, args.tracks, args.event, teams, args.max_votes,
                      backend=args.backend)
    if doc is None:
        return 0                      # no coverage yet; not a failure, see vote_tracks
    out = args.out or (C.STAGE3_DIR /
                       f"{stem}_reid{'_cnn' if args.backend == 'cnn' else ''}.json")
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[reid] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
