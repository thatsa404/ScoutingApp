"""Stage 2 -- project tracks from image pixels to field metres.

Reads the Stage 1 track file and the calibration, and writes positions in WPILib
metres. Every later consumer (route plots, zone occupancy, the eventual Dexie
import) reads this, never the pixel tracks.

    uv run -m rtrack.project GSxbsE42o5o --tracks out/stage1/MATCH3_st.jsonl

Two systematic biases are documented rather than silently "corrected", because both
are real and neither is measurable from a single camera:

  HEIGHT   The box bottom edge is where the robot meets the floor, which is what we
           want -- the detector was trained on whole-robot boxes, so no bumper-height
           correction is needed. Box-bottom jitter of a few px is the larger term.

  DEPTH    We see the robot's NEAR face. Its floor contact projects to the front of
           the chassis, not its centre, biasing position ~0.35 m toward the camera
           for a ~0.75 m deep robot. Correcting it needs robot heading, which we do
           not estimate. So "position" here means *near-face floor contact*, which is
           consistent frame to frame and fine for routes and zone occupancy.

Quality flags are attached, never dropped silently:
  offfield    projected outside the field rectangle (+ slack)
  fast        implied speed above ROBOT_MAX_SPEED -- a bad projection or an ID switch
  viewmoved   the camera was not in its calibrated pose here (see rtrack.viewcheck),
              so the homography does not apply and the metres are meaningless. FLAGGED
              RATHER THAN DELETED, and the detection behind it is untouched: identity
              work in that span is still perfectly good -- the robot was recognised,
              it just cannot be placed. Dropping the rows would throw away gallery
              evidence to fix a geometry problem.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import viewcheck as VC
from .acquire import video_id
from .calibrate import load_field_ref

SLACK_M = 0.6


def load_calib(stem: str) -> tuple[np.ndarray, dict | None]:
    p = C.CALIB_DIR / f"{stem}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found -- run rtrack.calibrate first")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return np.array(doc["H_video_to_fieldpx"], dtype=np.float64), doc.get("lens")


def project_points(pts_px: np.ndarray, H: np.ndarray, ref: dict,
                   lens: dict | None = None) -> np.ndarray:
    """Video pixels -> field metres (origin bottom-left of the field rect, +y up).

    Undistortion comes FIRST and is not optional when the calibration carries lens
    parameters: H was fitted in undistorted coordinates, so feeding it raw pixels
    silently reintroduces the whole barrel-distortion error the calibration removed.
    """
    pts = np.asarray(pts_px, np.float32)
    if lens:
        from . import lens as _lens
        pts = _lens.undistort_points(pts, lens).astype(np.float32)
    fp = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    r, ppm = ref["fieldRectPx"], ref["pxPerMeter"]
    X = (fp[:, 0] - r["x0"]) / ppm
    Y = (r["y1"] - fp[:, 1]) / ppm
    return np.stack([X, Y], axis=1)


# Visibility. Sampled in VIDEO space and projected FORWARD, never inverted: the same
# chain the positions themselves take, so the answer cannot disagree with them. An
# inverse would need re-distortion and a horizon guard, and would be a second
# implementation of the thing it is checking.
# VIDEO SAMPLES. This must be dense enough for the smallest the FIELD can be inside
# the frame, not for the frame. 320x180 was fine for a camera whose field fills the
# picture and badly wrong for one whose field does not: on 2026mawor the field occupies
# rows 481-711 of 1080 -- 21% of the frame height -- so only ~38 of 180 grid rows landed
# on it at all. The samples that survived were too scattered for the 5x5 close to
# bridge, the contour came out as a zig-zag through the middle of the field, and the
# export declared 52% of a fully-visible field unseen. The route plot then hatched the
# whole near half as "outside camera coverage", which is the exact misreading this
# overlay exists to prevent.
#
# Measured convergence, frac of field reported visible:
#
#     grid        2026mawor   2026necmp1     cost (mawor / necmp1)
#     320x180       0.484       0.865          5 ms / 5 ms
#     640x360       0.990       0.871          4 ms / 17 ms
#     960x540       0.993       0.872          9 ms / 39 ms
#     1920x1080     0.993       0.872         33 ms / 160 ms
#
# 960x540 is the first grid where both cameras have converged, and 40 ms once per export
# is not worth optimising. necmp1 barely moves because its field already fills the frame
# -- which is why this went unnoticed.
VIS_GRID_XY = (960, 540)        # video samples
VIS_MASK_XY = (400, 200)        # field raster the contour is traced on


def visible_region(H, ref: dict, lens: dict | None,
                   frame_wh: tuple[int, int] = (1920, 1080)):
    """(polygon in field metres, visible fraction) for this camera pose.

    WHY THIS IS WORTH SHIPPING. A route that stops at the far-left corner looks like a
    tracking failure and is nothing of the sort -- the camera simply cannot see there.
    On the 2026necmp1 camera 13.5% of the field is off-frame, almost all of it the two
    corners NEAREST the camera -- they sit at an extreme angle and fall outside the
    horizontal field of view, while the far corners are close to the image centre and
    stay in frame. (calib records cornersOutsideFrame = ['far-L', 'far-R'], but that
    naming is relative to the FIELD origin, not to the camera, so it is not evidence
    either way; see the note in visible_region on how the camera side is established.)

    Without the overlay the missing data is indistinguishable from lost data, and a
    reader draws conclusions about a robot that was never observable.
    """
    import cv2
    import numpy as np
    FL, FW = ref["fieldSizeM"]
    gw, gh = VIS_GRID_XY
    gx, gy = np.meshgrid(np.linspace(0, frame_wh[0] - 1, gw),
                         np.linspace(0, frame_wh[1] - 1, gh))
    XY = project_points(np.stack([gx.ravel(), gy.ravel()], axis=1), H, ref, lens)
    X, Y = XY[:, 0], XY[:, 1]
    ok = (X >= 0) & (X <= FL) & (Y >= 0) & (Y <= FW)
    if ok.sum() < 100:
        return None, None, None
    N, M = VIS_MASK_XY
    mask = np.zeros((M, N), np.uint8)
    ix = np.clip((X[ok] / FL * (N - 1)).astype(int), 0, N - 1)
    iy = np.clip((Y[ok] / FW * (M - 1)).astype(int), 0, M - 1)
    mask[iy, ix] = 255
    # Close pinholes between sample points; the grid is coarser than the raster.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    poly = [[round(float(px) / (N - 1) * FL, 2), round(float(py) / (M - 1) * FW, 2)]
            for px, py in approx]
    # WHICH SIDE THE CAMERA IS ON. The half with LESS coverage is the NEAR half, which
    # is the opposite of the intuition and was got wrong first time round.
    #
    # A side camera does not lose the far corners -- those sit near the middle of the
    # image and subtend a small angle. It loses its OWN corners, which are metres away
    # at an extreme angle and fall outside the horizontal field of view entirely.
    #
    # Verified two independent ways on 2026necmp1 rather than reasoned about:
    #   where each edge lands in the video   y~0 at row 491, y~8.1 at row 1039 of 1080
    #                                        -- the nearer edge sits low in frame
    #   pixel scale                          0.95 cm/px at y~0, 0.53 at y~8.1
    #                                        -- the nearer edge resolves finer
    # Both say the camera is at HIGH y, and the blind wedges in the polygon are at high
    # y too. Coverage: 0.991 of the low-y half, 0.770 of the high-y half.
    #
    # This exists so a route plot can be drawn the way the viewer saw it. Which end a
    # camera sits at is a property of the venue, not of FRC, so it is derived per
    # calibration rather than assumed once.
    filled = np.zeros((M, N), np.uint8)
    cv2.fillPoly(filled, [approx.astype(np.int32)], 255)
    cov_low = float(filled[:M // 2].mean())
    cov_high = float(filled[M // 2:].mean())
    side = "low-y" if cov_low <= cov_high else "high-y"
    return poly, round(float(cv2.contourArea(c)) / (N * M), 3), side


def auto_start_check(doc: dict, window_s: float = 1.0) -> dict:
    """At auto start the two alliances must occupy opposite ends. Free ground truth.

    DOES NOT ASSUME WHICH END IS WHICH. The camera can be on either side of the field
    at a different event, so "red starts at low x" is not a fact about FRC, only about
    this stream. Instead it derives the layout: take each alliance's median x at auto
    start, require the two to separate, and then flag any robot sitting closer to the
    OTHER alliance's end than its own.

    That formulation is orientation-free and self-calibrating, and it still catches
    what matters -- 6329 reading x=8.40 at t=0 when its alliance median was 4.4, which
    turned out to be the fuel pile detected as a robot and given a team label.

    Returns a report; it never raises. A failure here usually means a mislabelled
    track or a field element in the detections, not a broken homography.
    """
    try:
        from .routes import detect_auto_window
        t0, _ = detect_auto_window(doc)
    except Exception:
        return {"ran": False, "why": "could not detect the auto window"}

    per: dict[str, list] = defaultdict(list)
    for s in doc["samples"]:
        if s.get("team") and t0 <= s["t"] <= t0 + window_s:
            per[str(s["team"])].append(s["x"])
    if len(per) < 2:
        return {"ran": False, "why": "no team labels at auto start"}

    teams = {t: float(np.median(v)) for t, v in per.items()}

    # Which alliance a TEAM belongs to comes from TBA, never from observed hue.
    # Deriving it from hue skipped the one robot that needed checking: the fuel pile
    # wearing 6329's label is yellow, so it had no alliance evidence at all, the team
    # dropped out of the mapping, and the check passed while reporting nothing.
    team_alliance: dict[str, str] = {}
    rp = C.STAGE3_DIR / f"{doc['video']}_robots.json"
    if rp.exists():
        try:
            td = json.loads(rp.read_text(encoding="utf-8")).get("teams", {})
            for a in ("red", "blue"):
                for n in td.get(a, []):
                    team_alliance[str(n)] = a
        except Exception:
            pass
    no_hue = []
    if not team_alliance:                       # fall back to hue, and say so
        seen: dict[str, Counter] = defaultdict(Counter)
        for s in doc["samples"]:
            if s.get("team") and s.get("alliance") and t0 <= s["t"] <= t0 + window_s:
                seen[str(s["team"])][s["alliance"]] += 1
        team_alliance = {t: c.most_common(1)[0][0] for t, c in seen.items() if c}

    # A team with NO alliance colour at auto start is itself a red flag -- a robot
    # always shows a bumper, a field element does not.
    hue: dict[str, Counter] = defaultdict(Counter)
    for s in doc["samples"]:
        if s.get("team") and t0 <= s["t"] <= t0 + window_s and s.get("alliance"):
            hue[str(s["team"])][s["alliance"]] += 1
    no_hue = [t for t in teams if not hue.get(t)]

    side: dict[str, float] = {}
    for a in set(team_alliance.values()):
        xs = [teams[n] for n, aa in team_alliance.items() if aa == a and n in teams]
        if xs:
            side[a] = float(np.median(xs))
    if len(side) < 2:
        return {"ran": False, "why": "alliances not both present at auto start"}

    (a1, x1), (a2, x2) = sorted(side.items(), key=lambda kv: kv[1])
    sep = abs(x2 - x1)
    wrong = []
    for t, x in teams.items():
        a = team_alliance.get(t)
        if a not in side:
            continue
        mine, theirs = side[a], (x2 if side[a] == x1 else x1)
        if abs(x - theirs) < abs(x - mine):
            wrong.append({"team": t, "x": round(x, 2),
                          "ownEnd": round(mine, 2), "otherEnd": round(theirs, 2)})
    return {"ran": True, "autoStartS": round(t0, 2),
            "allianceEndsX": {a: round(x, 2) for a, x in side.items()},
            "separationM": round(sep, 2),
            "wrongEnd": wrong,
            "noAllianceColour": no_hue,
            "ok": bool(sep > 3.0 and not wrong and not no_hue)}


def run(stem: str, tracks: Path, out: Path, calib_stem: str | None = None) -> dict:
    """Project one track file into field metres.

    `calib_stem` lets a video reuse ANOTHER video's calibration, which is the normal
    case rather than the exception: a calibration describes a CAMERA, not a video, and
    an event's camera does not move between matches. Verified directly -- f1m3's
    calibration reprojects onto f1m2 with the field boundary still sitting on the real
    barriers, no adjustment. It is per camera, NOT per event: a second division uses a
    different camera and needs its own.
    """
    ref = load_field_ref()
    H, lens = load_calib(calib_stem or stem)
    FL, FW = ref["fieldSizeM"]

    rows = [json.loads(l) for l in tracks.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["f"])

    # Bottom-centre of every box, projected in one batch.
    idx, pts = [], []
    for ri, r in enumerate(rows):
        for di, d in enumerate(r["dets"]):
            x1, y1, x2, y2 = d["xyxy"]
            idx.append((ri, di))
            pts.append(((x1 + x2) / 2.0, y2))
    if not pts:
        raise SystemExit("no detections in the track file")
    XY = project_points(np.array(pts, np.float32), H, ref, lens)

    # Absent file means no finding, not a failure -- is_valid_at keeps everything when
    # the check has not been run, the same rule the scoreboard check follows.
    vdoc = VC.load(stem)

    samples = []
    for (ri, di), (X, Y) in zip(idx, XY):
        r, d = rows[ri], rows[ri]["dets"][di]
        flags = []
        if not (-SLACK_M <= X <= FL + SLACK_M and -SLACK_M <= Y <= FW + SLACK_M):
            flags.append("offfield")
        if not VC.is_valid_at(vdoc, r["t"]):
            flags.append("viewmoved")
        samples.append({"f": r["f"], "t": r["t"], "tid": d["tid"],
                        # Carried through when the input is a LABELLED track file, so
                        # routes can be drawn per team rather than per track id. A
                        # team's route is usually several tracks stitched by Stage 3;
                        # a track id is an implementation detail nobody asked about.
                        "team": d.get("team"),
                        "alliance": d.get("alliance"), "conf": d.get("conf"),
                        "x": round(float(X), 3), "y": round(float(Y), 3),
                        "flags": flags})

    # Kinematic check per track, in real metres now rather than pixel proxies.
    by_tid = defaultdict(list)
    for s in samples:
        if s["tid"] >= 0:
            by_tid[s["tid"]].append(s)
    n_fast = 0
    for tid, ss in by_tid.items():
        ss.sort(key=lambda s: s["t"])
        for a, b in zip(ss, ss[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            v = np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt
            if v > C.ROBOT_MAX_SPEED_MS:
                b["flags"].append("fast")
                n_fast += 1

    n_off = sum(1 for s in samples if "offfield" in s["flags"])
    n_vm = sum(1 for s in samples if "viewmoved" in s["flags"])
    quality = {
        "samples": len(samples),
        "tracks": len(by_tid),
        "offField": n_off,
        "offFieldPct": round(100 * n_off / len(samples), 2),
        "viewMoved": n_vm,
        "viewMovedPct": round(100 * n_vm / max(len(samples), 1), 2),
        "viewChecked": vdoc is not None,
        "kinematicViolations": n_fast,
        "kinematicPct": round(100 * n_fast / max(len(samples), 1), 2),
        "tSpan": [round(min(s["t"] for s in samples), 2),
                  round(max(s["t"] for s in samples), 2)],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"video": stem, "units": "meters", "fieldSizeM": [FL, FW],
           "convention": ref["convention"],
           "positionMeaning": "near-face floor contact; see module docstring",
           "quality": quality, "samples": samples}
    chk = auto_start_check(doc)
    doc["autoStartCheck"] = chk
    out.write_text(json.dumps(doc), encoding="utf-8")

    print(f"[project] {len(samples)} samples over {len(by_tid)} tracks")
    if chk.get("ran"):
        ends = "  ".join(f"{a} end x={x}" for a, x in chk["allianceEndsX"].items())
        print(f"[project] auto start t={chk['autoStartS']}s: {ends} "
              f"(separated by {chk['separationM']} m)")
        for t in chk.get("noAllianceColour", []):
            print(f"[project]   *** {t} shows NO bumper colour at auto start -- "
                  f"a robot always does; a field element does not")
        for w in chk["wrongEnd"]:
            print(f"[project]   *** {w['team']} sits at x={w['x']}, nearer the other "
                  f"alliance's end ({w['otherEnd']}) than its own ({w['ownEnd']})")
        if chk["ok"]:
            print("[project] auto-start layout consistent -- both alliances "
                  "at their own end")
    print(f"[project] off-field {n_off} ({quality['offFieldPct']}%)")
    print(f"[project] kinematic violations {n_fast} ({quality['kinematicPct']}%) "
          f"at >{C.ROBOT_MAX_SPEED_MS} m/s")
    print(f"[project] t span {quality['tSpan'][0]}-{quality['tSpan'][1]} s")
    print(f"[project] -> {out}")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: tracks -> field metres.")
    ap.add_argument("video")
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--calib-from", default=None, metavar="VIDEO",
                    help="reuse another video's calibration -- use this whenever the "
                         "same camera filmed both matches (see run())")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    stem = video_id(args.video)
    cal = video_id(args.calib_from) if args.calib_from else None
    if cal:
        print(f"[project] reusing calibration from {cal}")
    run(stem, args.tracks, args.out or (C.STAGE2_DIR / f"{stem}_positions.json"),
        calib_stem=cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
