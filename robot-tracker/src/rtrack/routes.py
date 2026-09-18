"""Stage 2 -- route plots. Robot paths drawn over the field, like receiver routes.

    uv run -m rtrack.routes GSxbsE42o5o --end 25          # the auto period
    uv run -m rtrack.routes GSxbsE42o5o --end 25 --panels # one panel per robot

Two views:
  combined  every route on one field, coloured by alliance, shaded light-to-dark
            with time so direction of travel is readable without arrows
  panels    small multiples, one robot per panel -- the format that actually gets
            used for comparing routes, because overlaid paths turn to spaghetti

Routes are drawn from `rtrack.project` output, so they are in real metres. Gaps
longer than `--max-gap` are BROKEN rather than bridged: a robot hidden behind a Hub
for two seconds could be anywhere, and a straight line across that gap would be an
invention at exactly the scale the plot is meant to reveal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C
from .acquire import video_id
from .calibrate import load_field_ref

RED, BLUE, GREY = "#ef4444", "#3b82f6", "#9ca3af"


def smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Light centred moving average. Detection jitter is a few px; this removes the
    visual buzz without moving the path meaningfully."""
    if len(a) < k or k < 2:
        return a
    pad = k // 2
    p = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(p[:, i], ker, mode="valid") for i in range(a.shape[1])],
                    axis=1)


def reachable(d: float, dt: float, v0: float) -> bool:
    """Could a robot already moving at v0 cover distance d in dt?

    Bounded by BOTH speed and acceleration. Elapsed time alone is the wrong test and
    it drew a real artefact: 1768 showed an 8.20 m step across a 0.87 s gap -- 9.4 m/s
    -- and because 0.87 < the 1.0 s gap threshold it was bridged and rendered as a
    long straight traverse across the field. The robot never went there; two tracks
    were handed over.

    Acceleration matters as much as speed. A pure speed bound allows 4.8 m in 0.87 s,
    which passes plenty of jumps made by a robot that was standing still a moment
    earlier. Starting from v0, the furthest reachable is v0*dt + a*dt^2/2, itself
    capped by the top speed.
    """
    if dt <= 0:
        return False
    lim = min(v0 * dt + 0.5 * C.ROBOT_MAX_ACCEL_MS2 * dt * dt,
              C.ROBOT_MAX_SPEED_MS * dt)
    return d <= lim * 1.25 + 0.35   # slack for position noise; see project.py biases


def segments(ts: np.ndarray, xy: np.ndarray, max_gap: float):
    """Split a path wherever the robot was unobserved too long, or moved impossibly.

    The second test is the one that matters for a route PLOT: a bridged gap is drawn
    as a straight line, so an unbroken segment is an assertion that the robot really
    travelled that way. Breaking on physics keeps the plot honest about what was seen.
    """
    out, start = [], 0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        d = float(np.hypot(*(xy[i] - xy[i - 1])))
        # speed entering the gap, from the step before it
        v0 = 0.0
        if i - 1 > start:
            pdt = ts[i - 1] - ts[i - 2]
            if pdt > 0:
                v0 = float(np.hypot(*(xy[i - 1] - xy[i - 2]))) / pdt
        if dt > max_gap or not reachable(d, dt, v0):
            if i - start >= 2:
                out.append((ts[start:i], xy[start:i]))
            start = i
    if len(ts) - start >= 2:
        out.append((ts[start:], xy[start:]))
    return out


def detect_auto_window(doc, bin_s: float = 0.5, moving: float = 0.35,
                       hold_s: float = 2.0) -> tuple[float, float]:
    """Find the auto period from robot motion instead of guessing a window.

    FRC matches have an unmistakable shape: robots sit still on the starting line,
    auto begins and everything moves, then there is a distinct pause before teleop.
    Measured on the test match this gives 8.7-29.5 s (~20.8 s), matching the score
    bug's 0:20 auto -- whereas a naive "first 25 s of the video" included 5.7 s of
    pre-match stillness and clipped the end of auto.
    """
    by = defaultdict(list)
    for s_ in doc["samples"]:
        if s_["tid"] >= 0 and "offfield" not in s_["flags"]:
            by[s_["tid"]].append(s_)
    bins = defaultdict(list)
    for ss in by.values():
        ss.sort(key=lambda z: z["t"])
        for a, b in zip(ss, ss[1:]):
            dt = b["t"] - a["t"]
            if 0 < dt < 0.5:
                v = np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt
                if v < C.ROBOT_MAX_SPEED_MS:
                    bins[round(b["t"] / bin_s) * bin_s].append(v)
    if not bins:
        raise SystemExit("no motion data")
    ts = sorted(bins)
    spd = np.array([np.mean(bins[t]) for t in ts])
    ts = np.array(ts)
    need = int(round(hold_s / bin_s))

    def sustained(mask, i):
        return mask[i:i + need].all() if i + need <= len(mask) else False

    mv = spd > moving
    start = next((i for i in range(len(ts)) if sustained(mv, i)), None)
    if start is None:
        raise SystemExit("could not find sustained motion")
    still = ~mv
    end = next((i for i in range(start + need, len(ts)) if sustained(still, i)), None)
    t0 = float(ts[start])
    t1 = float(ts[end]) if end is not None else float(ts[-1])
    return t0, t1


def load(stem: str, path: Path | None):
    p = path or (C.STAGE2_DIR / f"{stem}_positions.json")
    if not p.exists():
        raise SystemExit(f"{p} not found -- run rtrack.project first")
    return json.loads(p.read_text(encoding="utf-8"))


def team_alliances(stem: str) -> dict[str, str]:
    """Team -> alliance from TBA, via the Stage 3 robots file.

    Once a route is keyed by team, colouring it by bumper hue is strictly worse than
    looking the alliance up: TBA is ground truth, hue is inference. It was drawing
    9644 -- a blue team -- in red, because the hue vote inside the auto window
    happened to go the wrong way on its one contributing track.
    """
    p = C.STAGE3_DIR / f"{stem}_robots.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    t = doc.get("teams", {})
    return ({k: "red" for k in t.get("red", [])}
            | {k: "blue" for k in t.get("blue", [])})


def build(doc, t0: float, t1: float, min_samples: int, by_team: bool = True,
          alliance_of: dict[str, str] | None = None):
    """Group samples into routes, by TEAM where the positions file carries labels.

    A team's route through auto is normally several track fragments that Stage 3
    reassembled, so keying on track id splits one robot's path across several plots
    and names them after an internal counter. Falls back to track id when the
    positions came from an unlabelled track file.
    """
    keyed_by_team = by_team and any(s.get("team") for s in doc["samples"])
    by = defaultdict(list)
    for s in doc["samples"]:
        if s["tid"] < 0 or not (t0 <= s["t"] <= t1):
            continue
        if "offfield" in s["flags"]:
            continue
        key = s.get("team") if keyed_by_team else s["tid"]
        if key is None:
            continue
        by[key].append(s)

    tracks = {}
    for key, ss in by.items():
        if len(ss) < min_samples:
            continue
        ss.sort(key=lambda s: s["t"])
        # Several fragments can overlap in time after regrouping; averaging duplicate
        # timestamps keeps the path single-valued instead of zig-zagging between two
        # simultaneous detections of the same robot.
        merged: dict[float, list] = defaultdict(list)
        for s in ss:
            merged[round(s["t"], 3)].append(s)
        ts_sorted = sorted(merged)
        ts = np.array(ts_sorted)
        xy = smooth(np.array([[float(np.mean([q["x"] for q in merged[t]])),
                               float(np.mean([q["y"] for q in merged[t]]))]
                              for t in ts_sorted]))
        alliance = (alliance_of or {}).get(str(key)) if keyed_by_team else None
        if alliance is None:
            votes = defaultdict(int)
            for s in ss:
                if s["alliance"]:
                    votes[s["alliance"]] += 1
            alliance = max(votes, key=votes.get) if votes else None
        tracks[key] = {"t": ts, "xy": xy, "alliance": alliance,
                       "n": len(ts), "tids": sorted({s["tid"] for s in ss}),
                       "dist": float(np.abs(np.diff(xy, axis=0)).sum())}
    return tracks, keyed_by_team


def field_extent(ref):
    FL, FW = ref["fieldSizeM"]
    return [0, FL, 0, FW]


def draw_field(ax, ref):
    import matplotlib.image as mpimg
    img = mpimg.imread(str(C.REPO_ROOT / ref["image"]))
    r = ref["fieldRectPx"]
    crop = img[int(r["y0"]):int(r["y1"]), int(r["x0"]):int(r["x1"])]
    ax.imshow(crop, extent=field_extent(ref), origin="upper", alpha=0.85,
              interpolation="bilinear")
    FL, FW = ref["fieldSizeM"]
    ax.set_xlim(0, FL)
    ax.set_ylim(0, FW)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def colour(alliance):
    return {"red": RED, "blue": BLUE}.get(alliance, GREY)


def label_of(key, by_team: bool) -> str:
    return str(key) if by_team else f"#{key}"


def plot_combined(tracks, ref, dest: Path, title: str, max_gap: float,
                  by_team: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_field(ax, ref)
    for key, tr in sorted(tracks.items(), key=lambda kv: str(kv[0])):
        c = colour(tr["alliance"])
        for ts, xy in segments(tr["t"], tr["xy"], max_gap):
            # Shade along the path so direction reads without arrows.
            pts = xy.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            f = np.linspace(0.25, 1.0, len(segs))
            lc = LineCollection(segs, colors=[matplotlib.colors.to_rgba(c, a)
                                              for a in f], linewidths=2.6)
            ax.add_collection(lc)
        # Label OUTSIDE the start marker. A 4-digit team number does not fit inside a
        # dot at this scale -- it rendered as "644" and "201", which is worse than no
        # label because it looks like a team number and is not one.
        ax.plot(*tr["xy"][0], "o", color=c, ms=9, mec="white", mew=1.5, zorder=5)
        ax.annotate(label_of(key, by_team), tr["xy"][0],
                    xytext=(10, 10), textcoords="offset points",
                    color="white", fontsize=9, weight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.28", fc=c, ec="white", lw=1.1))
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def plot_panels(tracks, ref, dest: Path, title: str, max_gap: float,
                by_team: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Alliance then label, so the two alliances read as blocks rather than being
    # interleaved by sample count.
    items = sorted(tracks.items(),
                   key=lambda kv: (kv[1]["alliance"] or "z", str(kv[0])))
    n = len(items)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, tr) in zip(axes, items):
        draw_field(ax, ref)
        c = colour(tr["alliance"])
        for ts, xy in segments(tr["t"], tr["xy"], max_gap):
            ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=2.4, solid_capstyle="round")
        ax.plot(*tr["xy"][0], "o", color=c, ms=9, mec="white", mew=1.5)
        ax.plot(*tr["xy"][-1], "s", color=c, ms=7, mec="white", mew=1.2)
        head = (f"{key}" if by_team else f"track #{key}")
        ax.set_title(f"{head}  ({tr['alliance'] or 'unknown'})  "
                     f"{tr['n']} samples, {tr['dist']:.0f} m", fontsize=9)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: robot route plots.")
    ap.add_argument("video")
    ap.add_argument("--positions", type=Path, default=None)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=25.0, help="seconds")
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="break the path when unobserved longer than this (s)")
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--panels", action="store_true")
    ap.add_argument("--by-track", action="store_true",
                    help="key routes on track id even when team labels exist "
                         "(useful for auditing Stage 3's grouping)")
    ap.add_argument("--auto", action="store_true",
                    help="detect the auto period from robot motion and use it as "
                         "the window, instead of --start/--end")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    stem = video_id(args.video)
    ref = load_field_ref()
    doc = load(stem, args.positions)

    t_lo = doc["quality"]["tSpan"][0]
    if args.auto:
        t0, t1 = detect_auto_window(doc)
        print(f"[routes] detected auto period: {t0:.1f}-{t1:.1f}s "
              f"({t1 - t0:.1f}s) from the motion profile")
    else:
        t0, t1 = t_lo + args.start, t_lo + args.end
    tracks, by_team = build(doc, t0, t1, args.min_samples, not args.by_track,
                            team_alliances(stem))
    if not tracks:
        raise SystemExit("no tracks in that window")

    print(f"[routes] window {t0:.1f}-{t1:.1f}s -> {len(tracks)} routes, keyed by "
          f"{'TEAM' if by_team else 'track id'}")
    for key, tr in sorted(tracks.items(), key=lambda kv: -kv[1]["n"]):
        src = (f"  from tracks {tr['tids']}" if by_team else "")
        print(f"    {label_of(key, by_team):<6} {str(tr['alliance'] or '?'):<5} "
              f"{tr['n']:>4} samples  {tr['t'][0]:6.1f}-{tr['t'][-1]:6.1f}s  "
              f"{tr['dist']:5.1f} m{src}")

    title = (f"{stem}  AUTO period routes  {t0:.1f}-{t1:.1f}s ({t1-t0:.1f}s)"
             if args.auto else
             f"{stem}  routes  t={args.start:.0f}-{args.end:.0f}s "
             f"(match-relative from {t_lo:.1f}s)")
    if args.panels:
        dest = C.STAGE2_DIR / f"{stem}_routes_panels.png"
        plot_panels(tracks, ref, dest, title, args.max_gap, by_team)
    else:
        dest = C.STAGE2_DIR / f"{stem}_routes.png"
        plot_combined(tracks, ref, dest, title, args.max_gap, by_team)
    print(f"[routes] -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
