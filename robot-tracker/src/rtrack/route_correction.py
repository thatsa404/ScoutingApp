"""Per-camera correction applied to exported route samples.

    uv run -m rtrack.route_correction freeze --camera 2026mawor \\
        --report out/stage2/2026mawor_occupancy_fit_report.json --camera-xy 7.990075,19.633362
    uv run -m rtrack.route_correction show --camera 2026mawor

Two ordered steps, both measured rather than assumed, applied to the samples
rtrack.export publishes and to nothing upstream of it:

1. RADIAL SHIFT, away from the camera's ground point. A detection's floor contact is
   the bottom of its box, which is the robot's NEAR face as the camera sees it, not its
   centre. Moving each point a fixed distance directly away from where the camera stands
   on the floor carries it to roughly the centre of the robot.

2. SIMILARITY -- one uniform scale about the frame origin, then a translation. What is
   left is a residual in the projection itself, fitted by rtrack.occupancy_fit from
   where robots were actually seen across many matches rather than from any one of them.

ORDER AND ORIGIN ARE PART OF THE CORRECTION. The similarity was fitted to points that
had already been shifted, and its scale is about (0, 0) of this frame -- so applying it
first, or about the field centre, would move every point by a different amount than the
fit measured. The document therefore stores the steps as an ordered list with their
anchor points, and apply() walks them in that order.

KEYED BY CAMERA, like calib/<stem>.json and the occluder file, because both steps are
properties of one camera position: the shift direction depends on where the camera is,
and the residual is that camera's homography. A correction fitted at one venue says
nothing about another, so an event without its own file gets no correction at all,
never someone else's.

EXPORT ONLY, deliberately. The solver's kinematic limits and identity joins were
measured on uncorrected projections; correcting upstream would quietly move every
threshold they depend on. Applying it to the deliverable changes where routes are drawn
without changing how they are assembled.

A CONSTANT SHIFT IS AN APPROXIMATION. One distance for every robot assumes one robot
depth. It is right on average -- across 22 2026mawor matches the share of samples inside
the robot-centre inset rose from 93.95% to 98.23% -- but a smaller robot flush against
the far wall is overshot: 5347 in 2026mawor_qm5 lands ~0.16 m closer to the wall than
its own footprint allows. A per-team depth would remove that; it is not modelled here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

SCHEMA = 1
KIND = "routeCorrection"


def path_for(camera: str) -> Path:
    return C.CALIB_DIR / f"{camera}_route_correction.json"


def load(camera: str | None) -> dict | None:
    """The frozen correction for this camera, or None. Never raises.

    A malformed file is reported and IGNORED rather than half-applied: an export that
    silently used a broken correction would publish routes that are wrong and say they
    were corrected, which is worse than publishing them uncorrected and saying so.
    """
    if not camera:
        return None
    p = path_for(camera)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("kind") != KIND or doc.get("schemaVersion") != SCHEMA:
            raise ValueError(f"not a schema-{SCHEMA} {KIND} document")
        for step in doc["steps"]:
            op = step["op"]
            if op == "radialShift":
                float(step["meters"]); [float(v) for v in step["fromXY"]]
            elif op == "similarity":
                float(step["scale"]); [float(v) for v in step["translationM"]]
                if step.get("about", "origin") != "origin":
                    raise ValueError("similarity must be about the frame origin")
            else:
                raise ValueError(f"unknown step {op!r}")
        return doc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[route_correction] IGNORING {p.name}: {exc}", file=sys.stderr)
        return None


def apply_xy(x: float, y: float, doc: dict) -> tuple[float, float]:
    """Walk the steps in order. Identical arithmetic to rtrack.occupancy_fit."""
    for step in doc["steps"]:
        if step["op"] == "radialShift":
            cx, cy = step["fromXY"]
            dx, dy = x - cx, y - cy
            n = math.hypot(dx, dy)
            if n > 1e-9:
                m = float(step["meters"])
                x, y = x + m * dx / n, y + m * dy / n
        elif step["op"] == "similarity":
            s = float(step["scale"])
            tx, ty = step["translationM"]
            x, y = x * s + tx, y * s + ty
    return x, y


def summary(doc: dict) -> dict:
    """What a route document records about the correction it was given."""
    return {
        "applied": True,
        "camera": doc.get("camera"),
        "frozenAt": doc.get("frozenAt"),
        "steps": [{k: v for k, v in st.items() if k != "why"} for st in doc["steps"]],
    }


def freeze(camera: str, report_path: Path, camera_xy: tuple[float, float]) -> Path:
    """Write the correction from an occupancy_fit report. Refuses a mismatched report."""
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    params, fit = rep.get("parameters") or {}, rep.get("fit") or {}
    if params.get("cameraStem") != camera:
        raise SystemExit(f"[route_correction] report is for camera "
                         f"{params.get('cameraStem')!r}, not {camera!r}")
    # leaveOneOut is one evaluation per held-out match; record the means, which is what
    # the investigation reported (0.122 m boundary MAE, 99.79% inside the field).
    loo = [e.get("evaluation") or {} for e in (rep.get("leaveOneOut") or [])]
    def _mean(key):
        v = [float(e[key]) for e in loo if isinstance(e.get(key), (int, float))]
        return round(sum(v) / len(v), 4) if v else None
    doc = {
        "schemaVersion": SCHEMA,
        "kind": KIND,
        "camera": camera,
        "frame": "rtrack.project sample frame -- the x/y rtrack.export publishes",
        "frozenAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": [
            {"op": "radialShift", "meters": float(params["shiftM"]),
             "fromXY": [float(camera_xy[0]), float(camera_xy[1])],
             "why": "near-face floor contact -> robot centre, directly away from the "
                    "camera's ground point"},
            {"op": "similarity", "scale": float(fit["scale"]),
             "translationM": [float(v) for v in fit["translationM"]], "about": "origin",
             "why": "event-level projection residual from rtrack.occupancy_fit"},
        ],
        "provenance": {
            "fitReport": str(report_path.relative_to(C.TRACKER_ROOT)
                             if report_path.is_absolute() and C.TRACKER_ROOT in report_path.parents
                             else report_path),
            "matches": len(rep.get("matches") or []),
            "axisScale": fit.get("axisScale"),
            "axisScaleDisagreement": fit.get("axisScaleDisagreement"),
            "leaveOneOut": {"heldOutMatches": len(loo),
                            "meanBoundaryMaeM": _mean("boundaryMaeM"),
                            "meanInsideFieldFraction": _mean("insideFieldFrac")},
            "parameters": params,
        },
    }
    out = path_for(camera)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="write calib/<camera>_route_correction.json")
    f.add_argument("--camera", required=True)
    f.add_argument("--report", type=Path, required=True)
    f.add_argument("--camera-xy", required=True, metavar="X,Y",
                   help="camera ground point in the route frame, as the fit used it")
    s = sub.add_parser("show", help="print the correction for a camera")
    s.add_argument("--camera", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        xy = tuple(float(v) for v in args.camera_xy.split(","))
        out = freeze(args.camera, args.report, xy)
        print(f"[route_correction] wrote {out}")
        return 0
    doc = load(args.camera)
    if doc is None:
        print(f"[route_correction] no correction for {args.camera} -- routes publish "
              f"uncorrected")
        return 1
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
