"""Repair the `notrobot` labels that were written to deconflict a DUPLICATE BOX.

A one-off migration for a specific bug in the curation UI, kept in the tree because it
edits human-authored data and that should be reviewable.

THE BUG. `notrobot` is a TRACK-level verdict: corrections.pins_from() collects the
flagged tids into `blocked` and subtracts them from the pins, so flagging one box
discards every team label the curator gave that track, in any frame, with no message.
The old public/rtrack/curate.html told curators to do exactly that -- its banner for two
boxes on one robot read "give this one the team and mark box N 'not a robot'". Measured
on 2026necmp1: 70 notrobot labels across 18 correction files, 46 of them on tracks with
200+ detections. Re-curating qm24 without the flag took it from 84% to 99% agreement.

WHY THIS IS NOT GUESSWORK. That banner fired on exactly one condition -- another box
containing this one by >= CONTAIN -- so a notrobot label meeting the same condition, in
a frame where the containing box carries a team, IS that instruction being followed. The
team is read off the neighbour. Nothing is inferred from pixels, and a flag with no such
neighbour is left alone: that is a genuine false-positive call (a referee, a field
element), which is the only thing the flag should ever have meant.

TWO MODES, because they carry different risk:

  reteam    rewrite it as a team label taken from the containing box. This is the strong
            one and the default: it gives corrections.merge_duplicates the SECOND pin it
            needs to fuse the duplicate, rather than leaving it loose to be absorbed into
            another team's group.
  retract   delete the label. The track stops being dropped and its other labels survive,
            but the box goes back to unanswered. Invents nothing; also recovers less.

Every rewritten label keeps `wasNotrobot: true`. corrections.flag_of() only looks at
FLAGS, so the marker is inert downstream -- it exists so the edit is visible in the file
and reversible without consulting git.

MEASURED, on 2026necmp1_qm21, scored against its 100 UNTOUCHED human team labels so the
two runs are comparable:

    corrections     tracks pinned   dup merges   clashes   agree / scored
    as curated             61            0          2        95 / 98   97%
    reteam                 66            4          1        96 / 98   98%

One error fixed, none introduced. The accuracy delta is small because qm21's
contamination happened to be cheap; the structural change is the point -- merge_duplicates
fires where it previously could not, because it needs both boxes pinned to one team.

Usage:
    uv run -m rtrack.repair_notrobot --event 2026necmp1              # dry run
    uv run -m rtrack.repair_notrobot --event 2026necmp1 --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from . import config as C

# The threshold the old banner used. See overlapNote() in public/rtrack/curate.html:
# containment, not IoU, because a small fragment box sitting INSIDE a big one scores low
# IoU and high containment, and that is exactly the shape this case takes.
CONTAIN = 0.45


def containment(a, b) -> float:
    """Intersection over the SMALLER box."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    area = lambda x: max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])
    return iw * ih / max(1.0, min(area(a), area(b)))


def _nearest(boxes, xy):
    """The detection a label landed on. Labels carry a POINT in original video pixels --
    that is what makes them survive track renumbering -- so the box has to be recovered."""
    if not boxes:
        return None
    return min(boxes, key=lambda b: float(np.hypot((b[0] + b[2]) / 2 - xy[0],
                                                   (b[1] + b[3]) / 2 - xy[1])))


def repair(path: Path, mode: str = "reteam", apply: bool = False) -> dict:
    stem = path.name.replace("_corrections.json", "")
    doc = json.loads(path.read_text(encoding="utf-8"))
    labels = doc["labels"]
    flagged = [l for l in labels if l.get("src") == "human" and l.get("notrobot")]
    out = {"stem": stem, "notrobot": len(flagged), "changed": 0, "kept": 0,
           "unresolved": 0, "rows": []}
    if not flagged:
        return out

    lab_p = C.STAGE3_DIR / f"{stem}_labeled.jsonl"
    if not lab_p.exists():
        # Without the labelled tracks there are no boxes, so containment cannot be
        # evaluated and nothing is touched. Reported, never silently skipped.
        out["unresolved"] = len(flagged)
        return out
    by_frame = collections.defaultdict(list)
    for line in lab_p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            for d in r["dets"]:
                by_frame[r["f"]].append(d["xyxy"])

    teamed = [l for l in labels if l.get("src") == "human" and l.get("team")]
    for lab in flagged:
        mine = _nearest(by_frame.get(lab["f"], []), lab["xy"])
        if mine is None:
            out["unresolved"] += 1
            continue
        near = {str(o["team"]) for o in teamed if o["f"] == lab["f"]
                and (ob := _nearest(by_frame.get(o["f"], []), o["xy"])) is not None
                and containment(mine, ob) >= CONTAIN}
        if len(near) != 1:
            # Zero neighbours is a genuine not-a-robot. More than one means three boxes
            # in a pile and no single answer to read off; both are left for a human.
            out["kept"] += 1
            continue
        team = next(iter(near))
        out["rows"].append((lab["f"], tuple(lab["xy"]), team))
        out["changed"] += 1
        if apply:
            lab.pop("notrobot", None)
            lab["wasNotrobot"] = True
            if mode == "reteam":
                lab["team"] = team

    if apply and out["changed"]:
        if mode == "retract":
            # A repaired label in retract mode has no team and no flag, which is not a
            # label at all -- drop it rather than leave an empty answer behind.
            doc["labels"] = [l for l in labels
                             if l.get("team") or not l.get("wasNotrobot")]
        # indent=1 matches what curate.html's relay writes; anything else reformats
        # every line of a tracked file and buries the change.
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", default="2026necmp1")
    ap.add_argument("--mode", choices=("reteam", "retract"), default="reteam")
    ap.add_argument("--apply", action="store_true",
                    help="write the files. Without it, this only says what it would do.")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    a = ap.parse_args(argv)

    paths = sorted(
        Path("corrections").glob(f"{a.event}_*_corrections.json"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"qm(\d+)", p.name)) else 0)
    if not paths:
        print(f"[repair] no corrections for {a.event}")
        return 1

    print(f"[repair] {a.event}  mode={a.mode}  "
          f"{'APPLYING' if a.apply else 'DRY RUN -- nothing written'}\n")
    print(f"{'match':<22}{'notrobot':>9}{'repaired':>10}{'left as-is':>12}")
    tot = collections.Counter()
    for p in paths:
        r = repair(p, a.mode, a.apply)
        if not r["notrobot"]:
            continue
        left = r["kept"] + r["unresolved"]
        print(f"{r['stem']:<22}{r['notrobot']:>9}{r['changed']:>10}{left:>12}")
        if not a.quiet:
            for f, xy, team in r["rows"]:
                print(f"      f{f:<8}{str(xy):<16} -> {team}")
        tot["notrobot"] += r["notrobot"]
        tot["changed"] += r["changed"]
        tot["kept"] += left
    print(f"\n[repair] {tot['changed']} of {tot['notrobot']} repaired; "
          f"{tot['kept']} left as genuine not-a-robot calls")
    if not a.apply and tot["changed"]:
        print("[repair] re-run with --apply to write these")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
