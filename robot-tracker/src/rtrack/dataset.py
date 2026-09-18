"""Audit and convert third-party detection datasets.

Borrowed datasets are the fastest route to a Stage 1 detector, but they fail in
ways their own metrics will not show you. Three checks matter here, in order:

1. SCALE. Our bumpers are ~4-7% of frame width (80-130 px in 1920). A dataset of
   close-ups trains a model that has never seen an object that small.
2. DOMAIN. Broadcast wide shot vs stands-side vs on-robot POV. Eyeball the sheet.
3. LEAKAGE. Roboflow commonly augments before splitting, and video-frame datasets
   put adjacent near-identical frames in both splits. Either inflates val metrics.
   We re-split by source, and trust only our own eval set regardless.

    uv run -m rtrack.dataset audit roboflow
    uv run -m rtrack.dataset prepare roboflow --out data/datasets/bumpers
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C

SPLITS = ("train", "valid", "test")

# Measured off GSxbsE42o5o: bumpers span 80-130 px in a 1920-wide frame.
TARGET_W_LO, TARGET_W_HI = 0.042, 0.068

_RF_HASH = re.compile(r"\.rf\.[0-9a-f]+$")
_EXT_TAG = re.compile(r"_(jpg|jpeg|png)$", re.I)
_FRAME_NO = re.compile(r"[_-]\d{2,6}$")


def source_stem(p: Path) -> str:
    """Filename with Roboflow's augmentation hash stripped."""
    return _EXT_TAG.sub("", _RF_HASH.sub("", p.stem))


def source_clip(p: Path) -> str:
    """Coarser grouping: the clip a frame came from (v0_004 -> v0).

    Adjacent frames of one clip are near-identical. Splitting between them leaks
    just as badly as splitting augmented copies, and is easy to miss.
    """
    return _FRAME_NO.sub("", source_stem(p))


def read_yaml(root: Path) -> dict:
    """Minimal parse -- avoids a pyyaml dependency for four keys."""
    txt = (root / "data.yaml").read_text(encoding="utf-8")
    out: dict = {}
    m = re.search(r"^nc:\s*(\d+)", txt, re.M)
    if m:
        out["nc"] = int(m.group(1))
    m = re.search(r"^names:\s*\[(.*?)\]", txt, re.M | re.S)
    if m:
        out["names"] = [n.strip().strip("'\"") for n in m.group(1).split(",")]
    m = re.search(r"license:\s*(.+)", txt)
    if m:
        out["license"] = m.group(1).strip()
    m = re.search(r"url:\s*(.+)", txt)
    if m:
        out["url"] = m.group(1).strip()
    return out


def _labels(root: Path, split: str) -> list[Path]:
    d = root / split / "labels"
    return sorted(d.glob("*.txt")) if d.is_dir() else []


def _images(root: Path, split: str) -> list[Path]:
    d = root / split / "images"
    return sorted(p for p in d.glob("*") if p.suffix.lower() in
                  (".jpg", ".jpeg", ".png")) if d.is_dir() else []


def audit(root: Path, sheet: bool = True) -> dict:
    meta = read_yaml(root)
    names = meta.get("names", [])
    print(f"=== {root} ===")
    print(f"classes ({meta.get('nc')}): {names}")
    print(f"licence: {meta.get('license', '?')}")
    print(f"url: {meta.get('url', '?')}\n")

    report: dict = {"root": str(root), **meta, "splits": {}}
    all_w: list[float] = []

    for split in SPLITS:
        labs, imgs = _labels(root, split), _images(root, split)
        if not labs and not imgs:
            continue
        inst = Counter()
        per_img = []
        ws, hs = [], []
        for t in labs:
            n = 0
            for line in t.read_text(encoding="utf-8").split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                inst[int(parts[0])] += 1
                ws.append(float(parts[3]))
                hs.append(float(parts[4]))
                n += 1
            per_img.append(n)
        all_w += ws

        print(f"--- {split}: {len(imgs)} images, {sum(inst.values())} instances ---")
        for c in sorted(inst):
            nm = names[c] if c < len(names) else str(c)
            print(f"   {c:>2} {nm:<16} {inst[c]:>6}")
        if per_img:
            print(f"   objects/image mean {st.mean(per_img):.1f}, "
                  f"max {max(per_img)}, empty {per_img.count(0)}")
        if ws:
            q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]  # noqa: E731
            print(f"   box norm width  p10/50/90: "
                  f"{q(ws,.1):.3f}/{st.median(ws):.3f}/{q(ws,.9):.3f}")
            print(f"   box norm height p10/50/90: "
                  f"{q(hs,.1):.3f}/{st.median(hs):.3f}/{q(hs,.9):.3f}")
            ar = [w / h for w, h in zip(ws, hs) if h > 0]
            print(f"   aspect w/h      p10/50/90: "
                  f"{q(ar,.1):.2f}/{st.median(ar):.2f}/{q(ar,.9):.2f}")
        report["splits"][split] = {
            "images": len(imgs), "instances": sum(inst.values()),
            "byClass": {names[c] if c < len(names) else str(c): n
                        for c, n in sorted(inst.items())},
            "boxWidthMedian": round(st.median(ws), 4) if ws else None,
        }
        print()

    # ---- leakage
    print("--- split hygiene ---")
    stems = {s: {source_stem(p) for p in _images(root, s)} for s in SPLITS}
    clips = {s: {source_clip(p) for p in _images(root, s)} for s in SPLITS}
    leak = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if not stems.get(a) or not stems.get(b):
            continue
        so, co = stems[a] & stems[b], clips[a] & clips[b]
        leak[f"{a}|{b}"] = {"sharedSources": len(so), "sharedClips": len(co)}
        flag = "  <-- LEAK" if so or co else ""
        print(f"   {a} vs {b}: {len(so)} shared source images, "
              f"{len(co)} shared clips{flag}")
        if co:
            print(f"      e.g. {sorted(co)[:4]}")
    report["leakage"] = leak
    print(f"   distinct clips: "
          f"{ {s: len(v) for s, v in clips.items() if v} }\n")

    # ---- the verdict that actually matters
    print("--- fit against our footage ---")
    verdict = []
    if all_w:
        med = st.median(all_w)
        p10 = sorted(all_w)[int(0.1 * (len(all_w) - 1))]
        print(f"   our bumpers:      {TARGET_W_LO:.3f}-{TARGET_W_HI:.3f} norm width")
        print(f"   this dataset:     p10 {p10:.3f}, median {med:.3f}")
        ratio = med / ((TARGET_W_LO + TARGET_W_HI) / 2)
        if ratio > 2.5:
            verdict.append(
                f"SCALE MISMATCH: objects are {ratio:.1f}x larger than ours. Train with "
                "aggressive scale augmentation (scale=0.9) and expect to need tiled "
                "(SAHI) inference; a model that never saw a 90 px robot will miss ours.")
        elif ratio < 0.5:
            verdict.append(f"objects are {1/ratio:.1f}x SMALLER than ours -- unusual, check the sheet.")
        else:
            verdict.append(f"scale is compatible ({ratio:.1f}x ours).")
        report["scaleRatio"] = round(ratio, 2)

    nm_l = [n.lower() for n in names]
    if not any("red" in n or "blue" in n for n in nm_l):
        verdict.append(
            "NO ALLIANCE CLASSES: this dataset cannot give red/blue for free. Either "
            "classify alliance by bumper hue inside each detected box (rtrack.prelabel "
            "already has calibrated red/blue bands) or relabel.")
    for v in verdict:
        print(f"   * {v}")
    report["verdict"] = verdict

    if sheet:
        p = contact_sheet(root, names)
        print(f"\n   sample sheet: {p}\n   LOOK AT IT -- domain fit is the one thing "
              "these numbers cannot tell you.")
    return report


def contact_sheet(root: Path, names: list[str], n: int = 9,
                  seed: int = 7) -> Path:
    labs = _labels(root, "train") or _labels(root, "valid")
    random.seed(seed)
    tiles = []
    for t in random.sample(labs, min(len(labs), 300)):
        img_p = next((root / ("train" if (root / "train" / "labels" / t.name).exists()
                              else "valid") / "images").glob(t.stem + ".*"), None)
        if img_p is None:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        rows = [l.split() for l in t.read_text(encoding="utf-8").split("\n")
                if len(l.split()) >= 5]
        if not rows:
            continue
        for r in rows:
            cx, cy, bw, bh = (float(x) for x in r[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 120), 2)
            cv2.putText(img, f"{bw:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 120), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (426, 240)))
        if len(tiles) == n:
            break

    C.STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    dest = C.STAGE1_DIR / f"dsaudit_{root.name}.jpg"
    if tiles:
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows_img = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(dest), cv2.vconcat(rows_img))
    return dest


def prepare(root: Path, out: Path, mapping: dict[str, str], val_frac: float) -> None:
    """Remap classes and re-split by source clip so no clip spans both splits."""
    meta = read_yaml(root)
    names = meta.get("names", [])
    idx_of = {n: i for i, n in enumerate(names)}

    # Target schema comes from the mapping's destinations, in first-appearance order,
    # rather than being hardcoded: a single-class `robot` dataset and a two-class
    # red/blue one are both legitimate inputs, and which one we get is not our choice.
    target_names: list[str] = []
    for dst in mapping.values():
        if dst not in target_names:
            target_names.append(dst)

    remap: dict[int, int] = {}
    for src, dst in mapping.items():
        if src not in idx_of:
            raise SystemExit(f"'{src}' not in dataset classes {names}")
        remap[idx_of[src]] = target_names.index(dst)

    pairs: list[tuple[Path, Path]] = []
    for split in SPLITS:
        for t in _labels(root, split):
            img = next((root / split / "images").glob(t.stem + ".*"), None)
            if img:
                pairs.append((img, t))

    by_clip: dict[str, list] = defaultdict(list)
    for img, lab in pairs:
        by_clip[source_clip(img)].append((img, lab))

    clips = sorted(by_clip)
    random.Random(0).shuffle(clips)
    n_val = max(1, int(len(clips) * val_frac))
    val_clips = set(clips[:n_val])

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped_imgs = 0
    for clip in clips:
        split = "val" if clip in val_clips else "train"
        for img, lab in by_clip[clip]:
            lines = []
            for line in lab.read_text(encoding="utf-8").split("\n"):
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                if c not in remap:
                    continue
                lines.append(" ".join([str(remap[c])] + p[1:5]))
                kept[remap[c]] += 1
            if not lines:
                dropped_imgs += 1
                continue
            shutil.copy2(img, out / split / "images" / img.name)
            (out / split / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    # Absolute paths on purpose. Ultralytics resolves relative entries against the
    # yaml's own directory (not the CWD), and falls back to its global datasets_dir
    # setting -- both of which produce confusing "images not found" paths.
    (out / "data.yaml").write_text(
        f"train: {(out / 'train' / 'images').resolve().as_posix()}\n"
        f"val: {(out / 'val' / 'images').resolve().as_posix()}\n\n"
        f"nc: {len(target_names)}\n"
        f"names: {target_names}\n", encoding="utf-8")

    print(f"[prepare] {len(clips)} clips -> {len(clips)-n_val} train / {n_val} val "
          "(split by clip, so no clip spans both)")
    print(f"[prepare] instances kept: "
          f"{ {target_names[c]: n for c, n in sorted(kept.items())} }")
    print(f"[prepare] images dropped for having no kept labels: {dropped_imgs}")
    print(f"[prepare] wrote {out / 'data.yaml'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit / convert a borrowed dataset.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("root", type=Path)
    a.add_argument("--no-sheet", action="store_true")
    a.add_argument("--json", type=Path)

    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, default=C.DATASETS_DIR / "bumpers")
    p.add_argument("--map", action="append", default=[], metavar="SRC=DST",
                   help="e.g. --map red_robot=red_bumper --map robots=other_robot")
    p.add_argument("--val-frac", type=float, default=0.2)

    args = ap.parse_args(argv)
    C.ensure_dirs()

    if args.cmd == "audit":
        rep = audit(args.root, sheet=not args.no_sheet)
        if args.json:
            args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    else:
        mapping = dict(m.split("=", 1) for m in args.map)
        if not mapping:
            raise SystemExit("pass at least one --map SRC=DST")
        prepare(args.root, args.out, mapping, args.val_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
