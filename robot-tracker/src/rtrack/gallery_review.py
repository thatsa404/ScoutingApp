"""Human-reviewed, provenance-preserving appearance galleries.

This module is the safe boundary between route curation and appearance learning.  The
legacy ``reid gallery`` command learns from every solver-named detection in a match;
this module learns only from explicit human-anchor candidates and records enough source
information to rebuild or revoke every prototype.

The first slice is deliberately local and file based.  Relay transport, the Tracks-tab
UI, and replay consume the JSON artifacts produced here without changing their source of
truth.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .acquire import raw_path
from .appear import MIN_BOX
from .corrections import MATCH_PX, resolve
from .robots import drop_offfield, drop_offview
from .project import calibration_status_for

SCHEMA = 2
MANIFEST_SCHEMA = 1
BUNDLE_KIND = "galleryReviewBundle"
ANSWER_KIND = "galleryReviewAnswer"
GALLERY_KIND = "reviewed-gallery-v1"
MAX_CANDIDATES = 12          # candidate images per team
MAX_CURRENT_IMAGES = 6
MIN_TRACK_VIEWS = 3
MAX_REVIEW_BYTES = 4 * 1024 * 1024
CROP_PAD = 0.45              # match rtrack.curate._crop
CROP_MAX_H = 260
CROP_JPEG_Q = 78


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    if not path.exists():
        raise SystemExit(f"[gallery] {label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[gallery] invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"[gallery] {label} must be an object: {path}")
    return value


def manifest_path(season: int) -> Path:
    return C.TRACKER_ROOT / "gallery" / str(int(season)) / "manifest.json"


def latest_path(season: int) -> Path:
    return C.OUT_DIR / "gallery" / str(int(season)) / "latest.json"


def review_dir() -> Path:
    return C.OUT_DIR / "gallery" / "review"


def object_dir() -> Path:
    return C.OUT_DIR / "gallery" / "objects"


def load_manifest(season: int) -> dict:
    path = manifest_path(season)
    if not path.exists():
        return {"kind": "galleryManifest", "schemaVersion": MANIFEST_SCHEMA,
                "season": int(season), "decisions": [], "teamStates": {},
                "reviewedCandidateIds": []}
    doc = _read_json(path, label="gallery manifest")
    if doc.get("kind") != "galleryManifest" or doc.get("schemaVersion") not in (MANIFEST_SCHEMA, SCHEMA):
        raise SystemExit(f"[gallery] unsupported manifest schema: {path}")
    if int(doc.get("season", -1)) != int(season):
        raise SystemExit(f"[gallery] manifest season mismatch: {path}")
    return doc


def active_decisions(manifest: dict) -> list[dict]:
    """Return one active decision per candidate, with revocations applied."""
    current: dict[str, dict] = {}
    for decision in manifest.get("decisions", []):
        cid = decision.get("candidateId")
        if cid:
            current[cid] = decision
    return [d for d in current.values() if d.get("action") in ("accept", "relabel")]


def manifest_version(manifest: dict) -> str:
    active = sorted(active_decisions(manifest), key=lambda d: d.get("candidateId", ""))
    payload = {"schemaVersion": SCHEMA, "season": manifest["season"],
               "decisions": active,
               "teamStates": manifest.get("teamStates", {})}
    return _hash(payload)


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _filter_gallery_rows(match: str, rows: list[dict], *, calib_stem: str | None,
                         no_field_filter: bool, allow_unsafe_calibration: bool) -> tuple[list[dict], dict]:
    """Apply camera-view and field bounds before any gallery anchor is resolved.

    Gallery evidence is more dangerous than a route diagnostic: one foreign-camera
    robot becomes a reusable identity prototype.  Therefore missing/quarantined
    calibration fails closed unless the caller explicitly requests a diagnostic unsafe
    filter.  This mirrors the route pipeline's geometry admission policy while making
    the exception visible in the bundle provenance.
    """
    view_rows, view_dropped, view_total = drop_offview(rows, match)
    if no_field_filter:
        return view_rows, {"enabled": False, "reason": "explicit override",
                           "viewDropped": view_dropped, "viewTotal": view_total}
    camera = calib_stem or match.split("_", 1)[0]
    cal_path = C.CALIB_DIR / f"{camera}.json"
    if not cal_path.exists():
        raise SystemExit(f"[gallery] no calibration for {camera}; refusing unbounded "
                         "gallery candidates (use --no-field-filter only for diagnosis)")
    status = calibration_status_for(camera)
    if not status["usable"] and not allow_unsafe_calibration:
        detail = "; ".join(status["reasons"])
        raise SystemExit(f"[gallery] calibration {camera} is quarantined: {detail}; "
                         "refusing unbounded gallery candidates (use "
                         "--allow-unsafe-calibration only for diagnosis)")
    field_rows, field_dropped, field_total = drop_offfield(
        view_rows, match, calib_stem=camera,
        allow_unsafe_calibration=allow_unsafe_calibration)
    return field_rows, {"enabled": True, "calibration": camera,
                        "calibrationUsable": bool(status["usable"]),
                        "allowUnsafeCalibration": bool(allow_unsafe_calibration),
                        "viewDropped": view_dropped, "viewTotal": view_total,
                        "fieldDropped": field_dropped, "fieldTotal": field_total,
                        "calibrationStatus": status}


def _human_labels(corrections: dict) -> list[dict]:
    return [x for x in corrections.get("labels", [])
            if x.get("src", "human") == "human" and x.get("team")
            and not any(x.get(flag) for flag in ("mixed", "notrobot", "unknown"))]


def _anchor_map(rows: list[dict], labels: list[dict]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for item in resolve(rows, labels):
        if item.get("ok") and item.get("tid") is not None:
            result[int(item["tid"])].add(str(item["team"]))
    return result


def _appearance_rows(rows: list[dict], npz: Path,
                     *, calib_stem: str | None = None) -> list[dict]:
    z = np.load(npz, allow_pickle=False)
    tids, times = z["tid"], z["t"]
    by_tid: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i, (tid, t) in enumerate(zip(tids.tolist(), times.tolist())):
        by_tid[int(tid)].append((float(t), i))
    detections = []
    for row in rows:
        for det in row.get("dets", []):
            tid = int(det.get("tid", -1))
            if tid < 0 or tid not in by_tid:
                continue
            x1, y1, x2, y2 = (float(v) for v in det["xyxy"])
            if x2 - x1 < MIN_BOX:
                continue
            nearest = min(by_tid[tid], key=lambda pair: abs(pair[0] - float(row["t"])))
            detections.append({"tid": tid, "f": int(row["f"]), "t": float(row["t"]),
                               "xyxy": [x1, y1, x2, y2],
                               "boxArea": round((x2 - x1) * (y2 - y1), 1),
                               "boxWidth": round(x2 - x1, 1),
                               "boxHeight": round(y2 - y1, 1),
                               "embeddingIndex": nearest[1],
                               "embeddingTime": nearest[0]})
    if detections and calib_stem:
        try:
            from . import project as PJ
            H, lens = PJ.load_calib(calib_stem)
            ref = json.loads((C.CALIB_DIR / "field_ref_2026.json").read_text(encoding="utf-8"))
            field_l, field_w = ref["fieldSizeM"]
            feet = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2.0,
                              d["xyxy"][3]] for d in detections], np.float32)
            field = PJ.project_points(feet, H, ref, lens)
            for item, (x, y) in zip(detections, field):
                # Store normalized field coordinates so gallery selection remains
                # independent of the particular season's metre dimensions.
                item["fieldXY"] = [round(float(x) / field_l, 5),
                                    round(float(y) / field_w, 5)]
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            # The caller has already applied the safety filter. If a diagnostic
            # bundle has no usable projection, selection falls back to image space.
            pass
    return detections


def _choose_views(items: list[dict], limit: int) -> list[dict]:
    """Select temporally separated source views; cache embeddings add later diversity."""
    items = sorted(items, key=lambda x: (x["t"], x["f"]))
    if len(items) <= limit:
        return items
    # Deterministic farthest-in-time seed/expansion.  A later implementation can add
    # embedding-space farthest-first without changing the bundle contract.
    chosen = [items[0], items[-1]] if limit > 1 else [items[len(items) // 2]]
    while len(chosen) < limit:
        candidate = max((item for item in items if item not in chosen),
                        key=lambda item: min(abs(item["t"] - c["t"]) for c in chosen))
        chosen.append(candidate)
    return sorted(chosen, key=lambda x: (x["t"], x["f"]))


def _crop_geometry(shape: tuple[int, ...], box: list[float], *, max_width: int | None = None) -> dict:
    """Return the displayed crop size and detector box in crop pixels.

    The review image includes the same contextual padding as the curator. Keeping the
    transformed detector rectangle alongside the JPEG lets the Tracks UI draw the
    actual detection over the chicklet instead of asking the reviewer to infer it.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * CROP_PAD, bh * CROP_PAD
    a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
    c, d = min(w, int(x2 + px)), min(h, int(y2 + py))
    scale = CROP_MAX_H / (d - b) if d - b > CROP_MAX_H else 1.0
    out_w = int((c - a) * scale)
    out_h = CROP_MAX_H if scale != 1.0 else d - b
    if max_width is not None:
        out_w = min(out_w, max_width)
    return {
        "cropSize": [max(1, out_w), max(1, int(out_h))],
        "detectionBox": [round((x1 - a) * scale, 1),
                         round((y1 - b) * scale, 1),
                         round((x2 - a) * scale, 1),
                         round((y2 - b) * scale, 1)],
    }


def _encode_crop(img: np.ndarray, box: list[float]) -> bytes | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    # Gallery review is an identity decision, so show the whole detection plus the
    # same contextual buffer used by the curator.  The old appearance crop intentionally
    # showed only the superstructure and is not sufficient for a human selecting gallery
    # exemplars.
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * CROP_PAD, bh * CROP_PAD
    a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
    c, d = min(w, int(x2 + px)), min(h, int(y2 + py))
    crop = img[b:d, a:c]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    if crop.shape[0] > CROP_MAX_H:
        scale = CROP_MAX_H / crop.shape[0]
        crop = cv2.resize(crop, (max(8, int(crop.shape[1] * scale)), CROP_MAX_H),
                          interpolation=cv2.INTER_AREA)
    crop = crop[:, :420]
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_Q])
    return encoded.tobytes() if ok else None


def _object(crop: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(crop).hexdigest()
    path = object_dir() / f"{digest}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(crop)
    thumb = "data:image/jpeg;base64," + base64.b64encode(crop).decode("ascii")
    return digest, thumb


def _decode_requested(video: Path, requested: dict[int, list[dict]]) -> dict[tuple[int, int], dict]:
    if not video.exists():
        raise SystemExit(f"[gallery] source video not found: {video}")
    wanted = set(requested)
    out: dict[tuple[int, int], bytes] = {}
    cap = cv2.VideoCapture(str(video))
    frame = 0
    while wanted:
        ok, img = cap.read()
        if not ok:
            break
        if frame in wanted:
            for item in requested[frame]:
                crop = _encode_crop(img, item["xyxy"])
                if crop is not None:
                    out[(frame, int(item["tid"]))] = {
                        "bytes": crop,
                        "geometry": _crop_geometry(img.shape, item["xyxy"], max_width=420),
                    }
            wanted.remove(frame)
        frame += 1
    cap.release()
    return out


def _spread(items: list[dict], limit: int) -> list[dict]:
    """Choose deterministic evenly-spaced items from a sorted sequence."""
    if len(items) <= limit:
        return items
    indexes = np.linspace(0, len(items) - 1, limit).round().astype(int).tolist()
    return [items[i] for i in indexes]


def _curation_gallery_images(match: str, team: str, limit: int) -> list[dict]:
    """Use only crops anchored by human corrections when no gallery exists yet.

    The frame bundle also carries solver guesses in ``det.team``. Those guesses are
    useful context for the route curator, but are not authoritative gallery labels:
    using them here caused a corrected 2168 detection to be displayed as 2168 even
    though the human correction identified another robot.
    """
    path = C.STAGE3_DIR / f"{match}_curate_frames.json"
    if not path.exists():
        return []
    doc = _read_json(path, label="curation frames")
    corrections_path = C.TRACKER_ROOT / "corrections" / f"{match}_corrections.json"
    if not corrections_path.exists():
        return []
    corrections = _read_json(corrections_path, label="curation corrections")
    labels_by_frame: dict[int, list[dict]] = defaultdict(list)
    for label in _human_labels(corrections):
        labels_by_frame[int(label["f"])].append(label)
    video = raw_path(str(doc.get("video") or match))
    raw_size = None
    cap = cv2.VideoCapture(str(video)) if video.exists() else None
    if cap is not None and cap.isOpened():
        raw_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    images = []
    for frame in doc.get("frames", []):
        frame_w, frame_h = float(frame.get("w") or 0), float(frame.get("h") or 0)
        for det in frame.get("dets", []):
            if not det.get("crop"):
                continue
            xy = det.get("xy")
            labels = labels_by_frame.get(int(frame.get("f", -1)), [])
            if not isinstance(xy, (list, tuple)) or len(xy) != 2 or not labels:
                continue
            label = min(labels, key=lambda item: (
                (float(item["xy"][0]) - float(xy[0])) ** 2
                + (float(item["xy"][1]) - float(xy[1])) ** 2))
            label_distance = float(np.hypot(float(label["xy"][0]) - float(xy[0]),
                                            float(label["xy"][1]) - float(xy[1])))
            if label_distance > MATCH_PX or str(label.get("team")) != str(team):
                continue
            try:
                encoded = str(det["crop"]).split(",", 1)[1]
                crop = base64.b64decode(encoded)
                crop_img = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
            except (ValueError, TypeError, IndexError, binascii.Error):
                continue
            if crop_img is None:
                continue
            geometry = None
            if det.get("cropSize") and det.get("cropBox"):
                geometry = {"cropSize": det["cropSize"],
                            "detectionBox": det["cropBox"]}
            elif raw_size and frame_w > 0 and frame_h > 0 and det.get("box"):
                # Older curation bundles carry the box in the shipped frame but not
                # its crop transform. Reconstruct that transform from the source
                # frame dimensions, which is exact unless the crop touched a frame
                # edge (the common case is an interior robot).
                sx, sy = raw_size[0] / frame_w, raw_size[1] / frame_h
                raw_box = [float(det["box"][0]) * sx, float(det["box"][1]) * sy,
                           float(det["box"][2]) * sx, float(det["box"][3]) * sy]
                geometry = _crop_geometry((raw_size[1], raw_size[0]), raw_box)
            crop_hash, thumbnail = _object(crop)
            source = {"match": str(doc.get("match") or match),
                      "sourceTrack": det.get("tid"), "frame": frame.get("f"),
                      "time": frame.get("t"), "labelSource": "human-correction",
                      "humanTeam": str(label.get("team")),
                      "originalTeamGuess": det.get("team"),
                      "labelDistancePx": round(label_distance, 1)}
            if geometry:
                source.update(geometry)
            images.append({"imageId": crop_hash, "cropHash": crop_hash,
                           "role": "curation-reference", "thumbnail": thumbnail,
                           "source": source})
    images.sort(key=lambda x: (float(x["source"].get("time") or 0),
                               int(x["source"].get("frame") or 0),
                               str(x["imageId"])))
    return _spread(images, limit)


def _current_gallery_images(manifest: dict, team: str, limit: int,
                            *, match: str | None = None) -> list[dict]:
    """Return reviewed examples, falling back to this match's curation crops."""
    images = []
    seen = set()
    for decision in active_decisions(manifest):
        if str(decision.get("team")) != str(team):
            continue
        source = decision.get("source", {})
        for view in decision.get("views", []):
            crop_hash = view.get("cropHash")
            path = object_dir() / f"{crop_hash}.jpg" if crop_hash else None
            if not crop_hash or crop_hash in seen or not path.exists():
                continue
            seen.add(crop_hash)
            images.append({
                "imageId": crop_hash,
                "cropHash": crop_hash,
                "role": "current",
                "thumbnail": "data:image/jpeg;base64," +
                             base64.b64encode(path.read_bytes()).decode("ascii"),
                "source": {"match": source.get("match"),
                           "sourceTrack": source.get("sourceTrack"),
                           "frame": view.get("frame"), "time": view.get("time"),
                           "cropSize": view.get("cropSize") or source.get("cropSize"),
                           "detectionBox": (view.get("detectionBox")
                                            or source.get("detectionBox"))},
            })
    images.sort(key=lambda x: (str(x["source"].get("match", "")),
                               float(x["source"].get("time") or 0), x["imageId"]))
    if images:
        return _spread(images, limit)
    return _curation_gallery_images(match, team, limit) if match else []


def _candidate_images(items: list[dict], limit: int) -> list[dict]:
    """Cover time and field area first; use detection size as the tie-breaker.

    Gallery review should teach appearance across viewpoints, not just collect the
    largest detections. The greedy max-min pass therefore spreads selected images in
    normalized time and calibrated field coordinates before box area is consulted.
    """
    if not items:
        return []
    items = sorted(items, key=lambda x: (x["t"], x["f"], x["tid"]))
    if len(items) <= limit:
        return items
    lo_t, hi_t = float(items[0]["t"]), float(items[-1]["t"])
    duration = max(hi_t - lo_t, 1e-6)

    def point(item: dict) -> tuple[float, float]:
        field = item.get("fieldXY")
        if isinstance(field, (list, tuple)) and len(field) == 2:
            return float(field[0]), float(field[1])
        # Diagnostic/un-calibrated bundles still get image-space diversity, but the
        # calibrated path above is the normal source of this feature.
        x1, y1, x2, y2 = item.get("xyxy", [0.0, 0.0, 0.0, 0.0])
        return (float(x1 + x2) / (2 * 1920.0),
                float(y1 + y2) / (2 * 1080.0))

    def time_coord(item: dict) -> float:
        return (float(item["t"]) - lo_t) / duration

    def distance(a: dict, b: dict) -> float:
        ax, ay = point(a)
        bx, by = point(b)
        return float(np.hypot(ax - bx, ay - by))

    # Time endpoints are guaranteed anchors for the first two slots. Subsequent
    # choices maximize the least-covered time/field distance; area is only consulted
    # after the coverage score is tied.
    chosen: list[dict] = []
    used = set()
    chosen.append(items[0])
    used.add((items[0]["f"], items[0]["tid"]))
    if limit > 1:
        last = items[-1]
        chosen.append(last)
        used.add((last["f"], last["tid"]))
    max_area = max(x["boxArea"] for x in items) or 1.0
    while len(chosen) < limit:
        remaining = [x for x in items if (x["f"], x["tid"]) not in used]
        if not remaining:
            break
        def score(item: dict) -> tuple[float, float, float, float, int]:
            time_gap = min(abs(time_coord(item) - time_coord(c)) for c in chosen)
            field_gap = min(distance(item, c) for c in chosen)
            coverage = 0.5 * time_gap + 0.5 * min(field_gap / float(np.sqrt(2)), 1.0)
            area = item["boxArea"] / max_area
            track_bonus = 1.0 if not any(c["tid"] == item["tid"] for c in chosen) else 0.0
            return (coverage, time_gap, field_gap, area + 0.001 * track_bonus, -item["f"])
        best = max(remaining, key=score)
        chosen.append(best)
        used.add((best["f"], best["tid"]))
    return chosen


def _candidate_id(item: dict, *, season: int, match: str, team: str,
                  tracks_source: str, appearance_source: str) -> str:
    return _hash({"season": season, "match": match, "team": team,
                  "track": item["tid"], "frame": item["f"],
                  "tracks": tracks_source, "appearance": appearance_source})


def _unreviewed_candidates(items: list[dict], reviewed: set[str], *, season: int,
                           match: str, team: str, tracks_source: str,
                           appearance_source: str) -> list[dict]:
    return [item for item in items
            if _candidate_id(item, season=season, match=match, team=team,
                             tracks_source=tracks_source,
                             appearance_source=appearance_source) not in reviewed]


def prepare(match: str, *, season: int, corrections_path: Path | None = None,
            max_candidates: int = MAX_CANDIDATES,
            current_images: int = MAX_CURRENT_IMAGES,
            no_thumbnails: bool = False, calib_stem: str | None = None,
            no_field_filter: bool = False,
            allow_unsafe_calibration: bool = False) -> Path:
    rows_path = C.STAGE1_DIR / f"{match}_tracks_stitched.jsonl"
    npz_path = C.STAGE3_DIR / f"{match}_appearance_cnn.npz"
    corr_path = corrections_path or C.TRACKER_ROOT / "corrections" / f"{match}_corrections.json"
    rows = _load_rows(rows_path)
    rows, field_filter = _filter_gallery_rows(
        match, rows, calib_stem=calib_stem, no_field_filter=no_field_filter,
        allow_unsafe_calibration=allow_unsafe_calibration)
    corrections = _read_json(corr_path, label="corrections")
    anchors = _anchor_map(rows, _human_labels(corrections))
    if not anchors:
        raise SystemExit("[gallery] no human team anchors were resolved")
    appearances = _appearance_rows(
        rows, npz_path, calib_stem=calib_stem or match.split("_", 1)[0])
    by_tid: dict[int, list[dict]] = defaultdict(list)
    by_team: dict[str, list[dict]] = defaultdict(list)
    for item in appearances:
        by_tid[item["tid"]].append(item)
    for tid, team_set in anchors.items():
        if len(team_set) == 1:
            by_team[next(iter(team_set))].extend(by_tid.get(tid, []))

    track_sources = _file_hash(rows_path)
    appearance_source = _file_hash(npz_path)
    manifest = load_manifest(season)
    version = manifest_version(manifest)
    reviewed = {str(value) for value in manifest.get("reviewedCandidateIds", [])}
    # Accepted candidates from older manifests predate the reviewed-candidate
    # ledger. Treat them as reviewed so they are not reissued on the next bundle.
    reviewed.update(str(d.get("candidateId")) for d in active_decisions(manifest)
                    if d.get("candidateId"))
    requested: dict[int, list[dict]] = defaultdict(list)
    groups: list[dict] = []
    for team in sorted(by_team):
        eligible = _unreviewed_candidates(
            by_team[team], reviewed, season=season, match=match, team=team,
            tracks_source=track_sources, appearance_source=appearance_source)
        selected = _candidate_images(eligible, max_candidates)
        candidates = []
        for item in selected:
            candidate_id = _candidate_id(item, season=season, match=match, team=team,
                                         tracks_source=track_sources,
                                         appearance_source=appearance_source)
            candidates.append({"candidateId": candidate_id, "team": team,
                               "role": "candidate",
                               "cropHash": None, "thumbnail": None,
                               "source": {"match": match, "video": match,
                                          "sourceTrack": item["tid"], "frame": item["f"],
                                          "time": round(item["t"], 3), "xyxy": item["xyxy"],
                                          "fieldXY": item.get("fieldXY"),
                                          "boxArea": item["boxArea"],
                                          "boxWidth": item["boxWidth"],
                                          "boxHeight": item["boxHeight"],
                                          "tracksSha256": track_sources,
                                          "appearanceSha256": appearance_source},
                               "quality": {"boxArea": item["boxArea"],
                                           "boxWidth": item["boxWidth"],
                                           "boxHeight": item["boxHeight"],
                                           "padding": CROP_PAD},
                               "embeddingIndex": item["embeddingIndex"]})
            requested[item["f"]].append(item)
        groups.append({"team": team,
                       "currentGallery": _current_gallery_images(
                           manifest, team, current_images, match=match),
                       "candidates": candidates})

    crops = {} if no_thumbnails else _decode_requested(raw_path(match), requested)
    nonempty_groups = []
    for group in groups:
        usable = []
        for candidate in group["candidates"]:
            source = candidate["source"]
            crop_record = crops.get((int(source["frame"]), int(source["sourceTrack"])))
            if crop_record is None:
                continue
            crop_hash, thumbnail = _object(crop_record["bytes"])
            source.update(crop_record["geometry"])
            candidate["cropHash"] = crop_hash
            candidate["imageId"] = crop_hash
            candidate["thumbnail"] = thumbnail
            usable.append(candidate)
        group["candidates"] = usable
        if group["currentGallery"] or usable:
            nonempty_groups.append(group)

    body = {"kind": BUNDLE_KIND, "schemaVersion": SCHEMA,
            "reviewId": _hash({"season": season,
                                "fieldFilter": field_filter,
                                "teams": [{"team": g["team"],
                                           "current": [x["imageId"] for x in g["currentGallery"]],
                                           "candidates": [x["candidateId"] for x in g["candidates"]]}
                                          for g in nonempty_groups]}),
            "season": int(season), "match": match, "createdAt": _now(),
            "embeddingSpace": "resnet18-imagenet-v1/raw", "galleryVersion": version,
            "fieldFilter": field_filter,
            "limits": {"currentImagesPerTeam": current_images,
                        "candidateImagesPerTeam": max_candidates,
                        "cropPadding": CROP_PAD,
                        "candidateSelection": "time-field-coverage-then-box-area-v1"},
            "teams": nonempty_groups}
    encoded = _canonical(body)
    if len(encoded) > MAX_REVIEW_BYTES:
        raise SystemExit(f"[gallery] review bundle is {len(encoded) / 1e6:.2f} MB; reduce images per team")
    body["bundleHash"] = _hash(encoded)
    out = review_dir() / f"{body['reviewId']}.json"
    _save_json(out, body)
    print(f"[gallery] {len(nonempty_groups)} team(s), "
          f"{sum(len(g['currentGallery']) for g in nonempty_groups)} current + "
          f"{sum(len(g['candidates']) for g in nonempty_groups)} candidate image(s) -> {out}")
    return out


def validate_answer(answer: dict, bundle: dict) -> None:
    if answer.get("kind") != ANSWER_KIND or answer.get("schemaVersion") != SCHEMA:
        raise SystemExit("[gallery] unsupported answer schema")
    if answer.get("reviewId") != bundle.get("reviewId"):
        raise SystemExit("[gallery] answer reviewId does not match bundle")
    expected = bundle.get("bundleHash")
    if answer.get("bundleHash") != expected:
        raise SystemExit("[gallery] answer bundleHash is stale or missing")
    allowed: dict[str, set[str]] = {}
    for group in bundle.get("teams", []):
        team = str(group.get("team", ""))
        if not team or team in allowed:
            raise SystemExit(f"[gallery] invalid or duplicate team group: {team}")
        allowed[team] = {str(c.get("cropHash")) for c in group.get("candidates", [])
                         if c.get("cropHash")}
    seen = set()
    for selection in answer.get("selections", []):
        team = str(selection.get("team", ""))
        if team not in allowed or team in seen:
            raise SystemExit(f"[gallery] invalid or duplicate team selection: {team}")
        seen.add(team)
        include = selection.get("include", [])
        if not isinstance(include, list) or len(include) != len(set(include)):
            raise SystemExit(f"[gallery] duplicate candidate image in team selection: {team}")
        if not set(map(str, include)) <= allowed[team]:
            raise SystemExit(f"[gallery] selected image is not in bundle for team {team}")
        if "reviewed" in selection:
            reviewed = selection["reviewed"]
            if not isinstance(reviewed, list) or len(reviewed) != len(set(reviewed)):
                raise SystemExit(f"[gallery] duplicate reviewed candidate in team selection: {team}")
            candidate_ids = {str(c.get("candidateId"))
                             for group in bundle.get("teams", [])
                             if str(group.get("team")) == team
                             for c in group.get("candidates", []) if c.get("candidateId")}
            if not set(map(str, reviewed)) <= candidate_ids:
                raise SystemExit(f"[gallery] reviewed image is not in bundle for team {team}")
    states = {"needs-more", "sufficient", "robot-changed"}
    for state in answer.get("teamStates", []):
        if str(state.get("team")) not in allowed or state.get("state") not in states:
            raise SystemExit(f"[gallery] invalid team state: {state}")


def _append_answer(answer: dict, bundle: dict, manifest: dict, stamp: str) -> int:
    """Append one validated answer to an already version-checked manifest."""
    rid = str(answer["reviewId"])
    allowed = {str(c["cropHash"]): c
               for group in bundle.get("teams", [])
               for c in group.get("candidates", []) if c.get("cropHash")}
    selected = 0
    reviewed_ids = manifest.setdefault("reviewedCandidateIds", [])
    reviewed_set = {str(value) for value in reviewed_ids}
    all_candidate_ids = {str(c.get("candidateId"))
                         for group in bundle.get("teams", [])
                         for c in group.get("candidates", []) if c.get("candidateId")}
    for selection in answer.get("selections", []):
        team = str(selection["team"])
        for candidate_id in selection.get("reviewed", selection.get("include", [])):
            # The UI sends candidate IDs here. Older answers sent crop hashes in
            # include only, so those are added below when they resolve to a candidate.
            if str(candidate_id) in all_candidate_ids:
                reviewed_set.add(str(candidate_id))
        for crop_hash in selection.get("include", []):
            candidate = allowed[str(crop_hash)]
            reviewed_set.add(str(candidate["candidateId"]))
            source = candidate.get("source", {})
            # One selected image becomes one independently revocable prototype. The
            # source crop is retained without its transient base64 thumbnail.
            record = {"candidateId": candidate["candidateId"], "action": "accept",
                      "team": team, "acceptedViewHashes": [str(crop_hash)],
                      "source": source,
                      "views": [{"cropHash": str(crop_hash),
                                 "frame": source.get("frame"), "time": source.get("time"),
                                 "xyxy": source.get("xyxy"),
                                 "boxArea": source.get("boxArea"),
                                 "cropSize": source.get("cropSize"),
                                 "detectionBox": source.get("detectionBox"),
                                 "padding": CROP_PAD}],
                      "revisionId": "r1", "reviewId": rid,
                      "bundleHash": bundle["bundleHash"],
                      "baseGalleryVersion": answer.get("baseGalleryVersion")
                      or manifest_version(manifest),
                      "reviewedAt": stamp, "reviewer": answer.get("reviewer")}
            manifest.setdefault("decisions", []).append(record)
            selected += 1
    manifest["reviewedCandidateIds"] = sorted(reviewed_set)
    for state in answer.get("teamStates", []):
        team = str(state.get("team", ""))
        if team and state.get("state") in {"needs-more", "sufficient", "robot-changed"}:
            manifest.setdefault("teamStates", {})[team] = dict(state, updatedAt=stamp)
    return selected


def apply_answers(answer_paths: list[Path]) -> Path:
    """Apply independent reviews created from one gallery snapshot as one commit."""
    if not answer_paths:
        raise SystemExit("[gallery] no gallery answers supplied")
    pairs = []
    for answer_path in answer_paths:
        answer = _read_json(answer_path, label="gallery answer")
        rid = answer.get("reviewId")
        if not rid:
            raise SystemExit("[gallery] answer has no reviewId")
        bundle = _read_json(review_dir() / f"{rid}.json", label="gallery bundle")
        validate_answer(answer, bundle)
        pairs.append((answer, bundle))
    season = int(pairs[0][1]["season"])
    if any(int(bundle["season"]) != season for _, bundle in pairs):
        raise SystemExit("[gallery] batch answers span multiple seasons")
    manifest = load_manifest(season)
    base = manifest_version(manifest)
    for answer, _ in pairs:
        if answer.get("baseGalleryVersion") and answer["baseGalleryVersion"] != base:
            raise SystemExit("[gallery] batch answers were not based on the same current gallery")
    stamp = _now()
    selected = sum(_append_answer(answer, bundle, manifest, stamp)
                   for answer, bundle in pairs)
    manifest["schemaVersion"] = MANIFEST_SCHEMA
    path = manifest_path(season)
    _save_json(path, manifest)
    print(f"[gallery] applied {selected} selected image(s) from {len(pairs)} answer(s) -> {path}")
    return path


def apply_answer(answer_path: Path, *, bundle_path: Path | None = None) -> Path:
    """Apply one answer, retaining the original single-answer CLI behavior."""
    answer = _read_json(answer_path, label="gallery answer")
    rid = answer.get("reviewId")
    if not rid:
        raise SystemExit("[gallery] answer has no reviewId")
    resolved = bundle_path or review_dir() / f"{rid}.json"
    bundle = _read_json(resolved, label="gallery bundle")
    validate_answer(answer, bundle)
    season = int(bundle["season"])
    manifest = load_manifest(season)
    base = answer.get("baseGalleryVersion")
    if base and base != manifest_version(manifest):
        raise SystemExit("[gallery] answer was based on an older gallery; regenerate the bundle")
    stamp = _now()
    selected = _append_answer(answer, bundle, manifest, stamp)
    manifest["schemaVersion"] = MANIFEST_SCHEMA
    path = manifest_path(season)
    _save_json(path, manifest)
    print(f"[gallery] applied {selected} selected image(s) -> {path}")
    return path


def _source_embedding(match: str, tid: int, at: float) -> np.ndarray | None:
    path = C.STAGE3_DIR / f"{match}_appearance_cnn.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    indexes = np.flatnonzero(z["tid"] == tid)
    if not len(indexes):
        return None
    i = int(indexes[np.argmin(np.abs(z["t"][indexes] - at))])
    return np.asarray(z["feat"][i], dtype=np.float32)


def rebuild(season: int) -> Path:
    manifest = load_manifest(season)
    decisions = active_decisions(manifest)
    previous_version = None
    previous_latest = latest_path(season)
    if previous_latest.exists():
        try:
            previous_version = str(_read_json(previous_latest, label="gallery latest").get("version"))
        except SystemExit:
            previous_version = None
    embeddings, metadata = [], []
    for decision in decisions:
        source = decision.get("source", {})
        match, tid = source.get("match"), source.get("sourceTrack")
        if not match or tid is None:
            continue
        for view_hash in decision.get("acceptedViewHashes", []):
            # The bundle retains frame/time provenance; find the selected view without
            # making the manifest depend on transient base64 thumbnails.
            view = next((v for v in decision.get("views", [])
                         if v.get("cropHash") == view_hash), None)
            if view is None:
                # Answers from the first UI slice may keep views in the bundle only.
                bundle = review_dir() / f"{decision['reviewId']}.json"
                if bundle.exists():
                    b = _read_json(bundle, label="gallery bundle")
                    if b.get("schemaVersion") == 2:
                        pool = [c for group in b.get("teams", [])
                                for c in group.get("candidates", [])]
                        c = next((c for c in pool if c.get("cropHash") == view_hash), {})
                        source = c.get("source", {})
                        view = {"cropHash": view_hash, "time": source.get("time"),
                                "frame": source.get("frame")} if c else None
                    else:
                        c = next((c for c in b.get("candidates", [])
                                  if c.get("candidateId") == decision.get("candidateId")), {})
                        view = next((v for v in c.get("views", [])
                                     if v.get("cropHash") == view_hash), None)
            if view is None:
                continue
            feat = _source_embedding(match, int(tid), float(view.get("time", 0)))
            if feat is None:
                continue
            embeddings.append(feat)
            metadata.append((view_hash, f"{season}:{decision.get('team')}",
                             decision.get("revisionId", "r1"), match, int(tid),
                             float(view.get("time", 0))))
    if embeddings:
        matrix = np.stack(embeddings).astype(np.float32)
    else:
        matrix = np.zeros((0, 512), np.float32)
    version = manifest_version(manifest)
    out_dir = C.OUT_DIR / "gallery" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{version}.npz"
    np.savez_compressed(
        out, prototypeId=np.array([m[0] for m in metadata]),
        seasonTeam=np.array([m[1] for m in metadata]),
        revisionId=np.array([m[2] for m in metadata]),
        embedding=matrix, sourceMatch=np.array([m[3] for m in metadata]),
        sourceTrack=np.array([m[4] for m in metadata], np.int32),
        sourceTime=np.array([m[5] for m in metadata], np.float32),
        manifestVersion=np.array(version), embeddingSpace=np.array("resnet18-imagenet-v1/raw"),
        schemaVersion=np.array(1, np.int16), kind=np.array(GALLERY_KIND))
    latest = {"season": season, "version": version, "path": str(out),
              "prototypeCount": len(metadata), "manifestVersion": version,
              "embeddingSpace": "resnet18-imagenet-v1/raw", "updatedAt": _now()}
    _save_json(latest_path(season), latest)
    if previous_version and previous_version != version:
        # Import lazily to keep the review manifest/rebuild path usable without
        # importing the vote implementation during ordinary bundle preparation.
        from .gallery_replay import enqueue
        enqueue(season, previous_version, version,
                sorted({str(d.get("team")) for d in decisions if d.get("team")}))
    print(f"[gallery] rebuilt {len(metadata)} prototype(s) -> {out}")
    return out


def status(season: int, team: str | None = None) -> None:
    manifest = load_manifest(season)
    rows = Counter()
    for d in active_decisions(manifest):
        key = str(d.get("team", ""))
        if key and (team is None or key == team):
            rows[key] += len(d.get("acceptedViewHashes", []))
    print(json.dumps({"season": season, "version": manifest_version(manifest),
                      "teams": dict(rows),
                      "states": manifest.get("teamStates", {})}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Human-reviewed appearance gallery workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("match")
    p.add_argument("--season", type=int, default=C.YEAR)
    p.add_argument("--corrections", type=Path)
    p.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                   help="candidate images per team")
    p.add_argument("--current-images", type=int, default=MAX_CURRENT_IMAGES,
                   help="current gallery reference images per team")
    p.add_argument("--calib-from", default=None, metavar="VIDEO",
                   help="camera calibration stem; defaults to the event prefix")
    p.add_argument("--no-field-filter", action="store_true",
                   help="diagnostic override; do not publish unbounded results")
    p.add_argument("--allow-unsafe-calibration", action="store_true",
                   help="diagnostic override; use a quarantined calibration explicitly")
    p.add_argument("--no-thumbnails", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("answer", type=Path)
    p.add_argument("--bundle", type=Path)
    p = sub.add_parser("apply-batch")
    p.add_argument("answers", type=Path, nargs="+",
                   help="independent answers based on the same gallery version")
    p = sub.add_parser("rebuild")
    p.add_argument("--season", type=int, default=C.YEAR)
    p = sub.add_parser("status")
    p.add_argument("--season", type=int, default=C.YEAR)
    p.add_argument("--team")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.match, season=args.season, corrections_path=args.corrections,
                max_candidates=args.max_candidates, current_images=args.current_images,
                no_thumbnails=args.no_thumbnails, calib_stem=args.calib_from,
                no_field_filter=args.no_field_filter,
                allow_unsafe_calibration=args.allow_unsafe_calibration)
    elif args.cmd == "apply":
        apply_answer(args.answer, bundle_path=args.bundle)
    elif args.cmd == "apply-batch":
        apply_answers(args.answers)
    elif args.cmd == "rebuild":
        rebuild(args.season)
    elif args.cmd == "status":
        status(args.season, args.team)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
