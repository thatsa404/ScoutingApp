"""Dependency-aware replay for reviewed appearance-gallery revisions.

Replay is intentionally conservative.  A new gallery version may recompute cached
appearance votes, but route changes are reports for operator review until the replay
policy has been validated against the audited data.  Queue state is local and small;
the relay carries only bounded status summaries, never the season gallery or all old
match media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .reid import vote_tracks


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replay_dir() -> Path:
    return C.OUT_DIR / "gallery" / "replay"


def queue_path() -> Path:
    return replay_dir() / "queue.json"


def _load_queue() -> dict:
    if not queue_path().exists():
        return {"schemaVersion": 1, "entries": [], "updatedAt": _now()}
    try:
        doc = json.loads(queue_path().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[replay] invalid queue: {queue_path()}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("entries", []), list):
        raise SystemExit(f"[replay] malformed queue: {queue_path()}")
    return doc


def _save_queue(doc: dict) -> None:
    replay_dir().mkdir(parents=True, exist_ok=True)
    doc["updatedAt"] = _now()
    queue_path().write_text(json.dumps(doc, indent=1), encoding="utf-8")


def _run_manifests(season: int) -> list[tuple[str, Path, dict]]:
    result = []
    for path in sorted(C.STAGE3_DIR.glob("*_run.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        match = str(doc.get("match") or doc.get("video") or path.stem[:-4])
        if not match.startswith(str(int(season))):
            continue
        result.append((match, path, doc))
    return result


def enqueue(season: int, old: str, new: str, changed_teams: list[str] | None = None) -> int:
    if not old or not new or old == new:
        return 0
    doc = _load_queue()
    existing = {(str(e.get("match")), str(e.get("newVersion"))) for e in doc["entries"]}
    added = 0
    for match, manifest_path, run in _run_manifests(season):
        key = (match, new)
        if key in existing:
            continue
        doc["entries"].append({
            "match": match,
            "runManifest": str(manifest_path),
            "oldVersion": old,
            "newVersion": new,
            "changedTeams": sorted(set(changed_teams or [])),
            "state": "queued",
            "attempts": 0,
            "createdAt": _now(),
            "updatedAt": _now(),
        })
        existing.add(key)
        added += 1
    _save_queue(doc)
    print(f"[replay] queued {added} match(es) for {old[:12]} -> {new[:12]}")
    return added


def _resolve_recorded(path_value: str, run_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    # Run manifests store paths relative to robot-tracker, while the manifest itself
    # lives in out/stage3.  Prefer the recorded project-relative interpretation.
    candidate = C.TRACKER_ROOT / path
    if candidate.exists():
        return candidate
    return run_path.parent / path


def _inputs_match(run_path: Path, run: dict) -> tuple[bool, list[str]]:
    errors = []
    for name, info in (run.get("inputs") or {}).items():
        if not isinstance(info, dict) or not info.get("path") or not info.get("sha256"):
            continue
        path = _resolve_recorded(str(info["path"]), run_path)
        if not path.exists():
            errors.append(f"{name}: missing {path}")
            continue
        if _sha256(path) != info["sha256"]:
            errors.append(f"{name}: hash changed ({path})")
    return not errors, errors


def _event_for(match: str) -> str:
    return match.rsplit("_qm", 1)[0] if "_qm" in match else match


def diff_match(match: str, version: str) -> Path:
    run_path = C.STAGE3_DIR / f"{match}_run.json"
    if not run_path.exists():
        raise SystemExit(f"[replay] run manifest not found: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    ok, errors = _inputs_match(run_path, run)
    report = {"schemaVersion": 1, "kind": "galleryReplayReport", "match": match,
              "newVersion": version, "createdAt": _now(), "reproducible": ok,
              "state": "stale" if not ok else "vote-diffing", "inputErrors": errors}
    if not ok:
        return _write_report(match, version, report)

    identity_path = (run.get("inputs", {}).get("identity", {}) or {}).get("path")
    tracks_path = (run.get("inputs", {}).get("tracks", {}) or {}).get("path")
    if not identity_path or not tracks_path:
        report.update(state="stale", inputErrors=["run manifest lacks identity/tracks inputs"])
        return _write_report(match, version, report)
    identity = _resolve_recorded(str(identity_path), run_path)
    tracks = _resolve_recorded(str(tracks_path), run_path)
    if not identity.exists() or not tracks.exists():
        report.update(state="stale", inputErrors=["recorded identity or tracks file missing"])
        return _write_report(match, version, report)
    try:
        old_doc = json.loads(identity.read_text(encoding="utf-8"))
        teams = [str(t) for t in old_doc.get("teams", [])]
        if not teams:
            report.update(state="unchanged", reason="old vote document has no team set")
            return _write_report(match, version, report)
        backend = "cnn" if identity.name.endswith("_cnn.json") else "hist"
        new_doc = vote_tracks(match, tracks, _event_for(match), teams, backend=backend)
    except Exception as exc:  # replay should report a failed item, not kill the queue
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        return _write_report(match, version, report)
    if new_doc is None:
        report.update(state="unchanged", reason="new gallery has no coverage for this match")
        return _write_report(match, version, report)

    old_tracks = old_doc.get("tracks", {})
    changed = []
    for tid in sorted(set(old_tracks) | set(new_doc.get("tracks", {})), key=str):
        old_vote, new_vote = old_tracks.get(tid, {}), new_doc.get("tracks", {}).get(tid, {})
        fields = ("team", "share", "margin", "votes")
        delta = {f: [old_vote.get(f), new_vote.get(f)] for f in fields
                 if old_vote.get(f) != new_vote.get(f)}
        if delta:
            changed.append({"track": tid, "changes": delta})
    report.update({"state": "needs-review" if changed else "unchanged",
                   "changedTracks": changed, "changedTrackCount": len(changed),
                   "oldVotePath": str(identity), "newGalleryVersion": version,
                   "solverRun": False,
                   "policy": "route changes require explicit operator acceptance"})
    return _write_report(match, version, report)


def _write_report(match: str, version: str, report: dict) -> Path:
    path = replay_dir() / f"{match}_{version[:16]}.json"
    replay_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return path


def run(limit: int | None = None) -> None:
    doc = _load_queue()
    done = 0
    for entry in doc["entries"]:
        if entry.get("state") != "queued":
            continue
        if limit is not None and done >= limit:
            break
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["state"] = "vote-diffing"
        entry["updatedAt"] = _now()
        _save_queue(doc)
        report_path = diff_match(str(entry["match"]), str(entry["newVersion"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry["state"] = report.get("state", "failed")
        entry["report"] = str(report_path)
        entry["updatedAt"] = _now()
        done += 1
    _save_queue(doc)
    print(f"[replay] processed {done} queued match(es)")


def status(season: int | None = None) -> None:
    entries = _load_queue()["entries"]
    if season is not None:
        entries = [e for e in entries if str(e.get("match", "")).startswith(str(season))]
    counts = Counter(str(e.get("state", "unknown")) for e in entries)
    print(json.dumps({"season": season, "counts": dict(counts),
                      "entries": entries[-20:]}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay matches affected by gallery revisions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("enqueue")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--from", dest="old", required=True)
    p.add_argument("--to", dest="new", required=True)
    p.add_argument("--team", action="append", default=[])
    p = sub.add_parser("run")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("status")
    p.add_argument("--season", type=int)
    p = sub.add_parser("diff")
    p.add_argument("match")
    p.add_argument("--version", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "enqueue":
        enqueue(args.season, args.old, args.new, args.team)
    elif args.cmd == "run":
        run(args.limit)
    elif args.cmd == "status":
        status(args.season)
    elif args.cmd == "diff":
        print(diff_match(args.match, args.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
