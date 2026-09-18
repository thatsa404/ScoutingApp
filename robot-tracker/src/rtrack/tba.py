"""The Blue Alliance client -- mirrors the app's fetchTBA helper (main.js:735).

Reads VITE_TBA_KEY from the repo-root .env. The key is never copied into this
subproject; there is one source of truth and it is already gitignored.

    uv run -m rtrack.tba GSxbsE42o5o --event 2026necmp
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import requests

from . import config as C

TBA_BASE = "https://www.thebluealliance.com/api/v3"


@lru_cache(maxsize=1)
def _key() -> str:
    if not C.ENV_FILE.exists():
        raise RuntimeError(f"{C.ENV_FILE} not found")
    for line in C.ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("VITE_TBA_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"VITE_TBA_KEY not set in {C.ENV_FILE}")


def fetch(endpoint: str) -> object:
    """GET an endpoint. Same auth-header pattern as the app's fetchTBA()."""
    r = requests.get(
        f"{TBA_BASE}/{endpoint.lstrip('/')}",
        headers={"X-TBA-Auth-Key": _key()},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def team_events(team: int, year: int = C.YEAR) -> list[dict]:
    return fetch(f"team/frc{team}/events/{year}/simple")


def event_matches(event_key: str) -> list[dict]:
    return fetch(f"event/{event_key}/matches/simple")


def _teams(alliance: dict) -> list[int]:
    return [int(k.removeprefix("frc")) for k in alliance["team_keys"]]


def _shape(m: dict, event_key: str) -> dict:
    return {
        "key": m["key"],
        "eventKey": event_key,
        "compLevel": m["comp_level"],
        "setNumber": m.get("set_number"),
        "matchNumber": m.get("match_number"),
        "red": _teams(m["alliances"]["red"]),
        "blue": _teams(m["alliances"]["blue"]),
        "redScore": m["alliances"]["red"].get("score"),
        "blueScore": m["alliances"]["blue"].get("score"),
        "actualTime": m.get("actual_time"),
    }


def match_by_key(match_key: str) -> dict:
    """Direct lookup, e.g. '2026necmp_f1m3'.

    The realistic path in practice: TBA's videos[] is empty for a great many
    events (the whole 2026 New England championship included), so resolving a
    clip by its YouTube id often cannot work. Identifying the match by eye from
    the broadcast score bug and passing the key here always does.
    """
    m = fetch(f"match/{match_key}")
    return _shape(m, match_key.split("_", 1)[0])


def match_for_video(vid: str, event_keys: list[str]) -> dict | None:
    """Find the match whose videos[] contains this YouTube id.

    The app's fetchSchedule (main.js:762) filters to comp_level == 'qm', so playoff
    matches -- including the Final Tiebreaker test clip -- are never in Dexie. Going
    to TBA directly is both simpler and always correct.
    """
    for ek in event_keys:
        try:
            matches = event_matches(ek)
        except requests.HTTPError as e:
            print(f"[tba] {ek}: {e}")
            continue
        linked = 0
        for m in matches:
            keys = [v.get("key") for v in (m.get("videos") or [])
                    if v.get("type") == "youtube"]
            linked += bool(keys)
            if vid in keys:
                return _shape(m, ek)
        if linked == 0:
            print(f"[tba] {ek}: none of {len(matches)} matches have a linked video "
                  f"-- reverse lookup cannot work here; use --match <key> instead")
    return None


def search_events(year: int = C.YEAR, name_contains: str = "") -> list[dict]:
    """Event keys for a year, optionally filtered by name -- for locating a clip."""
    events = fetch(f"events/{year}/simple")
    needle = name_contains.lower()
    return [
        {"key": e["key"], "name": e["name"], "start": e.get("start_date")}
        for e in events
        if not needle or needle in e["name"].lower()
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve a video id to a TBA match.")
    ap.add_argument("video_id", nargs="?", help="11-char YouTube id")
    ap.add_argument("--event", action="append", default=[], dest="events",
                    help="event key to scan (repeatable)")
    ap.add_argument("--find-event", metavar="SUBSTR",
                    help="list event keys whose name contains SUBSTR, then exit")
    ap.add_argument("--match", metavar="KEY",
                    help="look the match up directly, e.g. 2026necmp_f1m3")
    ap.add_argument("--year", type=int, default=C.YEAR)
    args = ap.parse_args(argv)

    if args.find_event:
        for e in search_events(args.year, args.find_event):
            print(f"{e['key']:<14} {e['start']}  {e['name']}")
        return 0

    if args.match:
        print(json.dumps(match_by_key(args.match), indent=2))
        return 0

    if not args.video_id:
        ap.error("video_id is required unless --find-event or --match is given")
    if not args.events:
        ap.error("pass at least one --event (find them with --find-event)")

    match = match_for_video(args.video_id, args.events)
    if match is None:
        print(f"No match in {args.events} lists video {args.video_id}.")
        return 1
    print(json.dumps(match, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
