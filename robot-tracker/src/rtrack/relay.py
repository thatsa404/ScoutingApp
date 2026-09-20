"""Stage 4 -- talk to the rtrack relay.

    uv run -m rtrack.relay push-bundle 2026mawor_qm1
    uv run -m rtrack.relay wait-answer 2026mawor_qm1 --timeout 900
    uv run -m rtrack.relay push-calib 2026mawor_qm1
    uv run -m rtrack.relay wait-points 2026mawor_qm1
    uv run -m rtrack.relay wait-occl WFj_FsFQRkM            # occluders drawn on a phone

The home machine has the GPU, the video and the pipeline; the human is at an event with
a phone. Neither can reach the other -- residential NAT one side, venue wifi the other --
so a Cloudflare Worker sits in the middle (see rtrack-relay/worker.js). This module is
the home machine's end of that.

Configuration comes from the repo .env, same file the app reads:

    RTRACK_RELAY_URL=https://rtrack-relay.<you>.workers.dev
    RTRACK_TOKEN=<the same secret set with `wrangler secret put`>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

from . import config as C

DEFAULT_TIMEOUT_S = 900.0
POLL_S = 5.0


def _env() -> tuple[str, str]:
    """(url, token) from the repo .env, falling back to the process environment."""
    url = os.environ.get("RTRACK_RELAY_URL", "")
    tok = os.environ.get("RTRACK_TOKEN", "")
    envf = C.REPO_ROOT / ".env"
    if envf.exists() and not (url and tok):
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() == "RTRACK_RELAY_URL" and not url:
                url = v
            elif k.strip() == "RTRACK_TOKEN" and not tok:
                tok = v
    if not url:
        raise SystemExit(
            "[relay] RTRACK_RELAY_URL is not set.\n"
            "        Add it to .env after deploying rtrack-relay/:\n"
            "          RTRACK_RELAY_URL=https://rtrack-relay.<you>.workers.dev\n"
            "          RTRACK_TOKEN=<the wrangler secret>")
    return url.rstrip("/"), tok


def put(kind: str, ident: str, doc: dict) -> dict:
    url, tok = _env()
    body = json.dumps(doc, separators=(",", ":"))
    r = requests.post(f"{url}/{kind}/{ident}", data=body.encode("utf-8"),
                      headers={"Content-Type": "application/json", "Rtrack-Token": tok},
                      timeout=120)
    if r.status_code == 413:
        raise SystemExit(f"[relay] bundle too large for KV: {r.text[:200]}")
    r.raise_for_status()
    return r.json()


def get(kind: str, ident: str) -> dict | None:
    url, _ = _env()
    r = requests.get(f"{url}/{kind}/{ident}", timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def wait(kind: str, ident: str, timeout: float = DEFAULT_TIMEOUT_S,
         poll: float = POLL_S) -> dict | None:
    """Block until something appears, or give up. Prints progress so a long wait for a
    human to finish curating does not look like a hang."""
    t0 = time.time()
    last = -1.0
    while time.time() - t0 < timeout:
        doc = get(kind, ident)
        if doc is not None:
            print(f"\n[relay] {kind}/{ident} arrived after {time.time() - t0:.0f}s")
            return doc
        el = time.time() - t0
        if el - last >= 30:
            print(f"[relay] waiting for {kind}/{ident} ... {el:.0f}s", flush=True)
            last = el
        time.sleep(poll)
    print(f"[relay] timed out after {timeout:.0f}s waiting for {kind}/{ident}")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Push/pull curation work via the relay.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, helptext in (("push-bundle", "upload a curation bundle"),
                           ("push-calib", "upload a calibration frame"),
                           ("push-tracks", "upload exported routes for the app")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("ident", help="match key, or video id for calib")
        p.add_argument("--file", type=Path, default=None)

    for name, kind in (("wait-answer", "answer"), ("wait-points", "points"),
                       ("wait-occl", "occl")):
        p = sub.add_parser(name, help=f"block until {kind} come back")
        p.add_argument("ident")
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
        p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("index", help="what the relay currently holds")
    p = sub.add_parser("clear", help="delete one entry")
    p.add_argument("kind", choices=["bundle", "answer", "calib", "points"])
    p.add_argument("ident")

    args = ap.parse_args(argv)
    C.ensure_dirs()

    if args.cmd == "index":
        url, _ = _env()
        r = requests.get(f"{url}/index", timeout=60)
        r.raise_for_status()
        d = r.json()
        print(f"[relay] {d.get('count', 0)} item(s)")
        for it in d.get("items", []):
            kb = (it.get("bytes") or 0) / 1024
            age = (time.time() - (it.get("at", 0) / 1000)) if it.get("at") else None
            print(f"  {it['kind']:>7} {it['id']:<28} {kb:>8.0f} KB"
                  + (f"  {age / 60:.0f} min ago" if age else ""))
        return 0

    if args.cmd == "clear":
        url, tok = _env()
        r = requests.delete(f"{url}/{args.kind}/{args.ident}",
                            headers={"Rtrack-Token": tok}, timeout=60)
        print(f"[relay] {r.status_code} {r.text[:120]}")
        return 0

    if args.cmd in ("push-bundle", "push-calib", "push-tracks"):
        kind = {"push-bundle": "bundle", "push-calib": "calib",
                "push-tracks": "tracks"}[args.cmd]
        # Same shape as the wait-* destination table, and for the same reason: three
        # kinds through a two-way ternary is where a wrong default would hide.
        default = {
            "bundle": C.STAGE3_DIR / f"{args.ident}_curate_frames.json",
            "calib":  C.STAGE3_DIR / f"{args.ident}_calib_frame.json",
            "tracks": C.STAGE3_DIR / f"{args.ident}.json",
        }[kind]
        src = args.file or default
        if not src.exists():
            raise SystemExit(f"[relay] {src} not found")
        doc = json.loads(src.read_text(encoding="utf-8"))
        res = put(kind, args.ident, doc)
        mb = res.get("bytes", 0) / 1e6
        print(f"[relay] pushed {kind}/{args.ident} ({mb:.1f} MB) from {src.name}")
        if mb > 3:
            print(f"[relay] NOTE: {mb:.1f} MB is heavy for venue wifi. Rebuild with "
                  f"fewer --frames, or lower FRAME_W/FRAME_Q in rtrack.curate.")
        return 0

    kind = {"wait-answer": "answer", "wait-points": "points",
            "wait-occl": "occl"}[args.cmd]
    doc = wait(kind, args.ident, args.timeout)
    if doc is None:
        return 1
    # Where each kind belongs on disk. `occl` lands in calib/ next to the calibration it
    # was drawn against -- rtrack.occluders.path_for looks for exactly this name, and
    # robots.py picks it up from the calibration stem without being asked.
    dests = {
        "answer": C.REPO_ROOT / "robot-tracker" / "corrections"
                  / f"{args.ident}_corrections.json",
        "points": C.STAGE3_DIR / f"{args.ident}_points.json",
        "occl":   C.CALIB_DIR / f"{args.ident}_occluders.json",
    }
    out = args.out or dests[kind]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    n = len(doc.get("labels") or doc.get("points") or doc.get("regions") or [])
    print(f"[relay] {n} item(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
