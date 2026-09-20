# Cloudflare KV / R2 limits for rtrack-relay

Verified 2026-09-15 against developers.cloudflare.com.

## What broke (2026-09-15)

Worker threw Cloudflare error 1101:

    Error: KV list() limit exceeded for the day.
        at Object.fetch (worker.js:24:40)

Cause: `/index` was implemented as `RTRACK_KV.list()`. The free plan caps list at
1,000/day, a separate quota from the 100k reads. `/index` is polled by rtrack.watch,
the app's Tracks tab, and the curator's "What's available?". The watcher at a 20s poll
is 4,320 list calls/day, so the quota burned in ~5 hours and every /index after that
was a 500. Other endpoints stayed healthy because GET /bundle/<id> is a get, not a list.

Fix: `/index` now reads a maintained manifest key (`idx:manifest`); writes update it.
One list() remains, only to rebuild the manifest if missing. Degrades to an empty list
with a `degraded` field instead of a 500. Watcher default poll 20s -> 60s.

## Tier comparison

| | Free (per DAY) | Workers Paid $5/mo (per MONTH) |
|---|---|---|
| reads | 100,000 | 10M, then $0.50/M |
| writes | 1,000 | 1M, then $5.00/M |
| deletes | 1,000 | 1M, then $5.00/M |
| lists | 1,000 | 1M, then $5.00/M |
| storage | 1 GB | 1 GB, then $0.50/GB-month |
| value size | 25 MiB | 25 MiB (same) |

Structural difference that matters more than magnitude: paid limits are monthly
allowances with overage pricing, not hard daily caps. Free tier gives you a cliff
mid-event with no way to buy past it until UTC midnight.

Workers platform (separate from KV): free = 100,000 requests/day, 10 ms CPU,
50 subrequests/request, 128 MB memory. Paid = no daily request cap, 5 min CPU
(30 s default), 10,000 subrequests, 128 MB.

## Actual usage, ~60-match event day

| | used | free limit | headroom |
|---|---|---|---|
| writes | ~240 | 1,000 | 4x |
| reads | ~2,000 | 100,000 | 50x |
| lists | ~1 | 1,000 | 1000x |
| storage | ~228 MB | 1 GB | 4.4x |
| worker requests | ~2,000 | 100,000 | 50x |

Writes are 4 per match round-trip: bundle put + manifest get/put, then the same for
the answer. NOTE: the manifest fix doubled writes per operation, trading them for the
list() calls. Good trade at these ratios (lists were burned 4,000+/day by polling,
writes are per-match) but not free.

Nearest ceilings are writes and storage, both ~4x. What eats them is bundle size,
which went from 2.1 MB to 3.8 MB when best-views crops were added.

## If we outgrow it: R2, not Workers Paid

| | KV free | R2 free |
|---|---|---|
| storage | 1 GB | 10 GB/month |
| writes | 1,000/day | 1M Class A/month |
| reads | 100k/day | 10M Class B/month |
| egress | - | free, no charge |
| max value | 25 MiB | effectively unbounded |

R2 paid rates: $0.015/GB-month standard storage, $4.50/M Class A, $0.36/M Class B,
zero egress for all storage classes.

A 3.8 MB bundle is one Class A op regardless of size, so size stops affecting op
counts and only touches the 10 GB pool. Proposed split: R2 for bundles (large,
immutable blobs), KV for the control plane (answers, manifest, calib points - all
small). Both stay free with wide margins. Also retires the 25 MiB cap that MAX_BYTES
guards.

CAVEAT TO CONFIRM: enabling R2 may require a payment method on the account even for
free-tier usage. Not verified.

## Recommendation

Do nothing yet. We're at ~25% of the nearest limit and the failure is fixed.
Cheap lever if bundles keep growing: BEST_VIEWS 3 -> 2 in curate.py (~1/3 off),
at the cost of the time-spread that reveals a track changing robot.

Watch storage, not writes. Neither number is trustworthy until a full event has
actually been run through this.

## Multi-day events: the 24-hour TTL is a DATA-LOSS risk, not a convenience one

Everything on the relay expires after `TTL_S = 86400` (worker.js). That was chosen for
a one-day event and it is wrong for a district championship or a week-long trip, where
the machine at home may go 5-6 days without anyone touching it.

THE FAILURE IS SILENT AND IRREVERSIBLE. curate.html clears its local draft as soon as a
send is confirmed (deliberately -- see the DRAFT note there; a draft kept after a
successful send is a draft that resurrects stale answers). So once a curator sends, the
only copies of that work are:

  1. the `answer` key on the relay, and
  2. whatever rtrack.watch has already pulled to corrections/.

If the watcher is down for more than 24 hours, (1) expires and (2) never happened. The
curation is gone -- not stale, not degraded, gone -- and nothing anywhere reports it.
Over a six-day event with no physical access to the machine, an overnight crash on day 2
that is not noticed until day 4 destroys two days of scout labour.

Bundles expiring is a separate and much milder problem: a curator going back to an
earlier match finds nothing there, and the fix is to re-push a bundle that can always be
rebuilt from the video.

### Answers and bundles should not share a TTL

They have opposite economics and the single constant hides that:

| kind    | size    | replaceable?              | sensible TTL |
|---------|---------|---------------------------|--------------|
| answer  | ~9 KB   | NO -- irreplaceable human work | 30 days  |
| bundle  | ~4.3 MB | yes, rebuild from the clip     | 24h - 7d |

120 answers at ~10 KB is about 1.2 MB, so keeping every answer of a long event for a
month costs essentially nothing against the 1 GB free allowance. Bundles are where the
storage actually goes: 120 x 4.3 MB is ~516 MB, which fits but leaves little room if
more than one division is being covered at once.

Measured 2026-09-18 on 2026necmp1: 25 bundles averaging 4.2 MB, answers 8-10 KB each.

### Uptime is the other half, and it is not the same problem

Raising the TTL buys time for the watcher to come back; it does not keep the watcher
alive. Over 5-6 days the realistic killers, in rough order of likelihood:

  - the machine sleeping (residential power, nothing crashed, nothing looks wrong)
  - a Windows Update reboot -- near certain across six days, and a Task Scheduler
    "at logon" trigger does not fire if nothing logs back in
  - a power cut, which needs BIOS restore-on-AC as well as an auto-start
  - a crashed pipeline run holding the per-event lock: pipeline.STALE_LOCK_S is 3 hours,
    and for those 3 hours rtrack.watch logs "will retry" every 60 s and looks perfectly
    healthy while processing nothing

The last one is the argument for a heartbeat that reports STATE (processing / idle /
lock held) rather than mere liveness. A watcher that is alive and stuck is the case a
simple ping would miss.

Options, from least to most robust: a supervisor loop that restarts on exit; Task
Scheduler at logon with restart-on-failure; a real Windows service (NSSM) that starts
without a login. Anything unattended for days wants the third, plus `powercfg` sleep
disabled and update active-hours set.

Remote activation from the app is NOT among the options. The relay is a mailbox, not a
remote shell: something must already be running at home to receive a start request, and
that something may as well be the watcher. A supervisor accepting relay commands is
buildable and does buy one thing uptime alone does not -- changing `--event` mid-day
when covering more than one division -- but it cannot solve the bootstrap.

## Sources

- https://developers.cloudflare.com/kv/platform/limits/
- https://developers.cloudflare.com/kv/platform/pricing/
- https://developers.cloudflare.com/r2/pricing/
- https://developers.cloudflare.com/workers/platform/limits/
