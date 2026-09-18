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

## Sources

- https://developers.cloudflare.com/kv/platform/limits/
- https://developers.cloudflare.com/kv/platform/pricing/
- https://developers.cloudflare.com/r2/pricing/
- https://developers.cloudflare.com/workers/platform/limits/
