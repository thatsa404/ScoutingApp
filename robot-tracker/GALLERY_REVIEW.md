# Gallery Review Component

## Purpose

Build appearance galleries from human-approved, provenance-bearing tracklet views.
Human match curation supplies identity anchors; it does **not** authorize every
solver-assigned detection in the match as gallery truth.

The existing `reid gallery` accumulator is unsafe for authoritative use because it
samples every named route and irreversibly folds those samples into one team centroid.
The replacement keeps each contribution reviewable and removable.

## Team-oriented review design

The review unit is a team gallery, not a solver tracklet. The Tracks tab lists one row
per team. Opening a row shows:

- up to six read-only images already in that team’s current reviewed gallery;
- up to twelve new candidate images from human-anchored tracks currently associated with
  that team;
- candidates selected first for temporal and calibrated field-area coverage, then by
  detection size as a legibility/whole-robot tie-breaker
  used to avoid spending the entire set on one near-duplicate view;
- crops made from the full-resolution frame with 45% padding on every edge, capped at
  260 pixels tall, matching the curator’s contextual crop behavior.

The reviewer checks any number of candidate images. The answer contains only the
selected candidate crop hashes for each reviewed team, plus the deterministic candidate
ids that were shown and acknowledged. Those reviewed candidates are not reissued in a
later bundle; accepted images become current-gallery references and rejected/empty
selections are still removed from the immediate review pool. Current-gallery images are
comparison context and cannot be accidentally re-added. This makes the human decision
"these exact images are good exemplars for team X" rather than "this entire track is
team X".

## Workflow

```text
match corrections + source tracks + appearance cache
  -> gallery-review prepare
  -> team groups with current references and size-prioritized candidate images
  -> gallery-review bundle
  -> human: accept / relabel / reject / mixed / split
  -> gallery-review resolve
  -> approved prototype manifest
  -> deterministic derived gallery
  -> appearance votes for later matches
```

Gallery review is a separate human task from route curation. A route answer may seed
candidates, but only an explicit gallery-review decision may publish a prototype.

The workflow is iterative rather than a one-time approval pass:

```text
approved prototypes v1
  -> solve/re-solve matches
  -> propose uncertain, novel, or contradictory tracklet views
  -> human review round
  -> approved prototypes v2
  -> reassess every affected match
  -> repeat until the reviewer marks the team sufficient or new evidence appears
```

The reviewer may mark a team gallery `sufficient` when the proposed bundles have become
redundant and the assignments look stable. This pauses routine requests for that team;
novel views, strong contradictions, a robot revision, or a new camera domain may reopen
it. “Sufficient” is a review state, not a claim that the model can never be wrong.

## Candidate selection

- Begin with detections directly anchored by a human team label.
- Apply the calibrated camera-view and on-field bound before resolving anchors or
  selecting candidates. Missing or quarantined calibration fails closed; an unsafe
  diagnostic override must be explicit and recorded in the bundle.
- Expand only within a fragment whose identity is not contradicted by another human
  label, `mixed`, `notrobot`, co-detection, or an impossible transition.
- Never expand from a solver-only team assignment.
- Exclude detections outside the approved calibration/visibility mask.
- Select images rather than whole tracklets: seed each cleanly anchored source track,
  first maximize coverage across match time and calibrated field area, then prioritize
  larger detection boxes so the selected views are whole-robot and easy to interpret.
- Candidate crops use a 45% edge buffer from the full-resolution frame so the reviewer
  sees the whole robot and nearby context.
- Default review set per team: 6 current-gallery references and 12 new candidate images.
- Default mature target per team: 6--10 accepted tracklets and 24--40 views. Warn above
  50 and cap ordinary publication at 100 unless a reviewer explicitly overrides it.

After the seed round, prioritize candidates by expected information gain:

- solver/gallery disagreement or low identity margin;
- a fragment whose assignment changes under the newest gallery;
- embedding distance from every approved prototype (novel viewpoint or robot change);
- disagreement between appearance, alliance, continuity, and human anchors;
- representative examples from an under-covered camera, scale, or viewpoint slot.

Do not keep asking about near-duplicates merely because they are numerous. A review
round should show a small, ranked set and explain why each item was selected.

## Review bundle

The bundle is self-contained and compatible with the existing relay transport pattern.
Each candidate contains:

- stable candidate id and schema version;
- event, match, video, source track id, time range, and source detection anchors;
- proposed team and alliance;
- the exact human correction labels supporting the proposal;
- 3--5 representative crops with frame/time, box, crop hash, and embedding id;
- warnings for conflicts, low quality, long duration, or weak anchor proximity;
- current approved prototypes for side-by-side comparison.

The reviewer can `accept`, `relabel`, `reject`, mark `mixed`, or request `split`. The UI
must not pre-accept the machine proposal.

It can also mark a team's gallery `sufficient`, `needs-more`, or `robot-changed`. A
`robot-changed` decision starts a new revision while preserving the old one for earlier
matches.

## Approved prototype manifest

The authoritative artifact is an append-only JSON manifest, not an averaged `.npz`:

```json
{
  "schemaVersion": 1,
  "event": "2026mawor",
  "embeddingSpace": "resnet18-imagenet-v1/raw",
  "decisions": [
    {
      "candidateId": "...",
      "team": "190",
      "decision": "accept",
      "reviewedAt": "...",
      "source": {"match": "2026mawor_qm15", "track": 16},
      "views": [{"frame": 4932, "time": 82.2, "cropHash": "..."}]
    }
  ]
}
```

Relabels and revocations append decisions that supersede an earlier candidate id. Every
derived gallery can then be rebuilt exactly, and one bad contribution can be removed.

The durable identity key is `(season, team)`, not event. For example, team 190's 2026
gallery is shared across its 2026 events. Event and camera are prototype attributes used
for domain-aware selection, not identity boundaries. The manifest also carries a robot
revision id and optional validity interval:

```text
2026:190 / revision 1 / event A prototypes
2026:190 / revision 1 / event B prototypes
2026:190 / revision 2 / changed mechanism after event B
```

New-event voting starts with the season gallery, while retaining event-local prototypes
as the camera-specific layer. A revision does not delete old data: matches before the
change continue to use the prior revision, and uncertain transition dates can compare
both revisions until reviewed.

## Derived gallery and voting

- Preserve multiple prototypes per team; do not reduce the event to one permanent mean.
- Balance tracklets so a long video track cannot outweigh every other viewpoint.
- Restrict candidates to confirmed-present teams and use alliance as solver evidence or
  a calibrated closed-set restriction.
- Prefer camera/viewpoint-compatible and recent prototypes.
- Emit gallery maturity (`empty`, `seeded`, `developing`, `mature`) and prototype counts
  with every vote. Sparse evidence must not look fully confident.
- The `.npz` is a disposable cache derived from the manifest, never the source of truth.

Every published gallery has an immutable version/content hash. Match run manifests record
the exact season-gallery version, selected robot revisions, and prototype ids used for
each scheduled team. This is required for replay and for explaining why a route changed.

## Feedback and prior-match reassessment

Publishing a new gallery version queues every prior match whose lineup contains an
affected `(season, team)` and whose source embeddings are still available. Reassessment
recomputes appearance votes and the downstream solve without decoding video again.

For each replay, retain and report:

- old/new gallery versions and appearance margins;
- fragments whose team, path edge, parking state, or uncertainty changed;
- agreement against existing human corrections without using those corrections as test
  data when they trained the changed prototype;
- route continuity, custody, and teleport changes;
- whether the new result creates a new human conflict.

An improved solve may update a route after the normal publication safeguards, but its
machine-assigned fragments do not become gallery prototypes. This one-way boundary
prevents a mistaken gallery vote from relabelling old matches and then teaching the same
mistake back into the gallery.

Replay should be dependency-driven rather than rebuilding a whole season blindly. A
gallery update for `2026:190` affects only matches containing team 190; an event-camera
prototype affects only compatible domains unless explicitly promoted to the season
layer. Runs already evaluated against the same content hash are skipped.

### Replay cost controls

Review decisions do not publish immediately. They accumulate in a draft and one explicit
publish creates one immutable gallery version. A review round containing twenty decisions
therefore causes one dependency update, not twenty waves of replay.

Replay has two stages:

1. Recompute appearance similarities from cached embeddings and compare old/new votes.
   This requires no video decode and no CNN inference.
2. Run the identity/path solver only when a fragment's winning team, margin, gallery
   coverage, or edge evidence changes materially. Matches whose votes are unchanged are
   marked evaluated for the new gallery version without a solve.

The queue is keyed by `(match, gallery content hash)`. A newer unpublished version
supersedes stale pending work, each affected match is solved at most once per published
version, and multiple changed teams in one lineup coalesce into one solve. Replays are
background work with bounded concurrency so live-match processing remains higher priority.
Historical route publication is separate from reassessment: a replay produces a diff and
quality report first, and only materially improved, non-conflicting results advance under
the normal publication policy.

The expected fan-out is the number of matches containing the changed team, not the full
season. A typical team appears in a small fraction of an event's matches. Cached qm15
CP-SAT solves take only a few seconds; decoding and embedding are the expensive operations,
and replay deliberately avoids both.

## Relay and storage budget

Do not resend a growing season gallery in every review bundle. Crops are immutable,
content-addressed objects uploaded once by crop hash. A review bundle contains candidate
metadata, small thumbnails needed for the current round, and hashes for already-known
prototype views. Decisions contain only candidate ids, actions, optional corrected teams,
and manifest/version hashes.

Operational limits for the first implementation:

- at most 6 current-gallery references per team;
- at most 12 candidate images per team;
- target bundle size below 1 MB and hard refusal above 4 MB;
- no historical accepted crop is embedded again when its hash is already available;
- manifests and answers are retained; replaceable bundles and thumbnails may expire;
- season prototypes live in durable artifact storage, not the relay's transient mailbox.

The existing self-contained route-curation bundle remains appropriate for a short-lived
task. Gallery review is longitudinal, so its durable manifest/crops and transient review
messages must have separate storage lifecycles. If content-addressed object storage is not
available in the first increment, send bounded self-contained delta bundles and accept
the temporary duplication; never send the complete gallery.

## Pipeline integration

Replace the unconditional final pipeline step:

```text
reid gallery <curated labeled route>
```

with:

```text
gallery-review prepare <match>
gallery-review send/wait/resolve       # optional relay, same mailbox pattern
gallery-review publish <event>         # only accepted decisions
```

Publication is season-scoped even when preparation is event-scoped:

```text
gallery-review publish --season 2026
gallery-review replay --team 190 --season 2026
```

Until this component is active, legacy gallery updates should be explicitly marked
`unreviewed` and must not be presented as authoritative evidence.

## Acceptance criteria

- No solver-only assignment can enter an authoritative gallery.
- Every prototype displayed by voting can be traced to a review decision and source crop.
- Removing one decision deterministically removes its effect.
- Re-running preparation is idempotent and does not duplicate contributions.
- Bundle and answer schemas reject embedding-space or source-hash mismatches.
- Leave-match-out evaluation reports accuracy versus approved views per team and gallery
  maturity before the reviewed gallery replaces the legacy one.
- A gallery change identifies and reassesses all dependent prior matches exactly once.
- Replayed machine assignments cannot enter the authoritative manifest without a new
  human review decision.
- Season galleries survive event boundaries, while robot revisions and event/camera
  overlays preserve changes instead of averaging incompatible appearances together.
- Review traffic is bounded and sends deltas, never a complete season gallery.
- Publishing a batch of decisions creates one replay wave; unchanged votes do not invoke
  the solver, and duplicate/stale replay jobs coalesce by content hash.

# Implementation instructions

This section is the build plan. Implement it in the order shown; each phase leaves the
application in a usable state and has an explicit compatibility boundary.

## 1. Components and ownership

| Area | Existing integration point | Required work |
|---|---|---|
| App UI | `index.html` `#tools-tab-tracks`; `main.js::renderTracksTab()` | Add Gallery Review and Replay queues inside Tools → Tracks. |
| Review rendering | Existing Tracks tab styles and route-curation interaction patterns | Render candidate montages, decisions, draft persistence, submit, and team maturity controls. |
| Relay worker | `rtrack-relay/worker.js` | Add gallery bundle, answer, and status artifact kinds with bounded TTL/size policy. |
| Relay client | `src/rtrack/relay.py` | Add push/pull/wait/clear commands for gallery review artifacts. |
| Candidate generation | New `src/rtrack/gallery_review.py` | Resolve human anchors, select diverse tracklets/views, write review bundles, validate answers. |
| Authoritative store | New `robot-tracker/gallery/<season>/manifest.json` | Append reviewed decisions and supersessions keyed by season/team/revision. |
| Crop object store | New local `out/gallery/objects/` plus durable remote storage adapter | Store JPEGs once by SHA-256; never embed the season corpus in a bundle. |
| Derived gallery | Replace the authoritative role of `<event>_gallery_cnn.npz` | Compile reviewed manifests into versioned multi-prototype caches. |
| Voting | `src/rtrack/reid.py::vote_tracks()` | Load season/team prototypes, select compatible subsets, and emit prototype-level provenance/margins. |
| Pipeline | `src/rtrack/pipeline.py` final gallery step | Prepare review work; never auto-train from the complete solver-labelled route. |
| Replay | New `src/rtrack/gallery_replay.py`; integrate with `watch.py` | Diff votes cheaply, queue material re-solves, and report old/new route quality. |
| Published status | `src/rtrack/export.py::write_manifest()` and `public/tracks/index.json` | Publish gallery maturity/version and replay status for the Tracks tab. |

Do not add reviewed-gallery behavior to `build_gallery()` while leaving its current input
contract intact. That function consumes solver-labelled routes and is the unsafe boundary.
Keep it temporarily as `legacy/unreviewed`, then remove it after reviewed parity is proved.

## 2. Artifact layout

Use these paths so authoritative data, replaceable caches, and transient review work are
visibly different:

```text
robot-tracker/gallery/<season>/manifest.json          authoritative reviewed decisions
robot-tracker/gallery/<season>/schema.json            optional copied schema/version note
robot-tracker/out/gallery/objects/<sha256>.jpg         local content-addressed crop cache
robot-tracker/out/gallery/<season>/<version>.npz       derived multi-prototype embeddings
robot-tracker/out/gallery/<season>/latest.json         current content hash + cache path
robot-tracker/out/gallery/review/<reviewId>.json       transient outbound bundle
robot-tracker/out/gallery/review/<reviewId>_answer.json transient returned decisions
robot-tracker/out/gallery/replay/queue.json            replaceable local replay queue
robot-tracker/out/gallery/replay/<match>-<version>.json replay report
```

Only the season manifest is source-of-truth repository data. Generated NPZ files, crop
objects, bundles, queues, and reports stay ignored. Durable deployment must back up crop
objects separately; the manifest retains frame/box/hash provenance so a crop can be
reconstructed from source video when that video is available.

## 3. Identifiers and content hashes

Identifiers must survive retries and prevent accidental duplicate training:

- `seasonTeam`: `"2026:190"`.
- `revisionId`: stable id such as `"2026:190:r1"`.
- `candidateId`: SHA-256 of season, team proposal, match, immutable source track id,
  source interval, embedding space, and selected crop hashes.
- `prototypeId`: SHA-256 of candidate id, accepted team/revision, and accepted view hashes.
- `reviewId`: season plus a hash of the ordered candidate ids.
- `galleryVersion`: SHA-256 of the canonical active decision set, schema version,
  embedding space, and compiler configuration.

Canonical JSON uses sorted keys and stable list ordering before hashing. Retrying
`prepare`, `apply`, or `publish` with identical inputs must produce identical ids and no
new manifest decisions.

## 4. Schemas

### 4.1 Review bundle (`galleryReviewBundle`, schema 2)

```json
{
  "kind": "galleryReviewBundle",
  "schemaVersion": 2,
  "reviewId": "2026-...",
  "season": 2026,
  "createdAt": "ISO-8601",
  "embeddingSpace": "resnet18-imagenet-v1/raw",
  "galleryVersion": "sha256-before-review",
  "limits": {"currentImagesPerTeam": 6, "candidateImagesPerTeam": 12,
             "cropPadding": 0.45},
  "teams": [
    {
      "team": "190",
      "currentGallery": [{"imageId": "sha256", "cropHash": "sha256",
                          "thumbnail": "data:image/jpeg;base64,...",
                          "source": {"match": "2026mawor_qm8", "time": 82.2}}],
      "candidates": [{
        "candidateId": "sha256",
        "cropHash": "sha256",
        "thumbnail": "data:image/jpeg;base64,...",
        "source": {"match": "2026mawor_qm15", "sourceTrack": 16,
                   "frame": 4932, "time": 82.2, "boxArea": 12000},
        "quality": {"boxArea": 12000, "boxWidth": 100, "boxHeight": 120,
                    "padding": 0.45}
      }]
    }
  ]
}
```

Every bundle records input hashes. The app must refuse submission when required fields,
candidate ids, or embedding spaces are inconsistent.

Schema-2 bundles also record `fieldFilter`, including calibration stem, camera-view and
off-field drop counts, calibration usability, and whether an unsafe diagnostic override
was used. The field filter is part of the review id, so changing the bound invalidates
the prior review artifact.

### 4.2 Review answer (`galleryReviewAnswer`, schema 2)

```json
{
  "kind": "galleryReviewAnswer",
  "schemaVersion": 2,
  "reviewId": "2026-...",
  "bundleHash": "sha256",
  "baseGalleryVersion": "sha256-before-review",
  "reviewer": "optional-name",
  "submittedAt": "ISO-8601",
  "selections": [{"team": "190", "include": ["sha256"],
                  "reviewed": ["candidate-sha256"]}],
  "teamStates": [{"team": "190", "state": "sufficient"}]
}
```

Allowed candidate actions are `accept`, `relabel`, `reject`, `mixed`, `split`, and
`defer`. Allowed revision actions are `keep`, `new-revision`, and `compare-revisions`.
An answer may be applied only to its exact bundle hash and base gallery version. If the
base changed, regenerate or explicitly rebase; never silently apply stale decisions.

### 4.3 Authoritative season manifest (`galleryManifest`, schema 1)

The manifest is append-only. Each record contains decision id, superseded decision id,
review id, candidate/source provenance, accepted team/revision, view hashes, reviewer,
timestamps, embedding space, and active/revoked state. A separate canonical `teamStates`
map records maturity and review state. `reviewedCandidateIds` is a persistent ledger of
candidate ids already shown in submitted team reviews; it is used to advance future
bundles but is deliberately excluded from `galleryVersion`, since it does not change
the appearance model. Do not mutate old records to relabel or revoke; append a
superseding record so history remains explainable.

### 4.4 Derived prototype cache (`reviewed-gallery-v1`)

The NPZ must contain, at minimum:

```text
prototypeId[]       string
seasonTeam[]        string
revisionId[]        string
embedding[]         float32 [N,D]
sourceEvent[]       string
sourceCamera[]      string
sourceMatch[]       string
sourceTrack[]       int
startS[] / endS[]   float
viewCount[]         int
quality[]           float
manifestVersion     scalar string
embeddingSpace      scalar string
schemaVersion       scalar int
```

Never expose only `teams/cent/count`; that recreates the irreversible legacy design.

## 5. Back-end candidate generation

Create `src/rtrack/gallery_review.py` with these commands:

```text
gallery-review prepare <match> [--season YEAR] [--max-candidates 12]
gallery-review apply <answer.json>
gallery-review publish --season YEAR
gallery-review status [--season YEAR] [--team TEAM]
gallery-review rebuild --season YEAR
```

### 5.1 `prepare`

1. Load the exact stitched tracks, appearance cache, corrections, match lineup, field
   mask, current reviewed manifest, latest solver output, and run manifest.
2. Verify hashes and embedding-space compatibility before selecting anything.
3. Resolve only `src: "human"` team labels onto immutable stitched detections.
4. Construct authoritative intervals conservatively:
   - one uncontradicted human label seeds its containing source tracklet;
   - differing labels cut the interval at their midpoint;
   - `mixed`/`notrobot`, co-detection conflicts, impossible transitions, and unresolved
     cuts block expansion;
   - solver assignment alone never creates an authoritative interval.
5. Enumerate cached CNN embeddings/crops within those intervals and the valid field mask.
6. Group by source tracklet and team proposal. Remove candidates already represented by
   an active prototype with the same candidate hash.
7. Score candidate priority from:
   - new/empty team gallery;
   - low current vote margin or gallery/solver disagreement;
   - assignment change under latest gallery;
   - distance from approved prototypes;
   - under-covered camera, apparent-size bin, time/viewpoint cluster, or robot revision;
   - source track detection coverage and crop legibility.
8. Select views with a quality-filtered farthest-first/medoid pass. Enforce temporal
   separation and source-track balancing. Never select five adjacent frames.
9. Include only a bounded comparison set: nearest approved same-team prototype, nearest
   competing-team prototype, and one diverse same-team prototype where available.
10. Write the canonical bundle, its hash, and a summary of excluded candidates/reasons.

Candidate preparation must not decode the whole video when crop objects already exist.
When decoding is required, gather every requested frame in one sequential pass, matching
the existing `curate.py` strategy.

### 5.2 `apply`

Validate schema, bundle hash, base gallery version, candidate ids, accepted view hashes,
team/year, and embedding space. Reject the entire answer on structural mismatch. Semantic
warnings (for example accepting a candidate marked conflicted) require an explicit
override field and are recorded. Append decisions atomically using a temporary file plus
replace; preserve a timestamped backup of the previous manifest.

`split` produces a route/track correction task and does not publish a prototype until
the corrected source interval is regenerated and reviewed. `mixed` and `reject` become
negative provenance so the same candidate is not proposed repeatedly.

### 5.3 `publish` / `rebuild`

Compile active accepted decisions into separate tracklet-level prototypes. Average views
within one accepted tracklet, not across the whole team. Normalize embeddings after any
compatible head transform. Emit the deterministic cache and `latest.json`, then compute
the content hash and enqueue affected matches. `rebuild` must reproduce the same bytes or
the same canonical array content from the same manifest.

## 6. Voting with reviewed galleries

Update `reid.py` without changing legacy reads until migration is complete:

1. Add `load_reviewed_gallery(season, teams, match_context)`.
2. Select only scheduled, confirmed-present teams; keep `unknown` teams eligible and
   exclude confirmed-absent teams.
3. Select the robot revision valid at match time. If uncertain, score both revisions and
   expose that ambiguity rather than averaging them.
4. Rank prototypes by camera/event compatibility and recency, but retain a season-level
   fallback. Do not discard cross-event data merely because an event-local prototype
   exists.
5. Balance prototypes per source tracklet and cap the number contributing per team.
6. Score a query against every selected prototype. Start with a robust aggregate such as
   the mean of the best two compatible prototype similarities, calibrated by leave-match-
   out evaluation; do not let one nearest crop or one team with more prototypes win by
   count.
7. Emit per vote: winning team, runner-up, margin, prototype ids, gallery version,
   revision ids, maturity, and whether presence/alliance restricted the candidate set.
8. Preserve time-local vote lists so later fragment cuts can retally correctly.

The solver should consume confidence/margin rather than treating every appearance vote as
equal. Keep the reviewed gallery feature-gated until it beats legacy/raw control on
leave-match-out labels and qm15's audit.

## 7. Relay changes

In `rtrack-relay/worker.js`:

1. Add `gallery-bundle`, `gallery-answer`, and `gallery-status` to `KINDS`.
2. Add only `gallery-answer` to `DEVICE_WRITABLE`.
3. Set explicit TTLs: bundle 7 days, answer/status 30 days. The authoritative manifest
   never lives only in KV.
4. Enforce a 4 MiB limit specifically for `gallery-bundle`; retain the existing global
   safety limit for legacy artifacts.
5. Store index metadata for season, teams, candidate count, base gallery version, and
   state so Tools → Tracks can render the queue without fetching every bundle.
6. Keep `/index` as the maintained manifest key. Do not introduce relay `list()` calls.

In `relay.py`, add:

```text
push-gallery-bundle <reviewId> --file ...
wait-gallery-answer <reviewId> --out ...
push-gallery-status <reviewId> --file ...
clear gallery-bundle|gallery-answer|gallery-status <reviewId>
```

Extend the single `KINDS` table and destination maps rather than adding independent
request code. Print bundle bytes and reject locally above 4 MiB before consuming a relay
write. Answers remain tiny JSON documents.

Phase 1 may embed new thumbnails directly in each bounded delta bundle. Phase 2 should
add a content-addressed durable object endpoint and replace repeated thumbnail data with
hash references. Do not block the first reviewed-gallery implementation on R2, but never
send the full season corpus through KV.

## 8. Tools → Tracks front-end UI

Use the existing Tracks tab in `index.html` and `main.js`; do not add another top-level
navigation item. `renderTracksTab()` already loads the relay index, published track
manifest, event schedule, gallery counts, and camera tasks in one render.

### 8.1 Layout

Inside `#tools-tab-tracks`, render sections in this order:

1. Relay controls and reachability (existing).
2. **Gallery review** summary and queue (new).
3. **Replay status** (new).
4. Cameras (existing).
5. Match tracking queue (existing).

The Gallery Review summary shows:

- active season and gallery version;
- teams empty/seeded/developing/mature/sufficient;
- approved tracklets/views versus the 6--10 / 24--40 targets;
- number of review bundles waiting, answers returned, and drafts on this device;
- `Review next`, `Refresh`, and optional team filter controls.

Each queue row/card shows the team, current reference count, candidate count, bundle age,
base gallery version, and state (`ready`, `draft`, `submitted`, `stale`, `applied`). A
stale bundle is visible but cannot be submitted until rebased.

### 8.2 Review interaction in the Tracks tab

Clicking `Review` expands an in-tab full-width review panel (or a full-screen overlay
owned by the Tracks tab on small screens). Do not navigate to a disconnected page.

For each team display:

- 5--6 current-gallery images as read-only references;
- 10--12 padded candidate images, selected by time/field coverage first and detection
  size second;
- match/time/source-track and box-area context for every candidate;
- a checkbox/toggle for each candidate, with no default selections;
- a team state control for `needs-more`, `sufficient`, or `robot-changed`.

There is no accept/reject action on a whole track. The reviewer’s explicit selected crop
hashes are the only images eligible for publication; an empty selection is valid when no
candidate is good enough.

At the end of a team's candidates, offer `Needs more`, `Sufficient`, and `Robot changed`.
Show the effect of `Sufficient`: routine bundles pause, but novelty/conflict can reopen it.

### 8.3 Drafts and submission

Use `localStorage` keyed by review id plus bundle hash, as the curator does for drafts.
On bundle load:

- discard decisions for candidate ids no longer present;
- warn and refuse to reuse a draft whose bundle hash changed;
- preserve work across refresh/offline use;
- clear the draft only after the relay confirms the answer POST.

Before submit, show selected-image counts by team and the projected team maturity changes.
Submission POSTs `galleryReviewAnswer` to `/gallery-answer/<reviewId>`. If a device token is required,
reuse the curator's prompt/storage convention; never embed a secret in the static app.

The answer contains `selections: [{team, include: [cropHash], reviewed: [candidateId]}]`;
`reviewed` covers the candidate set shown for that submitted team, while current-gallery
images are never included in `include`. After success, update the queue locally to `submitted`,
return to the Gallery Review section, and leave match/camera queues intact. The relay
index refresh should eventually replace this optimistic state with authoritative status.

### 8.4 `main.js` structure

Do not make `renderTracksTab()` larger indefinitely. Extract helpers adjacent to the
existing Tracks code:

```text
galleryReviewIndex(items)
renderGalleryReviewQueue(host, state)
loadGalleryReviewBundle(reviewId)
openGalleryReview(reviewId)
renderGalleryCandidate(candidate)
saveGalleryDraft(reviewId, bundleHash, draft)
submitGalleryReview(reviewId)
renderGalleryReplayStatus(statusItems)
```

Continue using one `/index` request per render. Fetch bundle bodies lazily only when the
user opens them. Escape all relay-provided strings before assigning HTML; candidate ids,
notes, and metadata are untrusted. Validate schema before caching or rendering. Thumbnails
must be `data:image/jpeg;base64` or an approved same-origin/content-addressed URL—reject
arbitrary schemes.

### 8.5 Published manifest additions

Extend `public/tracks/index.json` with a backwards-compatible top-level block:

```json
{
  "reviewedGallery": {
    "season": 2026,
    "version": "sha256",
    "embeddingSpace": "...",
    "teams": {
      "190": {"maturity": "developing", "tracklets": 4, "views": 17,
              "revision": "r1", "updatedAt": "..."}
    }
  }
}
```

If absent, the UI labels the gallery `legacy/unreviewed`; it must not infer reviewed
status from legacy crop counts.

## 9. Pipeline and watcher changes

In `pipeline.py`, remove the unconditional authoritative call to `reid gallery` after
route export. During migration, either skip it or invoke it only with an explicit
`--legacy-gallery` flag and mark its output unreviewed.

After corrections and final solve:

1. Run `gallery-review prepare <match>`.
2. If no novel authoritative candidates exist, continue without relay work.
3. If candidates exist and relay is enabled, push the bounded bundle. Do not block live
   route publication waiting for gallery review.
4. The watcher polls gallery answers, applies them, and batches publication. A configurable
   explicit publish or short debounce closes the batch.
5. Publication writes a new gallery version and queues dependency-driven replay.

`watch.py` must prioritize new match answers over gallery review and replay. Suggested
priority: live route curation → calibration/occluders → gallery answers/publication → vote
diffs → historical CP-SAT replay. Use a single event/season publication lock so two
answers cannot race the manifest.

## 10. Replay implementation

Create `gallery_replay.py` with:

```text
gallery-replay enqueue --season YEAR --from OLD --to NEW
gallery-replay run [--limit N]
gallery-replay status [--season YEAR]
gallery-replay diff <match> --version NEW
```

On publication, inspect match run manifests and enqueue only matches containing changed
season/team/revision dependencies. Queue entries include match, old/new versions, changed
teams, state, attempts, and timestamps. A unique `(match,newVersion)` key coalesces work.

For each entry:

1. Verify source tracks, embeddings, corrections, and calibration hashes still match the
   recorded run. Mark stale rather than replaying mismatched inputs.
2. Recompute votes from cached embeddings under the new prototypes.
3. Compare per-fragment winner, margin, and contributing prototype ids with the old vote
   document.
4. If nothing material changed, write a `vote-unchanged` report and finish without CP-SAT.
5. Otherwise invoke the recorded solve arguments with only the gallery/vote dependency
   changed. Do not silently adopt current defaults.
6. Produce a report containing assignment changes, audit/correction agreement, custody,
   gaps/teleports, parked tracks, conflicts, objective composition, and reproducibility
   status.
7. Auto-publish only under an explicit policy. Initially require operator acceptance for
   route changes; unchanged or strictly diagnostic reports need no human action.

The Tracks tab Replay section reads `gallery-status` relay metadata and/or the published
manifest summary: queued, vote-diffing, solving, improved, unchanged, needs-review, failed.
Show counts and the newest material diffs, not a row for every completed no-op.

## 11. Migration of the existing gallery

Do not convert the current centroid/count NPZ into approved prototypes; its provenance is
already lost. Migration is:

1. Rename/status current galleries as `legacy/unreviewed` and retain them as an A/B control.
2. Generate seed review candidates from existing human corrections and source caches.
3. Review qm15's five present teams first, including known audit disagreements.
4. Publish the first reviewed season gallery alongside legacy, feature-gated.
5. Compare legacy, raw reviewed, and reviewed+solver results leave-match-out and on qm15.
6. Expand team coverage iteratively; do not wait for all 39 teams before testing.
7. Switch production voting only after reviewed galleries improve accuracy without route
   regressions. Keep rollback by gallery version.

## 12. Tests and verification

### Unit tests

- canonical hashes and idempotent candidate/prototype ids;
- human-anchor interval construction and blocking conflicts;
- diversity selection, temporal separation, tracklet balancing, and caps;
- answer schema validation, stale-base rejection, supersession, and revocation;
- deterministic manifest compilation and cache rebuild;
- robust prototype aggregation independent of prototype count;
- dependency fan-out and replay queue coalescing;
- relay kind permissions, TTLs, index metadata, and 4 MiB rejection;
- UI schema validation, HTML escaping, draft invalidation, and action serialization.

### Integration tests

- prepare → relay push → Tracks-tab review → answer pull → apply → publish;
- interrupted/offline draft recovery and successful-clear behavior;
- two answers racing one season manifest (one applies, one reports stale base);
- gallery update affecting two teams in one match produces one replay;
- unchanged votes skip CP-SAT;
- changed votes reproduce recorded solver arguments and produce a route diff;
- robot revision selects old prototypes before and new prototypes after the boundary;
- absent team cannot receive votes or gallery candidates.

### QM15 acceptance run

Use the later route audit only for evaluation, not candidate training in the scored fold.
Report embedding coverage, appearance label accuracy, within-alliance accuracy, held-out
fragment agreement, difficult-cluster result, route continuity, custody, teleport gaps,
and the three known hard fragment conflicts. Compare:

1. legacy gallery;
2. reviewed seed gallery at each maturity level;
3. reviewed gallery plus presence/alliance restrictions;
4. joint path-flow with reviewed appearance evidence;
5. audited identity ceiling.

Also report relay bytes per review decision and replay CPU time per published gallery
version. Capability is not accepted if accuracy improves by violating the 4 MiB bundle
cap, repeatedly solving unchanged matches, or losing provenance.

## 13. Delivery phases

1. **Safety boundary:** mark legacy unreviewed; remove automatic authoritative learning.
2. **Local vertical slice:** prepare/apply/publish from files, season manifest, reviewed
   multi-prototype votes, and qm15 A/B—no relay/UI dependency.
3. **Tracks-tab review:** bounded embedded-thumbnail bundles, drafts, submit, relay kinds.
4. **Operational loop:** watcher ingestion, batched publication, maturity/team states.
5. **Replay:** vote-diff gate, dependency queue, reports, operator-controlled republish.
6. **Durability:** content-addressed remote crops, cross-event season use, robot revisions.

Do not begin with replay or remote object storage. First prove that reviewed prototypes
can be generated, revoked, rebuilt, and outperform the legacy gallery locally; otherwise
the system would efficiently distribute and replay evidence that has not been shown to
be better.
