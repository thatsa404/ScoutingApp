# Route quality review

Reviewed `ROUTE_QUALITY.md` and traced its claims into the current implementation.
This is a compact investigation handoff, not a replacement for the original evidence.

## What the route product means

The app-facing route is `public/tracks/<matchKey>.json`. It contains six team-keyed
polylines in WPILib 2026 field metres, sampled at 5 Hz, with custody, gaps, camera
visibility, occluder shadows, and quality counters. `main.js` draws the route by team,
breaks lines across recorded gaps, and overlays camera-blind and structure-hidden areas.

A visible bad route can originate in different layers:

```text
wrong box -> wrong tracker fragment -> wrong stitched identity
          -> wrong team assignment -> wrong field projection -> wrong export/render
```

The layer matters: changing export smoothing cannot repair an identity fusion, and
changing CP-SAT cannot repair a detector box on a person or a bad homography.

## Confirmed lessons from the investigation

The five previously found bugs are coherent and all have concrete code locations:

1. `pipeline.py` now projects stitched tracks into `prepos.json` before `robots.py`.
   Without that, the kinematic and duplicate geometry constraints were inert on first
   solves.
2. `robots.geometric_duplicates()` no longer trusts low-IoU pairs across cameras;
   `DUP_IOU` is 0.35, with horizontal separation/vertical overlap doing the actual
   low/high duplicate test.
3. `solve._forbid_all_teams()` now forbids a kinematically impossible pair from sharing
   a pinned team too. Only the explicit “both pinned to the same team” contradiction
   can override that geometry.
4. `solve.distance_budget()` uses the measured displacement envelope rather than a
   linear top-speed bound that became larger than the field diagonal after ~2.5 s.
5. `robots.split_impossible_steps()` now keeps one lowest-box position per
   `(track, frame)`. A duplicate/rebound track can carry two boxes in one frame; reading
   both made one real jump look like 1,291 impossible steps on `qm10`.

These fixes establish a useful pattern: validate that the intended constraint is active,
then measure the exact rows/positions that the next stage consumes.

## Open head: `occluder_rebind`

`robots.occluder_rebind()` runs before heuristic splits, after stitched tracks and cached
CNN appearance descriptors exist. It can merge a later track into an earlier track when:

- both endpoints lie at the same marked occluder;
- the gap is within the transit budget, or `parked_at()` says the robot stayed parked;
- the appearance distance is at most `REBIND_MAX_COS = 0.70`; and
- the winner beats the runner-up by `REBIND_MARGIN = 1.25`.

The dangerous branch is the parked exception: a robot may be hidden for up to the parked
limit (currently 180 s), and “same place before/after” is weak evidence at a hub used by
several robots. A false merge buys custody but irreversibly combines two identities;
an unmerged track leaves an honest gap that later stages can see.

On `2026mawor_qm10`, the observed bad joins are long and concentrated at the blue hub:
3.7 s, 9.3 s, 39.7 s, and 58.5 s. The worst fused tracks produced an overlong
3,321-detection object and a blue-alliance identity failure. The single measured A/B
run with rebind disabled improved both curator alignment (97% -> 98%) and mean custody
(0.824 -> 0.866), so the default must not be defended by custody alone.

## Symptom-to-layer triage

| Symptom | First place to inspect | Why |
|---|---|---|
| Teleport inside one continuous track | `split_impossible_steps`; render the two frames | Usually a detector false positive or a bad box, not projection/CP-SAT |
| Teleport between two tracks with the same team | `kinematic_conflicts`, `pair_forbidden`, pins | The solver may have combined sequential tracks that cannot be one robot |
| Gap near a hub/trench | `occluder_rebind`, `hold_cells` | Rebind may have absorbed the track into another, or a real hidden robot may be parked |
| Swap that persists after the gap | Rebind and `split_chimeras` / appearance timeline | A fused track corrupts all downstream team votes |
| One robot appears twice | `geometric_duplicates`, curator duplicate labels, co-detection | Low/high duplicate boxes must be merged; separate robots must not be fused |
| Route absent for a visible robot | parking, custody window, `viewcheck`, camera visibility | Absence can be unassigned, filtered, outside the calibrated view, or genuinely occluded |
| All positions look globally wrong | `calib/<camera>.json`, lens model, `prepos` | Projection must be validated before interpreting identity metrics |

Track IDs in bug reports are not durable. Report the full match key, team, match-relative
time window, symptom (`teleport`, `gap`, `swap`, `ghost`, `double`), and physical context.

## Investigation protocol

1. Start from the exported route and record the exact team/time symptom in match seconds.
2. Re-run the solver with the exact stitched tracks, calibration stem, prepos file,
   identity file, corrections, `--seed 0`, and `--no-hint` as specified in `ROUTE_QUALITY.md`.
3. Re-project and re-export. Do not interpret a changed labelled file without rebuilding
   `positions.json` and the route export.
4. Inspect logs for: whether positions were supplied; `pair(s) forbidden`; duplicate
   merges; rebind joins and cosine/runner-up values; within-track cuts; custody conflicts.
5. Render the two relevant source frames with ffmpeg. Treat percentile tables as search
   tools, not explanations.
6. A/B one mechanism at a time: `--occluders __none__`, `--step-cut 0`, or
   `--kin-hard 0`. Preserve a run manifest containing command, inputs, output metrics,
   and the route file used for visual inspection.

## Metrics to trust

- **Curator alignment** is the primary correctness metric when corrections exist. It is
  the only reported measure that does not improve simply by labelling less.
- **Mean custody** measures route availability over the match-relative window. It is
  useful alongside alignment, not as a substitute for it.
- **Kinematic violations and teleport counts** are diagnostics. Measure route jumps by
  distance and inspect them; dividing noisy short steps by a small `dt` manufactures
  alarming speeds.
- **Coverage** is not cross-camera comparable when false detections inflate its
  denominator. Prefer custody and always record the window source.

## Recommended next experiment

Measure the parked rebind policy before changing its default:

- MAWOR: all 13 curated matches, with rebind on/off and the candidate parked-evidence
  rules held separately.
- NE championship: a representative curated sample after the current geometry/motion/
  pin changes, because its camera and occluder layout differ.
- For each run, compare curator alignment, mean custody, longest gaps, rebind count,
  gap duration, within-track cuts, custody conflicts, and route images around each join.

The first safe decision is not “delete the parked rule”; it is whether its evidence can
be made selective enough that long hidden intervals improve both metrics without fusing
different robots. If not, disabling it by default is better than silently rewriting
identity history, with explicit opt-in for cameras where it is validated.

## Baseline warning

The current working tree is not a clean baseline: `robots.py`, the MAWOR calibration,
the MAWOR `qm10` export/index, occluders, and correction files have uncommitted changes.
The current export includes 52 kinematic violations, and the working calibration has a
7.936 m maximum reprojection-error outlier. Any experiment must name its exact input
files and avoid comparing these outputs to older published routes without re-deriving
them.

## Current curated MAWOR route audit

Audited the 15 current curated exports listed in `public/tracks/index.json` (`qm1` through
`qm15`). The checks were performed on the published 5 Hz routes using match-relative time.
I searched for gaps of at least 5 seconds and adjacent exported samples moving at least
4 m in at most 1 second, then inspected source frames for representative cases. The
short-step screen is deliberately conservative: a large displacement can be a detector
handoff, a projection error, or a real identity problem; it is not proof of teleporting.

### Priority findings

| Match | Finding | Assessment |
|---|---|---|
| `qm14` | Teams `8567` and `8724` were not on the field; their long gaps are expected. The four robots actually present have custody `1768=.7796`, `4628=.8093`, `10254=.6973`, `716=.7938`. | **Not a primary route failure.** The six-robot mean is artificially depressed by two absent robots. Keep `4628`'s 6.98 m jump as a secondary check, but prioritize `qm15`. |
| `qm15` | Mean custody is 0.5945 with 13 gaps ≥5 s. Team `1027` has a 33.9 s pre-match gap and another 11.0 s gap; team `10393` has a 26.0 s gap. There are also several abrupt jumps: `2168` moves 9.32 m in 0.20 s at `149.6–149.8`, `190` moves 8.12 m in 0.267 s at `130.8–131.066`, `157` moves 7.99 m in 0.267 s at `130.866–131.133`, and `1027` moves 10.81 m in 1.0 s at `145.066–146.066`. | **Highest teleport/identity risk.** Most jumps cross `tid` values, so false handoff or identity fusion is more likely than a single smooth-track motion. The route needs source-frame and solver-join inspection. |
| `qm6` | Team `8544` has a 21.7 s gap at `133.833–155.566`, plus a 10.07 m jump in 0.267 s at `159.5–159.766`. The jump remains on `tid 743` on both sides. | **Strong within-track teleport candidate.** Because the track ID does not change, prioritize box/projection/rebound data and `split_impossible_steps`, not only team assignment. |
| `qm10` | Team `1768` has a 12.8 s gap at `5.8–18.6`; team `10254` jumps 6.64 m in 0.267 s at `43.466–43.733`. | **High-priority discontinuity.** The jump crosses `tid 309 → 302`, making a detector handoff or identity join plausible. |
| `qm12` | Team `5422` has an 11.1 s gap at `18.3–29.367`, custody 0.6813, and a 4.34 m jump in 0.267 s at `52.566–52.833`. The same `tid 444` appears on both sides. | **Moderate/high within-track candidate.** Similar to `qm6`, inspect the source detections and projection before changing identity constraints. |
| `qm13` | Team `1757` has a 9.4 s gap at `15.0–24.4`; team `1100` jumps 4.27 m in 0.20 s at `39.8–40.0`, remaining on `tid 12`. | **Moderate within-track candidate.** The same-track jump is too large for a normal 0.2 s motion step and merits a frame-level check. |

### Per-match triage

| Match | Route condition | Most useful next check |
|---|---|---|
| `qm1` | No gap ≥10 s. Team `716` has the largest gap (9.13 s); team `5422` has a 4.04 m / 0.267 s jump at `116.9–117.166`. | Low priority; inspect `5422` only if the rendered route visibly kinks. |
| `qm2` | No gap ≥10 s. Team `190` has a 5.87 s gap; team `6762` has a 5.59 m / 0.533 s jump at `96.5–97.033`. | Check whether the jump is a handoff or a route gap hidden by the export threshold. |
| `qm3` | One 9.27 s gap on team `3634` (custody 0.5973); no short-step jump ≥4 m within 1 s. | Treat as a visibility/coverage case, not a teleport case. |
| `qm4` | One 10.33 s gap on team `2370`; team `8724` has a 4.08 m / 0.20 s jump at `56.733–56.933`. | Inspect `8724` source boxes and the `tid 91 → 113` handoff. |
| `qm5` | Team `10393` has a 38.27 s gap at `116.2–154.466` (custody 0.6262); teams `1474` and `5347` also have long gaps. | Highest gap-priority after `qm14`/`qm15`; determine whether `10393` is genuinely hidden or simply unassigned. |
| `qm6` | Ten gaps ≥5 s, including team `8544` at 21.73 s and team `1100` at 16.80 s. | Investigate `8544` jump and its long preceding gap together; they may share a rebind/track-history cause. |
| `qm7` | No gap ≥10 s. Largest gap is 8.53 s on team `190`; no short-step jump ≥4 m within 1 s. | Routine review only. |
| `qm8` | No gap ≥5 s. Largest discontinuity is 4.15 m over 0.933 s on team `6762`, which is not an extreme speed by itself. | Lowest route-risk export in this audit. |
| `qm9` | Three gaps ≥5 s, largest 6.20 s on team `467`; no short-step jump ≥4 m within 1 s. | Routine review only. |
| `qm10` | Three gaps ≥5 s, including team `1768` at 12.8 s; 6.64 m / 0.267 s jump on `10254`. | Inspect the `10254` cross-track handoff and compare with `occluder_rebind` A/B. |
| `qm11` | Three gaps ≥5 s, including team `2370` at 10.6 s; no short-step jump ≥4 m within 1 s. Kinematic-violation count is nevertheless high (61). | Review violation rows even though no exported adjacent jump crossed the screen threshold. |
| `qm12` | Five gaps ≥5 s, including team `5422` at 11.07 s; same-track 4.34 m / 0.267 s jump. | Inspect within-track cut behavior and whether the route should be split at this step. |
| `qm13` | Six gaps ≥5 s; largest is 9.4 s on team `1757`; same-track 4.27 m / 0.20 s jump on `1100`. | Inspect the `1100` source box/projection pair first. |
| `qm14` | Teams `8567` and `8724` were absent. Among the four robots present, custody is roughly 0.70–0.81; team `4628` still has a 6.98 m / 0.334 s jump. | Use as a secondary control, focused on `4628`; do not treat the absent blue robots as missing-route defects. |
| `qm15` | Thirteen gaps ≥5 s, four ≥10 s; lowest custody after `qm14`. Multiple cross-track jumps cluster late in the match. | Highest identity/teleport investigation priority. Inspect every listed jump around the same solver run. |

### Interpretation and caveat

The strongest teleport-like signatures are `qm6/8544`, `qm12/5422`, and `qm13/1100`
because the reported track ID stays constant across the jump. The strongest identity
handoff signatures are concentrated in `qm15`, followed by `qm10/10254`, `qm14/4628`,
and `qm4/8724`. Long gaps should not be filled automatically: the route exporter only
breaks the line when the interval exceeds one second, so a short handoff can appear as a
continuous line with a large segment.

All 15 current MAWOR exports carry the same working calibration metadata, including the
known 7.936 m maximum reprojection-error outlier. Consequently, global projection error
is a live confounder for every spatial diagnosis. The findings above are triage targets
against the current published files, not final labels of tracker bugs. Before changing
the solver, re-run the exact route-quality command, inspect `prepos`/source detections,
and compare the route with the clean calibration or a calibration A/B.

## Follow-up: lineup context for `qm14` and `qm15`

The original gap ranking needs one important correction from video review. In `qm14`,
blue teams `8567` and `8724` never entered the field; only blue `1768` was present.
Their long gaps are therefore expected, and the low six-team mean custody is not a fair
measure of tracking quality for that match. Recomputed over the four robots actually on
the field, custody is approximately 0.70–0.81. `qm14` remains useful as a secondary
control for the `4628` discontinuity, but it is no longer the primary investigation case.

`qm15` is the stronger controlled test: all six teams are present, yet custody is only
0.5945. The per-team custody values are:

```text
2168  0.6499    190  0.6450    157  0.6677
1027  0.3003   10393  0.7241    237  0.5802
```

The `qm15` correction file contains 70 human anchors. The largest disagreements with the
pre-curation labels are repeated mappings `157 -> 2168` (15), `2168 -> 157` (8),
`190 -> 157` (8), `1027 -> 10393` (8), and `2168 -> 190` (6). The `was` value is the
previous automatic label, not a second ground-truth label, but this concentration is
strong evidence that the poor route is dominated by identity swaps around crossings and
track handoffs. Anchors occur repeatedly through the match, roughly from match time 40 s
through 163 s, rather than in one isolated camera failure.

The next `qm15` investigation should therefore be organized around the curator anchors
and the route jumps together:

1. For each jump, resolve the source detection at both endpoints and record the track
   transition, detector confidence, box centre, and nearest curator anchor.
2. Check whether `cuts_from()` actually separates the contradictory labels, or whether a
   long track still carries multiple robots between anchors.
3. Compare the route with `--occluders __none__` and `--step-cut 0` one at a time. A
   custody increase without curator-alignment improvement is not a fix.
4. Only after identity joins are understood, revisit the calibration outlier; otherwise
   projection error and identity error will be mixed together.

The current `qm15` solver summary supports that prioritization. Its six final groups use
17–27 source tracks each, and the strongest vote pool is not clean: the group ultimately
assigned to `1027` contains votes for every team (`1027:51`, `10393:39`, `190:22`,
`237:24`, `157:19`, `2168:17`). The `2168` group has `2168:74` against `157:46` and
`190:26`, while the `157` group has repeated `157 <-> 2168 <-> 237` vote changes. There
are no final custody conflicts, so this is not simply two simultaneously visible robots
being assigned the same team; it is a fragmentation/identity-history problem upstream of
the final route labels.
