# QM15 route instability: root cause and change plan

This document records the investigation baseline for `2026mawor_qm15`. It is the
starting point for implementation work; `ROUTE_QUALITY.md` remains the general
route-debugging procedure, and `ROUTE_QUALITY_REVIEW.md` contains the broader MAWOR
audit.

## Conclusion

The unstable `qm15` routes are primarily an identity-history failure, not an export or
rendering problem. The current pipeline destructively joins weak tracker fragments,
then repeatedly splits and renumbers them using evidence that is still keyed to the
pre-join track IDs. By the time the global solver runs, it is choosing among more than
one hundred short, internally mixed fragments rather than six coherent robot histories.

The working MAWOR calibration is also unsafe for hard geometric decisions. It amplifies
fragmentation and invalid kinematic decisions, but it does not by itself explain the
observed team swaps.

## Evidence from QM15

The reconstructed pre-solve stage counts are:

| Stage | Live tracks | Detections | Change |
|---|---:|---:|---|
| Raw stitched input | 77 | 22,238 | — |
| Field filtering | 42 | 11,331 | 10,907 detections dropped |
| Geometric duplicate merge | 42 | 11,331 | no merges |
| Occluder rebind | 33 | 11,331 | 9 destructive merges |
| Alliance split | 35 | 11,331 | 2 new fragments |
| Appearance split | 79 | 11,331 | 44 live fragments added |
| Vote/chimera split | 127 | 11,331 | 48 live fragments added |
| Curator cuts | 129 | 11,331 | 2 cuts |
| Final exported solve | 134 | — | further deconfliction cuts |

At least three of the nine occluder rebinds are disproved directly by curator labels:

- track `22` (`2168`) absorbed track `32` (`237`);
- track `13` (`2168`) absorbed track `38` (`237`);
- track `42` (`10393`) absorbed track `65` (`237`).

Some stitched inputs are already chimeric before rebind. For example, track `17`
contains curator labels for `2168` and `157`; track `18` contains `10393` and `237`;
track `28` changes from `190` to `157` across a long internal gap. Rebind therefore
operates on evidence that is not guaranteed to describe one robot.

The final six groups each contain 17–27 fragments and heavily mixed vote pools. The
group assigned to `1027`, for example, includes votes for all six teams. The reported
second-solution uncertainty identifies only two tracks, so the current uncertainty
metric materially understates semantic route instability.

## Root issues

### 1. Rebind uses an incompatible appearance space

`out/stage3/2026mawor_head.npz` does not exist. MAWOR rebind therefore compares raw CNN
embeddings, while `REBIND_MAX_COS = 0.70` was calibrated for whitened embeddings from a
different event. Raw and whitened cosine distances have different distributions and
quality. On `qm15`, different robots consequently appear to be extremely strong matches
at distances such as `0.059` and `0.080`.

The optional join checker has the same model-compatibility problem and made no cuts in
the reconstructed run, including on curator-confirmed chimeras.

### 2. Rebind destroys track identity before downstream evidence is consumed

`occluder_rebind()` rewrites an absorbed track ID to the keeper ID. Appearance caches
remain keyed to original stitched IDs, so descriptors and cut points for absorbed tracks
become unreachable. The identity structure merges `tally` and `voteList` but not every
timeline/provenance field used by later splitters. Later retallying repairs only some of
this state, depending on whether another splitter happens to cut the affected track.

This creates hidden, order-dependent coupling between rebind, appearance splitting,
vote splitting, curator cuts, and deconfliction.

### 3. Repeated mutation turns six histories into a fragment-assignment problem

After rebind reduces the input from 42 to 33 tracks, heuristic splitters expand it to
127 tracks before curator cuts and 134 by final output. The appearance stage reports 94
cuts/new IDs but produces only 44 additional live tracks, evidence that many proposed
cuts target IDs or time ranges no longer present after earlier mutations.

The global solver can remove simultaneous same-team conflicts, but it cannot reconstruct
identity evidence that was lost or attached to the wrong fragment. Zero final custody
conflicts therefore does not imply coherent routes.

### 4. Calibration quality is allowed to drive hard decisions

The working `2026mawor` calibration reports mean reprojection error `0.619 m` and maximum
error `7.936 m`, with no lens model. The previous committed calibration reported mean
`0.0724 m`, maximum `0.229 m`, and included a lens model. Despite the outlier, the working
calibration feeds field filtering, projected positions, kinematic exclusions, and
impossible-step splitting.

Bad calibration can remove valid detections and manufacture impossible motion. It must
not silently participate in hard constraints.

### 5. Run provenance is insufficient for reliable comparisons

Outputs do not provide one durable manifest tying together the exact command, source
hashes, calibration, appearance-head identity, occluder definition, corrections,
stage counts, joins, cuts, and solver settings. This makes it too easy to compare routes
that were produced by materially different evidence.

## Recommended implementation order

### Phase 1: stop unsafe behavior and make failures visible

1. **Fail closed on appearance-model mismatch.** Store an embedding-space version and
   whitening-head identity in appearance caches. Disable appearance-based rebind and
   join thresholds unless their calibration metadata matches the cache exactly.
2. **Gate calibration before geometry is used.** Reject or quarantine calibrations with
   excessive maximum/percentile reprojection error, invalid outliers, or incompatible
   lens metadata. Do not use a failed calibration for field filtering or hard kinematic
   constraints.
3. **Add a route validator.** Treat physically impossible adjacent exported positions as
   a failed route segment and emit a gap plus a diagnostic. This is a safety net, not an
   identity fix.
4. **Write a run manifest.** Record input hashes, command/arguments, calibration and head
   IDs, occluder/correction hashes, per-stage counts, proposed joins/cuts, solver
   objective, alignment, custody, and route validation results.

### Phase 2: remove destructive identity coupling

5. **Keep immutable detection and source-track IDs.** Every derived segment should carry
   explicit provenance instead of replacing the original ID.
6. **Represent rebinds as continuation hypotheses.** An occluder/appearance join should
   be a scored edge available to the optimizer, not an irreversible pre-solver fusion.
   Curator contradictions must be able to reject the edge cleanly.
7. **Key all evidence to immutable IDs.** Appearance descriptors, positions, votes,
   timelines, corrections, and diagnostics must survive segmentation without ad-hoc
   remapping.

### Phase 3: simplify segmentation and solving

8. **Build one segmentation plan.** Collect alliance, appearance, vote, kinematic, and
   curator cut proposals against immutable tracks, reconcile them, and apply the plan
   once. Avoid repeated ID mutation and conditional retallying.
9. **Separate evidence generation from optimization.** Construct a segment graph whose
   nodes are immutable segments and whose typed edges carry temporal, geometric,
   appearance, occluder, and curator evidence. Solve continuity and team assignment from
   that graph.
10. **Make deconfliction deterministic.** Do not let solution-search order decide which
    new cuts are introduced. Candidate cuts should be generated before optimization and
    included in the manifest.

### Phase 4: improve quality metrics

11. **Report identity-history diagnostics.** Include mixed-vote entropy, contradictory
    curator spans, selected weak continuation edges, and fragment count per team. The
    current second-best-solution metric alone misses obvious instability.
12. **Make custody presence-aware.** Record known lineup/presence state so absent robots,
    such as two blue teams in `qm14`, do not depress route-quality summaries.

## First regression experiment

Before changing architecture, preserve this snapshot and run an isolated `qm15` A/B
with identical inputs and seed:

- baseline behavior;
- occluder rebind disabled;
- only then, a model-compatible rebind implementation.

Compare curator alignment first, then mean custody, longest gaps, impossible route
steps, false joins, fragment counts, vote mixture, and rendered routes around the known
jumps. Do not overwrite the current stage outputs while producing this comparison.

Known exported jump windows to retain as regression cases include:

- `1027`, match seconds `145.066–146.066`, `10.81 m`;
- `2168`, `149.600–149.800`, `9.32 m`;
- `190`, `130.800–131.066`, `8.12 m`;
- `157`, `130.866–131.133`, `7.99 m`;
- `190`, `160.866–161.133`, `7.52 m`;
- `157`, `79.066–79.533`, `7.42 m`.

These cases are mostly cross-track handoffs. A successful change should improve route
continuity without merely hiding the jumps through smoothing or increased gap filling.
