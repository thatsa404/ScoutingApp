# Bumper recognition: what was planned, what was measured, what to do next

Status: **OCR is REMOVED from the live path** (`rtrack.identify` and `rtrack.assign`
deleted, `easyocr` dropped from `pyproject.toml`). Everything below the next section is
the record of how it was built and measured, kept because the measurements are what
justify the removal and because a future attempt should not repeat the dead ends.

---

## Why it was removed

Two independent reasons, both measured, not argued.

**It did not contribute.** With curator corrections loaded, blanking `identity.json`
and re-running produced an *identical* result: 97% coverage, 76% mean custody, 0
custody conflicts, 11 induced cuts, 7 crossing-explained clashes. Only the demoted-clash
count moved (6 vs 5). Running it with the file deleted entirely reproduces this exactly.

**It was not accurate enough to be worth its cost.** Scored against the 122 curator
labels, with the alliance gate applied and abstention counted as a miss:

| | per track | detection-weighted |
|---|---|---|
| OCR top vote | 35% (9/26) | 57% |
| appearance centroid, trained on half this match | **79%** (27/34) | **88%** |

It abstained completely on 7 of 26 tracks. And it cost **3m16s** of a ~6 min pipeline —
the single largest block in a latency budget whose whole purpose is getting frames in
front of a curator while the match is fresh in their memory.

`--solver greedy` named groups from pooled OCR votes, so it dies with OCR. The default
is now `--solver cpsat`, which is also the only solver that honours curator pins.
`identity.json` is still *read* when present, so old runs reproduce; its absence is now
the normal case rather than an error.

### What replaces it: a per-event appearance classifier (measured, promising, not built)

`rtrack.appear` already caches an HSV histogram of each detection's **superstructure**,
deliberately excluding the bumper band (`BODY_TOP, BODY_BOTTOM = 0.02, 0.58`). That
exclusion is what makes the idea viable across a whole event: FRC teams swap red and
blue between matches, so any descriptor that reads the bumper learns the alliance, not
the robot.

Labelling those descriptors with curated output (3891 crops, 6 teams, balanced) and
training a nearest-centroid classifier gives the table above. Note the split matters
enormously — a random split reports 60% where an honest time-based split reports 54%,
because adjacent frames are near-duplicates.

**The confound worth knowing about.** Train on one half of the field and test on the
other and within-alliance accuracy falls from 74% to 54–57%. So a real part of the
within-match signal is lighting and camera position, not robot identity. It is still
above the 33% within-alliance chance baseline, so there *is* genuine appearance signal —
just less than a time-split alone suggests.

**The decisive test has not been run**, and one match cannot run it: does a classifier
trained on match A identify the same robots in match B? That is the actual claim behind
"increasingly autonomous as an event unfolds". Curating a second match from the same
event answers it in about 90 seconds of human time. The field-side result above is a
mild prior *against*, so the test should be run before any of this is built.

### Descriptor ablation — does spatial structure help?

All colour variants spend the same ~384 dimensions, so a win is attributable to
*layout*, not to a bigger vector. `time` = train on first half of the match; `side` =
train on one half of the field, test on the other (the robustness proxy).
Within-alliance, chance = 33%.

| descriptor | dims | time | side |
|---|---|---|---|
| global HSV (the old one) | 384 | 73% | 56% |
| **3 horizontal bands HSV** | 384 | **77%** | **58%** |
| 3×2 grid HSV | 384 | 69% | 50% |
| 3×3 grid HSV | 288 | 67% | 53% |
| 3×3 gradient only | 72 | 48% | 39% |
| 3×2 rg-chromaticity | 600 | 54% | 44% |
| 3-band HSV + gradient | 456 | 78% | 56% |

Three results worth keeping:

**Bands beat grids, and rotation is part of why.** Robots turn, so binning left-to-right
scrambles a robot's own structure while top-to-bottom survives it. Bands beat columns at
every band count (72/68, 77/75, 75/70). But rotation is not the whole story — a 3×2 grid
(69%) scored worse than *either* of its factors alone, which is bin fragmentation:
splitting the pixel budget two ways leaves each cell's histogram too noisy. 3 bands is
the measured optimum; 2, 4, 6 and 8 are all worse.

**The robust features are weak and the strong features are fragile.** Gradient-only and
rg-chromaticity have by far the smallest time→side drop (+9%, +10%, against +18% for
HSV), which is exactly what cross-match transfer needs — but they discriminate barely
above chance. Fusing gradient into the colour descriptor bought +1% on time and cost
−2% on side, so it was rejected. Nothing tried here closes the condition gap.

**Data volume is not the constraint.** Learning curve, training on the first half:

| labelled crops | per-detection | per-track | detection-weighted |
|---|---|---|---|
| 30 | 61% | 65% | 73% |
| 60 | 70% | 78% | 85% |
| 250 | 74% | 79% | 88% |
| 1945 (all) | 77% | 79% | 88% |

It saturates at **~250 crops, roughly 40 per robot** — one match supplies 8× that. More
curation will not raise this ceiling. The representation is the ceiling.

### The representation is where the headroom is

An ImageNet ResNet18 embedding, penultimate layer, **no fine-tuning and no labels used
to build the features**, swapped in for the histogram:

| | time | side | per-track | det-weighted |
|---|---|---|---|---|
| 3-band HSV | 77% | 58% | 79% | 88% |
| ResNet18, untuned | 70% | 54% | **85%** | **91%** |

Worse per detection, better per track — its errors are less correlated within a track,
so aggregation cleans them up, and the per-robot spread is far flatter (58–77% against
53–85%, where the histogram collapses to 53% on 9644). An untuned generic embedding
already beating a hand-tuned descriptor on the operational metric is the strongest
evidence that a *fine-tuned* embedding is worth building. It also means per-detection
accuracy is the wrong number to optimise — always report the track-level one.

What did not change: the side-split gap, for either representation. Condition
robustness remains the open risk, and only a second curated match can measure it.

### Alliance invariance — the leak, and the fix

Teams switch red/blue between matches, so any alliance component in the descriptor
breaks centroid reuse across an event. The crop already excludes the bumper, but colour
leaks in anyway. Predicting *alliance* from the bumper-excluded descriptor (chance 50%):

| | alliance predictable at |
|---|---|
| global HSV | 64% |
| 3-band HSV | 62% |
| &nbsp;&nbsp;band 1 (top) | 59% |
| &nbsp;&nbsp;band 2 (middle) | 58% |
| &nbsp;&nbsp;**band 3 (bottom)** | **69%** |
| ResNet18 embedding | 72% |

The leak concentrates in the band nearest the bumper, as expected from reflection onto
the superstructure and alliance-coloured field structure inside a crop the robot does
not fill.

**Cropping the leak away does not work.** Every crude fix cost more team accuracy than
it removed leak, because band 3 carries real robot signal too:

| variant | alliance | team time | team side |
|---|---|---|---|
| 3 bands, as shipped | 62% | 77% | 58% |
| 2 bands (bottom dropped) | 63% | 72% | 53% |
| band 3 weighted 0.5 | 59% | 72% | 56% |
| `BODY_BOTTOM` 0.50 | 60% | 74% | 57% |
| `BODY_BOTTOM` 0.45 | 59% | 72% | 56% |

**Projecting it out does work, and the spatial layout is what makes it possible:**

| variant | alliance | team time | team side |
|---|---|---|---|
| 3 bands + proj | **50%** | 78% | 59% |
| 3 bands + cent | 53% | 76% | 62% |
| **3 bands + proj + cent** | **48%** | 77% | **64%** |
| global HSV + proj | 61% | 72% | 56% |

`proj` removes the red-minus-blue mean direction; `cent` subtracts the alliance mean.
Three things to keep:

1. **Alliance drops to chance (50%) on the 3-band descriptor and only to 61% on the
   global one.** In a global histogram the alliance component is entangled across the
   whole vector and one linear direction cannot capture it. Banding concentrates it, so
   it can be projected away cleanly. *The spatial structure is what makes the
   descriptor de-aliancable* — that is a stronger reason to keep bands than the +4
   points of raw accuracy.
2. **It costs nothing.** Team accuracy went *up* (77→78 time, 58→59 side).
3. **`proj + cent` gives the best cross-condition number in this whole investigation:
   side-split 58% → 64%.** That is the metric standing in for cross-match transfer.

Both transforms are **free of curator effort**: `proj` needs red/blue labels, which come
from the detector's own bumper class, and `cent` needs none at all. Better still, `cent`
is computed *per match* from that match's own detections, so it normalises out that
match's exposure and white balance along with the alliance — which is why it helps the
condition split so much. Neither belongs in `appear.descriptor` (both need a fitted
direction); both belong in the classifier that consumes it.

The same projection removes the CNN's larger leak (72% → 49%), but the CNN's
per-detection numbers stay below the histogram's (68% time / 52% side). Its advantage
remains the track-level one.

### The cross-match test was run. It FAILED.

Second match curated: `2026necmp1_sf11m1` (Burns Division Match 11, video `mnDNxmczFU0`).
Teams 1768 / 5687 / 9644 wear **blue** in f1m3 and **red** in sf11m1, and in sf11m1 they
are all on one alliance, so the alliance gate reduces the task to "which of these three
is which". Chance = 33%.

| variant | per-detection | per-track | det-weighted |
|---|---|---|---|
| raw 3-band | **24%** | 9/24 | 21% |
| + proj | 25% | 8/24 | 20% |
| + cent | **48%** | 11/24 | **53%** |
| + proj + cent | 42% | 10/24 | 51% |
| *control:* same-match | **74%** | 12/17 | **83%** |
| *control:* shuffled labels | 33% | 5/24 | 34% |

The raw descriptor transfers **below the chance floor** — confidently wrong, not merely
uninformed. The diagnostic says why: match the flipped robots' crops against all six
match-A centroids with no alliance gate, and

* **53% land on a match-A *red* team** — wrong robot, right colour
* **17% land on their own true identity**

against 19% / 72% for the same crops inside match A. So the bumper-excluded crop still
carries enough alliance colour that changing bumper colour makes a robot resemble a
*different* robot of the new colour more than it resembles itself. The 62% alliance
predictability measured earlier is not a minor contaminant; across matches it is the
dominant signal.

`cent` — per-match, per-alliance mean subtraction — is the only variant that clears
chance, and it still reaches only 48%/53% against a 74%/83% same-match reference.
`proj` did not help and hurt slightly when combined.

**This test is confounded three ways** and cannot be read as "alliance flip kills it"
alone. sf11m1 also changes the field and camera (Burns Division, not the finals field)
and the detections are much noisier: **23% of curated boxes are not robots, against 5%
in f1m3** — 4.7× more false positives, which is also why stitching saw 108 track ids
against 32. End-to-end that video reaches only 73% coverage and 57% mean custody,
against 97% and 76%.

### Then colour was removed entirely, and everything got better

The obvious response to "alliance colour dominates" is to stop looking at colour. Tested
on the same alliance-flip pair. `alli` = alliance predictability (50% = colour-blind),
`same` = 3-way inside f1m3, `xfer` = 3-way f1m3 → sf11m1 across the flip, `6-way` =
within-alliance 6-team inside f1m3, which is the pipeline's real task.

| descriptor | dims | alli | same | xfer | xfer-wtd | 6-way |
|---|---|---|---|---|---|---|
| 3-band HSV (shipped) | 384 | 72% | 74% | **24%** | 21% | 77% |
| 3-band HSV + cent | 384 | 52% | 73% | 48% | 53% | 76% |
| 3-band S+V (no hue) | 360 | 53% | **85%** | 58% | 61% | 80% |
| **3-band grayscale, 16 bins** | **48** | 60% | 82% | **73%** | **84%** | 81% |
| 3-band grayscale, 32 bins | 96 | 57% | 85% | 69% | 70% | 83% |
| 3-band gradient only | 48 | 51% | 50% | 29% | 40% | 55% |
| 3-band gray + gradient | 432 | 57% | 81% | 52% | 55% | 81% |

**Grayscale is not a trade. It wins on every axis at once** — transfer 24% → 73%,
same-match 74% → 82%, and the pipeline's 6-way task 77% → 81%, in **48 dimensions
instead of 384**. The shuffled-label floor came back at exactly 33%, so this is real.

Coarse bins beat fine ones for transfer (16 bins 73%, 256 bins 67%), which is the
expected direction: coarse luminance bins are insensitive to the exposure and white
balance differences between one venue's camera and another's. Bands matter much less
than they did for colour — 2, 3 and 4 are within a point of each other, 1 band is ~6
worse — so the rotation argument still applies but weakly.

Why colour was actively harmful, not merely redundant: a colour histogram spends most
of its dynamic range on the alliance and on the venue's white balance, both of which are
noise for identity. Luminance structure — dark intake, bright polycarbonate, the
vertical arrangement of mechanisms — is the robot's actual physical layout, and it is
stable across both.

### The clean cross-match test: f1m2. It passes.

`2026necmp_f1m2` — same six teams, **same alliances, same camera**, minutes from f1m3.
This isolates plain cross-match transfer from both confounds in the sf11m1 test. All six
teams are usable, so this is the pipeline's real 6-way within-alliance task.

| train → test | per-det | per-track | det-weighted |
|---|---|---|---|
| **gray16** f1m3 → f1m2 | 85% | 58/60 | **99%** |
| **gray16** f1m2 → f1m3 | 80% | 64/70 | **94%** |
| gray16 same-match reference | 81–82% | | 95–98% |
| HSV f1m3 → f1m2 | 80% | 51/60 | 92% |
| HSV f1m2 → f1m3 | 73% | 57/70 | 87% |

**Cross-match transfer is as good as same-match.** 99% and 94% detection-weighted,
against a 95–98% same-match reference and OCR's 57%. The premise behind the per-event
classifier holds.

Reading the three tests together gives the decomposition:

| condition | gray16 det-weighted |
|---|---|
| same match | 95–98% |
| different match, same camera, same alliance | **94–99%** |
| different match, different field/camera, alliance flipped | **84%** |

So plain cross-match transfer costs essentially nothing, and the sf11m1 shortfall is
attributable to the flip plus the camera change — not to cross-match transfer as such.
16-bin grayscale also repaired the weak robot there: 9644 went 34% → 53%.

Caveat worth keeping: f1m2 and f1m3 are finals minutes apart on one field, which is the
*easiest* cross-match case. sf11m1 at 84% is the better estimate of the hard case.

### Adopted

`appear.descriptor` is now 3 bands × 16 grayscale bins (48 dims, was 384), and
`--appear-thresh` defaults to **0.25** (was 0.55) because grayscale compresses the
Hellinger scale — at 0.55 it fires no cuts at all. The threshold was re-tuned on f1m3
end-to-end and then **validated on the two matches it was not tuned on**:

| match | coverage | mean custody | previously (HSV) |
|---|---|---|---|
| f1m3 | 98% | 77% | 97% / 76% |
| f1m2 | 99% | 61% | 97% / 60% |
| sf11m1 | 78% | 61% | 73% / 57% |

All three improved; all three remained custody-conflict-free.

### What this changes

The fix is **to stop using colour**, not to correct for it. `proj`, `cent` and tighter
crops all tried to subtract the alliance component out of a colour descriptor; the best
of them reached 53% detection-weighted transfer. Simply not looking at colour reaches
84% and is better within a match as well.

**Per-(team, alliance) centroids remain the belt-and-braces fallback** if a residual
colour dependence shows up: a team plays ~12 qualification matches, roughly half on each
colour, so after a few curated matches both colours are covered and a flip is looked up
rather than extrapolated. Grayscale looks likely to make that unnecessary.

That reframes the next test. `2026necmp1_f1m2` is the same six teams on the *same*
alliances with the *same* camera, which is exactly the per-(team, alliance) reuse case,
and it isolates plain cross-match transfer from both confounds above. If it scores near
the 74%/83% same-match reference, the classifier works across matches and only the flip
is fatal — which per-colour centroids solve. If it also scores near 50%, plain
cross-match transfer fails and the per-event classifier idea is in serious trouble
regardless of alliance handling.

### The classifier is wired into CP-SAT (`rtrack.reid`)

`rtrack.reid` replaces `rtrack.identify`. Two subcommands:

```
reid gallery <video> --event 2026necmp            # after a match is CURATED
reid votes <video> --event 2026necmp --match KEY  # before the NEXT one
```

The design decision that mattered: **`votes` emits exactly the shape `identify` did** —
`{"tracks": {tid: {"tally": {...}, "voteList": [[t, team], ...]}}}`. So `solve.py` and
`robots.py` needed **no changes at all**; `--identity <reid.json>` feeds the same
objective term the bumper votes used, and `robots.retally` redistributes votes across
track splits through the same timestamp machinery. Matching an existing interface beat
designing a better one.

Other choices, with reasons:

* **Voting reuses the cached `_appearance.npz`**, so it costs zero extra decodes on the
  critical path. Gallery building decodes once, but runs *after* curation, where latency
  does not matter.
* **Votes are cast only among the six teams TBA lists for that match.** An event gallery
  may hold forty teams; only six can be on the field. That closed set is free accuracy.
* **No alliance gating inside `reid`.** The solver already carries an alliance term worth
  250; gating votes too would count bumper hue twice and let a misread hue veto correct
  appearance evidence. Appearance says who it looks like; CP-SAT arbitrates.
* **Votes are capped per track** (`MAX_VOTES`, default 24) and spread evenly in time.
  Uncapped, one vote per detection would be hundreds, swamping every other term at
  `w.vote=10`. The cap is also honest: the learning curve saturated at ~40 crops.

**Result — gallery from f1m3 only, applied to f1m2 with NO corrections:**

| configuration | agreement with curated truth |
|---|---|
| no identity (geometry + hue only) | **13.3%** |
| appearance votes from one curated match | **84.2%** |

13.3% is what arbitrary naming gives: CP-SAT groups tracks correctly but has no basis to
*name* the groups. One curated match takes that to 84.2% on the next match with zero
human input. OCR's equivalent number was 52%.

(84.2% is the reproducible figure measured *after* the determinism fix below. Before it,
the same command returned anywhere from 70.5% to 84.2%.)

### A reproducibility bug found while measuring that

The number above is a range, not a point, and finding out why was the most useful thing
in this round. **Three runs of the identical command on byte-identical inputs scored
79.9%, 70.5% and 84.2%**, with coverage varying 11583 / 11958 / 11813.

Cause: `solve.py` ran CP-SAT with `num_search_workers = 8` against a **wall-clock**
limit. Whichever worker happened to be ahead when the clock expired supplied the answer,
so the team *naming* differed run to run even at identical coverage.

This invalidated a parameter sweep before it was believed — a `--max-votes` comparison
read 72% / 77% / 77%, entirely inside the 14-point run-to-run spread. **Any tuning done
before this was fixed would have been fitting noise.**

Fixed by counting search progress instead of seconds: `random_seed = 0`,
`interleave_search = True`, `max_deterministic_time = time_limit`, with a generous
wall-clock cap kept only as a safety net. Parallelism is retained.

Verified: three runs after the fix produced **byte-identical output** (same MD5 of the
full labelling, coverage 11813, agreement 84.2% each time). Note the stable value is the
*top* of the old range — the nondeterminism was costing accuracy, not just consistency.

**Lesson worth keeping:** before tuning anything in this pipeline, run the same command
twice and diff the output. The 52%, 69.4% and "93–94%" figures quoted earlier in this
document were all single samples from a distribution nobody had measured.

### Curated data as detector training data

Curated `notrobot` clicks are labelled negatives, and they multiply: a click anchors to a
detection, which belongs to a track, and the whole track is then negative.

| match | notrobot clicks | negative detections | share of all detections |
|---|---|---|---|
| f1m3 | 5 tracks | 273 | 2.2% |
| f1m2 | 2 tracks | 177 | 1.5% |
| sf11m1 | 23 tracks | 3425 | **22.8%** |

**That 3,875 figure is wrong, and the error is instructive.** It assumed a `notrobot`
click labels the whole track. Checked visually, it does not: these tracks DRIFT. Sampling
four of them across their lifetimes shows track 46 starting on a fuel pile at 86s and
sitting on robots by 104–143s; track 41 shows bumper "125" at 67s; track 23 shows "125"
again at 106s. Propagating the click track-wide would feed the detector real robots —
some with legible bumper numbers — as negatives, which would actively damage it.

The clicks themselves are sound. Rendering the exact crop at each clicked frame shows
fuel piles and a few field panels, which is precisely the false-positive population. It
is the propagation that is invalid.

What survives:

| match | clicks | track-wide (invalid) | within 3s of a click | within 1s |
|---|---|---|---|---|
| f1m3 | 6 | 273 | 224 | 84 |
| f1m2 | 2 | 177 | 78 | 26 |
| sf11m1 | 28 | 3425 | 1644 | 676 |
| **total** | **36** | ~~3875~~ | **1946** | **786** |

And even 786 is ~22 near-duplicate detections per click, so the effective independent
sample is closer to **36** than to anything in the thousands.

There is a further wrinkle: many of these are genuinely ambiguous rather than simply
wrong. A large share of the crops are fuel sitting in a robot's hopper or intake. It is
not obvious what the correct box even is for a robot whose hopper is full of game pieces,
so some of this may not be a labelling problem a detector can cleanly solve.

Two cheap shortcuts were tested first, and **both failed**:

*A geometry-and-motion filter* (box size, aspect, path length, speed, duration,
confidence, alliance-undecided share; logistic regression, leave-one-match-out) reached
only 58% / 82% / 76% track accuracy with negative recall of 20% / 50% / 72%. Not
deployable. With 33 robot and 25 non-robot tracks fitting 11 features, the coefficients
are not trustworthy either.

*A confidence threshold* looked promising because `mean_conf` carried the largest weight
in that model — but measuring the per-detection distributions directly killed it:
not-robot detections have median confidence **0.809** against robots' **0.858**, nearly
complete overlap. Cutting at 0.60 removes 26% of negatives while losing 5% of real
robots. **The detector is confidently wrong on field elements**, so no threshold helps.

Two hypotheses of mine were wrong here and the measurements corrected both: field
elements do *not* give themselves away by standing still (`med_speed` weight +0.26,
nearly useless), and the detector's own confidence does *not* separate them.

**Verdict on the question "is there an empirical case for improving robot/non-robot
detection from curated data?" — partial, and weaker than the raw counts suggest.**

Established:
* The problem is real and venue-dependent: 23% false-positive boxes on the unfamiliar
  field, against 5% and 2% on the two finals videos.
* The cheap fixes are ruled out by direct measurement, not by argument.
* The curator's clicks are accurate and cost nothing beyond curation already being done.
* Hard negatives are most valuable exactly where a detector is *confidently* wrong,
  which is what was measured here.

Not established, and the honest gaps:
* **The data is ~36 independent examples, not thousands.** Per match that is 2–28
  clicks. Across a full event (≈12 quals) it would be a few hundred, which is a usable
  hard-negative set — so the case is much stronger per *event* than per *match*.
* **Fine-tuning has not been shown to help.** Nothing here measures that.
* **Some of the population may be irreducibly ambiguous** (fuel inside a robot).
* The claim that false positives *cause* the downstream degradation is **not**
  established. sf11m1 also changed field and camera, and an attempt to separate the two
  by recomputing coverage over non-flagged detections failed on a track-id-space error.
  Treat "23% FP caused 78% coverage" as a hypothesis, not a finding.

The cheapest way to make the data usable is to **change the question asked of the
curator**, not to collect more of it. The per-track crop-strip mode already exists in
the viewer; asking a curator to mark the *span* of a track that is not a robot, rather
than one frame, would convert each click into a defensible run of negatives instead of
either one frame or a whole drifting track.

### Improvement path, in order

**1. ~~Curate a second match. ~~ ~~Then f1m2 to decompose the failure.~~ Both done, both
answered.** Cross-match transfer works (94–99%); colour was the problem and grayscale
fixed it; the descriptor swap is adopted and validated. What follows is superseded in
its first item only — the rest still stands.

**Next, in order:**

1. ~~**Put the classifier in the loop.**~~ Done — `rtrack.reid`, 13.3% → 84.2%.
2. **Re-tune now that the solver is deterministic.** `MAX_VOTES`, `--alli-weight` and
   `--park` were all tuned against a nondeterministic objective, so their settings are
   unverified. The `--max-votes` sweep must be redone; its earlier result was noise.
3. **Fine-tune the detector on the 3,875 curated hard negatives.** Largest remaining
   loss in the chain, and the shortcuts are already ruled out — see above.
4. **Surface per-track confidence** and keep low-confidence tracks in front of the
   curator. `reid` writes `share` and `margin` per track; nothing consumes them yet.
5. **Then** a fine-tuned embedding, if still wanted. The bar has moved a long way up:
   an untuned ResNet18 scored 47% detection-weighted on the flip test where 48-dim
   grayscale scored 84%.
6. **Single-pass decode**, independent of all the above.
Every number above is from one match, which holds lighting, camera and alliance
constant. The single untested premise — that a classifier trained on match A identifies
the same robots in match B — is the whole idea, and the side-split results are a mild
prior against it. Cost is ~90 s of curator time plus ~2 min of compute. Outcome decides
everything downstream: if cross-match transfer fails, the per-event classifier dies here
and curation stays manual, which is worth knowing before building anything.

**2. De-aliance in the classifier (`proj + cent`).** Measured, free, no curator cost,
and it is what makes the descriptor survive a team changing alliance mid-event. Fit
`proj` once per event, recompute `cent` per match from that match's own detections so it
also absorbs per-match exposure drift.

**3. Feed track-level appearance votes where the OCR votes used to go.** The vote
histogram interface into CP-SAT already exists and is now unused; the appearance
classifier produces exactly that shape at 88–91% detection-weighted, against OCR's 57%.
This is the step that actually buys autonomy: each curated match seeds the centroids for
the next, so the curator confirms rather than labels.

**4. Only then, a fine-tuned embedding.** The evidence that this pays is that an untuned
ImageNet ResNet18 already beats the hand-tuned histogram on the track-level metric. A
small metric-learning head over crops accumulated across an event is the natural form.
Gated on (1), because it is the most expensive item here and worthless if transfer fails.

**5. Single-pass decode**, independent of all the above. Five decodes at ~60 s each is
now the dominant term in curator latency.

**The risk to design against:** a classifier that is confidently wrong early in an event
poisons every match after it, because its output becomes the prior the curator confirms.
Whatever ships must surface per-track confidence and keep low-confidence tracks in front
of the curator rather than silently accepting them. The classifier proposes; CP-SAT's
constraints and the curator stay the arbiter.

---

## The correction that reframed everything

The original plan rested on this diagnosis, measured over 223 crops:

| failure mode | share |
|---|---|
| `no_text` — OCR returns nothing at all | **74.0%** |
| `unmatchable` | 13.9% |
| `HIT` | 9.9% |
| `ambiguous` | 2.2% |

and concluded *"three quarters of the failure is localisation"*. That conclusion was
wrong, because it never checked whether there was anything to localise.

**Hand-scoring 36 random crops: only ~12 (33%) contain a human-legible team number.**
Eight more are marginal. The remaining 16 show a robot facing away, buried in a fuel
scrum, clipped by the frame, or simply too far from camera. So roughly 44% of
`no_text` is not a failure at all — it is the correct answer.

That reset the target. Not "read the 74%", but "spend the OCR budget on the 33% that
can be read". Baseline was capturing 10 of an available ~33 points, not 10 of 100.

---

## Phase 1 — localisation. Tried, measured worse, removed.

Three variants were built and benchmarked against the existing path on identical
crops. None beat it.

| approach | `no_text` | HIT |
|---|---|---|
| **baseline: whole-box crop → CRAFT** | 73.6% | **10.0%** |
| bumper segmented by alliance hue → rectify → read | 93.2% | 5.5% |
| our own glyph localisation (global Otsu) | 74.5% | 9.5% |
| our own glyph localisation (adaptive threshold, both polarities) | 58.6% | 6.8% |

Why each failed, so none of them gets retried:

- **Hue segmentation.** These robots have red-painted chassis and stand in front of
  the red alliance wall. "Largest saturated red component in the box" is reliably the
  chassis or the wall. The colour cue is not discriminative on this footage.
- **Global Otsu on value-minus-saturation** merges the white digits with the bright
  carpet and field wall into a single blob, so digits never become separate
  components. This is a real bug and an adaptive threshold fixes it — glyph rows found
  went 24% → 53% — but fixing it did not fix the outcome.
- **Our own glyph localisation.** The prior "2–4 light components of similar height on
  a shared baseline" is too weak. Vent grilles, rivet rows, ball highlights and
  specular reflections all satisfy it. It found rows in 54% of crops, but they were
  mostly not numbers: 65 of 118 produced a single character, and dumped crops showed
  grilles and carpet. Distinguishing a number plate from a vent grille is a learned
  discrimination, not a geometric one.

The module was deleted rather than left in the tree. **CRAFT is good at finding text;
it was being handed crops with no text in them.**

---

## Phase 2 — crop selection. Implemented. This is where the win was.

Both signals are free: box width is already in the tracks file, and a Laplacian
variance is microseconds against a ~34 ms OCR call.

Yield per OCR call, measured over 500 crops:

| selector | crops kept | HIT rate | recall of all HITs |
|---|---|---|---|
| everything (previous behaviour) | 500 | 8.8% | 100% |
| width ≥ 120 px | 214 | 17.3% | 84% |
| **width ≥ 120 px AND sharpness ≥ median** | **87** | **32.2%** | 64% |
| width ≥ 100 AND Phase-1 glyph row found | 165 | 14.5% | 55% |

Both signals are strongly monotonic — by width 2.9 / 2.3 / 11.3 / 28.8%, by sharpness
quartile 3.2 / 4.0 / 13.6 / 14.4% — which is why their product beats either alone.
Note the last row: **the Phase 1 detector is worse than box width by itself**, even as
a prefilter. It earned no place in the pipeline in any role.

### Two things the naive version got wrong

1. **A global width floor starves far-field robots.** It reached 5 distinct tracks
   against 8 for the unfiltered budget. A robot that spends the match at the far end
   never produces a 120 px box, and dropping it does not make it unreadable — it makes
   it *unlabelled*. The floor is therefore applied per track and only when that track
   has enough large candidates to afford it.
2. **The sharpness threshold cannot be an absolute number.** Focus, exposure and
   encoding differ per broadcast. It is a running percentile over crops seen so far,
   and it is bypassed for any track that has not yet had its first dozen attempts.

### Result

Per-track vote shares, before and after:

| | before | after |
|---|---|---|
| best track | 33 votes @ 51% | 30 votes @ **100%** |
| tracks ≥70% share on ≥28 votes | 0 | **6** |
| typical share | 43–60% | 70–100% |
| runtime | 2m 50s | **3m 10s** |

Six tracks now clear the publishable bar: 5687 @ 100%/30, 9644 @ 100%/30,
1768 @ 90%/30, and three 6329 tracks at 77/73/70% on 30 votes each.

`DECIDE_MIN_VOTES` was also raised 10 → 30. Reads became accurate enough to satisfy
the old stopping rule almost immediately, which handed the freed budget straight back
instead of spending it on evidence.

---

## Where the bottleneck is now: grouping, not reading

The per-track gain does **not** reach the final output. Pooled into 6 robots the
shares are 16–63%, against 24–44% before — better at the top, no better overall.

The reason is visible in one group: tracks [4, 8, 13, 15, 21, 22] pooled
`{6329: 58, 9644: 42}`. Track 22 alone was 30/30 votes for 9644; tracks 8 and 15 were
6329 at 77% and 66%. Those are different robots merged into one group — a **colouring
error, not an OCR error**.

Two fixes were tried and both were reverted:

- **Weighting the vote-disagreement penalty by evidence** (30 votes @ 100% should
  outweigh 2 @ 50%). Changed the output by *nothing*.
- **Ordering the greedy colouring by vote confidence** so confident tracks anchor the
  groups. Measured worse: parked tracks 4 → 7, coverage 83% → 80%. It breaks the
  interval structure that makes six colours sufficient.

The diagnostic explains both. **Ordered by start time, 17 of 44 tracks had exactly one
legal group when placed and 10 had none; only 5 ever saw three or more options.** The
colouring is decided almost entirely by co-detection structure, so no cost function
can change it. Over-detection is not the culprit — only 0.4% of frames carry more than
6 tracks.

**This needs a different formulation, not a better weight.** Built as `rtrack.solve`
— see below.

---

## The global assignment (`rtrack.solve`, CP-SAT). Built and measured.

One solve assigns tracks directly to teams, replacing `colour()` + `name_groups()`
both. The two-stage split was the problem: grouping happened without knowing teams,
naming happened after groups were frozen, so the votes could never inform grouping.

Run with `--solver cpsat`. 222 booleans, ~900 pair terms, proven optimal in under 4 s.

| | coverage | robots ≥70% | worst share | mean share | alliance contradictions |
|---|---|---|---|---|---|
| greedy (baseline) | 83% | 2 | 17% | 48% | 1 |
| **cpsat, park=300** | 80% | 2 | 18% | **56%** | **0** |

Best robots: **9644 at 92%** (40 votes), **6329 at 77%** (84), **5687 at 69%** (49).

It is a real but modest win: +8 points of mean share, and it eliminates the alliance
contradiction that greedy could not avoid. It also reports *which* assignments are
uncertain, by re-solving with the optimum forbidden and listing the tracks that
change — 12 here, and those are the only decisions worth a human's time.

### A bug this exposed, affecting both solvers

`split_chimeras` gives later segments new ids but leaves segment 0 with the original
id, and `identity.json` is keyed by original id — so **a cut track kept its entire
tally on segment 0**, including votes cast during the windows that were split off. It
surfaced as a 15-detection segment reporting 24 votes, which is impossible. The split
was separating detections but not evidence, exactly defeating its purpose.

Fixed: `identify` now records each vote's timestamp (`voteList`), and `robots.retally`
re-attributes votes to the segment that was actually live when each was cast.

### Two things that did NOT work, measured

- **Raising the parking weight to buy coverage.** park=700 reaches 86% coverage but
  mean share collapses to 36% and the alliance contradiction returns. Coverage past
  ~80% is bought by forcing evidence-free tracks into some robot; that is not coverage
  worth having.
- **Pinning high-confidence tracks (the "human seed").** Pinning the two best tracks
  raised coverage to 86% and 1768 to 46%, but dropped 5687 from 69% to 25% and
  reintroduced an alliance contradiction. Seeding the tracks the solver *already gets
  right* adds nothing; see the revised guidance below.

---

## The binding constraint is now evidence supply, not assignment

| team | alliance | votes | share of all votes |
|---|---|---|---|
| 6329 | red | 115 | **36%** |
| 5687 | blue | 52 | 16% |
| 9644 | blue | 51 | 16% |
| 1768 | blue | 43 | 13% |
| 3467 | red | 29 | **9%** |
| 6201 | red | 29 | **9%** |

319 confident votes across the whole match, and 6329 holds nearly four times as many
as 6201 or 3467 — which are precisely the two robots every configuration labels worst
(18% and 40%). **No assignment algorithm can divide evidence that does not exist.**

One cause was found and fixed: a 2-character read like `"63"` is edit-distance 2 from
6329 and ≥4 from every other team here, so it passed `cost ≤ 2, margin ≥ 1` as a
*confident* vote. Since most failed reads return one or two characters, this
manufactured votes for whichever team a fragment happened to prefix. `MIN_READ_LEN=3`
now blocks it — worth 57 spurious votes and +11 points of worst-case share — but 6329
still holds 36%, so something else is also at work. Finding it is the highest-value
next step, ahead of any further solver work.

### Revised guidance on manual curation

The earlier expectation — that six clicks during auto would propagate through the
constraints and settle the whole match — is **not supported by measurement**. Pinning
the highest-confidence track per team reproduced the solver's own answer exactly and
only halved the uncertain-track count. Seeds confirm what the solver already believes.

What a human should be asked for instead:

1. **Identity for the evidence-starved robots** (6201, 3467 here) — tracks the OCR
   never read, not tracks it read well.
2. **The unstable set** — the tracks that change between the best and second-best
   solution, which `--solver cpsat` now prints. Everything else is settled.

---

## What would still help recognition, in order

1. **More candidate crops per track.** The selector is now good enough that the limit
   is how many large, sharp crops exist. Sampling every frame rather than every fifth
   would roughly triple the pool at roughly triple the decode cost.
2. **PaddleOCR instead of easyocr** (~1 hour). Attacks the ~14% `unmatchable`. Worth
   benchmarking before building anything.
3. **A trained digit model** (~1 day). Closed-set 4-digit recognition on a
   near-standard font is a much easier learning problem than general OCR and far
   faster at inference. Only worth it after the grouping problem is solved, since
   grouping — not reading — is what currently caps the output.

## What NOT to do

- **Resurrect OCR without a new reason.** Both items below (PaddleOCR, a trained digit
  model) were written before the removal. They address *reading accuracy*, which was
  never the binding constraint — the appearance classifier already beats OCR by 31
  points detection-weighted at a fraction of the runtime. Revisit only if the
  cross-match classifier test fails.

- **Deblurring.** Faster robots read *better* (15.6% at ≥2 m/s vs 7.8% below 0.5 m/s).
  Slow robots are slow because they are in scrums, occluded and badly angled.
- **Hand-built localisation.** Three variants, all measured worse than CRAFT.
- **More preprocessing variants on the crop.** The crop framing was never the problem;
  crop *selection* was.
- **Expecting `no_text` below ~45%.** Roughly that share of crops genuinely contain no
  legible number, and reporting it as a failure mode double-counts occlusion.
