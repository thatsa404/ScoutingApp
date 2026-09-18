# Curation loop

Status: **local loop built, used on a real match, and measured.** Relay not built.

## The headline number

Scored against 176 human-labelled anchors on `2026necmp_f1m3`:

| | correct | wrong | unlabelled |
|---|---|---|---|
| **automated, no human** | **52%** | 40% | 8% |
| **with corrections** | **94%** | 2% | 3% |

Automated identity is a coin flip. Better than the 17% a six-way guess would give, and
nowhere near usable — which settles the question: **automated labelling does not close
on this footage.** The architecture is human-in-the-loop, and the automated path's job
is to reduce the number of questions, not eliminate them.

Read the 94% carefully: corrections are treated as truth by construction, so that
figure measures *fidelity* — whether the plumbing faithfully transmits human judgment —
not independent accuracy. The 6% that is not honoured is the interesting part, and it
is all clash demotions and parked tracks. The 52% is not circular: it is the automated
pipeline scored against labels it never saw.

In detections rather than anchors: automated assigns 86% of detections at 52% accuracy
(~45% of all detections correctly identified); curated assigns 74% at 94% (~70%).
Curation trades some coverage for nearly double the correctly-labelled data.

### Where automation fails

Of ~70 wrong answers, about 55 are **within-alliance** confusions — 6201↔3467 (22),
5687↔1768, 1768→9644, 9644→5687 — against ~15 cross-alliance. The alliance machinery
(hue classification, `split_on_alliance`, the soft alliance term) is doing its job.
What is missing is any signal that separates the three robots *inside* one alliance,
which is exactly what the bumper number was supposed to provide and mostly cannot,
because only ~33% of crops contain a legible one.

## The loop

```bash
# 1. produce a bundle (one self-contained file, ~2.6 MB, 90% of detections)
uv run -m rtrack.curate GSxbsE42o5o --match 2026necmp_f1m3

# 2. open viewer/curate.html, drag the bundle in, adjudicate, save
#    -> corrections/<matchkey>_corrections.json

# 3. feed it back
uv run -m rtrack.robots GSxbsE42o5o --tracks out/stage1/MATCH3_st.jsonl \
    --match 2026necmp_f1m3 --solver cpsat \
    --corrections corrections/2026necmp_f1m3_corrections.json
```

No server, no CORS, no deployment, no Python on the curator's machine.

## The decision that makes it durable: detection anchors

Corrections are anchored to **a detection**, never to a track id:

```json
{"f": 1234, "xy": [812, 604], "team": "9644"}
```

Track ids are manufactured by `split_on_alliance`, `split_on_appearance` and
`split_chimeras`, all of which renumber, and we retune those thresholds constantly. A
correction saying "track 22 is 9644" is void the moment any threshold moves. A
correction saying "the robot at (812, 604) in frame 1234 is 9644" survives
resplitting, retracking and solver changes; it only breaks if the *detector* re-runs,
and then it degrades to a near-miss rather than a silent mislabel.

Resolution is nearest-box-centre within 90 px. Anything further is reported
UNRESOLVED, never silently dropped — a correction that quietly does nothing is worse
than one that errors. Verified: a deliberately bogus anchor was rejected at 676 px.

**Two labels on one track is a cut instruction.** If a curator marks frame 100 as 6329
and frame 500 as 9644 and both land on the same track, they have said something
stronger than either label: that track holds two robots. We cut between them rather
than making the labels argue. That beats every heuristic we have, because it comes
from someone who looked.

## What the curator sees

One track at a time: a grid of 12 crops spanning the track's life (the same crops
`rtrack.chicklets` uses, so the curator judges what we judge), plus span and detection
count.

| key | meaning |
|---|---|
| `1`–`6` | that team — whole track, or just the selected crops |
| `U` | **unknown** — it's a robot, can't say which team |
| `N` | **not a robot** — detector false positive |
| `M` | **mixed** — several robots, can't say where the join is |
| `S` | **skip** — not answered; comes back next pass |
| `A` / `Esc` | select all crops / clear selection |
| `←` `→` `L` `H` | navigate, list view, reveal the machine's guess |

`unknown` and `skip` are deliberately different: a skipped track is re-asked, an
`unknown` one is not, and the count of `unknown` answers is the honest ceiling on what
curation can achieve on this footage. `not a robot` is recorded rather than merely
dropped, so detector false positives can be counted.

### Per-crop labelling is the important control

Click a crop (ctrl/⌘-click to add more, shift-click for a run) and a team key labels
only those crops. Two differing labels on one track is read by `rtrack.corrections` as
**cut this track between them** — which is what a mid-track identity switch actually
is. Verified on the real case: track 16 reads 9644 at t86.5s and 6329 at t116.1s;
labelling both crops cut the track and pinned `16→9644`, `65→6329`.

That is strictly better than `mixed`, which only parks the track and recovers neither
robot. Prefer it whenever the switch point is visible.

### Two UI bugs that corrupted the first pass

Worth recording because both produced *plausible wrong labels* rather than obvious
breakage:

- **The crop strip scrolled horizontally**, so later crops were off-screen. Curators
  judged whole tracks from their first half. It now wraps; every crop is visible.
- **Crop selection ranked globally by size × sharpness**, which concentrated every
  crop wherever the robot passed nearest the camera. A 1077-detection track spanning
  39–124 s returned 11 of 12 crops from 44–65 s. Selection is now best-per-time-slot,
  which cut the largest inter-crop gap from ~35 s to ~7 s.

Together these are the likely cause of most of the six "label clashes" in the first
pass — they were not curator error so much as the tool showing half the evidence.

**The machine's guess is hidden by default.** Showing it makes confirmation fast and
makes anchoring certain, and anchoring is exactly how a wrong label gets laundered into
a verified one. It is one keypress away when genuinely needed.

Answers persist to `localStorage` per video, so a half-finished pass survives a
refresh.

### Ordering, and why partial passes are useful

Tracks are presented largest-first, because that is what a correction is worth.
Measured on the current output (63 tracks, 12,165 detections):

| answered | detections covered |
|---|---|
| 10 | 44% |
| 20 | 66% |
| **30** | **80%** |
| 40 | 90% |

So ~30 questions per match, 2–3 s each: **about 90 seconds**. A curator who stops at
10 has still fixed the largest 44%.

## What corrections do downstream

- **team label → CP-SAT hard pin** (`solve(pinned=...)`, already implemented)
- **conflicting labels → forced cut** via `_apply_cuts`, ahead of the solver
- **`mixed` → track dropped from grouping.** It keeps its detections in the output; it
  just stops claiming to be one robot.
- **everything → ground truth.** Every metric in this project so far has been a proxy:
  vote share, alliance consistency, whether a chicklet row looks mixed. None is truth.
  These labels are, so they are simultaneously solver input and the eval set we have
  never had — and later, training data for the digit model in OCR_PLAN Phase 3.

Corrections are also **~140× faster to solve**: the unpinned CP-SAT model took 252 s to
prove optimal on 63 tracks; with four pins it took 1.76 s. Pins collapse the search
space, so curation makes the pipeline quicker as well as more correct.

Coverage is reported against the detection count taken *before* any curator rejection,
so rejecting a track can never look like an improvement.

## Not built yet

**The relay.** Only needed when curation leaves this machine. The pattern is already
deployed in `nexus-relay/worker.js` (Cloudflare Worker + KV, CORS, token auth): add
`PUT /curate/:matchKey`, `GET /curate/:matchKey`, `POST /corrections/:matchKey`. About
40 lines and the same `wrangler deploy`. A 2.6 MB bundle sits well under KV's 25 MB
limit. **One deviation from the Nexus pattern:** that worker sets a 2-hour TTL, right
for match status and wrong here. Corrections are ground truth and belong in git at
`robot-tracker/corrections/<matchkey>.json`, with KV as transport only.

**Greedy solver support.** `--corrections` requires `--solver cpsat`; the greedy
colouring cannot honour pins and says so rather than ignoring them silently.

## The open question worth measuring first

When pins were placed on tracks the solver *already agreed with*, the assignment did
not change at all and only the uncertainty count halved. Corrective pins on tracks it
got wrong should do more, but that is unmeasured. It decides whether 30 answers resolve
30 tracks or considerably more — and therefore whether this scales to 80 matches at an
event. Measure it on the first real curated match.
