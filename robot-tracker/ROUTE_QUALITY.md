# Route quality: what we know, what we're asking, how to investigate

Working notes for the effort to make exported robot routes trustworthy. Written to be
picked up cold by someone with no memory of the session that produced it.

The deliverable this serves is `public/tracks/<matchKey>.json` — per-robot routes in field
metres, drawn by `renderFieldRoutes` in `main.js`. A route is wrong in ways a curator can
see: it jumps across the field, stops for ten seconds, or belongs to the wrong robot.

---

## 1. Current state

Branch `rtrack-identity-accuracy`, which is also `main` (pushed).

**Committed and validated:**

| commit | change |
|---|---|
| `607b924` | project field positions BEFORE the solve, not after |
| `5746496` | `DUP_IOU` 0.05 → 0.35 |
| `9be5b13` | a forbidden pair cannot share a pinned team either |
| `d62e07f` | measured motion model; demote same-team pins that cannot both be right |
| `f712dc6` | within-track kinematic cuts; occluder shading on route plots |

**Uncommitted, in the working tree, validated on qm10 only:**

- `robots.split_impossible_steps` now takes ONE position per (track, frame), keeping the
  lowest box. Without this it read a track carrying two boxes per frame as alternating
  between them and cut 1291 times on qm10 instead of ~1.

**Found, diagnosed, NOT fixed — the current head of the investigation:**

- `occluder_rebind` in `robots.py` fuses tracks that are different robots. See §3.

The last few 2026mawor runs are not committed, so published routes may lag the tree.
Re-derive rather than trusting what is live.

---

## 2. What we learned, with the evidence

Five bugs, all real, all found on 2026mawor and all invisible on 2026necmp1.

### 2.1 Field positions arrived after the solve that needed them

`rtrack.robots` consumes `out/stage2/<stem>_positions.json` for the kinematic constraint,
the geometric duplicate merge and field-space clash checks. `STEPS` in `pipeline.py` put
`robots` at index 5 and `project` at index 10, so that file could only ever be a PREVIOUS
run's — and on a first solve it did not exist. Every match the watcher handles is a first
solve, so the constraint was inert in the only unattended path. Proof from qm13:

```
22:28:15  out/stage3/2026mawor_qm13_labeled.jsonl   the solve output
22:28:40  out/stage2/2026mawor_qm13_positions.json  25s LATER
```

and zero `kinematically impossible` lines across all 13 matches the watcher had processed.
Fixed with a `prepos` step projecting the stitched tracks first.

**It hid because nothing reports a constraint that is not applied.** Absence of a warning
looked like absence of a problem.

### 2.2 The duplicate merge was fusing real robots

`geometric_duplicates` merged pairs at IoU 0.09–0.29, two of them co-detected for 137 and
140 frames — nine seconds of a pair 0.37 box widths apart. That is two robots driving
alongside each other. `DUP_IOU` was 0.05 because other criteria were meant to exclude;
that held on necmp1, whose genuine low/high duplicates sit at IoU 0.27–0.53.

### 2.3 A forbidden pair could still share a pinned team

The exclusion loop skipped the pinned team:

```python
for k in range(len(teams)):
    if teams[k] == pl or teams[k] == ph:
        continue      # that team is pinned here; leave it alone
```

With track A pinned to team T, the exclusion was skipped for T — the one team B must not
have. Both came out T. The guard was for a pin fighting an exclusion, but they only
conflict when BOTH are pinned to the same team, which is handled separately.

### 2.4 The motion model was linear, and vacuous past 2.5 s

`ROBOT_MAX_SPEED_MS * dt + 1.0` assumed top speed held for the whole interval. Measured
p99.9 displacement over 660 tracks:

| dt | observed p99.9 | old bound ×1.5 |
|---|---|---|
| 0.15 s | 0.96 m | 2.74 |
| 1.25 s | 4.53 m | 11.81 |
| 2.50 s | 7.32 m | **22.12** |
| 5.00 s | 10.79 m | **42.75** |

The field diagonal is 18.4 m, so past 2.5 s the bound exceeded the largest distance on the
field — nothing could be forbidden. Replaced by `EMPIRICAL_P999_M` in `solve.py`.

Two events agree closely in the tail, so the envelope describes ROBOTS and one curve
serves both cameras. Their medians differ (mawor's projection is noisier: 0.58 m/s
apparent motion for a stationary robot against necmp1's 0.22).

### 2.5 The within-track cut mis-saw tracks with two boxes per frame

`geometric_duplicates` fuses a low/high pair by RE-TAGGING, not deleting, so a track can
hold two detections in one frame. On qm10 one track held 3321 detections over ~2200
frames. Sorted by time it alternated between boxes, so every step read as impossible.

---

## 3. The open head: `occluder_rebind` fuses different robots

On 2026mawor_qm10 the rebind merged three tracks into one:

```
#28 -> #19 at blue hub: gap 3.7s
#31 -> #19 at blue hub: gap 9.3s
#40 -> #19 at blue hub: gap 39.7s     <- 40 seconds
#46 -> #5  at blue hub: gap 58.5s     <- 58 seconds
```

tid 31 (1648 dets) + tid 40 (1261) fused into the 3321-detection blob above. All at the
blue hub, which is why BLUE was the broken alliance in that match.

**The conflicting rules, in plain terms.** A robot behind a structure has not left, so
reconnect it (keeps routes whole). But only for about as long as a transit takes — capped
at twenty seconds. UNLESS it was parked there, in which case allow up to three minutes.

The third rule wins whenever it applies, and its evidence is nearly worthless at a hub: it
asks only whether the track vanished and reappeared near the same spot, which describes
almost any robot loitering near a hub. It then hands the choice to appearance, which is at
its weakest there — partly hidden robots, odd angles, same-alliance bumper colour. On qm10
the winning cosine was 0.118 against a runner-up of 0.169.

**The asymmetry that makes it a bad trade.** The rebind buys coverage; it risks identity.
A missed reconnection costs one visible gap. A wrong one fuses two robots, corrupting both
identities and everything downstream that learns from them.

**Measured on qm10, rebind disabled (`--occluders __none__`):**

| | rebind on | rebind off | no within-track cut |
|---|---|---|---|
| curator alignment | 97% | **98%** | 91% |
| mean custody | 0.824 | **0.866** | 0.778 |
| within-track cuts | 55 | **1** | — |
| parked tracks | 36 | **13** | 8 |
| 1768 custody | 0.590 | **0.736** | 0.466 |
| 1027 custody | 0.818 | **0.917** | 0.758 |

Turning it off improved coverage AND accuracy simultaneously. The 55 cuts were the
within-track check correctly flagging the rebind's damage — with the rebind off it finds
exactly 1, matching the rate measured on clean stitched tracks.

**Do not simply delete the parked rule.** The question is what evidence should be required
before allowing a long hide. Candidates: a much tighter position tolerance; requiring no
other track to be near that occluder during the gap; requiring a clear appearance margin
rather than merely a best match; or capping total hidden time far below 180 s. Measure
across BOTH events before changing a default — see §5.

---

## 4. Method lessons, learned expensively

**Render the frames. Look at them.** Three wrong theories today were built from percentile
tables: "box-bottom instability at the far edge", "frozen boxes", "micro-fragments". All
three were asserted confidently and all three were wrong. Extracting two frames took a
minute and showed the truth instantly — the detector had fired on a PERSON in an FTA vest
and scored them 0.613 against the real robot's 0.537.

```bash
# frame number -> timestamp; MAWOR is 60 fps, necmp1 30 fps
t=$(python -c "print(f'{FRAME/60:.4f}')")
ffmpeg -loglevel error -ss $t -i data/raw/<stem>.mp4 -frames:v 1 -y out.png
```

Use statistics to FIND the case; use the image to understand it.

**Measure the data the code actually sees.** Estimates were wrong three times because they
were taken from a convenient file on disk rather than the in-memory rows at that point in
the pipeline. Notably: at the stitch stage `rows` carry FRAGMENT ids and `mapping` turns
them into tracks — grouping by the raw id inspects something else entirely.

**When instrumenting beats analysing, instrument.** The pin-exemption bug was found in one
run by printing what the loop decided. Two prior rounds of aggregate analysis had missed it.

**Do not explain away a negative result.** A "principled" change that measures worse was
reported as "no measurable benefit, probably confounded". It was not confounded — it was
broken (§2.5). Take the number at face value and look for the bug.

---

## 5. How to investigate a reported instance

### 5.1 Report format

```
2026mawor_qm10  1768   115-125s  teleport   crossed the field, only 1 sample
2026mawor_qm10  1027   132-136s  gap        was parked by the blue hub
```

Full match key (both events have a qm10). Team. Time window in MATCH seconds — the route
scrubber's clock, which is what the export uses. Note if a time is video-clock instead;
MAWOR is offset ~35 s.

Symptom vocabulary, because each points somewhere different:

| word | means |
|---|---|
| **teleport** | route jumps across the field and back |
| **gap** | route stops and resumes |
| **swap** | two robots trade identities and continue |
| **ghost** | a route where that robot plainly is not |
| **double** | one robot drawn as two teams at once |

Where the robot actually was ("at the blue hub the whole time") is the single most useful
optional detail — it is what confirmed the parked track was the real 1768. Proximity to a
hub, trench or field edge is worth noting: occlusion failures cluster there.

Track ids are useless in a report; they change every solve.

### 5.2 Triage

| symptom | first hypothesis | check |
|---|---|---|
| gap + robot was near a hub | occluder rebind fused its track into another (§3) | `grep "rejoined across" <solve log>`; look for gaps > 20 s |
| gap, detections exist but unnamed | solver parked a real track | count detections per frame in the window; if 6 exist and 5 are named, the 6th is the answer |
| teleport across tracks | kinematic exclusion missed the pair | `grep "pair(s) forbidden"`; if zero, positions were missing |
| teleport inside one track | bad detection (person, field element) taken into the track | RENDER THE TWO FRAMES |
| swap | fused track upstream, usually the rebind | find the over-long track: dets ≫ frames |
| double | duplicate merge did not fire, or fired wrongly | `grep "duplicate track"`; check IoU and box-width separation |

### 5.3 Commands

```bash
cd robot-tracker

# solve one match, reproducibly
uv run -m rtrack.robots 2026mawor_qm10 \
  --tracks out/stage1/2026mawor_qm10_tracks_stitched.jsonl \
  --match 2026mawor_qm10 --deconflict 3 --no-hint --seed 0 \
  --calib-from 2026mawor --positions out/stage2/2026mawor_qm10_prepos.json \
  --identity out/stage3/2026mawor_qm10_reid_cnn.json \
  --corrections corrections/2026mawor_qm10_corrections.json

# then re-project and re-export before reading routes
uv run -m rtrack.project 2026mawor_qm10 --tracks out/stage3/2026mawor_qm10_labeled.jsonl --calib-from 2026mawor
uv run -m rtrack.export  2026mawor_qm10 --match 2026mawor_qm10 --hz 5 --publish --calib-from 2026mawor

# useful switches for A/B
--occluders __none__    # disable the occluder rebind and holding cells
--step-cut 0            # disable the within-track metres cut
--kin-hard 0            # disable the hard kinematic exclusion
```

Calibration stems are `2026mawor` and `2026necmp1`. They are NOT video ids — `export` and
`project` used to validate them as such, which broke necmp1 entirely after the rename.

### 5.4 Two metrics, and what each is worth

**Curator alignment** — printed by `rtrack.robots` when corrections exist. The only number
that cannot be gamed by labelling less. Trust it over custody, churn, or teleport counts.

**Mean custody** — share of the match each robot is tracked. Trades against alignment in
principle, but today every genuine fix improved BOTH. A change that helps one and hurts the
other deserves suspicion, not a trade-off narrative.

Teleport counts are a diagnostic, not a target. Measure them by DISTANCE, not speed:
dividing a noisy 2 m displacement by a 0.2 s interval manufactures 10 m/s. Of 43 such
"teleports" once, 22 were sub-3-metre hops inside the projection's own noise floor.

---

## 6. Open questions, in priority order

1. **What evidence should permit a long hide behind an occluder?** (§3) Fix, then measure
   alignment and custody on all 13 curated 2026mawor matches AND a necmp1 sample. necmp1
   has 6 occluder regions drawn and has not been re-measured since any of today's changes.
2. **Should the rebind be off by default?** It currently costs accuracy and coverage on the
   one event where it has been measured. necmp1 may differ — it has a different camera and
   more regions.
3. **Does the within-track cut still over-cut?** With the rebind off it makes 1 cut on
   qm10, matching expectation. Confirm across the event before trusting the threshold.
4. **The blue-alliance swap in qm10** — the real 1768 was parked while another track wore
   its label. Believed downstream of the rebind; confirm it disappears when §3 is fixed.
5. **Intra-track jumps not caused by bad detections.** ~38 remain event-wide. Unknown
   whether they are projection noise or more fused tracks.
6. **necmp1 has not been re-validated** since `DUP_IOU`, the motion model, the pin fix, or
   either cut. Everything today was tuned on MAWOR. Four constants have already broken when
   moved between these two cameras; assume more will.
