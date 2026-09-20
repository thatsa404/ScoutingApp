# robot-tracker

Proof-of-concept: track FRC robot positions over time from match video.

Self-contained Python subproject. Nothing here is imported by the Vite app — the
root `npm run build` and the Pages deploy are unaffected. It reads two things from
the parent repo and copies neither: `public/field/2026-field.png` and the
`VITE_TBA_KEY` in `.env`.

## Status

| Stage | What it proves | State |
|---|---|---|
| Env | CUDA torch actually on the GPU | ✅ torch 2.14.0+cu126, RTX 3060, 6.8 TFLOP/s |
| **0** | Is this footage usable at all? (risks R1, R5) | ✅ **BEST CASE** — see findings |
| **1a** | Open-vocab zero-label baseline (answers R2) | ✅ dead end, but R2 closed |
| **1** | ★ **PROOF GATE** — detection + stable tracking overlay | ✅ at the single-camera ceiling |
| **2** | Image → field coordinates | ✅ manual homography + lens model, **0.10 m mean / 0.26 m max** over 21 points. ⚠️ ONE camera only — each new venue needs re-clicking; `autocal` is degenerate (3 tags, conditioning 0.0) |
| **3** | Identity | ✅ **solved** — curation ~90 s/match, then 84.2% automatic on the *next* match via `rtrack.reid`. See OCR_PLAN.md |
| **3b** | JSON export (`rtrack-tracks v1`) | ✅ `rtrack.export` → `out/stage3/<matchKey>.json`, `--publish` to `public/tracks/`. 5 Hz, ~165 KB / 26 KB gzipped per match |
| **3c** | Replay viewer + app integration | ✅ standalone `viewer/replay.html` (drag a file, scrub) **and** in-app: Dexie v3 `matchTracks`, routes in the match modal, Routes tab on team detail |
| **3d** | Curation quality — is a frame worth a click? | ✅ legibility scored against the curator's own "can't tell" labels: **AUC 0.602 → 0.873**. See [Curation quality](#curation-quality--what-makes-a-frame-worth-a-click) |
| **4** | Capture → curate → publish, unattended | ✅ end-to-end on 2026mawor: 11 matches sliced from one YouTube archive, curated from a phone over a Cloudflare relay, published without touching the machine. 83–88% custody |

**Validated end to end on a real event.** 2026mawor (WPI) was processed from a single
multi-hour stream archive, curated on a phone in a different room from the machine, and
published to the app. That is the whole loop, and it is the thing the earlier stages
existed to make possible. What it is NOT yet: live. Everything here still runs against a
recorded stream, and the livestream source remains unwritten — though `source.py` is why
that is one new class rather than a rewrite.

## Portability: four cameras, measured

Everything below Stage 1 was tuned on one camera (the NE championship finals rig). Three
more videos say how much of that generalises. All four are 1080p, ~215 s, same season.

| | necmp f1m2 | necmp f1m3 | necmp1 sf11m1 | **mawor WPI f1m2** |
|---|---|---|---|---|
| source fps | 29.97 | 29.97 | 30.00 | **58.00** |
| det/frame (6 robots) | 3.78 | 4.58 | 4.72 | **7.72** |
| **curator says NOT a robot** | **2%** | **5%** | **23%** | **55%** |
| stitched tracks | 27 | 32 | 36 | **86** |
| median box width (px) | 117 | 116 | 115 | **94** |
| curator labels needed | 95 | 122 | 121 | **234** |
| mean custody | 76% | 77% | 73% | **76%** |
| robots found / conflicts | 6/6, 0 | 6/6, 0 | 6/6, 0 | **6/6, 0** |
| decode-only ceiling | — | 3.90× RT | — | **2.29× RT** |
| end-to-end throughput | 2.1× RT | 4.0× RT | 4.0× RT | **1.09× RT** |

**Three figures in this row were wrong when first published, and each for a different
reason. Worth reading before trusting any single number here.**

* **f1m2 read 61%.** That was one draw from the nondeterministic CP-SAT objective,
  before the determinism fix in `solve.py`. 76% is reproducible.
* **sf11m1 read 61%.** A measurement artifact: `match_window()` needs
  `positions.json`, which needs a calibration, and sf11m1 has none — so custody
  silently fell back to the WHOLE 214 s clip while every other match used a 150 s
  window. Over the same window it is **73%**, i.e. mid-pack rather than worst. Fixed
  with a pixel-space auto-start fallback (`auto_start_px`) that needs no homography,
  and the no-window case now prints "NOT COMPARABLE" instead of an innocent-looking
  percentage.
* **WPI read 59%.** Real, and fixed rather than re-measured: the alliance penalty was
  flat regardless of evidence. See the note below on team 190.

The pattern is the same each time — a number that looked like a property of the camera
was a property of the measurement. Custody is only comparable across matches when the
window is, and `custodyWindowSource` in `<stem>_robots.json` now records which path
produced it.

**The false-positive rate is a camera property, not noise: 2% → 5% → 23% → 55%.** The two
clean columns are the same physical camera. On the WPI camera the majority of detections
are field furniture — scoring towers, equipment, fuel — and it costs ~2× the curation
effort to say so. This is now the largest single loss in the chain and the clearest case
for spending curated `notrobot` data on detector hard negatives.

**What survived anyway:** grouping, alliance splitting and the CP-SAT assignment held on
all four. 6/6 robots, zero custody conflicts everywhere, and mean custody on the worst
camera (59%) is in line with two of the three championship runs. The identity chain is
more portable than the detector under it.

**Do not compare raw "coverage" across cameras.** It is named detections ÷ *all*
detections, so a camera with 55% junk detections has an inflated denominator and reports
47% while tracking robots about as well as one reporting 97%. Custody (per-robot share of
match time held) is the honest cross-camera metric.

### Frame rate is not 30 everywhere

The WPI stream is **58 fps**, and `--stride` was a hardcoded 2. That silently doubled the
processed rate and halved every frame-counted parameter downstream in real terms
(`track_buffer: 60`, `appear.STRIDE`, `split_on_appearance(win=12)` — all counted in
PROCESSED frames). Fixed: `rtrack.track` now takes **`--sample-hz` (default 15)** and
derives stride from the measured source fps, which makes all of those time-correct by
construction. `--stride` remains as an override for reproducing old runs.

Sampling coarser than ~10 Hz is not an option, and the reason is association rather than
resolution: a 5 m/s robot about 0.75 m wide travels ~1.25 m between samples at 4 Hz,
which is more than one box width, so IoU matching sees no overlap and the track breaks.
Decimate the OUTPUT as much as you like; the tracker cannot be decimated.

### Stage 4: the bottleneck is decode, not inference

Measured on the WPI file by halving the sample rate:

| sample rate | frames processed | wall | vs realtime |
|---|---|---|---|
| 15 Hz | 3109 | 195.9 s | 1.09× |
| 7.5 Hz | 1555 | 170.0 s | 1.26× |
| decode only, no model | 12436 | 93.7 s | **2.29×** |

Halving inference bought 13% of wall time, because every frame is decoded either way.
**Even with zero inference this file caps at 2.29× realtime**, against 3.90× for the
30 fps file. A livestream needs 1.0×; 1.09× is 9% headroom, which is none. The fix is a
cheaper decode path (NVDEC hardware decode, or letting ffmpeg decimate before the frames
reach Python), not a coarser sample rate. This is the same cost the ~5 separate decodes
in the current pipeline pay, so the single-pass refactor and the Stage 4 fix are the
same piece of work.

### A spatial "must be on the field" filter. Built, and it works.

Observation (JG): most WPI false positives are outside the arena, so the field boundary
could reject them. **Measured with a real homography on the 2026mawor camera, it removes
75% of curator-confirmed false positives for 2.0% of real robot detections.**

| class | detections | removed | removal rate |
|---|---|---|---|
| curator-confirmed robot | 3124 | 64 | **2.0%** |
| curator-confirmed NOT robot | 15009 | 11211 | **75%** |
| unlabelled | 5879 | 433 | 7% |

End to end on that match: coverage 47% → **91%**, stitched tracks **86 → 45**, and after
the split chain **110 tracks against f1m3's 103** — i.e. the curation burden on the worst
camera is now the same as on the clean one. Custody is unchanged at 59%, which is the
honest read: the filter does not track robots better, it removes junk that was never
robots. The coverage jump is mostly the denominator being cleaned up.

`rtrack.robots` applies it (`drop_offfield`, `--no-field-filter` to disable). It is a
no-op without a calibration and says so rather than failing quietly.

The rest of this section is the earlier homography-free exploration, kept because it
establishes what the filter can and cannot do.

Tested **without** a homography, since the two problem cameras have none. Confirmed-robot
box-bottoms define the drivable region by construction; the test is what share of
confirmed NON-robot detections fall outside their convex hull.

| match | false detections | outside the region | caught |
|---|---|---|---|
| necmp f1m3 | 273 | 0 | 0% |
| necmp f1m2 | 177 | 65 | 37% |
| sf11m1 | 3425 | 239 | **7%** |
| **WPI f1m2** | 11973 | 7860 | **66%** |

52 of WPI's 83 false tracks are mostly outside. **The filter helps most exactly where the
problem is worst**, which is the useful direction.

**It does not need a homography.** Seeding the region from just the 8 longest tracks — no
curation at all — still works, and the asymmetry is favourable:

| match | FP caught | confirmed robots wrongly dropped |
|---|---|---|
| necmp f1m3 | 1% | 3% |
| necmp f1m2 | 71% | 5% |
| sf11m1 | 6% | 2% |
| **WPI f1m2** | **50%** | **1%** |

Bootstrapping is weaker than the curated version (50% vs 66%) for a diagnosable reason:
**3 of WPI's 8 longest tracks are themselves false positives**, so they inflate the hull.
A homography-derived boundary would not have that contamination — which is the argument
for the original idea over the bootstrap.

**What it cannot fix:** sf11m1 gets 6–7%, because its false positives are fuel piles *on
the field*. Confirmed visually in the curator crops. Any spatial filter is blind to those,
so this is a partial remedy, not a replacement for detector work.

**Design note, and the reason this is cheap:** a filter needs only the field *boundary*,
not metric accuracy and not a lens model. That is roughly 4 clicks per camera against 21
for full metric calibration, and it is useful even where a proper homography is never
built. Where one does exist, reuse it.

**The pixel-space version was a dead end, and the reason is instructive.** An attempt to
verify it by dilating the hull 8% "as a safety margin" dropped only 3% of detections
instead of the expected ~25%: 8% in image pixels is enormous under perspective and
swallowed the whole gain. The shipped filter uses **0.6 m of slack in field metres**,
which is uniform where pixels are not — the single clearest practical argument for doing
this with a homography rather than an image-space hull.

### Camera motion: a one-time reframe, not a pan

The WPI broadcast opens on a full-screen title card, cuts to the field at t≈3.4 s, then
**reframes by +227 px vertically with a 1.2% zoom, settling at t≈6.5 s**. After that it
holds to within 0.6 px for the remaining 164 s — as static as the championship cameras,
so a single homography is valid *from the settle point*.

Two traps worth recording. First, the settle coincides with auto start (the 0:20
countdown begins ticking at ~6.5–7.5 s), which is exactly where the free ground-truth
check lives ("all 6 robots behind their starting line"). Second, **the naive way to
measure this gives the wrong answer**: optical flow across the whole frame reported the
camera static from t=3.5 s, because the static score overlay supplies ~99 perfect
matches that dominate the fit. The broadcast furniture must be masked before any
camera-motion estimate, and `cfg/botsort_frc.yaml` still ships `gmc_method: none`, which
is correct for the championship camera and wrong here.

The reframe is NOT what causes the fragmentation: new track ids appear at a steady ~10
per 10 s across the whole match (17 in the first bucket, 7–15 everywhere else). The false
positives are.

## Current best pipeline

One match, end to end, is one command. `rtrack.pipeline` sequences what already exists
rather than reimplementing any of it, and every step is skipped when its output is newer
than its input — which is what makes the curator round trip practical, since the second
pass must redo the solve and everything below it and nothing above it.

```powershell
# Slice the match out of an event archive, then run it
uv run -m rtrack.replay   2026mawor --match qm9
uv run -m rtrack.pipeline 2026mawor_qm9 --match 2026mawor_qm9 --calib-from 2026mawor
```

That is: track → stitch → appear → reid votes → robots (CP-SAT) → curate bundle →
[human] → robots again → project → export → gallery. See
[Stage 4](#stage-4--capture-curate-publish) for the loop around it.

The stages still run standalone, which is how anything gets debugged:

```powershell
uv run -m rtrack.track GSxbsE42o5o `
    --model runs/detect/runs/r2026_s/weights/best.pt `
    --start 91 --end 5398 --stride 2 `
    --imgsz 640 --square --conf 0.20 --lost-decay 0.7 --nms-iou 0.55
uv run -m rtrack.stitch  out/stage1/GSxbsE42o5o_tracks.jsonl
uv run -m rtrack.project GSxbsE42o5o --tracks out/stage1/GSxbsE42o5o_tracks_stitched.jsonl
uv run -m rtrack.routes  GSxbsE42o5o --auto --panels
```

Full match (2654 processed frames, 177 s): **30.2 fps**, 4.58 det/frame, **32 tracks**,
alliance decided on 85.3% of detections, 1.9% of frames with an impossible alliance
count. Projected to metres: 12,165 samples, **0% off-field**, 1.1% kinematic
violations.

## Setup

```powershell
winget install --id=astral-sh.uv -e
winget install --id=Gyan.FFmpeg -e
# reopen the shell so PATH picks both up
cd robot-tracker
uv sync            # ~2.5 GB, 10-20 min
```

`uv` provisions its own CPython 3.12 and leaves the system 3.14 alone. 3.12 is
required because **PyTorch ships no CUDA wheels for cp314** — on 3.14 you silently
get a CPU build.

### Verify CUDA — all three, the first one lies

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.6.x+cu126 12.6 True NVIDIA GeForce RTX 3060
# cuda=None -> CPU wheel; uninstall torch/torchvision and re-sync

uv run python -c "import torch,time; a=torch.randn(4096,4096,device='cuda'); torch.cuda.synchronize(); t=time.perf_counter(); [a@a for _ in range(20)]; torch.cuda.synchronize(); print('TFLOP/s', 20*2*4096**3/(time.perf_counter()-t)/1e12)"
# 3060: ~8-13. CPU fallback: <1. THIS is the real test.

uv run yolo predict model=yolo11n.pt source="https://ultralytics.com/images/bus.jpg" device=0
# 2-6 ms, not 40-80 ms
```

The `[tool.uv.sources]` block in `pyproject.toml` pins torch to the CUDA index.
Without it, resolution falls back to PyPI, which on Windows serves the CPU-only
build — you get a working pipeline at 3 fps while the 3060 sits idle, and nothing
warns you.

## Stage 0 — measure the broadcast first

A cutting or panning broadcast breaks homography **and** tracking continuity at
once. This is the cheapest fatal test in the project, so it runs before any
tracking code exists.

```powershell
uv run -m rtrack.acquire "https://www.youtube.com/watch?v=GSxbsE42o5o"
uv run -m rtrack.shots GSxbsE42o5o

# Identify the match. Reverse lookup by video id only works when TBA has the video
# linked, which is often not the case -- read the six teams off the score bug and:
uv run -m rtrack.tba --find-event "New England"
uv run -m rtrack.tba --match 2026necmp_f1m3
```

Writes to `out/stage0/`: `_motion.csv` (per-frame), `_shots.csv` (per-shot),
`_summary.json` (+ an automatic verdict), `_contact_sheet.png`.

**Then look at the contact sheet.** Five minutes of eyes beats the algorithm here.

### Decision tree

| Finding | Consequence |
|---|---|
| One static wide shot ≥ 85% of match | **Best case.** Proceed; Stage 2 ≈ 1 day. |
| Dominant static wide camera + cuts to closeups/replays | **Expected.** Wide shots only, one homography per camera, gap the timeline, re-associate identity across gaps. +0.5 day. |
| Continuous pan/zoom | **Bad.** Per-frame registration in Stage 2; BoT-SORT GMC mandatory in Stage 1. +2–3 days. |
| No sustained wide shot at all | **Kill case for this footage.** Concept still works but needs a fixed stands camera — it becomes a hardware project. |

### Findings — GSxbsE42o5o (run 2026-09-12)

| | |
|---|---|
| Match | **2026necmp_f1m3** — NE District Championship Final Tiebreaker |
| Alliances | red 6201 / 3467 / 6329 (Newsom) · blue 9644 / **1768** / 5687 (Burns) |
| Final score | 520 – 565, blue wins |
| Resolution / fps / duration | 1920×1080 / 29.97 / 214.7 s (6433 frames) |
| Codec | **AV1** — decodes fine in this OpenCV build at 130 fps, seek works |
| VFR warning? | No — `r_frame_rate == avg_frame_rate` |
| Shot count / median shot | 5 / 7.2 s |
| % time static / pan / zoom | 96.6 / 0 / 3.4 |
| Longest static shot | **177.1 s** (3.0 – 180.1 s), shot #1 |
| **Verdict** | **BEST CASE.** One locked-off shot (trans 0.04 px, scale 1.0000) holds the entire match. One homography, no re-association. |

The automated verdict initially read "expected case" because it scored the longest
shot as a percentage of *video* runtime (82.5%) — the other four shots are a title
card, a crowd reaction, a post-match field shot, and a fade to white. `verdict()`
now measures against a 150 s match length instead, which is the question that
actually matters.

**Shot table**

| # | Range | Dur | Class | What it is |
|---|---|---|---|---|
| 0 | 0.0–3.0 s | 3.0 s | static | Title card |
| **1** | **3.0–180.1 s** | **177.1 s** | **static** | **The match — locked-off wide camera** |
| 2 | 180.1–183.0 s | 2.9 s | static | Crowd reaction |
| 3 | 183.0–207.4 s | 24.4 s | static | Post-match field/crowd |
| 4 | 207.4–214.6 s | 7.2 s | zoom | Fade to white |

### By-eye check — and two findings that change the plan's assumptions

**Camera geometry is far better than assumed.** The plan budgeted for an end-on
view down the 54 ft length. It is actually an elevated **side** view across the
27 ft width, with the full field in frame and the long axis running left-right.
Depth range is therefore 27 ft, not 54 ft, so the perspective-error gradient
(risk R6) is much gentler than the plan's 25–40 cm/px far-field estimate. Confirm
with the Stage 2 Jacobian before relying on it.

**R2 (far-field robots too small) looks largely defused.** Bumper widths measure
roughly 80–130 px, not the feared sub-40 px. Bumper *numbers* are legible by eye
at 1080p ("6329", "1768", "5687"). Re-measure properly against the eval set, but
the `imgsz` sweep may matter far less here than budgeted.

Risks that got **worse** or are newly visible:

- **Occlusion is the real problem, not resolution.** The two Hub structures sit
  mid-field and hide robots behind them. Several sampled frames show only 4 of 6
  robots. `quality.framesWithAll6` will be the metric to watch, and risk R4
  (ID continuity through occlusion) is now the top Stage 1 risk.
- **Field staff in yellow shirts** walk the perimeter, and a dense crowd sits
  behind the glass. A generic person/robot detector will produce false positives
  there; a bumper-class model should not. Worth checking explicitly.
- **Hundreds of yellow game pieces** (fuel) cover the carpet.
- Score bug occupies the top ~250 px but does **not** overlap the playing field.
- **AprilTags on the Hub faces are visible** and look ~30–40 px, not the ~8–12 px
  the plan assumed. The §5.6 auto-calibration experiment is more promising than
  budgeted — still timebox it, but expect better odds.

TBA has **no video links for any 2026necmp match** (`videos: []` throughout), so
resolving a clip by YouTube id cannot work for this event. The match was
identified from the broadcast score bug and confirmed via `rtrack.tba --match`.
Expect this to be the normal path, not the exception.

## Stage 1a — open-vocabulary baseline: dead end, but informative

Ran `yolov8x-worldv2` (YOLO-World) against 7 full-res frames spanning auto through
endgame, at imgsz 640 and 1280, conf as low as 0.01, score bug cropped off.

| Prompt | Detections on the playing surface |
|---|---|
| `robot`, `machine`, `metal robot`, `robot with wheels` | **0** |
| `vehicle`, `cart`, `metal box`, `a small robot on a field` | **0** |
| `red bumper`, `blue bumper` | **0** |
| `toy car` | **1** (conf 0.14, 137 px, correctly on robot 5687) |

**This is not a scale problem and not a harness bug** — both were checked:

- Plain `yolo11x` on the same crop finds **69 objects**: 38 `person`, 30
  `sports ball` (the fuel), 1 `tv`. The image is fine.
- YOLO-World's text encoder is fine: `set_classes(["person"])` → 96 detections,
  `["person","ball"]` → 239.
- So `["robot"]` → 0 is a real result: **YOLO-World has no concept that matches an
  FRC robot.** Unsurprising — its "robot" prior is humanoid/industrial, not a boxy
  bumper-wrapped drivetrain.

**What this buys us:** risk R2 is answered and largely closed. A generic model
resolves 30 fuel balls and 38 people at this distance, and the one robot hit was
137 px wide — robots are comfortably resolved. The constraint is **vocabulary, not
resolution**, so the `imgsz` sweep budgeted in the plan may matter much less here.

**Next:** go straight to Stage 1b (Roboflow Universe FRC bumper datasets, trained
locally). Do not spend more time on open-vocab prompting. Needs a Roboflow account
to export the datasets.

## Stage 1b — borrowed dataset

```powershell
uv run -m rtrack.dataset audit roboflow
uv run -m rtrack.dataset prepare roboflow --map robots=robot --out data/datasets/frc_robot
uv run yolo detect train data=data/datasets/frc_robot/data.yaml model=yolo11s.pt `
    epochs=120 imgsz=1280 batch=12 device=0 project=runs name=robot_s patience=30
```

**Using: [`lucas-workspace-4nbar/frc-detection-r7jly`](https://universe.roboflow.com/lucas-workspace-4nbar/frc-detection-r7jly/dataset/4) v4** (CC BY 4.0).
813 images, 1920×729 broadcast frames, 2025 Reefscape.

The audit is the point of `rtrack.dataset`. Two candidates were tried:

| | `worbots-4145/2024-frc` | `lucas-workspace-4nbar/frc-detection` |
|---|---|---|
| Domain | stands-side, pits, close-ups | **broadcast wide shot** ✓ |
| Box width vs our 0.042–0.068 | p10 0.128, median 0.30 — **2–7× too big** | **p10 0.045, median 0.070** ✓ |
| Objects/image | mixed | **mean 4.8, max 6** ✓ |
| Season | mixed 2023 + 2024 | 2025 ✓ |
| Native width | 640 letterboxed | **1920** ✓ |
| Verdict | **rejected** | **in use** |

Scale and domain are what matter. Class schema and split hygiene are fixable;
a dataset of close-ups is not.

**Deviations this forces from the plan:**

1. **Single `robots` class — no alliance.** Red/blue cannot come free from the
   detector. `rtrack.alliance` recovers it from bumper hue in the lower 55–97% of
   each box, using the bands measured in `rtrack.prelabel`. It returns `None` rather
   than guessing when the margin is under 15%.
2. **Boxes are whole-robot, not bumper.** Worse for cross-year transfer than the
   plan's bumper rationale (2025 Reefscape robots have tall coral elevators that
   2026 Rebuilt robots do not) — but *better* for Stage 2, since a whole-robot box's
   bottom edge already is the floor contact point, needing no height correction.
3. **Ultralytics 8.4 defaults to TrackTrack** (CVPR 2025), not BoT-SORT as the plan
   assumed. Both are configured in `cfg/`; benchmark rather than assume.
4. **GMC is off** in both tracker configs. Stage 0 measured 0.04 px translation and
   1.0000 scale over the whole match — there is no camera motion to compensate, so
   global motion compensation can only add noise. Turn it back on for any handheld
   or panning source.

**Caveats:** only 10 distinct clips, so the model may overfit to those camera
setups. Their published split puts all 10 clips in *both* train and valid, so
adjacent near-identical frames straddle it and their reported metrics are inflated;
`prepare` re-splits by clip (8 train / 2 val). Validation mAP therefore measures
2025→2025 generalisation across two camera setups — it says nothing about 2026
transfer. **Only the 80-frame eval set decides the gate.**

### Result: 2025 → 2026 transfer FAILS

Trained YOLO11s to convergence (stopped at epoch 49/120; flat since ~25):

| 2025 val | 2026 footage |
|---|---|
| P 0.954, R 0.917, **mAP50 0.969** | **3.64 det/frame** against 6 · **44 track ids** in 400 frames · only **5% of frames** have 3 red + 3 blue |

A near-perfect in-domain score and an unusable result one season later. This is
exactly why the plan refused to let a borrowed dataset's own metrics decide the gate.

**Root cause: the whole-robot box taught a game-specific silhouette.** 2025 Reefscape
robots carry tall coral elevators, so the model learned *"tall vertical object on a
field = robot."* In 2026 Rebuilt that generalises to the wrong things:

- Predicted boxes run **2–3× too tall** (median height 155–320 px where real robots
  are ~60–90 px), stretching up off the robot into the crowd behind it.
- It fires confidently on the two mid-field **Hub structures**, which are genuinely
  tall objects standing on the field.
- It **misses low, boxy 2026 robots** that are large and obvious — 9644 in the near
  left corner goes undetected at conf 0.30.

Not a threshold artefact: swept conf 0.05/0.15/0.30 × imgsz 1280/1920. Lowering conf
to 0.05 yields 19–29 detections per frame that are mostly crowd and alliance wall,
while the box-height bias is unchanged. imgsz 1920 helps the aspect slightly and is
worth keeping.

**This vindicates the plan's original "bumpers, not robots" rule**, which we set
aside because the dataset was boxed that way. Bumpers are fixed by FRC rules — solid
alliance colour, 5–7.5 in tall, wide and low — so they cannot be confused with a Hub
tower and their geometry does not change between seasons.

### Dataset 3: `bear-metal-2046/2026-robot-detection` v11 — the right one

961 images, 1 class `robot`, CC BY 4.0, **actual 2026 Rebuilt season**, broadcast wide
shots from many events (PNW Bonney Lake, CA San Francisco, NE North Shore, FIM Lake
City, FNC Wake County). Audit: **scale 1.0× ours**, 6.1 objects/image, **zero leakage**
across all three splits. Splits and schema were already right — no `prepare` needed,
only absolute paths (`roboflow/data_abs.yaml`).

YOLO11s, imgsz 640, batch 32, stopped at epoch 57/150 (flat since ~45):
**P 0.91, R 0.89, mAP50 0.95**.

**Inference-mode sweep** on frames 1200–2000 (400 frames at 15 fps):

| | det/frame | track ids | after-open ids | 3+3 frames |
|---|---|---|---|---|
| 2025 model (for contrast) | 3.64 | 44 | 39 | 5.0% |
| A · letterbox 640 | 5.18 | 65 | 59 | 20.2% |
| B · **square** 640 | 4.72 | 28 | 22 | 18.2% |
| C · letterbox 1280 | 5.46 | 63 | 54 | 20.2% |
| D · square 640 + tracktrack | 4.56 | 33 | 27 | 16.0% |
| **E · square 640, conf 0.20** | 4.74 | **27** | **21** | 18.8% |

Three things this settled by measurement rather than argument:

1. **`--square` more than halves ID switches** (28 vs 65). Matching the dataset's
   "Resize (Stretch)" preprocessing gives stabler boxes frame to frame. Letterboxing
   finds slightly more detections but the tracks are far noisier.
2. **BoT-SORT beats TrackTrack here** (27 vs 33 ids), despite TrackTrack being the
   newer Ultralytics default. Worth re-checking if the detector changes.
3. Lowering conf 0.30 → 0.20 barely moves anything.

**Qualitatively the detector is now correct.** Boxes are tight and low on real
robots; no Hub towers, no boxes stretching into the crowd. Alliance balance is sane
(red 2.16 / blue 2.18, against the 2025 model's lopsided 0.91 / 2.54).

**Not yet at the gate**, and two of the three numbers are still short:

| Gate criterion | Target | Now |
|---|---|---|
| ID switches per 30 s | ≤ 2 | **~21 per 27 s** |
| Frames with 3+3 | (implied high) | **18.8%** |
| Throughput | ≥ 15 fps | **26.8 fps** ✓ |

**The recall number cannot be interpreted yet.** "4.74 of 6" assumes six robots are
visible, but Stage 0's by-eye pass found frames showing only four — robots hide
behind the two mid-field Hub structures. Some share of the apparent 21% miss rate is
robots that are genuinely not in frame. Separating "missed a visible robot" from "was
never visible" is precisely what the 80 labelled eval frames are for, and until they
exist neither a pass nor a fail can be declared honestly.

**Known alliance failure:** a robot sitting on a coloured field element can be
misread — 9644 (blue) reads red at f1800 in config B while parked on the red ramp,
because the hue sample band catches ramp pixels under the robot. `alliance.py`'s band
needs tightening, or weighting toward saturated pixels contiguous with the box centre.

### ID stability: 65 → 8 track ids

```powershell
uv run -m rtrack.track GSxbsE42o5o --imgsz 640 --square --conf 0.20
uv run -m rtrack.stitch out/stage1/GSxbsE42o5o_tracks.jsonl
```

**Diagnosis first.** The fragmentation was *not* detection flicker — only 1 of 27
tracks lived ≤5 frames, and surviving tracks were long and high-confidence (0.76–0.87).
Real robots were being lost and re-acquired under new ids. Every one of the 14
observed handoff gaps was ≤27 frames, **well inside the 60-frame `track_buffer`** — so
memory was never the constraint. One handoff (`#9 → #21`) spanned 23 frames while the
robot moved **5 px** and still spawned a new id.

Cause: **Kalman drift.** While a robot is occluded the filter extrapolates its
velocity, so the predicted box walks off the robot. On reappearance IoU is ~0 and no
threshold can match it.

**Beware the id-count metric.** A tracker that teleports one id between six robots
scores a perfect 6. Configs must therefore be ranked with the plan's kinematic check
(§5): an FRC robot tops out ~5.5 m/s ≈ 60 px/frame here, and anything beyond that is
guessing, not tracking.

| config | ids | det/f | **teleports** | worst px/f | clean ids |
|---|---|---|---|---|---|
| letterbox 640 | 65 | 5.18 | 0 | 54 | 65 |
| square 640, conf 0.20 | 27 | 4.74 | 0 | 56 | 27 |
| + ReID (proximity 0.1) | 23 | 4.78 | 0 | 56 | 23 |
| **+ match_thresh 0.95** | **21** | **4.81** | **0** | 56 | 21 |
| ReID, no IoU gate | 9 | 4.88 | **1190** | **1041** | **0** |
| **+ `rtrack.stitch`** | **8** | **4.81** | **0** | 56 | **8** |

The `ReID, no IoU gate` row is the trap: the most attractive id count in the table
and the only config with zero usable tracks. Its overlay shows trails criss-crossing
the whole field. **ReID stays off** — measured, not assumed.

**`rtrack.stitch` does the rest offline.** Online, the tracker must decide in real
time; offline we can look at a fragment's death and a later fragment's birth together
and ask whether one robot could plausibly have done both. It applies the plan's Stage
3 gates one level down — alliance match, reachability at ≤5.5 m/s, no temporal
overlap, and **uniqueness: refuse to merge when two predecessors are comparably
plausible**. On this segment: 21 → 8, 13 merges, 1 ambiguity correctly left split.

Remaining gap to 6 is honest: 8 tracks, 4.81 det/frame, 21.2% of frames at 3+3 — and
some of that shortfall is robots genuinely hidden behind the Hub structures rather
than missed. The eval set is what separates those.

### Lost-track velocity decay (`--lost-decay`)

Stitching repairs fragments after the fact. `rtrack.botpatch` attacks the cause, and
matters for Stage 4 where offline repair is impossible — a livestream cannot look
ahead.

Ultralytics' `BOTrack.multi_predict` already zeroes the **size** velocities of any
track that is not Tracked. It never extended that reasoning to **position**, so a
lost track keeps coasting at its last velocity: at 5 m/s a 2 s occlusion predicts 10 m
of travel, over half the field. Measured reality is the opposite — observed handoff
distances were 5–184 px (median ~80), because a robot going behind the Hub is usually
manoeuvring, not barrelling through.

The patch multiplies the position velocities by γ each frame while lost. Since
`predict` runs once per frame, that compounds to γⁿ — exponential decay, bounded
total drift of `v₀·γ/(1−γ)`.

| γ | ids (400 frames) | det/f | teleports |
|---|---|---|---|
| off | 21 | 4.81 | 0 |
| 0.9 | 19 | 4.83 | 0 |
| 0.8 | 18 | 4.84 | 0 |
| **0.7** | **15** | 4.84 | 0 |
| 0.0 (hard stop) | 15 | 4.85 | 0 |

All clean — unlike the ReID experiment, this reduction is real. γ=0.7 is the default
recommendation over 0.0 because it still lets a 1–2 frame gap benefit from genuine
motion while a long one converges to stationary.

**The compounding benefit is that it makes stitching safer**, not just shorter:

| full match (2654 frames) | ids | teleports | worst px/f |
|---|---|---|---|
| raw | 122 | 4 | 67 |
| raw + stitch | 49 | 4 | 67 |
| **γ=0.7** | **97** | 4 | 66 |
| **γ=0.7 + stitch** | **42** | **4** | **66** |

Decay cuts the merges stitch must make (85 → 55), and every merge is a chance to be
wrong.

`botpatch` monkeypatches ultralytics internals, so it is opt-in, and it asserts the
upstream function still matches what it was written against — it raises rather than
silently no-opping after an ultralytics upgrade.

### Fixing the "more than 3 on one alliance" spikes

An alliance cannot have four robots, so any frame where one side exceeds 3 is wrong
**by construction** — free ground truth, no hand labels required. That constraint
drove this whole round of fixes.

**Diagnosis first.** Of 123 over-3 frames on the full match, **104 had ≤6 total
detections** — so the cause was alliance *misclassification*, not extra detections.
Separately, 184 frames (6.9%) carried a duplicate box at IoU > 0.55.

| fix | what it addressed |
|---|---|
| `--nms-iou 0.55` (was Ultralytics' 0.70) | duplicate boxes stacked on one robot |
| tuned `alliance.BAND_*` | sample band catching coloured field elements |
| ~~track-level alliance vote~~ | **tried and reverted — made it worse** |

**The sample band was swept, not guessed** (0.55/0.97/0.00 → **0.65/0.90/0.25**):
start lower in the box (above is superstructure and background), drop the last rows
(carpet, shadow, and the coloured ramp a robot parks on), and trim the left/right
quarters (field elements bleeding in behind the robot).

| | before | after |
|---|---|---|
| frames with an alliance over 3 | 123 (4.6%) | **50 (1.9%)** |
| frames with a duplicate box | 184 (6.9%) | **9 (0.3%)** |
| frames with >6 detections | 56 | **11** |
| track ids (after stitch) | 42 | **32** |
| raw ids (before stitch) | 97 | **78** |
| kinematic violations | 4 | 4 |

**A negative result worth keeping.** Assigning one alliance per track by weighted
vote seemed obviously right — a robot cannot change alliance mid-match. It made
things **much worse**: over-3 frames went 123 → 412. Voting assumes per-frame errors
are random noise around a correct majority, but they are systematic: a robot parked
on the red ramp is misread for most of its visible life, so the vote entrenches the
error across the entire track instead of averaging it out. Kept as opt-in `--vote`,
off by default, and only worth revisiting once per-frame classification is right
more often than not.

**Deliberate trade:** undecided calls rose 9.7% → 14.7%, which *lowered* the 3+3 rate
(13.8% → 9.5%). That is the intended direction. An undecided detection still carries
a valid position and its alliance is recoverable at Stage 3 from the seeded team
number; a confidently wrong alliance silently corrupts a team's data. Precision over
recall here, because Stage 3 can supply the missing half and cannot detect the wrong
half.

Remaining: the counts plot ceiling is now capped at 4 (was 5–6), so the spikes are
largely resolved. The **dips** to 1 and 0 are what is left — see below.

### The dips are occlusion, not detector weakness

This was the open question blocking a verdict: "4.58 of 6" conflates *missed a
visible robot* with *robot was behind a Hub*. It turns out to be answerable without
any hand labels.

**1. The robots are all accounted for.** A frame inside a track's lifespan where that
track was not detected is a provable miss — the robot is demonstrably there, seen
before and after.

```
detections/frame 4.58  +  provable misses 1.36  =  5.95   (6.00 = fully explained)
```

The system is not failing to *know about* robots. It is failing to *see* ones it is
already tracking.

**2. The misses are localised.** Miss rate by field column, normalised by how often
robots are there at all:

| x band | miss rate | what is there |
|---|---|---|
| 720–960 | **0.07** | open centre field |
| 960–1200 | 0.14 | open centre field |
| 480–720 | 0.41 | red Hub |
| 1680–1920 | 0.80 | right edge / alliance station |
| **1200–1440** | **1.05** | **blue Hub** |

`out/stage1/miss_rate.png` shows the same thing: heat concentrated beside both Hubs
and at both alliance-station edges, with the open centre field clear. That is an
occlusion signature. A robot behind a 2 m tower is not a detection problem.

**3. Detector tuning confirms no headroom.** On a 400-frame segment:

| config | det/frame | track ids |
|---|---|---|
| square 640, conf 0.20 | 4.81 | **14** |
| square 960, conf 0.20 | 4.79 | 17 |
| square 640, conf 0.10 | 4.85 | 14 |
| letterbox 960, conf 0.20 | 5.00 | 26 |

Neither resolution nor threshold buys anything. Letterboxing finds ~4% more
detections and costs 86% more ids — the same trade seen earlier.

**4. Only ~18% of the lost time is cheaply recoverable.** Gap lengths (287 gaps,
3620 missed frame-slots):

| gap ≤ | share of all missed time |
|---|---|
| 10 frames (0.7 s) | **17.7%** |
| 20 frames (1.3 s) | 33.2% |
| 30 frames (2 s) | 42.1% |
| 60 frames (4 s) | 87.2% |

Short gaps can be interpolated with confidence. The bulk are 1–6 s occlusions, over
which a robot can cross much of the field, so interpolating them would be inventing
data at exactly the precision Stage 2 is meant to deliver.

**Conclusion: Stage 1 is at the practical ceiling for a single fixed camera.** The
residual loss is physical, not algorithmic — the camera cannot see through the Hub.
The right response is the one the plan's schema already anticipates: interpolate
short gaps, record long ones honestly in `gaps[]` with a reason, and let `quality`
carry the coverage number. Chasing the remaining 1.36 det/frame with a better
detector is not a good use of effort.

### Bug found: the reachability gate double-counted stride

`gap_frames` was already in processed frames, then `gap_s = gap_frames * step / fps`
multiplied by the stride **again**, making every budget 2× too generous. That, not the
linear growth I first blamed, is why merges were unconstrained: stitching the full
match **introduced 3 teleports and a 125 px/frame jump** the raw tracks did not have.

Fixed to `gap_s = gap_frames / fps`, plus a `HARD_CAP_PX` of 900 (~one field width)
past which the gate stops discriminating. Stitching now adds **zero** kinematic
violations. It costs a few ids (35 → 42 on the full match) because the gate is
correctly stricter — 42 honest tracks are worth more than 35 with fabricated merges.

**What does work** and should be kept:
- The pipeline runs end to end at **21.6 fps** with tracking on the 3060.
- `rtrack.alliance` resolves red/blue on **94.8%** of detections — the hue bands
  transfer fine even when the detector does not.
- `rtrack.dataset audit` correctly predicted the previous dataset was unusable and
  this one was worth trying. Its blind spot was *box semantics*: it checks scale,
  domain and class schema but not whether the boxes enclose the part of the object
  that stays constant across seasons.

## Stage 1 eval set — 80 frames, ready to label

```powershell
uv run -m rtrack.evalset sample GSxbsE42o5o        # choose frames -> eval/frames.index.json
uv run -m rtrack.evalset export GSxbsE42o5o --prelabel   # extract PNGs + draft labels
```

Built **before** trying any model, on purpose: the Universe datasets are 2024
Crescendo footage being asked to transfer to 2026 Rebuilt, and that is not something
you can judge by eye from an overlay.

**80 frames, all from the single static match shot** — 56 spaced uniformly across
auto/teleop/endgame, 24 hard-mined. The plan called for "20 from secondary shots",
which does not apply here: this clip's other four shots are a title card, a crowd
reaction, a different-camera driver-station view, and a fade. None show the field.

Hard-mining scores each frame with the colour pre-labeller and favours frames with
**fewer than six bumpers found** (occlusion, weight 0.5), **bumpers close together**
(scrums, 0.3), and **small bumpers** (distance, 0.2). Picks are spaced ≥25 frames
apart so they are not near-duplicates. The selected hard frames come back with 2–4
boxes against 6 expected, which is the signal we wanted.

*Caveat worth knowing when you label:* part of that hardness signal is an artifact.
The pre-labeller's ROI excludes the alliance-wall zones, so a frame with robots
parked at the walls scores "hard" partly because the aid cannot see them, not purely
because a detector would struggle. Those frames are still worth over-sampling —
robots against walls and clutter are genuinely hard — but do not read the score as
ground truth about difficulty.

### The draft labels

`--prelabel` writes a YOLO-format `.txt` beside each PNG (class 0 `red_bumper`,
1 `blue_bumper`), plus `eval/classes.txt`. Roboflow imports this directly.

**Correct them, do not trust them.** Measured on sample frames: about **6 true
positives and 3 false positives** per frame. The false positives cluster on the Hub
faces and ramp edges just outside the exclusion boxes. Deleting a wrong box is
faster than drawing a missing one, which is why the thresholds lean permissive.

`rtrack.prelabel` is a **labelling aid, not a detector.** It works only because
Stage 0 established a locked-off camera, one venue and constant arena lighting. It
cannot survive a venue change and has no concept of a robot as opposed to a red-ish
blob. It must not leak into Stage 1.

```powershell
uv run -m rtrack.prelabel --tune eval/frames/<one>.png   # hue histogram, to re-tune
uv run -m rtrack.prelabel --frames eval/frames --debug   # overlay JPEGs to inspect
```

`eval/frames/` (185 MB of PNGs) is gitignored; `frames.index.json` is tracked, so
the exact frame selection is reproducible from the video id alone.

## Stage 2 — field coordinates

### The field reference was wrong, and the AprilTag layout caught it

`calib/field_ref_2026.json` originally took the widely-quoted 54 × 27 ft and forced
the field-image rectangle to a 2.000 aspect, then "verified" that the implied edges
landed near guardrail structures. That verification was **circular** — with a ±25 px
search window and structures everywhere, it could hardly fail.

WPILib's authoritative layout says the field is **16.5410 × 8.0690 m**, aspect
**2.0499**. The width was off by 2%.

Re-derived without assuming an aspect: the horizontal rails are thin (17–20 px) and
unambiguous, giving 1410 px for 8.0690 m = **174.74 px/m**, with the x bounds
following from that scale about the detected centre line (x=1949.5). Both implied
edges land inside the 67 px-thick alliance walls, which is consistent.

This also prompted a better architecture: **metric truth comes from the tag layout,
and the field PNG is only a display backdrop.** The render is an illustration; any
inaccuracy in it now affects where a dot is drawn, never a recorded position.

### Getting the layout

Not in the allwpilib repo as JSON and not in the `robotpy-apriltag` wheel — the
layouts are compiled into the native library. Extracted by running the authoritative
library ephemerally and dumping it, so nothing enters the project environment:

```powershell
uv run --with robotpy-apriltag python scratch/dumplayout.py   # -> data/field/2026-apriltag-layout.json
```

32 tags at three heights (0.552, 0.889, 1.124 m).

### AprilTag auto-calibration: structurally blocked from this camera angle

`rtrack.autocal` implements the idea: use tag **corners** rather than centroids
(8+ constraints against 7 unknowns, so camera alignment need not be assumed), and
recover focal length by sweeping f and keeping the `solvePnP` with lowest
reprojection error. Corner-order and axis-sign conventions are **searched, not
guessed**, since both fail silently.

It runs, converges, and is wrong:

| | |
|---|---|
| reprojection error | 1.44 px over 12 points — looks excellent |
| recovered camera height | **0.15 m** (it is plainly elevated) |
| recovered focal length | **700 px — the sweep's lower bound** |
| centre-line validation | off by **1680 px** |

**Root cause:** the only detectable tags are 2, 11 and 21, and they sit at
*identical* y (4.638) and z (1.124) with *identical* yaw (90°), differing only in x.
PCA singular values `[6.10, 0, 0]` — exactly collinear, with coplanar parallel faces.
**Coplanar PnP with unknown focal length is degenerate**: focal length trades against
depth for pixel-identical projections. Hence f running to the boundary.

This is the conditioning risk flagged before starting, and it is a property of the
field and camera angle, not of the implementation. The camera sees one face of each
Hub, and those faces are parallel by construction.

**The centre-line validator earned its keep.** A 1.44 px reprojection error would
have been reported as success by any check internal to the fit. An independent
known-geometry check caught it immediately. Never calibrate without one.

**What would unblock it:** a tag at a different height or orientation. The layout has
them, but the perpendicular Hub faces are edge-on and the alliance-wall tags face
away. Failing that, a focal length from any independent source makes coplanar PnP
well-posed again — so tags remain useful for *re-deriving pose* at a venue already
calibrated once, just not for calibrating from nothing.

### Detection strategies tried, and the verdict

Four search strategies, on the theory that one non-coplanar tag breaks the
degeneracy:

| strategy | cost | tags found |
|---|---|---|
| median plate, 1-3x whole frame | 23 s | `[11]` |
| + raw frames, 1-2x | 15 s | `[2, 11, 21]` |
| + tiled 5x over plate (overlapping 320 px tiles) | 6 s | no change |
| + tiled 5x over 5 raw frames | 32 s | no change |
| bootstrap (predict all 32 tag positions, re-detect in ROIs) | 3 s | **0** — the seed pose is itself degenerate, so its predictions are wrong |

Tag 1 (z=0.889, yaw=180 — the one that would break coplanarity) *is* detectable, but
only ~5% of attempts. Recovering it took 150 frames × 3 whole-frame scales, several
minutes — which defeats the point of automatic calibration. And even with it, the
fit's centre-line slope was still wrong.

**Verdict for this footage: automatic calibration does not work.** The blocker is the
camera angle, not the implementation — this camera sees one parallel face of each
Hub. A camera positioned differently could see non-coplanar tags, so `autocal` is not
useless in general; it is blocked here.

**The valuable outcome is that it says so.** It reports conditioning 0.0000, flags the
implausible 134° FOV, and returns `verdict: NOT TRUSTWORTHY` rather than a confident
wrong homography. For unattended operation at venues nobody is attending, refusing is
the required behaviour — a silently wrong field mapping would corrupt every position
downstream with no signal that anything was wrong.

### Follow-up: lines + AprilTags combined. Tested, and it does NOT work.

The idea was to split the problem so neither leg has to do what it cannot: **lines
supply intrinsics and distortion** (no scene knowledge needed, so `autocal`'s
over-parameterised "3 coplanar tags for intrinsics *and* pose" is avoided), then
**tags supply pose only**, now that intrinsics are fixed. Tested on f1m3, which has a
manual calibration at 0.10 m to score against. Both legs failed independently.

**Leg 1 — lines → distortion. Failed, and the cause is measurable.** LSD finds plenty
of segments (54–81 per camera with robots and broadcast furniture masked) and the
barriers are clearly among them. But chaining segments and fitting distortion to their
endpoints does not recover the known lens:

| | f | k1 | cx, cy |
|---|---|---|---|
| known (21 clicked points, plumb-validated) | 398.9 | **−0.0308** | 881, 478 |
| fitted from chained LSD segments | 166.6 | **+0.0049** | 1600, 209 (at the constraint boundary) |

The diagnostic that settles it: scored on those chains, the **known-correct lens (3.23 px)
is worse than assuming no distortion at all (2.75 px)**, and a k1 sweep with everything
else fixed at known values minimises near 0.0 rather than at −0.031. The chains are not
predominantly world-straight lines, so the metric is fitting noise.

Why: LSD gives ~6 endpoints per chain with ±1–2 px localisation error over a ~260 px
span, and the distortion sagitta over that span is smaller than the noise.
`calibrate.trace_plumb` succeeds on the same footage (17.15 → 0.61 px) because it samples
**densely** along one long high-contrast edge. **The approach is sound; using LSD
endpoints as its input is not.** The real target is automating dense sub-pixel edge
tracing, not substituting a segment detector for it.

**Leg 2 — tags → pose. Failed, and this one is a resolution limit, not a bug.** Only
**2–3 of 32 tags decode**, consistently ids 2 and 11, and that is true on all three
cameras tested. Comparing `autocal`'s tags-only floor homography against the manual one
across 45 sampled image points:

* disagreement **mean 7.5 m, median 7.2 m, max 17.9 m** — on a field 16.46 m long
* the manual mapping spans y from −0.1 to 6.6 m; autocal compresses the same region
  into 2.9–5.3 m

This matches the prediction made before `autocal` was written: a 6.5 in tag at 12 m in
1080p is 8–12 px across, and 36h11 decoding wants 20+. No amount of implementation
effort adds tags that are not resolvable.

**Verdict: do not ship this as the baseline fallback.** A fallback that is wrong by
metres is worse than no fallback, because every downstream position inherits the error.
Manual calibration stays the baseline.

**What survives from the exercise:**

* **Calibration reuse is confirmed and is the real win.** f1m3's calibration applied to
  f1m2 — different match, same camera — lands the reprojected field boundary on the real
  barriers with no adjustment. Calibration is per *camera*, and should be keyed and
  cached that way rather than per video id. Note it is per camera, **not** per event:
  sf11m1 (Burns Division) uses a different, higher camera and f1m3's calibration is
  visibly wrong on it.
* **The line leg has an identified fix** (dense edge tracing) rather than a dead end.
* **Cross-video registration is now the more promising automation route**, precisely
  because it does not depend on tags. The field is identical at every event, so
  calibrating a new camera is registration against an already-calibrated reference view
  of the same physical object rather than calibration from scratch. Untested.

Runtime is ~80 s, most of it searching for tags that are not there. Dropping the two
tiled passes (which added nothing here but may help at other venues) brings it to
~40 s.

### Manual calibration — and a units bug that looked like bad clicking

```powershell
uv run -m rtrack.calibrate GSxbsE42o5o --frame 200 --interactive   # or --plate
uv run -m rtrack.project   GSxbsE42o5o --tracks out/stage1/MATCH3_st.jsonl
uv run -m rtrack.routes    GSxbsE42o5o --end 25 [--panels]
```

The first interactive run produced per-point errors of
`[0.0, 2.73, 1582.9, 14.9, 0.0, 0.0, 3.16, 0.0]` and a warp smeared into radial
streaks. The four **exact zeros** were the tell: a homography has 8 DOF, so any 4
point-pairs fit perfectly by construction. RANSAC had kept a minimal 4-point set and
called everything else an outlier.

**Cause: `ransacReprojThreshold=3.0` was in FIELD PIXELS.** At 174.74 px/m that is
**1.7 cm** — far tighter than achievable click precision, so every honest point was
rejected. The threshold is now specified in **metres** (default 0.40).

The clicks were fine — all eight of them, as the next section explains. Least-squares
over all 8 points gives:

| | |
|---|---|
| least-squares error, all 8 points | **mean 0.44 m, max 1.01 m** |
| per-point | `[0.574, 0.179, 0.26, 1.014, 0.466, 0.347, 0.745, 0.705]` |
| sensitivity (cm of field error per px of click error) | min 1.2, **median 2.3**, p90 4.1, max 5.5 |

Diagnostics added so this cannot recur silently: degeneracy detection *before*
fitting (vertical/horizontal span and quadrant coverage), an all-point least-squares
error printed alongside RANSAC, a warning when RANSAC keeps ≤4 points, and `--plate`
to calibrate against a median-stacked image with no robots obscuring the carpet.

### The lower residual was the WORSE fit

Two errors compounded here; both are worth keeping written down.

**1. Outlier rejection stripped a region of its only constraint.** With the threshold
at 0.40 m, RANSAC dropped 3 of 8 points. Residual fell 0.44 m → 0.22 m and it looked
like an improvement. It was not: the dropped points were the only ones on the LEFT of
the image, so the homography extrapolated unconstrained across half the field and
placed the boundary metres out. **With 8–10 points against 8 DOF there is no
redundancy to spare — rejecting any of them can leave a region unanchored.** The
default threshold is now 2.00 m, deliberately loose; tighten it only with many more
points than DOF.

**2. I assumed the two high-residual points were bad clicks. They were not.** They had
large residuals *because* the fit was being pulled away from them, not because they
were wrong. Refitting with all 8: mean 0.44 m, max 1.01 m — larger numbers, correct
geometry.

### Validation — use the reprojection overlay, not the warp

`out/stage2/<id>_reproject.png` draws the predicted field grid back onto the video
frame. This is the diagnostic that works. The warp shows where video pixels *land*,
which is hard to judge when most of the target is empty; a badly wrong homography can
look plausible there. The reprojection compares predicted geometry against the real
thing in the same image.

**The decisive check was corner visibility.** With the 8-point fit, `far-L` and
`far-R` land inside the frame while `near-L` and `near-R` fall outside — matching the
observed footage exactly, where the far corners are visible and the near ones are
not. The 6-point fit put `far-L` outside the frame at x=-84, which is what exposed it.

**Retracted:** an earlier note here claimed the fuel pile aligning in the warp was
independent validation. It is not — the fuel is a pile of balls with real height seen
obliquely, while the render draws it as a flat top-down array, so apparent agreement
there says nothing about the floor plane.

`rtrack.calibrate` now also warns when a third of the image has no calibration point
in it, since that is the condition that made this failure invisible.

**`rtrack.project` on the full match: 12,165 samples, 32 tracks, `off-field 0 (0.0%)`.**
Every projected position falls inside the field rectangle. A wrong homography would
scatter positions outside, so this is a second free check. Kinematic violations
(>5.5 m/s) are 1.07%, consistent with the known ID switches.

Two biases are documented rather than silently corrected: the box bottom edge is the
floor contact (no bumper-height correction needed, since the detector was trained on
whole-robot boxes), and we see the robot's *near* face, biasing position ~0.35 m
toward camera. "Position" therefore means **near-face floor contact** — consistent
frame to frame, fine for routes and zone occupancy.

### Route plots

`rtrack.routes` renders paths over the field in real metres, combined or as small
multiples (one robot per panel — overlaid paths turn to spaghetti). Gaps longer than
`--max-gap` are **broken, not bridged**: a robot hidden behind a Hub for two seconds
could be anywhere, and a straight line across that gap would invent data at exactly
the scale the plot exists to show.

First 25 s of the test match: 8 routes, red staying left and blue right as expected
for auto. Output at `out/stage2/GSxbsE42o5o_routes[_panels].png`.

## Stage 4 — capture, curate, publish

The loop, and where each piece lives:

```
event archive (one multi-hour YouTube VOD)
   │  rtrack.replay        slice one match out by TBA time
   ▼
data/raw/<event>_<match>.mp4
   │  rtrack.pipeline      the whole chain, resumable, one event at a time
   ▼
out/stage3/<match>_curate_frames.json      the bundle: 18 frames, crops, machine guess
   │  rtrack.relay push-bundle
   ▼
Cloudflare Worker + KV  ←──────→  public/rtrack/curate.html   (a phone, anywhere)
   │  rtrack.watch        notices answers and finishes the match unattended
   ▼
public/tracks/<match>.json + <event>_gallery.npz + <event>_refs.json
```

**Slicing (`replay.py`).** An event stream is one archive; matches are found by
`offset = TBA actual_time − stream release_timestamp`, then cut with ffmpeg range-seek.
Not yt-dlp `--download-sections`: that does not range-fetch a `was_live` VOD and pulls
the whole multi-hour file to extract 150 seconds.

**Sequencing (`pipeline.py`).** Takes a per-event lock, and that is correctness rather
than politeness — match N+1's identity votes come from the gallery that match N's
*curation* produced, so two runs in flight would both read a stale gallery and then race
to rewrite it. `--prep-only` runs just track/stitch/appear, none of which depend on the
gallery, so a whole event can be prepped unattended and each match then needs only ~1 min
of gallery-dependent work immediately before a human sees it.

**The relay (`rtrack-relay/worker.js`).** The machine holding the video is behind a home
NAT and the person doing the work is in a venue with a phone. Writing bundles needs the
home-machine token; **answering is open by default**, because the curator page is served
from public GitHub Pages and any token embedded in it to make the flow automatic would be
public too — friction with the appearance of security and none of the substance. The blast
radius of an unauthenticated answer is one match, in a 24-hour KV entry, reviewed before
it is applied. Set `RTRACK_ANSWER_TOKEN` if that stops being an acceptable trade.

Operating envelope and the free-tier limits that actually bite: [RELAY_LIMITS.md](RELAY_LIMITS.md).

**The watcher (`watch.py`).** Polls for answers and runs the back half of the pipeline for
each. A watcher rather than `pipeline --relay --wait` because that mode blocks on ONE
match, which is the wrong shape: a curator works through several in whatever order suits
them, and the machine should pick up each as it lands. Newness is the answer timestamp vs
the corrections file mtime, not a processed-set, so re-curating a match re-runs it instead
of being skipped as seen.

## Curation quality — what makes a frame worth a click

Set cover picks the frames that label the most of the match. It never asks whether a
curator can *read* those frames, and that gap cost a whole match: qm9 shipped 95% coverage
and 96% pre-fill, and the review came back "an awful lot of mistakes".

Coverage and legibility are different quantities and only one of them was in the
objective. Three things closed the gap.

### Legibility, measured against the curator rather than invented

Curators mark a box "can't tell" when they cannot read it, which is ground truth for
exactly this question — 607 readable against 34 unreadable boxes over seven curated
matches. Every candidate feature was scored by how well it separates them (AUC, 0.5 is a
coin flip):

| feature | AUC | note |
|---|---|---|
| `satw` — bumper-blob width ÷ box width | **0.750** | best single feature; this *is* "side-on view" |
| `sat` — bumper-coloured fraction of band | 0.748 | |
| `whiteon` — white **on the bumper blob** | 0.728 | |
| `sep` — separation from other boxes | 0.711 | |
| `boxh` — box height | 0.533 | |
| `blurn` — log Laplacian variance | **0.386** | ANTI-predictive |
| `white` — white anywhere in the band | **0.366** | ANTI-predictive |
| digit-shaped connected components | 0.43–0.48 | no better than chance |

**The first version scored 0.602 and weighted `white` highest, at 0.45.** It was the worst
feature available and pointed the wrong way: unreadable boxes average *more* white than
readable ones (0.079 vs 0.054), because a poorly-fitted box fills its band with bright
background. Asking the same pixels a better question — white *on* the bumper, found by
dilating the bumper mask — turns 0.366 into 0.728 with no new information.

Blur is a trap for the same reason: sharp background texture inflates Laplacian variance
precisely when the bumper is not filling the band. It was nearly added on the reasoning
that smeared bumpers are unreadable, which is true and irrelevant.

**Detecting the digits directly does not work at this resolution** and this is worth not
re-trying: at ~60 px of robot height the bumper band is ~25 px and a digit ~15 px, and
connected components filtered to digit-like aspect and height score 0.43–0.48. Whether a
bumper *face* is turned toward the camera is answerable here; whether its digits are
legible is not. The score asks the first and infers the second, and that is what gets it
to **0.873** — verified through the shipped code path, not the research script.

Legibility is also scored **relative to each track's own best view**. On an absolute
scale a robot parked at the far end reads badly in every frame it appears in, so it only
drags down whichever frame contains it — the objective cannot tell "a bad view of this
robot" from "this robot is never readable anywhere", and only the first is actionable.

### Best views: the frame need not carry the identification

A box that is not a good look at its own robot gets that robot's clearest crops attached,
from anywhere in the match — on qm10, 62% of boxes were not. The frame keeps doing the one
job only a frame can do: six boxes in one frame are provably six different robots, which
is what makes the answers mutually exclusive and stops one team being used twice. The
crops are spread ≥3 s apart on purpose, because a strip that disagrees with itself is a
track that changed robot partway, and that is the failure that has cost the most here.

### Team references: the gallery in a form a person can read

The appearance gallery holds 48 grayscale numbers per team. Long-pressing a team in the
curator shows the same knowledge as images: every box currently assigned to that team in
*this* match (free — the crops are already in the bundle, and it works on an event's first
match), plus crops confirmed in *earlier* curated matches. The two answer different
questions. In-match is a consistency check and is self-referential: six wrongly-seeded
boxes look perfectly consistent. The confirmed crops are the tiebreaker.

Written by the gallery step, which already decodes each curated match and was throwing the
pixels away. Ranked by `bumper_face` rather than full legibility — separation measures
whether a box is *ambiguous*, which is meaningless for a reference crop of an identity
that is already settled. At most 2 per match, 6 per team: six frames of one drive down the
field teach less than three views from three matches.

### Difficulty is measured, not asked

`unknown` is a contaminated signal — it mixes genuinely illegible with occluded, with
ambiguous-between-two-robots, with gave-up, and its rate depends on how hard the curator
was trying. The fix is not to ask them to rate photo quality: that competes with the label
for their attention and would tune the selector toward pretty crops over informative ones.
So each answer also carries `ms` (time on that box) and `aid` (whether the best-views strip
or reference sheet was opened before answering). Opening either is a direct admission that
the crop on screen was not enough, which is precisely what the score predicts. It rides
alongside the label and cannot corrupt it.

## Layout

```
src/rtrack/
  config.py     paths, field constants (WPILib meters), class map, Stage 0 tuning
  source.py     ★ FrameSource abstraction -- the reason Stage 4 is cheap
  acquire.py    Stage 0a: yt-dlp + ffprobe
  shots.py      Stage 0b: cut detection + camera motion + contact sheet
  tba.py        resolves a match -> the 6 team numbers (by key, or by video id)
  evalset.py    Stage 1: stratified frame sampling + hard-mining
  prelabel.py   colour-threshold draft boxes -- a labelling AID, not a detector
  dataset.py    audit / remap / re-split a borrowed dataset
  alliance.py   red|blue from bumper hue inside a detected box (real pipeline)
  track.py      Stage 1: detect + track -> tracks.jsonl (every later stage reads this)
  stitch.py     Stage 1: rejoin tracks the tracker dropped and re-acquired
  botpatch.py   BoT-SORT behaviour changes that belong to us, not to ultralytics
  overlay.py    Stage 1: the proof-gate video + counts plot
  calibrate.py  Stage 2: click correspondences -> homography (once per camera)
  autocal.py    Stage 2: AprilTag/line auto-calibration. DEGENERATE from this angle
  lens.py       Stage 2: radial distortion, r_d = r(1 + k1 r^2 + k2 r^4)
  project.py    Stage 2: image px -> field metres, + the off-field filter
  routes.py     Stage 2: route plots, auto/teleop window detection
  appear.py     Stage 3: grayscale band descriptors, cached per detection
  reid.py       Stage 3: the event gallery + per-track identity votes + team refs
  solve.py      Stage 3: CP-SAT global assignment (deterministic; see the notes)
  robots.py     Stage 3: the driver -- cuts, conflicts, corrections, custody
  corrections.py  resolves human labels onto tracks BY POSITION, not by track id
  chicklets.py  per-track crop strips (the desktop curation mode)
  curate.py     Stage 3: the curation bundle -- frame choice, legibility, best views
  export.py     Stage 3b: rtrack-tracks v1 -> public/tracks/, + the manifest
  replay.py     Stage 4: slice one match out of an event archive
  pipeline.py   Stage 4: one match end to end, resumable, per-event lock
  relay.py      Stage 4: push bundles / pull answers over the Worker
  watch.py      Stage 4: notice answers as they arrive and finish those matches
cfg/            botsort_frc.yaml, tracktrack_frc.yaml -- tuned, rationale inline
calib/  eval/  viewer/  corrections/   tracked (small, valuable)
data/  out/  models/  runs/  eval/frames/   gitignored
```

`corrections/` is tracked deliberately: it is hand-made ground truth that cost real
human time, it is a few KB per match, and it is now also the benchmark the legibility
score is tuned against.

Outside this directory, and part of the same system:

```
../public/rtrack/curate.html      the phone curator (served by GitHub Pages)
../public/rtrack/calibrate.html   loupe-based point collection for a new camera
../public/tracks/                 published routes + index.json, read by the app
../rtrack-relay/worker.js         the Cloudflare Worker; see RELAY_LIMITS.md
```

Everything runs as `uv run -m rtrack.<module>`.

## Design notes worth not re-litigating

- **Detect bumpers, not robots.** Bumper geometry is fixed by FRC rules (solid
  alliance color, 5–7.5 in tall, wrapping each corner) while robot silhouettes
  change every season. A bumper model has a real chance of transferring across
  years; a silhouette model has almost none.
- **WPILib field meters are the canonical unit**, not pixels and not feet. It is
  what `AprilTagFieldLayout` speaks, and it survives the field PNG being
  re-rendered at another resolution. Pixels are a render-time concern.
- **`source.py` is load-bearing.** No stage may touch `cv2.VideoCapture` directly.
  The livestream path is then one new source implementation and nothing else.
- **Never silently guess identity.** Every reassignment carries its evidence in the
  output. A track file you cannot audit is worse than one with holes, because the
  holes are honest.
- The video is at most `data/raw/<id>.mp4` and is gitignored; regenerate with
  `rtrack.acquire`. The 11-char YouTube id is the only thing worth committing.
- **Corrections resolve by POSITION, never by track id.** `corrections.resolve` matches a
  label to the nearest box centre in that frame. Track ids move every time anything is cut
  or re-segmented, so anchoring to them would invalidate a curator's work on the next run.
  This is what makes the round trip survivable, and it has been violated three separate
  times in analysis scripts — each producing impossible coverage figures (101%, 154%)
  rather than an error.
- **The curation bundle is segmented from the STITCHED file and gets its guess from the
  LABELLED one.** They are not interchangeable and both are needed: passing only the
  stitched file ships 0% pre-fill while the solver is getting 95% of detections right;
  passing the labelled file to `--tracks` re-segments an already-segmented file, and since
  anchoring is per track, splitting tracks spreads a fixed frame budget thinner and weakens
  the second anchor that catches an identity switch. Hence `--guess`.
- **Measure a proxy before tuning against it.** The legibility score was invented,
  weighted by intuition, and scored 0.602 against human judgement — with its
  highest-weighted feature pointing the *wrong way*. Two of the three things that felt
  obviously right (white in the bumper band, penalising blur) measured worse than chance.
  The curator's own "can't tell" labels were sitting on disk the whole time.
- **A metric computed in a score's own terms proves nothing about that score.** Best views
  was first reported as taking "never readable" from 30% to 0%, measured with the score
  that turned out to be near-chance. The mechanism was sound and the number was circular.
- **`/index` on the relay must never be a KV `list()`.** The free plan caps list at 1000
  calls per DAY, separately from the 100k reads, and `/index` is what everything polls. At
  a 20 s watcher poll that is 4,320 calls a day: the quota burned in about five hours and
  every `/index` after that threw, taking the app's Tracks listing down while every other
  endpoint stayed healthy. Writes maintain a manifest key; reads are ordinary gets.
- **Re-running `reid gallery` on an already-counted match double-counts it.** The merge is
  count-weighted, so a second pass inflates that team's count and makes it artificially
  hard to shift later. `--refs-only` exists for backfilling without touching descriptors.
- **Curation is irreplaceable human work and must survive a reload.** The curator held its
  answers in an in-memory Map; a dev-server reload destroyed a finished match with nothing
  anywhere recording that it had happened. Every answer is now written through to
  localStorage as it is made, and cleared only after the relay confirms the send.

## Prior art

- Roboflow, [Mapping Robot Paths in Robotics Competitions](https://blog.roboflow.com/robot-path-mapping/)
  — source of the bumper-not-robot insight.
- `NimbleValley/auto-scout` — same problem in Node. **Carries no license**: read it
  for approach, do not copy its code.
- Roboflow Universe FRC datasets (check each license before use):
  `greg-zetko-6wpfk/frc-robot-pov` (explicit red/blue bumper classes),
  `worbots-4145/2024-frc`, `frc-08aim/frc-robots-fx5cu`.
