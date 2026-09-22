# robot-tracker architecture notes

Reviewed against the current source tree and `README.md`, `ROUTE_QUALITY.md`,
`CURATION_PLAN.md`, `OCR_PLAN.md`, and `RELAY_LIMITS.md` on 2026-09-21.

## Scope and product boundary

`robot-tracker` is a self-contained Python 3.12/`uv` project. The Vite app does not
import it. Its product is `public/tracks/<matchKey>.json`, a schema-v1 document with
six team-keyed robot routes in field metres. The app consumes that document through
the local static file and/or the Cloudflare relay, then draws it in `main.js`.

The system is human-in-the-loop. Automatic identity is useful for pre-filling and
reducing questions, but curator corrections are the ground truth and are deliberately
anchored to `(video frame, box centre)`, not tracker IDs.

## End-to-end data flow

```text
video/archive
  -> acquire/replay/source + scoreboard/TBA match identity
  -> track              out/stage1/<stem>_tracks.jsonl       (15 Hz detections)
  -> stitch             out/stage1/<stem>_tracks_stitched.jsonl
  -> appear             out/stage3/<stem>_appearance*.npz
  -> reid votes         out/stage3/<stem>_reid*.json          (optional gallery evidence)
  -> pre-project        out/stage2/<stem>_prepos.json         (metres for constraints)
  -> robots/solve       out/stage3/<stem>_labeled.jsonl
  -> curate/corrections robot-tracker/corrections/<match>_corrections.json
  -> robots/solve again out/stage3/<stem>_labeled.jsonl + <stem>_robots.json
  -> viewcheck/project  out/stage2/<stem>_positions.json
  -> export             out/stage3/<match>.json
  -> publish/relay      public/tracks/<match>.json + index.json
  -> ScoutingApp        match modal and team Routes tab
```

`rtrack.pipeline` is the authoritative orchestration: `track -> stitch -> appear ->
votes -> prepos -> robots -> curate -> resolve -> viewcheck -> project -> export ->
gallery`. It is resumable and locks one event at a time because the appearance gallery
for match N changes the identity evidence for match N+1.

## Capability map

| Area | Main modules | What it provides |
|---|---|---|
| Source and match identity | `acquire`, `replay`, `source`, `scoreboard`, `tba` | Download/slice video, probe metadata, identify the broadcast match, retrieve six TBA teams |
| Stage 0 broadcast checks | `shots`, `viewcheck` | Detect cuts/camera pose changes and determine whether a calibrated view remains valid |
| Detection/tracking | `track`, `alliance`, `botpatch`, `overlay` | YOLO/Ultralytics + BoT-SORT boxes, alliance hue, proof-gate overlays; source-rate-aware sampling |
| Track repair | `stitch`, `robots` | Merge plausible tracker fragments, split alliance/appearance/chimeric/impossible steps, reject off-view/off-field detections |
| Calibration/geometry | `calibrate`, `autocal`, `lens`, `project`, `routes`, `occluders`, `occlude` | Manual homography and lens model, image-to-field metres, visibility/occluder overlays and diagnostic route plots |
| Identity | `appear`, `embed`, `reid`, `solve`, `robots` | Cached descriptors, per-event gallery, CNN whitening head, appearance votes, global CP-SAT assignment to six TBA teams |
| Human correction | `curate`, `corrections`, `repair_notrobot`, `chicklets`, `viewer/curate.html` | Full-frame or per-track bundles, anchored labels, forced cuts, not-robot/mixed/unknown decisions |
| Delivery | `export`, `relay`, `watch`, `pipeline` | Schema-v1 routes, manifest, relay transport, unattended re-solve/publish loop |
| App/viewers | `viewer/replay.html`, `viewer/curate.html`, root `main.js` | Scrub raw route exports, curate on desktop/phone, render routes in match/team views |

## Important data and identity boundaries

- A raw tracker `tid` is not stable. `stitch`, alliance/appearance splitting, chimeric
  splitting, duplicate merging, and deconfliction all create or renumber IDs.
- `corrections.py` resolves labels by nearest box at `(f, xy)`; a label more than 90 px
  away is unresolved rather than silently ignored. Two different team labels on one
  track mean “cut this track between them.”
- `project.py` writes one position per detection. `robots.positions_by_det()` rekeys
  those positions by `(frame, box)` so geometry survives later track renumbering.
- `prepos.json` is intentionally separate from canonical `positions.json`: the solver
  needs positions before assignment, while export/custody need positions from the final
  labelled tracks.
- `public/tracks` routes are keyed by TBA team number, not internal track ID. Export
  groups and samples by team, drops `offfield`/`viewmoved` samples, breaks long gaps,
  and samples at 5 Hz while the tracker remains at 15 Hz.

## Geometry and calibration assumptions

- A calibration belongs to a camera/event stem, not necessarily to the sliced video
  stem. For sliced clips, pass `--calib-from 2026mawor` or `--calib-from 2026necmp1`.
- `project_points()` undistorts first when a lens model exists, then applies the video
  pixel -> field-pixel homography and field reference. The exported position is the
  near-face floor contact, not the robot centre; heading/depth correction is not done.
- `drop_offfield()` is an early filter used by both `robots.py` and curation. Its 0.6 m
  slack is in field metres, not pixels. `viewcheck` can remove detections from foreign
  camera spans before identity grouping.
- Visibility polygons and occluder polygons are exported as explanatory overlays. A
  visibility gap means the camera could not see the field; an occluder shadow means a
  robot could be hidden behind a structure. They are not interchangeable.

## Solver model

`solve.py` assigns each surviving track to at most one of six match teams, requires each
team to receive something, hard-forbids co-detected tracks from sharing a team, and
allows parking. Votes, alliance mismatch, pairwise kinematic/gap costs, occlusion holds,
and optional appearance terms are objective terms. Curator pins are hard; contradictory
pins are demoted to preferences by `corrections.split_conflicts()`.

Hard geometry is intentionally limited: `pair_forbidden()`/`paths_forbidden()` can rule
out a physically impossible pairing, but a same-team curator pair is allowed to expose
the contradiction instead of making the run infeasible. The solve is repeated with the
best assignment forbidden to report unstable tracks—the real uncertainty surface.

## Operational invariants

1. Keep processed time consistent. `track --sample-hz 15` derives stride from source FPS;
   `--stride` is only for reproducing old runs. Downstream window sizes count processed
   frames, so a 58 FPS source must not be treated as a 30 FPS source.
2. Use the same preprocessing/segmentation inputs for curation and solving. Curation
   consumes the stitched file and applies the same filters/splits as `robots.py`; never
   feed `labeled.jsonl` back as the `--tracks` source.
3. Project stitched tracks before solving when kinematic or duplicate geometry is being
   used. Project labelled tracks again after the final solve before export.
4. Treat missing constraints as a diagnostic failure, not as evidence that no issue
   exists. Logs should say whether calibration, positions, viewcheck, and occluders were
   actually active.
5. Compare route quality with a match-relative motion window. Whole-clip custody mixes
   staging/post-match dead time into the denominator and is not comparable.
6. Render suspicious frames. Statistics locate candidates; frames distinguish detector
   false positives, camera geometry errors, and identity fusions.

## Useful command surface

```powershell
cd robot-tracker
uv sync
uv run -m rtrack.replay 2026mawor --match qm9
uv run -m rtrack.pipeline 2026mawor_qm9 --match 2026mawor_qm9 --calib-from 2026mawor

# standalone debugging
uv run -m rtrack.track <stem> --model <weights> --sample-hz 15 --square --conf 0.20
uv run -m rtrack.stitch out/stage1/<stem>_tracks.jsonl
uv run -m rtrack.project <stem> --tracks out/stage1/<stem>_tracks_stitched.jsonl --calib-from <camera>
uv run -m rtrack.routes <stem> --auto --panels
```

For route defects, the reproducible solve/export command and A/B switches are captured
in [`ROUTE_QUALITY.md`](ROUTE_QUALITY.md); that document is the investigation authority.

## Current worktree caution

The worktree is intentionally dirty. At review time, uncommitted changes include
`src/rtrack/robots.py`, `calib/2026mawor.json`, `public/tracks/2026mawor_qm10.json`,
`public/tracks/index.json`, a new MAWOR occluder file, and several correction files.
The current `qm10` export reports 52 kinematic violations and the working calibration
contains a 7.936 m maximum reprojection error outlier. Do not use the current published
MAWOR route or calibration as a clean regression baseline; preserve these changes and
re-derive comparisons from explicitly selected inputs.
