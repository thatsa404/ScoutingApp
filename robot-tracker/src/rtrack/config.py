"""Paths, field constants, and class maps.

Everything downstream imports its paths from here so that folding this subproject
into the main app later is a single-file change.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths

# .../ScoutingApp/robot-tracker/src/rtrack/config.py -> .../robot-tracker
TRACKER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRACKER_ROOT.parent

DATA_DIR = TRACKER_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets"
OUT_DIR = TRACKER_ROOT / "out"
MODELS_DIR = TRACKER_ROOT / "models"
EVAL_DIR = TRACKER_ROOT / "eval"
EVAL_FRAMES_DIR = EVAL_DIR / "frames"
CALIB_DIR = TRACKER_ROOT / "calib"
CFG_DIR = TRACKER_ROOT / "cfg"

STAGE0_DIR = OUT_DIR / "stage0"
STAGE1_DIR = OUT_DIR / "stage1"
STAGE2_DIR = OUT_DIR / "stage2"
STAGE3_DIR = OUT_DIR / "stage3"

# Referenced, never copied -- one source of truth, and the path is already correct
# for the day this folds into the Vite app.
FIELD_PNG = REPO_ROOT / "public" / "field" / "2026-field.png"
ENV_FILE = REPO_ROOT / ".env"

_ALL_DIRS = (
    RAW_DIR, DATASETS_DIR, MODELS_DIR, EVAL_FRAMES_DIR, CALIB_DIR,
    STAGE0_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR,
)


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- field

YEAR = 2026

# Auto length is a GAME RULE, not something to measure. routes.detect_auto_window finds
# where auto STARTS from robot motion, which is event-specific and genuinely has to be
# measured -- but it also reports an end, and that end is unreliable: across four curated
# 2026mawor matches it gave 2.0 s, 2.5 s, 5.0 s and 12.5 s for a period the scoreboard
# counts down from 0:20. It looks for a pause between auto and teleop and settles on one
# too early. So: measure the start, take the duration from the rulebook.
AUTO_S = 20.0

# Auto plus teleop plus the transition between them, measured from the FIRST SUSTAINED
# MOTION rather than from the buzzer -- which is what curate.match_window can actually
# find. Used as a CEILING on the motion-derived window, never as its start.
#
# MEASURED. Per-match motion profiles across all 25 curated 2026necmp1 clips, taking the
# last bin above 3 px that is followed by three quiet bins below 1 px: 14 matches have an
# unambiguous end and 12 of those span 162-164 s, the other two being mid-match lulls
# that the detector mistook for the end. The distribution is not broad -- it is a game
# rule seen through a 2 s bin.
#
# 166, not 164: two seconds of bin granularity above the widest clean observation, on the
# principle that over-running by a bin costs a little post-match noise while under-running
# by a bin cuts the endgame, which is the part of the match worth the most.
#
# DO NOT DERIVE THIS FROM CURATED FRAME SPANS. That was the first attempt and it gave
# 155, because the sampled frames are a subset of the window and stop wherever the last
# useful detection happened to be. It would have cut ~10 s of endgame off every match.
MATCH_SPAN_S = 166.0

GAME = "rebuilt"

# WPILib field coordinate convention, meters. 2026 Rebuilt is 27 ft x 54 ft.
FIELD_LENGTH_M = 16.4592  # 54 ft, +x from blue alliance wall toward red
FIELD_WIDTH_M = 8.2296    # 27 ft
FIELD_SIZE_M = (FIELD_LENGTH_M, FIELD_WIDTH_M)

# Canonical unit everywhere downstream is WPILib field METERS -- not pixels, not
# feet. It is what AprilTagFieldLayout speaks (free interop if the tag experiment
# pans out) and it survives the field PNG being re-rendered at a new resolution.
FIELD_UNITS = "meters"
FIELD_CONVENTION = "wpilib-2026"

# Physical constants used as sanity checks, not as tuning knobs.
ROBOT_MAX_SPEED_MS = 5.5       # FRC drivetrains top out ~4.5-5.5 m/s
# Speed alone is the wrong gate for "could a robot have got from A to B". Over a
# 0.87 s gap a pure speed bound permits 4.8 m, so an 8.2 m teleport is caught -- but
# a 2 m jump by a robot that was standing still is not, and that is the common case.
# Bounding acceleration too asks the real question: given how fast it was ALREADY
# moving, is the displacement reachable? Generous on purpose; a good swerve does
# 0-4.5 m/s in about a second, so 8 m/s^2 will not reject honest motion.
ROBOT_MAX_ACCEL_MS2 = 8.0
KINEMATIC_FLAG_MS = 6.0        # anything above this is a bad projection or an ID switch
BUMPER_BOTTOM_HEIGHT_M = 0.06  # FRC rules put bumper bottom 1-7.5 in above carpet
ROBOT_DEPTH_M = 0.75           # for documenting the near-face depth bias


# ---------------------------------------------------------------- classes

# Bumpers, not whole robots: bumper geometry is fixed by FRC rules (solid alliance
# color, 5-7.5 in tall, wrapping each corner) while robot silhouettes change every
# season. A bumper model has a real chance of transferring across years.
CLASS_NAMES = ["red_bumper", "blue_bumper"]
CLASS_RED, CLASS_BLUE = 0, 1
CLASS_TO_ALLIANCE = {CLASS_RED: "red", CLASS_BLUE: "blue"}

# Match the app's existing palette (BGR for OpenCV, hex for the HTML viewer).
COLOR_RED_HEX = "#ef4444"
COLOR_BLUE_HEX = "#3b82f6"
COLOR_RED_BGR = (68, 68, 239)
COLOR_BLUE_BGR = (246, 130, 59)


# ---------------------------------------------------------------- stage 0 tuning

# Cut detection: an HSV-histogram correlation drop AND a grayscale-thumbnail MAD
# spike must fire together. Requiring both suppresses flashbulbs and score-bug
# animations, which move the histogram but not the structure.
HIST_CORREL_CUT = 0.70
THUMB_MAD_CUT = 28.0

# Camera motion classification, measured on the 480p analysis frame.
STATIC_TRANS_PX = 1.5
STATIC_SCALE_EPS = 0.002

ANALYSIS_HEIGHT = 480  # decode height for Stage 0; full res is never needed here
