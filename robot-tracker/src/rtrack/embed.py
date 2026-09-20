"""Stage 3 -- learned appearance embedding, the alternative to appear.descriptor.

    uv run -m rtrack.appear <video> --tracks ... --backend both
    uv run -m rtrack.reid fit-head --event 2026necmp1
    uv run -m rtrack.reid votes <video> --event ... --match ... --backend cnn

WHY THIS EXISTS. The hand-crafted descriptor separates robots of the same alliance at
AUC 0.667 within a match and 0.604 across matches, and no amount of averaging moves it:
one crop scores 0.613, the mean of a whole track 0.666. That flatness is the diagnostic
-- the descriptor is not NOISY about identity, it is CONSISTENTLY measuring something
that is not identity, and you cannot average that away.

Measured over 63,606 curator-labelled crops (1,644 tracks, 75 teams, 20 matches), every
untrained representation lands in the same band, so this is not a tuning problem:

    representation                      within-match AUC   cross-match AUC
    current 3-band gray16 (48d)               0.667             0.604
    raw gray 4x4 .. 32x32                 0.632 -> 0.654    0.628 -> 0.647
    raw bgr 8x8 .. 24x24                      0.692             0.675
    HOG 2x2 / 4x4 / 6x6                       0.578             0.565
    resnet18 ImageNet, zero-shot              0.725             0.685
    resnet18 + whitening head                   --              0.862

Note the grayscale ladder saturates at 8x8: going from 64 pixels to 1024 buys 0.004.
Resolution was never the limit, which is why cutting the crop to 40x40 costs so little.
The limit is what is computed from the pixels.

The whitening head is the whole gain and it is a LINEAR map -- PCA to 256 dims, then
whitening by the within-team covariance (WCCN). Folds in that measurement were over
TEAMS, not crops and not matches, so every scored team was unseen when the head was fit.
That is the deployment condition: next season's robots are all unseen.

PREQUENTIAL, SCORED PER CURATED LABEL. Match N is voted on by a gallery and head built
only from matches 1..N-1, which is how an event actually unfolds, and every human label
is scored individually:

    backend                     correct    within-alliance
    chance                        16.7%         33.3%
    gray band histogram           33.7%         50.4%
    resnet18, no head             61.7%         68.1%
    resnet18 + whitening head     71.1%         75.2%

Within-alliance is the number that matters: 95-100% of real identity errors are
same-alliance swaps.

SCORE PER LABEL, NOT PER TRACK. 21% of curator-labelled stitched tracks carry more than
one team, so collapsing a track to its majority label mislabels part of it, and a purity
gate on the test set DELETES the hardest 23% of the curated evidence. The same
comparison scored per track reads 35.8 / 68.7 / 81.1 correct and 52.6 / 75.6 / 84.6
within-alliance -- inflated most for the strongest configuration, which gains most from
having the chimeras removed. Chimeras are still excluded from FITTING, where a mean
embedding blended from two robots genuinely does poison the within-team covariance;
filtering the training side is legitimate, filtering the test side is not.

The AUC tables above are likewise track-level and carry the same optimism. They are kept
because they compare representations against each other on identical data, which is what
they were for, but do not read them as absolute capability.

WHAT THIS DOES NOT REPLACE. robots.split_on_appearance still uses the histogram, whose
Hellinger distance scale APPEAR_THRESH is tuned against. Change-point detection and
re-identification are different jobs and the histogram is adequate at the first one, so
both descriptors are computed and neither backend is removed.

CACHING. The npz stores RAW embeddings, not whitened ones, so the head can be refit
from new curation without re-decoding any video. Whitening is applied at gallery and
vote time instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C

EMBED_SIDE = 112      # >= the p95 superstructure band (100 px), so the resize is not
                      # itself a bottleneck. 224 measured no better (0.671 vs 0.685).
BATCH = 256
NPC = 256             # PCA dims before whitening; 32/64/128 all scored lower
HEAD_RIDGE = 1e-3     # on the within-team covariance, or inversion is unstable
MIN_TEAM_TRACKS = 2   # a team with one track contributes no within-team scatter
MIN_HEAD_TRACKS = 250
"""Labelled tracks required before a head is worth fitting at all.

Below this the head is actively HARMFUL and raw embeddings win. Measured prequentially
over 20 curated 2026necmp1 matches -- each match scored with a gallery and head built
only from the matches before it, which is what happens at an event:

    head gate    correct   within-alliance
    none (raw)    61.7%        68.1%
    60-150        57.1%        65.5%
    200           59.9%        69.0%
    250           71.1%        75.2%
    300           66.1%        71.8%
    350           63.2%        69.0%

Per match the reason is stark: with the gate at 60 the head scores 29%, 17%, 7%, 12%
on qm9-qm12 and 83-100% from qm14 on. A within-team covariance estimated from a
hundred tracks of a dozen teams is mostly noise, and whitening by it amplifies exactly
the directions that noise occupies.

250 tracks is roughly 10-12 curated matches. Treat it as an order of magnitude, not a
precise constant -- it is tuned on one event. It does survive the change from track-level
to per-label scoring, which moved every other number here, so it is not an artefact of
the metric.
"""

_MODEL = None


def _model():
    """resnet18 penultimate features. Loaded once; ~11M params, milliseconds a batch."""
    global _MODEL
    if _MODEL is None:
        import torch
        import torchvision.models as tvm
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = (m.eval().to(dev), dev)
    return _MODEL


def embed(crops: list[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops of any size -> (n, 512) float32 embeddings."""
    import cv2
    import torch
    from torchvision import transforms as T
    if not crops:
        return np.zeros((0, 512), np.float32)
    model, dev = _model()
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    arr = np.array([cv2.resize(c, (EMBED_SIDE, EMBED_SIDE),
                               interpolation=cv2.INTER_AREA)[..., ::-1]
                    for c in crops], np.uint8)
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), BATCH):
            b = torch.from_numpy(arr[i:i + BATCH].copy())
            b = b.permute(0, 3, 1, 2).float().div_(255).to(dev)
            out.append(model(norm(b)).flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- the head


def head_path(event: str) -> Path:
    return C.STAGE3_DIR / f"{event}_head.npz"


def fit_head(V: np.ndarray, team: np.ndarray, npc: int = NPC) -> dict:
    """PCA-whiten, then whiten again by WITHIN-TEAM covariance.

    The second step is the one that matters. PCA alone rewards the directions with the
    most total variance, and on these crops that is lighting, pose and field position --
    exactly the consistent non-identity signal that made averaging useless. Dividing by
    the within-team scatter suppresses whatever varies WITHIN one robot's own crops and
    leaves what varies between robots.
    """
    mu = V.mean(0)
    X = V - mu
    _U, s, Vt = np.linalg.svd(X, full_matrices=False)
    npc = int(min(npc, (s > 1e-8).sum()))
    P = Vt[:npc].T / (s[:npc] / np.sqrt(len(X)) + 1e-6)
    Z = X @ P
    Sw = np.zeros((npc, npc))
    n = 0
    for t in set(team.tolist()):
        m = team == t
        if m.sum() < MIN_TEAM_TRACKS:
            continue
        Cc = Z[m] - Z[m].mean(0)
        Sw += Cc.T @ Cc
        n += int(m.sum()) - 1
    Sw = Sw / max(n, 1) + HEAD_RIDGE * np.eye(npc)
    w, Q = np.linalg.eigh(Sw)
    W = Q @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-6, None))) @ Q.T
    return {"mu": mu.astype(np.float32), "M": (P @ W).astype(np.float32),
            "npc": npc, "teams": int(len(set(team.tolist()))), "tracks": int(len(V))}


def save_head(event: str, h: dict) -> Path:
    p = head_path(event)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, mu=h["mu"], M=h["M"],
                        meta=np.array(json.dumps(
                            {k: h[k] for k in ("npc", "teams", "tracks")})))
    return p


def load_head(event: str):
    """-> callable applying the head, or None if this event has no head yet.

    None is NORMAL and must not be an error: an event's first matches are curated
    before any head exists, and reid falls back to raw embeddings (0.685 cross-match
    AUC against 0.862) rather than refusing to vote.
    """
    p = head_path(event)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    mu, M = z["mu"], z["M"]
    return lambda A: (np.asarray(A, np.float32) - mu) @ M
