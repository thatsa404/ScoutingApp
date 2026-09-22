"""Candidate graph for joint segment-continuity and team-assignment experiments.

Nodes are the immutable post-segmentation track ids.  This module only proposes local
edges; :mod:`rtrack.solve` chooses them jointly with team assignment.  Keeping proposal
generation separate from optimisation makes every rejected or selected handoff
inspectable in the run manifest.
"""

from __future__ import annotations

import math

from .solve import distance_budget


def candidates(info: dict, conflicts: dict[int, set[int]], *, max_gap_s: float = 3.0,
               reach_mult: float = 1.25, max_per_successor: int = 3,
               weight: int = 0) -> list[dict]:
    """Return plausible directed continuation edges between adjacent segments.

    An edge needs ordered, non-overlapping segments, compatible alliance evidence and
    calibrated field endpoints.  It is intentionally stricter than a same-team
    assignment: absent an edge, export reports a real route gap instead of silently
    joining two unrelated fragments.
    """
    if max_gap_s <= 0 or max_per_successor <= 0:
        return []
    out = []
    tids = sorted(info)
    for b in tids:
        ib = info[b]
        if "start" not in ib:
            continue
        choices = []
        for a in tids:
            if a == b or b in conflicts.get(a, ()):
                continue
            ia = info[a]
            if "end" not in ia or ia["t1"] > ib["t0"]:
                continue
            gap = float(ib["t0"] - ia["t1"])
            if gap > max_gap_s:
                continue
            aa, ab = ia.get("alliance"), ib.get("alliance")
            if aa and ab and aa != ab:
                continue
            dist = math.hypot(ib["start"][0] - ia["end"][0],
                              ib["start"][1] - ia["end"][1])
            budget = distance_budget(gap)
            if dist > reach_mult * budget:
                continue
            # Prefer short, physically easy handoffs.  The solver still decides whether
            # this evidence outweighs votes, pins and other candidate edges.
            score = gap / max(max_gap_s, 1e-6) + dist / max(budget, 1e-6)
            choices.append((score, a, gap, dist, budget))
        for _score, a, gap, dist, budget in sorted(choices)[:max_per_successor]:
            out.append({"a": a, "b": b, "kind": "local-continuity",
                        "gapS": round(gap, 3), "distanceM": round(dist, 3),
                        "budgetM": round(budget, 3), "weight": int(weight)})
    return out
