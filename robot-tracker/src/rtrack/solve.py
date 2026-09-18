"""Stage 3 -- assign tracks to teams in ONE global constrained solve.

Replaces colour() + name_groups(), and the point is that it replaces BOTH. Those two
stages are each optimal alone and jointly suboptimal: colour() groups tracks without
knowing which team is which, then name_groups() names the groups after they are
already frozen. The bumper votes -- the only direct evidence of identity we have --
therefore cannot influence grouping, which is exactly where the errors were.

MEASURED, on the test match, with greedy colouring: 17 of 44 tracks had exactly ONE
legal group by the time they were placed and 10 had none; only 5 ever saw three or
more options. The assignment was decided almost entirely by the order tracks were
visited, which is why raising the vote-disagreement penalty changed the output not at
all, and why a track carrying 30 of 30 votes for 9644 still landed in a 6329 group.

A global solve has no visit order. Every track's assignment is decided against every
other simultaneously:

    x[t][k] = 1   track t is robot k          (~44 x 6 booleans -- tiny)

    hard  sum_k x[t][k] <= 1                  <= 1, so parking stays possible
    hard  x[a][k] + x[b][k] <= 1              for co-detected a,b: cannot be one robot
    hard  sum_t x[t][k] >= 1                  every team must get something

    max   + VOTE   * votes[t][k]              direct bumper evidence
          - ALLI   * alliance mismatch        strong but SOFT: hue is fallible
          - KIN    * kinematically impossible co-assignment
          - GAP    * unobserved seconds between co-assigned tracks
          - PARK   * tracks left unassigned

Alliance is deliberately soft. It is recovered from bumper hue, which we measured
getting one alliance above three in 4.6% of frames; making it hard would let a single
misread colour veto correct vote evidence. Co-detection, by contrast, is hard: it is
geometry, not inference.

WHY CP-SAT AND NOT A MILP. Not for speed -- this model is trivial either way. CP-SAT
can enumerate NEAR-OPTIMAL solutions, so we can re-solve with the optimum forbidden
and report which tracks change their mind. That set is the honest uncertainty
surface, and it is what a human curation pass should be pointed at: the decisions
where the evidence genuinely does not decide.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from . import config as C

MAX_PAIR_GAP_S = 25.0     # beyond this the kinematic bound stops discriminating
# Worth more than any evidence term but finite: a contradicted curator label should
# win against votes and geometry, and still lose to a hard constraint.
PIN_SOFT = 5000


@dataclass(frozen=True)
class Weights:
    """All costs in 'one confident bumper vote' units, so they stay interpretable:
    alli=250 with vote=10 says contradicting bumper hue costs as much as 25 votes."""
    vote: int = 10
    alli: int = 250
    kin: int = 120
    gap: int = 2
    park: int = 40


def _pair_cost(a: dict, b: dict, w: Weights) -> tuple[int, int]:
    """(kinematic, gap) penalty for putting non-overlapping tracks a,b on one robot.

    Applied to every eligible pair rather than only to consecutive ones. That is
    conservative and correct: a longer gap buys a LARGER distance budget, so an
    intervening track can never make a penalised pair look worse than it is.
    """
    gap = max(0.0, b["t0"] - a["t1"])
    if gap > MAX_PAIR_GAP_S:
        return 0, 0
    kin = 0
    if "end" in a and "start" in b:
        d = float(np.hypot(b["start"][0] - a["end"][0], b["start"][1] - a["end"][1]))
        budget = C.ROBOT_MAX_SPEED_MS * max(gap, 0.1) + 1.0
        if d > budget:
            kin = int(w.kin * min((d - budget) / budget, 3.0))
    return kin, int(w.gap * gap)


def build_and_solve(info: dict, con: dict, ident: dict, red: list[str],
                    blue: list[str], time_limit: float = 30.0,
                    forbid: dict[int, int] | None = None,
                    pinned: dict[int, str] | None = None,
                    w: Weights = Weights(),
                    preferred: dict[int, str] | None = None):
    """Solve once. `forbid` blocks a previous solution; `pinned` fixes track->team."""
    from ortools.sat.python import cp_model

    teams = red + blue
    team_alli = {t: "red" for t in red} | {t: "blue" for t in blue}
    tids = sorted(info)
    m = cp_model.CpModel()
    x = {(t, k): m.NewBoolVar(f"x{t}_{k}") for t in tids for k in range(len(teams))}

    for t in tids:
        m.AddAtMostOne(x[t, k] for k in range(len(teams)))
    for k in range(len(teams)):
        m.AddAtLeastOne(x[t, k] for t in tids)

    # Hard: co-detected tracks are different robots. This is geometry, not inference.
    seen = set()
    for a in tids:
        for b in con.get(a, ()):
            if b <= a or b not in info or (a, b) in seen:
                continue
            seen.add((a, b))
            for k in range(len(teams)):
                m.AddAtMostOne(x[a, k], x[b, k])

    # Human seeds, if any: hard equality. One click should constrain the whole match.
    for t, team in (pinned or {}).items():
        if t in info and team in teams:
            m.Add(x[t, teams.index(team)] == 1)

    terms = []
    # Curator labels that CONTRADICT another curator label (two tracks on screen
    # together, both given the same team) cannot all be hard without making the model
    # infeasible. They enter as a large bonus instead: the solver honours them unless
    # the rest of the evidence makes that impossible, and the report says which lost.
    for t, team in (preferred or {}).items():
        if t in info and team in teams:
            terms.append(PIN_SOFT * x[t, teams.index(team)])
    for t in tids:
        votes = Counter(ident["tracks"].get(str(t), {}).get("tally", {}))
        alli = info[t].get("alliance")
        # Scale the alliance penalty by how much evidence backs this track's hue call.
        # A robot whose bumpers the detector can barely classify -- dark or off-standard
        # colours, measured on 2026mawor team 190 at 30% decided / 75% correct -- should
        # not veto a team as hard as one reading 98%/99%. Tracks with no hue at all are
        # already exempt via `if alli`; this is the graded version of the same idea.
        # alliConf defaults to 1.0 so an older info dict behaves exactly as before.
        conf = info[t].get("alliConf", 1.0)
        pen = int(round(w.alli * conf))
        assigned = []
        for k, team in enumerate(teams):
            v = w.vote * int(votes.get(team, 0))
            if alli and team_alli[team] != alli:
                v -= pen
            terms.append(v * x[t, k])
            assigned.append(x[t, k])
        park = m.NewBoolVar(f"park{t}")
        m.Add(sum(assigned) + park == 1)
        # Scale the parking penalty by track size: dropping a 1000-detection track is
        # far more costly than dropping a 12-detection fragment, and without this the
        # solver happily parks big awkward tracks to keep the objective tidy.
        terms.append(-int(w.park * np.log1p(info[t]["n"])) * park)

    # Pairwise terms. Only for non-overlapping pairs inside the gap horizon, which
    # keeps this in the hundreds rather than the thousands.
    npairs = 0
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            lo, hi = (a, b) if info[a]["t1"] <= info[b]["t0"] else (
                (b, a) if info[b]["t1"] <= info[a]["t0"] else (None, None))
            if lo is None:
                continue
            kin, gap = _pair_cost(info[lo], info[hi], w)
            if kin == 0 and gap == 0:
                continue
            for k in range(len(teams)):
                z = m.NewBoolVar(f"z{lo}_{hi}_{k}")
                m.AddMultiplicationEquality(z, [x[lo, k], x[hi, k]])
                terms.append(-(kin + gap) * z)
                npairs += 1

    if forbid:
        # Forbid an exact previous assignment, so the next solve must differ somewhere.
        lits = [x[t, k] for t, k in forbid.items() if (t, k) in x]
        if lits:
            m.AddBoolOr([lit.Not() for lit in lits])

    m.Maximize(sum(terms))
    solver = cp_model.CpSolver()
    # DETERMINISM IS REQUIRED HERE, not a nicety. With 8 workers on a wall-clock limit
    # this returned DIFFERENT team namings from byte-identical inputs: the same run
    # scored 69.4% and 79.9% against curated truth at identical coverage. That makes
    # every parameter comparison noise (a max-votes sweep showed 72%/77%/77%, entirely
    # inside the run-to-run spread) and means a curator can be asked different
    # questions about the same match.
    #
    # Wall-clock limits are the root cause: whichever worker happens to be ahead when
    # the clock expires supplies the answer. max_deterministic_time counts search
    # progress instead of seconds, and interleave_search makes multi-worker search
    # reproducible, so parallelism is kept.
    solver.parameters.random_seed = 0
    solver.parameters.interleave_search = True
    solver.parameters.num_search_workers = 8
    solver.parameters.max_deterministic_time = time_limit
    # WALL-CLOCK CAP, and it is a real cap rather than a formality. max_deterministic_time
    # counts search WORK, not seconds, so on a hard instance it can run far longer than
    # the same number of seconds would have. rtrack.robots calls solve() up to
    # 1 + --deconflict times and each call solves TWICE (the answer, then the answer
    # forbidden, to expose what is uncertain), so a match can hit this cap eight times.
    # At the 4x it used to be, that was ~16 minutes -- longer than the gap between
    # matches at an event, which is how a backlog starts.
    solver.parameters.max_time_in_seconds = max(30.0, time_limit * 1.5)
    status = solver.Solve(m)
    # If the wall clock stopped the search, the result is NOT reproducible -- that is
    # exactly the nondeterminism the deterministic limit exists to avoid. Say so, rather
    # than returning a number that looks like every other number.
    if solver.WallTime() >= solver.parameters.max_time_in_seconds - 0.5:
        print(f"[solve] WALL-CLOCK LIMIT hit at {solver.WallTime():.0f}s -- this "
              f"solution is NOT deterministic. Lower --deconflict, or raise the cap if "
              f"reproducibility matters more than latency.")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, {"status": solver.StatusName(status)}

    assign = {t: k for t in tids for k in range(len(teams))
              if solver.Value(x[t, k])}
    meta = {"status": solver.StatusName(status),
            "objective": solver.ObjectiveValue(),
            "bound": solver.BestObjectiveBound(),
            "wall": solver.WallTime(), "pairs": npairs,
            "vars": len(x), "parked": [t for t in tids if t not in assign]}
    return assign, teams, meta


def solve(info: dict, con: dict, ident: dict, red: list[str], blue: list[str],
          pinned: dict[int, str] | None = None, time_limit: float = 30.0,
          w: Weights = Weights(), preferred: dict[int, str] | None = None):
    """Solve, then solve again with that answer forbidden to expose what is uncertain.

    The second solve is the reason this module exists in CP-SAT rather than as a MILP.
    A global optimiser returns an equally confident-looking answer whether the evidence
    decides the question or merely breaks a tie, and we have been bitten by exactly
    that kind of false confidence before. Tracks that change between the best and
    second-best solution are the ones where the evidence does NOT decide -- so they
    are what a human should be asked about, and nothing else.
    """
    assign, teams, meta = build_and_solve(info, con, ident, red, blue, time_limit,
                                          pinned=pinned, w=w, preferred=preferred)
    if assign is None:
        return None, None, meta, set()

    alt, _, meta2 = build_and_solve(info, con, ident, red, blue, time_limit,
                                    forbid=assign, pinned=pinned, w=w,
                                    preferred=preferred)
    unstable: set[int] = set()
    if alt is not None:
        unstable = {t for t in assign if alt.get(t) != assign[t]}
        unstable |= {t for t in alt if t not in assign}
        meta["altObjective"] = meta2["objective"]
        meta["altGap"] = meta["objective"] - meta2["objective"]
    return assign, teams, meta, unstable


def groups_from(assign: dict[int, int], n_teams: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(n_teams)]
    for t, k in assign.items():
        groups[k].append(t)
    return [sorted(g) for g in groups]


def pooled_votes(groups: list[list[int]], ident: dict) -> list[Counter]:
    out = []
    for members in groups:
        c: Counter = Counter()
        for m in members:
            c.update(ident["tracks"].get(str(m), {}).get("tally", {}))
        out.append(c)
    return out


def timeline_flags(groups: list[list[int]], ident: dict, window: float = 20.0):
    """Flag a robot whose pooled votes disagree across time -- a likely chimera."""
    out = {}
    for gi, members in enumerate(groups):
        wins = defaultdict(Counter)
        for m in members:
            for t, team, n in ident["tracks"].get(str(m), {}).get("timeline", []):
                wins[int(t // window)][team] += n
        seq = [c.most_common(1)[0][0] for _, c in sorted(wins.items())
               if sum(c.values()) >= 3]
        out[gi] = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
    return out
