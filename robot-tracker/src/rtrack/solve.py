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

APP_SAME = 0.53
APP_DIFF = 0.89
"""Whitened-embedding cosine distance either side of which two tracks are the same
robot or different ones. Calibrated over 20 curated 2026necmp1 matches: same team
p90 = 0.53, different team p10 = 0.89, with NOTHING in between. Distances landing in
the gap get no term rather than a guess.

WHY THE OBJECTIVE NEEDS THIS AT ALL. The solver had no appearance term whatever -- it
saw votes, bumper hue, kinematics and gaps, but never "do these two look like the same
robot". Measured on 2026necmp1_qm24: stitch joined two different robots across a 3.73 s
gap (678 px, legal by its rules and blind to appearance since it runs before
rtrack.appear), split_on_appearance correctly cut the chimera back apart, and then the
solver reassembled it by giving both halves team 195. Nothing objected: 4.8 m in 3.7 s
is a slow drive, so kinematics was satisfied. The embeddings either side of that join
sit 0.964 apart -- as different as two robots get. 9 of 23 stitched tracks on that
match contain a jump that large.
"""

MAX_PAIR_GAP_S = 25.0     # beyond this the kinematic bound stops discriminating
KIN_CAP = 3.0             # multiples of the distance budget the penalty keeps scaling
                          # over; see _pair_cost. Raise to make gross teleports cost
                          # more than the vote evidence that buys them.
# Worth more than any evidence term but finite: a contradicted curator label should
# win against votes and geometry, and still lose to a hard constraint.
PIN_SOFT = 5000


@dataclass(frozen=True)
class Weights:
    """All costs in 'one confident bumper vote' units, so they stay interpretable:
    alli=250 with vote=10 says contradicting bumper hue costs as much as 25 votes."""
    vote: int = 10
    hold: int = 0      # occlusion hold; see robots.hold_cells
    app: int = 0       # appearance agreement between adjacent tracks; see APP_SAME
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
            # CAP WAS 3.0, i.e. 360 points for ANY violation however gross -- a 6 m/s
            # overshoot and a 130 m/s teleport cost exactly the same. Against vote
            # evidence worth up to MAX_VOTES * w.vote (4800 at w.vote=200) that can
            # never bind: measured on qm24, turning the kinematic term ON changed the
            # impossible-transition count from 23 to 24. The penalty must scale with
            # how impossible the jump is, so the cap is now a parameter and defaults
            # high enough to matter.
            kin = int(w.kin * min((d - budget) / budget, KIN_CAP))
    return kin, int(w.gap * gap)


def pair_forbidden(a: dict, b: dict, mult: float) -> bool:
    """True when a,b are too far apart in space to be one robot, at ANY price.

    Same reasoning as the co-detection constraint this sits beside: one robot cannot be
    in two places, and that is geometry rather than evidence to be weighed. A soft
    penalty says 'expensive'; past some multiple of the physical budget the right word
    is 'impossible'. mult <= 0 disables, preserving the old all-soft behaviour.
    """
    if mult <= 0 or "end" not in a or "start" not in b:
        return False
    gap = max(0.0, b["t0"] - a["t1"])
    if gap > MAX_PAIR_GAP_S:
        return False
    d = float(np.hypot(b["start"][0] - a["end"][0], b["start"][1] - a["end"][1]))
    return d > mult * (C.ROBOT_MAX_SPEED_MS * max(gap, 0.1) + 1.0)


def build_and_solve(info: dict, con: dict, ident: dict, red: list[str],
                    blue: list[str], time_limit: float = 30.0, *, opt_gap: float = 0.0,
                    hint: bool = True, seed: int = 0, kin_hard: float = 0.0,
                    holds: tuple = ((), ()), emb: dict | None = None,
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

    # Pairwise terms. Only for non-overlapping pairs inside the gap horizon. Measured
    # on 2026necmp1_qm21: 3704 such pairs out of 14878 possible, so the horizon is
    # doing real work -- but that is thousands, not the "hundreds" an earlier version
    # of this comment claimed.
    #
    # NOTE npairs counts (pair, team) COMBINATIONS, not pairs. The 22224 it reports on
    # qm21 is 3704 pairs x 6 teams, which is worth knowing before reading it as a
    # model-size problem.
    # Appearance applies ONLY between temporally ADJACENT tracks -- the single nearest
    # predecessor of each track. Applied to every pair inside the gap horizon it
    # ACCUMULATES: a fragment with a dozen dissimilar neighbours collects a dozen
    # penalties until parking it is cheaper than placing it, which is what happened --
    # at weight 600, 50 tracks parked and custody fell 77% -> 53%. Only the adjacent
    # pair is evidence about continuation anyway; the rest are unrelated robots that
    # happen to be nearby in time, and of course they look different.
    nearest_prev = {}
    for b in tids:
        best = None
        for a in tids:
            if a == b or info[a]["t1"] >= info[b]["t0"]:
                continue
            gap = info[b]["t0"] - info[a]["t1"]
            if best is None or gap < best[0]:
                best = (gap, a)
        if best is not None:
            nearest_prev[b] = best[1]

    npairs = 0
    n_forbid = 0
    n_pin_override = 0
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            lo, hi = (a, b) if info[a]["t1"] <= info[b]["t0"] else (
                (b, a) if info[b]["t1"] <= info[a]["t0"] else (None, None))
            if lo is None:
                continue
            if pair_forbidden(info[lo], info[hi], kin_hard):
                # THE CURATOR OUTRANKS THE GEOMETRY. A hard pin and a hard kinematic
                # exclusion are both absolute, so when they disagree the model is
                # INFEASIBLE and the run dies -- measured on 2026necmp1_qm21, which has
                # 65 pinned tracks and 60 forbidden pairs. A human who looked at both
                # crops and called them one robot is better evidence than a distance
                # bound built from box centres, and in any case the answer to a
                # contradiction is never "refuse to produce anything".
                pl, ph = (pinned or {}).get(lo), (pinned or {}).get(hi)
                if pl is not None and pl == ph:
                    n_pin_override += 1
                    continue
                for k in range(len(teams)):
                    if teams[k] == pl or teams[k] == ph:
                        continue      # that team is pinned here; leave it alone
                    m.AddAtMostOne(x[lo, k], x[hi, k])
                n_forbid += 1
                continue
            kin, gap = _pair_cost(info[lo], info[hi], w)
            # Appearance, on the same pairs the kinematic term already enumerates.
            # Positive = these look like one robot and should share a team; negative =
            # they look like two and should not.
            app = 0
            if (w.app and emb is not None and lo in emb and hi in emb
                    and nearest_prev.get(hi) == lo):
                d = 1.0 - float(emb[lo] @ emb[hi])
                if d <= APP_SAME:
                    app = int(w.app * (1.0 - d / APP_SAME))
                elif d >= APP_DIFF:
                    app = -int(w.app * min((d - APP_DIFF) / (1.0 - APP_DIFF) + 1.0, 2.0))
            if kin == 0 and gap == 0 and app == 0:
                continue
            for k in range(len(teams)):
                z = m.NewBoolVar(f"z{lo}_{hi}_{k}")
                # z = x[lo,k] AND x[hi,k], but only the FORCING direction is needed.
                # The term enters as -(kin+gap)*z under Maximize with kin+gap > 0, so
                # the solver already wants z = 0 and never needs to be stopped from
                # setting it; it only needs to be prevented from escaping the penalty
                # when both are 1. One linear constraint, identical optimum.
                #
                # AddMultiplicationEquality was building a general integer-product
                # constraint 22k times for what is a boolean AND. Products are far
                # heavier for presolve and propagation than clauses, and these
                # instances were failing to find ANY solution inside the deterministic
                # budget -- see robots.run_solve's retry.
                m.Add(z >= x[lo, k] + x[hi, k] - 1)
                if app > 0:
                    # A bonus has to be FORCED down when the pair splits, or the solver
                    # would claim it for free; the penalty direction needs no such care.
                    m.Add(z <= x[lo, k])
                    m.Add(z <= x[hi, k])
                terms.append((app - kin - gap) * z)
                npairs += 1

    # OCCLUSION HOLDS. A team that vanished into a structure is behind it, so it is
    # not simultaneously a robot somewhere else -- the co-detection rule extended
    # through the hidden interval. Priced rather than forbidden: the premise (that the
    # robot went behind the structure at all) can be wrong, and two hard constraints on
    # false premises have already cost this pipeline dearly.
    hx, he = holds
    if w.hold and (hx or he):
        for a, c in hx:
            if a not in info or c not in info:
                continue
            for k in range(len(teams)):
                z = m.NewBoolVar(f"h{a}_{c}_{k}")
                m.Add(z >= x[a, k] + x[c, k] - 1)
                terms.append(-w.hold * z)
        for a, c in he:
            if a not in info or c not in info:
                continue
            for k in range(len(teams)):
                z = m.NewBoolVar(f"e{a}_{c}_{k}")
                m.Add(z <= x[a, k])
                m.Add(z <= x[c, k])
                terms.append((w.hold // 2) * z)

    if forbid:
        # Forbid an exact previous assignment, so the next solve must differ somewhere.
        lits = [x[t, k] for t, k in forbid.items() if (t, k) in x]
        if lits:
            m.AddBoolOr([lit.Not() for lit in lits])

    if n_forbid:
        print(f"[solve] {n_forbid} pair(s) forbidden as kinematically impossible "
              f"(> {kin_hard}x the distance budget)")
    if n_pin_override:
        print(f"[solve] {n_pin_override} kinematically impossible pair(s) ALLOWED "
              f"because the curator pinned both to the same team -- the human wins, "
              f"but these are worth a look: a robot cannot be in two places")
    m.Maximize(sum(terms))
    # SOLUTION HINT. Votes are pure objective terms -- they change no constraint, so
    # they cannot make the model harder to SATISFY. What they do is send the search
    # down a far more expensive path before it reaches any first solution, and on
    # vote-carrying instances the first solve was returning UNKNOWN every time: no
    # feasible answer at all inside the deterministic budget. That triggered the 4x
    # retry in robots.run_solve, whose 90 s wall cap then terminated the search, and a
    # wall-clock stop is exactly the nondeterminism max_deterministic_time exists to
    # prevent. Measured on qm21: identical inputs gave 58%, 60% and 77% depending on
    # machine load.
    #
    # The evidence for a good first solution is already in hand -- the per-track vote
    # tally -- so hand it over instead of making the solver rediscover it. A hint only
    # seeds the search: it changes neither the feasible set nor the optimum, and CP-SAT
    # ignores it where it conflicts with a constraint.
    if hint:
        n_hint = 0
        for t in tids:
            tally = ident["tracks"].get(str(t), {}).get("tally", {})
            want = (preferred or {}).get(t) or (pinned or {}).get(t)
            if not want and tally:
                want = max(tally, key=tally.get)
            if want in teams:
                kk = teams.index(want)
                for k in range(len(teams)):
                    m.add_hint(x[t, k], 1 if k == kk else 0)
                n_hint += 1
        if n_hint:
            print(f"[solve] hinted {n_hint}/{len(tids)} track(s) from vote tallies")

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
    solver.parameters.random_seed = seed
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
    # OPTIMALITY GAP. Unlike the wall clock this does NOT cost determinism: it stops on
    # a property of the objective (proven within `opt_gap` of optimal), not on whichever
    # worker happened to be ahead when a timer expired, so the same input still gives
    # the same answer.
    #
    # It exists because identity votes made these instances far harder to PROVE than to
    # solve. The objective is denominated in "one confident bumper vote" units; without
    # votes it is highly degenerate and many assignments tie, so an optimum is found and
    # proved quickly. Votes give nearly every (track, team) pair a distinct cost, the
    # ties vanish, and the proof gets expensive while the answer does not get better.
    if opt_gap > 0:
        solver.parameters.relative_gap_limit = opt_gap
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

    # OBJECTIVE COMPOSITION. Reported because a seed sweep on qm21 found accuracy
    # uncorrelated with objective (r = -0.031 over 8 solutions; accuracy 52-66% while
    # the objective moved 4.9%). An objective that does not discriminate a correct
    # labelling cannot be optimised into one, so the first question about any solution
    # is where its mass actually sits -- not how close to optimal it is.
    comp = {"vote": 0, "alliance": 0, "park": 0, "pair": 0, "pin": 0}
    for t, team in (preferred or {}).items():
        if t in info and team in teams and assign.get(t) == teams.index(team):
            comp["pin"] += PIN_SOFT
    for t in tids:
        k = assign.get(t)
        if k is None:
            comp["park"] -= int(w.park * np.log1p(info[t]["n"]))
            continue
        team = teams[k]
        votes = Counter(ident["tracks"].get(str(t), {}).get("tally", {}))
        comp["vote"] += w.vote * int(votes.get(team, 0))
        alli = info[t].get("alliance")
        if alli and team_alli[team] != alli:
            comp["alliance"] -= int(round(w.alli * info[t].get("alliConf", 1.0)))
    comp["pair"] = int(round(solver.ObjectiveValue())) - sum(comp.values())
    tot = sum(abs(v) for v in comp.values()) or 1
    print("[solve] objective composition (share of total magnitude):")
    for kname in ("vote", "alliance", "park", "pair", "pin"):
        print(f"          {kname:<9}{comp[kname]:>9}  {100*abs(comp[kname])/tot:>5.1f}%")
    meta = {"status": solver.StatusName(status),
            "objective": solver.ObjectiveValue(),
            "bound": solver.BestObjectiveBound(),
            "wall": solver.WallTime(), "pairs": npairs,
            "vars": len(x), "parked": [t for t in tids if t not in assign]}
    return assign, teams, meta


def solve(info: dict, con: dict, ident: dict, red: list[str], blue: list[str],
          pinned: dict[int, str] | None = None, time_limit: float = 30.0,
          w: Weights = Weights(), preferred: dict[int, str] | None = None,
          *, opt_gap: float = 0.0, hint: bool = True, seed: int = 0,
          kin_hard: float = 0.0, holds: tuple = ((), ()), emb: dict | None = None):
    """Solve, then solve again with that answer forbidden to expose what is uncertain.

    The second solve is the reason this module exists in CP-SAT rather than as a MILP.
    A global optimiser returns an equally confident-looking answer whether the evidence
    decides the question or merely breaks a tie, and we have been bitten by exactly
    that kind of false confidence before. Tracks that change between the best and
    second-best solution are the ones where the evidence does NOT decide -- so they
    are what a human should be asked about, and nothing else.
    """
    assign, teams, meta = build_and_solve(info, con, ident, red, blue, time_limit,
                                          pinned=pinned, w=w, preferred=preferred,
                                          opt_gap=opt_gap, hint=hint, seed=seed,
                                          kin_hard=kin_hard, holds=holds, emb=emb)
    if assign is None:
        return None, None, meta, set()

    alt, _, meta2 = build_and_solve(info, con, ident, red, blue, time_limit,
                                    forbid=assign, pinned=pinned, w=w,
                                    preferred=preferred, opt_gap=opt_gap,
                                    hint=hint, seed=seed, kin_hard=kin_hard, holds=holds, emb=emb)
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
