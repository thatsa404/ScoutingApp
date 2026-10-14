// 2026 FRC Game: REBUILT
// Role-based per-shift tracking (Scorer/Shuttler/Defender/Died × 4 shifts).
// TBA fusion allocates per-shift fuel via fuseCustom, weighted by Scoring Effectiveness.
//
// TBA breakdown fuel lives in b.hubScore.{autoCount,shift1Count…shift4Count}.
// Inactive shifts already show 0 in TBA — no active-shift detection needed.
// Tower robots: b.autoTowerRobot{1|2|3}, b.endGameTowerRobot{1|2|3} (strings: "None"|"Level1"|"Level2"|"Level3")
//
// Sheet prerequisite: delete group-label header row, rename duplicate columns per CLAUDE.md plan.

// ── Galileo (2026gal) adapter ─────────────────────────────────────────────────
// Galileo's match scouting comes from another team's system: one wide row per robot per
// match (359 columns), letter-coded ratings and per-period role flags rather than our
// sheet's per-shift checkboxes. This rewrites each row into our column names so the rest
// of the pipeline (processRow, aggregation, TBA fusion) runs unchanged.
//
// How the fields line up, checked against the sheet's own values:
//   periods      Period1..4 -> our shifts TS1, S2, S3, S4 (their Transition is separate
//                and almost never used: one value across all 746 rows)
//   roles        a robot can hold several per period. Scorer = Alliance or Neutral;
//                Shuttler = Passing; Defender = Defending; Died = NonFunctioning.
//                Measured against their own per-period fuel estimates (2,984 robot-
//                periods): Neutral-only periods are credited ~21 fuel, so Neutral means
//                scoring from the neutral zone, not shuttling; Passing, Defending and
//                Stealing are never credited fuel. Hybrid is set exactly when Alliance
//                and Neutral both are (374/374), so it adds nothing. Stealing is left out
//                on purpose -- a tangential role, mostly alongside Passing or Defending.
//   auto fuel    CalculatedAuto -- a per-robot estimate; an alliance's robots sum to its
//                count (e.g. 4 + 4 + 36 against 46), so it works as our share weight
//   ratings      scoring and defense from the scout's direct inputs (see _gScoringEff);
//                passing from their computed PassingRolePerformance (display only)
//   climbs       DidEndRung1..3 -> L1..L3; AutoClimbYes -> auto climb
//   start        PrematchAutoStart [4]/[36] trench, [20] hub, [12]/[28] bump
const _G_ROLES = (row, p) => {
    const on = (r) => String(row[`Period${p}Role${r}`] ?? '').trim() === '1';
    return {
        scorer:   on('Alliance') || on('Neutral'),
        shuttler: on('Passing'),
        defender: on('Defending'),
        died:     on('NonFunctioning'),
    };
};
const _G_START = { '[4]': 'Trench', '[36]': 'Trench', '[20]': 'Hub', '[12]': 'Bump', '[28]': 'Bump' };

// EFFECTIVENESS FROM WHAT THE SCOUT ENTERED, not from the sheet's computed columns.
// The scout grades each period with one letter (the robot's play that period overall);
// the sheet's *RolePerformance columns copy that grade onto every role held and then
// average it by a formula that includes Transition/Endgame in ways we cannot see.
//   Scoring: the mean grade over exactly the periods we mark Scorer (Alliance or
//            Neutral); 0 when it never scored.
//   Defense: the scout's own post-match defense rating. "Not Observed" (490 of 746
//            rows) is 0, as a blank Defense Skill is on our sheets.
// The letter scale is the sheet's own fixed lookup (a=10, c=8, d=7, f=5, g=4, i=2, j=1,
// identical in every period); b/e/h fill the gaps on the same step.
const _G_GRADE = { a: 10, b: 9, c: 8, d: 7, e: 6, f: 5, g: 4, h: 3, i: 2, j: 1 };
const _G_PERIOD = ['PeriodOnePerformance', 'PeriodTwoPerformance', 'PeriodThreePerformance', 'PeriodFourPerformance'];
const _G_DEFENSE = { 'Not Effective': 2, 'Average': 5, 'Very Effective': 8 };
function _gScoringEff(row) {
    const grades = [1, 2, 3, 4]
        .filter(p => _G_ROLES(row, p).scorer)
        .map(p => _G_GRADE[String(row[_G_PERIOD[p - 1]] ?? '').trim().toLowerCase()])
        .filter(v => v !== undefined);
    return grades.length ? Math.round(10 * grades.reduce((s, v) => s + v, 0) / grades.length) / 10 : 0;
}
function galileoRow(row) {
    // Already translated (sync stores translated rows; see syncScoutingData): pass through.
    if (!('m (MatchScouting1)' in row) && 'Match Number' in row) return row;
    const g = (c) => String(row[c] ?? '').trim();
    const yes = (c) => g(c) === '1' || g(c).toUpperCase() === 'Y';
    const tf = (b) => (b ? 'TRUE' : 'FALSE');
    const level = yes('DidEndRung3') ? 'L3' : yes('DidEndRung2') ? 'L2' : yes('DidEndRung1') ? 'L1' : 'No';
    const collect = ['Depot', 'Neutral', 'Outpost'].filter(z => yes(`AutoCollect${z}`)).join('+');
    const out = {
        'Scouter Initials':        g('s (MatchScouting1)'),
        'Match Number':            g('m (MatchScouting1)'),
        'Team Number':             g('Team Number'),
        'Starting Position':       _G_START[g('PrematchAutoStart')] || '',
        'No Show':                 tf(yes('PostNoShow')),
        'Auto Action 1':           (parseFloat(g('CalculatedAuto')) || 0) > 0 ? 'Scored' : 'NoAction',
        'Auto Action 2':           collect ? `Collect ${collect}` : '',
        'Climbed Auto':            tf(yes('AutoClimbYes')),
        'Successful Climb':        tf(yes('AutoClimbYes')),
        'Fuel Scored in Auto?':    g('CalculatedAuto') || '0',
        'Climbed':                 level,
        'Scored Fuel in Endgame?': tf(yes('EndgameRoleAlliance') || yes('EndgameRoleNeutral')),
        'Scoring Effectiveness':   String(_gScoringEff(row)),
        'Passing Skill':           g('PassingRolePerformance') || '0',
        'Defense Skill':           String(_G_DEFENSE[g('PostDefenseRating')] ?? 0),
        'Shuttle Style?':          '',
        'Comments':                g('PostNote'),
    };
    [['TS1', 1], ['S2', 2], ['S3', 3], ['S4', 4]].forEach(([s, p]) => {
        const r = _G_ROLES(row, p);
        out[`${s} Scorer?`] = tf(r.scorer); out[`${s} Shuttler?`] = tf(r.shuttler);
        out[`${s} Defender?`] = tf(r.defender); out[`${s} Died?`] = tf(r.died);
    });
    return out;
}

export default {
    year: 2026,
    name: 'REBUILT',

    // ── Event sources ─────────────────────────────────────────────────────────
    eventSources: {
        '2026bc2': {
            label: 'Battle Cry @ WPI - Week 2',
            pitUrl: 'https://docs.google.com/spreadsheets/d/1JC95m3gYA65d7I_UuTGUxgQOvlCBv_sN18CRjH7iuas/edit?gid=2048789223#gid=2048789223'
        },
        '2026bc': {
            label:  'Battle Cry @ WPI - Week 1',
            pitUrl: 'https://docs.google.com/spreadsheets/d/1JC95m3gYA65d7I_UuTGUxgQOvlCBv_sN18CRjH7iuas/edit?gid=2048789223#gid=2048789223',
        },
        '2026necmp1': {
            label:  '2026 NE Championship - Burns',
            url:    'https://docs.google.com/spreadsheets/d/1ojsk9s9Tjc6-ZWRj-7yJsEaUwLkIFcg9wdnAA02aXUQ/edit?gid=0#gid=0',
            pitUrl: 'https://docs.google.com/spreadsheets/d/14ZDaH9QYnHnpaf0fCz_WMjRwNNyD7ud41nXrrQe2ncU/edit?gid=2048789223#gid=2048789223',
        },
        '2026gal': {
            label:  '2026 Championship - Galileo',
            // Another team's scouting system; rowAdapter translates it (see galileoRow).
            url:    'https://docs.google.com/spreadsheets/d/1sXE9_dJb7oGlKwj_kT0TkRKGpLpthdDCPRNZsZHsRm0/edit?gid=0#gid=0',
            // Same pit form as ours, so no adapter needed for it.
            pitUrl: 'https://docs.google.com/spreadsheets/d/1sXE9_dJb7oGlKwj_kT0TkRKGpLpthdDCPRNZsZHsRm0/edit?gid=2048789223#gid=2048789223',
            rowAdapter: galileoRow,
        },
        '2026mawor': {
            'label':  '2026 Week 5 - WPI - IN PROCESS',
        },
        '2026rikin': {
            'label':  '2026 Week 3 - URI - IN PROCESS',
        },
        '2026mabil': {
            'label':  '2026 Week 1 - Minuteman - IN PROCESS',
        },
        '2026week0': {
            'label':  '2026 Week 0 - IN PROCESS',
        },
        
    },

    // ── Column map ────────────────────────────────────────────────────────────
    defaultColumns: {
        matchNumber:      'Match Number',
        teamNumber:       'Team Number',
        startingPos:      'Starting Position',
        noShow:           'No Show',
        autoAction1:      'Auto Action 1',
        autoAction2:      'Auto Action 2',
        climbedAuto:      'Climbed Auto',
        successClimbAuto: 'Successful Climb',
        fuelScoredAuto:   'Fuel Scored in Auto?',
        shift1Scorer:     'TS1 Scorer?',
        shift1Shuttler:   'TS1 Shuttler?',
        shift1Defender:   'TS1 Defender?',
        shift1Died:       'TS1 Died?',
        shift2Scorer:     'S2 Scorer?',
        shift2Shuttler:   'S2 Shuttler?',
        shift2Defender:   'S2 Defender?',
        shift2Died:       'S2 Died?',
        shift3Scorer:     'S3 Scorer?',
        shift3Shuttler:   'S3 Shuttler?',
        shift3Defender:   'S3 Defender?',
        shift3Died:       'S3 Died?',
        shift4Scorer:     'S4 Scorer?',
        shift4Shuttler:   'S4 Shuttler?',
        shift4Defender:   'S4 Defender?',
        shift4Died:       'S4 Died?',
        climbedEndgame:   'Climbed',  // 'No' | 'L1' | 'L2' | 'L3' — level encodes both climb and height
        whereClimbed:     'Climbed',  // same column — level re-used as the level string downstream
        fuelEndgame:      'Scored Fuel in Endgame?',
        scoringEff:       'Scoring Effectiveness',
        passingSkill:     'Passing Skill',
        defenseSkill:     'Defense Skill',
        shuttleStyle:     'Shuttle Style?',
        comments:         'Comments',
    },

    // ── Row processing ────────────────────────────────────────────────────────
    processRow(row, colMap) {
        const get  = (f) => row[colMap[f]] ?? '';
        const num  = (f) => parseFloat(get(f)) || 0;
        const bool = (f) => { const v = get(f).trim().toLowerCase(); return v === 'true' || v === 'yes' || v === '1' || v === 'x'; };

        // 'Climbed' column contains the level string ('No'|'L1'|'L2'|'L3'), not a boolean
        const climbLevel = get('climbedEndgame').trim().toLowerCase();

        return {
            matchNumber:      num('matchNumber'),
            teamNumber:       get('teamNumber').trim(),
            startingPos:      get('startingPos').trim(),
            noShow:           bool('noShow'),
            autoAction1:      get('autoAction1').trim(),
            autoAction2:      get('autoAction2').trim(),
            climbedAuto:      bool('climbedAuto'),
            successClimbAuto: bool('successClimbAuto'),
            fuelScoredAuto:   num('fuelScoredAuto'),  // estimated count scored by this robot in auto
            shift1Scorer:     bool('shift1Scorer'),
            shift1Shuttler:   bool('shift1Shuttler'),
            shift1Defender:   bool('shift1Defender'),
            shift1Died:       bool('shift1Died'),
            shift2Scorer:     bool('shift2Scorer'),
            shift2Shuttler:   bool('shift2Shuttler'),
            shift2Defender:   bool('shift2Defender'),
            shift2Died:       bool('shift2Died'),
            shift3Scorer:     bool('shift3Scorer'),
            shift3Shuttler:   bool('shift3Shuttler'),
            shift3Defender:   bool('shift3Defender'),
            shift3Died:       bool('shift3Died'),
            shift4Scorer:     bool('shift4Scorer'),
            shift4Shuttler:   bool('shift4Shuttler'),
            shift4Defender:   bool('shift4Defender'),
            shift4Died:       bool('shift4Died'),
            climbedEndgame:   climbLevel !== '' && climbLevel !== 'no',
            whereClimbed:     climbLevel,  // 'l1'|'l2'|'l3' (or 'no'/'' for no climb)
            fuelEndgame:      bool('fuelEndgame'),
            scoringEff:       num('scoringEff'),   // 1–10
            passingSkill:     num('passingSkill'),  // 1–10
            defenseSkill:     num('defenseSkill'),  // 1–10
            shuttleStyle:     get('shuttleStyle').trim(),
            comments:         get('comments').trim(),
        };
    },

    // ── Team aggregation ──────────────────────────────────────────────────────
    aggregateTeam(rows) {
        const played = rows.filter(r => !r.noShow);
        const n = played.length || 1;

        const pct    = (fn) => Math.round(played.filter(fn).length / n * 100);
        const avg    = (fn) => played.reduce((s, r) => s + fn(r), 0) / n;
        const climbW = (r) => { const w = (r.whereClimbed || '').toLowerCase(); return r.climbedEndgame ? (w === 'l3' ? 30 : w === 'l2' ? 20 : 10) : 0; };
        const climbed = played.filter(r => r.climbedEndgame);
        const nc = climbed.length || 1;

        return {
            matches:  played.length,
            noShows:  rows.filter(r => r.noShow).length,

            // Auto
            autoFuelPct:         pct(r => (r.fuelScoredAuto ?? 0) > 0),
            avgAutoFuel:         avg(r => r.fuelScoredAuto ?? 0),
            autoClimbPct:        pct(r => r.climbedAuto),
            autoClimbSuccessPct: pct(r => r.successClimbAuto),
            autoClimbPts:        avg(r => r.climbedAuto ? 15 : 0),

            // Per-shift role rates
            shift1ScorerPct:   pct(r => r.shift1Scorer),   shift1ShuttlerPct: pct(r => r.shift1Shuttler),
            shift1DefenderPct: pct(r => r.shift1Defender), shift1DiedPct:     pct(r => r.shift1Died),
            shift2ScorerPct:   pct(r => r.shift2Scorer),   shift2ShuttlerPct: pct(r => r.shift2Shuttler),
            shift2DefenderPct: pct(r => r.shift2Defender), shift2DiedPct:     pct(r => r.shift2Died),
            shift3ScorerPct:   pct(r => r.shift3Scorer),   shift3ShuttlerPct: pct(r => r.shift3Shuttler),
            shift3DefenderPct: pct(r => r.shift3Defender), shift3DiedPct:     pct(r => r.shift3Died),
            shift4ScorerPct:   pct(r => r.shift4Scorer),   shift4ShuttlerPct: pct(r => r.shift4Shuttler),
            shift4DefenderPct: pct(r => r.shift4Defender), shift4DiedPct:     pct(r => r.shift4Died),

            // Convenience roll-ups
            pctScored:   pct(r => r.shift1Scorer || r.shift2Scorer || r.shift3Scorer || r.shift4Scorer),
            pctDefended: pct(r => r.shift1Defender || r.shift2Defender || r.shift3Defender || r.shift4Defender),
            pctDied:     pct(r => r.shift1Died || r.shift2Died || r.shift3Died || r.shift4Died),

            // Endgame
            climbPct:      pct(r => r.climbedEndgame),
            pctL1:         climbed.length ? Math.round(climbed.filter(r => (r.whereClimbed || '').toLowerCase() === 'l1').length / nc * 100) : 0,
            pctL2:         climbed.length ? Math.round(climbed.filter(r => (r.whereClimbed || '').toLowerCase() === 'l2').length / nc * 100) : 0,
            pctL3:         climbed.length ? Math.round(climbed.filter(r => (r.whereClimbed || '').toLowerCase() === 'l3').length / nc * 100) : 0,
            endgamePts:    avg(r => climbW(r)),
            endgameFuelPct: pct(r => r.fuelEndgame),

            // Qualitative
            avgScoringEff:  avg(r => r.scoringEff),
            avgPassingSkill: avg(r => r.passingSkill),
            avgDefenseSkill: avg(r => r.defenseSkill),

            comments: rows
                .filter(r => r.comments)
                .map(r => ({ matchNumber: r.matchNumber, text: r.comments }))
                .sort((a, b) => a.matchNumber - b.matchNumber),

            observations: rows,
        };
    },

    // ── TBA fusion stat definitions ───────────────────────────────────────────

    // Auto fuel is now allocated in fuseCustom using numeric estimates with per-team avg fallback.
    autoFuseStats: [],

    teleopFuseStats: [],
    splitFuseStats:  [],

    robotFuseStats: [
        {
            key: 'autoClimbPts',
            tba: (b, i) => {
                const v = b[`autoTowerRobot${i}`];
                if (v == null || v === 'None') return null;
                return (v === 'Level1' || v === true || v === 'true') ? 15 : 0;
            },
            scout: r => r.climbedAuto ? 15 : 0,
        },
        {
            key: 'endgamePts',
            tba: (b, i) => {
                const v = (b[`endGameTowerRobot${i}`] || '').toLowerCase();
                if (!v || v === 'none') return null;
                return v === 'level3' ? 30 : v === 'level2' ? 20 : v === 'level1' ? 10 : 0;
            },
            scout: r => {
                const w = (r.whereClimbed || '').toLowerCase();
                return r.climbedEndgame ? (w === 'l3' ? 30 : w === 'l2' ? 20 : 10) : 0;
            },
        },
    ],

    // Keys produced by fuseCustom — included in final per-match averages.
    customFuseKeys: ['autoFuelFused', 'teleFuelFused', 'endgameFuelFused'],

    // ── Active-scorer rate enrichment ─────────────────────────────────────────
    // Mutates agg to add activeScorerRate: fraction of active-hub period appearances
    // where this team was a Scorer. Called once per team before per-match fusion.
    enrichAggregateWithTBA(tn, rows, agg, tbaByMatch) {
        let activeAppearances = 0;
        let activeScored      = 0;

        // Conditional shift role counters — denominator is hub-active appearances only.
        // TS1 uses all matches with TBA data (Transition always runs).
        // S2-S4 use only matches where that shift's hub count > 0.
        const sd = { 1: {n:0,sc:0,sh:0,def:0,d:0}, 2: {n:0,sc:0,sh:0,def:0,d:0},
                     3: {n:0,sc:0,sh:0,def:0,d:0}, 4: {n:0,sc:0,sh:0,def:0,d:0} };

        for (const row of rows) {
            if (row.noShow) continue;
            const m  = tbaByMatch[row.matchNumber];
            if (!m) continue;
            const isRed = m.red?.includes(String(tn));
            const hs = (isRed ? m.redBreakdown : m.blueBreakdown)?.hubScore;
            if (!hs) continue;

            // TS1 — Transition always runs, so always count this match
            sd[1].n++;
            if (row.shift1Scorer)   sd[1].sc++;
            if (row.shift1Shuttler) sd[1].sh++;
            if (row.shift1Defender) sd[1].def++;
            if (row.shift1Died)     sd[1].d++;

            // S2-S4 — only when that hub was active this match
            for (const n of [2, 3, 4]) {
                if ((hs[`shift${n}Count`] ?? 0) > 0) {
                    sd[n].n++;
                    if (row[`shift${n}Scorer`])   sd[n].sc++;
                    if (row[`shift${n}Shuttler`]) sd[n].sh++;
                    if (row[`shift${n}Defender`]) sd[n].def++;
                    if (row[`shift${n}Died`])     sd[n].d++;
                }
            }

            // activeScorerRate — existing logic unchanged
            const periodDefs = [
                { active: true,                       scored: row.shift1Scorer },
                { active: (hs.shift1Count ?? 0) > 0, scored: row.shift1Scorer },
                { active: (hs.shift2Count ?? 0) > 0, scored: row.shift2Scorer },
                { active: (hs.shift3Count ?? 0) > 0, scored: row.shift3Scorer },
                { active: (hs.shift4Count ?? 0) > 0, scored: row.shift4Scorer },
                { active: true,                       scored: row.fuelEndgame  },
            ];
            for (const { active, scored } of periodDefs) {
                if (active) { activeAppearances++; if (scored) activeScored++; }
            }
        }

        agg.activeScorerRate = activeAppearances > 0 ? activeScored / activeAppearances : 0;

        // Overwrite unconditional shift percentages with conditional versions.
        // Falls back to aggregateTeam values when no TBA data is available (n === 0).
        for (const n of [1, 2, 3, 4]) {
            const { n: tot, sc, sh, def, d } = sd[n];
            if (tot > 0) {
                agg[`shift${n}ScorerPct`]   = Math.round(sc  / tot * 100);
                agg[`shift${n}ShuttlerPct`] = Math.round(sh  / tot * 100);
                agg[`shift${n}DefenderPct`] = Math.round(def / tot * 100);
                agg[`shift${n}DiedPct`]     = Math.round(d   / tot * 100);
            }
        }
    },

    // ── Per-period fuel allocation (custom fusion hook) ───────────────────────
    // Covers Transition+Shift1, Shifts 2-4, and Endgame.
    // Unscouted alliance partners get assumed weight = activeScorerRate × avgScoringEff.
    fuseCustom(obs, allianceRows, tbaMatch, teamStr, { allianceTeams, teamAggregates } = {}) {
        const isRed = tbaMatch.red?.includes(teamStr);
        const myBd  = isRed ? tbaMatch.redBreakdown : tbaMatch.blueBreakdown;
        if (!myBd?.hubScore) return {};

        const hs = myBd.hubScore;

        // TS1 combines Transition + Shift 1 (scouting tracks them as one TS1 role).
        const shiftDefs = [
            { tbaFuel: (hs.transitionCount ?? 0) + (hs.shift1Count ?? 0), scorerField: 'shift1Scorer', endgame: false },
            { tbaFuel: hs.shift2Count  ?? null, scorerField: 'shift2Scorer', endgame: false },
            { tbaFuel: hs.shift3Count  ?? null, scorerField: 'shift3Scorer', endgame: false },
            { tbaFuel: hs.shift4Count  ?? null, scorerField: 'shift4Scorer', endgame: false },
            { tbaFuel: hs.endgameCount ?? null, scorerField: 'fuelEndgame',  endgame: true  },
        ];

        // Scouted row: use actual flag × scoringEff.
        // Unscouted (null): assume activeScorerRate × avgScoringEff from historical aggregates.
        const getWeight = (row, slotIdx, scorerField) => {
            if (row !== null) {
                return row[scorerField] ? (row.scoringEff || 1) : 0;
            }
            const agg = teamAggregates?.[allianceTeams?.[slotIdx]];
            if (!agg) return 1; // no history — assume equal share
            return (agg.activeScorerRate ?? 0) * (agg.avgScoringEff || 1);
        };

        const myIdx = allianceTeams?.indexOf(teamStr) ?? -1;

        // Assumed weight for any team based solely on historical aggregates.
        // Used as a fallback when all scouted teams report non-scorer.
        const getAssumedWeight = tn => {
            const agg = teamAggregates?.[tn];
            if (!agg) return 1;
            return (agg.activeScorerRate ?? 0) * (agg.avgScoringEff || 1);
        };

        // Auto fuel: share TBA's autoCount by each robot's scouted estimate.
        // Unscouted partners use their historical avgAutoFuel as the weight.
        let autoFuelFused = null;
        const autoTotal = hs.autoCount ?? null;
        if (autoTotal != null) {
            const getAutoWeight = (row, slotIdx) => {
                if (row !== null) return row.fuelScoredAuto ?? 0;
                const agg = teamAggregates?.[allianceTeams?.[slotIdx]];
                return agg?.avgAutoFuel ?? 1;
            };
            const myAutoWeight       = getAutoWeight(myIdx >= 0 ? (allianceRows[myIdx] ?? obs) : obs, myIdx);
            const allianceAutoWeight = allianceRows.reduce((sum, r, i) => sum + getAutoWeight(r, i), 0);
            const autoShare = allianceAutoWeight > 0 ? myAutoWeight / allianceAutoWeight : 1 / 3;
            autoFuelFused = autoShare * autoTotal;
        }

        let teleFuelFused    = null;
        let endgameFuelFused = null;
        for (const { tbaFuel, scorerField, endgame } of shiftDefs) {
            if (tbaFuel == null) continue;

            const myWeight       = getWeight(myIdx >= 0 ? (allianceRows[myIdx] ?? obs) : obs, myIdx, scorerField);
            const allianceWeight = allianceRows.reduce((sum, r, i) => sum + getWeight(r, i, scorerField), 0);

            let share;
            if (allianceWeight > 0) {
                share = myWeight / allianceWeight;
            } else {
                // All scouts report non-scorer — fall back to historical assumed weights
                const assumedMine = getAssumedWeight(teamStr);
                const assumedAll  = (allianceTeams ?? []).reduce((sum, tn) => sum + getAssumedWeight(tn), 0);
                share = assumedAll > 0 ? assumedMine / assumedAll : 1 / 3;
            }

            const contribution = share * tbaFuel;
            if (endgame) endgameFuelFused = (endgameFuelFused ?? 0) + contribution;
            else         teleFuelFused    = (teleFuelFused    ?? 0) + contribution;
        }

        return { autoFuelFused, teleFuelFused, endgameFuelFused };
    },

    // ── Derived fused totals ──────────────────────────────────────────────────
    deriveFusedTotals(s) {
        if (s.autoFuelFused != null && s.teleFuelFused != null)
            s.totalFuelFused = s.autoFuelFused + s.teleFuelFused + (s.endgameFuelFused ?? 0);
    },

    // ── Per-match breakdown columns ───────────────────────────────────────────
    matchBreakdownColumns: [
        {
            label: 'Auto Fuel',
            raw:   r => (r.fuelScoredAuto ?? 0) > 0 ? String(r.fuelScoredAuto) : '—',
            fused: s => s.autoFuelFused ?? null,
        },
        {
            label: 'Tele Fuel',
            raw:   r => `S:${[1,2,3,4].filter(n => r[`shift${n}Scorer`]).join(',') || '—'}`,
            fused: s => s.teleFuelFused ?? null,
        },
        {
            label: 'Endgame',
            raw: r => {
                const w = (r.whereClimbed || '').toLowerCase();
                const climb = (r.climbedAuto ? 'A ' : '') + (r.climbedEndgame ? w.toUpperCase() : '—');
                const fuel  = r.fuelEndgame ? ' +F' : '';
                return climb + fuel;
            },
            fused: s => (s.endgamePts ?? 0) + (s.autoClimbPts ?? 0) + (s.endgameFuelFused ?? 0),
        },
    ],

    // ── EPA breakdowns ────────────────────────────────────────────────────────

    // Raw estimate when TBA fusion is unavailable — uses Scoring Effectiveness as proxy.
    computeEPABreakdown(s) {
        const autoClimbPts = (s.autoClimbPct || 0) / 100 * 15;
        const endgamePts   = ((s.pctL1 || 0) * 10 + (s.pctL2 || 0) * 20 + (s.pctL3 || 0) * 30) / 100;
        const teleopEst    = (s.avgScoringEff || 0) * 2; // 1–10 scale; rough pre-fusion estimate
        return { auto: autoClimbPts, teleop: teleopEst, endgame: endgamePts, total: autoClimbPts + teleopEst + endgamePts };
    },

    computeFusedEPABreakdown(s) {
        const auto    = (s.autoFuelFused ?? 0) + (s.autoClimbPts ?? 0);
        const teleop  = s.teleFuelFused ?? 0;
        const endgame = (s.endgamePts ?? 0) + (s.endgameFuelFused ?? 0);
        return { auto, teleop, endgame, total: auto + teleop + endgame };
    },

    scoringWeights: { autoFuelFused: 1, teleFuelFused: 1, autoClimbPts: 1, endgamePts: 1, endgameFuelFused: 1 },

    // Per-match EPA used for outlier detection. scoringWeights keys are fused outputs
    // unavailable on raw rows, so compute directly via computeEPABreakdown instead.
    computeMatchEPA(row) {
        return this.computeEPABreakdown(this.aggregateTeam([row])).total;
    },

    // ── Team detail scouting tab ──────────────────────────────────────────────

    // Columns shown in the analysis table's "Functional" toggle view.
    functionalColumns: [
        { label: 'Tele Fuel', sortKey: 'funcTeleFuel', decimals: 1,
          getValue: (raw, fused) => fused?.available ? (fused.stats?.teleFuelFused ?? null) : null },
        { label: 'Auto Pts',  sortKey: 'funcAutoPts',  decimals: 1,
          getValue: (raw, fused) => fused?.available ? (fused.stats?.autoFuelFused ?? 0) + (fused.stats?.autoClimbPts ?? 0) : null },
        { label: 'Scorer%',   sortKey: 'funcScorer',   suffix: '%',
          getValue: (raw) => raw.pctScored },
        { label: 'Avg Eff',   sortKey: 'funcAvgEff',   decimals: 1,
          getValue: (raw) => raw.avgScoringEff },
        { label: 'Climb%',    sortKey: 'funcClimb',    suffix: '%',
          getValue: (raw) => raw.climbPct },
    ],

    // Rich scouting detail rendered in the team detail overlay's Scouting tab.
    // Replaces the generic displayFields grid when defined.
    renderScoutingDetail({ rawStats: s, fusedStats, getValue, isFused, fused, allScoutBreakdowns = [], rankTier }) {
        const BG   = { S: '#f59e0b', A: '#4ade80', B: '#a855f7', C: '#64748b' };
        const tier = (val, fieldFn) => {
            if (val == null) return 'C';
            return rankTier ? rankTier(allScoutBreakdowns.map(fieldFn), val) : 'C';
        };
        const badge = (t) => {
            const fg = t === 'C' ? '#f8fafc' : '#0f172a';
            return `<span style="display:inline-block;padding:1px 7px;border-radius:4px;font-size:0.72em;font-weight:800;background:${BG[t]};color:${fg};">${t}</span>`;
        };
        const dot = (key) => isFused(key)
            ? `<span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#4ade80;margin-left:4px;vertical-align:middle;"></span>`
            : '';
        const sHdr = (label, color) =>
            `<div style="color:${color};font-size:0.7em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin:0 0 8px;border-left:3px solid ${color};padding-left:10px;">${label}</div>`;

        const epa = fused?.available
            ? this.computeFusedEPABreakdown(fusedStats)
            : this.computeEPABreakdown(s);

        const teleFuel   = getValue('teleFuelFused');
        const autoPts    = (getValue('autoFuelFused') ?? 0) + (getValue('autoClimbPts') ?? 0);
        const endPts     = (getValue('endgamePts') ?? 0) + (getValue('endgameFuelFused') ?? 0);

        // ── Top-line metric cards ─────────────────────────────────────────────
        const topMetrics = [
            { label: 'Total EPA',    val: epa.total,       fmt: v => v.toFixed(1), tierFn: b => b.total,        fDot: dot('teleFuelFused') || dot('autoFuelFused') },
            { label: 'Tele Fuel',    val: teleFuel,        fmt: v => v.toFixed(1), tierFn: b => b.teleFuelFused, fDot: dot('teleFuelFused') },
            { label: 'Auto Pts',     val: autoPts,         fmt: v => v.toFixed(1), tierFn: b => b.auto,          fDot: dot('autoFuelFused') },
            { label: 'Endgame Pts',  val: endPts,          fmt: v => v.toFixed(1), tierFn: b => b.endgame,       fDot: dot('endgamePts') },
            { label: 'Avg Eff',      val: s.avgScoringEff, fmt: v => v.toFixed(1), tierFn: b => b.avgScoringEff, fDot: '' },
        ];

        const topCardsHtml = topMetrics.map(m => {
            const t  = m.val != null ? tier(m.val, m.tierFn) : 'C';
            const bc = BG[t];
            return `
            <div style="background:rgba(255,255,255,0.02);border:1px solid ${bc}44;border-radius:8px;padding:12px;text-align:center;">
                <div style="color:#94a3b8;font-size:0.63em;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">${m.label}</div>
                <div style="font-size:1.25em;font-weight:800;display:flex;align-items:center;justify-content:center;margin-bottom:4px;">${m.val != null ? m.fmt(m.val) : '—'}${m.fDot}</div>
                ${badge(t)}
            </div>`;
        }).join('');

        // ── Shift breakdown table ─────────────────────────────────────────────
        const shiftRows = [
            { label: 'TS1', sc: s.shift1ScorerPct, sh: s.shift1ShuttlerPct, def: s.shift1DefenderPct, d: s.shift1DiedPct },
            { label: 'S2',  sc: s.shift2ScorerPct, sh: s.shift2ShuttlerPct, def: s.shift2DefenderPct, d: s.shift2DiedPct },
            { label: 'S3',  sc: s.shift3ScorerPct, sh: s.shift3ShuttlerPct, def: s.shift3DefenderPct, d: s.shift3DiedPct },
            { label: 'S4',  sc: s.shift4ScorerPct, sh: s.shift4ShuttlerPct, def: s.shift4DefenderPct, d: s.shift4DiedPct },
        ].map(r => {
            const sc  = v => `color:${v>=60?'#4ade80':v>=30?'#f59e0b':'#475569'};font-weight:${v>=30?700:400}`;
            const sh  = v => `color:${v>=40?'#60a5fa':v>=20?'#93c5fd':'#475569'}`;
            const def = v => `color:${v>=40?'#f87171':v>=20?'#fca5a5':'#475569'}`;
            const die = v => `color:${v>=20?'#ef4444':'#475569'}`;
            return `<tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:5px 8px;color:#94a3b8;font-weight:600;font-size:0.85em;">${r.label}</td>
                <td style="text-align:center;padding:5px 6px;${sc(r.sc)}">${r.sc}%</td>
                <td style="text-align:center;padding:5px 6px;${sh(r.sh)}">${r.sh}%</td>
                <td style="text-align:center;padding:5px 6px;${def(r.def)}">${r.def}%</td>
                <td style="text-align:center;padding:5px 6px;${die(r.d)}">${r.d}%</td>
            </tr>`;
        }).join('');

        const shiftTableHtml = `
        <table style="width:100%;border-collapse:collapse;font-size:0.8em;">
            <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                <th style="text-align:left;padding:4px 8px;"></th>
                <th style="text-align:center;padding:4px 6px;">Scorer%</th>
                <th style="text-align:center;padding:4px 6px;">Shuttler%</th>
                <th style="text-align:center;padding:4px 6px;">Defender%</th>
                <th style="text-align:center;padding:4px 6px;">Died%</th>
            </tr></thead>
            <tbody>${shiftRows}</tbody>
        </table>`;

        // ── Endgame cards ─────────────────────────────────────────────────────
        const endItems = [
            { label: 'Climbed', val: `${s.climbPct}%`,       color: s.climbPct>=70?'#4ade80':s.climbPct>=40?'#f59e0b':'#94a3b8' },
            { label: 'L1',      val: `${s.pctL1}%` },
            { label: 'L2',      val: `${s.pctL2}%` },
            { label: 'L3',      val: `${s.pctL3}%` },
            { label: 'Fuel%',   val: `${s.endgameFuelPct}%` },
        ];
        const endCardsHtml = endItems.map(e =>
            `<div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.12);border-radius:6px;padding:10px;text-align:center;min-width:60px;">
                <div style="color:#64748b;font-size:0.63em;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">${e.label}</div>
                <div style="font-size:1.05em;font-weight:700;color:${e.color||'#f1f5f9'};">${e.val}</div>
            </div>`
        ).join('');

        // ── Qualitative cards ─────────────────────────────────────────────────
        const effTier = s.avgScoringEff != null ? tier(s.avgScoringEff, b => b.avgScoringEff) : 'C';
        const qualCardsHtml = [
            { label: 'Scoring Eff',   val: s.avgScoringEff,   t: effTier, suffix: '/10' },
            { label: 'Passing Skill', val: s.avgPassingSkill, t: null,    suffix: '/10' },
            { label: 'Defense Skill', val: s.avgDefenseSkill, t: null,    suffix: '/10' },
        ].map(c =>
            `<div style="background:rgba(167,139,250,0.05);border:1px solid rgba(167,139,250,0.12);border-radius:6px;padding:10px;text-align:center;min-width:80px;">
                <div style="color:#64748b;font-size:0.63em;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">${c.label}</div>
                <div style="font-size:1.05em;font-weight:700;margin-bottom:${c.t?'4px':'0'};">${c.val!=null?c.val.toFixed(1)+''+c.suffix:'—'}</div>
                ${c.t ? badge(c.t) : ''}
            </div>`
        ).join('');

        return `
        <div style="margin-bottom:20px;">
            ${sHdr('Performance', '#60a5fa')}
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:8px;">${topCardsHtml}</div>
        </div>
        <div style="margin-bottom:20px;">
            ${sHdr('Shift Breakdown', '#3b82f6')}
            <div style="overflow-x:auto;">${shiftTableHtml}</div>
        </div>
        <div style="margin-bottom:20px;">
            ${sHdr('Endgame', '#10b981')}
            <div style="display:flex;gap:8px;flex-wrap:wrap;">${endCardsHtml}</div>
        </div>
        <div style="margin-bottom:20px;">
            ${sHdr('Qualitative', '#a78bfa')}
            <div style="display:flex;gap:8px;flex-wrap:wrap;">${qualCardsHtml}</div>
        </div>`;
    },

    // ── Game-specific curation section ────────────────────────────────────────
    curationSection(tbaMatches, matchCoverage, scoutIndex, _isCumulative, _reportingMode, { summaryStyle, hdrStyle, chevron }) {
        const hasBreakdowns = tbaMatches.some(m => m.redBreakdown);
        if (!hasBreakdowns) return '';

        // ── Part 1: Hub Fuel by Shift summary ────────────────────────────────
        const summaryRows = [];
        for (const { mn, m } of matchCoverage) {
            for (const [alliance, bd] of [['red', m?.redBreakdown], ['blue', m?.blueBreakdown]]) {
                if (!bd?.hubScore) continue;
                const teams = m?.[alliance] || [];
                const hs = bd.hubScore;
                const shifts = [hs.shift1Count ?? 0, hs.shift2Count ?? 0, hs.shift3Count ?? 0, hs.shift4Count ?? 0];
                const activeShifts = shifts.map((c, i) => c > 0 ? `S${i+1}:${c}` : null).filter(Boolean);
                summaryRows.push({
                    mn, alliance,
                    activeShifts,
                    totalActive:  shifts.reduce((a, b) => a + b, 0),
                    autoCount:    hs.autoCount ?? 0,
                    fullyScouted: teams.every(t => scoutIndex[mn]?.[t]),
                });
            }
        }
        if (summaryRows.length === 0) return '';

        const totalFuel  = summaryRows.reduce((s, r) => s + r.autoCount + r.totalActive, 0);
        const part1 = `
        <details style="margin-bottom:20px;">
            <summary style="${summaryStyle('#f59e0b')}">
                <span style="${hdrStyle('#f59e0b')}">2026 — Hub Fuel by Shift <span style="color:#475569;font-weight:400;font-size:0.9em;">(${summaryRows.length} alliances · ${totalFuel} total fuel)</span></span>
                ${chevron}
            </summary>
            <div style="margin-top:12px;">
            <p style="color:#94a3b8;font-size:0.82em;margin:0 0 10px;">Active shifts only (inactive = 0). Fusion weights Scorers by Scoring Effectiveness within each active shift.</p>
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.75em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:3px 6px;">Match</th><th style="padding:3px 6px;">Alliance</th>
                    <th style="padding:3px 6px;">Auto</th><th style="text-align:left;padding:3px 6px;">Active Shifts</th>
                    <th style="padding:3px 6px;">Tele Total</th><th style="padding:3px 6px;">Scouted</th>
                </tr></thead>
                <tbody>
                ${summaryRows.map(r => `<tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:3px 6px;color:#60a5fa;">QM ${r.mn}</td>
                    <td style="padding:3px 6px;color:${r.alliance === 'red' ? '#f87171' : '#60a5fa'};">${r.alliance}</td>
                    <td style="text-align:center;padding:3px 6px;">${r.autoCount}</td>
                    <td style="padding:3px 6px;color:#94a3b8;">${r.activeShifts.join(' · ') || '—'}</td>
                    <td style="text-align:center;padding:3px 6px;">${r.totalActive}</td>
                    <td style="text-align:center;padding:3px 6px;color:${r.fullyScouted ? '#4ade80' : '#f59e0b'};">${r.fullyScouted ? '✓' : '~'}</td>
                </tr>`).join('')}
                </tbody>
            </table>
            </div></div>
        </details>`;

        // ── Part 2: Fusion debug ──────────────────────────────────────────────
        // Build per-team stats from scoutIndex (scorer rates + activeScorerRate)
        const teamObs = {};
        for (const matchTeams of Object.values(scoutIndex)) {
            for (const [tn, row] of Object.entries(matchTeams)) {
                if (!teamObs[tn]) teamObs[tn] = [];
                teamObs[tn].push(row);
            }
        }
        const tbaByMatch = {};
        for (const m of tbaMatches) tbaByMatch[m.matchNumber] = m;

        const teamDebug = {};
        for (const [tn, tRows] of Object.entries(teamObs)) {
            const played = tRows.filter(r => !r.noShow);
            const n = played.length || 1;
            const pct = fn => Math.round(played.filter(fn).length / n * 100);
            const avg = fn => (played.reduce((s, r) => s + fn(r), 0) / n).toFixed(1);
            let app = 0, sc = 0;
            for (const row of played) {
                const m = tbaByMatch[row.matchNumber];
                if (!m) continue;
                const isRed = m.red?.includes(tn);
                const hs = (isRed ? m.redBreakdown : m.blueBreakdown)?.hubScore;
                if (!hs) continue;
                const defs = [
                    { active: true,                       scored: row.shift1Scorer },
                    { active: (hs.shift1Count ?? 0) > 0, scored: row.shift1Scorer },
                    { active: (hs.shift2Count ?? 0) > 0, scored: row.shift2Scorer },
                    { active: (hs.shift3Count ?? 0) > 0, scored: row.shift3Scorer },
                    { active: (hs.shift4Count ?? 0) > 0, scored: row.shift4Scorer },
                    { active: true,                       scored: row.fuelEndgame  },
                ];
                for (const { active, scored } of defs) {
                    if (active) { app++; if (scored) sc++; }
                }
            }
            teamDebug[tn] = {
                matches: played.length,
                avgEff:  avg(r => r.scoringEff),
                activeRate: app > 0 ? Math.round(sc / app * 100) : null,
                ts1: pct(r => r.shift1Scorer), s2: pct(r => r.shift2Scorer),
                s3:  pct(r => r.shift3Scorer), s4: pct(r => r.shift4Scorer),
                end: pct(r => r.fuelEndgame),
            };
        }

        // Per-match attribution rows
        const attrData = [];
        for (const { mn, m } of matchCoverage) {
            for (const [alliance, bd] of [['red', m?.redBreakdown], ['blue', m?.blueBreakdown]]) {
                if (!bd?.hubScore) continue;
                const teams = m?.[alliance] || [];
                const hs = bd.hubScore;
                const teamRows = teams.map(t => scoutIndex[mn]?.[t] ?? null);
                const ts1 = (hs.transitionCount ?? 0) + (hs.shift1Count ?? 0);
                const periods = [];
                if (ts1 > 0) periods.push({ label: 'TS1', fuel: ts1, field: 'shift1Scorer' });
                for (const [count, label, field] of [
                    [hs.shift2Count, 'S2', 'shift2Scorer'], [hs.shift3Count, 'S3', 'shift3Scorer'],
                    [hs.shift4Count, 'S4', 'shift4Scorer'], [hs.endgameCount, 'End', 'fuelEndgame'],
                ]) {
                    if ((count ?? 0) > 0) periods.push({ label, fuel: count, field });
                }
                if (periods.length) attrData.push({ mn, alliance, teams, teamRows, periods });
            }
        }

        // Render per-team cells for a period: green S·eff→share%, gray N, italic ? for unscouted
        const periodCells = (teamRows, field) => {
            const weights = teamRows.map(r => r === null ? null : (r[field] ? (r.scoringEff || 1) : 0));
            const total   = weights.reduce((s, w) => s + (w ?? 0), 0);
            return weights.map((w, i) => {
                if (w === null) return `<td style="text-align:center;padding:2px 4px;" title="${teamRows.length > i ? '' : ''}"><span style="color:#475569;font-style:italic;">?</span></td>`;
                if (w === 0)   return `<td style="text-align:center;padding:2px 4px;color:#334155;">N</td>`;
                const share = total > 0 ? Math.round(w / total * 100) : 33;
                return `<td style="text-align:center;padding:2px 4px;"><span style="color:#4ade80;">S·${Number(w).toFixed(0)}→${share}%</span></td>`;
            }).join('');
        };

        const part2 = `
        <details style="margin-bottom:20px;">
            <summary style="${summaryStyle('#38bdf8')}">
                <span style="${hdrStyle('#38bdf8')}">2026 — Fusion Debug</span>
                ${chevron}
            </summary>
            <div style="margin-top:12px;">

            <p style="color:#94a3b8;font-size:0.82em;margin:0 0 4px;font-weight:600;">Team Scorer Rates</p>
            <p style="color:#64748b;font-size:0.78em;margin:0 0 8px;">ActiveRate = Scorer appearances ÷ active-hub period appearances (Transition + active Shifts 1–4 + Endgame). Green ≥60%, amber ≥30%, red &lt;30%.</p>
            <div style="overflow-x:auto;margin-bottom:16px;">
            <table style="width:100%;border-collapse:collapse;font-size:0.75em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:3px 6px;">Team</th>
                    <th style="padding:3px 6px;">M</th><th style="padding:3px 6px;">AvgEff</th>
                    <th style="padding:3px 6px;">ActiveRate</th>
                    <th style="padding:3px 6px;">TS1%</th><th style="padding:3px 6px;">S2%</th>
                    <th style="padding:3px 6px;">S3%</th><th style="padding:3px 6px;">S4%</th>
                    <th style="padding:3px 6px;">End%</th>
                </tr></thead>
                <tbody>
                ${Object.entries(teamDebug).sort(([a],[b]) => Number(a)-Number(b)).map(([tn, s]) => {
                    const rc = s.activeRate === null ? '#64748b' : s.activeRate >= 60 ? '#4ade80' : s.activeRate >= 30 ? '#f59e0b' : '#ef4444';
                    const ec = parseFloat(s.avgEff) >= 6 ? '#4ade80' : parseFloat(s.avgEff) >= 3 ? '#f59e0b' : '#ef4444';
                    return `<tr style="border-bottom:1px solid #1e293b;">
                        <td style="padding:3px 6px;color:#f1f5f9;font-weight:600;">${tn}</td>
                        <td style="text-align:center;padding:3px 6px;">${s.matches}</td>
                        <td style="text-align:center;padding:3px 6px;color:${ec};">${s.avgEff}</td>
                        <td style="text-align:center;padding:3px 6px;color:${rc};">${s.activeRate !== null ? s.activeRate+'%' : '—'}</td>
                        <td style="text-align:center;padding:3px 6px;">${s.ts1}%</td><td style="text-align:center;padding:3px 6px;">${s.s2}%</td>
                        <td style="text-align:center;padding:3px 6px;">${s.s3}%</td><td style="text-align:center;padding:3px 6px;">${s.s4}%</td>
                        <td style="text-align:center;padding:3px 6px;">${s.end}%</td>
                    </tr>`;
                }).join('')}
                </tbody>
            </table>
            </div>

            <p style="color:#94a3b8;font-size:0.82em;margin:0 0 4px;font-weight:600;">Per-Match Fuel Attribution</p>
            <p style="color:#64748b;font-size:0.78em;margin:0 0 8px;">S·eff→share% = Scorer (green). N = not scorer. ? = unscouted (assumption used). Hover T1/T2/T3 for team number.</p>
            <div style="overflow-x:auto;max-height:420px;overflow-y:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.75em;">
                <thead style="position:sticky;top:0;background:#0f172a;z-index:1;">
                <tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:3px 6px;">QM</th>
                    <th style="padding:3px 6px;">Alln</th><th style="padding:3px 6px;">Period</th>
                    <th style="padding:3px 6px;">Fuel</th>
                    <th style="padding:3px 6px;">T1</th><th style="padding:3px 6px;">T2</th><th style="padding:3px 6px;">T3</th>
                </tr></thead>
                <tbody>
                ${attrData.flatMap(({ mn, alliance, teams, teamRows, periods }) =>
                    periods.map(({ label, fuel, field }, pi) => {
                        const ac = alliance === 'red' ? '#f87171' : '#60a5fa';
                        return `<tr style="border-bottom:1px solid #0f172a;${pi%2?'background:rgba(255,255,255,0.02)':''}">
                            <td style="padding:2px 6px;color:#60a5fa;">${pi===0?`QM ${mn}`:''}</td>
                            <td style="padding:2px 4px;color:${ac};">${pi===0?alliance:''}</td>
                            <td style="padding:2px 4px;color:#94a3b8;">${label}</td>
                            <td style="text-align:center;padding:2px 4px;">${fuel}</td>
                            ${periodCells(teamRows.map((r,i)=>r===null?null:{...r,_team:teams[i]}), field)
                                .replace(/<td /g, `<td title="${teams.join(' / ')}" `)}
                        </tr>`;
                    })
                ).join('')}
                </tbody>
            </table>
            </div>
            </div>
        </details>`;

        return part1 + part2;
    },

    // ── Display fields ────────────────────────────────────────────────────────
    displayFields: [
        { group: 'Overview' },
        { label: 'Matches Scouted', key: 'matches' },
        { label: 'Scout EPA',       key: 'scoutingEPA', decimals: 1 },
        { label: 'No Shows',        key: 'noShows' },

        { group: 'Auto' },
        { label: 'Scored Fuel',        key: 'autoFuelPct',         suffix: '%' },
        { label: 'Climbed (L1)',        key: 'autoClimbPct',        suffix: '%' },
        { label: 'Climb Success',       key: 'autoClimbSuccessPct', suffix: '%' },
        { label: 'Auto Climb Pts',      key: 'autoClimbPts',        decimals: 1 },

        { group: 'Shift 1' },
        { label: 'Scorer',   key: 'shift1ScorerPct',   suffix: '%' },
        { label: 'Shuttler', key: 'shift1ShuttlerPct', suffix: '%' },
        { label: 'Defender', key: 'shift1DefenderPct', suffix: '%' },
        { label: 'Died',     key: 'shift1DiedPct',     suffix: '%' },

        { group: 'Shift 2' },
        { label: 'Scorer',   key: 'shift2ScorerPct',   suffix: '%' },
        { label: 'Shuttler', key: 'shift2ShuttlerPct', suffix: '%' },
        { label: 'Defender', key: 'shift2DefenderPct', suffix: '%' },
        { label: 'Died',     key: 'shift2DiedPct',     suffix: '%' },

        { group: 'Shift 3' },
        { label: 'Scorer',   key: 'shift3ScorerPct',   suffix: '%' },
        { label: 'Shuttler', key: 'shift3ShuttlerPct', suffix: '%' },
        { label: 'Defender', key: 'shift3DefenderPct', suffix: '%' },
        { label: 'Died',     key: 'shift3DiedPct',     suffix: '%' },

        { group: 'Shift 4' },
        { label: 'Scorer',   key: 'shift4ScorerPct',   suffix: '%' },
        { label: 'Shuttler', key: 'shift4ShuttlerPct', suffix: '%' },
        { label: 'Defender', key: 'shift4DefenderPct', suffix: '%' },
        { label: 'Died',     key: 'shift4DiedPct',     suffix: '%' },

        { group: 'Roles (Overall)' },
        { label: 'Any Shift Scored',   key: 'pctScored',   suffix: '%' },
        { label: 'Any Shift Defended', key: 'pctDefended', suffix: '%' },
        { label: 'Any Shift Died',     key: 'pctDied',     suffix: '%' },

        { group: 'Endgame' },
        { label: 'Climbed',        key: 'climbPct',       suffix: '%' },
        { label: 'Level 1',        key: 'pctL1',          suffix: '%' },
        { label: 'Level 2',        key: 'pctL2',          suffix: '%' },
        { label: 'Level 3',        key: 'pctL3',          suffix: '%' },
        { label: 'Endgame Pts',    key: 'endgamePts',     decimals: 1 },
        { label: 'Endgame Fuel',   key: 'endgameFuelPct', suffix: '%' },

        { group: 'Qualitative' },
        { label: 'Scoring Eff.',   key: 'avgScoringEff',  decimals: 1 },
        { label: 'Passing Skill',  key: 'avgPassingSkill', decimals: 1 },
        { label: 'Defense Skill',  key: 'avgDefenseSkill', decimals: 1 },
    ],

    // ── Match detail breakdown ────────────────────────────────────────────────
    matchBreakdown: {
        scoreRows: (rbd, bbd) => [
            ['Auto',         rbd.totalAutoPoints,  bbd.totalAutoPoints],
            ['Teleop',       rbd.totalTeleopPoints, bbd.totalTeleopPoints],
            ['Endgame',      (rbd.hubScore?.endgamePoints || 0) + (rbd.endGameTowerPoints || 0),
                             (bbd.hubScore?.endgamePoints || 0) + (bbd.endGameTowerPoints || 0)],
            ['Fouls Earned', rbd.foulPoints,        bbd.foulPoints],
        ],
        bonusRPFields: [
            { label: 'Energized RP',    field: 'energizedAchieved' },
            { label: 'Supercharged RP', field: 'superchargedAchieved' },
            { label: 'Traversal RP',    field: 'traversalAchieved' },
        ],
    },

    // ── Watch List: RP threshold prediction ──────────────────────────────────
    // scoreComponent keys are returned by componentScores(breakdown).
    // threshold: null → binary RP, use historical achievement rate only.
    // threshold defaults are editable per-event in the Watch List UI.
    rpThresholds: [
        { label: 'Energized RP',    rpField: 'energizedAchieved',    scoreComponent: 'allianceFuel', threshold: 190,  fuelEPAKey: 'total_fuel' },
        { label: 'Supercharged RP', rpField: 'superchargedAchieved', scoreComponent: 'allianceFuel', threshold: 360,  fuelEPAKey: 'total_fuel' },
    ],

    // Extracts scoring components from a TBA alliance breakdown object.
    // hubScore.totalCount is not a real TBA field — sum the period counts directly.
    componentScores: (bd) => {
        const hs = bd?.hubScore;
        if (!hs || hs.autoCount == null) return { allianceFuel: null };
        return {
            allianceFuel: (hs.autoCount ?? 0) + (hs.transitionCount ?? 0) +
                          (hs.shift1Count ?? 0) + (hs.shift2Count ?? 0) +
                          (hs.shift3Count ?? 0) + (hs.shift4Count ?? 0) +
                          (hs.endgameCount ?? 0),
        };
    },
};
