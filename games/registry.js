// Game config registry.
// To add a new year: import the config and add it to GAME_REGISTRY.
// Event sources (sheet URLs, column overrides) live in each game config's
// `eventSources` field — no changes to this file needed for new events.

import reefscape2025 from './2025_reefscape.js';
import rebuilt2026   from './2026_rebuilt.js';

const GAME_REGISTRY = {
    '2025': reefscape2025,
    '2026': rebuilt2026,
};

// Merged event sources from all registered game configs.
// Keyed by event key (e.g. '2025necmp2'). Each value: { url?, pitUrl?, columnOverrides?, rowAdapter? }
export const EVENT_SOURCES = Object.values(GAME_REGISTRY)
    .reduce((acc, config) => ({ ...acc, ...(config.eventSources || {}) }), {});

// Returns the game config for a given event key by extracting the year prefix.
// e.g. '2025necmp' → GAME_REGISTRY['2025']
export function getGameConfig(eventKey) {
    const year = eventKey?.match(/^(\d{4})/)?.[1];
    return GAME_REGISTRY[year] || null;
}

// Merges a game config's defaultColumns with any per-event overrides,
// producing the final canonical-field → column-header map used by processRow.
export function resolveColumnMap(gameConfig, columnOverrides = {}) {
    return { ...gameConfig.defaultColumns, ...columnOverrides };
}

// Process all raw rows for an event using the appropriate game config.
// Returns { config, observations, byTeam } or null if no game config found.
export function processScoutingData(eventKey, rawRows, columnOverrides = {}) {
    const config = getGameConfig(eventKey);
    if (!config) return null;

    const colMap = resolveColumnMap(config, columnOverrides);
    // A sheet from another scouting system is translated into this game's columns first
    // (eventSources[key].rowAdapter), so everything downstream reads one format.
    const adapt = EVENT_SOURCES[eventKey]?.rowAdapter;
    const observations = (adapt ? rawRows.map(adapt) : rawRows)
        .map(row => config.processRow(row, colMap))
        .filter(r => r.teamNumber);

    const byTeam = {};
    for (const obs of observations) {
        const t = obs.teamNumber;
        if (!byTeam[t]) byTeam[t] = [];
        byTeam[t].push(obs);
    }

    return { config, observations, byTeam };
}

// Get per-team aggregated stats for an event.
// Returns { config, teamStats: { teamNumber: aggregatedStats } } or null.
export function aggregateScoutingData(eventKey, rawRows, columnOverrides = {}) {
    const result = processScoutingData(eventKey, rawRows, columnOverrides);
    if (!result) return null;

    const { config, byTeam } = result;
    const teamStats = {};
    for (const [teamNumber, rows] of Object.entries(byTeam)) {
        teamStats[teamNumber] = config.aggregateTeam(rows);
    }
    return { config, teamStats };
}

// Build a { matchNumber: { teamNumber: processedRow } } index from all observations.
// Pass the result to fuseScoutingWithTBA as allByMatch.
export function indexObservationsByMatch(observations) {
    const index = {};
    for (const obs of observations) {
        if (!index[obs.matchNumber]) index[obs.matchNumber] = {};
        index[obs.matchNumber][obs.teamNumber] = obs;
    }
    return index;
}

// Detect whether TBA reports a counting stat cumulatively (including auto)
// or as period-only values, by scanning all available match breakdowns.
//
// Two signals (checked across all matches and both alliances):
//   cumulative evidence: tbaAuto + tbaRaw > maxPerAlliance — values can't both
//     be independent counts if they exceed the physical maximum.
//   separate evidence:   tbaRaw < tbaAuto — teleop can't be less than auto if
//     it's cumulative (cumulative must be >= auto by definition).
//
// Returns 'cumulative', 'separate', or 'unknown'.
export function detectCumulativeReportingMode(tbaMatches, teleopFuseStats) {
    let cumulativeEvidence = 0;
    let separateEvidence   = 0;

    for (const match of tbaMatches) {
        for (const breakdown of [match.redBreakdown, match.blueBreakdown]) {
            if (!breakdown) continue;
            for (const stat of teleopFuseStats) {
                if (stat.maxPerAlliance == null) continue;
                const auto   = stat.tbaAuto(breakdown) ?? 0;
                const teleop = stat.tbaRaw(breakdown)  ?? 0;
                if (auto + teleop > stat.maxPerAlliance) cumulativeEvidence++;
                if (teleop < auto)                       separateEvidence++;
            }
        }
    }

    if (cumulativeEvidence > 0 && separateEvidence === 0) return 'cumulative';
    if (separateEvidence   > 0 && cumulativeEvidence === 0) return 'separate';
    if (cumulativeEvidence > separateEvidence) return 'cumulative';
    if (separateEvidence > cumulativeEvidence) return 'separate';
    return 'unknown';
}

// Fuse per-team scouting estimates with TBA-validated alliance totals.
//
// teamNumber:    team to compute fused stats for
// observations:  processed rows for this team (from processScoutingData)
// allByMatch:    { matchNumber: { teamNumber: processedRow } } for the whole event
// tbaMatches:    TBA match records with .red/.blue and .redBreakdown/.blueBreakdown
// gameConfig:    full game config object (provides fuseStats arrays)
//
// Game config fusion arrays (all optional, default to []):
//   autoFuseStats    — share × TBA auto count; each: { key, scout, tba }
//   teleopFuseStats  — share × net teleop count (cumulative-corrected);
//                      each: { key, scout, tbaRaw, tbaAuto, maxPerAlliance }
//   splitFuseStats   — fuse total then re-split by scouting ratio;
//                      each: { autoKey, teleopKey, scoutAuto, scoutTeleop, tba }
//   robotFuseStats   — read directly by robot index; each: { key, tba(b,i), scout }
//
// Returns:
//   { available: false, fusedByMatch: {}, coverage, reportingMode }
//   { available: true,  stats: { key: fusedAvg }, fusedByMatch, coverage, reportingMode }
export function fuseScoutingWithTBA(teamNumber, observations, allByMatch, tbaMatches, gameConfig) {
    const {
        autoFuseStats   = [],
        teleopFuseStats = [],
        splitFuseStats  = [],
        robotFuseStats  = [],
    } = gameConfig;
    const teamStr = String(teamNumber);

    const reportingMode = detectCumulativeReportingMode(tbaMatches, teleopFuseStats);
    const isCumulative  = reportingMode !== 'separate'; // treat 'unknown' as cumulative (safer default)

    // Pre-compute per-team aggregates for assumption filling in fuseCustom (2026+).
    // Skipped entirely for game configs that don't define fuseCustom (e.g. 2025).
    const teamAggregates = {};
    if (gameConfig.fuseCustom) {
        const allObsByTeam = {};
        for (const matchObs of Object.values(allByMatch)) {
            for (const [tn, row] of Object.entries(matchObs)) {
                if (!allObsByTeam[tn]) allObsByTeam[tn] = [];
                allObsByTeam[tn].push(row);
            }
        }
        const tbaByMatch = {};
        for (const m of tbaMatches) tbaByMatch[m.matchNumber] = m;

        for (const [tn, rows] of Object.entries(allObsByTeam)) {
            const agg = gameConfig.aggregateTeam(rows);
            if (gameConfig.enrichAggregateWithTBA) {
                gameConfig.enrichAggregateWithTBA(tn, rows, agg, tbaByMatch);
            }
            teamAggregates[tn] = agg;
        }
    }

    const fusedByMatch = {};
    const coverage = { total: 0, withTBA: 0, withFullScouting: 0 };

    for (const obs of observations) {
        if (obs.noShow) continue;
        coverage.total++;

        const tbaMatch = tbaMatches.find(m => m.matchNumber === obs.matchNumber);
        if (!tbaMatch) continue;

        const breakdown = _getAllianceBreakdown(tbaMatch, teamStr);
        if (!breakdown) continue;
        coverage.withTBA++;

        const allianceTeams = _getAllianceTeams(tbaMatch, teamStr);
        const matchObs      = allByMatch[obs.matchNumber] || {};
        const allianceRows  = allianceTeams.map(t => matchObs[t] ?? null);
        if (allianceRows.every(r => r !== null)) coverage.withFullScouting++;

        const fused = {};

        // ── Auto: share × TBA auto count ─────────────────────────────────────
        for (const stat of autoFuseStats) {
            fused[stat.key] = _shareWeighted(stat.scout, obs, allianceRows, stat.tba(breakdown));
        }

        // ── Teleop: share × net teleop count ─────────────────────────────────
        // If TBA reports cumulatively (teleop includes auto), subtract auto to
        // get the teleop-only contribution before applying the share.
        for (const stat of teleopFuseStats) {
            const raw  = stat.tbaRaw(breakdown);
            const auto = stat.tbaAuto(breakdown) ?? 0;
            if (raw == null) { fused[stat.key] = null; continue; }
            const netTeleop = isCumulative ? Math.max(0, raw - auto) : raw;
            fused[stat.key] = _shareWeighted(stat.scout, obs, allianceRows, netTeleop);
        }

        // ── Split stats: fuse total, then re-split using scouting ratio ───────
        for (const stat of splitFuseStats) {
            const tbaTotal = stat.tba(breakdown);
            if (tbaTotal == null) { fused[stat.autoKey] = fused[stat.teleopKey] = null; continue; }

            const combinedScout = r => stat.scoutAuto(r) + stat.scoutTeleop(r);
            const fusedTotal = _shareWeighted(combinedScout, obs, allianceRows, tbaTotal);

            const scoutAuto   = stat.scoutAuto(obs);
            const scoutTeleop = stat.scoutTeleop(obs);
            const scoutTotal  = scoutAuto + scoutTeleop;
            const autoRatio   = scoutTotal > 0 ? scoutAuto / scoutTotal : 0.5;

            fused[stat.autoKey]   = fusedTotal * autoRatio;
            fused[stat.teleopKey] = fusedTotal * (1 - autoRatio);
        }

        // ── Per-robot stats: read directly by robot index, no share-weighting ─
        if (robotFuseStats.length > 0) {
            const robotIdx = _getRobotIndex(tbaMatch, teamStr);
            for (const stat of robotFuseStats) {
                const val = robotIdx != null ? stat.tba(breakdown, robotIdx) : null;
                fused[stat.key] = val ?? stat.scout(obs);
            }
        }

        // ── Game-specific custom fusion hook (e.g. per-shift allocation) ──────
        if (gameConfig.fuseCustom) {
            const custom = gameConfig.fuseCustom(obs, allianceRows, tbaMatch, teamStr,
                                                 { allianceTeams, teamAggregates });
            Object.assign(fused, custom);
        }

        fusedByMatch[obs.matchNumber] = fused;
    }

    const matchFusions = Object.values(fusedByMatch);
    if (matchFusions.length === 0) return { available: false, fusedByMatch: {}, coverage, reportingMode };

    const allKeys = [
        ...autoFuseStats.map(s => s.key),
        ...teleopFuseStats.map(s => s.key),
        ...splitFuseStats.flatMap(s => [s.autoKey, s.teleopKey]),
        ...robotFuseStats.map(s => s.key),
        ...(gameConfig.customFuseKeys ?? []),
    ];

    const stats = {};
    for (const key of allKeys) {
        const values = matchFusions.map(f => f[key]).filter(v => v != null);
        stats[key] = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
    }

    return { available: true, stats, fusedByMatch, coverage, reportingMode };
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function _shareWeighted(scoutFn, obs, allianceRows, tbaTotal) {
    if (tbaTotal == null) return null;
    const allianceScouted = allianceRows.reduce((s, r) => s + (r ? scoutFn(r) : 0), 0);
    const share = allianceScouted > 0 ? scoutFn(obs) / allianceScouted : 1 / 3;
    return share * tbaTotal;
}

function _getAllianceTeams(tbaMatch, teamStr) {
    if (tbaMatch.red?.includes(teamStr))  return tbaMatch.red;
    if (tbaMatch.blue?.includes(teamStr)) return tbaMatch.blue;
    return [];
}

function _getAllianceBreakdown(tbaMatch, teamStr) {
    if (tbaMatch.red?.includes(teamStr))  return tbaMatch.redBreakdown  ?? null;
    if (tbaMatch.blue?.includes(teamStr)) return tbaMatch.blueBreakdown ?? null;
    return null;
}

// Returns 1-based robot position index within the alliance, or null if not found.
function _getRobotIndex(tbaMatch, teamStr) {
    const ri = tbaMatch.red?.indexOf(teamStr);
    if (ri >= 0) return ri + 1;
    const bi = tbaMatch.blue?.indexOf(teamStr);
    if (bi >= 0) return bi + 1;
    return null;
}
