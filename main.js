import Dexie from 'dexie';
import Chart from 'chart.js/auto';
import { getGameConfig, resolveColumnMap, aggregateScoutingData, processScoutingData, fuseScoutingWithTBA, indexObservationsByMatch, detectCumulativeReportingMode, EVENT_SOURCES as SCOUTING_SOURCES } from './games/registry.js';
import { getTeamQuip, logFractionToTier, QUIP_TIERS } from './teamQuips.js';

// 1. DATABASE SETUP
// We use Dexie to handle larger storage (images/multiple events)
const db = new Dexie('ScoutingAppDB');
db.version(1).stores({
    teams: 'teamNumber, eventKey',
    matches: 'key, eventKey, matchNumber'
});
db.version(2).stores({
    teams: 'teamNumber, eventKey',
    matches: 'key, eventKey, matchNumber',
    tbaTeams: 'teamNumber, eventKey'
});
// v3 adds robot position tracks, produced offline by robot-tracker/ and published as
// public/tracks/<matchKey>.json. Keyed by TBA match key so it joins `matches` for free.
// Every existing table must be repeated here or Dexie drops it (see CLAUDE.md).
db.version(3).stores({
    teams: 'teamNumber, eventKey',
    matches: 'key, eventKey, matchNumber',
    tbaTeams: 'teamNumber, eventKey',
    matchTracks: 'key, eventKey'
});
window.db = db;

// 2. CONFIG & API KEYS
const TBA_BASE   = 'https://www.thebluealliance.com/api/v3';
const TBA_KEY    = import.meta.env.VITE_TBA_KEY;
const NEXUS_KEY  = import.meta.env.VITE_NEXUS_KEY || '';
const _tbaCache = new Map(); // endpoint → { etag, data }
let _tbaGotFreshData = false;  // reset per _runTBASyncs cycle
let _tbaLastFreshMs  = null;   // timestamp of last non-304 TBA response
const YT_KEY = import.meta.env.VITE_YOUTUBE_KEY || '';

// SCOUTING_SOURCES is imported as EVENT_SOURCES from ./games/registry.js above.
// To add events, edit the eventSources field in the appropriate games/ config file.

// ── NOTIFICATIONS ─────────────────────────────────────────────────────────────
const _firedNotifIds = new Set(); // prevents re-firing within a session

function fireNotif(title, body, tag, vibrate = false) {
    if (localStorage.getItem('notifEnabled') !== 'true') return;
    if (Notification.permission === 'granted') {
        new Notification(title, { body, tag, icon: '/favicon.ico' });
    }
    if (document.visibilityState !== 'visible') {
        document.title = `🔔 ${title}`;
    }
    if (vibrate && navigator.vibrate) {
        navigator.vibrate([300, 150, 300, 150, 300]);
    }
}

function updateNotifBtn() {
    const btn   = document.getElementById('notifBtn');
    const label = document.getElementById('notifBtnLabel');
    if (!btn) return;
    const enabled = localStorage.getItem('notifEnabled') === 'true';
    const blocked = Notification.permission === 'denied';
    btn.style.opacity = enabled ? '1' : '0.35';
    btn.title = blocked
        ? 'Notifications blocked — allow in browser settings'
        : enabled ? 'Notifications on (click to disable)' : 'Notifications off (click to enable)';
    if (label) label.textContent = blocked ? 'Blocked' : enabled ? 'On' : 'Off';
}

function initNotifications() {
    updateNotifBtn();
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') document.title = '1768 Scouting';
    });
}

window.toggleNotifications = async function () {
    if (!('Notification' in window)) { alert('Your browser does not support notifications.'); return; }
    const enabled = localStorage.getItem('notifEnabled') === 'true';
    if (!enabled) {
        if (Notification.permission === 'denied') {
            alert('Notifications are blocked. Allow them in your browser settings, then try again.');
            return;
        }
        if (Notification.permission !== 'granted') {
            const perm = await Notification.requestPermission();
            if (perm !== 'granted') return;
        }
        localStorage.setItem('notifEnabled', 'true');
    } else {
        localStorage.setItem('notifEnabled', 'false');
    }
    updateNotifBtn();
};


window.currentFocusedTeam = null;

// ── NEXUS INTEGRATION ─────────────────────────────────────────────────────────
// Polls a Cloudflare Worker relay (nexus-relay/worker.js) that receives
// POST webhooks from frc.nexus and stores the latest payload in KV.

let _nexusInterval       = null;
let _nexusDirectInterval = null; // direct API polling — runs independently of relay toggle
let _queueNotifInterval  = null; // periodic fallback for time-based queueing notifications
let _nexusLastKey        = sessionStorage.getItem('nexusLastKey') || null; // "label|status" dedup key
let _nexusStatusBarKey   = null; // db match key currently shown in the status bar
let nexusMatchCache      = {}; // matchKey → { status, isQueuing }

const NEXUS_DEFAULT_URL = 'https://nexus-relay.thatsa404.workers.dev';

function getNexusUrl()     { return (localStorage.getItem('nexusRelayUrl') || NEXUS_DEFAULT_URL).trim(); }
function isNexusEnabled()  { return localStorage.getItem('nexusEnabled') === 'true'; }

async function maybeAutoActivateNexus() {
    if (isNexusEnabled()) return;
    const now = Math.floor(Date.now() / 1000);
    const matches = await db.matches.toArray();
    const hasFuture = matches.some(m => (m.predictedTime ?? 0) > now && (m.redScore ?? -1) < 0);
    if (!hasFuture) return;
    localStorage.setItem('nexusEnabled', 'true');
    updateNexusUI();
    startNexusPolling();
}

function updateNexusUI() {
    const btn   = document.getElementById('nexusToggleBtn');
    const label = document.getElementById('nexusToggleLabel');
    const urlIn = document.getElementById('nexusRelayUrlInput');
    if (!btn) return;
    const enabled = isNexusEnabled();
    const hasUrl  = !!getNexusUrl();
    if (urlIn && !urlIn.value) urlIn.value = getNexusUrl();
    const keyIn = document.getElementById('nexusEventKeyOverride');
    if (keyIn && !keyIn.value) keyIn.value = localStorage.getItem('nexusEventKeyOverride') || '';
    btn.style.opacity = (enabled && hasUrl) ? '1' : '0.45';
    if (label) label.textContent = (enabled && hasUrl) ? 'Connected' : hasUrl ? 'Off' : 'No URL';
}

function _nexusStatusBadge(status) {
    if (!status) return '';
    const lc = status.toLowerCase();
    if (lc === 'now queuing') return `<span style="display:inline-block;padding:1px 5px;border-radius:3px;font-size:0.7em;font-weight:700;background:#14532d;color:#86efac;border:1px solid #16a34a;">NOW QUEUING</span>`;
    if (lc === 'on deck')     return `<span style="display:inline-block;padding:1px 5px;border-radius:3px;font-size:0.7em;font-weight:700;background:#431407;color:#fdba74;border:1px solid #c2410c;">ON DECK</span>`;
    if (lc === 'on field')    return `<span style="display:inline-block;padding:1px 5px;border-radius:3px;font-size:0.7em;font-weight:700;background:#164e63;color:#67e8f9;border:1px solid #0891b2;">ON FIELD</span>`;
    return '';
}

// Parse all Nexus event data, update predicted times + match status cache.
async function applyNexusEventData(data) {
    if (!data?.matches) return;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;

    const updates = [];
    const now = Math.floor(Date.now() / 1000);

    for (const nm of data.matches) {
        const label = (nm.label ?? '').trim();
        let dbKey = null;

        const qualMatch = label.match(/^qualification\s+(\d+)$/i);
        if (qualMatch) {
            dbKey = `${eventKey}_qm${qualMatch[1]}`;
        } else {
            const playoffMatch = label.match(/^playoff\s+(\d+)$/i);
            if (playoffMatch) dbKey = `${eventKey}_sf${playoffMatch[1]}m1`;
        }
        if (!dbKey) continue;

        const status = (nm.status ?? '').trim();
        const isQueuing = label === data.nowQueuing;
        nexusMatchCache[dbKey] = { status, isQueuing };

        // Nexus estimated time overrides predictedTime for unplayed matches.
        // Once TBA sets actualTime (post-match), that is treated as truth.
        const estMs = nm.times?.estimatedStartTime;
        if (estMs && estMs > 0) {
            const estSec = Math.floor(estMs / 1000);
            const dbMatch = await db.matches.get(dbKey);
            if (dbMatch && (dbMatch.redScore ?? -1) < 0 && !dbMatch.actualTime) {
                updates.push(db.matches.update(dbKey, { predictedTime: estSec }));
            }
        }
    }

    if (updates.length) await Promise.all(updates);

    updateNexusScheduleStatus();
    updateScheduleCountdowns();
    updateHomeBanner();
    check1768QueueNotifications();
    await _clearScoredNexusStatusBar();
}

// Hide the status bar if the match it's showing has already been scored in TBA.
async function _clearScoredNexusStatusBar() {
    if (!_nexusStatusBarKey) return;
    const m = await db.matches.get(_nexusStatusBarKey);
    if (m && (m.redScore ?? -1) >= 0) updateNexusStatusBar(null);
}

// Update status badges on unscored cells in the schedule table.
function updateNexusScheduleStatus() {
    document.querySelectorAll('[data-unscored-key]').forEach(td => {
        const key = td.dataset.unscoredKey;
        const cached = nexusMatchCache[key];
        if (!cached?.status) return;
        const badge = _nexusStatusBadge(cached.status);
        if (!badge) return;
        // Replace or update the badge span, preserving the data-unscored-key attribute
        let span = td.querySelector('.nexus-match-badge');
        if (!span) {
            span = document.createElement('span');
            span.className = 'nexus-match-badge';
            span.style.display = 'block';
            td.innerHTML = '';
            td.appendChild(span);
        }
        span.innerHTML = badge;
    });
}

function startNexusPolling() {
    if (_nexusInterval) clearInterval(_nexusInterval);
    pollNexus();
    _nexusInterval = setInterval(pollNexus, 20_000);
}

function stopNexusPolling() {
    if (_nexusInterval) { clearInterval(_nexusInterval); _nexusInterval = null; }
    updateNexusStatusBar(null);
}

async function pollNexus() {
    // ── Relay worker (1768-focused, notifications only) ────────────────────────
    const url = getNexusUrl();
    if (!url) return;
    const now = Math.floor(Date.now() / 1000);
    const allMatches = await db.matches.toArray();
    const our1768 = allMatches.filter(m => m.red?.includes('1768') || m.blue?.includes('1768'));
    const soonUnplayed = our1768.length === 0 || our1768.some(m =>
        (m.redScore ?? -1) < 0 &&
        (m.predictedTime ?? 0) > now &&
        (m.predictedTime ?? 0) <= now + 3600
    );
    if (!soonUnplayed) return;
    try {
        const resp = await fetch(url, { cache: 'no-store' });
        if (resp.status === 204) return;
        if (!resp.ok) return;
        const data = await resp.json();
        handleNexusPayload(data);
    } catch { /* network error — silently ignore */ }
}

// Direct Nexus API polling — timing & status for all matches, independent of relay toggle.
async function pollNexusDirect() {
    if (!NEXUS_KEY) return;
    const allMatches = await db.matches.toArray();
    if (!allMatches.some(m => (m.redScore ?? -1) < 0)) return;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;
    const data = await fetchNexusLiveStatus(eventKey);
    if (data) await applyNexusEventData(data);
}

function startNexusDirectPolling() {
    if (!NEXUS_KEY) return;
    if (_nexusDirectInterval) clearInterval(_nexusDirectInterval);
    pollNexusDirect();
    _nexusDirectInterval = setInterval(pollNexusDirect, 20_000);
    // 30s fallback for time-based queueing notifications (covers when neither Nexus
    // nor TBA sync has fired recently — e.g. auto-sync disabled, no NEXUS_KEY)
    if (_queueNotifInterval) clearInterval(_queueNotifInterval);
    _queueNotifInterval = setInterval(check1768QueueNotifications, 30_000);
}

function handleNexusPayload(data) {
    const match = data?.match;
    if (!match) return;
    const label  = (match.label  ?? '').trim();
    const status = (match.status ?? '').trim();
    if (!label && !status) return;

    // Notify only on status change
    const key = `${label}|${status}`;
    if (key === _nexusLastKey) return;
    _nexusLastKey = key;
    sessionStorage.setItem('nexusLastKey', key);

    // Write relay data into nexusMatchCache so check1768QueueNotifications can read it
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    let dbKey = null;
    if (eventKey) {
        const qualM = label.match(/^qualification\s+(\d+)$/i);
        const playM = label.match(/^playoff\s+(\d+)$/i);
        dbKey = qualM ? `${eventKey}_qm${qualM[1]}`
              : playM ? `${eventKey}_sf${playM[1]}m1`
              : null;
        if (dbKey) nexusMatchCache[dbKey] = { ...(nexusMatchCache[dbKey] || {}), status };
    }

    updateNexusStatusBar({ label, status }, dbKey);

    check1768QueueNotifications();

    const lc = status.toLowerCase();
    if (lc.includes('result') || lc.includes('posted')) {
        fireNotif(`📡 Results — ${label}`, status, `nexus-${key}`);
    }
}

// Checks whether team 1768 should receive a queueing-stage notification for their
// next match, using (in priority order): Nexus status cache, match-completion position,
// or a 25-minute time-based fallback.
async function check1768QueueNotifications() {
    if (localStorage.getItem('notifEnabled') !== 'true') return;

    const allMatches = await db.matches.orderBy('matchNumber').toArray();
    const qualMatches = allMatches.filter(m => !m.compLevel || m.compLevel === 'qm');

    const next = qualMatches.find(m =>
        (m.red?.includes('1768') || m.blue?.includes('1768')) &&
        (m.redScore ?? -1) < 0
    );
    if (!next) return;

    const matchKey    = next.key;
    const matchNum    = next.matchNumber;
    const allianceStr = next.red?.includes('1768') ? 'Red alliance' : 'Blue alliance';
    const matchIdx    = qualMatches.findIndex(m => m.key === matchKey);

    const idQueued  = `1768-queued-${matchKey}`;
    const idOnDeck  = `1768-ondeck-${matchKey}`;
    const idOnField = `1768-onfield-${matchKey}`;

    // Determine the highest status level we should be at right now.
    // Priority: Nexus cache → match completions → time fallback
    let level = null;

    const nexusStatus = (nexusMatchCache[matchKey]?.status ?? '').toLowerCase();
    if (nexusStatus === 'on field' || nexusStatus === 'now on field') {
        level = 'onfield';
    } else if (nexusStatus === 'on deck') {
        level = 'ondeck';
    } else if (nexusStatus === 'now queuing' || nexusStatus === 'queued') {
        level = 'queued';
    } else {
        // Match-completion-based: N-1 done → on field, N-2 done → on deck, N-3 done → queued
        const isDone = m => (m?.redScore ?? -1) >= 0;
        const prev1 = matchIdx > 0 ? qualMatches[matchIdx - 1] : null;
        const prev2 = matchIdx > 1 ? qualMatches[matchIdx - 2] : null;
        const prev3 = matchIdx > 2 ? qualMatches[matchIdx - 3] : null;

        if (isDone(prev1))      level = 'onfield';
        else if (isDone(prev2)) level = 'ondeck';
        else if (isDone(prev3)) level = 'queued';
        else {
            // Time fallback: fire "queued" when ≤25 min to estimated start
            const now = Math.floor(Date.now() / 1000);
            const est = next.predictedTime ?? 0;
            if (est > 0 && (est - now) > 0 && (est - now) <= 25 * 60) level = 'queued';
        }
    }

    if (!level) return;

    // Fire the highest-priority un-fired notification.
    // Higher levels (onfield > ondeck > queued) are independent — each fires once.
    if (level === 'onfield' && !_firedNotifIds.has(idOnField)) {
        _firedNotifIds.add(idOnField);
        fireNotif(`🟢 On Field — QM ${matchNum}`, `1768 (${allianceStr}) — head to the field now!`, idOnField, true);
    }
    if ((level === 'ondeck' || level === 'onfield') && !_firedNotifIds.has(idOnDeck)) {
        _firedNotifIds.add(idOnDeck);
        fireNotif(`🟡 On Deck — QM ${matchNum}`, `1768 (${allianceStr}) — next in queue`, idOnDeck, true);
    }
    if (!_firedNotifIds.has(idQueued)) {
        _firedNotifIds.add(idQueued);
        fireNotif(`🔵 Queued — QM ${matchNum}`, `1768 (${allianceStr}) — head to queuing`, idQueued, true);
    }
}

window.updateNexusStatusBar = function updateNexusStatusBar(data, matchKey) {
    const bar = document.getElementById('nexus-status-bar');
    if (!bar) return;
    if (!data) { bar.style.display = 'none'; _nexusStatusBarKey = null; return; }

    const { label, status } = data;
    const lc = status.toLowerCase();
    let bg = '#1a2332', border = '#334155', textColor = '#94a3b8';
    if (lc === 'on deck')                              { bg = '#431407'; border = '#c2410c'; textColor = '#fdba74'; }
    else if (lc === 'on field')                        { bg = '#14532d'; border = '#16a34a'; textColor = '#86efac'; }
    else if (lc.includes('result') || lc.includes('posted')) { bg = '#1e1b4b'; border = '#4f46e5'; textColor = '#a5b4fc'; }

    bar.style.cssText = `display:flex; background:${bg}; border-color:${border};`;
    const dotEl   = bar.querySelector('.nexus-dot');
    const labelEl = bar.querySelector('.nexus-label');
    if (dotEl)   dotEl.style.background = textColor;
    if (labelEl) { labelEl.style.color = textColor; labelEl.textContent = `${label} · ${status}`; }
    const dismissEl = bar.querySelector('.nexus-dismiss');
    if (dismissEl) dismissEl.style.color = textColor;

    _nexusStatusBarKey = matchKey || null;

    // Auto-clear after results are posted — no need to persist a terminal status
    if (lc.includes('result') || lc.includes('posted')) {
        setTimeout(() => updateNexusStatusBar(null), 20_000);
    }
}

window.toggleNexus = function () {
    if (!isNexusEnabled() && !getNexusUrl()) { alert('Enter the Relay URL first.'); return; }
    const nowEnabled = !isNexusEnabled();
    localStorage.setItem('nexusEnabled', String(nowEnabled));
    updateNexusUI();
    if (nowEnabled) startNexusPolling(); else stopNexusPolling();
};

window.saveNexusUrl = function () {
    const url = document.getElementById('nexusRelayUrlInput')?.value.trim();
    if (url) localStorage.setItem('nexusRelayUrl', url);
    updateNexusUI();
};

function initNexusIntegration() {
    updateNexusUI();
    if (isNexusEnabled() && getNexusUrl()) startNexusPolling();
    startNexusDirectPolling();
    // Always start the queueing-notification fallback timer, even without NEXUS_KEY
    if (!NEXUS_KEY) {
        if (_queueNotifInterval) clearInterval(_queueNotifInterval);
        _queueNotifInterval = setInterval(check1768QueueNotifications, 30_000);
    }
}

// ── Nexus REST API ─────────────────────────────────────────────────────────────
// Direct pull (not webhook) — fetches full event status including all matches.

async function fetchNexusLiveStatus(eventKey) {
    if (!NEXUS_KEY || !eventKey) return null;
    try {
        const resp = await fetch(`https://frc.nexus/api/v1/event/${eventKey}`, {
            headers: { 'Nexus-Api-Key': NEXUS_KEY },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        localStorage.setItem(`nexusEventData_${eventKey}`, JSON.stringify({ data, ts: Date.now() }));
        return data;
    } catch (e) {
        console.warn('[Nexus] fetchLiveStatus failed:', e.message);
        return null;
    }
}

function updateAppEventKey(eventKey) {
    const subtitle = document.getElementById('headerSubtitle');
    const selectBtn = document.getElementById('selectEventBtn');
    if (eventKey) {
        if (subtitle) subtitle.textContent = eventKey;
        document.title = `Nashoba Robotics — ${eventKey}`;
        if (selectBtn) selectBtn.textContent = 'Switch Event';
    } else {
        if (subtitle) subtitle.textContent = 'Event Hub';
        document.title = 'Nashoba Robotics — Event Hub';
        if (selectBtn) selectBtn.textContent = 'Select Event';
    }
}

// ── EVENT SELECTOR ─────────────────────────────────────────────────────────────

function showEventSelector() {
    const overlay = document.getElementById('event-selector-overlay');
    if (!overlay) return;

    const localReg = JSON.parse(localStorage.getItem('localArchiveRegistry') || '{}');

    const archiveBadge = type => type === 'config'
        ? `<span style="color:#f59e0b;font-size:0.72em;font-weight:700;margin-top:2px;">Config archive</span>`
        : type === 'full'
        ? `<span style="color:#60a5fa;font-size:0.72em;font-weight:700;margin-top:2px;">Full archive</span>`
        : `<span style="color:#94a3b8;font-size:0.72em;margin-top:2px;">Archive</span>`;

    const makeCard = (key, sublabel, archiveType) => `
        <button class="event-preset-card" onclick="window.selectPresetEvent('${key}')">
            <span class="preset-key">${key}</span>
            ${sublabel ? `<span class="preset-sublabel">${sublabel}</span>` : ''}
            ${archiveType ? archiveBadge(archiveType) : ''}
        </button>`;

    // Build preset cards grouped by year, newest first
    const byYear = {};
    for (const [key, src] of Object.entries(SCOUTING_SOURCES)) {
        const year = key.slice(0, 4);
        if (!byYear[year]) byYear[year] = [];
        byYear[year].push({ key, src });
    }
    const years = Object.keys(byYear).sort((a, b) => b - a);

    // Registry-only events: saved/loaded archives not in SCOUTING_SOURCES
    const registryOnlyKeys = Object.keys(localReg).filter(k => !SCOUTING_SOURCES[k]);

    const makeSection = (label, cards, isOpen) =>
        `<details class="event-year-section"${isOpen ? ' open' : ''}>
            <summary class="event-selector-year-label"><span>${label}</span><span class="year-arrow">▾</span></summary>
            <div class="year-cards">${cards}</div>
        </details>`;

    const presetsEl = document.getElementById('event-selector-presets');
    if (presetsEl) {
        const sourceSections = years.map((year, i) => {
            const cards = byYear[year].map(({ key, src }) => {
                const gameConfig = getGameConfig(key);
                const sublabel = [src.label, gameConfig ? `${gameConfig.name} ${year}` : null]
                    .filter(Boolean).join(' · ');
                return makeCard(key, sublabel, localReg[key]?.archiveType);
            }).join('');
            return makeSection(year, cards, i === 0);
        }).join('');

        const archiveSection = registryOnlyKeys.length ? (() => {
            const cards = registryOnlyKeys.map(key => {
                const gameConfig = getGameConfig(key);
                const sublabel = gameConfig ? `${gameConfig.name} ${gameConfig.year}` : null;
                return makeCard(key, sublabel, localReg[key]?.archiveType);
            }).join('');
            return makeSection('My Archives', cards, true);
        })() : '';

        presetsEl.innerHTML = archiveSection + sourceSections;
    }

    // Show "Continue with [key]" if a key is already saved
    const savedKey = localStorage.getItem('lastEventKey');
    const continueDiv = document.getElementById('event-selector-continue');
    const continueBtn = document.getElementById('event-selector-continue-btn');
    if (savedKey && continueDiv && continueBtn) {
        continueBtn.textContent = `Continue with ${savedKey}`;
        continueDiv.style.display = 'block';
    } else if (continueDiv) {
        continueDiv.style.display = 'none';
    }

    overlay.style.display = 'flex';
}

window.closeEventSelector = function () {
    const overlay = document.getElementById('event-selector-overlay');
    if (overlay) overlay.style.display = 'none';
};

window.selectPresetEvent = async function (key) {
    const input = document.getElementById('eventKeyInput');
    const existingKey = input?.value.trim().toLowerCase();
    if (existingKey && existingKey !== key) {
        await _silentClearEvent(existingKey);
        nexusMatchCache = {};
    }

    if (input) input.value = key;
    localStorage.setItem('lastEventKey', key);
    updateAppEventKey(key);
    updateOBEStatus(key);
    renderScoutingSection();
    window.closeEventSelector();

    // Auto-load archive if one exists for this event
    const hint = document.getElementById('archiveHint');
    if (hint) hint.innerHTML = `<span style="color:#64748b;font-size:0.82em;">Checking for archive…</span>`;
    await window.loadEventArchive(key);
};

window.selectCustomEvent = function () {
    const input = document.getElementById('event-selector-custom-input');
    const key = input?.value.trim().toLowerCase();
    if (!key) return;
    window.selectPresetEvent(key);
};

window.openEventSelector = function () { showEventSelector(); };

// 3. UTILITY FUNCTIONS
const sleep = (ms) => new Promise(res => setTimeout(res, ms));

function getTeamNotes() {
    try { return JSON.parse(localStorage.getItem('teamNotes') || '{}'); } catch { return {}; }
}
// Returns { matchKey: { text, qm } } for one team; migrates legacy formats transparently.
function getTeamNotesMap(teamNumber) {
    const raw = getTeamNotes()[String(teamNumber)];
    if (!raw) return {};
    if (typeof raw === 'string') return { general: { text: raw, qm: null } };
    // Old single-note format: { text, qm }
    if (typeof raw.text === 'string') {
        const key = raw.qm != null ? String(raw.qm) : 'general';
        return { [key]: { text: raw.text, qm: raw.qm ?? null } };
    }
    return raw;
}
// Returns { text, qm } for a specific match context (qm = number or null = general).
function getTeamNote(teamNumber, qm = null) {
    const map = getTeamNotesMap(teamNumber);
    return map[qm != null ? String(qm) : 'general'] || null;
}
// Formatted string for a specific match context.
function noteDisplayText(teamNumber, qm = null) {
    const note = getTeamNote(teamNumber, qm);
    if (!note?.text) return '';
    return note.qm != null ? `(QM ${note.qm}) ${note.text}` : note.text;
}
// All note lines for a team, sorted by match number (for overview tab).
function allNoteDisplayLines(teamNumber) {
    return Object.values(getTeamNotesMap(teamNumber))
        .filter(n => n.text)
        .sort((a, b) => (a.qm ?? Infinity) - (b.qm ?? Infinity))
        .map(n => n.qm != null ? `(QM ${n.qm}) ${n.text}` : n.text);
}
function _getEventNotes(eventKey) {
    if (!eventKey) return [];
    try { return JSON.parse(localStorage.getItem(`eventNotes_${eventKey}`) || '[]'); } catch { return []; }
}
function _saveEventNote(note, eventKey) {
    if (!eventKey) return;
    const notes = _getEventNotes(eventKey);
    const idx = notes.findIndex(n => n.id === note.id);
    if (idx >= 0) notes[idx] = note; else notes.push(note);
    localStorage.setItem(`eventNotes_${eventKey}`, JSON.stringify(notes));
}
function _deleteEventNote(id, eventKey) {
    if (!eventKey) return;
    localStorage.setItem(`eventNotes_${eventKey}`, JSON.stringify(_getEventNotes(eventKey).filter(n => n.id !== id)));
}

function saveTeamNote(teamNumber, text, qm = null) {
    const all = getTeamNotes();
    const teamKey = String(teamNumber);
    const map = getTeamNotesMap(teamNumber);
    const matchKey = qm != null ? String(qm) : 'general';
    if (text.trim()) map[matchKey] = { text: text.trim(), qm: qm ?? null };
    else delete map[matchKey];
    if (Object.keys(map).length === 0) delete all[teamKey];
    else all[teamKey] = map;
    localStorage.setItem('teamNotes', JSON.stringify(all));
}

// ── Scouting data helpers ─────────────────────────────────────────────────────

// Returns the source URL for an event: checks SCOUTING_SOURCES first, then a
// per-device localStorage override (saved when the user pastes a URL manually).
function getScoutingSource(eventKey) {
    if (!eventKey) return null;
    const entry = SCOUTING_SOURCES[eventKey];
    if (entry?.url) return entry.url;
    if (typeof entry === 'string') return entry; // legacy string form
    return localStorage.getItem(`scoutingSheetUrl_${eventKey}`) || null;
}

function getPitSource(eventKey) {
    if (!eventKey) return null;
    return SCOUTING_SOURCES[eventKey]?.pitUrl || localStorage.getItem(`pitSheetUrl_${eventKey}`) || null;
}

// Returns any per-event column name overrides defined in SCOUTING_SOURCES.
function getScoutingColumnOverrides(eventKey) {
    return SCOUTING_SOURCES[eventKey]?.columnOverrides || {};
}

// Returns aggregated scouting stats for all teams at an event, or null if no
// raw data or no game config exists for that event key.
function getScoutingStats(eventKey) {
    const raw = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!raw) return null;
    return aggregateScoutingData(eventKey, JSON.parse(raw), getScoutingColumnOverrides(eventKey));
}

// Returns scout comments for one team at an event, sorted by match number.
// Each entry: { matchNumber, text }
function getScoutingComments(teamNumber, eventKey) {
    if (!eventKey) return [];
    const raw = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!raw) return [];
    const result = processScoutingData(eventKey, JSON.parse(raw), getScoutingColumnOverrides(eventKey));
    if (!result) return [];
    return (result.byTeam[String(teamNumber)] || [])
        .filter(r => r.comments)
        .map(r => ({ matchNumber: r.matchNumber, text: r.comments }))
        .sort((a, b) => a.matchNumber - b.matchNumber);
}

// RFC-4180-compliant CSV parser. Handles quoted fields containing commas/newlines.
function parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter(l => l.trim());
    if (lines.length < 2) return [];
    function parseLine(line) {
        const fields = [];
        let cur = '', inQ = false;
        for (let i = 0; i < line.length; i++) {
            const c = line[i];
            if (c === '"') { if (inQ && line[i + 1] === '"') { cur += '"'; i++; } else inQ = !inQ; }
            else if (c === ',' && !inQ) { fields.push(cur); cur = ''; }
            else cur += c;
        }
        fields.push(cur);
        return fields;
    }
    const headers = parseLine(lines[0]).map(h => h.trim());
    return lines.slice(1).map(line => {
        const vals = parseLine(line);
        const obj = {};
        headers.forEach((h, i) => { obj[h] = (vals[i] ?? '').trim(); });
        return obj;
    });
}

const TIER_BG  = { S: '#f59e0b', A: '#4ade80', B: '#a855f7', C: '#64748b' };
const TIER_STYLE = {
    S: { color: '#f59e0b', bg: 'rgba(245,158,11,0.08)' },
    A: { color: '#4ade80', bg: 'rgba(74,222,128,0.07)' },
    B: { color: '#a855f7', bg: 'rgba(168,85,247,0.06)' },
    C: { color: '#64748b', bg: 'rgba(100,116,139,0.03)' },
};
function tierBadge(tier, extraClass = '', label = '') {
    const bg = TIER_BG[tier] || TIER_BG.C;
    const fg = tier === 'C' ? '#f8fafc' : '#0f172a';
    const text = label ? `${tier} ${label}` : tier;
    return `<span class="${extraClass}" style="display:inline-block;padding:1px 7px;border-radius:4px;font-size:0.72em;font-weight:800;background:${bg};color:${fg};letter-spacing:0.06em;vertical-align:middle;">${text}</span>`;
}
const OWN_TEAM = '1768';
function ownStar(tn) {
    return String(tn) === OWN_TEAM
        ? `<span style="color:#fbbf24;font-size:0.65em;vertical-align:middle;margin-left:3px;line-height:1;">★</span>`
        : '';
}

function epaRankTier(allTeams, myVal, fieldFn) {
    const vals = allTeams.map(fieldFn).filter(v => v != null && !isNaN(v)).sort((a, b) => b - a);
    const rank = vals.findIndex(v => v <= myVal + 0.001);
    const r = rank < 0 ? vals.length : rank;
    return r < 8 ? 'S' : r < 20 ? 'A' : r < 32 ? 'B' : 'C';
}


async function fetchTBA(endpoint) {
    const cached = _tbaCache.get(endpoint);
    const headers = { 'X-TBA-Auth-Key': TBA_KEY };
    if (cached?.etag) headers['If-None-Match'] = cached.etag;

    const response = await fetch(`${TBA_BASE}${endpoint}`, { headers });

    if (response.status === 304 && cached) return cached.data;
    if (!response.ok) throw new Error(`TBA ${response.status}: ${endpoint}`);

    const etag = response.headers.get('ETag');
    const data = await response.json();
    if (etag) _tbaCache.set(endpoint, { etag, data });
    _tbaGotFreshData = true;
    _tbaLastFreshMs  = Date.now();
    return data;
}

window.fetchSchedule = async function (eventKey) {
    const statusDiv = document.getElementById('status');
    try {
        const response = await fetch(`https://www.thebluealliance.com/api/v3/event/${eventKey}/matches/simple`, {
            headers: { 'X-TBA-Auth-Key': TBA_KEY }
        });
        const matches = await response.json();

        // Filter for Qualifications only and sort by match number
        const qualMatches = matches
            .filter(m => m.comp_level === 'qm')
            .sort((a, b) => a.match_number - b.match_number);

        // Save to Dexie
        await db.matches.bulkPut(qualMatches.map(m => ({
            key: m.key,
            eventKey: eventKey,
            matchNumber: m.match_number,
            red: m.alliances.red.team_keys.map(t => t.replace('frc', '')),
            blue: m.alliances.blue.team_keys.map(t => t.replace('frc', '')),
            redScore: m.alliances.red.score,
            blueScore: m.alliances.blue.score,
            predictedTime: m.predicted_time || null,
            actualTime: m.actual_time || null,
            videos: (m.videos || []).filter(v => v.type === 'youtube').map(v => v.key),
        })));

        console.log(`Loaded ${qualMatches.length} matches into schedule.`);
    } catch (err) {
        console.error("TBA Fetch Error:", err);
    }
};

window.displaySchedule = async function () {
    const body = document.getElementById('scheduleBody');
    if (!body) return;

    const matches = (await db.matches.orderBy('matchNumber').toArray())
        .filter(m => !m.compLevel || m.compLevel === 'qm');
    const isMobile = document.body.classList.contains('mobile-ui');
    const thead = document.querySelector('#scheduleTable thead');

    if (matches.length === 0) {
        body.innerHTML = `<tr><td colspan="${isMobile ? 5 : 8}" style="text-align:center; padding:20px;">No matches cached. Hit "Sync Schedule" on the Home tab.</td></tr>`;
        return;
    }

    // Rebuild thead to match layout
    if (isMobile) {
        thead.innerHTML = `<tr>
            <th style="text-align:center;">Match</th>
            <th>1</th><th>2</th><th>3</th>
            <th style="text-align:center;min-width:3.2rem;">Score</th>
        </tr>`;
    } else {
        thead.innerHTML = `
            <tr>
                <th rowspan="2">Match</th>
                <th colspan="3" class="red-header">Red Alliance</th>
                <th colspan="3" class="blue-header">Blue Alliance</th>
                <th rowspan="2">Result</th>
            </tr>
            <tr>
                <th class="red-header">1</th><th class="red-header">2</th><th class="red-header">3</th>
                <th class="blue-header">1</th><th class="blue-header">2</th><th class="blue-header">3</th>
            </tr>`;
    }

    body.innerHTML = '';

    const teamCell = (team, cls) =>
        `<td class="${cls}" data-team="${team}" onclick="highlightTeam('${team}')" style="cursor:pointer;"><strong>${team}</strong></td>`;

    const nowTs = Math.floor(Date.now() / 1000);

    matches.forEach(m => {
        const redWon  = m.redScore > -1 && m.redScore > m.blueScore;
        const blueWon = m.redScore > -1 && m.blueScore > m.redScore;
        const hasVideo = m.videos && m.videos.length > 0;
        const matchPassed = m.redScore <= -1 && m.predictedTime && m.predictedTime < nowTs;
        const videoIcon = hasVideo
            ? `<svg style="width:12px;height:12px;vertical-align:middle;margin-left:4px;color:#f59e0b;flex-shrink:0;" viewBox="0 0 20 20" fill="currentColor"><circle cx="10" cy="10" r="10"/><polygon points="8,6 15,10 8,14" fill="#080d16"/></svg>`
            : '';

        if (isMobile) {
            const redRow  = document.createElement('tr');
            const blueRow = document.createElement('tr');
            redRow.dataset.matchStart = 'true';
            redRow.dataset.teams = [...m.red, ...m.blue].join(',');

            const redCells  = m.red.map(t  => teamCell(t,  'red-cell')).join('');
            const blueCells = m.blue.map(t => teamCell(t, 'blue-cell')).join('');

            const scoreCell = m.redScore > -1
                ? `<td rowspan="2" onclick="viewMatchDetail('${m.key}')"
                       style="cursor:pointer;border-left:2px solid #334155;vertical-align:middle;text-align:center;white-space:nowrap;padding:4px 8px;min-width:2.8rem;">
                       <div style="color:${redWon  ? '#4ade80' : '#94a3b8'};font-weight:${redWon  ? '800' : 'normal'};white-space:nowrap;">${m.redScore}</div>
                       <div style="color:#334155;font-size:0.65em;line-height:1.4;white-space:nowrap;">${hasVideo ? videoIcon : '—'}</div>
                       <div style="color:${blueWon ? '#4ade80' : '#94a3b8'};font-weight:${blueWon ? '800' : 'normal'};white-space:nowrap;">${m.blueScore}</div>
                   </td>`
                : matchPassed
                    ? `<td rowspan="2" onclick="viewMatchDetail('${m.key}')"
                           style="cursor:pointer;border-left:2px solid #334155;vertical-align:middle;text-align:center;white-space:nowrap;min-width:2.8rem;">⏩</td>`
                    : (() => {
                        const nb = _nexusStatusBadge(nexusMatchCache[m.key]?.status);
                        const inner = nb ? `<span class="nexus-match-badge" style="display:block;">${nb}</span>` : '—';
                        return `<td rowspan="2" data-unscored-key="${m.key}" data-unscored-time="${m.predictedTime}" style="color:#64748b;font-style:italic;border-left:2px solid #334155;vertical-align:middle;text-align:center;white-space:nowrap;min-width:2.8rem;">${inner}</td>`;
                    })();

            const mobileCountdown = m.redScore <= -1 && m.predictedTime
                ? `<div data-predicted-time="${m.predictedTime}" data-match-key="${m.key}" style="font-size:0.65em;color:#64748b;margin-top:2px;"></div>`
                : '';
            redRow.innerHTML = `
                <td class="match-number" rowspan="2" onclick="viewMatchPrep('${m.key}')"
                    style="cursor:pointer;text-decoration:underline;color:#3b82f6;vertical-align:middle;text-align:center;white-space:nowrap;padding:4px 6px;">
                    QM ${m.matchNumber}${mobileCountdown}
                </td>
                ${redCells}${scoreCell}`;
            blueRow.innerHTML = blueCells;

            body.appendChild(redRow);
            body.appendChild(blueRow);
        } else {
            const row = document.createElement('tr');
            row.dataset.matchStart = 'true';
            row.dataset.teams = [...m.red, ...m.blue].join(',');
            const redCells  = m.red.map(t  => teamCell(t,  'red-cell')).join('');
            const blueCells = m.blue.map(t => teamCell(t, 'blue-cell')).join('');
            const resultCell = m.redScore > -1
                ? `<td onclick="viewMatchDetail('${m.key}')" style="cursor:pointer;border-left:2px solid #334155;white-space:nowrap;">
                       <span style="color:${redWon  ? '#4ade80' : '#94a3b8'};font-weight:${redWon  ? 'bold' : 'normal'}">${m.redScore}</span>
                       <span style="color:#475569;"> – </span>
                       <span style="color:${blueWon ? '#4ade80' : '#94a3b8'};font-weight:${blueWon ? 'bold' : 'normal'}">${m.blueScore}</span>
                       ${videoIcon}
                   </td>`
                : matchPassed
                    ? `<td onclick="viewMatchDetail('${m.key}')" style="cursor:pointer;border-left:2px solid #334155;text-align:center;color:#64748b;">⏩</td>`
                    : (() => {
                        const nb = _nexusStatusBadge(nexusMatchCache[m.key]?.status);
                        const inner = nb ? `<span class="nexus-match-badge" style="display:block;">${nb}</span>` : 'Upcoming';
                        return `<td data-unscored-key="${m.key}" data-unscored-time="${m.predictedTime}" style="color:#64748b;font-style:italic;border-left:2px solid #334155;">${inner}</td>`;
                    })();
            const desktopCountdown = m.redScore <= -1 && m.predictedTime
                ? `<div data-predicted-time="${m.predictedTime}" data-match-key="${m.key}" style="font-size:0.65em;color:#64748b;margin-top:2px;"></div>`
                : '';
            row.innerHTML = `
                <td class="match-number" onclick="viewMatchPrep('${m.key}')"
                    style="cursor:pointer;text-decoration:underline;color:#3b82f6;">QM ${m.matchNumber}${desktopCountdown}</td>
                ${redCells}${blueCells}${resultCell}`;
            body.appendChild(row);
        }
    });
    applyScheduleFilter();
    clearInterval(_scheduleCountdownInterval);
    updateScheduleCountdowns();
    _scheduleCountdownInterval = setInterval(updateScheduleCountdowns, 1_000);
    updateHomeBanner();
};

let prepChartInstance = null; // Global variable to handle chart destruction
let breakdownRadarChartRed  = null;
let breakdownRadarChartBlue = null;

const rightPanelHistory = [];

let _scheduleCountdownInterval = null;

// ── MATCH COUNTDOWN BANNER (team 1768) ────────────────────────────────────────
let _bannerMatchNum  = null;
let _bannerMatchTime = null;
let _bannerAlliance  = null; // 'red' | 'blue'

function updateBannerTick() {
    const textEl = document.getElementById('match-countdown-text');
    const banner = document.getElementById('match-countdown-banner');
    if (!textEl || !_bannerMatchTime) return;

    const now = Math.floor(Date.now() / 1000);
    const remaining = _bannerMatchTime - now;

    if (remaining < 0) {
        // Predicted time passed — clear and advance to the next future match
        _bannerMatchTime = null;
        updateHomeBanner();
        return;
    }

    let countdown;
    if (remaining >= 3600) {
        const h = Math.floor(remaining / 3600);
        const m = Math.floor((remaining % 3600) / 60);
        countdown = `${h}h ${m}m`;
    } else {
        const m = Math.floor(remaining / 60);
        const s = String(remaining % 60).padStart(2, '0');
        countdown = m > 0 ? `${m}m ${s}s` : `${s}s`;
    }

    textEl.textContent = `QM ${_bannerMatchNum}  ·  ${countdown}`;
    banner.style.display = 'flex';
}

async function updateHomeBanner() {
    const banner = document.getElementById('match-countdown-banner');
    const allianceEl = document.getElementById('match-countdown-alliance');
    if (!banner) return;

    const now = Math.floor(Date.now() / 1000);
    const matches = await db.matches.orderBy('matchNumber').toArray();
    const next = matches.find(m =>
        m.redScore <= -1 &&
        m.predictedTime &&
        m.predictedTime > now &&
        (m.red?.includes('1768') || m.blue?.includes('1768'))
    );

    if (!next) {
        banner.style.display = 'none';
        _bannerMatchTime = null;
        return;
    }

    _bannerMatchNum  = next.matchNumber;
    _bannerMatchTime = next.predictedTime;
    _bannerAlliance  = next.red?.includes('1768') ? 'red' : 'blue';

    if (allianceEl) {
        allianceEl.textContent = _bannerAlliance === 'red' ? 'Red' : 'Blue';
        allianceEl.style.color      = _bannerAlliance === 'red' ? '#fca5a5' : '#93c5fd';
        allianceEl.style.background = _bannerAlliance === 'red' ? '#7f1d1d55' : '#1e3a5f55';
    }

    updateBannerTick();
}

function updateScheduleCountdowns() {
    const now = Math.floor(Date.now() / 1000);
    document.querySelectorAll('[data-predicted-time]').forEach(el => {
        const predicted = parseInt(el.dataset.predictedTime, 10);
        const remaining = predicted - now;
        if (remaining < 0) {
            el.textContent = '';
        } else if (remaining >= 3600) {
            const t = new Date(predicted * 1000);
            el.textContent = '~' + t.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
        } else if (remaining >= 300) {
            el.textContent = `${Math.floor(remaining / 60)}m`;
        } else {
            const m = Math.floor(remaining / 60);
            const s = String(remaining % 60).padStart(2, '0');
            el.textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
        }
    });
    // Flip "Upcoming / —" cells to ⏩ once their predicted time passes
    document.querySelectorAll('[data-unscored-key]').forEach(td => {
        const t = parseInt(td.dataset.unscoredTime, 10);
        if (t && t < now) {
            const key = td.dataset.unscoredKey;
            td.removeAttribute('data-unscored-key');
            td.removeAttribute('data-unscored-time');
            td.style.cssText += ';cursor:pointer;text-align:center;color:#64748b;font-style:normal;';
            td.textContent = '⏩';
            td.onclick = () => window.viewMatchDetail(key);
        }
    });
}

let currentPrepMatch = null;

window.viewMatchPrep = async function (matchKey) {
    const match = await db.matches.get(matchKey);
    if (!match) return;
    currentPrepMatch = match;

    document.getElementById('prepMatchLabel').innerText = `Match Prep: Qual ${match.matchNumber}`;
    const isSplit = document.body.classList.contains('split-ui');
    if (isSplit) {
        pushCurrentRightPanel();
        document.getElementById('splitRightPanel').style.display = 'none';
        document.getElementById('matchPrepView').style.display = 'block';
    } else {
        window.switchView('matchPrepView');
        pushNavState('matchPrep');
    }

    // Preload all teams to compute overall tiers
    const allTeamsForTier = await db.teams.toArray();
    const overallTierOf = (team) => {
        const myVal = team.analysis?.ceiling != null ? parseFloat(team.analysis.ceiling) : (team.currentEPA || 0);
        return epaRankTier(allTeamsForTier, myVal,
            t => t.analysis?.ceiling != null ? parseFloat(t.analysis.ceiling) : (t.currentEPA || 0));
    };

    // Helper to get team data and return a stats object
    const getTeamStats = async (teamNum) => {
        const team = await db.teams.get(parseInt(teamNum));
        return {
            number: teamNum,
            total: team?.currentEPA || 0,
            auto: team?.autoEPA || 0,
            teleop: team?.teleopEPA || 0,
            endgame: team?.endgameEPA || 0
        };
    };

    // 1. Calculate Alliance Totals
    const redTeamsData = await Promise.all(match.red.map(num => getTeamStats(num)));
    const blueTeamsData = await Promise.all(match.blue.map(num => getTeamStats(num)));

    // 2. Render the Comparison Chart
    renderPrepChart(redTeamsData, blueTeamsData);

    // 2. Helper function to build a team card
    const createTeamCard = async (teamNum, matchNumber = null) => {
        const team = await db.teams.get(parseInt(teamNum));

        // 1. DATA CLEANUP: Force both to strings and trim any whitespace
        const globalFocus = (window.currentFocusedTeam || "").toString().trim();
        const currentCardTeam = (teamNum || "").toString().trim();

        // 2. THE CHECK:
        const isFocused = (globalFocus !== "" && globalFocus === currentCardTeam);

        // 3. LOGGING: Keep this in for one refresh to see the truth in the console
        console.log(`Comparing: [${globalFocus}] to [${currentCardTeam}] -> Result: ${isFocused}`);

        const focusClass = isFocused ? 'highlight-active' : '';

        // Fallback if team data hasn't been synced yet
        if (!team) {
            return `<div class="prep-team-card"><h3>Team ${teamNum}</h3><p>No data. Sync Statbotics.</p></div>`;
        }

        const tier = overallTierOf(team);
        const hasNote = !!getTeamNote(teamNum, matchNumber)?.text;
        const qmArg = matchNumber != null ? matchNumber : 'null';

        return `
        <div class="prep-team-card ${focusClass}" id="prep-card-${teamNum}">
            <div class="prep-card-header" onclick="highlightTeam('${teamNum}')" style="cursor:pointer;">
                <div class="header-left">
                    <span class="prep-team-number">${teamNum}</span>
                    <div style="color:#94a3b8;font-size:0.78em;font-weight:600;">EPA ${team.currentEPA.toFixed(1)}${localEpaBadge(team)}</div>
                </div>
                ${tierBadge(tier, 'prep-tier-badge', 'Tier')}
            </div>

            <div class="prep-action-btns" style="display:flex;gap:8px;margin-top:10px;">
                <button onclick="viewTeamDetail(${teamNum})" style="flex:1;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:7px;font-size:0.82em;font-weight:600;cursor:pointer;">View Profile</button>
                <button id="note-toggle-btn-${teamNum}" onclick="togglePrepNote('${teamNum}')" style="flex:1;background:#1e293b;color:#94a3b8;border:1px solid ${hasNote ? '#3b82f6' : '#334155'};border-radius:6px;padding:7px;font-size:0.82em;font-weight:600;cursor:pointer;">${hasNote ? 'Note ▾' : 'Notes ▾'}</button>
            </div>
            <div id="prep-note-section-${teamNum}" data-view-all="false" style="display:none;margin-top:8px;">
                <div style="display:flex;justify-content:flex-end;margin-bottom:5px;">
                    <button id="note-view-toggle-${teamNum}" onclick="togglePrepNoteView('${teamNum}', ${qmArg})" style="background:none;border:1px solid #334155;color:#64748b;border-radius:4px;padding:2px 8px;font-size:0.72em;cursor:pointer;">Show All</button>
                </div>
                <div id="prep-note-content-${teamNum}">
                    ${renderPrepNoteSection(teamNum, matchNumber)}
                </div>
            </div>
        </div>
    `;
    };

    // 3. Populate Red and Blue Lists
    const redCards = await Promise.all(match.red.map(num => createTeamCard(num, match.matchNumber)));
    const blueCards = await Promise.all(match.blue.map(num => createTeamCard(num, match.matchNumber)));

    document.getElementById('redPrepList').innerHTML = redCards.join('');
    document.getElementById('bluePrepList').innerHTML = blueCards.join('');
};

window.openScoutingBreakdown = async function () {
    if (!currentPrepMatch) return;
    const modal   = document.getElementById('scoutingBreakdownModal');
    const content = document.getElementById('scoutingBreakdownContent');
    modal.style.display = 'block';
    pushNavState('scoutingBreakdown');
    content.innerHTML = '<p style="color:#64748b;text-align:center;margin-top:40px;">Loading…</p>';

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const rawStr   = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!rawStr) {
        content.innerHTML = '<p style="color:#64748b;padding:20px;">No scouting data synced for this event.</p>';
        return;
    }
    const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
    if (!processed || !processed.config.displayFields) {
        content.innerHTML = '<p style="color:#64748b;padding:20px;">No scouting comparison available for this game.</p>';
        return;
    }

    const { config, byTeam } = processed;
    const redTeams  = currentPrepMatch.red  || [];
    const blueTeams = currentPrepMatch.blue || [];
    const allTeams  = [...redTeams, ...blueTeams];

    const tbaMatches = await db.matches.where('eventKey').equals(eventKey).toArray();
    const tbaByMatch = {};
    const hasTBABreakdowns = tbaMatches.some(m => m.redBreakdown);
    if (hasTBABreakdowns && config.enrichAggregateWithTBA) {
        for (const m of tbaMatches) tbaByMatch[m.matchNumber] = m;
    }

    const teamStats = {};
    for (const tn of allTeams) {
        const rows = byTeam[String(tn)];
        if (rows?.length) {
            const { rows: deduped } = deduplicateTeamRows(rows);
            const stats = config.aggregateTeam(deduped);
            if (hasTBABreakdowns && config.enrichAggregateWithTBA) {
                config.enrichAggregateWithTBA(String(tn), deduped, stats, tbaByMatch);
            }
            teamStats[tn] = stats;
        } else {
            teamStats[tn] = null;
        }
    }

    const fmtVal = (stats, field) => {
        if (!stats) return '<span style="color:#334155;">N/A</span>';
        const v = stats[field.key];
        if (v == null) return '<span style="color:#475569;">—</span>';
        if (field.suffix === '%') return `${Math.round(v)}%`;
        if (field.decimals != null) return v.toFixed(field.decimals);
        return String(Math.round(v));
    };

    // ── Radar chart ───────────────────────────────────────────────────────────
    // Fuse each team to get autoFuelFused, then combine with autoClimbPts for auto EPA.
    const allByMatch = indexObservationsByMatch(processed.observations);
    const teamEPABreakdown = {};
    const teamAutoEPA = {};
    for (const tn of allTeams) {
        const agg  = teamStats[tn];
        if (!agg) { teamAutoEPA[tn] = 0; teamEPABreakdown[tn] = { total: 0 }; continue; }
        const rows = byTeam[String(tn)];
        const { rows: deduped } = deduplicateTeamRows(rows);
        const fused = hasTBABreakdowns
            ? fuseScoutingWithTBA(String(tn), deduped, allByMatch, tbaMatches, config)
            : { available: false };
        const fs = fused.available ? fused.stats : {};
        const merged = { ...agg, ...fs };
        const bd = config.computeFusedEPABreakdown?.(merged) ?? { auto: agg.autoClimbPts ?? 0, total: 0 };
        teamAutoEPA[tn] = bd.auto;
        teamEPABreakdown[tn] = bd;
    }
    const redPred  = redTeams.reduce((s, tn)  => s + (teamEPABreakdown[tn]?.total ?? 0), 0);
    const bluePred = blueTeams.reduce((s, tn) => s + (teamEPABreakdown[tn]?.total ?? 0), 0);
    const maxAutoEPA = Math.max(...Object.values(teamAutoEPA), 1);

    const radarAxes = [
        { label: 'Auto EPA',      fn: (s, tn) => teamAutoEPA[tn] / maxAutoEPA * 100 },
        { label: 'Endgame Climb', fn: (s)     => s?.climbPct ?? 0 },
        { label: 'Scoring Eff',   fn: (s)     => s?.avgScoringEff   != null ? (s.avgScoringEff   - 1) / 9 * 100 : 0 },
        { label: 'Shuttling Eff', fn: (s)     => s?.avgPassingSkill != null ? (s.avgPassingSkill - 1) / 9 * 100 : 0 },
        { label: 'Defense Eff',   fn: (s)     => s?.avgDefenseSkill != null ? (s.avgDefenseSkill - 1) / 9 * 100 : 0 },
        { label: 'Reliability',   fn: (s)     => 100 - (s?.pctDied ?? 0) },
    ];

    const redPalette  = ['rgba(239,68,68',  'rgba(248,113,113', 'rgba(252,165,165'];
    const bluePalette = ['rgba(59,130,246', 'rgba(96,165,250',  'rgba(147,197,253'];

    const makeDatasets = (teams, palette) => teams.map((tn, i) => {
        const stats = teamStats[tn];
        const c = palette[i];
        return {
            label: `${tn}${String(tn) === OWN_TEAM ? ' ★' : ''}`,
            data: radarAxes.map(ax => ax.fn(stats, tn)),
            borderColor: `${c},1)`,
            backgroundColor: `${c},0.08)`,
            pointBackgroundColor: `${c},1)`,
            pointRadius: 3,
            borderWidth: 2,
        };
    });

    // ── Table ─────────────────────────────────────────────────────────────────
    const thBase = 'padding:8px 10px;text-align:center;font-weight:700;font-size:0.82em;white-space:nowrap;border-bottom:2px solid';
    let tableHtml = `<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">
        <thead><tr>
            <th style="padding:8px 10px;text-align:left;color:#475569;font-size:0.72em;border-bottom:2px solid #334155;"></th>
            ${redTeams.map(t  => `<th style="${thBase} #7f1d1d;color:#fca5a5;">${t}${ownStar(t)}</th>`).join('')}
            ${blueTeams.map(t => `<th style="${thBase} #1e3a8a;color:#93c5fd;">${t}${ownStar(t)}</th>`).join('')}
        </tr></thead>
        <tbody>`;

    for (const field of config.displayFields) {
        if (field.group) {
            tableHtml += `<tr style="background:#1e293b;">
                <td colspan="${1 + allTeams.length}" style="padding:5px 10px;color:#64748b;font-size:0.7em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">${field.group}</td>
            </tr>`;
            continue;
        }
        tableHtml += `<tr style="border-bottom:1px solid #1e293b;">
            <td style="padding:5px 10px;color:#94a3b8;font-size:0.8em;white-space:nowrap;">${field.label}</td>
            ${redTeams.map(t  => `<td style="padding:5px 10px;text-align:center;font-size:0.82em;">${fmtVal(teamStats[t],  field)}</td>`).join('')}
            ${blueTeams.map(t => `<td style="padding:5px 10px;text-align:center;font-size:0.82em;">${fmtVal(teamStats[t], field)}</td>`).join('')}
        </tr>`;
    }

    tableHtml += `</tbody></table></div>`;

    content.innerHTML = `
        <div style="display:flex;justify-content:center;gap:32px;margin-bottom:16px;font-weight:700;">
            <div style="text-align:center;">
                <div style="font-size:0.7em;font-weight:700;color:#fca5a5;letter-spacing:0.06em;margin-bottom:2px;">RED PREDICTED</div>
                <div style="font-size:1.6em;color:#f87171;">${Math.round(redPred)}</div>
            </div>
            <div style="align-self:center;color:#475569;font-size:0.9em;">vs</div>
            <div style="text-align:center;">
                <div style="font-size:0.7em;font-weight:700;color:#93c5fd;letter-spacing:0.06em;margin-bottom:2px;">BLUE PREDICTED</div>
                <div style="font-size:1.6em;color:#60a5fa;">${Math.round(bluePred)}</div>
            </div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:20px;">
            <div style="flex:1;min-width:260px;text-align:center;">
                <div style="font-size:0.75em;font-weight:700;color:#fca5a5;letter-spacing:0.06em;margin-bottom:6px;">RED ALLIANCE</div>
                <canvas id="breakdownRadarRed"></canvas>
            </div>
            <div style="flex:1;min-width:260px;text-align:center;">
                <div style="font-size:0.75em;font-weight:700;color:#93c5fd;letter-spacing:0.06em;margin-bottom:6px;">BLUE ALLIANCE</div>
                <canvas id="breakdownRadarBlue"></canvas>
            </div>
        </div>
        ${tableHtml}
    `;

    // Render both radars after DOM is updated
    if (breakdownRadarChartRed)  { breakdownRadarChartRed.destroy();  breakdownRadarChartRed  = null; }
    if (breakdownRadarChartBlue) { breakdownRadarChartBlue.destroy(); breakdownRadarChartBlue = null; }

    const radarOptions = {
        responsive: true,
        aspectRatio: 1,
        layout: { padding: 0 },
        scales: {
            r: {
                min: 0, max: 100,
                ticks: { stepSize: 25, color: '#475569', backdropColor: 'transparent', font: { size: 10 } },
                grid:        { color: '#1e293b' },
                angleLines:  { color: '#334155' },
                pointLabels: {
                    color: '#94a3b8',
                    font: { size: 10 },
                    padding: 4,
                    callback: label => label.includes(' ') ? label.split(' ') : label,
                },
            },
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: { color: '#94a3b8', boxWidth: 12, font: { size: 11 }, padding: 8 },
            },
            tooltip: {
                callbacks: { label: ctx => `${ctx.dataset.label}: ${Math.round(ctx.raw)}` },
            },
        },
    };

    const axisLabels = radarAxes.map(a => a.label);
    breakdownRadarChartRed  = new Chart(document.getElementById('breakdownRadarRed').getContext('2d'),
        { type: 'radar', data: { labels: axisLabels, datasets: makeDatasets(redTeams,  redPalette)  }, options: radarOptions });
    breakdownRadarChartBlue = new Chart(document.getElementById('breakdownRadarBlue').getContext('2d'),
        { type: 'radar', data: { labels: axisLabels, datasets: makeDatasets(blueTeams, bluePalette) }, options: radarOptions });
};

window.closeScoutingBreakdown = function () {
    document.getElementById('scoutingBreakdownModal').style.display = 'none';
    if (breakdownRadarChartRed)  { breakdownRadarChartRed.destroy();  breakdownRadarChartRed  = null; }
    if (breakdownRadarChartBlue) { breakdownRadarChartBlue.destroy(); breakdownRadarChartBlue = null; }
};

function renderPrepChart(redTeams, blueTeams) {
    const ctx = document.getElementById('allianceComparisonChart').getContext('2d');

    if (prepChartInstance) {
        prepChartInstance.destroy();
    }

    const redShades = ['#b91c1c', '#ef4444', '#f87171'];
    const blueShades = ['#1e3a8a', '#3b82f6', '#93c5fd'];

    const datasets = [
        ...redTeams.map((team, i) => {
            // Check if this specific team segment should be highlighted
            const isFocused = (window.currentFocusedTeam === team.number.toString());
            return {
                label: `Team ${team.number}`,
                data: [team.total, team.auto, team.teleop, team.endgame],
                backgroundColor: isFocused ? '#fde047' : redShades[i],
                borderColor: isFocused ? '#000' : 'transparent',
                borderWidth: isFocused ? 2 : 0,
                stack: 'Red'
            };
        }),
        ...blueTeams.map((team, i) => {
            const isFocused = (window.currentFocusedTeam === team.number.toString());
            return {
                label: `Team ${team.number}`,
                data: [team.total, team.auto, team.teleop, team.endgame],
                backgroundColor: isFocused ? '#fde047' : blueShades[i],
                borderColor: isFocused ? '#000' : 'transparent',
                borderWidth: isFocused ? 2 : 0,
                stack: 'Blue'
            };
        })
    ];

    prepChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: ['Total EPA', 'Auto', 'Teleop', 'Endgame'], datasets: datasets },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                y: { stacked: true, grid: { display: false }, ticks: { color: '#f8fafc', font: { weight: 'bold' } } }
            },
            plugins: {
                legend: { display: true, position: 'bottom', labels: { color: '#f8fafc', boxWidth: 12 } }
            }
        }
    });
}





window.viewMatchDetail = async function (matchKey) {
    const match = await db.matches.get(matchKey);
    if (!match) return;

    const redWon = match.redScore > match.blueScore;
    const blueWon = match.blueScore > match.redScore;

    document.getElementById('matchDetailLabel').innerText = `Match Details: Qual ${match.matchNumber}`;

    const redScoreEl = document.getElementById('redTotalScore');
    const blueScoreEl = document.getElementById('blueTotalScore');
    redScoreEl.innerText = match.redScore;
    blueScoreEl.innerText = match.blueScore;
    redScoreEl.style.color = redWon ? '#4ade80' : '#f8fafc';
    blueScoreEl.style.color = blueWon ? '#4ade80' : '#f8fafc';

    document.getElementById('redMatchTeams').innerHTML = match.red.map(t => `<div>${t}</div>`).join('');
    document.getElementById('blueMatchTeams').innerHTML = match.blue.map(t => `<div>${t}</div>`).join('');

    const breakdownEl = document.getElementById('scoreBreakdown');
    const rbd = match.redBreakdown;
    const bbd = match.blueBreakdown;

    if (rbd && bbd) {
        const eventKeyFromMatch = matchKey.split('_')[0];
        const mb = getGameConfig(eventKeyFromMatch)?.matchBreakdown;

        const redResultRP  = redWon ? 3 : match.redScore === match.blueScore ? 1 : 0;
        const blueResultRP = blueWon ? 3 : match.redScore === match.blueScore ? 1 : 0;

        const bonusRPFields = mb?.bonusRPFields ?? [];
        const check = val => val ? `<span style="color:#4ade80;">✓</span>` : `<span style="color:#475569;">✗</span>`;

        const totalRedRP  = rbd.rp ?? (redResultRP  + bonusRPFields.reduce((s, { field }) => s + (rbd[field] ? 1 : 0), 0));
        const totalBlueRP = bbd.rp ?? (blueResultRP + bonusRPFields.reduce((s, { field }) => s + (bbd[field] ? 1 : 0), 0));

        const scoreRows = mb?.scoreRows(rbd, bbd) ?? [
            ['Auto',         rbd.totalAutoPoints,   bbd.totalAutoPoints],
            ['Teleop',       rbd.totalTeleopPoints,  bbd.totalTeleopPoints],
            ['Fouls Earned', rbd.foulPoints,         bbd.foulPoints],
        ];
        const rpRows = [
            ['Match Result', redResultRP, blueResultRP],
            ...bonusRPFields.map(({ label, field }) => [label, rbd[field], bbd[field]]),
        ];

        breakdownEl.innerHTML = `
            <table class="breakdown-table">
                <thead><tr>
                    <th></th>
                    <th style="color:#ef4444;">Red</th>
                    <th style="color:#3b82f6;">Blue</th>
                </tr></thead>
                <tbody>
                    ${scoreRows.map(([label, r, b]) => `<tr>
                        <td>${label}</td>
                        <td>${r ?? '—'}</td>
                        <td>${b ?? '—'}</td>
                    </tr>`).join('')}
                    <tr class="breakdown-total">
                        <td>Total</td>
                        <td style="color:${redWon ? '#4ade80' : 'inherit'}">${rbd.totalPoints ?? match.redScore}</td>
                        <td style="color:${blueWon ? '#4ade80' : 'inherit'}">${bbd.totalPoints ?? match.blueScore}</td>
                    </tr>
                    <tr><td colspan="3" style="padding:8px 0 4px; color:#64748b; font-size:0.8em; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Ranking Points</td></tr>
                    ${rpRows.map(([label, r, b]) => `<tr>
                        <td style="color:#94a3b8;">${label}</td>
                        <td>${typeof r === 'number' ? r : check(r)}</td>
                        <td>${typeof b === 'number' ? b : check(b)}</td>
                    </tr>`).join('')}
                    <tr class="breakdown-total">
                        <td>Total RP</td>
                        <td style="color:#fbbf24;">${totalRedRP}</td>
                        <td style="color:#fbbf24;">${totalBlueRP}</td>
                    </tr>
                </tbody>
            </table>`;
    } else {
        breakdownEl.innerHTML = `<p style="color:#64748b; font-style:italic; font-size:0.9em; margin-top:12px;">Run "Sync TBA Matches" for a detailed breakdown.</p>`;
    }

    const videoSection = document.getElementById('matchVideoSection');
    const ytKeys = match.videos || [];
    if (ytKeys.length === 0) {
        let webcasts = [];
        try {
            const ek = document.getElementById('eventKeyInput')?.value.trim().toLowerCase() || match.eventKey;
            webcasts = JSON.parse(localStorage.getItem(`webcasts_${ek}`) || '[]');
        } catch {}
        const stream = findStreamForMatch(match, webcasts);
        if (stream) {
            const matchTs = match.actualTime ?? match.predictedTime;
            const offset = Math.max(0, matchTs - stream.startTimestamp - 2);
            const thumbId = 'stream-seek-thumb';
            videoSection.innerHTML = ytSpeedBar() + `
                <div style="color:#64748b;font-size:0.78em;font-style:italic;margin-bottom:6px;">No match video yet — live stream seeked to approx. match time</div>
                <div id="${thumbId}" onclick="loadYTEmbedAtTime('${stream.channel}','${thumbId}',${offset})"
                    style="position:relative;cursor:pointer;border-radius:8px;overflow:hidden;background:#000;">
                    <img src="https://img.youtube.com/vi/${stream.channel}/hqdefault.jpg"
                        style="width:100%;display:block;opacity:0.75;"
                        onerror="this.style.display='none'" loading="lazy">
                    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;gap:6px;">
                        <div style="width:56px;height:40px;background:rgba(15,23,42,0.82);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3em;">⏩</div>
                        <div style="background:rgba(15,23,42,0.7);color:#94a3b8;font-size:0.72em;padding:2px 8px;border-radius:4px;">Live Stream</div>
                    </div>
                </div>`;
        } else {
            videoSection.innerHTML = `<p style="color:#64748b; font-style:italic; font-size:0.85em; margin:0;">No match video available.</p>`;
        }
    } else {
        videoSection.innerHTML = ytSpeedBar() + ytKeys.map((key, i) => {
            const thumbId = `yt-thumb-${i}`;
            return `<div id="${thumbId}" onclick="loadYTEmbed('${key}','${thumbId}')"
                style="position:relative; cursor:pointer; border-radius:8px; overflow:hidden; background:#000; ${i > 0 ? 'margin-top:12px;' : ''}">
                <img src="https://img.youtube.com/vi/${key}/hqdefault.jpg"
                    style="width:100%; display:block; opacity:0.85;"
                    onerror="this.style.display='none'" loading="lazy">
                <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; pointer-events:none;">
                    <div style="width:64px; height:44px; background:rgba(255,0,0,0.85); border-radius:10px; display:flex; align-items:center; justify-content:center;">
                        <div style="border-style:solid; border-width:10px 0 10px 20px; border-color:transparent transparent transparent #fff; margin-left:4px;"></div>
                    </div>
                </div>
            </div>`;
        }).join('');
    }

    const isSplitDetail = document.body.classList.contains('split-ui');
    if (isSplitDetail) pushCurrentRightPanel();
    document.getElementById('matchDetailView').style.display = isSplitDetail ? 'block' : 'flex';
    pushNavState('matchDetail');

    // Robot routes, if this match has been tracked. Deliberately AFTER the modal is
    // shown: the canvas cannot measure itself while its container is display:none, the
    // same constraint that makes performanceChart render lazily. Fire-and-forget so a
    // slow/absent tracks file never delays the modal.
    renderMatchTracks(matchKey);
};

// Populates #matchTracksSection. Silent no-op when the match has no tracks, which is
// the normal case for almost every match.
async function renderMatchTracks(matchKey) {
    const host = document.getElementById('matchTracksSection');
    if (!host) return;
    host.innerHTML = '';
    const doc = await loadMatchTracks(matchKey);
    if (!doc) return;

    const q = doc.quality || {};
    const vid = doc.source?.videoId;
    host.innerHTML = `
        <h3 style="margin:18px 0 8px; font-size:15px;">Robot Routes</h3>
        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:baseline;
                    color:#94a3b8; font-size:12px; margin-bottom:8px;">
            <span>${q.samplesOut ?? 0} samples @ ${doc.sampling?.outputHz ?? '?'} Hz</span>
            <span>custody ${Math.round(100 * (q.meanCustody || 0))}%</span>
            ${q.curated ? '<span style="color:#22c55e;">curated</span>'
                        : '<span style="color:#f59e0b;">auto-labelled</span>'}
        </div>
        <div id="mtWrap" style="position:relative; width:100%; border-radius:8px; overflow:hidden;">
            <img id="mtImg" src="${import.meta.env.BASE_URL}${doc.field.imageRef}"
                 alt="field" style="display:block; width:100%; height:auto;">
            <canvas id="mtCanvas" style="position:absolute; inset:0; width:100%; height:100%;"></canvas>
        </div>
        <div style="display:flex; gap:10px; align-items:center; margin-top:8px; flex-wrap:wrap;">
            <button id="mtPlay" style="padding:3px 10px;font-size:0.78em;border-radius:4px;cursor:pointer;border:1px solid #334155;background:transparent;color:#94a3b8;">▶</button>
            <input type="range" id="mtScrub" min="0" max="1000" value="1000" style="flex:1; min-width:160px;">
            <span id="mtClock" style="font-variant-numeric:tabular-nums; font-size:12px;
                  color:#94a3b8; min-width:64px;">full</span>
            ${vid ? `<button id="mtSeek" title="Open the match video at the moment the slider is showing"
                 style="padding:3px 10px;font-size:0.78em;border-radius:4px;cursor:pointer;border:1px solid #334155;background:transparent;color:#94a3b8;">▶ Watch this moment</button>` : ''}
        </div>
        <div id="mtTeams" style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;"></div>
        <div id="mtLegend" style="display:flex; gap:14px; flex-wrap:wrap; margin-top:7px;
             font-size:11px; color:#64748b;"></div>
        <div style="margin-top:8px;">
          <button id="mtFull" title="Open this route view full screen"
                  style="padding:4px 11px; font-size:12px; border-radius:6px;
                         border:1px solid #334155; background:transparent; color:#94a3b8;
                         cursor:pointer;">⛶ Full screen</button>
          <button id="mtAuto" style="padding:4px 11px; font-size:12px; border-radius:6px;
                  cursor:pointer; border:1px solid #334155; background:transparent;
                  color:#94a3b8; font-weight:600;">Auto only</button>
          <span id="mtAutoNote" style="color:#64748b; font-size:11px; margin-left:8px;"></span>
        </div>`;

    applyFieldOrientation(document.getElementById('mtWrap'), doc);
    const img = document.getElementById('mtImg');
    const cv  = document.getElementById('mtCanvas');
    const shown = new Set(doc.robots.map(r => String(r.team)));
    let tMin = Infinity, tMax = -Infinity;
    for (const r of doc.robots) for (const s of r.samples) {
        if (s.t < tMin) tMin = s.t;
        if (s.t > tMax) tMax = s.t;
    }
    let tNow = tMax, playing = false, raf = 0;
    // Autonomous is the slice scouts care about most and it is ~20 s of ~170, so at full
    // scale it is a knot in the corner of the plot. Sample times are relative to auto
    // start, so isolating it is just an upper bound on t -- no second dataset, no
    // re-fetch, and the scrubber keeps working inside the clipped range.
    const autoEnd = autoEndOf(doc);
    let autoOnly = false;

    const paint = () => renderFieldRoutes(cv, doc, {
        teams: shown, tNow, trailOnly: tNow < tMax, dots: tNow < tMax,
        tMax: autoOnly ? autoEnd : null,
    });
    const resize = () => { if (_trackSizeCanvas(img, cv)) paint(); };

    document.getElementById('mtTeams').innerHTML = doc.robots.map((r, i) => `
        <button class="mt-team" data-team="${r.team}" style="padding:4px 9px; font-size:12px;
                border:1px solid #334155; border-radius:6px; cursor:pointer;
                background:transparent; color:${window.trackColourFor(r, i)};
                font-weight:700;">${r.team}</button>`).join('');
    // Only name a shading that is actually on screen. A legend entry for an overlay the
    // reader cannot see teaches them to look for something that is not there -- and on a
    // camera that sees the whole field, "outside camera coverage" is genuinely absent.
    const _lgChip = (border, fill) =>
        `<span style="display:inline-block;width:22px;height:11px;border-radius:2px;
         border:1px dashed ${border};background:${fill};vertical-align:-1px;
         margin-right:5px;"></span>`;
    const _lg = [];
    if (Array.isArray(doc.field?.occluderPolysM) && doc.field.occluderPolysM.length) {
        _lg.push(_lgChip('rgba(240,168,51,0.85)', 'rgba(240,168,51,0.18)')
                 + 'hidden behind a structure');
    }
    if (Array.isArray(doc.field?.visiblePolyM) && (doc.field.visibleFrac ?? 1) < 0.98) {
        _lg.push(_lgChip('rgba(226,232,240,0.75)', 'rgba(170,180,190,0.22)')
                 + 'outside camera coverage');
    }
    document.getElementById('mtLegend').innerHTML =
        _lg.map(t => `<span>${t}</span>`).join('');

    document.getElementById('mtTeams').querySelectorAll('.mt-team').forEach(b => {
        b.onclick = () => {
            const t = b.dataset.team;
            if (shown.has(t)) { shown.delete(t); b.style.opacity = '0.35'; }
            else { shown.add(t); b.style.opacity = '1'; }
            paint();
        };
    });

    const fullBtn = document.getElementById('mtFull');
    if (fullBtn) fullBtn.onclick = () => window.openRoutesFull(doc, {
        // Hand over the CURRENT view, not a default one: whatever teams are shown and
        // whether auto-only is on should survive the jump to full screen.
        teams: shown, tNow, trailOnly: tNow < tMax, dots: tNow < tMax,
        tMax: autoOnly ? autoEnd : null,
    });
    const autoBtn = document.getElementById('mtAuto');
    const autoNote = document.getElementById('mtAutoNote');
    const syncAuto = () => {
        autoBtn.style.background = autoOnly ? '#1e3a5f' : 'transparent';
        autoBtn.style.borderColor = autoOnly ? '#3b82f6' : '#334155';
        autoBtn.style.color = autoOnly ? '#60a5fa' : '#94a3b8';
        autoNote.textContent = autoOnly ? `first ${autoEnd.toFixed(0)}s of the match` : '';
    };
    autoBtn.onclick = () => {
        autoOnly = !autoOnly;
        // Snap the scrubber to the end of whichever range is now showing, so the plot is
        // never blank because tNow sits past the clip.
        if (autoOnly && tNow > autoEnd) setT(autoEnd); else paint();
        syncAuto();
    };
    syncAuto();

    const setT = (t) => {
        tNow = Math.max(tMin, Math.min(tMax, t));
        document.getElementById('mtScrub').value =
            String(Math.round((tNow - tMin) / (tMax - tMin) * 1000));
        document.getElementById('mtClock').textContent =
            tNow >= tMax ? 'full' : `${tNow.toFixed(1)}s`;
        paint();
    };
    document.getElementById('mtScrub').oninput = (e) => {
        if (playing) { playing = false; cancelAnimationFrame(raf);
                       document.getElementById('mtPlay').textContent = '▶'; }
        setT(tMin + (e.target.value / 1000) * (tMax - tMin));
    };
    const step = (prev) => {
        if (!playing) return;
        const now = performance.now();
        setT(tNow + (now - prev) / 1000);
        if (tNow >= tMax) { playing = false;
                            document.getElementById('mtPlay').textContent = '▶'; return; }
        raf = requestAnimationFrame(() => step(now));
    };
    document.getElementById('mtPlay').onclick = () => {
        playing = !playing;
        document.getElementById('mtPlay').textContent = playing ? '❚❚' : '▶';
        if (playing) { if (tNow >= tMax) setT(tMin);
                       raf = requestAnimationFrame(() => step(performance.now())); }
        else cancelAnimationFrame(raf);
    };
    const seek = document.getElementById('mtSeek');
    if (seek) seek.onclick = () => {
        // Opens the match video at whatever instant the slider is on. One-way only:
        // reading playback position BACK would need the full YouTube IFrame API with
        // event listeners, and this app only ever posts commands to embeds.
        const at = Math.max(0, (doc.source?.matchStartVideoSec || 0)
                               + (tNow >= tMax ? 0 : tNow));
        // Thumbnails are '#yt-thumb-<i>' (~main.js:1504) or '#stream-seek-thumb'.
        const holder = document.querySelector(
            '#matchVideoSection [id^="yt-thumb-"], #matchVideoSection #stream-seek-thumb');
        if (holder) { loadYTEmbedAtTime(vid, holder.id, at); return; }
        // Embed already running: rewriting iframe.src mid-playback was the error you
        // hit -- the player is initialised and a src swap tears it down uncleanly. Post
        // a seek instead, which is what the API is for, and only touch src as a last
        // resort when there is no iframe at all.
        const frame = document.querySelector('#matchVideoSection iframe[data-yt]');
        if (frame && frame.contentWindow) {
            try {
                frame.contentWindow.postMessage(JSON.stringify({
                    event: 'command', func: 'seekTo', args: [Math.floor(at), true],
                }), '*');
                frame.contentWindow.postMessage(JSON.stringify({
                    event: 'command', func: 'playVideo', args: [],
                }), '*');
                return;
            } catch { /* fall through */ }
        }
        // Nothing to talk to: build the embed fresh at the right time.
        const host = document.getElementById('matchVideoSection');
        if (host) {
            const id = 'mt-yt-host';
            host.insertAdjacentHTML('afterbegin', `<div id="${id}"></div>`);
            loadYTEmbedAtTime(vid, id, at);
        }
    };

    // The img may already be cached (complete) or still loading — handle both, exactly
    // as initFieldTab does for the drawing canvas.
    if (img.complete && img.naturalWidth) resize();
    else img.addEventListener('load', resize, { once: true });
    window.addEventListener('resize', resize);
    host._trackResize = resize;   // so closeMatchDetail can unhook it
}

window.openLightbox = function (url) {
    const lb = document.getElementById('photoLightbox');
    document.getElementById('lightboxImg').src = url;
    lb.style.display = 'flex';
    pushNavState('lightbox');
};

window.closeLightbox = function () {
    document.getElementById('photoLightbox').style.display = 'none';
    document.getElementById('lightboxImg').src = '';
};

window.closeMatchDetail = function () {
    document.getElementById('matchDetailView').style.display = 'none';
    document.getElementById('matchVideoSection').innerHTML = '';
    // Same teardown for the routes section, plus its window-level resize listener --
    // clearing innerHTML alone would leak one listener per modal open.
    const mt = document.getElementById('matchTracksSection');
    if (mt) {
        if (mt._trackResize) { window.removeEventListener('resize', mt._trackResize);
                               mt._trackResize = null; }
        mt.innerHTML = '';
    }
    if (document.body.classList.contains('split-ui') && !popRightPanel()) {
        document.getElementById('splitRightPanel').style.display = 'flex';
    }
};

window.closePrepView = function () {
    if (document.body.classList.contains('split-ui')) {
        document.getElementById('matchPrepView').style.display = 'none';
        if (!popRightPanel()) {
            document.getElementById('splitRightPanel').style.display = 'flex';
        }
    } else {
        window.switchView('scheduleView');
    }
};

// ── Note editor (match prep cards) ──────────────────────────────────────────

const NOTE_TA_STYLE = 'width:100%;box-sizing:border-box;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#f8fafc;padding:8px;font-size:0.85em;font-family:inherit;resize:vertical;min-height:60px;margin-top:6px;';
const NOTE_BTN = (label, color, onclick) =>
    `<button onclick="${onclick}" style="background:${color};color:#f8fafc;border:none;border-radius:6px;padding:5px 12px;font-size:0.82em;font-weight:600;cursor:pointer;">${label}</button>`;

function renderPrepNoteSection(teamNum, qm) {
    const note = getTeamNote(teamNum, qm);
    const qmArg = qm != null ? qm : 'null';
    if (note?.text) {
        return `<div class="prep-note-display" style="color:#94a3b8;font-size:0.82em;line-height:1.4;padding:6px 8px;background:#0f172a;border-radius:4px;border-left:2px solid #3b82f6;white-space:pre-wrap;margin-bottom:6px;">${noteDisplayText(teamNum, qm)}</div>
            <div style="display:flex;gap:6px;">
                ${NOTE_BTN('Edit', '#334155', `showPrepNoteEditor('${teamNum}', ${qmArg})`)}
                ${NOTE_BTN('Delete', '#7f1d1d', `deletePrepNote('${teamNum}', ${qmArg})`)}
            </div>`;
    }
    const label = qm != null ? ` for QM ${qm}` : '';
    return `<div style="color:#475569;font-size:0.82em;margin-bottom:6px;font-style:italic;">No note${label}.</div>
        ${NOTE_BTN('Add Note', '#3b82f6', `showPrepNoteEditor('${teamNum}', ${qmArg})`)}`;
}

window.togglePrepNote = function (teamNum) {
    const section = document.getElementById(`prep-note-section-${teamNum}`);
    const btn = document.getElementById(`note-toggle-btn-${teamNum}`);
    if (!section) return;
    const opening = section.style.display === 'none';
    section.style.display = opening ? 'block' : 'none';
    if (btn) btn.textContent = btn.textContent.replace(/[▾▴]/, opening ? '▴' : '▾');
};

window.togglePrepNoteView = function (teamNum, qm) {
    const section = document.getElementById(`prep-note-section-${teamNum}`);
    const viewBtn = document.getElementById(`note-view-toggle-${teamNum}`);
    const content = document.getElementById(`prep-note-content-${teamNum}`);
    if (!section || !content) return;
    const showingAll = section.dataset.viewAll === 'true';
    if (showingAll) {
        section.dataset.viewAll = 'false';
        if (viewBtn) viewBtn.textContent = 'Show All';
        content.innerHTML = renderPrepNoteSection(teamNum, qm);
    } else {
        section.dataset.viewAll = 'true';
        if (viewBtn) viewBtn.textContent = 'This Match';
        const lines = allNoteDisplayLines(teamNum);
        content.innerHTML = lines.length
            ? `<div class="prep-note-display" style="color:#94a3b8;font-size:0.82em;line-height:1.6;white-space:pre-wrap;">${lines.join('\n')}</div>`
            : `<div style="color:#475569;font-size:0.82em;font-style:italic;">No notes for this team yet.</div>`;
    }
};

window.showPrepNoteEditor = function (teamNum, qm) {
    const content = document.getElementById(`prep-note-content-${teamNum}`);
    if (!content) return;
    const qmArg = qm != null ? qm : 'null';
    const existing = getTeamNote(teamNum, qm);
    content.innerHTML = `<textarea id="note-ta-${teamNum}" style="${NOTE_TA_STYLE}">${existing?.text ?? ''}</textarea>
        <div style="display:flex;gap:6px;margin-top:6px;">
            ${NOTE_BTN('Save', '#3b82f6', `savePrepNote('${teamNum}', ${qmArg})`)}
            ${NOTE_BTN('Cancel', '#334155', `cancelPrepNoteEditor('${teamNum}', ${qmArg})`)}
        </div>`;
    document.getElementById(`note-ta-${teamNum}`)?.focus();
};

window.savePrepNote = function (teamNum, qm) {
    const text = document.getElementById(`note-ta-${teamNum}`)?.value || '';
    saveTeamNote(teamNum, text, qm);
    const section = document.getElementById(`prep-note-section-${teamNum}`);
    const content = document.getElementById(`prep-note-content-${teamNum}`);
    if (section) section.dataset.viewAll = 'false';
    const viewBtn = document.getElementById(`note-view-toggle-${teamNum}`);
    if (viewBtn) viewBtn.textContent = 'Show All';
    if (content) content.innerHTML = renderPrepNoteSection(teamNum, qm);
    _updateNoteToggleBtn(teamNum, qm);
};

window.cancelPrepNoteEditor = function (teamNum, qm) {
    const content = document.getElementById(`prep-note-content-${teamNum}`);
    if (content) content.innerHTML = renderPrepNoteSection(teamNum, qm);
};

window.deletePrepNote = function (teamNum, qm) {
    saveTeamNote(teamNum, '', qm);
    const section = document.getElementById(`prep-note-section-${teamNum}`);
    const content = document.getElementById(`prep-note-content-${teamNum}`);
    if (section) section.dataset.viewAll = 'false';
    const viewBtn = document.getElementById(`note-view-toggle-${teamNum}`);
    if (viewBtn) viewBtn.textContent = 'Show All';
    if (content) content.innerHTML = renderPrepNoteSection(teamNum, qm);
    _updateNoteToggleBtn(teamNum, qm);
};

function _updateNoteToggleBtn(teamNum, qm) {
    const btn = document.getElementById(`note-toggle-btn-${teamNum}`);
    if (!btn) return;
    const hasNote = !!getTeamNote(teamNum, qm)?.text;
    const isOpen = document.getElementById(`prep-note-section-${teamNum}`)?.style.display !== 'none';
    btn.textContent = `${hasNote ? 'Note' : 'Notes'} ${isOpen ? '▴' : '▾'}`;
    btn.style.borderColor = hasNote ? '#3b82f6' : '#334155';
}

// Note editor wired into the team detail Overview tab (adds/edits general notes only)
window.showOverviewNoteEditor = function (teamNum) {
    const section = document.getElementById('overview-notes-section');
    if (!section) return;
    const existing = getTeamNote(teamNum, null); // general note
    const allLines = allNoteDisplayLines(teamNum);
    const allDisplay = allLines.length
        ? `<div style="background:#1e293b;padding:10px 14px;border-radius:8px;border:1px solid #334155;color:#cbd5e1;font-size:0.9em;line-height:1.5;white-space:pre-wrap;margin-bottom:10px;">${allLines.join('\n')}</div>`
        : '';
    section.innerHTML = `${allDisplay}
        <div style="color:#64748b;font-size:0.8em;margin-bottom:4px;">General note (not tied to a match)</div>
        <textarea id="overview-note-ta" style="${NOTE_TA_STYLE}">${existing?.text ?? ''}</textarea>
        <div style="display:flex;gap:6px;margin-top:6px;">
            ${NOTE_BTN('Save', '#3b82f6', `saveOverviewNote(${teamNum})`)}
            ${NOTE_BTN('Cancel', '#334155', `cancelOverviewNote(${teamNum})`)}
            ${existing?.text ? NOTE_BTN('Delete', '#7f1d1d', `deleteOverviewNote(${teamNum})`) : ''}
        </div>`;
    document.getElementById('overview-note-ta')?.focus();
};

window.saveOverviewNote = function (teamNum) {
    const text = document.getElementById('overview-note-ta')?.value || '';
    saveTeamNote(teamNum, text, null); // always saves as general note
    renderNoteSection(teamNum);
};

window.cancelOverviewNote = function (teamNum) { renderNoteSection(teamNum); };

window.deleteOverviewNote = function (teamNum) {
    saveTeamNote(teamNum, '', null); // delete general note only
    renderNoteSection(teamNum);
};

function renderNoteSection(teamNum) {
    const section = document.getElementById('overview-notes-section');
    if (!section) return;

    const userLines  = allNoteDisplayLines(teamNum);
    const hasGeneral = !!getTeamNote(teamNum, null)?.text;
    const eventKey   = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const scoutComments = getScoutingComments(teamNum, eventKey);

    // Build user notes block
    const userBlock = userLines.length
        ? `<div style="margin-bottom:10px;">
               <div style="color:#60a5fa;font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Your Notes</div>
               <div style="background:#1e293b;padding:12px 14px;border-radius:8px;border:1px solid #334155;color:#cbd5e1;font-size:0.9em;line-height:1.5;white-space:pre-wrap;">${userLines.join('\n')}</div>
           </div>`
        : '';

    // Build scout comments block (read-only)
    const scoutBlock = scoutComments.length
        ? `<div style="margin-bottom:10px;">
               <div style="color:#a78bfa;font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Scout Observations</div>
               <div style="background:#1e293b;padding:12px 14px;border-radius:8px;border:1px solid #334155;color:#cbd5e1;font-size:0.9em;line-height:1.5;">
                   ${scoutComments.map(c => `<div style="padding:4px 0;border-bottom:1px solid #1e293b;"><span style="color:#64748b;font-size:0.85em;margin-right:8px;">QM ${c.matchNumber}</span>${c.text}</div>`).join('')}
               </div>
           </div>`
        : '';

    const hasAnything = userLines.length || scoutComments.length;

    section.innerHTML = `
        ${userBlock}
        ${scoutBlock}
        ${!hasAnything ? `<div style="background:#1e293b;padding:16px;border-radius:8px;border:1px dashed #334155;display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <p style="color:#475569;font-style:italic;margin:0;font-size:0.9em;">No notes yet.</p>
            <button onclick="showOverviewNoteEditor(${teamNum})" style="background:#334155;color:#f8fafc;border:none;border-radius:6px;padding:6px 14px;font-size:0.82em;font-weight:600;cursor:pointer;white-space:nowrap;">Add Note</button>
        </div>` : `<button onclick="showOverviewNoteEditor(${teamNum})" style="background:#334155;color:#f8fafc;border:none;border-radius:6px;padding:6px 14px;font-size:0.82em;font-weight:600;cursor:pointer;">${hasGeneral ? 'Edit General Note' : 'Add General Note'}</button>`}`;
}

function getYTPlaybackRate() {
    return parseFloat(localStorage.getItem('ytPlaybackRate') || '1');
}

function ytSpeedBar() {
    const cur = getYTPlaybackRate();
    const btns = [0.5, 0.75, 1, 1.25, 1.5, 2].map(r => {
        const active = r === cur;
        return `<button class="yt-speed-btn" data-rate="${r}" onclick="setYTPlaybackRate(${r})"
            style="padding:2px 8px;font-size:0.72em;border-radius:4px;cursor:pointer;
            border:1px solid ${active ? '#60a5fa' : '#334155'};
            background:${active ? 'rgba(96,165,250,0.12)' : 'transparent'};
            color:${active ? '#60a5fa' : '#64748b'};">${r}×</button>`;
    }).join('');
    return `<div style="display:flex;align-items:center;gap:5px;margin-bottom:8px;">
        <span style="color:#475569;font-size:0.72em;font-weight:600;margin-right:2px;">SPEED</span>${btns}
    </div>`;
}

window.setYTPlaybackRate = function(rate) {
    localStorage.setItem('ytPlaybackRate', String(rate));
    document.querySelectorAll('.yt-speed-btn').forEach(btn => {
        const active = parseFloat(btn.dataset.rate) === rate;
        btn.style.borderColor = active ? '#60a5fa' : '#334155';
        btn.style.background  = active ? 'rgba(96,165,250,0.12)' : 'transparent';
        btn.style.color       = active ? '#60a5fa' : '#64748b';
    });
    document.querySelectorAll('iframe[data-yt]').forEach(iframe => {
        try { iframe.contentWindow.postMessage(JSON.stringify({ event: 'command', func: 'setPlaybackRate', args: [rate] }), '*'); } catch {}
    });
};

window.applyYTPlaybackRate = function(iframeId) {
    const rate = getYTPlaybackRate();
    if (rate === 1) return;
    const send = () => {
        const iframe = document.getElementById(iframeId);
        if (!iframe) return;
        try { iframe.contentWindow.postMessage(JSON.stringify({ event: 'command', func: 'setPlaybackRate', args: [rate] }), '*'); } catch {}
    };
    setTimeout(send, 800);
    setTimeout(send, 2000);
};

window.loadYTEmbed = function (key, thumbId) {
    const container = document.getElementById(thumbId);
    if (!container) return;
    const iframeId = `yt-iframe-${thumbId}`;
    container.outerHTML = `<div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; border-radius:8px;">
        <iframe id="${iframeId}" data-yt="1" src="https://www.youtube-nocookie.com/embed/${key}?autoplay=1&enablejsapi=1"
            style="position:absolute; top:0; left:0; width:100%; height:100%; border:0;"
            allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen loading="lazy"
            onload="applyYTPlaybackRate('${iframeId}')"></iframe>
    </div>`;
};

window.loadYTEmbedAtTime = function (key, thumbId, startSecs) {
    const container = document.getElementById(thumbId);
    if (!container) return;
    const iframeId = `yt-iframe-${thumbId}`;
    container.outerHTML = `<div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; border-radius:8px;">
        <iframe id="${iframeId}" data-yt="1" src="https://www.youtube-nocookie.com/embed/${key}?autoplay=1&enablejsapi=1&start=${Math.floor(startSecs)}"
            style="position:absolute; top:0; left:0; width:100%; height:100%; border:0;"
            allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen loading="lazy"
            onload="applyYTPlaybackRate('${iframeId}')"></iframe>
    </div>`;
};

// Returns the webcast record (with startTimestamp) whose date matches the match's play time.
// Uses actualTime when available; falls back to predictedTime if the predicted time has already passed.
function findStreamForMatch(match, webcasts) {
    const now = Math.floor(Date.now() / 1000);
    const ts = match.actualTime ?? (match.predictedTime < now ? match.predictedTime : null);
    if (!ts || !webcasts.length) return null;
    const matchDate = new Date(ts * 1000).toISOString().slice(0, 10);
    return webcasts.find(w => w.date === matchDate && w.type === 'youtube' && w.startTimestamp) ?? null;
}





window.highlightTeam = function (teamNumber) {
    const allCells = document.querySelectorAll('.red-cell, .blue-cell');
    const allRows = document.querySelectorAll('#scheduleBody tr');

    // 1. Clear previous
    allCells.forEach(cell => cell.classList.remove('highlight-active'));
    allRows.forEach(row => row.classList.remove('row-highlight'));

    // 2. Toggle check (using window.currentFocusedTeam)
    if (window.currentFocusedTeam === teamNumber.toString()) {
        window.currentFocusedTeam = null;
        window.refreshPrepHighlight(); // Keep these in sync
        if (document.getElementById('schedule-sub-watchlist')?.style.display !== 'none') renderWatchList();
        return;
    }

    // 3. Apply new
    const targets = document.querySelectorAll(`[data-team="${teamNumber}"]`);
    if (targets.length > 0) {
        targets.forEach(cell => {
            cell.classList.add('highlight-active');
            const parentRow = cell.closest('tr');
            if (parentRow) parentRow.classList.add('row-highlight');
        });

        window.currentFocusedTeam = teamNumber.toString();
    }

    // 4. Update the Prep cards if they are currently visible
    window.refreshPrepHighlight();
    applyScheduleFilter();
    if (document.getElementById('schedule-sub-watchlist')?.style.display !== 'none') renderWatchList();
};

window.refreshPrepHighlight = function () {
    const allCards = document.querySelectorAll('.prep-team-card');

    // 1. Update the Team Cards
    allCards.forEach(card => {
        const teamNum = card.querySelector('.prep-card-header span').innerText;
        if (window.currentFocusedTeam === teamNum) {
            card.classList.add('highlight-active');
        } else {
            card.classList.remove('highlight-active');
        }
    });

    // 2. Update the Chart segments
    if (prepChartInstance) {
        const redShades = ['#b91c1c', '#ef4444', '#f87171'];
        const blueShades = ['#1e3a8a', '#3b82f6', '#93c5fd'];

        prepChartInstance.data.datasets.forEach((dataset, index) => {
            // Extract the team number from the label "Team 1768"
            const teamNum = dataset.label.replace('Team ', '');
            const isFocused = (window.currentFocusedTeam === teamNum);

            if (dataset.stack === 'Red') {
                dataset.backgroundColor = isFocused ? '#fde047' : redShades[index % 3];
            } else {
                // Blue teams are the 4th, 5th, and 6th datasets
                dataset.backgroundColor = isFocused ? '#fde047' : blueShades[(index - 3) % 3];
            }

            // Add a border to the highlighted segment to make it "pop"
            dataset.borderColor = isFocused ? '#000' : 'transparent';
            dataset.borderWidth = isFocused ? 2 : 0;
        });

        prepChartInstance.update();
    }
};






async function getMatchHistory(teamNumber, year) {
    const url = `https://api.statbotics.io/v3/team_matches?team=${teamNumber}&year=${year}`;
    const response = await fetch(url);
    const json = await response.json();
    const matchArray = json.data || json.results || json;

    if (!matchArray || matchArray.length === 0) return [];

    // We still sort it so the order is preserved in the database
    matchArray.sort((a, b) => a.time - b.time);

    return matchArray; // Return the full objects, not just EPA
}

// teamEventData: the full record from team_events?event= (passed in from syncProjections).
// Keeps component EPAs event-specific without an extra per-team network call.
async function processTeamPerformance(teamNumber, eventKey, force = false, teamEventData = null) {
    const year = eventKey.slice(0, 4);

    // 1. Check local DB
    const cachedTeam = await db.teams.get(teamNumber);

    const nameResp = await fetch(`https://api.statbotics.io/v3/team/${teamNumber}`);
    const nameData = await nameResp.json();
    const teamName = nameData.name || "Unknown Team";

    // 2. Handshake (team_year) — used for match count and as fallback for EPA values
    const summaryResp = await fetch(`https://api.statbotics.io/v3/team_year/${teamNumber}/${year}`);
    const summary = await summaryResp.json();
    const apiMatchCount = summary.count || summary.data?.count || 0;

    console.log(`Team ${teamNumber}: Local Count ${cachedTeam?.matchCount || 0}, API Count ${apiMatchCount}`);

    // Event-specific EPA from the bulk team_events record; falls back to season-level team_year.
    const evEPA       = teamEventData?.epa ?? null;
    const bd          = evEPA?.breakdown ?? null;
    const autoEPA     = bd?.auto_points    ?? summary.epa?.breakdown?.auto_points    ?? 0;
    const teleopEPA   = bd?.teleop_points  ?? summary.epa?.breakdown?.teleop_points  ?? 0;
    const endgameEPA  = bd?.endgame_points ?? summary.epa?.breakdown?.endgame_points ?? 0;
    const eventEpaEnd = evEPA?.end ?? null;

    // 3. Only skip deep dive if cache is fresh
    const needsUpdate = force || !cachedTeam || cachedTeam.matchCount !== apiMatchCount;

    if (!needsUpdate) {
        await db.teams.update(teamNumber, {
            ...(eventEpaEnd != null ? { currentEPA: eventEpaEnd } : {}),
            autoEPA,
            teleopEPA,
            endgameEPA,
            epa: evEPA ?? summary.epa ?? null,
        });
        console.log(`-> Skipping deep dive for ${teamNumber}, but updated summary stats.`);
        return null;
    }

    // 4. Deep dive — full match history for EPA timeline and ceiling analysis
    console.log(`-> Fetching full matches for ${teamNumber}...`);
    const fullMatchData = await getMatchHistory(teamNumber, year);

    if (!fullMatchData || fullMatchData.length === 0) {
        console.warn(`-> No match data found for ${teamNumber}`);
        return null;
    }

    const playedMatches = fullMatchData.filter(m => m.status === 'Completed' && m.epa?.post);
    const currentEPA = eventEpaEnd
        ?? (playedMatches.length > 0 ? playedMatches[playedMatches.length - 1].epa.post : 0);

    await db.teams.put({
        teamNumber,
        teamName,
        eventKey,
        matchCount: apiMatchCount,
        currentEPA,
        autoEPA,
        teleopEPA,
        endgameEPA,
        epa: evEPA ?? summary.epa ?? null,
        rawStatboticsData: fullMatchData,
        analysis:    cachedTeam?.analysis    ?? null,
        lastUpdated: Date.now(),
        // Preserve local EPA fields — put() is a full replace, these would otherwise be wiped,
        // causing computeLocalEPA to re-seed from currentEPA instead of the original baseline.
        preEventEPA:        cachedTeam?.preEventEPA        ?? null,
        preEventAutoEPA:    cachedTeam?.preEventAutoEPA    ?? null,
        preEventTeleopEPA:  cachedTeam?.preEventTeleopEPA  ?? null,
        preEventEndgameEPA: cachedTeam?.preEventEndgameEPA ?? null,
        localEPATimeline:   cachedTeam?.localEPATimeline   ?? [],
    });

    return null;
}


function setSyncTimestamp(key) {
    const str = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    localStorage.setItem(`lastSync_${key}`, str);
    const el = document.getElementById(`ts-${key}`);
    if (el) el.textContent = `Last sync: ${str}`;
}

window.syncProjections = async function () {
    const input = document.getElementById('eventKeyInput');
    const eventKey = input ? input.value.trim().toLowerCase() : "";
    if (!eventKey) { alert("Please enter a valid Event Key first!"); return; }

    localStorage.setItem('lastEventKey', eventKey);

    const statusDiv = document.getElementById('status');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');

    // 1. Get team list from TBA (primary source — always available)
    statusDiv.innerText = `Fetching team list for ${eventKey} from TBA…`;
    let tbaTeams = [];
    try {
        const resp = await fetchTBA(`/event/${eventKey}/teams`);
        tbaTeams = Array.isArray(resp) ? resp : [];
    } catch (e) {
        statusDiv.innerText = `❌ Failed to reach TBA. Check your event key and network connection.`;
        return;
    }
    if (!tbaTeams.length) {
        statusDiv.innerText = `❌ No teams found for ${eventKey} on TBA. Check the event key.`;
        return;
    }

    // Build a name map from TBA data
    const tbaNameMap = {};
    for (const t of tbaTeams) tbaNameMap[t.team_number] = t.nickname || `Team ${t.team_number}`;

    // 2. Try Statbotics event bulk endpoint to get event-specific EPA breakdowns (optional)
    //    This only returns data if the event is hosted on Statbotics; if not, we still
    //    fetch per-team history via processTeamPerformance using year-level data.
    statusDiv.innerText = `Fetching Statbotics event data for ${eventKey}…`;
    const sbMap = {};
    try {
        const sbResp = await fetch(`https://api.statbotics.io/v3/team_events?event=${eventKey}&limit=100`);
        if (sbResp.ok) {
            const sbJson = await sbResp.json();
            const sbList = sbJson.data || sbJson.results || sbJson;
            if (Array.isArray(sbList)) {
                for (const te of sbList) sbMap[te.team] = te;
            }
        }
    } catch (e) {
        console.warn('Statbotics event endpoint unreachable — will use year-level data per team');
    }

    // 3. Progress bar setup
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    const totalTeams = tbaTeams.length;
    let sbFailed = 0;

    // 4. Sync loop — always try processTeamPerformance (works without event-specific data);
    //    only fall back to a TBA stub if Statbotics is completely unreachable for that team.
    for (let i = 0; i < totalTeams; i++) {
        const tn = tbaTeams[i].team_number;
        statusDiv.innerText = `Syncing Team ${tn} (${i + 1}/${totalTeams})…`;

        try {
            // Pass event-specific data if available; processTeamPerformance handles null gracefully
            await processTeamPerformance(tn, eventKey, false, sbMap[tn] ?? null);
        } catch (e) {
            // Statbotics unreachable for this team — write a minimal TBA-only record
            console.warn(`Statbotics unavailable for team ${tn}: ${e.message}`);
            sbFailed++;
            const existing = await db.teams.get(tn);
            if (!existing) {
                await db.teams.put({
                    teamNumber:  tn,
                    teamName:    tbaNameMap[tn],
                    eventKey,
                    currentEPA:  null,
                    autoEPA:     null,
                    teleopEPA:   null,
                    endgameEPA:  null,
                    epa:         null,
                    rawStatboticsData: [],
                    lastUpdated: Date.now(),
                });
            } else if (tbaNameMap[tn] && tbaNameMap[tn] !== `Team ${tn}`) {
                await db.teams.update(tn, { teamName: tbaNameMap[tn] });
            }
        }

        progressBar.style.width = `${((i + 1) / totalTeams) * 100}%`;
        displayTeams();
    }

    // 5. Wrap up
    const sbNote = sbFailed > 0 ? ` (${sbFailed} team${sbFailed > 1 ? 's' : ''} missing Statbotics data)` : '';
    statusDiv.innerText = `✅ Sync complete! Loaded ${totalTeams} teams${sbNote}.`;
    setSyncTimestamp('statboticsProjections');
    progressBar.style.background = '#10b981';
    setTimeout(() => {
        progressContainer.style.display = 'none';
        progressBar.style.background = '#3b82f6';
    }, 2000);

    displayTeams();
}

window.syncStatboticsLive = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an Event Key first.'); return; }

    const statusDiv = document.getElementById('status');
    statusDiv.textContent = 'Fetching live Statbotics match data…';

    // Two parallel calls: match-by-match EPA for timeline, and event-level for component EPAs
    const [matchResp, teamEvResp] = await Promise.all([
        fetch(`https://api.statbotics.io/v3/team_matches?event=${eventKey}&limit=1000`),
        fetch(`https://api.statbotics.io/v3/team_events?event=${eventKey}&limit=100`),
    ]);
    const json = await matchResp.json();
    const eventMatches = json.data || json.results || json;

    if (!Array.isArray(eventMatches) || !eventMatches.length) {
        statusDiv.textContent = 'No Statbotics match data returned for this event.';
        return;
    }

    // Build team_event map for component EPA lookup
    const teamEvMap = {};
    if (teamEvResp.ok) {
        const teJson = await teamEvResp.json();
        const teList = teJson.data || teJson.results || teJson;
        if (Array.isArray(teList)) {
            for (const te of teList) teamEvMap[te.team] = te;
        }
    }

    // Group match records by team
    const byTeam = {};
    for (const m of eventMatches) {
        const tn = m.team;
        if (!byTeam[tn]) byTeam[tn] = [];
        byTeam[tn].push(m);
    }
    for (const tn of Object.keys(byTeam)) {
        byTeam[tn].sort((a, b) => (a.time || 0) - (b.time || 0));
    }

    const allTeams = await db.teams.toArray();
    let updated = 0;

    // Create records for teams not yet in db — allows live-only sync without history sync
    const existingNums = new Set(allTeams.map(t => t.teamNumber));
    const toCreate = [];
    for (const [tnStr, matches] of Object.entries(byTeam)) {
        const tn = parseInt(tnStr);
        if (existingNums.has(tn)) continue;
        const teData = teamEvMap[tn];
        const evEPA  = teData?.epa ?? null;
        const teBd   = evEPA?.breakdown ?? null;
        const played = matches.filter(m => m.status === 'Completed' && m.epa?.post != null);
        const latestEPA = played.length ? played[played.length - 1].epa.post
                        : (evEPA?.total_points?.mean ?? null);
        toCreate.push({
            teamNumber:  tn,
            teamName:    `Team ${tn}`,
            eventKey,
            currentEPA:  latestEPA,
            autoEPA:     teBd?.auto_points    ?? null,
            teleopEPA:   teBd?.teleop_points  ?? null,
            endgameEPA:  teBd?.endgame_points ?? null,
            epa:         evEPA,
            rawStatboticsData: matches,
            lastUpdated: Date.now(),
        });
        updated++;
    }
    if (toCreate.length) await db.teams.bulkPut(toCreate);

    for (const team of allTeams) {
        const tn = team.teamNumber;
        const newMatches = byTeam[tn];
        if (!newMatches) continue;

        const played = newMatches.filter(m => m.status === 'Completed' && m.epa?.post != null);
        if (!played.length) continue;

        const latestEPA = played[played.length - 1].epa.post;

        // Merge event matches into existing history (replace same match keys, keep the rest)
        const eventKeys = new Set(newMatches.map(m => m.match));
        const baseHistory = (team.rawStatboticsData || []).filter(m => !eventKeys.has(m.match));
        const merged = [...baseHistory, ...newMatches].sort((a, b) => (a.time || 0) - (b.time || 0));

        const teData = teamEvMap[tn];
        const evEPA  = teData?.epa ?? null;
        const teBd   = evEPA?.breakdown ?? null;

        await db.teams.update(tn, {
            currentEPA: latestEPA,
            rawStatboticsData: merged,
            ...(evEPA ? {
                autoEPA:    teBd?.auto_points    ?? team.autoEPA,
                teleopEPA:  teBd?.teleop_points  ?? team.teleopEPA,
                endgameEPA: teBd?.endgame_points ?? team.endgameEPA,
                epa:        evEPA,
            } : {}),
        });
        updated++;
    }

    displayTeams();
    await renderAtAGlance();
    setSyncTimestamp('statboticsLive');
    statusDiv.textContent = `✅ Live Statbotics sync complete — ${updated} team${updated !== 1 ? 's' : ''} updated.`;
};

async function _fetchAndStoreWebcasts(eventKey) {
    const evData = await fetchTBA(`/event/${eventKey}`);
    const webcasts = (evData.webcasts || []).filter(w => w.type === 'youtube');
    if (YT_KEY && webcasts.length > 0) {
        const ids = webcasts.map(w => w.channel).join(',');
        const ytData = await fetch(
            `https://www.googleapis.com/youtube/v3/videos?id=${ids}&part=liveStreamingDetails&key=${YT_KEY}`
        ).then(r => r.json());
        for (const item of (ytData.items || [])) {
            const wc = webcasts.find(w => w.channel === item.id);
            const startStr = item.liveStreamingDetails?.actualStartTime;
            if (wc && startStr) wc.startTimestamp = Math.floor(new Date(startStr).getTime() / 1000);
        }
    }
    localStorage.setItem(`webcasts_${eventKey}`, JSON.stringify(webcasts));
}

window.syncSchedule = async function () {
    const eventKey = document.getElementById('eventKeyInput').value.trim().toLowerCase();
    if (!eventKey) return alert("Please enter an Event Key.");

    const statusDiv = document.getElementById('status');
    statusDiv.innerText = "Fetching Schedule from TBA...";

    try {
        // TBA_API_KEY should be defined at the top of your file
        const response = await fetch(`https://www.thebluealliance.com/api/v3/event/${eventKey}/matches/simple`, {
            headers: { 'X-TBA-Auth-Key': TBA_KEY }
        });

        if (!response.ok) throw new Error("TBA Key invalid or Event not found.");

        const matches = await response.json();
        const qualMatches = matches
            .filter(m => m.comp_level === 'qm')
            .sort((a, b) => a.match_number - b.match_number);

        await db.matches.bulkPut(qualMatches.map(m => ({
            key: m.key,
            eventKey: eventKey,
            matchNumber: m.match_number,
            red: m.alliances.red.team_keys.map(t => t.replace('frc', '')),
            blue: m.alliances.blue.team_keys.map(t => t.replace('frc', '')),
            redScore: m.alliances.red.score,
            blueScore: m.alliances.blue.score,
            predictedTime: m.predicted_time || null,
            actualTime: m.actual_time || null,
            videos: (m.videos || []).filter(v => v.type === 'youtube').map(v => v.key),
        })));

        // Fetch webcasts from event metadata, then enrich with YouTube stream start times
        try { await _fetchAndStoreWebcasts(eventKey); } catch (e) { console.warn('Could not fetch webcasts:', e); }

        statusDiv.innerText = "✅ Schedule Sync Complete!";
        displaySchedule();
        maybeAutoActivateNexus();
        startNexusDirectPolling();
    } catch (err) {
        console.error(err);
        statusDiv.innerText = "❌ TBA Schedule Sync Failed.";
    }

    localStorage.setItem('lastEventKey', eventKey);
};

window.syncAll = async function () {
    const eventKey = document.getElementById('eventKeyInput').value.trim();
    if (!eventKey) return alert("Please enter an Event Key.");

    const statusDiv = document.getElementById('status');

    // Run them sequentially so the status messages don't fight
    statusDiv.innerText = "🚀 Starting Master Sync...";

    await window.syncProjections();
    await window.syncSchedule();
    await window.syncTBAOPR();
    await window.syncTBAMatches();
    await renderAtAGlance();

    statusDiv.innerText = "🎉 All systems up to date!";
};

// ── Auto-sync ─────────────────────────────────────────────────────────────

let _autoSyncTimer = null;
let _autoSyncTick  = null;
let _autoSyncCountdown = 0;

// Statbotics is triggered by TBA non-304s, not a fixed clock.
// A random 3–5 min per-device jitter spreads load across devices.
let _statboticsJitterTimer  = null;  // one-shot setTimeout
let _statboticsJitterFireAt = null;  // epoch ms when it will fire
let _statboticsLastMs       = null;  // epoch ms of last completed syncStatboticsLive
let _localEpaLastMs         = null;  // epoch ms of last completed computeLocalEPA
let _snapshotStale          = false; // true when EPA/OPR changed since last prediction recompute

function _scheduleJitteredStatbotics() {
    if (_statboticsJitterTimer) return; // already pending
    const delayMs = 180_000 + Math.random() * 120_000; // 3–5 min
    _statboticsJitterFireAt = Date.now() + delayMs;
    _updateSyncDataStatus();
    _statboticsJitterTimer = setTimeout(async () => {
        _statboticsJitterTimer  = null;
        _statboticsJitterFireAt = null;
        await window.syncStatboticsLive();
        _statboticsLastMs = Date.now();
        _updateSyncDataStatus();
    }, delayMs);
}

function _fmtAgo(ms) {
    const s = Math.floor(ms / 1000);
    if (s < 90)  return 'just now';
    const m = Math.floor(s / 60);
    if (m < 60)  return `${m}m ago`;
    return `${Math.floor(m / 60)}h ago`;
}

function _updateSyncDataStatus() {
    const el = document.getElementById('sync-data-status');
    if (!el) return;
    const now = Date.now();
    const parts = [];

    if (_tbaLastFreshMs) {
        parts.push(`TBA update: ${_fmtAgo(now - _tbaLastFreshMs)}`);
    }

    if (isLocalEpaEnabled()) {
        if (_localEpaLastMs) parts.push(`Local EPA: ${_fmtAgo(now - _localEpaLastMs)}`);
    } else {
        if (_statboticsJitterFireAt) {
            const inMin = Math.ceil(Math.max(0, _statboticsJitterFireAt - now) / 60_000);
            parts.push(`Statbotics: updating in ~${inMin}m`);
        } else if (_statboticsLastMs) {
            parts.push(`Statbotics: ${_fmtAgo(now - _statboticsLastMs)}`);
        }
    }

    el.textContent = parts.join('  ·  ');
}

async function _runTBASyncs() {
    _tbaGotFreshData = false;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const hasSchedule = eventKey
        ? (await db.matches.where('eventKey').equals(eventKey).count()) > 0
        : false;

    if (!hasSchedule) {
        // No schedule yet — try to fetch it first
        await window.syncTBAMatches();
        const nowHasSchedule = eventKey
            ? (await db.matches.where('eventKey').equals(eventKey).count()) > 0
            : false;
        if (!nowHasSchedule) return;
        // Schedule just loaded with fresh data — run OPR and EPA immediately
        // (don't re-call syncTBAMatches; it would 304 and suppress the EPA run)
        await window.syncTBAOPR();
        if (isLocalEpaEnabled()) await computeLocalEPA();
        else _scheduleJitteredStatbotics();
        return;
    }

    await window.syncTBAOPR();
    await window.syncTBAMatches();
    if (_tbaGotFreshData) {
        if (isLocalEpaEnabled()) await computeLocalEPA();
        else _scheduleJitteredStatbotics();
    }
}

// ── Local EPA Engine ──────────────────────────────────────────────────────────
// Replaces live Statbotics sync with client-side EPA computation when enabled.
// Uses Statbotics' Kalman update: per match, each team on the alliance gets
// Δ = K × (actual − predicted_alliance) / 3  where K decays as matches played.

function isLocalEpaEnabled() { return localStorage.getItem('localEpaEnabled') === 'true'; }

function teamHasSbEventData(team) {
    const ek = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!ek || !team) return false;
    return (team.rawStatboticsData || []).some(m => m.event === ek && m.epa?.post);
}

function localEpaBadge(team = null) {
    if (!isLocalEpaEnabled()) return '';
    if (team && teamHasSbEventData(team)) return '';
    return `<span style="color:#60a5fa;font-size:0.65em;font-weight:600;margin-left:3px;">EST</span>`;
}

async function _capturePreEventEPA(teams) {
    const updates = teams
        .filter(t => t.preEventEPA == null && t.currentEPA != null)
        .map(t => db.teams.update(t.teamNumber, {
            preEventEPA:        t.currentEPA,
            preEventAutoEPA:    t.autoEPA    ?? null,
            preEventTeleopEPA:  t.teleopEPA  ?? null,
            preEventEndgameEPA: t.endgameEPA ?? null,
        }));
    await Promise.all(updates);
}

async function computeLocalEPA() {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const [teams, matches, tbaTeams] = await Promise.all([
        db.teams.toArray(), db.matches.toArray(), db.tbaTeams.toArray(),
    ]);
    if (!teams.length) return;

    // Build ignore maps to mirror OPR ignore behaviour in the local EPA replay.
    const globalIgnoredKeys = new Set(matches.filter(m => m.globallyIgnored).map(m => m.key));
    // teamNumber (int) → Set of match keys ignored for that team specifically
    const teamIgnoredKeys = {};
    for (const t of tbaTeams) {
        const keys = getTeamIgnoredKeys(t);
        if (keys.length) teamIgnoredKeys[t.teamNumber] = new Set(keys);
    }

    // Capture pre-event baseline for any teams added after local EPA was first enabled
    // (e.g. via syncStatboticsLive bulkPut). Without this, their seed falls back to
    // currentEPA which may already reflect mid-event Statbotics values.
    await _capturePreEventEPA(teams);

    // Partition teams: those where statbotics already has current-event match data
    // should never be overwritten by local estimates — restore their statbotics values.
    const sbTeams = [];
    const localTeams = [];
    for (const t of teams) {
        const sbEventMatches = (t.rawStatboticsData || [])
            .filter(m => eventKey && m.event === eventKey && m.epa?.post)
            .sort((a, b) => (a.time || 0) - (b.time || 0));
        if (sbEventMatches.length > 0) {
            t._sbLastEPA = sbEventMatches[sbEventMatches.length - 1].epa.post;
            sbTeams.push(t);
        } else {
            localTeams.push(t);
        }
    }

    // Restore statbotics values for teams that have current-event data, clearing any local overrides
    if (sbTeams.length > 0) {
        await Promise.all(sbTeams.map(t => db.teams.update(t.teamNumber, {
            currentEPA:       t._sbLastEPA,
            localEPATimeline: [],
        })));
    }

    if (!localTeams.length) {
        _localEpaLastMs = Date.now();
        _updateSyncDataStatus();
        if (activeTeamNumber) {
            const refreshed = await db.teams.get(activeTeamNumber);
            if (refreshed) activeTeamData = refreshed;
        }
        await refreshEPADisplays(activeTeamNumber);
        if (activeTeamData && lastDetailDataSubTab === 'epa') renderChart(activeTeamData);
        return;
    }

    // Local Kalman computation for teams without statbotics event data.
    // Seed n from career match count so K starts at the right level per Statbotics percent_func.
    const epaState = {};
    for (const t of localTeams) {
        const careerN = (t.rawStatboticsData || []).filter(m => m.epa?.post).length;
        epaState[t.teamNumber] = {
            current:  t.preEventEPA        ?? t.currentEPA  ?? 0,
            auto:     t.preEventAutoEPA     ?? t.autoEPA     ?? 0,
            endgame:  t.preEventEndgameEPA  ?? t.endgameEPA  ?? 0,
            n: careerN,
        };
    }
    // Seed sbTeams with their pre-event EPA so alliance predictions are accurate during replay.
    // Without this, getE() falls back to {current:0}, corrupting error terms for localTeams
    // that share alliances with sbTeams and causing their EPA to drift on every sync.
    for (const t of sbTeams) {
        const firstEventMatch = (t.rawStatboticsData || [])
            .filter(m => eventKey && m.event === eventKey && m.epa?.post)
            .sort((a, b) => (a.time || 0) - (b.time || 0))[0];
        const preEpa = firstEventMatch?.epa?.pre ?? t.preEventEPA ?? t.currentEPA ?? 0;
        const careerN = (t.rawStatboticsData || []).filter(m => m.epa?.post).length;
        epaState[t.teamNumber] = {
            current: preEpa,
            auto:    t.preEventAutoEPA    ?? t.autoEPA    ?? 0,
            endgame: t.preEventEndgameEPA ?? t.endgameEPA ?? 0,
            n:       careerN,
        };
    }
    const getE = tn => epaState[tn] || (epaState[tn] = { current: 0, auto: 0, endgame: 0, n: 0 });

    const played = matches
        .filter(m => (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !globalIgnoredKeys.has(m.key))
        .sort((a, b) => (a.actualTime || a.predictedTime || 0) - (b.actualTime || b.predictedTime || 0));

    // Per-team timeline: String(teamNumber) -> [{label, epa}]
    const teamTimeline = {};

    for (const m of played) {
        const matchLabel = (!m.compLevel || m.compLevel === 'qm') ? `Q${m.matchNumber}` : `P${m.matchNumber}`;
        const teamsInMatch = [];

        for (const [alliance, score, bd] of [
            [m.red  || [], m.redScore,  m.redBreakdown],
            [m.blue || [], m.blueScore, m.blueBreakdown],
        ]) {
            const members = alliance.map(t => String(t).replace(/^frc/i, ''));
            if (!members.length) continue;

            // Predictions use all alliance members (including any ignored ones) so
            // the baseline score expectation is accurate even when a team is skipped.
            const predTotal   = members.reduce((s, t) => s + getE(t).current, 0);
            const predAuto    = members.reduce((s, t) => s + getE(t).auto,    0);
            const predEndgame = members.reduce((s, t) => s + getE(t).endgame, 0);
            const autoActual    = bd?.totalAutoPoints ?? null;
            const endgameActual = bd ? ((bd.endGameTowerPoints || 0) + (bd['Hub Endgame Fuel Count'] || 0)) : null;
            const N = members.length || 1;

            for (const t of members) {
                // Skip EPA update if this match is individually ignored for this team.
                // The team still participates in the alliance prediction above.
                if (teamIgnoredKeys[parseInt(t)]?.has(m.key)) continue;
                const e = getE(t);
                e.n++;
                const prev = Math.min(0.5, Math.max(0.3, 0.5 - (0.2 / 6) * (e.n - 6)));
                const K = (2 / 3) * prev; // matches Statbotics percent_func for 2016+
                e.current += K * (score - predTotal)   / N;
                if (autoActual    != null) e.auto    += K * (autoActual    - predAuto)    / N;
                if (endgameActual != null) e.endgame += K * (endgameActual - predEndgame) / N;
                teamsInMatch.push(t);
            }
        }

        // Snapshot EPA for all teams that played in this match (after both alliances updated)
        for (const t of teamsInMatch) {
            if (!teamTimeline[t]) teamTimeline[t] = [];
            teamTimeline[t].push({ label: matchLabel, epa: getE(t).current });
        }
    }

    await Promise.all(localTeams.map(t => {
        const e = epaState[t.teamNumber];
        if (!e || e.n === 0) return null;
        return db.teams.update(t.teamNumber, {
            currentEPA:         e.current,
            autoEPA:            e.auto,
            teleopEPA:          e.current - e.auto - e.endgame,
            endgameEPA:         e.endgame,
            localEPATimeline:   teamTimeline[String(t.teamNumber)] || [],
        });
    }).filter(Boolean));

    _localEpaLastMs = Date.now();
    _updateSyncDataStatus();

    // Refresh in-memory active team so renderOverview + chart use fresh EPA values
    if (activeTeamNumber) {
        const refreshed = await db.teams.get(activeTeamNumber);
        if (refreshed) activeTeamData = refreshed;
    }
    await refreshEPADisplays(activeTeamNumber);

    // Re-render chart if the EPA sub-tab is currently visible
    if (activeTeamData && lastDetailDataSubTab === 'epa') {
        renderChart(activeTeamData);
    }

    _setSnapshotStale(true);
}

function updateLocalEpaUI() {
    const btn   = document.getElementById('localEpaBtn');
    const label = document.getElementById('localEpaLabel');
    const enabled = isLocalEpaEnabled();
    if (btn)   btn.textContent   = enabled ? '✅' : '⬜';
    if (label) label.textContent = enabled ? 'On' : 'Off';
}

window.toggleLocalEpa = async function () {
    const enabling = !isLocalEpaEnabled();
    localStorage.setItem('localEpaEnabled', String(enabling));
    updateLocalEpaUI();
    _updateSyncDataStatus();

    if (enabling) {
        const teams = await db.teams.toArray();
        await _capturePreEventEPA(teams);
        await computeLocalEPA();
    } else {
        // Restore pre-event Statbotics baseline
        const teams = await db.teams.toArray();
        const ek = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
        await Promise.all(teams
            .filter(t => t.preEventEPA != null && !(ek && (t.rawStatboticsData || []).some(m => m.event === ek && m.epa?.post)))
            .map(t => db.teams.update(t.teamNumber, {
                currentEPA:       t.preEventEPA,
                autoEPA:          t.preEventAutoEPA    ?? t.autoEPA,
                teleopEPA:        t.preEventTeleopEPA  ?? t.teleopEPA,
                endgameEPA:       t.preEventEndgameEPA ?? t.endgameEPA,
                localEPATimeline: [],
            })));
        if (activeTeamNumber) {
            const refreshed = await db.teams.get(activeTeamNumber);
            if (refreshed) activeTeamData = refreshed;
        }
        await refreshEPADisplays(activeTeamNumber);
        if (activeTeamData && lastDetailDataSubTab === 'epa') {
            renderChart(activeTeamData);
        }
    }
};

function _updateAutoSyncStatus() {
    const el = document.getElementById('autoSyncStatus');
    if (!el) return;
    const m = Math.floor(_autoSyncCountdown / 60);
    const s = String(_autoSyncCountdown % 60).padStart(2, '0');
    el.textContent = `Next sync in ${m}:${s}`;
}

window.toggleAutoSync = function () {
    if (_autoSyncTimer) {
        clearInterval(_autoSyncTimer);
        clearInterval(_autoSyncTick);
        clearTimeout(_statboticsJitterTimer);
        _autoSyncTimer         = null;
        _autoSyncTick          = null;
        _statboticsJitterTimer  = null;
        _statboticsJitterFireAt = null;
        const btn = document.getElementById('autoSyncBtn');
        const sel = document.getElementById('autoSyncInterval');
        if (btn) { btn.textContent = 'Start Auto-Sync'; btn.style.background = '#059669'; }
        if (sel) sel.disabled = false;
        const status = document.getElementById('autoSyncStatus');
        if (status) status.textContent = '';
        _updateSyncDataStatus();
    } else {
        const minutes = parseInt(document.getElementById('autoSyncInterval')?.value || '5', 10);
        const totalSeconds = minutes * 60;
        _autoSyncCountdown = totalSeconds;

        // TBA immediately; if fresh data comes back, triggers local EPA or jittered Statbotics
        _runTBASyncs();
        // Statbotics baseline — skipped when local EPA mode is active
        if (!isLocalEpaEnabled()) {
            window.syncStatboticsLive().then(() => {
                _statboticsLastMs = Date.now();
                _updateSyncDataStatus();
            });
        }

        _autoSyncTimer = setInterval(() => {
            _autoSyncCountdown = totalSeconds;
            _runTBASyncs();
        }, totalSeconds * 1000);

        _autoSyncTick = setInterval(() => {
            _autoSyncCountdown = Math.max(0, _autoSyncCountdown - 1);
            _updateAutoSyncStatus();
            _updateSyncDataStatus();
        }, 1000);

        const btn = document.getElementById('autoSyncBtn');
        const sel = document.getElementById('autoSyncInterval');
        if (btn) { btn.textContent = 'Stop Auto-Sync'; btn.style.background = '#ef4444'; }
        if (sel) sel.disabled = true;
        _updateAutoSyncStatus();
    }
};






// ── Scouting data sync ────────────────────────────────────────────────────────

let _scoutingAutoSyncTimer = null;
let _scoutingAutoSyncTick  = null;
let _scoutingAutoSyncCountdown = 0;

function _updateScoutingAutoSyncStatus() {
    const el = document.getElementById('scoutingAutoSyncStatus');
    if (!el) return;
    const m = Math.floor(_scoutingAutoSyncCountdown / 60);
    const s = String(_scoutingAutoSyncCountdown % 60).padStart(2, '0');
    el.textContent = `Next sync in ${m}:${s}`;
}

// Imports a parsed archive bundle into localStorage + IndexedDB.
// Returns a summary string.
async function importArchiveBundle(data, eventKey) {
    localStorage.removeItem(`scoutingFusedStats_${eventKey}`);
    // Detect a structured archive bundle by eventKey presence (not by scoutingRows, which may be empty)
    if (!Array.isArray(data) && (data.eventKey || data.teams || data.scoutingRows)) {
        if (Array.isArray(data.scoutingRows) && data.scoutingRows.length) {
            localStorage.setItem(`scoutingData_${eventKey}`, JSON.stringify(data.scoutingRows));
        }
        if (data.pitRows?.length) localStorage.setItem(`pitData_${eventKey}`, JSON.stringify(data.pitRows));
        if (data.teams?.length)    await db.teams.bulkPut(data.teams);
        if (data.tbaTeams?.length) await db.tbaTeams.bulkPut(data.tbaTeams);
        if (data.matches?.length)  await db.matches.bulkPut(data.matches);
        if (data.matchTracks?.length) {
            try {
                await db.matchTracks.bulkPut(data.matchTracks);
                // Both caches key off what is in Dexie, and neither notices a bulkPut.
                _tracksManifest = null;
                routesRenderedFor = null;
            } catch { /* older schema without the table: routes simply stay unavailable */ }
        }
        if (data.tbaAlliances?.length) {
            localStorage.setItem(`tbaAlliances_${eventKey}`, JSON.stringify(data.tbaAlliances));
            const alliances = Array.from({ length: 8 }, (_, i) => {
                const a = data.tbaAlliances[i];
                if (!a) return { captain: null, pick1: null, pick2: null };
                const strip = key => a.picks[key]?.replace(/^frc/i, '') || null;
                return { captain: strip(0), pick1: strip(1), pick2: strip(2) };
            });
            localStorage.setItem('realDraftState', JSON.stringify({ alliances, currentAlliance: 8, currentRound: 2 }));
        }
        if (typeof data.calibrationBeta === 'number' && !isNaN(data.calibrationBeta)) {
            localStorage.setItem(`wlCalibrationBeta_${eventKey}`, String(data.calibrationBeta));
        }
        if (data.preEventSnapshot) {
            localStorage.setItem(`wlPreEventSnapshot_${data.eventKey}`, JSON.stringify(data.preEventSnapshot));
        }
        // Config fields (present in both config and full archives)
        if (data.rpThresholds)     localStorage.setItem(`rpThresholds_${eventKey}`, JSON.stringify(data.rpThresholds));
        if (data.webcasts)         localStorage.setItem(`webcasts_${eventKey}`, JSON.stringify(data.webcasts));
        if (data.eventNotes)       localStorage.setItem(`eventNotes_${eventKey}`, JSON.stringify(data.eventNotes));
        if (data.scoutingSheetUrl) localStorage.setItem(`scoutingSheetUrl_${eventKey}`, data.scoutingSheetUrl);
        if (data.pitSheetUrl)      localStorage.setItem(`pitSheetUrl_${eventKey}`, data.pitSheetUrl);
        if (data.pickListOrder)    localStorage.setItem('pickListOrder', JSON.stringify(data.pickListOrder));
        if (data.draftConfig) {
            if (data.draftConfig.mode)             localStorage.setItem('draftMode', data.draftConfig.mode);
            if (data.draftConfig.numAlliances)     localStorage.setItem('draftNumAlliances', data.draftConfig.numAlliances);
            if (data.draftConfig.picksPerAlliance) localStorage.setItem('draftPicksPerAlliance', data.draftConfig.picksPerAlliance);
            if (data.draftConfig.epaWeights)       localStorage.setItem('draftEPAWeights', JSON.stringify(data.draftConfig.epaWeights));
        }
        if (data.nexusConfig) {
            if (data.nexusConfig.relayUrl)          localStorage.setItem('nexusRelayUrl', data.nexusConfig.relayUrl);
            if (data.nexusConfig.enabled != null)   localStorage.setItem('nexusEnabled', data.nexusConfig.enabled);
            if (data.nexusConfig.eventKeyOverride)  localStorage.setItem('nexusEventKeyOverride', data.nexusConfig.eventKeyOverride);
            const urlIn = document.getElementById('nexusRelayUrlInput');
            if (urlIn) urlIn.value = data.nexusConfig.relayUrl || '';
            const keyIn = document.getElementById('nexusEventKeyOverride');
            if (keyIn) keyIn.value = data.nexusConfig.eventKeyOverride || '';
            updateNexusUI();
        }
        if (data.localEpaEnabled != null) { localStorage.setItem('localEpaEnabled', data.localEpaEnabled); updateLocalEpaUI(); }

        _recordLocalArchive(eventKey, data.archiveType);
        const scoutCount = data.scoutingRows?.length || 0;
        const type = data.archiveType === 'config' ? 'Config archive' : 'Archive';
        return `${type} loaded: ${data.teams?.length || 0} teams + ${data.matches?.length || 0} matches + ${scoutCount} scouting rows`;
    } else {
        // Legacy: bare array of scouting rows
        localStorage.setItem(`scoutingData_${eventKey}`, JSON.stringify(data));
        return `${data.length} rows loaded`;
    }
}

// Updates OBE (overtaken by events) indicators on sync buttons based on stored archive coverage.
// coverage keys: teams, tbaTeams, matches, breakdowns, scouting
function updateOBEStatus(eventKey) {
    const raw = eventKey ? localStorage.getItem(`archiveCoverage_${eventKey}`) : null;
    const cov = raw ? JSON.parse(raw) : {};

    const entries = [
        { btnId: 'btn-syncProjections',    spanId: 'ts-statboticsProjections', covered: !!cov.teams,      setText: true },
        { btnId: 'btn-syncSchedule',       spanId: 'ts-schedule',              covered: !!cov.matches,    setText: true },
        { btnId: 'btn-syncStatboticsLive', spanId: 'ts-statboticsLive',        covered: !!cov.teams,      setText: false },
        { btnId: 'btn-syncTBAOPR',         spanId: 'ts-tbaOPR',                covered: !!cov.tbaTeams,   setText: false },
        { btnId: 'btn-syncTBAMatches',     spanId: 'ts-tbaMatches',            covered: !!cov.breakdowns, setText: false },
        { btnId: 'btn-syncScoutingData',   spanId: 'scouting-sync-status',     covered: !!cov.scouting,   setText: false },
        { btnId: 'btn-syncPitData',        spanId: 'pit-sync-status',          covered: !!cov.pit,        setText: false },
    ];

    for (const { btnId, spanId, covered, setText } of entries) {
        const btn  = document.getElementById(btnId);
        const span = document.getElementById(spanId);
        if (btn)  btn.style.opacity = covered ? '0.5' : '';
        if (span) {
            if (setText) span.textContent = covered ? '✓ from archive' : '';
            span.style.color = covered ? '#4ade80' : '#475569';
        }
    }
}

// Returns the first available archive URL for the event key, or null if none found.
// Priority: full → config → legacy plain archive.
async function findArchiveUrl(eventKey) {
    const base = import.meta.env.BASE_URL;
    const candidates = [
        `${base}scouting/${eventKey}_full_archive.json`,
        `${base}scouting/${eventKey}_config_archive.json`,
        `${base}scouting/${eventKey}_archive.json`,
    ];
    for (const url of candidates) {
        try {
            const resp = await fetch(url, { method: 'HEAD' });
            const ct = resp.headers.get('content-type') || '';
            if (resp.ok && ct.includes('json')) return url;
        } catch {}
    }
    return null;
}

// Checks for a public archive file for the given event key and updates #archiveHint.
let _archiveCheckTimer = null;
async function checkEventArchive(eventKey) {
    const hint = document.getElementById('archiveHint');
    if (!hint) return;
    if (!eventKey || eventKey.length < 6) { hint.innerHTML = ''; return; }

    const url = await findArchiveUrl(eventKey);
    if (url) {
        hint.innerHTML = `<button onclick="loadEventArchive('${eventKey}')" style="background:#1e3a5f;color:#60a5fa;border:1px solid #3b82f6;font-size:0.82em;padding:6px 12px;">↓ Load Archived Data</button>`;
    } else {
        hint.innerHTML = `<span style="color:#475569;font-size:0.8em;">No archive available</span>`;
    }
}

window.loadEventArchive = async function (eventKey) {
    if (!eventKey) eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('No event key — type one in the Event Key field first.'); return; }

    const hint = document.getElementById('archiveHint');
    if (hint) hint.innerHTML = `<span style="color:#64748b;font-size:0.82em;">Loading…</span>`;
    const url = await findArchiveUrl(eventKey);
    if (!url) {
        if (hint) hint.innerHTML = `<span style="color:#475569;font-size:0.8em;">No archive available</span>`;
        return;
    }
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const summary = await importArchiveBundle(data, eventKey);

        // Store per-field coverage so updateOBEStatus can dim the right buttons
        localStorage.setItem(`archiveCoverage_${eventKey}`, JSON.stringify({
            teams:      !!(data.teams?.length),
            tbaTeams:   !!(data.tbaTeams?.length),
            matches:    !!(data.matches?.length),
            breakdowns: !!(data.matches?.some(m => m.redBreakdown || m.blueBreakdown)),
            scouting:   !!(data.scoutingRows?.length),
            pit:        !!(data.pitRows?.length),
        }));
        updateOBEStatus(eventKey);

        // Restore event key to the input so subsequent actions work
        const keyInput = document.getElementById('eventKeyInput');
        if (keyInput) keyInput.value = eventKey;
        localStorage.setItem('lastEventKey', eventKey);
        updateAppEventKey(eventKey);

        // Set sync timestamps for all bundled data sources so the Home tab shows them as synced
        const archiveTime = data.archived ? new Date(data.archived).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const archiveLabel = `Archive (${archiveTime})`;
        for (const key of ['statboticsLive', 'tbaOPR', 'tbaMatches', 'scoutingData']) {
            localStorage.setItem(`lastSync_${key}`, archiveLabel);
            const el = document.getElementById(`ts-${key}`);
            if (el) el.textContent = `Last sync: ${archiveLabel}`;
        }

        if (hint) hint.innerHTML = `<span style="color:#4ade80;font-size:0.82em;">✓ ${summary}</span>`;

        // Run TBA fusion now so the dashboard shows fused EPA immediately
        try { await computeScoutingFusion(); } catch (_) {}

        // Refresh all display surfaces
        renderScoutingSection();
        await Promise.all([
            displayTeams(),
            displaySchedule(),
            displayTBATeams(),
        ]);
        await renderAtAGlance();
        displayScoutingTeams();
        renderPickList();
        renderDraft();
        maybeAutoActivateNexus();
        startNexusDirectPolling();

        // Auto-fetch webcasts if the archive had matches but no webcast data was bundled
        const hasMatches = !!(data.matches?.length);
        const hasWebcasts = !!(JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) || '[]').length);
        if (hasMatches && !hasWebcasts) {
            _fetchAndStoreWebcasts(eventKey).catch(e => console.warn('Could not fetch webcasts:', e));
        }
    } catch (err) {
        if (hint) hint.innerHTML = `<span style="color:#ef4444;font-size:0.82em;">Error: ${err.message}</span>`;
    }
};

window.syncScoutingData = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an Event Key first.'); return; }
    const rawSource = getScoutingSource(eventKey);
    if (!rawSource) { alert('No scouting sheet source configured for this event.'); return; }
    const source = rawSource.endsWith('.json') ? rawSource : (sheetsInputToCsvUrl(rawSource) || rawSource);

    const statusEl = document.getElementById('scouting-sync-status');
    if (statusEl) statusEl.textContent = 'Syncing…';

    try {
        const resp = await fetch(source);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = rawSource.endsWith('.json') ? await resp.json() : parseCSV(await resp.text());
        const now = new Date().toLocaleTimeString();
        localStorage.setItem('lastSync_scoutingData', now);
        const summary = await importArchiveBundle(data, eventKey);
        if (statusEl) statusEl.textContent = summary.startsWith('Archive') ? summary : `Last sync: ${now} · ${summary}`;
        renderScoutingSection();
    } catch (err) {
        const msg = err.message === 'Failed to fetch'
            ? 'Failed to fetch — check that the sheet is shared publicly (Anyone with the link → Viewer)'
            : `Error: ${err.message}`;
        if (statusEl) statusEl.textContent = msg;
    }
};

window.toggleScoutingAutoSync = function () {
    if (_scoutingAutoSyncTimer) {
        clearInterval(_scoutingAutoSyncTimer);
        clearInterval(_scoutingAutoSyncTick);
        _scoutingAutoSyncTimer = _scoutingAutoSyncTick = null;
        const btn = document.getElementById('scoutingAutoSyncBtn');
        const sel = document.getElementById('scoutingAutoSyncInterval');
        const sts = document.getElementById('scoutingAutoSyncStatus');
        if (btn) { btn.textContent = 'Start Auto-Sync'; btn.style.background = '#059669'; }
        if (sel) sel.disabled = false;
        if (sts) sts.textContent = '';
    } else {
        const minutes = parseInt(document.getElementById('scoutingAutoSyncInterval')?.value || '5', 10);
        const total = minutes * 60;
        _scoutingAutoSyncCountdown = total;
        const _runScoutingAutoSync = () => {
            const ek = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
            window.syncScoutingData();
            if (ek) _syncPitDataForEvent(ek);
        };
        _runScoutingAutoSync();
        _scoutingAutoSyncTimer = setInterval(() => {
            _scoutingAutoSyncCountdown = total;
            _runScoutingAutoSync();
        }, total * 1000);
        _scoutingAutoSyncTick = setInterval(() => {
            _scoutingAutoSyncCountdown = Math.max(0, _scoutingAutoSyncCountdown - 1);
            _updateScoutingAutoSyncStatus();
        }, 1000);
        const btn = document.getElementById('scoutingAutoSyncBtn');
        const sel = document.getElementById('scoutingAutoSyncInterval');
        if (btn) { btn.textContent = 'Stop Auto-Sync'; btn.style.background = '#ef4444'; }
        if (sel) sel.disabled = true;
        _updateScoutingAutoSyncStatus();
    }
};

function sheetsInputToCsvUrl(input) {
    // Accept: bare sheet ID, any Google Sheets URL (edit/view/pub), or existing CSV export URL
    const idMatch = input.match(/\/spreadsheets\/d\/([A-Za-z0-9_-]+)/);
    const id = idMatch ? idMatch[1] : input.replace(/\s/g, '');
    if (!id) return null;
    // Preserve gid if present in the pasted URL (query param or fragment), otherwise default to first sheet (gid=0)
    const gidMatch = input.match(/[?&#]gid=(\d+)/);
    const gid = gidMatch ? gidMatch[1] : '0';
    return `https://docs.google.com/spreadsheets/d/${id}/export?format=csv&gid=${gid}`;
}

window.saveScoutingSheetUrl = function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const raw = document.getElementById('scoutingUrlInput')?.value.trim();
    if (!raw || !eventKey) return;
    const url = sheetsInputToCsvUrl(raw);
    if (!url) return;
    localStorage.setItem(`scoutingSheetUrl_${eventKey}`, url);
    renderScoutingSection();
};

window.savePitSheetUrl = function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const raw = document.getElementById('pitUrlInput')?.value.trim();
    if (!raw || !eventKey) return;
    const url = sheetsInputToCsvUrl(raw);
    if (!url) return;
    localStorage.setItem(`pitSheetUrl_${eventKey}`, url);
    renderScoutingSection();
};

async function _syncPitDataForEvent(eventKey) {
    const raw = getPitSource(eventKey);
    if (!raw) return; // no pit source — skip silently
    const source = sheetsInputToCsvUrl(raw) || raw;
    const statusEl = document.getElementById('pit-sync-status');
    if (statusEl) statusEl.textContent = 'Syncing…';
    try {
        const resp = await fetch(source);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const rows = parseCSV(await resp.text());
        const now = new Date().toLocaleTimeString();
        localStorage.setItem(`pitData_${eventKey}`, JSON.stringify(rows));
        localStorage.setItem(`lastSync_pitData`, now);
        if (statusEl) statusEl.textContent = `Last sync: ${now} · ${rows.length} teams`;
        renderScoutingSection();
    } catch (err) {
        const msg = err.message === 'Failed to fetch'
            ? 'Failed to fetch — check that the sheet is shared publicly (Anyone with the link → Viewer)'
            : `Error: ${err.message}`;
        if (statusEl) statusEl.textContent = msg;
    }
}

window.syncPitData = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an Event Key first.'); return; }
    if (!getPitSource(eventKey)) { alert('No pit scouting sheet configured for this event.'); return; }
    await _syncPitDataForEvent(eventKey);
};

function _recordLocalArchive(eventKey, archiveType) {
    const reg = JSON.parse(localStorage.getItem('localArchiveRegistry') || '{}');
    reg[eventKey] = { archiveType: archiveType || 'full' };
    localStorage.setItem('localArchiveRegistry', JSON.stringify(reg));
}

function _downloadBundle(bundle) {
    _recordLocalArchive(bundle.eventKey, bundle.archiveType);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }));
    a.download = `${bundle.eventKey}_${bundle.archiveType}_archive.json`;
    a.click();
    URL.revokeObjectURL(a.href);
}

function showArchiveSummaryModal(bundle) {
    const existing = document.getElementById('archiveSummaryModal');
    if (existing) existing.remove();

    const played = (bundle.matches || []).filter(m => (m.redScore ?? -1) >= 0).length;
    const unplayed = (bundle.matches || []).length - played;
    const isConfig = bundle.archiveType === 'config';

    const row = (label, value) =>
        `<tr><td style="color:#94a3b8;padding:4px 12px 4px 0;white-space:nowrap;">${label}</td>` +
        `<td style="color:#f1f5f9;font-weight:600;">${value}</td></tr>`;
    const check = v => v ? '<span style="color:#4ade80;">✓</span>' : '<span style="color:#475569;">—</span>';

    const rows = [
        row('Archive type', `<span style="color:${isConfig ? '#f59e0b' : '#60a5fa'}">${isConfig ? 'Config (pre-event)' : 'Full (post-event)'}</span>`),
        row('Teams (EPA/history)', `${bundle.teams?.length || 0} teams, ${bundle.tbaTeams?.length || 0} TBA`),
        row('Matches', isConfig ? '— (config only)' : `${bundle.matches?.length || 0} total (${played} played, ${unplayed} unplayed)`),
        row('Scouting data', isConfig ? '— (config only)' : `${bundle.scoutingRows?.length || 0} match rows, ${bundle.pitRows?.length || 0} pit rows`),
        row('Alliance picks', isConfig ? '— (config only)' : check(bundle.tbaAlliances?.length)),
        row('Robot routes', isConfig ? '— (config only)'
            : bundle.matchTracks?.length
              ? `${bundle.matchTracks.length} match${bundle.matchTracks.length === 1 ? '' : 'es'}`
                + ` (~${Math.round(JSON.stringify(bundle.matchTracks).length / 1024)} KB)`
              : check(false)),
        row('RP thresholds', check(bundle.rpThresholds)),
        row('Webcast URLs', bundle.webcasts?.length ? `${bundle.webcasts.length} streams` : check(false)),
        row('Scouting sheet URL', check(bundle.scoutingSheetUrl)),
        row('Pit sheet URL', check(bundle.pitSheetUrl)),
        row('Pick list order', bundle.pickListOrder?.length ? `${bundle.pickListOrder.filter(e => e !== '---separator---').length} teams` : check(false)),
        row('Draft config', check(bundle.draftConfig?.mode)),
        row('Nexus relay URL', check(bundle.nexusConfig?.relayUrl)),
        row('Nexus enabled', bundle.nexusConfig?.enabled != null ? bundle.nexusConfig.enabled === 'true' ? '<span style="color:#4ade80;">Yes</span>' : 'No' : check(false)),
        row('Local EPA', bundle.localEpaEnabled != null ? bundle.localEpaEnabled === 'true' ? '<span style="color:#4ade80;">Enabled</span>' : 'Disabled' : check(false)),
        row('W-L calibration beta', bundle.calibrationBeta != null ? bundle.calibrationBeta.toFixed(4) : check(false)),
        row('Pre-event prediction snapshot', check(bundle.preEventSnapshot)),
        row('Event notes', bundle.eventNotes?.length ? `${bundle.eventNotes.length} entries` : check(false)),
    ];

    const modal = document.createElement('div');
    modal.id = 'archiveSummaryModal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:500;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;padding:20px;';
    modal.innerHTML = `
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;max-width:480px;width:100%;max-height:90vh;overflow-y:auto;padding:24px;">
            <div style="font-size:1.1em;font-weight:700;color:#f1f5f9;margin-bottom:4px;">Archive Summary</div>
            <div style="font-size:0.78em;color:#64748b;margin-bottom:16px;">${bundle.eventKey} · ${new Date(bundle.archived).toLocaleString()}</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-bottom:20px;">${rows.join('')}</table>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
                <button onclick="document.getElementById('archiveSummaryModal').remove()"
                    style="background:transparent;color:#64748b;border:1px solid #334155;border-radius:6px;padding:7px 18px;cursor:pointer;font-size:0.9em;">Cancel</button>
                <button id="archiveDownloadBtn"
                    style="background:#1e40af;color:#f1f5f9;border:1px solid #3b82f6;border-radius:6px;padding:7px 18px;cursor:pointer;font-size:0.9em;font-weight:600;">↓ Download</button>
            </div>
        </div>`;
    document.body.appendChild(modal);
    document.getElementById('archiveDownloadBtn').onclick = () => {
        _downloadBundle(bundle);
        modal.remove();
    };
}

window.saveArchiveBundle = async function (mode = 'full') {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an event key first.'); return; }

    const [teams, tbaTeams, matches] = await Promise.all([
        db.teams.where('eventKey').equals(eventKey).toArray(),
        db.tbaTeams.where('eventKey').equals(eventKey).toArray(),
        db.matches.where('eventKey').equals(eventKey).toArray(),
    ]);

    const savedBeta = parseFloat(localStorage.getItem(`wlCalibrationBeta_${eventKey}`));

    // Effective RP thresholds: merge game-config defaults with any saved overrides.
    // Saved as {rpField: value} so importArchiveBundle can restore them correctly.
    const gameConfig = getGameConfig(eventKey);
    const effectiveThresholds = getEffectiveThresholds(gameConfig, eventKey);
    const rpThresholdsForArchive = effectiveThresholds.length
        ? Object.fromEntries(effectiveThresholds.filter(r => r.threshold != null).map(r => [r.rpField, r.threshold]))
        : null;

    // Filter pick list to teams at this event only (the global pickListOrder accumulates across events).
    const eventTeamSet = new Set(teams.map(t => String(t.teamNumber)));
    const rawPickOrder = JSON.parse(localStorage.getItem('pickListOrder') ?? 'null');
    const filteredPickOrder = rawPickOrder
        ? rawPickOrder.filter(entry => entry === '---separator---' || eventTeamSet.has(String(entry)))
        : null;

    const bundle = {
        eventKey,
        archived: new Date().toISOString(),
        archiveType: mode,
        // ── Team data (both modes) ──────────────────────────────────────
        teams: teams.map(({ photoUrl: _, ...rest }) => rest),
        tbaTeams,
        // ── Config / settings (both modes) ─────────────────────────────
        rpThresholds:     rpThresholdsForArchive,
        webcasts:         JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) ?? 'null'),
        eventNotes:       JSON.parse(localStorage.getItem(`eventNotes_${eventKey}`) ?? 'null'),
        scoutingSheetUrl: localStorage.getItem(`scoutingSheetUrl_${eventKey}`) ?? null,
        pitSheetUrl:      localStorage.getItem(`pitSheetUrl_${eventKey}`) ?? null,
        pickListOrder:    filteredPickOrder,
        draftConfig: {
            mode:             localStorage.getItem('draftMode') ?? null,
            numAlliances:     localStorage.getItem('draftNumAlliances') ?? null,
            picksPerAlliance: localStorage.getItem('draftPicksPerAlliance') ?? null,
            epaWeights:       JSON.parse(localStorage.getItem('draftEPAWeights') ?? 'null'),
        },
        nexusConfig: {
            relayUrl:         localStorage.getItem('nexusRelayUrl') ?? null,
            enabled:          localStorage.getItem('nexusEnabled') ?? null,
            eventKeyOverride: localStorage.getItem('nexusEventKeyOverride') ?? null,
        },
        localEpaEnabled:  localStorage.getItem('localEpaEnabled') ?? null,
        // Model calibration — useful pre-event to seed predictions
        calibrationBeta:  isNaN(savedBeta) ? null : savedBeta,
        preEventSnapshot: JSON.parse(localStorage.getItem(`wlPreEventSnapshot_${eventKey}`) ?? 'null'),
    };

    if (mode === 'full') {
        bundle.scoutingRows  = JSON.parse(localStorage.getItem(`scoutingData_${eventKey}`) ?? '[]');
        bundle.pitRows       = JSON.parse(localStorage.getItem(`pitData_${eventKey}`) ?? 'null');
        bundle.matches       = matches;
        bundle.tbaAlliances  = JSON.parse(localStorage.getItem(`tbaAlliances_${eventKey}`) ?? 'null');
        // Robot routes. Without these an archive restores everything about a past event
        // EXCEPT where the robots were, and the Routes tabs come back empty on any device
        // that cannot reach public/tracks/ -- which is the case an archive exists for.
        //
        // Size scales with how many matches were actually processed, not with the event:
        // ~143 KB each, and a realistic event has tracks for a fraction of its matches
        // (11 of 100 on 2026mawor). Only what is already cached locally is included; the
        // archive is a snapshot of what this device has, not a fetch-everything trigger.
        try {
            bundle.matchTracks = await db.matchTracks.where('eventKey').equals(eventKey).toArray();
        } catch { bundle.matchTracks = []; }
    }

    showArchiveSummaryModal(bundle);
};

// Backwards-compatible alias
window.saveScoutingArchive = () => saveArchiveBundle('full');

// ── Adjustments bundle (ignored matches, ceilings, notes) ────────────────────

async function buildAdjBundle(eventKey) {
    const [allTeams, allTBATeams] = await Promise.all([
        db.teams.where('eventKey').equals(eventKey).toArray(),
        db.tbaTeams.where('eventKey').equals(eventKey).toArray(),
    ]);
    const allNotes = JSON.parse(localStorage.getItem('teamNotes') || '{}');
    const teams = {};
    for (const t of allTBATeams) {
        if (t.ignoredMatchKeys?.length || t.scoutingIgnoreActive) {
            teams[t.teamNumber] = {
                ignoredMatchKeys: t.ignoredMatchKeys ?? [],
                scoutingIgnoreActive: t.scoutingIgnoreActive ?? false,
            };
        }
    }
    for (const t of allTeams) {
        if (t.analysis) {
            teams[t.teamNumber] = { ...(teams[t.teamNumber] ?? {}), analysis: t.analysis };
        }
    }
    return { version: 1, eventKey, teams, notes: allNotes };
}

async function applyAdjBundle(bundle) {
    const [allTBATeams, allMatches] = await Promise.all([
        db.tbaTeams.toArray(),
        db.matches.toArray(),
    ]);
    const allTeamNums = allTBATeams.map(t => t.teamNumber);
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));

    for (const [tn, adj] of Object.entries(bundle.teams ?? {})) {
        const num = parseInt(tn);
        const existing = (await db.tbaTeams.get(num)) ?? { teamNumber: num };
        const keys = adj.ignoredMatchKeys ?? existing.ignoredMatchKeys ?? [];

        let adjustedOPR = null;
        if (keys.length > 0) {
            const keySet = new Set(keys);
            const subset = allMatches.filter(m =>
                (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 &&
                !globalIgnored.has(m.key) && !keySet.has(m.key)
            );
            const result = computeLocalOPR(subset, allTeamNums);
            const idx = allTeamNums.findIndex(n => n === num);
            if (result && idx !== -1) adjustedOPR = result[idx];
        }

        await db.tbaTeams.put({
            ...existing,
            ignoredMatchKeys: keys.length > 0 ? keys : null,
            adjustedOPR: keys.length > 0 ? adjustedOPR : null,
            scoutingIgnoreActive: adj.scoutingIgnoreActive ?? existing.scoutingIgnoreActive ?? false,
        });
        if (adj.analysis) {
            const team = (await db.teams.get(num)) ?? { teamNumber: num };
            await db.teams.put({ ...team, analysis: adj.analysis });
        }
    }
    const existingNotes = JSON.parse(localStorage.getItem('teamNotes') || '{}');
    for (const [tn, matchNotes] of Object.entries(bundle.notes ?? {})) {
        existingNotes[tn] = { ...existingNotes[tn], ...matchNotes };
    }
    localStorage.setItem('teamNotes', JSON.stringify(existingNotes));
    await refreshEPADisplays(activeTeamNumber);
}

window.exportAdjBundle = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an event key first.'); return; }
    const bundle = await buildAdjBundle(eventKey);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }));
    a.download = `${eventKey}_adjustments.json`;
    a.click();
    URL.revokeObjectURL(a.href);
};

window.importAdjFile = async function (input) {
    const file = input.files[0];
    if (!file) return;
    input.value = '';
    try {
        const bundle = JSON.parse(await file.text());
        if (!bundle.version || !bundle.teams) throw new Error('Not a valid adjustments file.');
        if (!confirm(`Import adjustments for ${bundle.eventKey}?\nThis merges ignored matches, ceilings, and notes without overwriting your raw data.`)) return;
        await applyAdjBundle(bundle);
        alert('Adjustments imported.');
    } catch (err) {
        alert(`Import failed: ${err.message}`);
    }
};

window.shareAdjLink = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('Enter an event key first.'); return; }
    if (typeof LZString === 'undefined') { alert('LZString not loaded — check your network connection.'); return; }
    const bundle = await buildAdjBundle(eventKey);
    const compressed = LZString.compressToEncodedURIComponent(JSON.stringify(bundle));
    const url = `${location.origin}${location.pathname}#adj=${compressed}`;
    const btn = document.getElementById('shareAdjLinkBtn');
    try {
        await navigator.clipboard.writeText(url);
        if (btn) { const orig = btn.textContent; btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = orig; }, 2000); }
    } catch {
        prompt('Copy this link:', url);
    }
};

async function checkAdjustmentsFromURL() {
    const hash = location.hash;
    if (!hash.startsWith('#adj=')) return;
    const compressed = hash.slice(5);
    history.replaceState(null, '', location.pathname + location.search);
    if (typeof LZString === 'undefined') return;
    try {
        const bundle = JSON.parse(LZString.decompressFromEncodedURIComponent(compressed));
        if (!bundle?.version || !bundle?.teams) return;
        if (!confirm(`Import adjustments for ${bundle.eventKey} from shared link?\nThis merges ignored matches, ceilings, and notes without overwriting your raw data.`)) return;
        await applyAdjBundle(bundle);
        alert('Adjustments imported.');
    } catch {
        // malformed or expired link — silently ignore
    }
}

function renderScoutingSection() {
    const container = document.getElementById('scouting-data-section');
    if (!container) return;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();

    if (!eventKey) {
        container.innerHTML = `<p style="color:#475569;font-size:0.85em;margin:0;">Enter an event key above to configure scouting data.</p>`;
        return;
    }

    const source    = getScoutingSource(eventKey);
    const isLive    = source?.startsWith('http');
    const hasData   = !!localStorage.getItem(`scoutingData_${eventKey}`);
    const lastSync  = localStorage.getItem('lastSync_scoutingData');
    const gameConfig = getGameConfig(eventKey);
    const noConfig  = !gameConfig;

    const pitSource = getPitSource(eventKey);

    if (!source) {
        const hasPitData  = !!localStorage.getItem(`pitData_${eventKey}`);
        const pitLastSync = localStorage.getItem('lastSync_pitData');
        const pitSyncStatus = pitLastSync && hasPitData ? `Last sync: ${pitLastSync}` : hasPitData ? 'Data loaded' : 'No data yet';
        const pitSectionHtml = pitSource
            ? `<div>
                   <div style="color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Pit Scouting</div>
                   <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                       <span style="color:#34d399;font-size:0.78em;font-weight:700;">● Configured</span>
                   </div>
                   <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                       <button id="btn-syncPitData" onclick="syncPitData()">Sync Pit Data</button>
                       <span id="pit-sync-status" style="color:#475569;font-size:0.78em;">${pitSyncStatus}</span>
                   </div>
               </div>`
            : `<div>
                   <div style="color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Pit Scouting</div>
                   <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                       <input type="text" id="pitUrlInput" placeholder="Sheet ID or Google Sheets URL"
                           style="flex:1;min-width:260px;padding:8px 10px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:0.85em;">
                       <button onclick="savePitSheetUrl()">Save</button>
                   </div>
               </div>`;
        container.innerHTML = `
            <p style="color:#64748b;font-size:0.85em;margin:0 0 10px;">No match sheet configured for <strong style="color:#f8fafc;">${eventKey}</strong>.</p>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div>
                    <div style="color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Match Scouting</div>
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                        <input type="text" id="scoutingUrlInput" placeholder="Sheet ID or Google Sheets URL"
                            style="flex:1;min-width:260px;padding:8px 10px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:0.85em;">
                        <button onclick="saveScoutingSheetUrl()">Save</button>
                    </div>
                </div>
                ${pitSectionHtml}
            </div>`;
        return;
    }

    const matchBadge = isLive
        ? `<span style="color:#34d399;font-size:0.78em;font-weight:700;">● Live Sheet</span>`
        : `<span style="color:#60a5fa;font-size:0.78em;font-weight:700;">● Archive</span>`;
    const gameLabel = noConfig
        ? `<span style="color:#fbbf24;font-size:0.78em;">⚠ No game config for ${eventKey.match(/^\d{4}/)?.[0] ?? '?'}</span>`
        : `<span style="color:#64748b;font-size:0.78em;">Game: <strong style="color:#94a3b8;">${gameConfig.name} ${gameConfig.year}</strong></span>`;

    const hasPitData   = !!localStorage.getItem(`pitData_${eventKey}`);
    const pitLastSync  = localStorage.getItem('lastSync_pitData');
    const syncStatus    = lastSync && hasData ? `Last sync: ${lastSync}` : hasData ? 'Data loaded' : 'No data yet';
    const pitSyncStatus = pitLastSync && hasPitData ? `Last sync: ${pitLastSync}` : hasPitData ? 'Data loaded' : 'No data yet';

    const pitBlock = pitSource
        ? `<div style="display:flex;flex-direction:column;gap:6px;margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;">
               <div style="display:flex;align-items:center;gap:8px;">
                   <span style="color:#64748b;font-size:0.85em;min-width:100px;">Pit Scouting</span>
                   <span style="color:#34d399;font-size:0.78em;font-weight:700;">● Configured</span>
               </div>
               <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                   <button id="btn-syncPitData" onclick="syncPitData()">Sync Pit Data</button>
                   <span id="pit-sync-status" style="color:#475569;font-size:0.78em;">${pitSyncStatus}</span>
               </div>
           </div>`
        : `<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;">
               <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                   <span style="color:#64748b;font-size:0.85em;min-width:100px;">Pit Scouting</span>
                   <input type="text" id="pitUrlInput" placeholder="Sheet ID or Google Sheets URL"
                       style="flex:1;min-width:220px;padding:8px 10px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:0.85em;">
                   <button onclick="savePitSheetUrl()">Save</button>
               </div>
           </div>`;

    container.innerHTML = `
        <div style="margin-bottom:12px;">
            <div style="display:flex;flex-direction:column;gap:6px;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span style="color:#64748b;font-size:0.85em;min-width:100px;">Match Scouting</span>
                    ${matchBadge}
                    ${gameLabel}
                </div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                    <button id="btn-syncScoutingData" onclick="syncScoutingData()">Sync Match Data</button>
                    <span id="scouting-sync-status" style="color:#475569;font-size:0.78em;">${syncStatus}</span>
                </div>
            </div>
            ${pitBlock}
        </div>
        ${isLive ? `
        <div style="padding-top:10px;border-top:1px solid #1e293b;">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <button id="scoutingAutoSyncBtn" onclick="toggleScoutingAutoSync()" style="background:#059669;">Start Auto-Sync</button>
                <select id="scoutingAutoSyncInterval" style="padding:7px 10px;border-radius:4px;border:1px solid #334155;background:#1e293b;color:#f8fafc;font-size:0.9rem;cursor:pointer;">
                    <option value="5">Every 5 min</option>
                    <option value="7">Every 7 min</option>
                    <option value="10">Every 10 min</option>
                </select>
                <span id="scoutingAutoSyncStatus" style="color:#94a3b8;font-size:0.85em;font-variant-numeric:tabular-nums;"></span>
            </div>
        </div>` : ''}`;

    updateOBEStatus(eventKey);
}

window.clearCache = async function () {
    if (!confirm("Clear all API data (Statbotics + TBA)? This cannot be undone.")) {
        return;
    }

    try {
        await db.teams.clear();
        await db.tbaTeams.clear();
        await db.matches.clear();
        await _clearTrackCache();

        // Clear persisted sync state
        localStorage.removeItem('lastEventKey');
        updateAppEventKey(null);
        localStorage.removeItem('pickListOrder');
        localStorage.removeItem('mockDraftState');
        localStorage.removeItem('realDraftState');
        localStorage.removeItem('draftState');
        for (const key of ['statboticsLive', 'tbaOPR', 'tbaMatches']) {
            localStorage.removeItem(`lastSync_${key}`);
            const el = document.getElementById(`ts-${key}`);
            if (el) el.textContent = '';
        }
        Object.keys(localStorage).filter(k => k.startsWith('archiveCoverage_')).forEach(k => localStorage.removeItem(k));
        updateOBEStatus(null);

        // Clear the event key input
        const input = document.getElementById('eventKeyInput');
        if (input) input.value = '';

        const statusDiv = document.getElementById('status');
        if (statusDiv) statusDiv.innerText = 'Cache cleared.';

        displayTeams();
        displaySchedule();

        console.log("Database cache cleared.");
    } catch (err) {
        console.error("Error clearing cache:", err);
        alert("Failed to clear cache. Check console for details.");
    }
};

// Track data is cached in THREE places and clearing one without the others is worse than
// clearing none: the Dexie rows, the in-memory manifest, and the Routes tab's memo. Leave
// the memo set and the tab keeps painting the old event's routes from a stale closure;
// leave the manifest and the next render re-populates Dexie from it immediately.
async function _clearTrackCache() {
    try { await db.matchTracks.clear(); } catch { /* table absent on a stale schema */ }
    _tracksManifest = null;
    routesRenderedFor = null;
}

async function _silentClearEvent(eventKey) {
    await db.teams.clear();
    await db.tbaTeams.clear();
    await db.matches.clear();
    // Cleared wholesale like the tables above, not filtered to eventKey, so that "clear
    // the event" means the same thing for every table. Losing another event's cached
    // routes costs one re-fetch from public/tracks/, which is the cheapest thing here.
    await _clearTrackCache();

    for (const key of ['statboticsLive', 'tbaOPR', 'tbaMatches', 'statboticsProjections']) {
        localStorage.removeItem(`lastSync_${key}`);
        const el = document.getElementById(`ts-${key}`);
        if (el) el.textContent = '';
    }
    localStorage.removeItem(`archiveCoverage_${eventKey}`);
    localStorage.removeItem(`scoutingData_${eventKey}`);
    localStorage.removeItem(`scoutingFusedStats_${eventKey}`);
    localStorage.removeItem('lastSync_scoutingData');
    localStorage.removeItem(`pitData_${eventKey}`);
    localStorage.removeItem('lastSync_pitData');
    localStorage.removeItem(`tbaAlliances_${eventKey}`);
    localStorage.removeItem('mockDraftState');
    localStorage.removeItem('realDraftState');
    localStorage.removeItem(`rpThresholds_${eventKey}`);
    localStorage.removeItem(`wlCalibrationBeta_${eventKey}`);
    localStorage.removeItem(`wlPreEventSnapshot_${eventKey}`);
    localStorage.removeItem(`webcasts_${eventKey}`);
    localStorage.removeItem(`eventNotes_${eventKey}`);

    localStorage.setItem('nexusEnabled', 'false');
    localStorage.removeItem('nexusEventKeyOverride');

    _bannerMatchTime = null;
    const _cb = document.getElementById('match-countdown-banner');
    if (_cb) _cb.style.display = 'none';
    wlDetailCache = null;
    wlPreEventCache = null;
    wlComputedAsOf = null;
    wlCalibrationBeta = 0.982;
    watchListDirty = true;

    if (teamChartInstance) { teamChartInstance.destroy(); teamChartInstance = null; }
    if (tbaChartInstance) { tbaChartInstance.destroy(); tbaChartInstance = null; }
    if (matchInfluenceChartInstance) { matchInfluenceChartInstance.destroy(); matchInfluenceChartInstance = null; }
    if (dashboardChartInstance) { dashboardChartInstance.destroy(); dashboardChartInstance = null; }
}

window.clearEvent = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('No event key set — enter one first.'); return; }
    if (!confirm(`Clear all data for ${eventKey} (API cache, scouting, and pit data)? This cannot be undone.`)) return;

    try {
        nexusMatchCache = {};
        await _silentClearEvent(eventKey);
        updateOBEStatus(eventKey);

        await displayTeams();
        await displayTBATeams();
        await displaySchedule();
        await renderAtAGlance();
        await renderPickList();
        renderScoutingSection();
        displayScoutingTeams();

        localStorage.removeItem('lastEventKey');
        const keyInput = document.getElementById('eventKeyInput');
        if (keyInput) keyInput.value = '';
        updateAppEventKey(null);

        const nexusKeyInput = document.getElementById('nexusEventKeyOverride');
        if (nexusKeyInput) nexusKeyInput.value = '';
        updateNexusUI();
    } catch (err) {
        console.error('Error clearing event:', err);
        alert('Failed to clear event data. Check console for details.');
    }
};

window.clearScoutingData = function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;
    if (!confirm(`Clear scouting data for ${eventKey}? This cannot be undone.`)) return;
    localStorage.removeItem(`scoutingData_${eventKey}`);
    localStorage.removeItem(`scoutingFusedStats_${eventKey}`);
    localStorage.removeItem('lastSync_scoutingData');
    renderScoutingSection();
    displayScoutingTeams();
};

window.clearPitData = function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;
    if (!confirm(`Clear pit scouting data for ${eventKey}? This cannot be undone.`)) return;
    localStorage.removeItem(`pitData_${eventKey}`);
    localStorage.removeItem('lastSync_pitData');
    renderScoutingSection();
};

window.clearScouting = function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;
    if (!confirm(`Clear all scouting data (match + pit) for ${eventKey}? This cannot be undone.`)) return;
    localStorage.removeItem(`scoutingData_${eventKey}`);
    localStorage.removeItem(`scoutingFusedStats_${eventKey}`);
    localStorage.removeItem('lastSync_scoutingData');
    localStorage.removeItem(`pitData_${eventKey}`);
    localStorage.removeItem('lastSync_pitData');
    renderScoutingSection();
    displayScoutingTeams();
};




window.runManualAnalysis = async function () {
    if (!activeTeamNumber) {
        console.error("No active team selected.");
        return;
    }

    // --- FIX: The Bulletproof Fetch ---
    // Try it exactly as stored first, then fallback to Integer just in case
    let team = await db.teams.get(activeTeamNumber);
    if (!team) {
        team = await db.teams.get(parseInt(activeTeamNumber));
    }

    if (!team) {
        alert("Error: Could not load team data from database.");
        return;
    }

    const startInput = parseInt(document.getElementById('mathStart').value);
    const endInput = parseInt(document.getElementById('mathEnd').value);

    // Get only the completed matches with EPA
    const playedMatches = team.rawStatboticsData.filter(m => m.epa?.post);

    // Slice based on match index (1-based for user friendliness)
    const selection = playedMatches.slice(startInput - 1, endInput);
    const epaTimeline = selection.map(m => m.epa.post);

    if (epaTimeline.length < 5) {
        document.getElementById('analysisFeedback').innerText = "❌ Need at least 5 matches in range.";
        return;
    }

    document.getElementById('analysisFeedback').innerText = "Calculating...";

    // Run the Math
    const fitParams = fitExponentialGrowth(epaTimeline);
    const analysis = getBootstrappedCeiling(epaTimeline);

    // Temporarily store it on the object so the chart can see it
    team.analysis = analysis;
    team.analysis.rawParams = fitParams;
    team.analysis.startIndex = startInput - 1; // Used for chart alignment

    // Update UI Stats
    document.getElementById('detailStats').innerHTML = `
        <div style="background:#333; padding:15px; border-radius:8px;">
            <label style="color:#888; font-size:0.8em;">RANGE CURRENT</label>
            <div style="font-size:1.5em; font-weight:bold;">${epaTimeline[epaTimeline.length - 1]}</div>
        </div>
        <div style="background:#333; padding:15px; border-radius:8px;">
            <label style="color:#888; font-size:0.8em;">PROJECTED CEILING</label>
            <div style="font-size:1.5em; font-weight:bold; color:#4ade80;">${analysis.ceiling}</div>
        </div>
        <div style="background:#333; padding:15px; border-radius:8px;">
            <label style="color:#888; font-size:0.8em;">CONFIDENCE</label>
            <div>${analysis.lowerBound} - ${analysis.upperBound}</div>
        </div>
    `;

    // SAVE THE ANALYSIS: This makes the result show up in the main table permanently
    await db.teams.update(team.teamNumber, {
        analysis: team.analysis
    });

    refreshEPADisplays();

    document.getElementById('analysisFeedback').innerText = `✅ Analysis complete for matches ${startInput} to ${Math.min(endInput, playedMatches.length)}.`;

    // Re-draw the chart with the new trendline
    renderChart(team);
};




// 7. UI HOOKS
const eventInput = document.getElementById('eventKeyInput');


function fitExponentialGrowth(timeline) {
    const n = timeline.length;
    const maxObserved = Math.max(...timeline);

    let bestA = 0, bestB = 0, bestK = 0;
    let lowestError = Infinity;

    // We know the true ceiling (A) must be higher than their current highest score.
    // We will test every possible ceiling from just above their max, up to 3x their max.
    const startA = maxObserved + 0.1;
    const endA = maxObserved * 3.0;
    const stepA = 0.5;

    for (let A = startA; A <= endA; A += stepA) {

        // --- LOG-LINEARIZATION ---
        // By looking at ln(A - y), we turn the exponential curve into a straight line.
        // This allows us to use Exact Algebraic Least Squares to find the perfect slope.
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        let validPoints = 0;

        for (let i = 0; i < n; i++) {
            const x = i + 1;
            const y = timeline[i];

            // Calculate the linearized Y value
            const Y = Math.log(A - y);

            sumX += x;
            sumY += Y;
            sumXY += x * Y;
            sumXX += x * x;
            validPoints++;
        }

        // Closed-form linear regression formulas (No learning rates, perfect accuracy)
        const denominator = (validPoints * sumXX) - (sumX * sumX);
        if (denominator === 0) continue;

        const m = ((validPoints * sumXY) - (sumX * sumY)) / denominator;
        const C = (sumY - m * sumX) / validPoints;

        // Convert the linear line back into our exponential variables
        const k = -m;
        const B = Math.exp(C);

        // We only care about positive growth. If the math suggests they are getting worse, ignore it.
        if (k <= 0) continue;

        // --- NON-LINEAR ERROR CHECK ---
        // Check how well these exact parameters fit the actual raw dots on the chart
        let totalSquaredError = 0;
        for (let i = 0; i < n; i++) {
            const x = i + 1;
            const prediction = A - B * Math.exp(-k * x);
            const error = prediction - timeline[i];
            totalSquaredError += error * error; // True Least Squares calculation
        }

        // If this ceiling produced the lowest overall error, save it as the winner
        if (totalSquaredError < lowestError) {
            lowestError = totalSquaredError;
            bestA = A;
            bestB = B;
            bestK = k;
        }
    }

    // Fallback if the data is entirely flat
    if (lowestError === Infinity) {
        return { A: maxObserved, B: 0, k: 0.1, n };
    }

    return { A: bestA, B: bestB, k: bestK, n };
}

function getBootstrappedCeiling(timeline) {
    if (timeline.length < 5) return { ceiling: "N/A" };

    const originalFit = fitExponentialGrowth(timeline);

    const residuals = timeline.map((y, i) => {
        const x = i + 1;
        return y - (originalFit.A - originalFit.B * Math.exp(-originalFit.k * x));
    });

    const bootstrapResults = [];
    for (let b = 0; b < 100; b++) {
        const syntheticTimeline = timeline.map((y, i) => {
            const x = i + 1;
            const randomResid = residuals[Math.floor(Math.random() * residuals.length)];
            const val = (originalFit.A - originalFit.B * Math.exp(-originalFit.k * x)) + randomResid;
            return val;
        });

        const fit = fitExponentialGrowth(syntheticTimeline);
        bootstrapResults.push(fit.A);
    }

    bootstrapResults.sort((a, b) => a - b);

    return {
        ceiling: originalFit.A.toFixed(1),
        lowerBound: bootstrapResults[Math.floor(bootstrapResults.length * 0.05)].toFixed(1),
        upperBound: bootstrapResults[Math.floor(bootstrapResults.length * 0.95)].toFixed(1),
        rawParams: originalFit
    };
}

let currentSortKey = 'ceiling'; // Default sort
let currentSortOrder = 1; // 1 for descending, -1 for ascending
let currentSortColumn = 'ceiling'; // Add this line

window.sortBy = function (column) {
    if (currentSortColumn === column) {
        // If clicking the same column, flip the direction
        currentSortOrder *= -1;
    } else {
        // If clicking a new column, set it as active and default to Highest First
        currentSortColumn = column;
        currentSortOrder = 1;

        // Exception: Team Numbers usually make more sense sorted Lowest First
        if (column === 'teamNumber') currentSortOrder = -1;
    }
    displayTeams(); // Redraw the table with the new sorting rules
};




let teamChartInstance     = null;
let dashboardChartInstance = null;

function renderTeamChart(sortedTeams) {
    const ctx = document.getElementById('teamComparisonChart').getContext('2d');
    if (teamChartInstance) teamChartInstance.destroy();

    const isMobile = document.body.classList.contains('mobile-ui');
    // Prepare labels (Team Numbers)
    const labels = sortedTeams.map(t => t.teamNumber.toString());

    teamChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Auto',
                    data: sortedTeams.map(t => t.autoEPA || 0),
                    backgroundColor: '#fbbf24',
                    stack: 'EPA',
                    order: 10,
                },
                {
                    label: 'Teleop',
                    data: sortedTeams.map(t => t.teleopEPA || 0),
                    backgroundColor: '#3b82f6',
                    stack: 'EPA',
                    order: 10,
                },
                {
                    label: 'Endgame',
                    data: sortedTeams.map(t => t.endgameEPA || 0),
                    backgroundColor: '#10b981',
                    stack: 'EPA',
                    order: 10,
                },
                {
                    type: 'scatter',
                    label: 'EPA',
                    data: sortedTeams.map(t => t.currentEPA || 0),
                    backgroundColor: sortedTeams.map(t => {
                        const c = t.analysis?.ceiling;
                        return (c != null && c !== '—' && c !== 'N/A') ? 'rgba(0,0,0,0)' : '#ffffff';
                    }),
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    pointStyle: 'circle',
                    pointRadius: isMobile ? 2.5 : 5,
                    pointHoverRadius: isMobile ? 4 : 7,
                    order: 1,
                },
                {
                    type: 'line',
                    label: 'Ceiling',
                    data: sortedTeams.map(t => {
                        const c = t.analysis?.ceiling;
                        return (c != null && c !== '—' && c !== 'N/A') ? parseFloat(c) : null;
                    }),
                    borderColor: '#4ade80',
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#ffffff',
                    pointStyle: 'circle',
                    pointRadius: isMobile ? 3 : 6,
                    pointHoverRadius: isMobile ? 4 : 8,
                    spanGaps: false,
                    fill: false,
                    order: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Expected Points Added (EPA)', color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#f8fafc', usePointStyle: true }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}




window.displayTeams = async function () {
    const allTeams = await db.teams.toArray();
    const tableBody = document.getElementById('teamBody');
    //const searchVal = document.getElementById('teamSearch')?.value || '';
    const table = document.getElementById('teamTable');

    if (allTeams.length === 0) {
        table.style.display = 'none';
        return;
    }
    table.style.display = 'table';

    // Compute tier by ceiling EPA rank across all teams
    const sortedForTier = [...allTeams].sort((a, b) =>
        (b.analysis?.ceiling || b.currentEPA || 0) - (a.analysis?.ceiling || a.currentEPA || 0));
    const teamTierMap = new Map(sortedForTier.map((t, i) => [
        t.teamNumber, i < 8 ? 'S' : i < 20 ? 'A' : i < 32 ? 'B' : 'C'
    ]));

    // 2. Sort logic (Maintains your preferred order)
    allTeams.sort((a, b) => {
        let valA, valB;

        switch (currentSortColumn) {
            case 'teamNumber':
                valA = a.teamNumber;
                valB = b.teamNumber;
                break;
            case 'autoEPA':
                valA = a.autoEPA || 0;
                valB = b.autoEPA || 0;
                break;
            case 'teleopEPA':
                valA = a.teleopEPA || 0;
                valB = b.teleopEPA || 0;
                break;
            case 'endgameEPA':
                valA = a.endgameEPA || 0;
                valB = b.endgameEPA || 0;
                break;
            case 'currentEPA':
                valA = a.currentEPA || 0;
                valB = b.currentEPA || 0;
                break;
            case 'ceiling':
            default:
                valA = a.analysis?.ceiling || a.currentEPA || 0;
                valB = b.analysis?.ceiling || b.currentEPA || 0;
                break;
        }

        // parseFloat ensures we are doing math on numbers, not strings
        return (parseFloat(valB) - parseFloat(valA)) * currentSortOrder;
    });

    // 2. Render the Chart with the current list
    renderTeamChart(allTeams);

    tableBody.innerHTML = '';

    allTeams.forEach(team => {
        const analysis = team.analysis || { ceiling: "—", lowerBound: "—", upperBound: "—" };
        const { ceiling, lowerBound, upperBound } = analysis;

        const tier = teamTierMap.get(team.teamNumber) || 'C';
        const row = document.createElement('tr');
        row.style.backgroundColor = TIER_STYLE[tier].bg;
        row.style.borderLeft = `6px solid ${TIER_STYLE[tier].color}`;

        // --- THE FIX ---
        // 1. Change the mouse to a pointer so it feels like a button
        row.style.cursor = 'pointer';

        // 2. Attach the click event to the entire row safely
        row.onclick = () => viewTeamDetail(team.teamNumber, 'epa-opr');

        // Notice we removed the onclick="" from the <td> string
        const _eb = localEpaBadge(team);
        row.innerHTML = `
            <td>${tierBadge(tier)}</td>
            <td style="white-space:nowrap;"><strong>${team.teamNumber}</strong>${ownStar(team.teamNumber)}</td>
            <td>${team.currentEPA ? team.currentEPA.toFixed(1) : 'N/A'}${_eb}</td>
            <td class="ceiling-cell"><strong>${ceiling}</strong></td>
            <td style="color:#aaa;">${team.autoEPA ? team.autoEPA.toFixed(1) + _eb : '-'}</td>
            <td style="color:#aaa;">${team.teleopEPA ? team.teleopEPA.toFixed(1) + _eb : '-'}</td>
            <td style="color:#aaa;">${team.endgameEPA ? team.endgameEPA.toFixed(1) + _eb : '-'}</td>
        `;

        tableBody.appendChild(row);
    });
}


// And add this at the very bottom of main.js to load on startup
displayTeams();




window.setSort = function (key) {
    if (currentSortKey === key) {
        currentSortOrder *= -1;
    } else {
        currentSortKey = key;
        currentSortOrder = -1;
    }
    displayTeams();
};

// ─── OPR COMPUTATION ────────────────────────────────────────────────────────

// Works on local db.matches records (red/blue string arrays, redScore/blueScore).
// Returns OPR array indexed the same as teamNumbers, or null if underdetermined.
function computeLocalOPR(matches, teamNumbers) {
    const keys = teamNumbers.map(String);
    const n = keys.length;
    if (n === 0) return null;
    const idx = Object.fromEntries(keys.map((k, i) => [k, i]));
    const rows = [], scores = [];
    for (const m of matches) {
        if ((m.redScore ?? -1) < 0) continue;
        for (const [alliance, score] of [
            [(m.red || []).map(String), m.redScore],
            [(m.blue || []).map(String), m.blueScore]
        ]) {
            const row = new Array(n).fill(0);
            for (const t of alliance) { if (idx[t] !== undefined) row[idx[t]] = 1; }
            rows.push(row);
            scores.push(score);
        }
    }
    if (rows.length < n) return null;
    const ATA = Array.from({ length: n }, () => new Array(n).fill(0));
    const ATb = new Array(n).fill(0);
    for (let k = 0; k < rows.length; k++) {
        for (let i = 0; i < n; i++) {
            if (!rows[k][i]) continue;
            ATb[i] += scores[k];
            for (let j = 0; j < n; j++) ATA[i][j] += rows[k][j];
        }
    }
    return gaussianElim(ATA, ATb);
}

function gaussianElim(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
        }
        [M[col], M[maxRow]] = [M[maxRow], M[col]];
        if (Math.abs(M[col][col]) < 1e-10) continue;
        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const f = M[row][col] / M[col][col];
            for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
        }
    }
    return Array.from({ length: n }, (_, i) => M[i][i] === 0 ? 0 : M[i][n] / M[i][i]);
}

function computeOPR(matches, teamKeys, getScore) {
    const n = teamKeys.length;
    if (n === 0) return null;
    const idx = Object.fromEntries(teamKeys.map((k, i) => [k, i]));
    const rows = [], scores = [];
    for (const m of matches) {
        if (m.comp_level !== 'qm' || !m.alliances) continue;
        for (const color of ['red', 'blue']) {
            const score = m.score_breakdown ? getScore(m.score_breakdown[color]) : null;
            if (score == null || isNaN(score)) continue;
            const row = new Array(n).fill(0);
            for (const key of (m.alliances[color].team_keys || [])) {
                if (idx[key] !== undefined) row[idx[key]] = 1;
            }
            rows.push(row);
            scores.push(score);
        }
    }
    if (rows.length < n) return null;
    const ATA = Array.from({ length: n }, () => new Array(n).fill(0));
    const ATb = new Array(n).fill(0);
    for (let k = 0; k < rows.length; k++) {
        for (let i = 0; i < n; i++) {
            if (!rows[k][i]) continue;
            ATb[i] += scores[k];
            for (let j = 0; j < n; j++) ATA[i][j] += rows[k][j];
        }
    }
    return gaussianElim(ATA, ATb);
}

// ─── TBA OPR SYNC ────────────────────────────────────────────────────────────

window.syncTBAOPR = async function () {
    const eventKey = document.getElementById('eventKeyInput').value.trim().toLowerCase();
    if (!eventKey) return alert("Please enter an Event Key.");
    const statusDiv = document.getElementById('status');
    statusDiv.innerText = "Fetching TBA OPR & COPR data...";
    try {
        const [oprData, coprData] = await Promise.all([
            fetchTBA(`/event/${eventKey}/oprs`),
            fetchTBA(`/event/${eventKey}/coprs`)
        ]);
        if (!oprData?.oprs) throw new Error("No OPR data found. Event may not have played yet.");

        // Detect component fields from COPRs response
        let autoMap = null, teleopMap = null, endgameMap = null;
        if (coprData && typeof coprData === 'object' && !coprData.Errors) {
            const keys = Object.keys(coprData);
            console.log('[TBA COPRs] Available components:', keys.join(', '));
            const autoKey = ['totalAutoPoints', 'autoPoints'].find(k => keys.includes(k));
            const teleopKey = ['totalTeleopPoints', 'teleopPoints'].find(k => keys.includes(k));
            if (autoKey) autoMap = coprData[autoKey];
            if (teleopKey) teleopMap = coprData[teleopKey];
            // Endgame = tower climbing + fuel scored during the endgame window
            const towerMap = coprData['endGameTowerPoints'] || null;
            const endgameFuelMap = coprData['Hub Endgame Fuel Count'] || null;
            if (towerMap || endgameFuelMap) {
                const allKeys = Object.keys(towerMap || endgameFuelMap);
                endgameMap = Object.fromEntries(
                    allKeys.map(k => [k, (towerMap?.[k] || 0) + (endgameFuelMap?.[k] || 0)])
                );
            }
        }

        const teamKeys = Object.keys(oprData.oprs).sort();
        const records = teamKeys.map(key => ({
            teamNumber: parseInt(key.replace('frc', '')),
            teamKey: key,
            eventKey,
            opr: +(oprData.oprs[key] || 0).toFixed(2),
            dpr: +(oprData.dprs[key] || 0).toFixed(2),
            ccwm: +(oprData.ccwms[key] || 0).toFixed(2),
            autoOPR: autoMap ? +(autoMap[key] || 0).toFixed(2) : null,
            teleopOPR: teleopMap ? +(teleopMap[key] || 0).toFixed(2) : null,
            endgameOPR: endgameMap ? +(endgameMap[key] || 0).toFixed(2) : null,
            lastUpdated: Date.now()
        }));
        await db.tbaTeams.bulkPut(records);
        setSyncTimestamp('tbaOPR');
        statusDiv.innerText = `✅ TBA OPR synced for ${records.length} teams.`;
        await displayTBATeams();
        _setSnapshotStale(true);
    } catch (err) {
        console.error(err);
        statusDiv.innerText = `❌ TBA OPR Sync Failed: ${err.message}`;
    }
};

// Detects score breakdown field names and recomputes component OPRs from stored match data.
async function recomputeComponentOPRs(matches, eventKey) {
    const tbaTeams = await db.tbaTeams.where('eventKey').equals(eventKey).toArray();
    if (tbaTeams.length === 0) return;
    const teamKeys = tbaTeams.map(t => t.teamKey).sort();

    const sampleBd = matches.find(m => m.score_breakdown?.red)?.score_breakdown?.red;
    if (!sampleBd) {
        console.log('[TBA] No score breakdowns available — matches may not have been played yet.');
        return;
    }
    const fields = Object.keys(sampleBd);
    console.log('[TBA] Score breakdown fields:', fields.join(', '));

    const autoF = ['autoPoints'].find(f => fields.includes(f));
    const teleopF = ['teleopPoints'].find(f => fields.includes(f));
    const endgameF = ['endgamePoints', 'endGamePoints', 'endgameBargePoints', 'endgameTotalStagePoints']
        .find(f => fields.includes(f));
    console.log(`[TBA] Mapping → auto:${autoF} | teleop:${teleopF} | endgame:${endgameF}`);

    const autoOPRs = autoF ? computeOPR(matches, teamKeys, bd => bd?.[autoF] ?? null) : null;
    const teleopOPRs = teleopF ? computeOPR(matches, teamKeys, bd => bd?.[teleopF] ?? null) : null;
    const endgameOPRs = endgameF ? computeOPR(matches, teamKeys, bd => bd?.[endgameF] ?? null) : null;

    const updates = tbaTeams.map(team => {
        const i = teamKeys.indexOf(team.teamKey);
        if (i === -1) return team;
        return {
            ...team,
            autoOPR: autoOPRs ? +autoOPRs[i].toFixed(2) : null,
            teleopOPR: teleopOPRs ? +teleopOPRs[i].toFixed(2) : null,
            endgameOPR: endgameOPRs ? +endgameOPRs[i].toFixed(2) : null,
        };
    });
    await db.tbaTeams.bulkPut(updates);
}

window.syncTBAMatches = async function () {
    const eventKey = document.getElementById('eventKeyInput').value.trim().toLowerCase();
    if (!eventKey) return alert("Please enter an Event Key.");
    const statusDiv = document.getElementById('status');
    statusDiv.innerText = "Fetching full match data from TBA...";
    try {
        const matches = await fetchTBA(`/event/${eventKey}/matches`);
        if (!Array.isArray(matches)) throw new Error("Invalid match data from TBA.");

        // Snapshot already-scored matches before overwriting, for new-score notifications
        const scoredBefore = new Set(
            (await db.matches.where('eventKey').equals(eventKey).toArray())
                .filter(m => m.redScore > -1).map(m => m.key)
        );

        const records = matches.map(m => ({
            key: m.key,
            eventKey,
            compLevel: m.comp_level,
            matchNumber: m.match_number,
            setNumber: m.set_number ?? null,
            red: (m.alliances?.red?.team_keys || []).map(k => k.replace('frc', '')),
            blue: (m.alliances?.blue?.team_keys || []).map(k => k.replace('frc', '')),
            redScore: m.alliances?.red?.score ?? -1,
            blueScore: m.alliances?.blue?.score ?? -1,
            redBreakdown: m.score_breakdown?.red || null,
            blueBreakdown: m.score_breakdown?.blue || null,
            predictedTime: m.predicted_time || null,
            actualTime: m.actual_time || null,
            videos: (m.videos || []).filter(v => v.type === 'youtube').map(v => v.key),
        }));
        await db.matches.bulkPut(records);

        // Notify for each newly posted score that involves the focused team
        const focused = window.currentFocusedTeam;
        if (focused) {
            for (const r of records) {
                if (r.redScore > -1 && !scoredBefore.has(r.key)) {
                    const allTeams = [...(r.red || []), ...(r.blue || [])];
                    if (allTeams.includes(focused)) {
                        const notifId = `score-${r.key}`;
                        if (!_firedNotifIds.has(notifId)) {
                            _firedNotifIds.add(notifId);
                            const redWon = r.redScore > r.blueScore;
                            const result = redWon ? 'Red wins' : r.blueScore > r.redScore ? 'Blue wins' : 'Tie';
                            fireNotif(
                                `QM ${r.matchNumber} scored — ${result}`,
                                `${r.redScore}–${r.blueScore} · Red: ${(r.red||[]).join(', ')} · Blue: ${(r.blue||[]).join(', ')}`,
                                notifId
                            );
                        }
                    }
                }
            }
        }

        setSyncTimestamp('tbaMatches');
        watchListDirty = true;
        const qualCount = records.filter(r => r.compLevel === 'qm').length;
        const playoffCount = records.length - qualCount;
        statusDiv.innerText = `✅ TBA Matches synced (${qualCount} qual${playoffCount ? `, ${playoffCount} playoff` : ''}).`;
        displaySchedule();
        maybeAutoActivateNexus();
        startNexusDirectPolling();
        check1768QueueNotifications();

        // Auto-fetch webcasts if they haven't been stored yet for this event
        if (!JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) || '[]').length) {
            _fetchAndStoreWebcasts(eventKey).catch(e => console.warn('Could not fetch webcasts:', e));
        }
    } catch (err) {
        console.error(err);
        statusDiv.innerText = `❌ TBA Matches Sync Failed: ${err.message}`;
    }
};



// ─── TBA CHART & TABLE ───────────────────────────────────────────────────────

let tbaChartInstance = null;
let tbaSortColumn = 'opr';
let tbaSortOrder = 1;
let matchInfluenceChartInstance = null;
let currentTBATab = 'teams';

window.sortTBABy = function (column) {
    if (tbaSortColumn === column) {
        tbaSortOrder *= -1;
    } else {
        tbaSortColumn = column;
        tbaSortOrder = column === 'teamNumber' ? -1 : 1;
    }
    displayTBATeams();
};

function renderTBAChart(teams, effOPR) {
    const ctx = document.getElementById('tbaComparisonChart').getContext('2d');
    if (tbaChartInstance) tbaChartInstance.destroy();
    const isMobile = document.body.classList.contains('mobile-ui');
    const hasComponents = teams.some(t => t.autoOPR != null);
    const labels = teams.map(t => t.teamNumber.toString());
    const getOPR = effOPR || (t => t.opr || 0);
    const datasets = hasComponents ? [
        { label: 'Auto OPR', data: teams.map(t => t.autoOPR || 0), backgroundColor: '#fbbf24', stack: 'OPR', order: 10 },
        { label: 'Teleop OPR', data: teams.map(t => t.teleopOPR || 0), backgroundColor: '#3b82f6', stack: 'OPR', order: 10 },
        { label: 'Endgame OPR', data: teams.map(t => t.endgameOPR || 0), backgroundColor: '#10b981', stack: 'OPR', order: 10 }
    ] : [
        { label: 'OPR', data: teams.map(t => t.opr || 0), backgroundColor: '#3b82f6', stack: 'OPR', order: 10 }
    ];
    datasets.push({
        type: 'scatter',
        label: 'OPR',
        data: teams.map(getOPR),
        backgroundColor: '#ffffff',
        borderColor: '#ffffff',
        borderWidth: 2,
        pointStyle: 'circle',
        pointRadius: isMobile ? 2.5 : 5,
        pointHoverRadius: isMobile ? 4 : 7,
        order: 1,
    });
    tbaChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } },
                y: {
                    beginAtZero: true, stacked: true, grid: { color: '#334155' }, ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'OPR', color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#f8fafc', usePointStyle: true } },
                tooltip: { mode: 'index', intersect: false }
            }
        }
    });
}

window.displayTBATeams = async function () {
    const allTeams = await db.tbaTeams.toArray();
    const allMatches = await db.matches.toArray();
    const tableBody = document.getElementById('tbaBody');
    const table = document.getElementById('tbaTable');
    const globalIgnoreNote = document.getElementById('tbaGlobalIgnoreNote');
    if (allTeams.length === 0) { table.style.display = 'none'; return; }

    // Recompute OPR excluding any globally ignored matches.
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
    let globalOPRMap = null;
    if (globalIgnored.size > 0) {
        const teamNums = allTeams.map(t => t.teamNumber);
        const activePlayed = allMatches.filter(m =>
            (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !globalIgnored.has(m.key)
        );
        const recomputed = computeLocalOPR(activePlayed, teamNums);
        if (recomputed) {
            globalOPRMap = Object.fromEntries(teamNums.map((num, i) => [num, recomputed[i]]));
        }
    }
    if (globalIgnoreNote) {
        if (globalIgnored.size > 0) {
            globalIgnoreNote.textContent = `OPRs recomputed excluding ${globalIgnored.size} globally ignored match${globalIgnored.size > 1 ? 'es' : ''}.`;
            globalIgnoreNote.style.display = 'block';
        } else {
            globalIgnoreNote.style.display = 'none';
        }
    }

    // Use individually-ignored LOO, then globally-recomputed OPR, then raw TBA OPR.
    const effOPR = t => {
        const keys = getTeamIgnoredKeys(t);
        if (keys.length > 0 && t.adjustedOPR != null && keys.some(k => !globalIgnored.has(k)))
            return t.adjustedOPR;
        if (globalOPRMap) return globalOPRMap[t.teamNumber] ?? (t.opr || 0);
        return t.opr || 0;
    };

    allTeams.sort((a, b) => {
        let valA, valB;
        switch (tbaSortColumn) {
            case 'teamNumber': valA = a.teamNumber; valB = b.teamNumber; break;
            case 'dpr': valA = a.dpr || 0; valB = b.dpr || 0; break;
            case 'ccwm': valA = a.ccwm || 0; valB = b.ccwm || 0; break;
            case 'autoOPR': valA = a.autoOPR ?? 0; valB = b.autoOPR ?? 0; break;
            case 'teleopOPR': valA = a.teleopOPR ?? 0; valB = b.teleopOPR ?? 0; break;
            case 'endgameOPR': valA = a.endgameOPR ?? 0; valB = b.endgameOPR ?? 0; break;
            default: valA = effOPR(a); valB = effOPR(b);
        }
        return (parseFloat(valB) - parseFloat(valA)) * tbaSortOrder;
    });
    const tbaForTier = [...allTeams].sort((a, b) => effOPR(b) - effOPR(a));
    const tbaTierMap = new Map(tbaForTier.map((t, i) => [
        t.teamNumber, i < 8 ? 'S' : i < 20 ? 'A' : i < 32 ? 'B' : 'C'
    ]));
    renderTBAChart(allTeams, effOPR);
    table.style.display = 'table';
    tableBody.innerHTML = '';
    const hasComponents = allTeams.some(t => t.autoOPR != null);
    allTeams.forEach(team => {
        const eff = effOPR(team);
        const tier = tbaTierMap.get(team.teamNumber) || 'C';
        const row = document.createElement('tr');
        row.style.backgroundColor = TIER_STYLE[tier].bg;
        row.style.borderLeft = `6px solid ${TIER_STYLE[tier].color}`;
        row.style.cursor = 'pointer';
        row.onclick = () => viewTeamDetail(team.teamNumber, 'epa-opr');
        const oprCell = getTeamIgnoredKeys(team).some(k => !globalIgnored.has(k))
            ? `${eff.toFixed(1)}&thinsp;<span style="color:#fbbf24; font-size:0.7em; font-weight:600;">ADJ</span>`
            : eff.toFixed(1);
        row.innerHTML = `
            <td>${tierBadge(tier)}</td>
            <td style="white-space:nowrap;"><strong>${team.teamNumber}</strong>${ownStar(team.teamNumber)}</td>
            <td>${oprCell}</td>
            <td style="color:${team.ccwm >= 0 ? '#4ade80' : '#f87171'}">${team.ccwm.toFixed(1)}</td>
            <td style="color:#aaa;">${hasComponents && team.autoOPR != null ? team.autoOPR.toFixed(1) : '—'}</td>
            <td style="color:#aaa;">${hasComponents && team.teleopOPR != null ? team.teleopOPR.toFixed(1) : '—'}</td>
            <td style="color:#aaa;">${hasComponents && team.endgameOPR != null ? team.endgameOPR.toFixed(1) : '—'}</td>
        `;
        tableBody.appendChild(row);
    });
};


// ─── HOME: AT A GLANCE ───────────────────────────────────────────────────────

let glanceSortColumn = 'rp';
let glanceSortOrder = 1; // 1 = descending

window.switchHomeTab = function (tab) {
    ['setup', 'overview', 'stream'].forEach(t => {
        const display = t === tab ? (t === 'setup' ? 'grid' : 'block') : 'none';
        document.getElementById(`home-tab-${t}`).style.display = display;
    });
    document.querySelectorAll('#homeTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['setup', 'overview', 'stream'][i] === tab);
    });
    if (tab === 'overview') renderAtAGlance();
    if (tab === 'stream')   renderStreamsTab();
};

window.sortGlanceBy = function (col) {
    if (glanceSortColumn === col) {
        glanceSortOrder *= -1;
    } else {
        glanceSortColumn = col;
        glanceSortOrder = 1;
    }
    renderAtAGlance();
};

let dashboardChartCondensed = true;
let lastDashboardRows       = null;

window.toggleDashboardCondense = function () {
    dashboardChartCondensed = !dashboardChartCondensed;
    const btn = document.getElementById('dashboardCondenseBtn');
    if (btn) btn.textContent = dashboardChartCondensed ? 'Expand' : 'Condense';
    if (lastDashboardRows) renderDashboardChart(lastDashboardRows);
};

function renderDashboardChart(rows) {
    lastDashboardRows = rows;
    const canvas = document.getElementById('dashboardComparisonChart');
    if (!canvas) return;
    if (dashboardChartInstance) { dashboardChartInstance.destroy(); dashboardChartInstance = null; }

    const labels  = rows.map(r => String(r.team.teamNumber));
    const epas    = rows.map(r => r.epaVal   ?? null);
    const oprs    = rows.map(r => r.opr      ?? null);
    const scouts  = rows.map(r => r.scoutEPA ?? null);
    const hasOPR   = oprs.some(v => v != null);
    const hasScout = scouts.some(v => v != null);

    const datasets = [
        { label: 'Statbotics EPA', data: epas,   backgroundColor: '#f59e0baa', borderColor: '#f59e0b', borderWidth: 1 },
        ...(hasOPR   ? [{ label: 'TBA OPR',   data: oprs,   backgroundColor: '#60a5faaa', borderColor: '#60a5fa', borderWidth: 1 }] : []),
        ...(hasScout ? [{ label: 'Scout EPA',  data: scouts, backgroundColor: '#4ade80aa', borderColor: '#4ade80', borderWidth: 1 }] : []),
    ];

    // Per-team column width: condensed aims to fit ~800px; expanded gives comfortable spacing.
    const perTeam = dashboardChartCondensed
        ? Math.max(14, Math.floor(800 / Math.max(rows.length, 1)))
        : 52;
    const w = Math.max(640, rows.length * perTeam);
    canvas.width  = w;  canvas.style.width  = w + 'px';
    canvas.height = 240; canvas.style.height = '240px';

    // Subtle tier-colored background band behind each team's bar group.
    const TIER_RGBA = { S: 'rgba(245,158,11,0.13)', A: 'rgba(74,222,128,0.10)',
                        B: 'rgba(168,85,247,0.09)',  C: 'rgba(100,116,139,0.05)' };
    const tierColors = rows.map(r => TIER_RGBA[r.tier] ?? TIER_RGBA.C);

    const tierBgPlugin = {
        id: 'dashboardTierBg',
        beforeDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            if (!chartArea) return;
            const step = scales.x.width / Math.max(labels.length, 1);
            const half = step / 2;
            ctx.save();
            tierColors.forEach((color, i) => {
                const cx = scales.x.getPixelForValue(i);
                ctx.fillStyle = color;
                ctx.fillRect(cx - half, chartArea.top, step, chartArea.bottom - chartArea.top);
            });
            ctx.restore();
        },
    };

    const condensed = dashboardChartCondensed;
    dashboardChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets },
        plugins: [tierBgPlugin],
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
                tooltip: { callbacks: { title: ctx => 'Team ' + ctx[0].label } },
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { size: condensed ? 8 : 10 },
                        maxRotation: condensed ? 90 : 0,
                        minRotation: condensed ? 90 : 0,
                    },
                    grid: { color: '#1e293b' },
                },
                y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#334155' }, beginAtZero: true },
            },
        },
    });
}

async function renderAtAGlance() {
    const statusEl = document.getElementById('atAGlanceStatus');
    const table = document.getElementById('atAGlanceTable');
    const tbody = document.getElementById('atAGlanceBody');
    if (!statusEl || !table || !tbody) return;

    const eventKey   = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const tbaLink    = document.getElementById('dashboardTBALink');
    const statLink   = document.getElementById('dashboardStatboticsLink');
    const linksRow   = document.getElementById('dashboardEventLinks');
    if (eventKey && tbaLink && statLink && linksRow) {
        tbaLink.href  = `https://www.thebluealliance.com/event/${eventKey}`;
        statLink.href = `https://www.statbotics.io/event/${eventKey}`;
        linksRow.style.display = 'flex';
    } else if (linksRow) {
        linksRow.style.display = 'none';
    }

    const [allTeams, allTBATeams, allMatches] = await Promise.all([
        db.teams.toArray(), db.tbaTeams.toArray(), db.matches.toArray()
    ]);

    if (!allTeams.length) {
        statusEl.textContent = 'No team data — sync team list/history or Statbotics Live first.';
        table.style.display = 'none';
        if (tbody) tbody.innerHTML = '';
        return;
    }

    // ── Effective OPR (mirrors displayTBATeams) ──────────────────────────────
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
    const tbaTeamMap = Object.fromEntries(allTBATeams.map(t => [t.teamNumber, t]));

    let globalOPRMap = null;
    if (globalIgnored.size > 0) {
        const teamNums = allTBATeams.map(t => t.teamNumber);
        const activePlayed = allMatches.filter(m =>
            (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !globalIgnored.has(m.key)
        );
        const recomputed = computeLocalOPR(activePlayed, teamNums);
        if (recomputed) globalOPRMap = Object.fromEntries(teamNums.map((n, i) => [n, recomputed[i]]));
    }

    const effOPR = tba => {
        if (!tba) return null;
        const keys = getTeamIgnoredKeys(tba);
        if (keys.length > 0 && tba.adjustedOPR != null && keys.some(k => !globalIgnored.has(k)))
            return tba.adjustedOPR;
        if (globalOPRMap) return globalOPRMap[tba.teamNumber] ?? tba.opr ?? null;
        return tba.opr ?? null;
    };

    // ── Ranking points from match history ────────────────────────────────────
    const rpMap = {};
    const playedMatches = allMatches.filter(m => (m.redScore ?? -1) >= 0 && (!m.compLevel || m.compLevel === 'qm'));
    let hasBreakdown = false;

    for (const m of playedMatches) {
        const redWon = m.redScore > m.blueScore;
        const blueWon = m.blueScore > m.redScore;
        const tie = m.redScore === m.blueScore;

        // Use TBA's pre-computed rankingPoints when available; otherwise compute from
        // match result (3/1/0) + 2026 bonus RPs (Energized, Supercharged, Traversal).
        const bonusRP = bd => bd ? (
            (bd.energizedAchieved ? 1 : 0) +
            (bd.superchargedAchieved ? 1 : 0) +
            (bd.traversalAchieved ? 1 : 0)
        ) : 0;
        const redRP = m.redBreakdown?.rp ?? ((redWon ? 3 : tie ? 1 : 0) + bonusRP(m.redBreakdown));
        const blueRP = m.blueBreakdown?.rp ?? ((blueWon ? 3 : tie ? 1 : 0) + bonusRP(m.blueBreakdown));
        if (m.redBreakdown != null) hasBreakdown = true;

        for (const team of (m.red || [])) {
            if (!rpMap[team]) rpMap[team] = { rp: 0, played: 0, wins: 0, ties: 0, losses: 0, totalScore: 0 };
            rpMap[team].rp += redRP; rpMap[team].played++; rpMap[team].totalScore += m.redScore;
            if (redWon) rpMap[team].wins++; else if (tie) rpMap[team].ties++; else rpMap[team].losses++;
        }
        for (const team of (m.blue || [])) {
            if (!rpMap[team]) rpMap[team] = { rp: 0, played: 0, wins: 0, ties: 0, losses: 0, totalScore: 0 };
            rpMap[team].rp += blueRP; rpMap[team].played++; rpMap[team].totalScore += m.blueScore;
            if (blueWon) rpMap[team].wins++; else if (tie) rpMap[team].ties++; else rpMap[team].losses++;
        }
    }

    // ── Scouting EPA ─────────────────────────────────────────────────────────
    const scoutEPAMap = {};
    if (eventKey) {
        const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
        if (rawStr) {
            const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
            const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
            if (processed?.config?.computeEPABreakdown) {
                const { config, byTeam } = processed;
                for (const [tn, rawRows] of Object.entries(byTeam)) {
                    const tbaEntry = tbaTeamMap[parseInt(tn)];
                    const scoutIgnoreKeys = tbaEntry?.scoutingIgnoreActive ? getTeamIgnoredKeys(tbaEntry) : [];
                    let { rows: deduped } = deduplicateTeamRows(rawRows);
                    let ignoredMatchNums = new Set();
                    if (scoutIgnoreKeys.length > 0) {
                        ignoredMatchNums = new Set(allMatches.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber));
                        deduped = deduped.filter(r => !ignoredMatchNums.has(r.matchNumber));
                    }
                    const rawStats = config.aggregateTeam(deduped);
                    const fusedResult = fusedCache?.teams?.[tn];
                    const effectiveFused = (fusedResult?.available && ignoredMatchNums.size > 0)
                        ? refilteredFusedStats(fusedResult, ignoredMatchNums) : fusedResult;
                    const isFused = !!(effectiveFused?.available && config.computeFusedEPABreakdown);
                    const breakdown = isFused
                        ? config.computeFusedEPABreakdown(effectiveFused.stats)
                        : config.computeEPABreakdown(rawStats);
                    scoutEPAMap[tn] = { total: breakdown.total, isFused, isAdj: ignoredMatchNums.size > 0 };
                }
            }
        }
    }

    // ── Build rows ────────────────────────────────────────────────────────────
    const rows = allTeams.map(team => {
        const tn = parseInt(team.teamNumber);
        const tba = tbaTeamMap[tn];
        const rp = rpMap[String(tn)] || { rp: 0, played: 0, wins: 0, ties: 0, losses: 0 };
        const opr = effOPR(tba);
        const analysis = team.analysis || {};
        const hasCeil = analysis.ceiling != null && analysis.ceiling !== '—';
        const epaVal = hasCeil ? parseFloat(analysis.ceiling) : (team.currentEPA || 0);
        const hasLOO = getTeamIgnoredKeys(tba).some(k => !globalIgnored.has(k)) && tba?.adjustedOPR != null;
        const hasAdj = !hasLOO && globalOPRMap != null;
        const scoutData = scoutEPAMap[tn];
        return { team, tba, rp, opr, epaVal, hasCeil, hasLOO, hasAdj,
                 scoutEPA: scoutData?.total ?? null, scoutFused: scoutData?.isFused ?? false, scoutAdj: scoutData?.isAdj ?? false };
    });

    const hasOPR = allTBATeams.length > 0;
    const hasRP = playedMatches.length > 0;

    // ── Composite score + tier (must run before sort) ────────────────────────
    // Composite = average of EPA and OPR percentile ranks (0 = best, 1 = worst).
    // Tier cutoffs: top 8 → S, next 12 → A, next 12 → B, rest → C.
    {
        const pctRank = (arr, val) => {
            const sorted = [...arr].sort((a, b) => b - a);
            const idx = sorted.findIndex(v => v <= val + 0.001);
            return idx < 0 ? 1 : idx / (sorted.length || 1);
        };
        const epaVals = rows.map(r => r.epaVal);
        const oprVals = rows.map(r => r.opr ?? 0);
        const scoutEPAVals = rows.map(r => r.scoutEPA ?? 0);
        const hasAnyOPR = rows.some(r => r.opr != null);
        const hasAnyScout = rows.some(r => r.scoutEPA != null);
        const composite = r => {
            const sources = [pctRank(epaVals, r.epaVal)];
            if (hasAnyOPR)   sources.push(pctRank(oprVals, r.opr ?? 0));
            if (hasAnyScout) sources.push(pctRank(scoutEPAVals, r.scoutEPA ?? 0));
            return sources.reduce((a, b) => a + b, 0) / sources.length;
        };
        rows.forEach(r => { r.composite = composite(r); });
        const tierOrder = [...rows].sort((a, b) => a.composite - b.composite);
        const tierMap = new Map(tierOrder.map((r, i) => [
            r.team.teamNumber,
            i < 8 ? 'S' : i < 20 ? 'A' : i < 32 ? 'B' : 'C'
        ]));
        rows.forEach(r => { r.tier = tierMap.get(r.team.teamNumber); });
    }

    const avgScore = rp => rp.played ? rp.totalScore / rp.played : 0;

    // Sort — default rp desc
    rows.sort((a, b) => {
        let va, vb;
        switch (glanceSortColumn) {
            case 'epa': va = a.epaVal; vb = b.epaVal; break;
            case 'opr': va = a.opr ?? -999; vb = b.opr ?? -999; break;
            case 'scoutEPA': va = a.scoutEPA ?? -999; vb = b.scoutEPA ?? -999; break;
            case 'composite': va = -a.composite; vb = -b.composite; break;
            // Swap a/b so glanceSortOrder=1 means ascending (smallest team number first)
            case 'teamNumber': va = b.team.teamNumber; vb = a.team.teamNumber; break;
            default: va = a.rp.played > 0 ? a.rp.rp / a.rp.played : 0; vb = b.rp.played > 0 ? b.rp.rp / b.rp.played : 0; break;
        }
        return (vb - va) * glanceSortOrder || avgScore(b.rp) - avgScore(a.rp) || (b.epaVal - a.epaVal);
    });

    // Compute RP-based rank separately so it stays stable regardless of current sort
    const avgRP = rp => rp.played > 0 ? rp.rp / rp.played : 0;
    const rpRank = Object.fromEntries(
        [...rows].sort((a, b) => avgRP(b.rp) - avgRP(a.rp) || avgScore(b.rp) - avgScore(a.rp) || b.epaVal - a.epaVal)
            .map((r, i) => [r.team.teamNumber, i + 1])
    );
    const TIER = TIER_STYLE;

    table.style.display = 'table';
    tbody.innerHTML = rows.map(r => {
        const { team, rp, opr, epaVal, hasCeil, hasLOO, hasAdj, composite, scoutEPA, scoutFused, scoutAdj } = r;
        const rank = hasRP ? rpRank[team.teamNumber] : '—';
        const record = hasRP ? `${rp.wins}–${rp.losses}${rp.ties ? `–${rp.ties}` : ''}` : null;
        const rpStr = hasRP ? rp.rp : '—';
        const compStr = composite != null ? ((1 - composite) * 100).toFixed(1) : '—';
        const epaStr = epaVal.toFixed(1);
        const ceilBadge = hasCeil
            ? `<span style="color:#4ade80; font-size:0.65em; font-weight:600; margin-left:3px;">CEIL</span>` : '';
        const oprStr = opr != null ? opr.toFixed(1) : '—';
        const oprBadge = (hasLOO || hasAdj)
            ? `<span style="color:${hasLOO ? '#fbbf24' : '#f97316'}; font-size:0.65em; font-weight:600; margin-left:3px;">ADJ</span>`
            : '';
        const scoutStr = scoutEPA != null ? scoutEPA.toFixed(1) : '—';
        const fusedBadge = scoutFused
            ? `<span style="color:#818cf8; font-size:0.65em; font-weight:600; margin-left:3px;">F</span>` : '';
        const scoutAdjBadge = scoutAdj
            ? `<span style="color:#fbbf24; font-size:0.65em; font-weight:600; margin-left:3px;">ADJ</span>` : '';

        const tier = r.tier;
        const ts = TIER[tier];
        const td = (content, center = true) =>
            `<td style="padding:13px 10px; border-bottom:1px solid #1e293b;${center ? ' text-align:center;' : ''}">${content}</td>`;
        const rankCell = `<td style="padding:13px 10px; border-bottom:1px solid #1e293b; text-align:center; box-shadow:inset 3px 0 0 ${ts.color};">
            <div style="display:flex; flex-direction:column; align-items:center; gap:2px;">
                <span style="color:${ts.color}; font-size:0.75em; font-weight:800; letter-spacing:0.08em;">${tier}</span>
                <span style="color:#64748b; font-weight:700;">${rank}</span>
            </div>
        </td>`;

        const teamCell = `<td style="padding:13px 10px; border-bottom:1px solid #1e293b; white-space:nowrap;">
            <strong style="color:#f8fafc;">${team.teamNumber}</strong>${ownStar(team.teamNumber)}
        </td>
        <td style="padding:13px 10px; border-bottom:1px solid #1e293b;">
            <span style="color:#94a3b8; font-size:0.85em; font-weight:600;">${team.teamName || ''}</span>
        </td>`;

        return `<tr style="cursor:pointer; background:${ts.bg};" onclick="viewTeamDetail(${team.teamNumber})">
            ${rankCell}
            ${teamCell}
            ${td(`<span style="color:${ts.color};">${compStr}</span>`)}
            ${td(`${rpStr}${record ? `<div style="color:#94a3b8;font-size:0.75em;font-weight:600;margin-top:2px;white-space:nowrap;">${record}</div>` : ''}`)}
            ${td(`${epaStr}${ceilBadge}${localEpaBadge(team)}`)}
            ${td(hasOPR ? `${oprStr}${oprBadge}` : '—')}
            ${td(`${scoutStr}${fusedBadge}${scoutAdjBadge}`)}
        </tr>`;
    }).join('');

    renderDashboardChart(rows);

    statusEl.textContent = !hasRP
        ? 'Sync schedule to see records and ranking points.'
        : !hasBreakdown
            ? 'Ranking points show win/tie/loss only — run "Sync TBA Matches" to include bonus RPs.'
            : '';
}

// ─── TBA MATCH INFLUENCE TAB ────────────────────────────────────────────────

window.switchTBATab = function (tab) {
    currentTBATab = tab;
    document.getElementById('tba-tab-teams').style.display = tab === 'teams' ? 'block' : 'none';
    document.getElementById('tba-tab-matches').style.display = tab === 'matches' ? 'block' : 'none';
    document.querySelectorAll('#tbaTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['teams', 'matches'][i] === tab);
    });
    if (tab === 'matches') renderMatchInfluenceTab();
};

// Returns [{m, influence, isIgnored}] for every played match.
// influence = Σ|ΔOPR| across all teams when this match is removed (or added back if ignored).
async function computeMatchInfluences() {
    const allMatches = await db.matches.toArray();
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
    const allPlayed = allMatches.filter(m => (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0);
    const activePlayed = allPlayed.filter(m => !globalIgnored.has(m.key));

    if (!activePlayed.length) return null;

    // Use only teams from active played matches — allows OPR to be solved for early-event
    // states where far fewer than all registered teams have appeared on the field.
    const allTeamNums = [...new Set(
        activePlayed.flatMap(m => [...(m.red ?? []), ...(m.blue ?? [])]).map(Number)
    )];

    const baseOPRs = computeLocalOPR(activePlayed, allTeamNums);
    if (!baseOPRs) return null;

    return allPlayed.map(m => {
        const isIgnored = globalIgnored.has(m.key);
        // Non-ignored: LOO (remove m). Ignored: reverse (add m back to active set).
        const subset = isIgnored ? [...activePlayed, m] : activePlayed.filter(pm => pm.key !== m.key);
        const looOPRs = computeLocalOPR(subset, allTeamNums);
        const influence = looOPRs
            ? baseOPRs.reduce((sum, opr, i) => sum + Math.abs(opr - looOPRs[i]), 0)
            : null;
        return { m, influence, isIgnored };
    });
}

function renderMatchInfluenceChart(sorted) {
    const ctx = document.getElementById('matchInfluenceChart');
    if (!ctx) return;
    if (matchInfluenceChartInstance) matchInfluenceChartInstance.destroy();

    const influences = sorted.map(r => r.influence ?? 0);
    const total = influences.reduce((s, v) => s + v, 0);
    let cum = 0;
    const cdfData = influences.map(v => {
        cum += v;
        return total > 0 ? +((cum / total) * 100).toFixed(1) : 0;
    });

    matchInfluenceChartInstance = new Chart(ctx, {
        data: {
            labels: sorted.map(r => `Q${r.m.matchNumber}`),
            datasets: [
                {
                    type: 'bar',
                    label: 'Influence (Σ|ΔOPR|)',
                    data: influences,
                    backgroundColor: sorted.map(r => r.isIgnored ? '#334155' : '#3b82f6'),
                    yAxisID: 'y',
                    order: 2
                },
                {
                    type: 'line',
                    label: 'Cumulative %',
                    data: cdfData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y2',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 45 } },
                y: {
                    beginAtZero: true,
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Σ|ΔOPR|', color: '#94a3b8' }
                },
                y2: {
                    position: 'right',
                    beginAtZero: true,
                    max: 100,
                    grid: { display: false },
                    ticks: { color: '#f59e0b', callback: v => v + '%' },
                    title: { display: true, text: 'Cumulative %', color: '#f59e0b' }
                }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#f8fafc', usePointStyle: true } },
                tooltip: { mode: 'index', intersect: false }
            }
        }
    });
}

window.renderMatchInfluenceTab = async function () {
    const statusEl = document.getElementById('matchInfluenceStatus');
    const tableBody = document.getElementById('matchInfluenceBody');
    const tableWrapper = document.getElementById('matchInfluenceTable');
    const chartWrapper = document.getElementById('matchInfluenceChartContainer');
    if (!statusEl || !tableBody) return;

    statusEl.innerText = 'Computing match influences…';
    tableWrapper.style.display = 'none';
    chartWrapper.style.display = 'none';

    const results = await computeMatchInfluences();
    if (!results) {
        statusEl.innerText = 'Run "Sync TBA OPR" and "Sync Schedule" first.';
        return;
    }
    statusEl.innerText = '';

    // Sort by influence descending; globally-ignored shown interleaved by their re-inclusion influence.
    const sorted = [...results].sort((a, b) => (b.influence ?? 0) - (a.influence ?? 0));

    chartWrapper.style.display = 'block';
    tableWrapper.style.display = 'table';
    renderMatchInfluenceChart(sorted);

    tableBody.innerHTML = sorted.map(r => {
        const { m, influence, isIgnored } = r;
        const played = (m.redScore ?? -1) >= 0;
        const redWon = played && m.redScore > m.blueScore;
        const blueWon = played && m.blueScore > m.redScore;
        const resultCell = played
            ? `<span style="color:${redWon ? '#4ade80' : '#94a3b8'}; font-weight:${redWon ? 'bold' : 'normal'}">${m.redScore}</span>
               <span style="color:#475569"> – </span>
               <span style="color:${blueWon ? '#4ade80' : '#94a3b8'}; font-weight:${blueWon ? 'bold' : 'normal'}">${m.blueScore}</span>`
            : '<span style="color:#475569; font-style:italic;">Upcoming</span>';

        const redTeams = (m.red || []).map(t =>
            `<span onclick="event.stopPropagation();viewTeamDetail(${t})" style="color:#ef4444;cursor:pointer;">${t}</span>`
        ).join(' ');
        const blueTeams = (m.blue || []).map(t =>
            `<span onclick="event.stopPropagation();viewTeamDetail(${t})" style="color:#3b82f6;cursor:pointer;">${t}</span>`
        ).join(' ');

        const influenceDisplay = influence != null
            ? `${influence.toFixed(2)}${isIgnored ? '<span title="Re-inclusion influence" style="color:#64748b; font-size:0.8em;">*</span>' : ''}`
            : '—';

        const matchLabel = `Q${m.matchNumber}${isIgnored
            ? ' <span style="color:#f59e0b; font-size:0.7em; font-weight:600;">IGNORED</span>' : ''}`;

        const actionBtn = `<button onclick="event.stopPropagation();setGloballyIgnored('${m.key}',${!isIgnored})"
            style="padding:4px 14px; font-size:0.85em; background:${isIgnored ? '#92400e' : '#1e293b'}; border:1px solid ${isIgnored ? '#d97706' : '#475569'}; color:${isIgnored ? '#fde68a' : '#94a3b8'}; border-radius:4px; cursor:pointer;">
            ${isIgnored ? 'Restore' : 'Ignore'}
        </button>`;

        return `<tr onclick="viewMatchDetail('${m.key}')" style="cursor:pointer; opacity:${isIgnored ? '0.55' : '1'};">
            <td style="padding:12px 8px; font-weight:bold; color:#f8fafc;">${matchLabel}</td>
            <td style="padding:12px 8px;">${redTeams}</td>
            <td style="padding:12px 8px;">${blueTeams}</td>
            <td style="padding:12px 8px; white-space:nowrap;">${resultCell}</td>
            <td style="padding:12px 8px; font-weight:bold; color:#f8fafc;">${influenceDisplay}</td>
            <td style="padding:12px 8px;" onclick="event.stopPropagation();">${actionBtn}</td>
        </tr>`;
    }).join('');
};

window.setGloballyIgnored = async function (matchKey, ignored) {
    await db.matches.update(matchKey, { globallyIgnored: ignored || null });

    // When globally ignoring a match, remove it from any team's individual ignore list.
    if (ignored) {
        const affected = await db.tbaTeams.filter(t =>
            (Array.isArray(t.ignoredMatchKeys) && t.ignoredMatchKeys.includes(matchKey)) ||
            t.ignoredMatchKey === matchKey
        ).toArray();
        if (affected.length > 0) {
            await Promise.all(affected.map(t => {
                const keys = getTeamIgnoredKeys(t).filter(k => k !== matchKey);
                return db.tbaTeams.update(t.teamNumber, {
                    ignoredMatchKeys: keys.length > 0 ? keys : null,
                    ignoredMatchKey:  null,
                    adjustedOPR:      null,
                });
            }));
        }
    }

    if (isLocalEpaEnabled()) await computeLocalEPA(); // affects all teams in the match
    await displayTBATeams();
    if (currentTBATab === 'matches') await renderMatchInfluenceTab();
    if (activeTBAData) {
        activeTBAData = await db.tbaTeams.get(activeTBAData.teamNumber);
        await renderTBADetail(activeTBAData.teamNumber, activeTBAData);
        if (activeTeamData) await renderOverview(activeTeamData, activeTBAData);
    }
    // Refresh the performance chart if the matches tab is open
    if (lastDetailTab === 'matches' && activeTeamData) {
        await renderMatchesTab(activeTeamData.teamNumber, 'matches-tab-perf-chart');
    }
};

window.viewTeamDetail = async function (teamNumber, tab = lastDetailTab) {
    activeTeamNumber = teamNumber; // <--- ADD THIS LINE
    const team = await db.teams.get(teamNumber);
    if (!team) return;

    const view = document.getElementById('teamDetailView');
    const label = document.getElementById('detailTeamLabel');
    const stats = document.getElementById('detailStats');

    if (!view || !label || !stats) {
        console.error("Missing Detail View elements in HTML.");
        return;
    }

    label.innerText = `Team ${teamNumber}: ${team.teamName || ''}`;

    // Quip: log-scale EPA fraction determines tier + unique within-tier rank
    {
        const quipContainer = document.getElementById('detailTeamQuip');
        const quipTextEl = document.getElementById('detailTeamQuipText');
        const quipsEnabled = localStorage.getItem('quipsEnabled') === 'true';
        if (quipContainer) quipContainer.style.display = quipsEnabled ? '' : 'none';
        if (quipTextEl && quipsEnabled) {
            const allTeams = await db.teams.where('eventKey').equals(team.eventKey).toArray();
            let quipTier = 'B';
            let rankInTier = null;
            if (allTeams.length > 0) {
                // Fused EPA: prefer scouting-fused, fall back to Statbotics currentEPA
                const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${team.eventKey}`)); } catch { return null; } })();
                const gameConfig = getGameConfig(team.eventKey);
                const getEPA = t => {
                    const fr = fusedCache?.teams?.[String(t.teamNumber)];
                    if (fr?.available && gameConfig?.computeFusedEPABreakdown) {
                        return gameConfig.computeFusedEPABreakdown(fr.stats).total;
                    }
                    return t.currentEPA ?? 0;
                };

                const epas = allTeams.map(getEPA);
                const logMin = Math.log(Math.max(Math.min(...epas), 0.1));
                const logMax = Math.log(Math.max(Math.max(...epas), 0.1));
                const getLogFraction = t => logMax > logMin
                    ? Math.max(0, Math.min(1, (Math.log(Math.max(getEPA(t), 0.1)) - logMin) / (logMax - logMin)))
                    : 0.5;

                quipTier = logFractionToTier(getLogFraction(team));

                // Rank within tier by log-fraction descending → unique quip slot per team
                const tierPeers = allTeams
                    .filter(t => logFractionToTier(getLogFraction(t)) === quipTier)
                    .sort((a, b) => getLogFraction(b) - getLogFraction(a));
                rankInTier = Math.max(tierPeers.findIndex(t => t.teamNumber === Number(teamNumber)), 0);
            }
            const randomMode = localStorage.getItem('quipRandomMode') === 'true';
            quipTextEl.textContent = Number(teamNumber) === 1768
                ? 'Mechanis Lupus.'
                : getTeamQuip(Number(teamNumber), quipTier, randomMode, rankInTier, getQuipUserSeed());
            if (quipContainer) {
                quipContainer.dataset.tier = quipTier;
                quipContainer.dataset.team = teamNumber;
            }
        }
    }

    // --- FIX: The Safe Check ---
    // If analysis is null, provide default "blank" values
    const analysis = team.analysis || { ceiling: "—", lowerBound: "—", upperBound: "—" };

    stats.innerHTML = `
        <div style="background:#333; padding:15px; border-radius:8px;" id="detailEpaCard">
            <label style="color:#888; font-size:0.8em;">${isLocalEpaEnabled() && !teamHasSbEventData(team) ? 'LOCAL EPA' : 'CURRENT EPA'}</label>
            <div style="font-size:1.5em; font-weight:bold;">${team.currentEPA ? team.currentEPA.toFixed(1) : '0'}</div>
        </div>
        <div style="background:#333; padding:15px; border-radius:8px;">
            <label style="color:#888; font-size:0.8em;">PROJECTED CEILING</label>
            <div style="font-size:1.5em; font-weight:bold; color:#4ade80;">${analysis.ceiling}</div>
        </div>
        <div style="background:#333; padding:15px; border-radius:8px;">
            <label style="color:#888; font-size:0.8em;">90% CONFIDENCE</label>
            <div>${analysis.lowerBound} - ${analysis.upperBound}</div>
        </div>
    `;

    activeTeamData = team;
    activeTBAData = await db.tbaTeams.get(teamNumber) || await db.tbaTeams.get(parseInt(teamNumber));
    if (tab === 'epa-opr') { lastDetailDataSubTab = 'epa'; tab = 'data'; }
    switchDetailTab(tab);

    window.switchView('teamDetailView');
    pushNavState('teamDetail');
    fitDetailLabel();
};

function fitDetailLabel() {
    const el = document.getElementById('detailTeamLabel');
    if (!el) return;
    el.style.fontSize = '';
    const cs = getComputedStyle(el);
    const lineH = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
    const maxH = lineH * 2 + 4; // 2 lines + small rounding buffer
    let size = parseFloat(cs.fontSize);
    const minSize = 13;
    while (el.scrollHeight > maxH && size > minSize) {
        size -= 1;
        el.style.fontSize = size + 'px';
    }
}

window.closeDetail = function () {
    document.getElementById('teamDetailView').style.display = 'none';
};


// At the top of main.js
window.currentView = 'scheduleView';
window.previousView = 'scheduleView';
window.scheduleFilterActive = false;

function applyScheduleFilter() {
    const active = window.scheduleFilterActive && window.currentFocusedTeam;
    const team = window.currentFocusedTeam;
    const isMobile = document.body.classList.contains('mobile-ui');

    // Remove any existing gap rows before re-evaluating
    document.querySelectorAll('#scheduleBody tr.schedule-gap').forEach(r => r.remove());

    document.querySelectorAll('#scheduleBody tr[data-teams]').forEach(row => {
        const teams = (row.dataset.teams || '').split(',');
        const show = !active || teams.includes(team);
        row.style.display = show ? '' : 'none';
        if (isMobile) {
            const next = row.nextElementSibling;
            if (next && !next.dataset.teams) next.style.display = show ? '' : 'none';
        }
    });

    if (!active) return;

    // Insert gap indicator rows between visible match groups
    const allMainRows = [...document.querySelectorAll('#scheduleBody tr[data-teams]')];
    const visibleRows = allMainRows.filter(r => r.style.display !== 'none');
    const cols = isMobile ? 5 : 8;

    visibleRows.forEach((row, i) => {
        if (i === 0) return;
        const prevVisible = visibleRows[i - 1];

        // Walk from after the previous visible match to count hidden matches in between
        // On mobile each visible match has a trailing blue row, so skip it first
        let cursor = isMobile
            ? prevVisible.nextElementSibling?.nextElementSibling
            : prevVisible.nextElementSibling;

        let hiddenCount = 0;
        while (cursor && cursor !== row) {
            if (cursor.dataset.teams) hiddenCount++;
            cursor = cursor.nextElementSibling;
        }

        if (hiddenCount > 0) {
            const gapRow = document.createElement('tr');
            gapRow.className = 'schedule-gap';
            gapRow.innerHTML = `<td colspan="${cols}">· · · ${hiddenCount} match${hiddenCount !== 1 ? 'es' : ''} not shown · · ·</td>`;
            row.parentNode.insertBefore(gapRow, row);
        }
    });
}

window.toggleScheduleFilter = function () {
    window.scheduleFilterActive = !window.scheduleFilterActive;
    document.getElementById('scheduleFilterBtn')?.classList.toggle('active', window.scheduleFilterActive);
    applyScheduleFilter();
};

// ── SCHEDULE SUB-TABS ────────────────────────────────────────────────────────

let watchListDirty = true;
let watchListCutoff = null;   // null = live mode; integer = treat matches > N as unplayed
let wlYourCollapsed = false;
let wlOtherCollapsed = false;
let wlStandingsCollapsed = false;
let wlDetailCache = null;
let wlPreEventCache = null;
let wlComputedAsOf = null; // label shown in banner, e.g. "Q12" or null for pre-event
// Linear calibration factor: p_cal = 0.5 + wlCalibrationBeta*(p - 0.5).
// 1.0 = no correction; <1.0 = shrink toward 50% (fixes overconfidence).
// Set by runBacktest → applyWLCalibration(), resets to 1.0 on page load.
let wlCalibrationBeta = 0.982;

window.switchScheduleTab = function (tab) {
    ['matches', 'watchlist', 'bracket'].forEach(t => {
        document.getElementById(`schedule-sub-${t}`).style.display = t === tab ? '' : 'none';
    });
    document.querySelectorAll('#scheduleTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['matches', 'watchlist', 'bracket'][i] === tab);
    });
    if (tab === 'watchlist' && watchListDirty) showWatchListStale();
    if (tab === 'bracket')  renderBracketTab();
};

function renderStreamsTab() {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const container = document.getElementById('home-tab-stream');
    if (!container) return;

    let webcasts = [];
    try { webcasts = JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) || '[]'); } catch {}

    if (!eventKey || webcasts.length === 0) {
        container.innerHTML = `<p style="color:#64748b;font-style:italic;text-align:center;margin-top:32px;">No streams found. Sync Schedule to check for webcasts.</p>`;
        return;
    }

    container.innerHTML = webcasts.map((w, i) => {
        const thumbId = `stream-thumb-${i}`;
        const dateLabel = w.date ? `<div style="color:#94a3b8;font-size:0.8em;margin-bottom:6px;">${w.date}</div>` : '';
        return `<div style="margin-bottom:24px;">
            ${dateLabel}
            <div id="${thumbId}" onclick="loadYTEmbed('${w.channel}','${thumbId}')"
                style="position:relative;cursor:pointer;border-radius:8px;overflow:hidden;background:#000;max-width:640px;">
                <img src="https://img.youtube.com/vi/${w.channel}/hqdefault.jpg"
                    style="width:100%;display:block;opacity:0.85;"
                    onerror="this.style.display='none'" loading="lazy">
                <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
                    <div style="width:64px;height:44px;background:rgba(255,0,0,0.85);border-radius:10px;display:flex;align-items:center;justify-content:center;">
                        <div style="border-style:solid;border-width:10px 0 10px 20px;border-color:transparent transparent transparent #fff;margin-left:4px;"></div>
                    </div>
                </div>
            </div>
        </div>`;
    }).join('');
}

// ── BRACKET TAB ──────────────────────────────────────────────────────────────

window.saveNexusEventKeyOverride = function () {
    const val = document.getElementById('nexusEventKeyOverride')?.value.trim().toLowerCase() || '';
    if (val) localStorage.setItem('nexusEventKeyOverride', val);
    else localStorage.removeItem('nexusEventKeyOverride');
};

function getNexusEventKey() {
    return localStorage.getItem('nexusEventKeyOverride')?.trim().toLowerCase()
        || document.getElementById('eventKeyInput')?.value.trim().toLowerCase()
        || '';
}

let _bracketMode = localStorage.getItem('bracketMode') || 'bracket';

window.setBracketMode = function (mode) {
    _bracketMode = mode;
    localStorage.setItem('bracketMode', mode);
    document.getElementById('bracketViewBtn')?.classList.toggle('active', mode === 'bracket');
    document.getElementById('bracketListBtn')?.classList.toggle('active', mode === 'list');
    window.renderBracketTab();
};

// FRC double-elimination bracket: match number → {round, upper/lower}
// Official FRC numbering: M1-M4 upper R1, M5-M6 lower R2, M7-M8 upper R2,
// M9-M10 lower R3, M11 lower R4, M12 upper final (R4), M13 lower final (R5), M14-M15 grand finals
const _FRC_BRACKET = {
    1:  { col:0, row:0, label:'Match 1',  upper:true  },
    2:  { col:0, row:2, label:'Match 2',  upper:true  },
    3:  { col:0, row:4, label:'Match 3',  upper:true  },
    4:  { col:0, row:6, label:'Match 4',  upper:true  },
    5:  { col:1, row:1, label:'Match 5',  upper:false },
    6:  { col:1, row:5, label:'Match 6',  upper:false },
    7:  { col:1, row:0, label:'Match 7',  upper:true  },
    8:  { col:1, row:4, label:'Match 8',  upper:true  },
    9:  { col:2, row:2, label:'Match 9',  upper:false },
    10: { col:2, row:4, label:'Match 10', upper:false },
    11: { col:3, row:3, label:'Match 11', upper:false },
    12: { col:3, row:1, label:'Match 12', upper:true  },
    13: { col:4, row:2, label:'Match 13', upper:false },
    14: { col:5, row:2, label:'F1',  upper:null  },
    15: { col:5, row:3, label:'F2',  upper:null  },
};

window.renderBracketTab = async function () {
    const container = document.getElementById('bracket-content');
    if (!container) return;

    // Sync override input state
    const savedOverride = localStorage.getItem('nexusEventKeyOverride') || '';
    const overrideInput = document.getElementById('nexusEventKeyOverride');
    if (overrideInput && !overrideInput.value && savedOverride) overrideInput.value = savedOverride;

    // Sync mode buttons
    document.getElementById('bracketViewBtn')?.classList.toggle('active', _bracketMode === 'bracket');
    document.getElementById('bracketListBtn')?.classList.toggle('active', _bracketMode === 'list');

    const tbaKey   = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const nexusKey = getNexusEventKey();

    if (!tbaKey) {
        container.innerHTML = `<p style="color:#64748b;font-style:italic;text-align:center;margin-top:32px;">No event selected.</p>`;
        return;
    }

    container.innerHTML = `<p style="color:#64748b;text-align:center;margin-top:32px;">Loading…</p>`;

    const tbaMatches = await db.matches.toArray();

    // Convert a TBA match key to bracket match number (1-15).
    // 2023+ double-elim: sf{N}m1 → N (set_number = bracket match), f1m1 → 14, f1m2 → 15
    // Also supports p{N} canonical format used by some TBA responses.
    function _tbaKeyToBracketNum(key) {
        const suffix = key?.replace(`${tbaKey}_`, '');
        if (!suffix) return null;
        if (suffix.startsWith('p')) { const n = parseInt(suffix.slice(1)); return isNaN(n) ? null : n; }
        const sf = suffix.match(/^sf(\d+)m\d+$/);
        if (sf) return parseInt(sf[1]);
        const f = suffix.match(/^f\d+m(\d+)$/);
        if (f) return 13 + parseInt(f[1]);
        return null;
    }

    // Build TBA score lookup: bracket match number → match record
    const tbaScoreMap = {};
    for (const m of tbaMatches) {
        const n = _tbaKeyToBracketNum(m.key);
        if (n != null) tbaScoreMap[n] = m;
    }

    // Try Nexus
    let nexusData = null;
    if (NEXUS_KEY) nexusData = await fetchNexusLiveStatus(nexusKey);

    // Build unified match list from Nexus (with TBA score enrichment) or pure TBA
    let matchEntries = []; // { n, label, redTeams, blueTeams, status, redScore, blueScore, isQueuing, isDone }
    let nowQueuing = null;
    let dataSource = 'tba';

    if (nexusData) {
        dataSource = 'nexus';
        nowQueuing = nexusData.nowQueuing || null;
        const nexusPlayoff = (nexusData.matches || []).filter(m => /^playoff\s*\d+/i.test(m.label || ''));
        for (const m of nexusPlayoff) {
            const n = parseInt((m.label || '').match(/\d+/)?.[0] || '0');
            const tba = tbaScoreMap[n] ?? null;
            // Use TBA compLevel when available (reliable for non-standard brackets);
            // fall back to n >= 14 only when TBA hasn't synced this match yet.
            const isFinals = tba ? tba.compLevel === 'f' : n >= 14;
            matchEntries.push({
                n, label: m.label,
                redTeams:  m.redTeams  || [],
                blueTeams: m.blueTeams || [],
                status: m.status || '',
                redScore:  tba?.redScore  ?? null,
                blueScore: tba?.blueScore ?? null,
                isQueuing: m.label === nowQueuing,
                isDone: /complet|result/i.test(m.status || ''),
                isFinals,
            });
        }
    }

    // Fall through to TBA-only if Nexus failed or returned no playoff matches
    if (!matchEntries.length) {
        dataSource = 'tba';
        const sfM = tbaMatches.filter(m => m.compLevel === 'sf' && (m.setNumber ?? 0) > 0);
        const fM  = tbaMatches.filter(m => m.compLevel === 'f');
        // Finals are numbered after the last bracket match, not hardcoded at 13+
        const maxSfN = sfM.length ? Math.max(...sfM.map(m => m.setNumber)) : 13;
        for (const m of [...sfM, ...fM]) {
            const n = m.compLevel === 'f'
                ? maxSfN + (m.matchNumber ?? 1)
                : (m.setNumber ?? _tbaKeyToBracketNum(m.key));
            if (n == null) continue;
            matchEntries.push({
                n, label: `Playoff ${n}`,
                redTeams:  (m.red  || []).map(t => String(t).replace(/^frc/i,'')),
                blueTeams: (m.blue || []).map(t => String(t).replace(/^frc/i,'')),
                status: (m.redScore ?? -1) >= 0 ? 'Completed' : '',
                redScore:  m.redScore  ?? null,
                blueScore: m.blueScore ?? null,
                isQueuing: false,
                isDone: (m.redScore ?? -1) >= 0,
                isFinals: m.compLevel === 'f',
            });
        }
    }

    if (!matchEntries.length) {
        const nexusMsg = !NEXUS_KEY
            ? `No Nexus API key — configure <code>VITE_NEXUS_KEY</code>.`
            : nexusData
                ? `No playoff matches from Nexus yet.`
                : `Nexus unreachable (check key / network).`;
        container.innerHTML = `
            <div style="text-align:center;margin-top:32px;color:#64748b;">
                <p>${nexusMsg}</p>
                <p style="font-size:0.85em;margin-top:6px;">No TBA playoff matches synced either. Sync TBA Matches once playoffs begin.</p>
            </div>`;
        return;
    }

    matchEntries.sort((a, b) => a.n - b.n);
    const matchMap = Object.fromEntries(matchEntries.map(e => [e.n, e]));

    // Standard double-elim: 13 bracket + up to 3 finals = max 16 total.
    // F3 only exists after F1+F2 are both played (split), so if 16 are listed, at least 15 must be played.
    // Anything beyond that is a non-standard bracket.
    const playedCount = matchEntries.filter(e => e.isDone).length;
    const isNonStandard = matchEntries.length > 16 || (matchEntries.length === 16 && playedCount < 15);
    const bracketViewBtn = document.getElementById('bracketViewBtn');
    const bracketListBtn = document.getElementById('bracketListBtn');
    if (bracketViewBtn) bracketViewBtn.style.display = isNonStandard ? 'none' : '';
    if (bracketListBtn) bracketListBtn.style.display = isNonStandard ? 'none' : '';

    const nexusKeyNote = dataSource === 'nexus' && nexusKey !== tbaKey ? ` · Nexus key: ${nexusKey}` : '';

    const effectiveMode = isNonStandard ? 'list' : _bracketMode;

    if (nowQueuing) {
        container.innerHTML = `<div style="background:#14532d;border:1px solid #16a34a;border-radius:6px;padding:7px 12px;margin-bottom:12px;color:#86efac;font-size:0.88em;">Now queuing: <strong>${nowQueuing}</strong></div>`;
    } else {
        container.innerHTML = '';
    }

    if (effectiveMode === 'list') {
        _renderBracketList(container, matchEntries, nowQueuing, nexusKeyNote, dataSource, isNonStandard);
    } else {
        _renderBracketVisual(container, matchMap, nowQueuing, nexusKeyNote, dataSource);
    }
};

// ── List mode ────────────────────────────────────────────────────────────────
function _renderBracketList(container, entries, nowQueuing, nexusKeyNote, dataSource, isNonStandard) {
    const ROUND_LABEL = {
        1:'Upper R1', 2:'Upper R1', 3:'Upper R1', 4:'Upper R1',
        5:'Lower R2', 6:'Lower R2', 7:'Upper R2', 8:'Upper R2',
        9:'Lower R3', 10:'Lower R3', 11:'Upper Final', 12:'Lower R4',
        13:'Lower Final',
    };

    const isMobile = document.body.classList.contains('mobile-ui');

    const maxNonFinalsN = entries.filter(e => !e.isFinals).reduce((acc, e) => Math.max(acc, e.n), 0);
    const matchLabel = e => e.isFinals ? `F${e.n - maxNonFinalsN}` : `M${e.n}`;

    const maxTeams = Math.max(3, ...entries.flatMap(e => [(e.redTeams||[]).length, (e.blueTeams||[]).length]));

    const teamCell = (team, cls, shrink) => {
        const t = String(team).replace(/^frc/i, '');
        const sz = shrink ? 'font-size:0.8em;padding:2px 3px;' : '';
        return `<td class="${cls}" onclick="highlightTeam('${t}')" style="cursor:pointer;${sz}"><strong>${t}</strong></td>`;
    };

    const allianceCells = (teams, cls, shrink) => {
        const cells = teams.map(t => teamCell(t, cls, shrink));
        for (let i = teams.length; i < maxTeams; i++) cells.push(`<td class="${cls}"></td>`);
        return cells.join('');
    };

    const nonStandardNote = isNonStandard
        ? `<div style="background:#1e1b4b;border:1px solid #4338ca;border-radius:5px;padding:7px 11px;margin-bottom:10px;font-size:0.82em;color:#a5b4fc;">Non-standard bracket — visual view unavailable.</div>`
        : '';

    let html = `${nonStandardNote}<div style="font-size:0.78em;color:#475569;margin-bottom:10px;">Source: ${dataSource === 'nexus' ? 'Nexus' : 'TBA'}${nexusKeyNote}</div>`;

    const colSpan = isMobile ? maxTeams + 2 : maxTeams * 2 + 2;
    const numCols = Array.from({length: maxTeams}, (_, i) => i + 1);
    if (isMobile) {
        html += `<table style="width:100%;border-collapse:collapse;">
            <thead><tr>
                <th style="text-align:center;">Match</th>
                ${numCols.map(i => `<th>${i}</th>`).join('')}
                <th style="text-align:center;min-width:3.2rem;">Score</th>
            </tr></thead><tbody>`;
    } else {
        html += `<table style="width:100%;border-collapse:collapse;">
            <thead>
            <tr>
                <th rowspan="2">Match</th>
                <th colspan="${maxTeams}" class="red-header">Red Alliance</th>
                <th colspan="${maxTeams}" class="blue-header">Blue Alliance</th>
                <th rowspan="2">Result</th>
            </tr><tr>
                ${numCols.map(i => `<th class="red-header">${i}</th>`).join('')}
                ${numCols.map(i => `<th class="blue-header">${i}</th>`).join('')}
            </tr>
            </thead><tbody>`;
    }

    let lastRound = null;
    for (const e of entries) {
        const rLabel = isNonStandard ? null
            : (e.isFinals || e.n > 13) ? 'Finals'
            : (ROUND_LABEL[e.n] || `Playoff ${e.n}`);
        if (rLabel !== null && rLabel !== lastRound) {
            html += `<tr><td colspan="${colSpan}" style="font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#475569;padding:10px 4px 4px;border-bottom:1px solid #1e293b;">${rLabel}</td></tr>`;
            lastRound = rLabel;
        }

        const lbl = matchLabel(e);
        const { redTeams, blueTeams, redScore, blueScore, isQueuing, isDone } = e;
        const red  = redTeams  || [];
        const blue = blueTeams || [];
        const redWon  = redScore != null && blueScore != null && redScore  > blueScore;
        const blueWon = redScore != null && blueScore != null && blueScore > redScore;
        const scored  = redScore != null && redScore >= 0;
        const shrink  = Math.max(red.length, blue.length) > 3;

        const queuingBadge = isQueuing
            ? `<span style="background:#14532d;color:#86efac;border-radius:3px;padding:1px 4px;font-size:0.65em;font-weight:700;margin-left:4px;">▶</span>`
            : '';

        const redCells  = allianceCells(red,  'red-cell',  shrink);
        const blueCells = allianceCells(blue, 'blue-cell', shrink);

        if (isMobile) {
            const scoreCell = scored
                ? `<td rowspan="2" style="border-left:2px solid #334155;vertical-align:middle;text-align:center;white-space:nowrap;padding:4px 8px;">
                       <div style="color:${redWon?'#4ade80':'#94a3b8'};font-weight:${redWon?'800':'normal'}">${redScore}</div>
                       <div style="color:#334155;font-size:0.65em;">·</div>
                       <div style="color:${blueWon?'#4ade80':'#94a3b8'};font-weight:${blueWon?'800':'normal'}">${blueScore}</div>
                   </td>`
                : `<td rowspan="2" style="color:#64748b;font-style:italic;border-left:2px solid #334155;vertical-align:middle;text-align:center;">—</td>`;
            html += `<tr>
                <td class="match-number" rowspan="2" style="color:#3b82f6;font-weight:700;vertical-align:middle;text-align:center;white-space:nowrap;padding:4px 6px;">${lbl}${queuingBadge}</td>
                ${redCells}${scoreCell}
            </tr><tr>${blueCells}</tr>`;
        } else {
            const resultCell = scored
                ? `<td style="border-left:2px solid #334155;white-space:nowrap;">
                       <span style="color:${redWon?'#4ade80':'#94a3b8'};font-weight:${redWon?'bold':'normal'}">${redScore}</span>
                       <span style="color:#475569;"> – </span>
                       <span style="color:${blueWon?'#4ade80':'#94a3b8'};font-weight:${blueWon?'bold':'normal'}">${blueScore}</span>
                   </td>`
                : `<td style="color:#64748b;font-style:italic;border-left:2px solid #334155;">Upcoming</td>`;
            html += `<tr>
                <td class="match-number" style="color:#3b82f6;font-weight:700;white-space:nowrap;">${lbl}${queuingBadge}</td>
                ${redCells}${blueCells}${resultCell}
            </tr>`;
        }
    }

    html += `</tbody></table>`;
    container.insertAdjacentHTML('beforeend', html);
}

// ── Visual bracket mode ───────────────────────────────────────────────────────
function _renderBracketVisual(container, matchMap, nowQueuing, nexusKeyNote, dataSource) {
    // 6 columns: R1, R2, R3, R4, R5, Finals
    const COL_HEADERS = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5', 'Finals'];
    // Each column lists match numbers in top-to-bottom display order
    // Upper bracket on top, lower bracket below, with spacing
    const COLS = [
        [1, 2, null, 3, 4],                    // R1: M1,M2 | gap | M3,M4
        [7, 5, null, null, 8, 6],              // R2: M7(upper),M5(lower) | gap | M8(upper),M6(lower)
        [11, null, 9, 10],                      // R3: M11(upper final) | gap | M9,M10(lower)
        [null, 12],                             // R4: gap(upper) | M12(lower R4)
        [null, null, 13],                       // R5: M13
        [null, 14, 15],                         // Finals
    ];

    let html = `<div style="font-size:0.78em;color:#475569;margin-bottom:10px;">Source: ${dataSource === 'nexus' ? 'Nexus' : 'TBA'}${nexusKeyNote}</div><div class="bracket-view">`;

    for (let ci = 0; ci < COLS.length; ci++) {
        const col = COLS[ci];
        html += `<div class="bracket-col">
            <div class="bracket-col-header">${COL_HEADERS[ci]}</div>`;

        for (const mn of col) {
            if (mn === null) {
                html += `<div class="bracket-slot spacer"></div>`;
            } else {
                const e = matchMap[mn] ?? null;
                html += `<div class="bracket-slot">${_bracketCard(e, nowQueuing, 'visual', `Playoff ${mn}`)}</div>`;
            }
        }
        html += `</div>`;

        // Connector column between rounds (not after last)
        if (ci < COLS.length - 1) {
            html += `<div style="display:flex;flex-direction:column;width:16px;flex:0 0 16px;padding-top:38px;">`;
            // Draw bracket arms aligned with the pairs that feed into the next column
            // Upper bracket arms
            if (ci === 0) {
                // M1+M2 feed M7, M3+M4 feed M8
                html += `<div style="flex:2;border-right:1px solid #334155;border-bottom:1px solid #334155;"></div>
                          <div style="flex:2;border-right:1px solid #334155;border-top:1px solid #334155;"></div>
                          <div style="flex:1;"></div>
                          <div style="flex:2;border-right:1px solid #334155;border-bottom:1px solid #334155;"></div>
                          <div style="flex:2;border-right:1px solid #334155;border-top:1px solid #334155;"></div>`;
            } else {
                html += `<div style="flex:1;"></div>`;
            }
            html += `</div>`;
        }
    }

    html += `</div>`;
    container.insertAdjacentHTML('beforeend', html);
}

// ── Shared card renderer ──────────────────────────────────────────────────────
function _bracketCard(e, nowQueuing, mode, fallbackLabel) {
    if (!e) {
        // Placeholder for unplayed/unknown match
        const lbl = fallbackLabel || 'TBD';
        return mode === 'visual'
            ? `<div class="bracket-card"><div class="bracket-card-label">${lbl}</div>
               <div class="bracket-alliance red bracket-tbd">Red TBD</div>
               <div class="bracket-alliance blue bracket-tbd">Blue TBD</div></div>`
            : '';
    }
    const { label, redTeams, blueTeams, redScore, blueScore, isQueuing, isDone, status } = e;
    const redWon  = redScore  != null && blueScore != null && redScore  > blueScore;
    const blueWon = redScore  != null && blueScore != null && blueScore > redScore;

    const teamList = (teams) => teams.length
        ? teams.map(t => `<span onclick="highlightTeam('${t}')" style="cursor:pointer;text-decoration:underline dotted;">${t}</span>`).join(' ')
        : '<span class="bracket-tbd">TBD</span>';

    const scoreEl = (score) => score != null ? `<span class="bracket-score">${score}</span>` : '';
    const redClass  = `bracket-alliance red${redWon?' winner':isDone?' loser':''}`;
    const blueClass = `bracket-alliance blue${blueWon?' winner':isDone?' loser':''}`;

    if (mode === 'visual') {
        const cardClass = `bracket-card${isQueuing?' queuing':isDone?' done':''}`;
        return `<div class="${cardClass}">
            <div class="bracket-card-label">${label}${isQueuing ? ' <span style="color:#86efac;font-size:0.85em;">▶</span>' : ''}</div>
            <div class="${redClass}">${teamList(redTeams)}${scoreEl(redScore)}</div>
            <div class="${blueClass}">${teamList(blueTeams)}${scoreEl(blueScore)}</div>
        </div>`;
    }
    // list mode card
    const badge = isQueuing
        ? `<span style="background:#14532d;color:#86efac;border-radius:3px;padding:1px 5px;font-size:0.7em;font-weight:700;margin-left:6px;">QUEUING</span>`
        : isDone ? `<span style="background:#1e1b4b;color:#a5b4fc;border-radius:3px;padding:1px 5px;font-size:0.7em;margin-left:6px;">Done</span>`
        : status ? `<span style="background:#1e293b;color:#94a3b8;border-radius:3px;padding:1px 5px;font-size:0.7em;margin-left:6px;">${status}</span>` : '';
    return `<div style="background:#0f172a;border:1px solid ${isQueuing?'#16a34a':'#1e293b'};border-radius:5px;padding:9px 11px;margin-bottom:7px;max-width:320px;">
        <div style="font-size:0.78em;font-weight:700;color:#64748b;margin-bottom:5px;">${label}${badge}</div>
        <div class="${redClass}" style="margin-bottom:3px;">${teamList(redTeams)}${scoreEl(redScore)}</div>
        <div class="${blueClass}">${teamList(blueTeams)}${scoreEl(blueScore)}</div>
    </div>`;
}

// ── WATCH LIST ENGINE ────────────────────────────────────────────────────────

// Merge saved per-event threshold overrides onto game config defaults.
function getEffectiveThresholds(gameConfig, eventKey) {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(`rpThresholds_${eventKey}`) || '{}'); } catch {}
    return (gameConfig?.rpThresholds ?? []).map(rpt => ({
        ...rpt,
        threshold: saved[rpt.rpField] != null ? Number(saved[rpt.rpField]) : rpt.threshold,
    }));
}

// Average OPR across all known teams; used when a team has no OPR.
function wlEventAvgOPR(tbaMap) {
    const vals = Object.values(tbaMap).map(t => t.opr).filter(v => v != null && v > 0);
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 30;
}

// Standard deviation of an array (sample, N-1 denominator).
function wlStd(arr) {
    if (arr.length < 2) return Infinity;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (arr.length - 1));
}

// Compute all fusion intermediate values for one team — shared by predictor and debug tool.
// Returns: { epa, opr, sigmaEpaEst, sigmaEpaGen, sigmaEpa, sigmaOpr, fused, oprWeightPct, hasAdj }
function wlTeamFusionStats(tn, tbaMap, teamsMap, oprSigmaRel) {
    const tba    = tbaMap[parseInt(tn)];
    const stat   = teamsMap[parseInt(tn)];
    const hasAdj = !!(tba && getTeamIgnoredKeys(tba).length > 0 && tba.adjustedOPR != null);
    const opr    = hasAdj ? tba.adjustedOPR : (tba?.opr ?? null);
    const epa    = stat?.currentEPA ?? null;

    const epaSd       = stat?.epa?.total_points?.sd ?? (epa != null ? 0.15 * Math.abs(epa) : 0);
    const matchCount  = Math.max(1, stat?.matchCount ?? 1);
    const sigmaEpaEst = epaSd / Math.sqrt(matchCount);
    const sigmaEpaGen = epa != null ? 0.12 * Math.abs(epa) : 0;
    const sigmaEpa    = Math.sqrt(sigmaEpaEst ** 2 + sigmaEpaGen ** 2);
    const sigmaOpr    = (isFinite(oprSigmaRel) && opr != null) ? oprSigmaRel * Math.abs(opr) : Infinity;

    let fused, oprWeightPct;
    if (opr == null && epa == null) {
        fused = null; oprWeightPct = null;
    } else if (opr == null || hasAdj) {
        fused = opr ?? epa; oprWeightPct = opr != null ? 100 : 0;
    } else if (epa == null) {
        fused = opr; oprWeightPct = 100;
    } else if (!isFinite(sigmaOpr) || sigmaOpr <= 0) {
        fused = epa; oprWeightPct = 0;
    } else if (sigmaEpa <= 0) {
        fused = opr; oprWeightPct = 100;
    } else {
        const wEpa = 1 / (sigmaEpa ** 2);
        const wOpr = 1 / (sigmaOpr ** 2);
        fused = (wEpa * epa + wOpr * opr) / (wEpa + wOpr);
        oprWeightPct = wOpr / (wEpa + wOpr) * 100;
    }

    return { tn: parseInt(tn), epa, opr, sigmaEpaEst, sigmaEpaGen, sigmaEpa, sigmaOpr, fused, oprWeightPct, hasAdj };
}

// Inverse-variance fusion of EPA and OPR for one team.
// oprSigmaRel: coefficient of variation of OPR predictions derived from event residuals.
//   Infinity (no residuals yet) → weight collapses to zero → pure EPA.
//   As residuals accumulate, OPR earns weight proportional to its precision.
// EPA uncertainty = √(σ_estimation² + σ_generalization²):
//   σ_estimation = epa.sd / √matchCount (shrinks with more historical matches)
//   σ_generalization = 12% of EPA (floor: even perfect estimation leaves event-to-event drift)
// Adjusted OPR (for ignored-match teams) always takes priority.
function wlPredictedContribution(tn, tbaMap, teamsMap, avg, oprSigmaRel = Infinity) {
    const stats = wlTeamFusionStats(tn, tbaMap, teamsMap, oprSigmaRel);
    return stats.fused ?? avg;
}

function wlAlliancePredictedScore(teams, tbaMap, teamsMap, avg, oprSigmaRel = Infinity) {
    return (teams ?? []).reduce((s, tn) => s + (wlPredictedContribution(tn, tbaMap, teamsMap, avg, oprSigmaRel) ?? avg), 0);
}

// Box-Muller standard normal sample.
function wlGaussian() {
    return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// Gaussian prior σ as a fraction of predicted score, derived from per-team EPA standard deviations.
// Assumes team scoring contributions are independent: σ_alliance = √(σ₁² + σ₂² + σ₃²).
function wlAllianceSigmaRel(teams, teamsMap, predicted) {
    let sumSq = 0, found = 0;
    for (const tn of (teams ?? [])) {
        const sd = teamsMap[parseInt(tn)]?.epa?.total_points?.sd;
        if (sd != null) { sumSq += sd * sd; found++; }
    }
    if (!found || predicted <= 0) return 0.18;  // flat 18% CV if no EPA SDs available
    return Math.sqrt(sumSq) / predicted;
}

// Blended residual pool: empirical relative residuals + N_PRIOR synthetic Gaussian draws.
// Prior weight = 10 / (empirical_count + 10), shrinking as real data accumulates.
function wlBuildRelPool(relResiduals, teams, teamsMap, predicted) {
    const N_PRIOR  = 500;
    const sigmaRel = wlAllianceSigmaRel(teams, teamsMap, predicted);
    const prior    = Array.from({ length: N_PRIOR }, () => sigmaRel * wlGaussian());
    return [...relResiduals, ...prior];
}

// Differential pool: empirical margin residuals + Gaussian prior for the predicted margin.
// σ_diff = √(σ_red_abs² + σ_blue_abs²) where σ_x_abs = σ_x_rel × predicted_x.
// Sampling from this pool for win probability captures shared match-level noise (both alliances
// affected by field conditions, refs, etc.), preventing probabilities from reaching 100%/0%.
function wlBuildDiffPool(diffResiduals, redTeams, blueTeams, teamsMap, redPred, bluePred) {
    const N_PRIOR   = 500;
    const sigmaRed  = wlAllianceSigmaRel(redTeams,  teamsMap, redPred)  * redPred;
    const sigmaBlue = wlAllianceSigmaRel(blueTeams, teamsMap, bluePred) * bluePred;
    const sigmaDiff = Math.sqrt(sigmaRed * sigmaRed + sigmaBlue * sigmaBlue);
    const prior = Array.from({ length: N_PRIOR }, () => sigmaDiff * wlGaussian());
    return [...diffResiduals, ...prior];
}

// Relative residuals: (actualScore − predicted) / predicted, one per alliance per played match.
// Differential residuals: (actualRed − actualBlue) − (predRed − predBlue), in absolute points.
// Uses the same inverse-variance fusion as wlPredictedContribution: oprSigmaRel at match i is
// derived from residuals accumulated *before* match i, keeping predictions self-consistent.
function wlCollectResiduals(playedMatches, tbaMap, teamsMap) {
    const avg     = wlEventAvgOPR(tbaMap);
    const relRes  = [];
    const diffRes = [];
    for (const m of playedMatches) {
        if ((m.redScore ?? -1) < 0) continue;
        const oprSigmaRel = wlStd(relRes);   // Infinity until ≥2 residuals exist
        const rp = wlAlliancePredictedScore(m.red,  tbaMap, teamsMap, avg, oprSigmaRel);
        const bp = wlAlliancePredictedScore(m.blue, tbaMap, teamsMap, avg, oprSigmaRel);
        if (rp > 0) relRes.push((m.redScore  - rp) / rp);
        if (bp > 0) relRes.push((m.blueScore - bp) / bp);
        diffRes.push((m.redScore - m.blueScore) - (rp - bp));
    }
    return { relResiduals: relRes, diffResiduals: diffRes };
}

// Solve a fuel-specific OPR using computeLocalOPR on component scores instead of total scores.
// Falls back to a reduced team set (only teams seen in fuel matches) when the full matrix is
// underdetermined — e.g. early in an event or when using the debug cutoff control.
function wlComputeFuelOPR(playedMatches, allTeamNums, scoreComponent, gameConfig) {
    if (!gameConfig?.componentScores) return null;
    const fuelMs = playedMatches.map(m => {
        const rf = gameConfig.componentScores(m.redBreakdown)?.[scoreComponent];
        const bf = gameConfig.componentScores(m.blueBreakdown)?.[scoreComponent];
        return (rf != null && bf != null) ? { ...m, redScore: rf, blueScore: bf } : null;
    }).filter(Boolean);
    if (fuelMs.length < 2) return null;

    let result = computeLocalOPR(fuelMs, allTeamNums);
    if (result) return result;

    // Underdetermined (fewer matches than teams) — retry with only teams seen in fuel matches,
    // then re-expand back to allTeamNums-indexed array so callers stay unchanged.
    const seenNums = [...new Set(
        fuelMs.flatMap(m => [...(m.red ?? []), ...(m.blue ?? [])]).map(t => parseInt(t)).filter(n => !isNaN(n))
    )];
    const reduced = computeLocalOPR(fuelMs, seenNums);
    if (!reduced) return null;
    const full = new Array(allTeamNums.length).fill(0);
    seenNums.forEach((tn, i) => {
        const idx = allTeamNums.indexOf(tn);
        if (idx !== -1) full[idx] = reduced[i];
    });
    return full;
}

// Historical RP achievement rate for an alliance.  Returns max over 3 teams (optimistic).
function wlHistoricalRPRate(teamNumbers, rpField, playedMatches) {
    const rates = (teamNumbers ?? []).map(tn => {
        const played = playedMatches.filter(m =>
            (m.redScore ?? -1) >= 0 &&
            (m.red?.includes(String(tn)) || m.blue?.includes(String(tn)))
        );
        if (!played.length) return null;
        const hit = played.filter(m => {
            const bd = m.red?.includes(String(tn)) ? m.redBreakdown : m.blueBreakdown;
            return bd?.[rpField];
        });
        return hit.length / played.length;
    }).filter(r => r != null);
    return rates.length ? Math.max(...rates) : 0.5;
}

function wlSample(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

// Monte Carlo prediction for one unplayed match.
// Returns { redProb, tieProb, blueProb, redPredicted, bluePredicted, rpProbs: {red, blue} }
// Win probability uses differential residuals so shared match-level noise is captured; RP
// probability uses per-alliance relative residuals since RP thresholds are per-alliance.
function wlSimulateMatch(match, tbaMap, teamsMap, allTeamNums, relResiduals, diffResiduals, gameConfig, effectiveThresholds, playedMatches, fuelOPRCache, N = 500) {
    const avg         = wlEventAvgOPR(tbaMap);
    const oprSigmaRel = wlStd(relResiduals);   // Infinity pre-event → pure EPA until OPR validated
    const redPred     = wlAlliancePredictedScore(match.red,  tbaMap, teamsMap, avg, oprSigmaRel);
    const bluePred    = wlAlliancePredictedScore(match.blue, tbaMap, teamsMap, avg, oprSigmaRel);

    // Differential pool for win probability — models the score margin, not each alliance in isolation.
    const diffPool = wlBuildDiffPool(diffResiduals, match.red, match.blue, teamsMap, redPred, bluePred);
    // Per-alliance pools for RP threshold probability (still need per-alliance score estimates).
    const redPool  = wlBuildRelPool(relResiduals, match.red,  teamsMap, redPred);
    const bluePool = wlBuildRelPool(relResiduals, match.blue, teamsMap, bluePred);

    let redWins = 0, ties = 0;
    const rpRed = {}, rpBlue = {};
    for (const rpt of effectiveThresholds) { rpRed[rpt.rpField] = 0; rpBlue[rpt.rpField] = 0; }

    // Pre-compute predicted fuel per alliance (constant across simulations).
    const fuelPred = {};
    for (const rpt of effectiveThresholds) {
        if (rpt.threshold == null) continue;
        const fuelArr = fuelOPRCache[rpt.scoreComponent];
        // fuelArr is valid only if the OPR solve produced at least one positive value.
        // An all-zeros result (truthy array) occurs when breakdown data is missing/zero —
        // fall through to the EPA fallback in that case.
        const fuelOPRValid = Array.isArray(fuelArr) && fuelArr.some(v => v > 0.5);
        if (fuelOPRValid) {
            // Clip individual team OPR contributions to ≥ 0 — negative OPR is an artifact
            // of the least-squares solve and would wrongly suppress the alliance sum.
            fuelPred[rpt.rpField] = {
                r: (match.red  ?? []).reduce((s, tn) => { const idx = allTeamNums.indexOf(parseInt(tn)); return s + Math.max(0, fuelArr[idx] ?? 0); }, 0),
                b: (match.blue ?? []).reduce((s, tn) => { const idx = allTeamNums.indexOf(parseInt(tn)); return s + Math.max(0, fuelArr[idx] ?? 0); }, 0),
            };
        } else if (rpt.fuelEPAKey) {
            // OPR unavailable or degenerate — sum each team's Statbotics EPA fuel component.
            const epaSum = (tns) => (tns ?? []).reduce((s, tn) =>
                s + Math.max(0, teamsMap[parseInt(tn)]?.epa?.breakdown?.[rpt.fuelEPAKey] ?? 0), 0);
            fuelPred[rpt.rpField] = { r: epaSum(match.red), b: epaSum(match.blue) };
        } else {
            fuelPred[rpt.rpField] = { r: redPred * 0.4, b: bluePred * 0.4 };
        }
    }

    const diffPred = redPred - bluePred;
    for (let i = 0; i < N; i++) {
        const diffSim = diffPred + wlSample(diffPool);
        if (diffSim > 0)                   redWins++;
        else if (Math.abs(diffSim) < 1)    ties++;

        for (const rpt of effectiveThresholds) {
            if (rpt.threshold == null) continue;
            const { r: rf, b: bf } = fuelPred[rpt.rpField];
            const rsim = Math.max(0, rf * (1 + wlSample(redPool)));
            const bsim = Math.max(0, bf * (1 + wlSample(bluePool)));
            if (rsim >= rpt.threshold) rpRed[rpt.rpField]++;
            if (bsim >= rpt.threshold) rpBlue[rpt.rpField]++;
        }
    }

    // Binary RPs use historical rate, not per-trial sampling
    for (const rpt of effectiveThresholds) {
        if (rpt.threshold != null) continue;
        rpRed[rpt.rpField]  = Math.round(wlHistoricalRPRate(match.red,  rpt.rpField, playedMatches) * N);
        rpBlue[rpt.rpField] = Math.round(wlHistoricalRPRate(match.blue, rpt.rpField, playedMatches) * N);
    }

    // Apply linear calibration: p_cal = 0.5 + β(p − 0.5), then renormalize with tieProb intact.
    const β = wlCalibrationBeta;
    const rawRed  = redWins / N;
    const rawBlue = (N - redWins - ties) / N;
    const rawTie  = ties / N;
    const calRed  = Math.max(0, 0.5 + β * (rawRed  - 0.5));
    const calBlue = Math.max(0, 0.5 + β * (rawBlue - 0.5));
    const calSum  = calRed + calBlue + rawTie;

    return {
        redProb: calRed / calSum, tieProb: rawTie / calSum, blueProb: calBlue / calSum,
        redPredicted: redPred, bluePredicted: bluePred,
        rpProbs: {
            red:  Object.fromEntries(effectiveThresholds.map(r => [r.rpField, rpRed[r.rpField]  / N])),
            blue: Object.fromEntries(effectiveThresholds.map(r => [r.rpField, rpBlue[r.rpField] / N])),
        },
    };
}

// Sum actual RPs earned from played matches.
function wlComputeActualRP(playedMatches) {
    const rpMap = {};
    for (const m of playedMatches) {
        if ((m.redScore ?? -1) < 0) continue;
        const rRP = m.redBreakdown?.rp  ?? (m.redScore  > m.blueScore  ? 3 : m.redScore  === m.blueScore  ? 1 : 0);
        const bRP = m.blueBreakdown?.rp ?? (m.blueScore > m.redScore   ? 3 : m.blueScore === m.redScore   ? 1 : 0);
        for (const tn of (m.red  ?? [])) rpMap[String(tn)] = (rpMap[String(tn)] ?? 0) + rRP;
        for (const tn of (m.blue ?? [])) rpMap[String(tn)] = (rpMap[String(tn)] ?? 0) + bRP;
    }
    return rpMap;
}

// Simulate all remaining matches N times. Returns { [teamNumber]: { mean, p10, p90, meanRP } }.
// meanRP is computed analytically (sum of expected values) rather than by sampling, so it is
// stable across recomputes. Rank distribution (mean/p10/p90) still uses Monte Carlo.
function wlSimulateStandings(baseRP, unplayed, matchPredictions, effectiveThresholds, N = 1000) {
    // Analytical expected RP: deterministic, no sampling variance.
    const analyticalRP = { ...baseRP };
    for (const m of unplayed) {
        const pred = matchPredictions[m.key];
        if (!pred) continue;
        const blueProb   = Math.max(0, 1 - pred.redProb - pred.tieProb);
        const rExpWin    = 3 * pred.redProb + pred.tieProb;
        const bExpWin    = 3 * blueProb     + pred.tieProb;
        const rExpBonus  = effectiveThresholds.reduce((s, rpt) => s + (pred.rpProbs.red[rpt.rpField]  ?? 0), 0);
        const bExpBonus  = effectiveThresholds.reduce((s, rpt) => s + (pred.rpProbs.blue[rpt.rpField] ?? 0), 0);
        for (const tn of (m.red  ?? [])) analyticalRP[String(tn)] = (analyticalRP[String(tn)] ?? 0) + rExpWin + rExpBonus;
        for (const tn of (m.blue ?? [])) analyticalRP[String(tn)] = (analyticalRP[String(tn)] ?? 0) + bExpWin + bExpBonus;
    }

    // Monte Carlo for rank distribution only.
    const rankSamples = {};
    for (let i = 0; i < N; i++) {
        const simRP = { ...baseRP };
        for (const m of unplayed) {
            const pred = matchPredictions[m.key];
            if (!pred) continue;
            const r = Math.random();
            let rRP, bRP;
            if      (r < pred.redProb)                     { rRP = 3; bRP = 0; }
            else if (r < pred.redProb + pred.tieProb)      { rRP = 1; bRP = 1; }
            else                                            { rRP = 0; bRP = 3; }
            for (const rpt of effectiveThresholds) {
                if (Math.random() < (pred.rpProbs.red[rpt.rpField]  ?? 0)) rRP++;
                if (Math.random() < (pred.rpProbs.blue[rpt.rpField] ?? 0)) bRP++;
            }
            for (const tn of (m.red  ?? [])) simRP[String(tn)] = (simRP[String(tn)] ?? 0) + rRP;
            for (const tn of (m.blue ?? [])) simRP[String(tn)] = (simRP[String(tn)] ?? 0) + bRP;
        }
        Object.entries(simRP).sort((a, b) => b[1] - a[1]).forEach(([tn], idx) => {
            (rankSamples[tn] = rankSamples[tn] ?? []).push(idx + 1);
        });
    }
    return Object.fromEntries(Object.keys(analyticalRP).map(tn => {
        const rs = (rankSamples[tn] ?? []).sort((a, b) => a - b);
        return [tn, {
            mean:   rs.length ? +(rs.reduce((s, r) => s + r, 0) / rs.length).toFixed(1) : null,
            p10:    rs[Math.floor(rs.length * 0.10)] ?? null,
            p90:    rs[Math.floor(rs.length * 0.90)] ?? null,
            meanRP: +analyticalRP[tn].toFixed(1),
        }];
    }));
}

// Estimate how much a single match outcome shifts the focused team's projected rank.
// Runs two fixed-outcome simulations (red wins / blue wins) and returns the larger delta.
function wlComputeImpact(focusedTN, baselineMean, matchKey, unplayed, matchPredictions, baseRP, effectiveThresholds, N = 500) {
    const runFixed = (forceRed) => {
        const sums = [];
        for (let i = 0; i < N; i++) {
            const simRP = { ...baseRP };
            for (const m of unplayed) {
                const pred = matchPredictions[m.key];
                if (!pred) continue;
                let rRP, bRP;
                if (m.key === matchKey) {
                    if (forceRed > 0)       { rRP = 3; bRP = 0; }
                    else if (forceRed === 0){ rRP = 1; bRP = 1; }
                    else                    { rRP = 0; bRP = 3; }
                } else {
                    const r = Math.random();
                    if      (r < pred.redProb)                    { rRP = 3; bRP = 0; }
                    else if (r < pred.redProb + pred.tieProb)     { rRP = 1; bRP = 1; }
                    else                                           { rRP = 0; bRP = 3; }
                }
                for (const rpt of effectiveThresholds) {
                    if (Math.random() < (pred.rpProbs.red[rpt.rpField]  ?? 0)) rRP++;
                    if (Math.random() < (pred.rpProbs.blue[rpt.rpField] ?? 0)) bRP++;
                }
                for (const tn of (m.red  ?? [])) simRP[String(tn)] = (simRP[String(tn)] ?? 0) + rRP;
                for (const tn of (m.blue ?? [])) simRP[String(tn)] = (simRP[String(tn)] ?? 0) + bRP;
            }
            const ranked = Object.entries(simRP).sort((a, b) => b[1] - a[1]);
            const idx = ranked.findIndex(([tn]) => tn === String(focusedTN));
            sums.push(idx >= 0 ? idx + 1 : ranked.length + 1);
        }
        return sums.reduce((s, r) => s + r, 0) / sums.length;
    };
    const redMean  = runFixed(1);
    const blueMean = runFixed(-1);
    return {
        impact: Math.max(Math.abs(baselineMean - redMean), Math.abs(baselineMean - blueMean)),
        rankIfRedWins:  redMean,
        rankIfBlueWins: blueMean,
    };
}

// ── PRE-EVENT BASELINE SNAPSHOT ──────────────────────────────────────────────

function computePreEventSnapshot(eventKey, allMatches, teamsMap, tbaMap, allTeamNums, gameConfig, effectiveThresholds) {
    const matchPredictions = {};
    for (const m of allMatches) {
        matchPredictions[m.key] = wlSimulateMatch(
            m, tbaMap, teamsMap, allTeamNums,
            [], [],       // empty residuals → Gaussian prior only
            gameConfig, effectiveThresholds,
            [],           // no played matches
            {}            // empty fuel OPR → EPA fallback via fuelEPAKey
        );
    }
    const rankDistrib = wlSimulateStandings({}, allMatches, matchPredictions, effectiveThresholds);
    const snapshot = { computed: new Date().toISOString(), rankDistrib, matchPredictions };
    localStorage.setItem(`wlPreEventSnapshot_${eventKey}`, JSON.stringify(snapshot));
    return snapshot;
}

window.resetPreEventSnapshot = function(eventKey) {
    if (!wlDetailCache) return;
    const { allMatches, teamsMap, tbaMap, allTeamNums, gameConfig, effectiveThresholds } = wlDetailCache;
    localStorage.removeItem(`wlPreEventSnapshot_${eventKey}`);
    wlPreEventCache = computePreEventSnapshot(eventKey, allMatches, teamsMap, tbaMap, allTeamNums, gameConfig, effectiveThresholds);
    _setSnapshotStale(false);
    wlMatchesRenderedFor = null;
    renderWatchList();
};

function _setSnapshotStale(stale) {
    _snapshotStale = stale;
    const btn = document.getElementById('wl-recompute-btn');
    if (!btn) return;
    if (stale) {
        btn.style.background = 'rgba(120,53,15,0.35)';
        btn.style.color = '#fbbf24';
        btn.style.borderColor = '#92400e';
    } else {
        btn.style.background = 'rgba(20,83,45,0.35)';
        btn.style.color = '#4ade80';
        btn.style.borderColor = '#166534';
    }
}

// ── WATCH LIST CONTROLS (also rendered in Dev tab) ───────────────────────────

function buildWLControlsHTML(eventKey, effectiveThresholds, totalMatchCount) {
    const cutoffN = watchListCutoff;

    const thresholdCtrls = effectiveThresholds.filter(r => r.threshold != null).map(rpt => `
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span style="color:#94a3b8;font-size:0.85em;white-space:nowrap;">${rpt.label} threshold</span>
            <input type="number" min="0" step="1" value="${rpt.threshold}"
                style="width:64px;padding:3px 6px;background:#0f172a;border:1px solid #334155;color:#f8fafc;border-radius:4px;font-size:0.9em;"
                onchange="saveWatchRPThreshold('${rpt.rpField}',this.value,'${eventKey}')">
            <span style="color:#64748b;font-size:0.85em;">fuel</span>
        </div>`).join('');

    const resetBtn = effectiveThresholds.some(r => r.threshold != null)
        ? `<button onclick="resetWatchRPThresholds('${eventKey}')" style="background:transparent;color:#64748b;border:1px solid #334155;border-radius:4px;padding:3px 10px;font-size:0.82em;cursor:pointer;">Reset thresholds</button>`
        : '';

    const cutoffCtrl = `
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span style="color:#94a3b8;font-size:0.85em;white-space:nowrap;">Simulate from after Q</span>
            <input type="number" min="1" max="${totalMatchCount}" value="${cutoffN ?? ''}" placeholder="all"
                style="width:56px;padding:3px 6px;background:#0f172a;border:1px solid #334155;color:#f8fafc;border-radius:4px;font-size:0.9em;"
                onchange="setWatchListCutoff(this.value||null)">
            <span style="color:#64748b;font-size:0.85em;">/ ${totalMatchCount}</span>
            ${cutoffN != null ? `<button onclick="setWatchListCutoff(null)" style="background:transparent;color:#64748b;border:1px solid #334155;border-radius:4px;padding:3px 8px;font-size:0.82em;cursor:pointer;">Live</button>` : ''}
        </div>`;

    return `<div style="display:flex;flex-direction:column;gap:10px;padding:10px 0;">
        ${thresholdCtrls}
        ${resetBtn}
        ${cutoffCtrl}
    </div>`;
}

// ── WATCH LIST RENDERER ──────────────────────────────────────────────────────

function showWatchListStale() {
    const updateBtn = document.getElementById('wl-update-btn');
    if (updateBtn) {
        // Watch list already rendered — just surface the Update button in the banner.
        updateBtn.style.display = '';
        return;
    }
    // No rendered content yet — show a compute placeholder.
    const container = document.getElementById('schedule-sub-watchlist');
    if (!container) return;
    container.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:48px 20px;gap:14px;">
            <button onclick="renderWatchList()"
                style="background:#1d4ed8;color:#f8fafc;border:none;border-radius:8px;padding:10px 28px;font-size:1em;font-weight:600;cursor:pointer;letter-spacing:0.02em;">
                Compute Watch List
            </button>
        </div>`;
}

window.renderWatchList = async function renderWatchList() {
    watchListDirty = false;
    const container = document.getElementById('schedule-sub-watchlist');
    if (!container) return;
    container.innerHTML = `<p style="color:#64748b;padding:20px 0;">Computing Watch List…</p>`;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (eventKey) {
        const saved = parseFloat(localStorage.getItem(`wlCalibrationBeta_${eventKey}`));
        wlCalibrationBeta = isNaN(saved) ? 1.0 : saved;
    }
    if (!eventKey) {
        container.innerHTML = `<p style="color:#64748b;padding:20px 0;">Enter an event key first.</p>`;
        return;
    }

    const gameConfig = getGameConfig(eventKey);
    const effectiveThresholds = getEffectiveThresholds(gameConfig, eventKey);

    const [allMatches, allTBATeams, allTeams] = await Promise.all([
        db.matches.where('eventKey').equals(eventKey).toArray(),
        db.tbaTeams.toArray(),
        db.teams.toArray(),
    ]);

    if (!allMatches.length) {
        const effectiveThresholdsNoSched = getEffectiveThresholds(gameConfig, eventKey);
        const controlsNoSched = effectiveThresholdsNoSched.some(r => r.threshold != null)
            ? buildWLControlsHTML(eventKey, effectiveThresholdsNoSched, 0)
            : '';
        container.innerHTML = controlsNoSched +
            `<p style="color:#64748b;padding:20px 0;">No schedule loaded — sync TBA matches first.</p>`;
        return;
    }

    const tbaMap   = Object.fromEntries(allTBATeams.map(t => [t.teamNumber, t]));
    const teamsMap = Object.fromEntries(allTeams.map(t => [t.teamNumber, t]));
    const allTeamNums = [...new Set(
        allMatches.flatMap(m => [...(m.red ?? []), ...(m.blue ?? [])]).map(tn => parseInt(tn)).filter(n => !isNaN(n))
    )];
    const cutoffN  = watchListCutoff;
    const totalMatchCount = allMatches.length;

    // Load or silently compute pre-event baseline (run once per event key)
    const snapshotKey = `wlPreEventSnapshot_${eventKey}`;
    const storedSnap = localStorage.getItem(snapshotKey);
    wlPreEventCache = storedSnap ? JSON.parse(storedSnap) : null;
    if (!wlPreEventCache) {
        wlPreEventCache = computePreEventSnapshot(eventKey, allMatches, teamsMap, tbaMap, allTeamNums, gameConfig, effectiveThresholds);
    }

    const playedMatches = allMatches.filter(m =>
        (m.redScore ?? -1) >= 0 && (cutoffN == null || m.matchNumber <= cutoffN)
    );
    const unplayed = allMatches.filter(m =>
        (m.redScore ?? -1) < 0 || (cutoffN != null && m.matchNumber > cutoffN)
    );

    const lastPlayedNum = playedMatches.length > 0
        ? Math.max(...playedMatches.map(m => m.matchNumber)) : null;
    wlComputedAsOf = cutoffN != null ? `Q${cutoffN}` : lastPlayedNum != null ? `Q${lastPlayedNum}` : null;

    const focusedTN  = String(window.currentFocusedTeam || OWN_TEAM);
    const { relResiduals, diffResiduals } = wlCollectResiduals(playedMatches, tbaMap, teamsMap);

    // Pre-build fuel OPR for each unique score component used in threshold RPs.
    // Falls back to Statbotics EPA (fuelEPAKey) when OPR is underdetermined.
    const fuelOPRCache = {};
    for (const rpt of effectiveThresholds) {
        if (rpt.threshold != null && rpt.scoreComponent && !(rpt.scoreComponent in fuelOPRCache)) {
            fuelOPRCache[rpt.scoreComponent] = wlComputeFuelOPR(playedMatches, allTeamNums, rpt.scoreComponent, gameConfig);
        }
    }

    // Predict all unplayed matches
    const matchPredictions = {};
    for (const m of unplayed) {
        matchPredictions[m.key] = wlSimulateMatch(m, tbaMap, teamsMap, allTeamNums, relResiduals, diffResiduals, gameConfig, effectiveThresholds, playedMatches, fuelOPRCache);
    }


    const baseRP      = wlComputeActualRP(playedMatches);
    const rankDistrib = wlSimulateStandings(baseRP, unplayed, matchPredictions, effectiveThresholds);

    wlDetailCache = { allMatches, matchPredictions, baseRP, playedMatches, effectiveThresholds, rankDistrib, teamsMap,
                      tbaMap, allTeamNums, relResiduals, diffResiduals, fuelOPRCache, gameConfig };

    const focDist = rankDistrib[focusedTN] ?? { mean: '?', p10: '?', p90: '?' };
    const focRP   = baseRP[focusedTN] ?? 0;
    const top12Set = new Set(
        Object.entries(rankDistrib).sort((a, b) => a[1].mean - b[1].mean).slice(0, 12).map(([tn]) => tn)
    );
    const focusedRank = typeof rankDistrib[focusedTN]?.mean === 'number' ? rankDistrib[focusedTN].mean : null;

    const yourMatches  = unplayed.filter(m =>  m.red?.includes(focusedTN) || m.blue?.includes(focusedTN));
    const otherMatches = unplayed.filter(m => !m.red?.includes(focusedTN) && !m.blue?.includes(focusedTN));

    // ── Controls (rendered into Dev tab, not Watch List banner) ──────────────
    const devCtrlEl = document.getElementById('dev-wl-controls-content');
    if (devCtrlEl) devCtrlEl.innerHTML = buildWLControlsHTML(eventKey, effectiveThresholds, totalMatchCount);

    const debugBanner = cutoffN != null
        ? `<div style="background:rgba(251,191,36,0.08);border:1px solid #b45309;border-radius:6px;padding:8px 14px;margin-bottom:12px;color:#fbbf24;font-size:0.85em;">
               ⚠ Debug mode — simulating from match ${cutoffN} (${playedMatches.length} played, ${unplayed.length} remaining)
           </div>` : '';

    // ── Standings table ───────────────────────────────────────────────────────

    const standingsRows = Object.entries(rankDistrib)
        .sort((a, b) => a[1].mean - b[1].mean)
        .slice(0, 12)
        .map(([tn, d], i) => {
            const isFoc  = tn === focusedTN;
            const stat   = teamsMap[parseInt(tn)];
            const epa    = stat?.currentEPA;
            const sd     = stat?.epa?.total_points?.sd;
            const epaStr = epa != null
                ? `<span style="color:#64748b;font-size:0.78em;font-weight:400;margin-left:5px;">${epa.toFixed(0)}${sd != null ? ` ±${sd.toFixed(0)}` : ''}</span>`
                : '';
            const preD = wlPreEventCache?.rankDistrib?.[tn];
            const prePart = preD ? `<span style="color:#475569;font-size:0.78em;"> · pre&nbsp;${preD.mean}</span>` : '';
            return `<tr onclick="viewTeamDetail(${tn},'matches')" style="cursor:pointer;${isFoc ? 'background:rgba(251,191,36,0.06);font-weight:600;' : ''}">
                <td style="padding:4px 8px;text-align:center;color:#64748b;">${i + 1}</td>
                <td style="padding:4px 8px;${isFoc ? 'color:#fbbf24;' : ''}">${tn}${ownStar(tn)}${epaStr}</td>
                <td style="padding:4px 8px;text-align:center;color:#94a3b8;">${baseRP[tn] ?? 0}</td>
                <td style="padding:4px 8px;text-align:center;">${d.mean}${prePart} <span style="color:#64748b;font-size:0.82em;font-weight:400;">(${d.meanRP.toFixed(1)} RP)</span></td>
                <td style="padding:4px 8px;text-align:center;color:#64748b;font-size:0.82em;">${d.p10}–${d.p90}</td>
            </tr>`;
        }).join('');

    // ── Match card builder ────────────────────────────────────────────────────

    const matchCard = (m, pred, impactLabel = '', rankInfo = null) => {
        if (!m || !pred) return '';
        const redFoc  = m.red?.includes(focusedTN);
        const blueFoc = m.blue?.includes(focusedTN);

        // Focused team's alliance always on the left; default red-left
        const leftIsBlue  = blueFoc && !redFoc;
        const leftTeams   = leftIsBlue ? (m.blue ?? []) : (m.red  ?? []);
        const rightTeams  = leftIsBlue ? (m.red  ?? []) : (m.blue ?? []);
        const leftLabel   = leftIsBlue ? 'BLU' : 'RED';
        const rightLabel  = leftIsBlue ? 'RED' : 'BLU';
        const leftColor   = leftIsBlue ? '#3b82f6' : '#ef4444';
        const rightColor  = leftIsBlue ? '#ef4444' : '#3b82f6';
        const leftBarClr  = leftIsBlue ? 'rgba(59,130,246,0.5)'  : 'rgba(239,68,68,0.5)';
        const rightBarClr = leftIsBlue ? 'rgba(239,68,68,0.5)'   : 'rgba(59,130,246,0.5)';
        const leftPctNum  = leftIsBlue ? pred.blueProb  : pred.redProb;
        const rightPctNum = leftIsBlue ? pred.redProb   : pred.blueProb;
        const leftPred    = leftIsBlue ? pred.bluePredicted : pred.redPredicted;
        const rightPred   = leftIsBlue ? pred.redPredicted  : pred.bluePredicted;
        const leftRank    = rankInfo ? (leftIsBlue  ? rankInfo.rankIfBlueWins : rankInfo.rankIfRedWins)  : null;
        const rightRank   = rankInfo ? (leftIsBlue  ? rankInfo.rankIfRedWins  : rankInfo.rankIfBlueWins) : null;

        const leftPct  = Math.round(leftPctNum  * 100);
        const rightPct = Math.round(rightPctNum * 100);

        // Favorable = focused team's own alliance (always left), or whichever side gives a better rank
        let favorableLeft = null;
        if (redFoc || blueFoc) {
            favorableLeft = true;
        } else if (leftRank !== null && rightRank !== null && leftRank !== rightRank) {
            favorableLeft = leftRank < rightRank;  // lower rank number = better
        }

        const teamSpan = (tn) => {
            const isFoc = String(tn) === focusedTN;
            let skull = '';
            if (!isFoc && focusedRank !== null && top12Set.has(String(tn))) {
                const tnRank = rankDistrib[String(tn)]?.mean;
                if (typeof tnRank === 'number') {
                    const color = tnRank < focusedRank ? '#ef4444' : '#fbbf24';
                    const title = tnRank < focusedRank
                        ? `Projected ahead of ${focusedTN} (~rank ${tnRank.toFixed(1)})`
                        : `Projected behind ${focusedTN} (~rank ${tnRank.toFixed(1)})`;
                    skull = `<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="${color}" fill-rule="evenodd" style="vertical-align:middle;margin-left:2px;flex-shrink:0;" title="${title.replace(/"/g,'&quot;')}"><path d="M12,2A9,9 0 0,1 21,11C21,14.03 19.5,16.82 17,18.5V21A1,1 0 0,1 16,22H8A1,1 0 0,1 7,21V18.5C4.5,16.82 3,14.03 3,11A9,9 0 0,1 12,2M9,9A2,2 0 0,0 7,11A2,2 0 0,0 9,13A2,2 0 0,0 11,11A2,2 0 0,0 9,9M15,9A2,2 0 0,0 13,11A2,2 0 0,0 15,13A2,2 0 0,0 17,11A2,2 0 0,0 15,9M12,17A1,1 0 0,0 11,18A1,1 0 0,0 12,19A1,1 0 0,0 13,18A1,1 0 0,0 12,17Z"/></svg>`;
                }
            }
            return `<span onclick="event.stopPropagation();highlightTeam('${tn}')" style="cursor:pointer;${isFoc?'color:#fbbf24;font-weight:700;':''}">${tn}${ownStar(tn)}${skull}</span>`;
        };

        const rankRow = `
            <div style="display:flex;justify-content:space-between;font-size:0.78em;color:#64748b;margin-top:3px;">
                <span>${leftRank != null ? `rank if won: ${leftRank.toFixed(1)}` : '…'}</span>
                <span>${rightRank != null ? `rank if won: ${rightRank.toFixed(1)}` : '…'}</span>
            </div>`;

        const rpRow = effectiveThresholds.length ? `
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:7px;font-size:0.78em;color:#94a3b8;">
                ${effectiveThresholds.map(rpt => {
                    const prob = redFoc ? (pred.rpProbs.red[rpt.rpField] ?? 0)
                               : blueFoc ? (pred.rpProbs.blue[rpt.rpField] ?? 0)
                               : Math.max(pred.rpProbs.red[rpt.rpField] ?? 0, pred.rpProbs.blue[rpt.rpField] ?? 0);
                    const pct = Math.round(prob * 100);
                    const fill = `linear-gradient(to right,#22c55e ${pct}%,#1e293b ${pct}%)`;
                    return `<span>${rpt.label} <span style="display:inline-block;width:44px;height:6px;border-radius:3px;background:${fill};vertical-align:middle;margin:0 3px;"></span>${pct}%</span>`;
                }).join('')}
            </div>` : '';

        return `<div id="wmc-${m.key}" onclick="viewMatchDetail('${m.key}')" style="cursor:pointer;background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px 14px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
                <strong style="color:#3b82f6;cursor:pointer;" onclick="event.stopPropagation();viewMatchPrep('${m.key}')">Q${m.matchNumber}</strong>
                ${impactLabel ? `<span style="color:#64748b;font-size:0.78em;">rank impact ±${impactLabel}</span>` : ''}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:6px;gap:8px;">
                <div>
                    <div style="color:${leftColor};font-size:0.72em;font-weight:700;letter-spacing:0.05em;margin-bottom:2px;">${leftLabel}</div>
                    <div style="font-size:0.88em;">${leftTeams.map(teamSpan).join(' · ')}</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:${rightColor};font-size:0.72em;font-weight:700;letter-spacing:0.05em;margin-bottom:2px;">${rightLabel}</div>
                    <div style="font-size:0.88em;">${rightTeams.map(teamSpan).join(' · ')}</div>
                </div>
            </div>
            <div style="display:flex;height:18px;border-radius:4px;overflow:hidden;">
                <div style="flex:${Math.max(leftPct,1)};background:${leftBarClr};display:flex;align-items:center;padding:0 6px;font-size:0.75em;font-weight:700;${favorableLeft===true?'box-shadow:inset 0 0 0 2px rgba(255,255,255,0.55);':''}">
                    <span style="color:${leftIsBlue?'#93c5fd':'#fca5a5'};">${leftPct}%</span>
                </div>
                <div style="flex:${Math.max(rightPct,1)};background:${rightBarClr};display:flex;align-items:center;justify-content:flex-end;padding:0 6px;font-size:0.75em;font-weight:700;${favorableLeft===false?'box-shadow:inset 0 0 0 2px rgba(255,255,255,0.55);':''}">
                    <span style="color:${leftIsBlue?'#fca5a5':'#93c5fd'};">${rightPct}%</span>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.82em;color:#475569;margin-top:5px;">
                <span>${leftPred.toFixed(0)} pts</span>
                <span>${rightPred.toFixed(0)} pts</span>
            </div>
            ${rankRow}
            ${rpRow}
        </div>`;
    };

    // ── Initial render ────────────────────────────────────────────────────────

    const snapDate = wlPreEventCache
        ? new Date(wlPreEventCache.computed).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
        : '—';
    const hasOPR = Object.values(tbaMap).some(t => t.opr != null);
    const oprNote = hasOPR ? ' · <span style="color:#475569;font-size:0.78em;" title="OPR is now available — baseline was computed without it">+OPR</span>' : '';
    const recomputeStyle = _snapshotStale
        ? 'background:rgba(120,53,15,0.35);color:#fbbf24;border:1px solid #92400e;'
        : wlPreEventCache
            ? 'background:rgba(20,83,45,0.35);color:#4ade80;border:1px solid #166534;'
            : 'background:transparent;color:#64748b;border:1px solid #334155;';
    const recomputeRow = `
        <div style="display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="color:#475569;font-size:0.78em;">Snapshot: ${snapDate}${oprNote}</span>
            <button id="wl-recompute-btn" onclick="resetPreEventSnapshot('${eventKey}')"
                style="${recomputeStyle}border-radius:4px;padding:3px 10px;font-size:0.82em;cursor:pointer;"
                title="Recompute pre-event baseline using current EPA/OPR values">Recompute Predictions</button>
        </div>`;

    container.innerHTML = `
        ${recomputeRow}
        ${debugBanner}
        <div id="wl-main-banner" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#0f172a;border:1px solid #1e293b;border-radius:8px;margin-bottom:14px;">
            <span style="color:#f8fafc;font-weight:600;">Watching: ${focusedTN}</span>
            <span style="color:#334155;">|</span>
            <span style="color:#94a3b8;font-size:0.88em;">${focRP} RP · Proj rank ${focDist.p10}–${focDist.p90} (avg ${focDist.mean})${wlComputedAsOf ? ` · as of ${wlComputedAsOf}` : ''}</span>
            <button id="wl-update-btn" onclick="renderWatchList()" style="display:none;margin-left:auto;background:#1d4ed8;color:#f8fafc;border:none;border-radius:6px;padding:4px 14px;font-size:0.82em;font-weight:600;cursor:pointer;">Update</button>
        </div>

        <div id="wl-progress-wrap" style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;font-size:0.75em;color:#64748b;margin-bottom:4px;">
                <span>Computing impact analysis…</span>
                <span id="wl-progress-pct">0%</span>
            </div>
            <div style="height:4px;background:#1e293b;border-radius:2px;">
                <div id="wl-progress-bar" style="height:100%;width:0%;background:#3b82f6;border-radius:2px;transition:width 0.15s;"></div>
            </div>
        </div>

        <div style="margin-bottom:20px;">
            <div onclick="window.toggleWLSection('standings')" style="cursor:pointer;display:flex;align-items:center;justify-content:space-between;color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;user-select:none;">
                <span>Projected Standings (top 12)</span>
                <span id="wl-standings-arrow" style="font-size:0.9em;">${wlStandingsCollapsed ? '▶' : '▼'}</span>
            </div>
            <div id="wl-standings-body" style="${wlStandingsCollapsed ? 'display:none' : ''}">
                <div style="overflow-x:auto;">
                    <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
                        <thead><tr style="color:#64748b;font-size:0.78em;text-transform:uppercase;letter-spacing:0.04em;">
                            <th style="padding:4px 8px;text-align:center;">#</th>
                            <th style="padding:4px 8px;">Team · EPA ±SD</th>
                            <th style="padding:4px 8px;text-align:center;">RP Now</th>
                            <th style="padding:4px 8px;text-align:center;">Proj. Rank · Total RP</th>
                            <th style="padding:4px 8px;text-align:center;">Range</th>
                        </tr></thead>
                        <tbody>${standingsRows}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <div onclick="window.toggleWLSection('your')" style="cursor:pointer;display:flex;align-items:center;justify-content:space-between;color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;user-select:none;">
            <span>Your Remaining Matches (${yourMatches.length})</span>
            <span id="wl-your-arrow" style="font-size:0.9em;">${wlYourCollapsed ? '▶' : '▼'}</span>
        </div>
        <div id="wl-your-matches" style="${wlYourCollapsed ? 'display:none' : ''}">
            ${yourMatches.length
                ? yourMatches.map(m => matchCard(m, matchPredictions[m.key], '…')).join('')
                : `<p style="color:#475569;font-size:0.85em;margin-bottom:16px;">No remaining matches for team ${focusedTN}.</p>`}
        </div>

        <div id="wl-other-header" onclick="window.toggleWLSection('other')" style="cursor:pointer;display:flex;align-items:center;justify-content:space-between;color:#94a3b8;font-size:0.75em;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin:16px 0 8px;user-select:none;">
            <span id="wl-other-header-text">Matches to Watch — computing impact…</span>
            <span id="wl-other-arrow" style="font-size:0.9em;">${wlOtherCollapsed ? '▶' : '▼'}</span>
        </div>
        <div id="wl-other-matches" style="${wlOtherCollapsed ? 'display:none' : ''}">
            ${otherMatches.map(m => matchCard(m, matchPredictions[m.key], '')).join('')}
        </div>`;

    // ── Async impact analysis ─────────────────────────────────────────────────
    // Process one match per idle frame (your + other combined), then re-render both sections.

    const allWLMatches = [
        ...yourMatches.map(m => ({ match: m, isYours: true })),
        ...otherMatches.map(m => ({ match: m, isYours: false })),
    ];
    const impactResults = [];
    const _focusedTN   = focusedTN;
    const _baselineMean = +focDist.mean;
    const _unplayed    = unplayed;
    const _matchPred   = matchPredictions;
    const _baseRP      = baseRP;
    const _thresholds  = effectiveThresholds;
    const _matchCard   = matchCard;
    const _yourLen     = yourMatches.length;
    let _idx = 0;

    const sched = typeof requestIdleCallback !== 'undefined'
        ? (fn) => requestIdleCallback(fn, { timeout: 200 })
        : (fn) => setTimeout(fn, 0);

    const _total = allWLMatches.length;

    const updateProgress = (done) => {
        const pct = _total > 0 ? Math.round((done / _total) * 100) : 100;
        const bar  = document.getElementById('wl-progress-bar');
        const pctEl = document.getElementById('wl-progress-pct');
        if (bar)   bar.style.width = pct + '%';
        if (pctEl) pctEl.textContent = pct + '%';
    };

    const processNext = () => {
        if (_idx >= _total) {
            // Hide progress bar
            const wrap = document.getElementById('wl-progress-wrap');
            if (wrap) wrap.style.display = 'none';

            // Re-render "Your Matches" in match-number order with impact labels
            if (_yourLen > 0) {
                const yourResults = impactResults
                    .filter(r => r.isYours)
                    .sort((a, b) => a.match.matchNumber - b.match.matchNumber);
                const yourSection = document.getElementById('wl-your-matches');
                if (yourSection) yourSection.innerHTML =
                    yourResults.map(r => _matchCard(r.match, _matchPred[r.match.key], r.impact.toFixed(1) + ' pos', { rankIfRedWins: r.rankIfRedWins, rankIfBlueWins: r.rankIfBlueWins })).join('');
            }

            // Re-render "Matches to Watch" sorted by impact × uncertainty
            const significant = impactResults
                .filter(r => !r.isYours && r.impact >= 0.05)
                .sort((a, b) => b.score - a.score)
                .slice(0, 10);
            const headerText = document.getElementById('wl-other-header-text');
            const section    = document.getElementById('wl-other-matches');
            if (headerText) headerText.textContent = `Matches to Watch (${significant.length}) — by impact × uncertainty`;
            if (section) section.innerHTML = significant.length
                ? significant.map(r => _matchCard(r.match, _matchPred[r.match.key], r.impact.toFixed(1) + ' pos', { rankIfRedWins: r.rankIfRedWins, rankIfBlueWins: r.rankIfBlueWins })).join('')
                : `<p style="color:#475569;font-size:0.85em;">No matches found that significantly affect ${_focusedTN}'s ranking.</p>`;
            return;
        }
        const { match: m, isYours } = allWLMatches[_idx++];
        const pred = _matchPred[m.key];
        const uncertainty = 1 - Math.abs((pred?.redProb ?? 0.5) - (pred?.blueProb ?? 0.5));
        const scaledN = Math.max(25, Math.round(500 * uncertainty));
        const { impact, rankIfRedWins, rankIfBlueWins } = wlComputeImpact(_focusedTN, _baselineMean, m.key, _unplayed, _matchPred, _baseRP, _thresholds, scaledN);
        impactResults.push({ match: m, isYours, impact, score: impact * uncertainty, rankIfRedWins, rankIfBlueWins });
        updateProgress(_idx);
        sched(processNext);
    };
    sched(processNext);
}

window.saveWatchRPThreshold = function (rpField, value, eventKey) {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(`rpThresholds_${eventKey}`) || '{}'); } catch {}
    saved[rpField] = Number(value);
    localStorage.setItem(`rpThresholds_${eventKey}`, JSON.stringify(saved));
    renderWatchList();
};

window.resetWatchRPThresholds = function (eventKey) {
    localStorage.removeItem(`rpThresholds_${eventKey}`);
    renderWatchList();
};

window.setWatchListCutoff = function (val) {
    watchListCutoff = val == null ? null : parseInt(val);
    renderWatchList();
};

// Apply a linear calibration factor derived from backtesting.
// p_cal = 0.5 + beta*(p − 0.5). Resets to 1.0 (no-op) on page reload.
window.applyWLCalibration = function (beta) {
    wlCalibrationBeta = parseFloat(beta) || 1.0;
    const evk = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (evk) localStorage.setItem(`wlCalibrationBeta_${evk}`, String(wlCalibrationBeta));
    watchListDirty = true;
    if (document.getElementById('schedule-sub-watchlist')?.style.display !== 'none') {
        renderWatchList();
    }
    // Re-run backtest so the badge updates immediately to show the new current β
    runBacktest();
};

// --- Backtest event data fetcher ---

// Session-level cache so repeated runs don't re-fetch the same events.
const btEventCache = new Map();

// Fetch match results, OPR, and Statbotics EPA for any event key that isn't the current event.
// Returns { matches, tbaMap, teamsMap } in the same shape wlSimulateMatch expects.
async function btFetchEventData(eventKey) {
    if (btEventCache.has(eventKey)) return btEventCache.get(eventKey);

    const [matchResp, oprResp, sbResp] = await Promise.all([
        fetchTBA(`/event/${eventKey}/matches`),
        fetchTBA(`/event/${eventKey}/oprs`),
        fetch(`https://api.statbotics.io/v3/team_events?event=${eventKey}`).then(r => r.json()),
    ]);

    const matches = (Array.isArray(matchResp) ? matchResp : [])
        .filter(m => m.comp_level === 'qm')
        .sort((a, b) => a.match_number - b.match_number)
        .map(m => ({
            key:           m.key,
            matchNumber:   m.match_number,
            red:           m.alliances.red.team_keys.map(k => k.replace('frc', '')),
            blue:          m.alliances.blue.team_keys.map(k => k.replace('frc', '')),
            redScore:      m.alliances.red.score,
            blueScore:     m.alliances.blue.score,
            redBreakdown:  m.score_breakdown?.red  ?? null,
            blueBreakdown: m.score_breakdown?.blue ?? null,
        }));

    const tbaMap = {};
    for (const [teamKey, opr] of Object.entries(oprResp?.oprs ?? {})) {
        const tn = parseInt(teamKey.replace('frc', ''));
        tbaMap[tn] = { teamNumber: tn, opr };
    }

    const sbList = Array.isArray(sbResp) ? sbResp : (sbResp?.data ?? sbResp?.results ?? []);
    const teamsMap = {};
    for (const t of sbList) {
        teamsMap[t.team] = {
            teamNumber: t.team,
            currentEPA: t.epa?.total_points?.mean ?? null,
            epa:        t.epa ?? null,
        };
    }

    const result = { matches, tbaMap, teamsMap };
    btEventCache.set(eventKey, result);
    return result;
}

// --- Statistical helpers for backtest calibration ---

// Standard normal CDF (Abramowitz & Stegun 26.2.17, accurate to ~7 decimal places)
function btNormalCDF(z) {
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const p = 1 - d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))));
    return z >= 0 ? p : 1 - p;
}
function btP2(z) { return 2 * (1 - btNormalCDF(Math.abs(z))); }

// Chi-squared upper-tail p-value via Wilson-Hilferty normal approximation (good for df >= 3)
function btChi2P(chi2, df) {
    if (chi2 <= 0 || df < 1) return 1;
    const z = (Math.pow(chi2 / df, 1 / 3) - (1 - 2 / (9 * df))) / Math.sqrt(2 / (9 * df));
    return 1 - btNormalCDF(z);
}

// Spiegelhalter (1986) Z-test — tests calibration without binning.
// Z = Σ(y_i − p_i)(1 − 2p_i) / √(Σ p_i(1−p_i)(1−2p_i)²)
// Z > 0: overconfident (probs too extreme). Z < 0: underconfident (probs too conservative).
// H0: Z ~ N(0,1). High p-value = no significant miscalibration.
function btSpiegelhalter(results) {
    let num = 0, denom = 0;
    for (const { p, won } of results) {
        const y = won ? 1 : 0;
        num   += (y - p) * (1 - 2 * p);
        denom += p * (1 - p) * (1 - 2 * p) ** 2;
    }
    if (denom <= 0) return { z: 0, p: 1 };
    const z = num / Math.sqrt(denom);
    return { z, p: btP2(z) };
}

// Hosmer-Lemeshow C-statistic (bin-based chi-squared calibration test).
// C = Σ_k (O_k − E_k)² / (n_k · p̄_k · (1 − p̄_k)),  C ~ χ²(bins_used − 2)
// E_k = sum of predicted p_i in bin k (not just midpoint × count).
function btHosmerLemeshow(bins) {
    let C = 0, used = 0;
    for (const b of bins) {
        if (b.count < 1) continue;
        const Ek = b.predicted;     // sum of p_i, i.e. expected wins in bin
        const pk = Ek / b.count;
        if (pk <= 0 || pk >= 1) continue;
        C += (b.actual - Ek) ** 2 / (b.count * pk * (1 - pk));
        used++;
    }
    const df = Math.max(1, used - 2);
    return { C, df, p: btChi2P(C, df) };
}

// Wilson score 95% confidence interval for a proportion.
function btWilson(k, n) {
    if (n === 0) return [0, 1];
    const z = 1.96, p = k / n;
    const denom = 1 + z * z / n;
    const center = (p + z * z / (2 * n)) / denom;
    const margin = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom;
    return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

// Backtest the win-probability model across one or more event keys.
// For each match i within an event, predicts using only matches 0..i-1 (forward simulation).
// The current event is read from IndexedDB; all others are fetched from TBA + Statbotics.
window.runBacktest = async function () {
    const resultsEl = document.getElementById('algo-backtest-results');
    if (!resultsEl) return;

    const currentEv = localStorage.getItem('selectedEvent') || '';
    const input     = document.getElementById('bt-event-keys');
    if (input && !input.value.trim()) input.value = currentEv;

    const eventKeys = (input?.value || currentEv)
        .split(/[\s,]+/)
        .map(k => k.trim().toLowerCase())
        .filter(Boolean);

    if (!eventKeys.length) {
        resultsEl.innerHTML = '<p style="color:#64748b;">Enter at least one event key.</p>';
        return;
    }

    // Pre-load current event data from IndexedDB once
    const [rawMatchesDB, tbaArrDB, teamsArrDB] = await Promise.all([
        db.matches.toArray(),
        db.tbaTeams.toArray(),
        db.teams.toArray(),
    ]);
    const dbTBAMap   = {};  tbaArrDB.forEach(t   => { dbTBAMap[t.teamNumber]   = t; });
    const dbTeamsMap = {};  teamsArrDB.forEach(t  => { dbTeamsMap[t.teamNumber] = t; });
    const dbPlayed   = rawMatchesDB
        .filter(m => (m.redScore ?? -1) >= 0)
        .sort((a, b) => a.matchNumber - b.matchNumber);

    const results = [];        // { eventKey, matchNum, p, won }
    const eventRows = [];      // { key, count, error } — one per event for the header table

    for (let ei = 0; ei < eventKeys.length; ei++) {
        const eventKey = eventKeys[ei];
        resultsEl.innerHTML = `<p style="color:#64748b;font-style:italic;">Fetching ${eventKey} (${ei + 1}/${eventKeys.length})…</p>`;

        let matches, tbaMap, teamsMap;
        try {
            if (eventKey === currentEv) {
                matches  = dbPlayed;
                tbaMap   = dbTBAMap;
                teamsMap = dbTeamsMap;
            } else {
                ({ matches, tbaMap, teamsMap } = await btFetchEventData(eventKey));
                matches = matches.filter(m => (m.redScore ?? -1) >= 0);
            }
        } catch (e) {
            eventRows.push({ key: eventKey, count: 0, error: `Fetch failed: ${e.message ?? e}` });
            continue;
        }

        const played = [...matches].sort((a, b) => a.matchNumber - b.matchNumber);
        if (played.length < 3) {
            eventRows.push({ key: eventKey, count: 0, error: 'Too few played matches' });
            continue;
        }

        const evGameConfig = getGameConfig(eventKey);
        let evCount = 0;

        for (let i = 0; i < played.length; i++) {
            const m = played[i];
            if (m.redScore === m.blueScore) continue;   // skip ties

            const history = played.slice(0, i);
            const { relResiduals, diffResiduals } = wlCollectResiduals(history, tbaMap, teamsMap);

            const pred = wlSimulateMatch(
                m, tbaMap, teamsMap, [],
                relResiduals, diffResiduals,
                evGameConfig, [],   // win probability only, no RP thresholds
                history, {},
                400
            );

            results.push({ eventKey, matchNum: m.matchNumber, p: pred.redProb, won: m.redScore > m.blueScore });
            evCount++;
            if (i % 8 === 0) await new Promise(r => setTimeout(r, 0));
        }

        eventRows.push({ key: eventKey, count: evCount, error: null });
    }

    if (!results.length) {
        resultsEl.innerHTML = '<p style="color:#64748b;">No non-tie matches found across the selected events.</p>';
        return;
    }

    // Fold to "favorite" perspective: favP ∈ [0.5, 1] always represents the predicted winner.
    // Doubles effective n per bucket vs the red/blue framing and eliminates the asymmetric tail.
    // Spiegelhalter Z and β_opt are provably invariant to this folding.
    const favResults = results.map(r => ({
        p:   Math.max(r.p, 1 - r.p),
        won: (r.p >= 0.5) === r.won,   // did the predicted favorite actually win?
    }));

    // 5-bin calibration (50–60% … 90–100%) — track predicted (sum of p_i) for proper H-L E_k
    const bins = Array.from({ length: 5 }, (_, i) => ({
        label: `${50 + i * 10}–${60 + i * 10}%`, midpoint: 55 + i * 10,
        actual: 0, predicted: 0, count: 0,
    }));
    for (const r of favResults) {
        const b = Math.min(4, Math.floor((r.p - 0.5) * 10));
        bins[b].actual    += r.won ? 1 : 0;
        bins[b].predicted += r.p;
        bins[b].count++;
    }

    // Summary stats (Brier score and accuracy are invariant to the favorite-folding)
    const brier    = favResults.reduce((s, r) => s + (r.p - (r.won ? 1 : 0)) ** 2, 0) / favResults.length;
    const accuracy = favResults.filter(r => r.won).length / favResults.length;

    // Statistical tests (Spiegelhalter Z is also invariant to folding — proved in comments above)
    const sp = btSpiegelhalter(favResults);
    const hl = btHosmerLemeshow(bins);

    // β_opt = Σ(y-0.5)(p-0.5) / Σ(p-0.5)² — also invariant to folding
    let betaNum = 0, betaDenom = 0;
    for (const { p, won } of favResults) {
        betaNum   += ((won ? 1 : 0) - 0.5) * (p - 0.5);
        betaDenom += (p - 0.5) ** 2;
    }
    const betaOpt = betaDenom > 0 ? Math.max(0.1, Math.min(2.0, betaNum / betaDenom)) : 1.0;
    const betaCurrent = wlCalibrationBeta;

    const pBadge = (p, label) => {
        const [color, verdict] = p > 0.10 ? ['#22c55e', 'calibrated']
                               : p > 0.05 ? ['#f59e0b', 'marginal']
                               :            ['#ef4444', 'miscalibrated'];
        const pStr = p < 0.001 ? '<0.001' : p.toFixed(3);
        return `<span style="color:${color};font-weight:600">${verdict}</span> <span style="color:#64748b">(${label}, p = ${pStr})</span>`;
    };

    // Calibration table rows — each bucket gets a Wilson 95% CI bar
    const rows = bins.map(b => {
        if (!b.count) {
            return `<tr><td style="color:#2d3f57;padding:5px 8px;font-size:0.82em">${b.label}</td>
                <td colspan="3" style="color:#2d3f57;text-align:center;font-size:0.82em">—</td></tr>`;
        }
        const [lo, hi]  = btWilson(b.actual, b.count);
        const actualPct = b.actual / b.count * 100;
        const midInCI   = b.midpoint >= lo * 100 && b.midpoint <= hi * 100;
        const barColor  = midInCI ? '#3b82f6' : '#f59e0b';

        // 100px bar = 0–100%; CI band + actual tick + expected tick
        const toBar = (pct) => Math.max(0, Math.min(100, Math.round(pct)));
        const loW  = toBar(lo  * 100);
        const hiW  = toBar(hi  * 100);
        const actW = toBar(actualPct);
        const midW = toBar(b.midpoint);
        const bar = `<div style="position:relative;height:10px;width:100px;background:#1e293b;border-radius:3px;display:inline-block;vertical-align:middle;">
            <div style="position:absolute;top:0;left:${loW}px;width:${Math.max(1,hiW-loW)}px;height:100%;background:${barColor};opacity:0.22;"></div>
            <div style="position:absolute;top:0;left:${midW}px;width:1px;height:100%;background:#475569;"></div>
            <div style="position:absolute;top:0;left:${Math.max(0,actW-1)}px;width:2px;height:100%;background:${barColor};border-radius:1px;"></div>
        </div>`;
        const ciText = `${(lo*100).toFixed(0)}–${(hi*100).toFixed(0)}%`;
        const checkMark = midInCI
            ? `<span style="color:#22c55e">✓</span>`
            : `<span style="color:#f59e0b" title="Expected midpoint ${b.midpoint}% falls outside 95% CI">⚠</span>`;

        return `<tr>
            <td style="color:#94a3b8;padding:5px 8px;font-size:0.82em">${b.label}</td>
            <td style="text-align:center;padding:5px 8px;font-size:0.82em">${b.count}</td>
            <td style="padding:5px 8px">${bar}</td>
            <td style="text-align:center;padding:5px 8px;font-size:0.82em;color:#94a3b8">${actualPct.toFixed(0)}% <span style="color:#475569">[${ciText}]</span></td>
            <td style="text-align:center;padding:5px 8px;font-size:0.82em">${checkMark}</td>
        </tr>`;
    }).join('');

    // Per-event summary pills
    const evPills = eventRows.map(ev => {
        if (ev.error) {
            return `<span style="display:inline-flex;align-items:center;gap:5px;background:#1c0f0f;border:1px solid #7f1d1d;border-radius:4px;padding:2px 8px;font-size:0.78em;color:#fca5a5;">${ev.key} <span style="color:#64748b">— ${ev.error}</span></span>`;
        }
        return `<span style="display:inline-flex;align-items:center;gap:5px;background:#0c1929;border:1px solid #1e3a5f;border-radius:4px;padding:2px 8px;font-size:0.78em;color:#93c5fd;">${ev.key} <span style="color:#475569">${ev.count} matches</span></span>`;
    }).join(' ');

    resultsEl.innerHTML = `
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">${evPills}</div>
        <div style="display:flex;gap:24px;margin-bottom:14px;flex-wrap:wrap;">
            <div><span style="color:#64748b;font-size:0.82em;">Matches tested</span><br><b style="font-size:1.1em">${results.length}</b></div>
            <div><span style="color:#64748b;font-size:0.82em;">Accuracy</span><br><b style="font-size:1.1em">${(accuracy * 100).toFixed(0)}%</b></div>
            <div><span style="color:#64748b;font-size:0.82em;">Brier score</span><br><b style="font-size:1.1em">${brier.toFixed(3)}</b><span style="color:#475569;font-size:0.79em;margin-left:5px;">(random = 0.25)</span></div>
        </div>
        <div style="background:#0c1929;border:1px solid #1e293b;border-radius:7px;padding:12px 14px;margin-bottom:14px;font-size:0.83em;line-height:1.9;">
            <div>Spiegelhalter Z = ${sp.z.toFixed(2)} &nbsp;→&nbsp; ${pBadge(sp.p, 'no binning')}
                <span style="color:#475569;font-size:0.9em;margin-left:6px;">
                    ${sp.z > 0 ? '(overconfident — probabilities too extreme)' : '(underconfident — probabilities too conservative)'}
                </span>
            </div>
            <div>Hosmer-Lemeshow C = ${hl.C.toFixed(2)}, df = ${hl.df} &nbsp;→&nbsp; ${pBadge(hl.p, 'bin-based χ²')}</div>
        </div>
        <div style="background:#0c1929;border:1px solid #1e3a5f;border-radius:7px;padding:12px 14px;margin-bottom:14px;font-size:0.83em;">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <div>
                    <span style="color:#64748b;">Optimal calibration β</span>
                    <b style="margin-left:6px;font-size:1.05em;color:${betaOpt < 0.97 ? '#f59e0b' : betaOpt > 1.03 ? '#60a5fa' : '#22c55e'}">${betaOpt.toFixed(3)}</b>
                    <span style="color:#475569;margin-left:6px;font-size:0.9em;">
                        ${betaOpt < 0.97 ? 'shrinks probabilities toward 50%' : betaOpt > 1.03 ? 'sharpens probabilities away from 50%' : 'no adjustment needed'}
                    </span>
                </div>
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                    <button onclick="applyWLCalibration(${betaOpt.toFixed(4)})"
                        style="background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:5px;padding:4px 12px;font-size:0.85em;font-weight:600;cursor:pointer;">
                        Apply β = ${betaOpt.toFixed(3)} to Watch List
                    </button>
                    ${betaCurrent !== 1.0 ? `<button onclick="applyWLCalibration(1.0)"
                        style="background:#1a1a2e;color:#64748b;border:1px solid #334155;border-radius:5px;padding:4px 12px;font-size:0.85em;cursor:pointer;">
                        Reset (currently β = ${betaCurrent.toFixed(3)})
                    </button>` : ''}
                </div>
            </div>
            <p style="color:#475569;font-size:0.85em;margin:8px 0 0;line-height:1.5;">
                β = Σ(y−½)(p−½) / Σ(p−½)² — OLS estimate of the true calibration slope.
                Applying it corrects systematic over/underconfidence without changing the model's ranking of matches.
            </p>
        </div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="color:#475569;font-size:0.78em;text-align:left;border-bottom:1px solid #1e293b;">
                    <th style="padding:5px 8px;">Predicted Favorite Win%</th>
                    <th style="text-align:center;padding:5px 8px;">n</th>
                    <th style="padding:5px 8px;">Actual rate <span style="font-weight:400;color:#334155;">(95% CI band)</span></th>
                    <th style="text-align:center;padding:5px 8px;">Actual [95% CI]</th>
                    <th style="text-align:center;padding:5px 8px;">Expected in CI?</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
        <p style="color:#475569;font-size:0.79em;margin-top:12px;line-height:1.6;">
            Predictions are folded to the favorite's perspective (favP = max(redP, blueP)), doubling the effective sample per bucket.
            The bar shows a 95% Wilson CI (shaded) with the actual win rate (solid tick) and expected midpoint (gray tick).
            ✓ = the expected midpoint falls inside the CI. Spiegelhalter Z and β are invariant to this folding.
        </p>`;
};

window.toggleAlgoSection = function (id) {
    const body  = document.getElementById(id);
    const arrow = document.getElementById(id + '-arrow');
    if (!body) return;
    const collapsed = body.style.display === 'none';
    body.style.display  = collapsed ? '' : 'none';
    if (arrow) arrow.textContent = collapsed ? '▼' : '▶';
};

window.toggleWLSection = function (section) {
    if (section === 'standings') {
        wlStandingsCollapsed = !wlStandingsCollapsed;
        document.getElementById('wl-standings-body').style.display = wlStandingsCollapsed ? 'none' : '';
        document.getElementById('wl-standings-arrow').textContent = wlStandingsCollapsed ? '▶' : '▼';
    } else if (section === 'your') {
        wlYourCollapsed = !wlYourCollapsed;
        document.getElementById('wl-your-matches').style.display = wlYourCollapsed ? 'none' : '';
        document.getElementById('wl-your-arrow').textContent = wlYourCollapsed ? '▶' : '▼';
    } else {
        wlOtherCollapsed = !wlOtherCollapsed;
        document.getElementById('wl-other-matches').style.display = wlOtherCollapsed ? 'none' : '';
        document.getElementById('wl-other-arrow').textContent = wlOtherCollapsed ? '▶' : '▼';
    }
};

function pushCurrentRightPanel() {
    if (!document.body.classList.contains('split-ui')) return;
    for (const id of ['teamDetailView', 'matchDetailView', 'matchPrepView']) {
        const el = document.getElementById(id);
        const d = el?.style.display;
        if (d && d !== 'none') {
            rightPanelHistory.push({ id, display: d });
            el.style.display = 'none';
            return;
        }
    }
}

function popRightPanel() {
    if (rightPanelHistory.length === 0) return false;
    const { id, display } = rightPanelHistory.pop();
    document.getElementById(id).style.display = display;
    return true;
}

// Push a history entry so the native back gesture can dismiss overlays
function pushNavState(overlay) {
    history.pushState({ overlay }, '');
}

// Native back gesture/button: close the topmost visible overlay
window.addEventListener('popstate', () => {
    if (document.getElementById('scoutingBreakdownModal').style.display !== 'none') {
        window.closeScoutingBreakdown();
    } else if (document.getElementById('photoLightbox').style.display !== 'none') {
        window.closeLightbox();
    } else if (document.getElementById('matchDetailView').style.display !== 'none') {
        window.closeMatchDetail();
    } else if (document.getElementById('teamDetailView').style.display !== 'none') {
        window.goBack();
    } else if (window.currentView === 'matchPrepView') {
        window.switchView('scheduleView');
    }
});

window.setUIMode = function (mode) {
    localStorage.setItem('uiMode', mode);
    document.body.classList.toggle('mobile-ui', mode === 'mobile');
    document.body.classList.toggle('split-ui', mode === 'split');
    document.getElementById('desktopModeBtn')?.classList.toggle('active', mode === 'desktop');
    document.getElementById('mobileModeBtn')?.classList.toggle('active', mode === 'mobile');
    document.getElementById('splitModeBtn')?.classList.toggle('active', mode === 'split');
    displaySchedule();
};

function initUIMode() {
    const saved = localStorage.getItem('uiMode');
    const isMobile = /Mobi|Android|iPhone/i.test(navigator.userAgent);
    const mode = saved || (isMobile ? 'mobile' : 'desktop');
    window.setUIMode(mode);
}

window.setColorMode = function (mode) {
    localStorage.setItem('colorMode', mode);
    document.body.classList.toggle('light-mode', mode === 'light');
    document.getElementById('darkModeBtn')?.classList.toggle('active', mode === 'dark');
    document.getElementById('lightModeBtn')?.classList.toggle('active', mode === 'light');
};

window.toggleConfigPanel = function () {
    document.getElementById('config-panel')?.classList.toggle('open');
};

document.addEventListener('click', e => {
    const panel = document.getElementById('config-panel');
    const btn   = document.getElementById('configBtn');
    if (panel?.classList.contains('open') && !panel.contains(e.target) && !btn?.contains(e.target)) {
        panel.classList.remove('open');
    }
});

window.rerollQuip = function () {
    const container = document.getElementById('detailTeamQuip');
    const textEl = document.getElementById('detailTeamQuipText');
    if (!container || !container.dataset.tier || !textEl) return;
    // Re-roll always uses random tier-based quip, bypassing any event-specific quip
    textEl.textContent = getTeamQuip(Number(container.dataset.team), container.dataset.tier, true);
    container.dataset.hasEventQuip = 'false';
};

window.toggleQuipsEnabled = function () {
    const nowEnabled = !(localStorage.getItem('quipsEnabled') === 'true');
    localStorage.setItem('quipsEnabled', String(nowEnabled));
    const el = document.getElementById('detailTeamQuip');
    if (el) el.style.display = nowEnabled ? '' : 'none';
    renderDevTab();
};

function getQuipUserSeed() {
    let seed = parseInt(localStorage.getItem('quipUserSeed'), 10);
    if (!seed || isNaN(seed)) {
        seed = Math.floor(Math.random() * 0xffffffff);
        localStorage.setItem('quipUserSeed', String(seed));
    }
    return seed;
}

function initColorMode() {
    const saved = localStorage.getItem('colorMode') || 'dark';
    window.setColorMode(saved);
}

window.switchView = function (viewId, btn) {
    // In split mode, showing the team detail uses pushCurrentRightPanel to save
    // whatever is open (including matchPrepView) — handle it first, before the
    // prep-close block below would interfere.
    if (document.body.classList.contains('split-ui') && viewId === 'teamDetailView') {
        pushCurrentRightPanel();
        window.previousView = window.currentView;
        document.getElementById('teamDetailView').style.display = 'flex';
        updateDetailBackButton();
        return;
    }

    // In split mode, close the prep panel when navigating to a main left-panel view.
    if (document.body.classList.contains('split-ui')) {
        const prep = document.getElementById('matchPrepView');
        if (prep && prep.style.display === 'block') {
            prep.style.display = 'none';
            if (!popRightPanel()) {
                document.getElementById('splitRightPanel').style.display = 'flex';
            }
        }
    }

    // 1. Hide the current view
    const current = document.getElementById(window.currentView);
    if (current) current.style.display = 'none';

    // 2. Store the current view as 'previous' before we swap
    if (window.currentView !== viewId) {
        window.previousView = window.currentView;
    }

    // 3. Show the new view
    const next = document.getElementById(viewId);
    if (next) {
        next.style.display = viewId === 'teamDetailView' ? 'flex' : 'block';
        window.currentView = viewId;
    }

    // 4. Sync all nav items (top-nav and mobile bottom nav) by data-view attribute
    const MAIN_VIEWS = new Set(['homeView', 'scheduleView', 'dataView', 'toolsView']);
    if (MAIN_VIEWS.has(viewId)) {
        document.querySelectorAll('[data-view]').forEach(b => {
            b.classList.toggle('active', b.dataset.view === viewId);
        });
    }

    // 5. Lazy-render tools tab when first opened
    if (viewId === 'toolsView' && currentToolsTab === 'picklist') renderPickList();
    if (viewId === 'toolsView' && currentToolsTab === 'draft') renderDraft();

    // 6. Update the Back button label on the Team Detail page
    updateDetailBackButton();

    // 7. Re-anchor tab bars flush against nav in mobile layout
    positionMobileTabBars();
};

let currentDataTab = 'statbotics';
let dataChartVisible = true;

window.toggleDataChart = function () {
    dataChartVisible = !dataChartVisible;
    const display = dataChartVisible ? '' : 'none';
    const label   = dataChartVisible ? 'Hide Chart' : 'Show Chart';
    ['statboticsChartContainer', 'tbaChartContainer', 'scoutingChartContainer', 'dashboardChartContainer']
        .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = display; });
    ['dataChartToggleBtn', 'dashboardChartToggleBtn']
        .forEach(id => { const el = document.getElementById(id); if (el) el.textContent = label; });
};

function positionMobileTabBars() {
    if (!document.body.classList.contains('mobile-ui')) return;
    const nav = document.querySelector('.mobile-bottom-nav');
    if (!nav) return;
    requestAnimationFrame(() => {
        const navTop = nav.getBoundingClientRect().top;
        const primaryBottom = window.innerHeight - navTop;
        // Position every visible primary tab bar flush against the nav bar
        document.querySelectorAll('.app-view > .detail-tabs').forEach(el => {
            el.style.bottom = primaryBottom + 'px';
        });
        // Force reflow, then position sub-tabs flush against dataTabs specifically
        const dataTabs = document.getElementById('dataTabs');
        if (dataTabs) {
            const tabTop = dataTabs.getBoundingClientRect().top;
            const subBottom = window.innerHeight - tabTop;
            ['tbaTabs', 'scoutingSubTabs'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.bottom = subBottom + 'px';
            });
        }
    });
}
// Keep old name as alias so existing call sites don't break
const positionMobileSubTabs = positionMobileTabBars;

window.switchDataTab = function (tab) {
    currentDataTab = tab;
    ['statbotics', 'tba', 'scouting', 'algorithms'].forEach(t => {
        document.getElementById(`data-tab-${t}`).style.display = t === tab ? 'block' : 'none';
    });
    document.querySelectorAll('#dataTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['statbotics', 'tba', 'scouting', 'algorithms'][i] === tab);
    });
    positionMobileSubTabs();
    if (tab === 'scouting') {
        displayScoutingTeams();   // immediate render from cache or raw scouting
        computeScoutingFusion();  // async — re-renders when fusion completes
    }
};

// ─── SCOUTING DATA TAB ────────────────────────────────────────────────────────

let scoutingChartInstance = null;
let scoutingSortCol = 'total';
let scoutingSortDir = 1;
let scoutingTableView   = 'epa';    // 'epa' | 'functional'
let scoutingTableFormat = 'tiered'; // 'tiered' | 'gradient'

window.sortScoutingBy = function (col) {
    if (scoutingSortCol === col) scoutingSortDir *= -1;
    else { scoutingSortCol = col; scoutingSortDir = col === 'teamNumber' ? -1 : 1; }
    displayScoutingTeams();
};

window.setScoutingTableView = function (view) {
    scoutingTableView = view;
    displayScoutingTeams();
};

window.setScoutingTableFormat = function (fmt) {
    scoutingTableFormat = fmt;
    displayScoutingTeams();
};

function renderScoutingChart(rows) {
    const ctx = document.getElementById('scoutingComparisonChart').getContext('2d');
    if (scoutingChartInstance) scoutingChartInstance.destroy();
    const isMobile = document.body.classList.contains('mobile-ui');
    const labels = rows.map(r => r.teamNumber);
    scoutingChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Auto',    data: rows.map(r => r.auto.toFixed(1)),    backgroundColor: '#f59e0b', stack: 'epa' },
                { label: 'Teleop',  data: rows.map(r => r.teleop.toFixed(1)),  backgroundColor: '#3b82f6', stack: 'epa' },
                { label: 'Endgame', data: rows.map(r => r.endgame.toFixed(1)), backgroundColor: '#10b981', stack: 'epa' },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } },
                y: { stacked: true, beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#94a3b8' },
                     title: { display: true, text: 'Scouting EPA (pts/match)', color: '#94a3b8' } },
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#f8fafc', usePointStyle: true } },
                tooltip: { mode: 'index', intersect: false },
            },
        },
    });
}

window.computeScoutingFusion = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) return;
    const statusEl = document.getElementById('scouting-fusion-status');

    const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!rawStr) { if (statusEl) statusEl.textContent = 'No scouting data to fuse.'; return; }

    if (statusEl) statusEl.textContent = 'Computing…';

    const tbaMatches = await db.matches.where('eventKey').equals(eventKey).toArray();
    if (!tbaMatches.some(m => m.redBreakdown)) {
        if (statusEl) statusEl.textContent = 'No TBA breakdowns — run "Sync TBA Matches" first.';
        return;
    }

    const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
    if (!processed) { if (statusEl) statusEl.textContent = 'No game config for this event.'; return; }

    const { config, byTeam, observations } = processed;
    const allByMatch = indexObservationsByMatch(observations);
    const teams = {};

    for (const [teamNumber, rawRows] of Object.entries(byTeam)) {
        const { rows } = deduplicateTeamRows(rawRows);
        teams[teamNumber] = fuseScoutingWithTBA(teamNumber, rows, allByMatch, tbaMatches, config);
    }

    const fusedCount = Object.values(teams).filter(r => r.available).length;
    const now = new Date().toLocaleTimeString();
    localStorage.setItem(`scoutingFusedStats_${eventKey}`, JSON.stringify({ computed: now, teams }));
    if (statusEl) statusEl.textContent = `${fusedCount}/${Object.keys(teams).length} teams fused · ${now}`;
    displayScoutingTeams();
};

window.displayScoutingTeams = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const table = document.getElementById('scoutingTeamTable');
    const body  = document.getElementById('scoutingTeamBody');
    if (!table || !body) return;

    const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!rawStr) { table.style.display = 'none'; return; }

    const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
    if (!processed || !processed.config.computeEPABreakdown) { table.style.display = 'none'; return; }

    const { config, byTeam } = processed;

    // Load cached fusion results if available
    const fusedCache = (() => {
        try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; }
    })();

    const statusEl = document.getElementById('scouting-fusion-status');
    if (fusedCache && statusEl && !statusEl.textContent.includes('fused')) {
        const fusedCount = Object.values(fusedCache.teams).filter(r => r.available).length;
        statusEl.textContent = `${fusedCount}/${Object.keys(byTeam).length} teams fused · ${fusedCache.computed}`;
    }

    // When fusion data is present, filter to teams that appear in TBA match alliances.
    // Typo'd team numbers won't appear in any alliance and are excluded.
    let knownTeams = null;
    if (fusedCache) {
        const tbaMatches = await db.matches.where('eventKey').equals(eventKey).toArray();
        if (tbaMatches.length > 0) {
            knownTeams = new Set();
            for (const m of tbaMatches) {
                for (const t of [...(m.red || []), ...(m.blue || [])]) knownTeams.add(t);
            }
        }
    }

    // Load TBA team records so we can respect per-team scouting exclusions
    const allTBATeamsForScout = await db.tbaTeams.toArray();
    const tbaTeamMapForScout  = Object.fromEntries(allTBATeamsForScout.map(t => [String(t.teamNumber), t]));
    const allMatchesForScout  = await db.matches.toArray();

    // Build rows: use fused EPA breakdown when available, raw scouting otherwise
    let rows = Object.entries(byTeam)
        .filter(([teamNumber]) => !knownTeams || knownTeams.has(teamNumber))
        .map(([teamNumber, rawRows]) => {
        const tbaEntry = tbaTeamMapForScout[teamNumber];
        const scoutIgnoreKeys = tbaEntry?.scoutingIgnoreActive ? getTeamIgnoredKeys(tbaEntry) : [];
        let { rows: deduped } = deduplicateTeamRows(rawRows);
        let ignoredMatchNums = new Set();
        if (scoutIgnoreKeys.length > 0) {
            ignoredMatchNums = new Set(allMatchesForScout.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber));
            deduped = deduped.filter(r => !ignoredMatchNums.has(r.matchNumber));
        }
        const rawStats = config.aggregateTeam(deduped);
        const fusedResult = fusedCache?.teams?.[teamNumber];
        const effectiveFused = (fusedResult?.available && ignoredMatchNums.size > 0)
            ? refilteredFusedStats(fusedResult, ignoredMatchNums) : fusedResult;
        const isFused = effectiveFused?.available && config.computeFusedEPABreakdown;
        const breakdown = isFused
            ? config.computeFusedEPABreakdown(effectiveFused.stats)
            : config.computeEPABreakdown(rawStats);
        const funcVals = {};
        if (config.functionalColumns) {
            for (const col of config.functionalColumns) {
                funcVals[col.sortKey] = col.getValue(rawStats, effectiveFused) ?? null;
            }
        }
        return { teamNumber, matches: rawStats.matches, isFused, ...breakdown, ...funcVals };
    });

    // Sort
    rows.sort((a, b) => {
        const va = scoutingSortCol === 'teamNumber' ? parseInt(a.teamNumber) : (a[scoutingSortCol] ?? 0);
        const vb = scoutingSortCol === 'teamNumber' ? parseInt(b.teamNumber) : (b[scoutingSortCol] ?? 0);
        return (vb - va) * scoutingSortDir;
    });

    // Tier by scout EPA rank
    const sorted = [...rows].sort((a, b) => b.total - a.total);
    const tierOf = (tn) => {
        const i = sorted.findIndex(r => r.teamNumber === tn);
        return i < 8 ? 'S' : i < 20 ? 'A' : i < 32 ? 'B' : 'C';
    };

    renderScoutingChart([...rows].sort((a, b) => b.total - a.total));
    table.style.display = 'table';

    // Inject view + format toggles
    const toggleEl = document.getElementById('scouting-table-toggle');
    if (toggleEl) {
        const isEpa  = scoutingTableView === 'epa';
        const isTier = scoutingTableFormat === 'tiered';
        const btn = (label, onclick, active, activeColor = '#f8fafc') =>
            `<button onclick="${onclick}" style="border:none;padding:4px 12px;font-size:0.72em;cursor:pointer;font-weight:600;${active?`background:#1e293b;color:${activeColor}`:'background:transparent;color:#64748b'}">${label}</button>`;
        const viewPart = config.functionalColumns
            ? `<div style="display:inline-flex;gap:0;border:1px solid #334155;border-radius:5px;overflow:hidden;">
                   ${btn('EPA Breakdown', "window.setScoutingTableView('epa')",        isEpa,  '#f8fafc')}
                   ${btn('Functional',    "window.setScoutingTableView('functional')", !isEpa, '#60a5fa')}
               </div>`
            : '';
        const fmtPart = `<div style="display:inline-flex;gap:0;border:1px solid #334155;border-radius:5px;overflow:hidden;">
                   ${btn('Tiered',   "window.setScoutingTableFormat('tiered')",   isTier,  '#f8fafc')}
                   ${btn('Gradient', "window.setScoutingTableFormat('gradient')", !isTier, '#a78bfa')}
               </div>`;
        toggleEl.innerHTML = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">${viewPart}${fmtPart}</div>`;
    }

    // Update column headers to match view
    const theadRow = document.querySelector('#scoutingTeamTable thead tr');
    if (theadRow) {
        if (scoutingTableView === 'functional' && config.functionalColumns) {
            theadRow.innerHTML =
                `<th></th><th onclick="sortScoutingBy('teamNumber')" style="cursor:pointer;">Team ↕</th>` +
                config.functionalColumns.map(c =>
                    `<th onclick="sortScoutingBy('${c.sortKey}')" style="cursor:pointer;">${c.label} ↕</th>`
                ).join('') +
                `<th onclick="sortScoutingBy('matches')" style="cursor:pointer;">Matches ↕</th>`;
        } else {
            theadRow.innerHTML =
                `<th></th><th onclick="sortScoutingBy('teamNumber')" style="cursor:pointer;">Team ↕</th>` +
                `<th onclick="sortScoutingBy('total')" style="cursor:pointer;">Scout EPA ↕</th>` +
                `<th onclick="sortScoutingBy('auto')" style="cursor:pointer;">Auto ↕</th>` +
                `<th onclick="sortScoutingBy('teleop')" style="cursor:pointer;">Teleop ↕</th>` +
                `<th onclick="sortScoutingBy('endgame')" style="cursor:pointer;">Endgame ↕</th>` +
                `<th onclick="sortScoutingBy('matches')" style="cursor:pointer;">Matches ↕</th>`;
        }
    }

    const isFuncView  = scoutingTableView === 'functional' && config.functionalColumns;
    const isGradient  = scoutingTableFormat === 'gradient';

    // Per-column min/max for gradient shading
    const gradCols = isFuncView
        ? config.functionalColumns.map(c => c.sortKey)
        : ['total', 'auto', 'teleop', 'endgame'];
    const colMin = {}, colMax = {};
    if (isGradient) {
        for (const col of gradCols) {
            const vals = rows.map(r => r[col]).filter(v => v != null && isFinite(v));
            colMin[col] = vals.length > 0 ? Math.min(...vals) : 0;
            colMax[col] = vals.length > 0 ? Math.max(...vals) : 1;
        }
    }
    // Returns a CSS background declaration for a cell given its value and column key.
    // Interpolates hue green(142)→amber(42)→red(0) as value goes from best to worst.
    const gradBg = (val, col) => {
        if (!isGradient || val == null) return '';
        const range = colMax[col] - colMin[col];
        const t = range > 0 ? 1 - (val - colMin[col]) / range : 0.5;
        // RGB interpolation avoids hue-space paths that pass through unwanted colors.
        // Anchors: green rgb(35,67,47) → bg rgb(15,23,42) → red rgb(67,35,35)
        const p = t < 0.5 ? t * 2 : (t - 0.5) * 2;
        const r = t < 0.5 ? Math.round(35 - 20 * p) : Math.round(15 + 52 * p);
        const g = t < 0.5 ? Math.round(67 - 44 * p) : Math.round(23 + 12 * p);
        const b = t < 0.5 ? Math.round(47 -  5 * p) : Math.round(42 -  7 * p);
        return `background:rgb(${r},${g},${b});`;
    };

    body.innerHTML = rows.map(r => {
        const tier = tierOf(r.teamNumber);
        const rowStyle = isGradient
            ? `background:#0f172a;border-left:3px solid #334155;cursor:pointer;`
            : `background:${TIER_STYLE[tier].bg};border-left:6px solid ${TIER_STYLE[tier].color};cursor:pointer;`;
        const tierCell = isGradient
            ? `<td style="color:#475569;font-size:0.75em;font-weight:700;padding:4px 6px;">${tier}</td>`
            : `<td>${tierBadge(tier)}</td>`;

        if (isFuncView) {
            return `<tr style="${rowStyle}" onclick="viewTeamDetail(${r.teamNumber}, 'scouting')">
                ${tierCell}
                <td style="white-space:nowrap;"><strong>${r.teamNumber}</strong>${ownStar(r.teamNumber)}</td>
                ${config.functionalColumns.map(c => {
                    const val = r[c.sortKey];
                    const display = val == null ? '—'
                        : c.suffix === '%' ? Math.round(val) + '%'
                        : c.decimals != null ? val.toFixed(c.decimals)
                        : String(Math.round(val));
                    return `<td style="${gradBg(val, c.sortKey)}">${display}</td>`;
                }).join('')}
                <td style="color:#64748b;">${r.matches}</td>
            </tr>`;
        }
        const fusedDot = r.isFused
            ? `<span title="TBA-fused" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#4ade80;margin-left:5px;vertical-align:middle;"></span>`
            : '';
        return `<tr style="${rowStyle}" onclick="viewTeamDetail(${r.teamNumber}, 'scouting')">
            ${tierCell}
            <td style="white-space:nowrap;"><strong>${r.teamNumber}</strong>${ownStar(r.teamNumber)}</td>
            <td style="${gradBg(r.total,'total')}white-space:nowrap;"><strong>${r.total.toFixed(1)}</strong>${fusedDot}</td>
            <td style="${gradBg(r.auto,'auto')}color:#f59e0b;">${r.auto.toFixed(1)}</td>
            <td style="${gradBg(r.teleop,'teleop')}color:#3b82f6;">${r.teleop.toFixed(1)}</td>
            <td style="${gradBg(r.endgame,'endgame')}color:#10b981;">${r.endgame.toFixed(1)}</td>
            <td style="color:#64748b;">${r.matches}</td>
        </tr>`;
    }).join('');
};

// ─── SCOUTING SUB-TABS ──────────────────────────────────────────────────────

let currentScoutingSubTab = 'teams';
window.switchScoutingSubTab = function (tab) {
    currentScoutingSubTab = tab;
    ['teams', 'curation', 'notes'].forEach(t => {
        document.getElementById(`scouting-subtab-${t}`).style.display = t === tab ? 'block' : 'none';
    });
    document.querySelectorAll('#scoutingSubTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['teams', 'curation', 'notes'][i] === tab);
    });
    if (tab === 'curation') renderCurationTab();
    if (tab === 'notes') renderNotesTab();
};

async function renderCurationTab() {
    const container = document.getElementById('scouting-curation-content');
    if (!container) return;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const rawStr    = localStorage.getItem(`scoutingData_${eventKey}`);
    const pitRawStr = localStorage.getItem(`pitData_${eventKey}`);
    if (!rawStr && !pitRawStr) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:20px;">No scouting data loaded.</p>';
        return;
    }

    container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:20px;">Computing…</p>';

    let config = null, byTeam = {}, observations = [];
    let dedupedByTeam = {}, scoutIndex = {};
    let reportingMode = 'unknown', isCumulative = true;

    if (rawStr) {
        const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
        if (!processed) { container.innerHTML = '<p style="color:#64748b;font-style:italic;">No game config for this event.</p>'; return; }
        ({ config, byTeam, observations } = processed);
        for (const [tn, rows] of Object.entries(byTeam)) {
            dedupedByTeam[tn] = deduplicateTeamRows(rows).rows;
        }
        for (const [tn, rows] of Object.entries(dedupedByTeam)) {
            for (const r of rows) {
                if (!scoutIndex[r.matchNumber]) scoutIndex[r.matchNumber] = {};
                scoutIndex[r.matchNumber][tn] = r;
            }
        }
    }

    const tbaMatches    = await db.matches.where('eventKey').equals(eventKey).toArray();
    const hasBreakdowns = tbaMatches.some(m => m.redBreakdown);

    if (rawStr && config) {
        reportingMode = hasBreakdowns
            ? detectCumulativeReportingMode(tbaMatches, config.teleopFuseStats ?? [])
            : 'unknown';
        isCumulative = reportingMode !== 'separate';
    }

    let html = '';
    if (!rawStr) {
        html += '<p style="color:#64748b;font-style:italic;font-size:0.85em;margin:0 0 16px;">No match scouting data loaded — showing pit scouting only.</p>';
    }

    // ── 1. MATCH COVERAGE ───────────────────────────────────────────────────
    const allMatchNums = tbaMatches.map(m => m.matchNumber).sort((a, b) => a - b);
    const matchCoverage = allMatchNums.map(mn => {
        const m = tbaMatches.find(x => x.matchNumber === mn);
        const sc = scoutIndex[mn] || {};
        const redScouted  = (m?.red  || []).filter(t => sc[t]).length;
        const blueScouted = (m?.blue || []).filter(t => sc[t]).length;
        return { mn, m, sc, redScouted, blueScouted,
            redTotal: (m?.red || []).length, blueTotal: (m?.blue || []).length };
    });

    const fullyUnscouted = matchCoverage.filter(c => c.redScouted === 0 && c.blueScouted === 0);
    const partial        = matchCoverage.filter(c => (c.redScouted > 0 || c.blueScouted > 0) &&
                                                      (c.redScouted < c.redTotal || c.blueScouted < c.blueTotal));
    const fullyCovered   = matchCoverage.filter(c => c.redScouted === c.redTotal && c.blueScouted === c.blueTotal);

    const summaryStyle = (color) => `cursor:pointer;list-style:none;display:flex;align-items:center;gap:0;margin-bottom:0;padding:6px 0;`;
    const hdrStyle = (color) => `font-size:0.7em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;border-left:3px solid ${color};padding-left:10px;color:#94a3b8;flex:1;`;
    const chevron = `<span class="curation-chevron" style="color:#475569;font-size:0.9em;margin-left:8px;transition:transform 0.15s;">▼</span>`;

    if (rawStr) html += `
    <details style="margin-bottom:20px;">
        <summary style="${summaryStyle('#64748b')}">
            <span style="${hdrStyle('#64748b')}">Match Coverage</span>
            <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:#4ade80;">${fullyCovered.length}</span> full · <span style="color:#f59e0b;">${partial.length}</span> partial · <span style="color:#ef4444;">${fullyUnscouted.length}</span> unscouted</span>
            ${chevron}
        </summary>
        <div style="margin-top:12px;">
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
            <div style="background:#0f2010;border:1px solid #166534;border-radius:6px;padding:10px 16px;text-align:center;">
                <div style="color:#4ade80;font-size:1.4em;font-weight:700;">${fullyCovered.length}</div>
                <div style="color:#64748b;font-size:0.72em;">Fully scouted</div>
            </div>
            <div style="background:#1a1500;border:1px solid #854d0e;border-radius:6px;padding:10px 16px;text-align:center;">
                <div style="color:#f59e0b;font-size:1.4em;font-weight:700;">${partial.length}</div>
                <div style="color:#64748b;font-size:0.72em;">Partially scouted</div>
            </div>
            <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:6px;padding:10px 16px;text-align:center;">
                <div style="color:#ef4444;font-size:1.4em;font-weight:700;">${fullyUnscouted.length}</div>
                <div style="color:#64748b;font-size:0.72em;">Unscouted</div>
            </div>
        </div>
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
            <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                <th style="text-align:left;padding:4px 8px;">Match</th>
                <th style="text-align:center;padding:4px 8px;">Red scouted</th>
                <th style="text-align:center;padding:4px 8px;">Blue scouted</th>
                <th style="text-align:left;padding:4px 8px;">Missing teams</th>
            </tr></thead>
            <tbody>
            ${matchCoverage.map(({ mn, m, sc, redScouted, blueScouted, redTotal, blueTotal }) => {
                const full = redScouted === redTotal && blueScouted === blueTotal;
                const none = redScouted === 0 && blueScouted === 0;
                const missing = [...(m?.red || []).filter(t => !sc[t]), ...(m?.blue || []).filter(t => !sc[t])];
                return `<tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:4px 8px;color:#60a5fa;">QM ${mn}</td>
                    <td style="text-align:center;padding:4px 8px;color:${redScouted === redTotal ? '#4ade80' : '#f59e0b'};">${redScouted}/${redTotal}</td>
                    <td style="text-align:center;padding:4px 8px;color:${blueScouted === blueTotal ? '#4ade80' : '#f59e0b'};">${blueScouted}/${blueTotal}</td>
                    <td style="padding:4px 8px;color:#94a3b8;">${missing.length ? missing.join(', ') : '—'}</td>
                </tr>`;
            }).join('')}
            </tbody>
        </table>
        </div>
        </div>
    </details>`;

    // ── 2. PIT SCOUTING COVERAGE ─────────────────────────────────────────────
    {
        // Build full team list: TBA match alliances → scouting byTeam → db.teams (synced team list)
        const eventTeams = new Set();
        for (const m of tbaMatches) {
            for (const t of [...(m.red || []), ...(m.blue || [])]) eventTeams.add(t);
        }
        if (!eventTeams.size) {
            for (const tn of Object.keys(byTeam)) eventTeams.add(tn);
        }
        if (!eventTeams.size) {
            const dbTeams = await db.teams.where('eventKey').equals(eventKey).toArray();
            for (const t of dbTeams) eventTeams.add(String(t.teamNumber));
        }

        if (!pitRawStr) {
            html += `
            <details style="margin-bottom:20px;">
                <summary style="${summaryStyle('#64748b')}">
                    <span style="${hdrStyle('#64748b')}">Pit Scouting Coverage</span>
                    <span style="font-size:0.75em;color:#475569;margin-right:8px;">No pit data loaded</span>
                    ${chevron}
                </summary>
                <div style="margin-top:12px;">
                    <p style="color:#475569;font-size:0.85em;font-style:italic;">Sync a pit scouting sheet from the Home tab to see coverage.</p>
                </div>
            </details>`;
        } else {
            const pitRows = JSON.parse(pitRawStr);

            // Find which event teams have a pit row (value-match heuristic)
            const pittedTeams = new Set();
            for (const row of pitRows) {
                for (const v of Object.values(row)) {
                    const trimmed = String(v).trim();
                    if (eventTeams.has(trimmed)) { pittedTeams.add(trimmed); break; }
                }
            }

            const sort = arr => [...arr].sort((a, b) => parseInt(a) - parseInt(b));
            const missing  = sort([...eventTeams].filter(t => !pittedTeams.has(t)));
            const scouted  = sort([...eventTeams].filter(t => pittedTeams.has(t)));

            const chip = (tn, color) =>
                `<span onclick="viewTeamDetail(${tn}, 'pit-data')" style="cursor:pointer;display:inline-block;padding:2px 8px;border-radius:12px;background:${color}22;border:1px solid ${color}55;color:${color};font-size:0.78em;font-weight:600;margin:2px;">${tn}</span>`;

            html += `
            <details style="margin-bottom:20px;">
                <summary style="${summaryStyle('#64748b')}">
                    <span style="${hdrStyle('#64748b')}">Pit Scouting Coverage</span>
                    <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:#4ade80;">${scouted.length}</span> scouted · <span style="color:#ef4444;">${missing.length}</span> missing</span>
                    ${chevron}
                </summary>
                <div style="margin-top:12px;">
                    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;">
                        <div style="background:#0f2010;border:1px solid #166534;border-radius:6px;padding:10px 16px;text-align:center;">
                            <div style="color:#4ade80;font-size:1.4em;font-weight:700;">${scouted.length}</div>
                            <div style="color:#64748b;font-size:0.72em;">Pit scouted</div>
                        </div>
                        <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:6px;padding:10px 16px;text-align:center;">
                            <div style="color:#ef4444;font-size:1.4em;font-weight:700;">${missing.length}</div>
                            <div style="color:#64748b;font-size:0.72em;">Not scouted</div>
                        </div>
                        <div style="background:#0c1220;border:1px solid #1e3a5f;border-radius:6px;padding:10px 16px;text-align:center;">
                            <div style="color:#60a5fa;font-size:1.4em;font-weight:700;">${pitRows.length}</div>
                            <div style="color:#64748b;font-size:0.72em;">Pit entries total</div>
                        </div>
                    </div>
                    ${missing.length ? `
                    <div style="margin-bottom:12px;">
                        <div style="color:#ef4444;font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Not yet scouted</div>
                        <div>${missing.map(t => chip(t, '#ef4444')).join('')}</div>
                    </div>` : `<p style="color:#4ade80;font-size:0.85em;margin:0 0 12px;">All ${eventTeams.size} teams have been pit scouted.</p>`}
                    ${scouted.length ? `
                    <div>
                        <div style="color:#4ade80;font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Scouted</div>
                        <div>${scouted.map(t => chip(t, '#4ade80')).join('')}</div>
                    </div>` : ''}
                </div>
            </details>`;
        }
    }

    // ── 3. FOUL POINTS ──────────────────────────────────────────────────────
    if (hasBreakdowns) {
        const foulRows = [];
        for (const m of tbaMatches) {
            for (const [alliance, breakdown, oppBreakdown] of [
                ['red',  m.redBreakdown,  m.blueBreakdown],
                ['blue', m.blueBreakdown, m.redBreakdown],
            ]) {
                if (!breakdown || !oppBreakdown) continue;
                // foulPoints on the breakdown = pts awarded TO this alliance FROM opponent fouls
                const foulPts = breakdown.foulPoints ?? 0;
                foulRows.push({ mn: m.matchNumber, alliance, foulPts, teams: m[alliance] || [] });
            }
        }
        const avgFoul = foulRows.length ? foulRows.reduce((s, r) => s + r.foulPts, 0) / foulRows.length : 0;
        const perRobot = (avgFoul / 3).toFixed(1);

        html += `
        <details style="margin-bottom:20px;">
            <summary style="${summaryStyle('#a78bfa')}">
                <span style="${hdrStyle('#a78bfa')}">Foul Points Received</span>
                <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:#f8fafc;">${avgFoul.toFixed(1)}</span> pts/alliance · <span style="color:#f8fafc;">${perRobot}</span> pts/robot</span>
                ${chevron}
            </summary>
            <div style="margin-top:12px;">
            <p style="color:#94a3b8;font-size:0.82em;margin:0 0 10px;">Average <strong style="color:#f8fafc;">${avgFoul.toFixed(1)} pts/alliance/match</strong> received from opponent fouls — roughly <strong style="color:#f8fafc;">${perRobot} pts/robot/match</strong> not captured by scouting.</p>
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:4px 8px;">Match</th>
                    <th style="padding:4px 8px;">Alliance</th>
                    <th style="text-align:right;padding:4px 8px;">Foul pts received</th>
                    <th style="text-align:right;padding:4px 8px;">Per robot</th>
                </tr></thead>
                <tbody>
                ${foulRows.map(r => `
                    <tr style="border-bottom:1px solid #1e293b;">
                        <td style="padding:4px 8px;color:#60a5fa;">QM ${r.mn}</td>
                        <td style="padding:4px 8px;color:${r.alliance==='red'?'#f87171':'#60a5fa'};">${r.alliance}</td>
                        <td style="text-align:right;padding:4px 8px;color:${r.foulPts>10?'#f59e0b':'#94a3b8'};">${r.foulPts}</td>
                        <td style="text-align:right;padding:4px 8px;color:#64748b;">${(r.foulPts/3).toFixed(1)}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
            </div>
            </div>
        </details>`;
    }

    // ── 4. OUTLIER MATCHES ──────────────────────────────────────────────────
    if (rawStr) {
        const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();

        const getMatchEPA = (tn, row) => {
            const matchFused = fusedCache?.teams?.[tn]?.fusedByMatch?.[String(row.matchNumber)];
            if (matchFused && config.computeFusedEPABreakdown) {
                return config.computeFusedEPABreakdown(matchFused).total;
            }
            return config.computeMatchEPA ? config.computeMatchEPA(row) : 0;
        };

        const outliers = [];
        for (const [tn, rows] of Object.entries(dedupedByTeam)) {
            const played = rows.filter(r => !r.noShow);
            if (played.length < 2) continue;
            const epas = played.map(r => ({ r, epa: getMatchEPA(tn, r) }));
            const mean = epas.reduce((s, x) => s + x.epa, 0) / epas.length;
            const std  = Math.sqrt(epas.reduce((s, x) => s + (x.epa - mean) ** 2, 0) / epas.length);
            for (const { r, epa } of epas) {
                const z = std > 0 ? (epa - mean) / std : 0;
                if (Math.abs(z) >= 1.5) outliers.push({ tn, mn: r.matchNumber, epa, mean, z });
            }
        }
        outliers.sort((a, b) => Math.abs(b.z) - Math.abs(a.z));

        html += `
        <details style="margin-bottom:20px;">
            <summary style="${summaryStyle('#f59e0b')}">
                <span style="${hdrStyle('#f59e0b')}">Match Outliers <span style="color:#475569;font-weight:400;font-size:0.9em;">(≥1.5σ from team mean)</span></span>
                <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:${outliers.length>0?'#f59e0b':'#4ade80'};">${outliers.length}</span> found</span>
                ${chevron}
            </summary>
            <div style="margin-top:12px;">
            ${outliers.length === 0 ? '<p style="color:#64748b;font-style:italic;font-size:0.85em;">No significant outliers found.</p>' : `
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:4px 8px;">Team</th>
                    <th style="text-align:left;padding:4px 8px;">Match</th>
                    <th style="text-align:right;padding:4px 8px;">Match EPA</th>
                    <th style="text-align:right;padding:4px 8px;">Team avg</th>
                    <th style="text-align:right;padding:4px 8px;">Deviation</th>
                </tr></thead>
                <tbody>
                ${outliers.slice(0, 30).map(o => {
                    const dir = o.z > 0 ? '+' : '';
                    const c   = o.z > 0 ? '#4ade80' : '#ef4444';
                    return `<tr style="border-bottom:1px solid #1e293b;cursor:pointer;" onclick="viewTeamDetail(${o.tn}, 'scouting')">
                        <td style="padding:4px 8px;color:#f8fafc;font-weight:600;white-space:nowrap;">${o.tn}</td>
                        <td style="padding:4px 8px;color:#60a5fa;white-space:nowrap;">QM ${o.mn}</td>
                        <td style="text-align:right;padding:4px 8px;white-space:nowrap;">${o.epa.toFixed(1)}</td>
                        <td style="text-align:right;padding:4px 8px;color:#64748b;white-space:nowrap;">${o.mean.toFixed(1)}</td>
                        <td style="text-align:right;padding:4px 8px;white-space:nowrap;color:${c};">${dir}${o.z.toFixed(2)}σ</td>
                    </tr>`;
                }).join('')}
                </tbody>
            </table>
            </div>`}
            </div>
        </details>`;
    }

    // ── 5. EPA COMPARISON ───────────────────────────────────────────────────
    if (rawStr) {
        const statboticsTeams = await db.teams.toArray();
        const statByTeam = Object.fromEntries(statboticsTeams.map(t => [String(t.teamNumber), t]));
        const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
        const compRows = Object.entries(dedupedByTeam).map(([tn, rows]) => {
            const rawStats = config.aggregateTeam(rows);
            const fusedResult = fusedCache?.teams?.[tn];
            const breakdown = (fusedResult?.available && config.computeFusedEPABreakdown)
                ? config.computeFusedEPABreakdown(fusedResult.stats)
                : config.computeEPABreakdown(rawStats);
            const statTeam = statByTeam[tn];
            const statEPA = statTeam?.currentEPA ?? null;
            return { tn, scoutEPA: breakdown.total, statEPA, diff: statEPA != null ? breakdown.total - statEPA : null, matches: rawStats.matches, fused: !!fusedResult?.available };
        }).filter(r => r.statEPA != null).sort((a, b) => b.statEPA - a.statEPA);

        const avgGap = compRows.length ? (compRows.reduce((s,r) => s + r.diff, 0) / compRows.length) : 0;
        html += `
        <details style="margin-bottom:20px;">
            <summary style="${summaryStyle('#10b981')}">
                <span style="${hdrStyle('#10b981')}">EPA Comparison <span style="color:#475569;font-weight:400;font-size:0.9em;">(scouting vs Statbotics)</span></span>
                <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:${avgGap < -5 ? '#ef4444' : avgGap > 5 ? '#f59e0b' : '#4ade80'};">${avgGap>=0?'+':''}${avgGap.toFixed(1)}</span> avg gap · ${compRows.length} teams</span>
                ${chevron}
            </summary>
            <div style="margin-top:12px;">
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:4px 8px;">Team</th>
                    <th style="text-align:right;padding:4px 8px;">Scout EPA</th>
                    <th style="text-align:right;padding:4px 8px;">Statbotics EPA</th>
                    <th style="text-align:right;padding:4px 8px;">Gap</th>
                    <th style="text-align:right;padding:4px 8px;">Matches</th>
                </tr></thead>
                <tbody>
                ${compRows.map(r => {
                    const gapColor = r.diff > 5 ? '#f59e0b' : r.diff < -5 ? '#ef4444' : '#64748b';
                    const dot = r.fused ? `<span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#4ade80;margin-left:4px;vertical-align:middle;"></span>` : '';
                    return `<tr style="border-bottom:1px solid #1e293b;cursor:pointer;" onclick="viewTeamDetail(${r.tn}, 'scouting')">
                        <td style="padding:4px 8px;font-weight:600;">${r.tn}</td>
                        <td style="text-align:right;padding:4px 8px;">${r.scoutEPA.toFixed(1)}${dot}</td>
                        <td style="text-align:right;padding:4px 8px;color:#64748b;">${r.statEPA.toFixed(1)}</td>
                        <td style="text-align:right;padding:4px 8px;color:${gapColor};">${r.diff>=0?'+':''}${r.diff.toFixed(1)}</td>
                        <td style="text-align:right;padding:4px 8px;color:#475569;">${r.matches}</td>
                    </tr>`;
                }).join('')}
                </tbody>
            </table>
            </div>
            </div>
        </details>`;
    }

    // ── 6. UNKNOWN TEAM NUMBERS ──────────────────────────────────────────────
    if (rawStr) {
        const knownTeams = new Set();
        for (const m of tbaMatches) {
            for (const t of [...(m.red || []), ...(m.blue || [])]) knownTeams.add(t);
        }

        if (knownTeams.size > 0) {
            // Collect all distinct (teamNumber, matchNumber) pairs not in the event roster
            const unknownMap = {}; // teamNum → Set of matchNumbers
            for (const obs of observations) {
                const tn = String(obs.teamNumber);
                if (!tn || tn === '0') continue;
                if (!knownTeams.has(tn)) {
                    if (!unknownMap[tn]) unknownMap[tn] = new Set();
                    unknownMap[tn].add(obs.matchNumber);
                }
            }

            // Per-match index of scouted team numbers (for finding unscouted alliance partners)
            const matchScoutedTeams = {};
            for (const obs of observations) {
                const tn = String(obs.teamNumber);
                if (!tn || tn === '0') continue;
                if (!matchScoutedTeams[obs.matchNumber]) matchScoutedTeams[obs.matchNumber] = new Set();
                matchScoutedTeams[obs.matchNumber].add(tn);
            }
            const matchIndex = Object.fromEntries(tbaMatches.map(m => [m.matchNumber, m]));

            const unknownEntries = Object.entries(unknownMap)
                .sort(([a], [b]) => Number(a) - Number(b));

            html += `
            <details style="margin-bottom:20px;">
                <summary style="${summaryStyle('#f87171')}">
                    <span style="${hdrStyle('#f87171')}">Unknown Team Numbers <span style="color:#475569;font-weight:400;font-size:0.9em;">(not in TBA event roster)</span></span>
                    <span style="font-size:0.75em;color:#64748b;margin-right:8px;"><span style="color:${unknownEntries.length > 0 ? '#ef4444' : '#4ade80'};">${unknownEntries.length}</span> found</span>
                    ${chevron}
                </summary>
                <div style="margin-top:12px;">
                ${unknownEntries.length === 0
                    ? '<p style="color:#4ade80;font-size:0.85em;margin:0;">All scouted team numbers match the TBA event roster.</p>'
                    : `<p style="color:#94a3b8;font-size:0.82em;margin:0 0 10px;">These team numbers appear in scouting data but not in any TBA match alliance. Likely data entry errors — check the matches listed and correct the team number in the sheet.</p>
                    <div style="overflow-x:auto;">
                    <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
                        <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                            <th style="text-align:left;padding:4px 8px;">Scouted #</th>
                            <th style="text-align:right;padding:4px 8px;">Rows</th>
                            <th style="text-align:left;padding:4px 8px;">Matches</th>
                            <th style="text-align:left;padding:4px 8px;">Unscouted in those matches</th>
                        </tr></thead>
                        <tbody>
                        ${unknownEntries.map(([tn, mnSet]) => {
                            const sortedMns = [...mnSet].sort((a, b) => a - b);
                            const matches = sortedMns.map(mn => `QM ${mn}`).join(', ');

                            // Teams in those TBA matches that have no scouting row
                            const unscouted = new Set();
                            for (const mn of sortedMns) {
                                const tbaMatch = matchIndex[mn];
                                if (!tbaMatch) continue;
                                const scouted = matchScoutedTeams[mn] || new Set();
                                for (const t of [...(tbaMatch.red || []), ...(tbaMatch.blue || [])]) {
                                    if (!scouted.has(t)) unscouted.add(t);
                                }
                            }
                            const unscoutedStr = unscouted.size > 0
                                ? [...unscouted].sort((a, b) => Number(a) - Number(b)).join(', ')
                                : '<span style="color:#475569;">—</span>';

                            return `<tr style="border-bottom:1px solid #1e293b;">
                                <td style="padding:4px 8px;color:#f87171;font-weight:600;">${tn}</td>
                                <td style="text-align:right;padding:4px 8px;color:#94a3b8;">${mnSet.size}</td>
                                <td style="padding:4px 8px;color:#94a3b8;">${matches}</td>
                                <td style="padding:4px 8px;color:#fbbf24;font-weight:600;">${unscoutedStr}</td>
                            </tr>`;
                        }).join('')}
                        </tbody>
                    </table>
                    </div>`}
                </div>
            </details>`;
        }
    }

    // ── 7. Game-specific curation section ────────────────────────────────────
    if (config?.curationSection) {
        html += config.curationSection(tbaMatches, matchCoverage, scoutIndex, isCumulative, reportingMode, { summaryStyle, hdrStyle, chevron });
    }

    container.innerHTML = html;
}

// ─── NOTES TAB ────────────────────────────────────────────────────────────────

function renderNotesTab() {
    const container = document.getElementById('scouting-notes-content');
    if (!container) return;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const showAll  = container.dataset.mode !== 'user';
    const query    = (container.dataset.search || '').trim().toLowerCase();

    const hlText = (rawText, q) => {
        const escaped = String(rawText).replace(/&/g,'&amp;').replace(/</g,'&lt;');
        if (!q) return escaped;
        const escapedQ = q.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        return escaped.replace(new RegExp(`(${escapedQ})`, 'gi'), '<mark style="background:#854d0e;color:#fef08a;border-radius:2px;padding:0 1px;">$1</mark>');
    };

    // User-created team notes
    const userNotes = [];
    for (const [teamNum, map] of Object.entries(getTeamNotes())) {
        for (const note of Object.values(map)) {
            if (note.text) userNotes.push({ source: 'user', teamNumber: teamNum, matchNumber: note.qm, text: note.text });
        }
    }

    // Standalone event notes (no team)
    const eventNotes = _getEventNotes(eventKey).map(n => ({ source: 'event', teamNumber: null, matchNumber: null, ...n }));

    // Scouting sheet comments (all-mode only)
    const scoutNotes = [];
    if (showAll && eventKey) {
        const raw = localStorage.getItem(`scoutingData_${eventKey}`);
        if (raw) {
            const result = processScoutingData(eventKey, JSON.parse(raw), getScoutingColumnOverrides(eventKey));
            if (result?.byTeam) {
                for (const [tn, rows] of Object.entries(result.byTeam)) {
                    for (const r of rows) {
                        if (r.comments) scoutNotes.push({ source: 'scouting', teamNumber: tn, matchNumber: r.matchNumber, text: r.comments });
                    }
                }
            }
        }
    }

    const totalUser = userNotes.length + eventNotes.length;
    const totalAll  = totalUser + scoutNotes.length;

    // Filter by search query
    const matchNote = (n) => {
        if (!query) return true;
        if (n.text?.toLowerCase().includes(query)) return true;
        if (String(n.teamNumber ?? '').includes(query)) return true;
        return false;
    };
    const filteredUser   = userNotes.filter(matchNote);
    const filteredEvent  = eventNotes.filter(matchNote);
    const filteredScout  = scoutNotes.filter(matchNote);
    const filteredTotal  = filteredUser.length + filteredEvent.length + filteredScout.length;

    // Group all notes by team; '__none__' = no team
    const teamMap = new Map();
    const addNote = (n) => {
        const key = n.teamNumber != null ? String(n.teamNumber) : '__none__';
        if (!teamMap.has(key)) teamMap.set(key, []);
        teamMap.get(key).push(n);
    };
    filteredUser.forEach(addNote);
    filteredScout.forEach(addNote);   // merged under same team in all-mode
    filteredEvent.forEach(addNote);   // goes to '__none__'

    // Within each team: user notes first, then sheet, sorted by match number
    for (const notes of teamMap.values()) {
        notes.sort((a, b) => {
            if (a.source !== b.source) return a.source === 'scouting' ? 1 : -1;
            return (a.matchNumber ?? Infinity) - (b.matchNumber ?? Infinity);
        });
    }

    // Teams sorted numerically; general notes last
    const sortedKeys = [...teamMap.keys()].sort((a, b) => {
        if (a === '__none__') return 1;
        if (b === '__none__') return -1;
        return Number(a) - Number(b);
    });

    const modeBtn = (mode, label, count) => {
        const active = showAll === (mode === 'all');
        return `<button onclick="setNotesMode('${mode}')" style="padding:6px 14px;border-radius:6px;border:1px solid ${active ? '#3b82f6' : '#334155'};background:${active ? '#1e3a5f' : 'transparent'};color:${active ? '#60a5fa' : '#64748b'};font-size:0.82em;cursor:pointer;">${label} <span style="color:${active ? '#93c5fd' : '#475569'};">(${count})</span></button>`;
    };

    // Row inside a team section — team already shown in header, so just match context
    const noteRow = (n, i) => {
        const matchLabel = n.matchNumber != null ? `QM ${n.matchNumber}` : (n.source !== 'scouting' ? 'General' : '');
        const accent   = n.source === 'scouting' ? '#a78bfa' : '#60a5fa';
        const teamArg  = n.teamNumber  != null ? `'${n.teamNumber}'`  : 'null';
        const qmArg    = n.matchNumber != null ? n.matchNumber        : 'null';
        const idArg    = n.id ? `'${n.id}'` : "''";
        const editable = n.source !== 'scouting';
        return `<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:10px 14px;${i > 0 ? 'border-top:1px solid #0f172a;' : ''}">
            <div style="min-width:0;flex:1;">
                ${matchLabel ? `<span style="color:${accent};font-size:0.76em;font-weight:700;margin-right:8px;">${matchLabel}</span>` : ''}
                <span style="color:#cbd5e1;font-size:0.88em;line-height:1.55;white-space:pre-wrap;word-break:break-word;">${hlText(n.text, query)}</span>
            </div>
            <div style="display:flex;gap:5px;flex-shrink:0;align-items:center;padding-top:1px;">
                ${n.source === 'scouting' ? `<span style="color:#475569;font-size:0.72em;background:#0f172a;border-radius:4px;padding:2px 6px;">Sheet</span>` : ''}
                ${editable
                    ? `<button onclick="editNoteRow('${n.source}',${teamArg},${qmArg},${idArg})" style="background:#334155;color:#f8fafc;border:none;border-radius:6px;padding:3px 9px;font-size:0.76em;cursor:pointer;">Edit</button>
                       <button onclick="deleteNoteRow('${n.source}',${teamArg},${qmArg},${idArg})" style="background:#7f1d1d;color:#fca5a5;border:none;border-radius:6px;padding:3px 9px;font-size:0.76em;cursor:pointer;">Del</button>`
                    : ''}
            </div>
        </div>`;
    };

    const teamSection = (key, notes) => {
        const isNone     = key === '__none__';
        const label      = isNone ? 'General Notes' : `Team ${key}`;
        const userCount  = notes.filter(n => n.source !== 'scouting').length;
        const sheetCount = notes.filter(n => n.source === 'scouting').length;
        const countStr   = sheetCount > 0 ? `${userCount} user · ${sheetCount} sheet` : `${notes.length} note${notes.length !== 1 ? 's' : ''}`;
        return `<details class="notes-team-section" open style="background:#1e293b;border-radius:8px;margin-bottom:10px;border:1px solid #334155;overflow:hidden;">
            <summary style="display:flex;align-items:center;justify-content:space-between;padding:9px 14px;background:#162032;border-bottom:1px solid #334155;user-select:none;">
                <span style="color:#f1f5f9;font-weight:700;font-size:0.9em;">${label}</span>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#475569;font-size:0.76em;">${countStr}</span>
                    <span class="notes-chevron" style="color:#475569;font-size:0.8em;">▾</span>
                </div>
            </summary>
            ${notes.map((n, i) => noteRow(n, i)).join('')}
        </details>`;
    };

    const emptyState = query
        ? `<div style="background:#1e293b;padding:20px 16px;border-radius:8px;border:1px dashed #334155;text-align:center;color:#475569;font-size:0.88em;">No notes match <strong style="color:#94a3b8;">"${query.replace(/&/g,'&amp;')}"</strong>.</div>`
        : `<div style="background:#1e293b;padding:20px 16px;border-radius:8px;border:1px dashed #334155;text-align:center;color:#475569;font-size:0.88em;">No notes yet. Use <strong style="color:#94a3b8;">+ Add Note</strong> to create one.</div>`;

    const bulkBtns = sortedKeys.length > 1
        ? `<div style="display:flex;gap:6px;">
               <button onclick="expandAllNotes()" style="background:transparent;color:#64748b;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:0.76em;cursor:pointer;">Expand all</button>
               <button onclick="collapseAllNotes()" style="background:transparent;color:#64748b;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:0.76em;cursor:pointer;">Collapse all</button>
           </div>`
        : '';

    const resultHint = query
        ? `<span style="color:#64748b;font-size:0.82em;white-space:nowrap;">${filteredTotal} result${filteredTotal !== 1 ? 's' : ''}</span>`
        : '';

    container.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
            <button onclick="showNoteEditorModal()" style="background:#1e3a5f;color:#60a5fa;border:1px solid #3b82f6;border-radius:6px;padding:8px 16px;font-weight:700;cursor:pointer;white-space:nowrap;">+ Add Note</button>
            <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                ${modeBtn('all',  'All Notes',  totalAll)}
                ${modeBtn('user', 'User Notes', totalUser)}
                ${bulkBtns}
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
            <input id="notes-search-input" type="search" value="${query.replace(/&/g,'&amp;')}" placeholder="Search notes…"
                oninput="filterNotes(this.value)"
                style="flex:1;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px 12px;font-size:0.88em;outline:none;box-sizing:border-box;">
            ${resultHint}
        </div>
        ${sortedKeys.length ? sortedKeys.map(k => teamSection(k, teamMap.get(k))).join('') : emptyState}`;

    if (query) {
        const si = container.querySelector('#notes-search-input');
        if (si) { si.focus(); si.setSelectionRange(si.value.length, si.value.length); }
    }
}

window.collapseAllNotes = function () {
    document.querySelectorAll('#scouting-notes-content .notes-team-section').forEach(d => d.removeAttribute('open'));
};
window.expandAllNotes = function () {
    document.querySelectorAll('#scouting-notes-content .notes-team-section').forEach(d => d.setAttribute('open', ''));
};

window.setNotesMode = function (mode) {
    const c = document.getElementById('scouting-notes-content');
    if (c) c.dataset.mode = mode;
    renderNotesTab();
};

window.filterNotes = function (value) {
    const c = document.getElementById('scouting-notes-content');
    if (c) { c.dataset.search = value; renderNotesTab(); }
};

window.showNoteEditorModal = function (opts = {}) {
    document.getElementById('note-editor-overlay')?.remove();
    const { teamNumber = '', matchNumber = '', text = '', source = 'user', id = null,
            _origTeam = null, _origMatch = undefined } = opts;
    const overlay = document.createElement('div');
    overlay.id = 'note-editor-overlay';
    overlay.dataset.origSource = source;
    overlay.dataset.origTeam   = _origTeam  != null ? String(_origTeam)  : '';
    overlay.dataset.origMatch  = _origMatch != null ? String(_origMatch) : '';
    overlay.dataset.origId     = id || '';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:9999;display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;';
    overlay.innerHTML = `
        <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:24px;width:100%;max-width:480px;box-sizing:border-box;">
            <h3 style="color:#f8fafc;margin:0 0 16px;font-size:1.05rem;font-weight:700;">${id ? 'Edit Note' : 'New Note'}</h3>
            <textarea id="note-modal-text" placeholder="Note text…" style="width:100%;min-height:90px;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:10px;font-size:0.9rem;resize:vertical;box-sizing:border-box;font-family:inherit;">${String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;')}</textarea>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">
                <label style="display:flex;flex-direction:column;gap:4px;">
                    <span style="color:#64748b;font-size:0.78em;">Team # <span style="color:#475569;">(optional)</span></span>
                    <input id="note-modal-team" type="text" value="${teamNumber}" placeholder="e.g. 1768" style="background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px;font-size:0.9rem;box-sizing:border-box;">
                </label>
                <label style="display:flex;flex-direction:column;gap:4px;">
                    <span style="color:#64748b;font-size:0.78em;">Match # <span style="color:#475569;">(optional)</span></span>
                    <input id="note-modal-match" type="number" value="${matchNumber}" placeholder="e.g. 5" min="1" style="background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px;font-size:0.9rem;box-sizing:border-box;">
                </label>
            </div>
            <p style="color:#475569;font-size:0.76em;margin:10px 0 0;">Notes tied to a team appear on that team’s detail page.</p>
            <div style="display:flex;gap:8px;margin-top:16px;justify-content:flex-end;">
                <button onclick="document.getElementById('note-editor-overlay').remove()" style="background:#334155;color:#f8fafc;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:0.88rem;">Cancel</button>
                <button onclick="_saveNoteFromModal()" style="background:#3b82f6;color:#f8fafc;border:none;border-radius:6px;padding:8px 18px;font-weight:700;cursor:pointer;font-size:0.88rem;">Save Note</button>
            </div>
        </div>`;
    document.body.appendChild(overlay);
    document.getElementById('note-modal-text')?.focus();
};

window._saveNoteFromModal = function () {
    const overlay = document.getElementById('note-editor-overlay');
    if (!overlay) return;
    const origSource = overlay.dataset.origSource;
    const origTeam   = overlay.dataset.origTeam;
    const origMatch  = overlay.dataset.origMatch;
    const origId     = overlay.dataset.origId;

    const text = document.getElementById('note-modal-text')?.value.trim() || '';
    if (!text) { alert('Note text cannot be empty.'); return; }
    const teamNum  = document.getElementById('note-modal-team')?.value.trim() || '';
    const matchRaw = document.getElementById('note-modal-match')?.value.trim();
    const qm       = matchRaw ? (parseInt(matchRaw, 10) || null) : null;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();

    // Delete old note when editing (handles team/match context changes)
    if (origTeam && origSource === 'user') {
        saveTeamNote(origTeam, '', origMatch ? parseInt(origMatch, 10) : null);
    } else if (origId && origSource === 'event') {
        _deleteEventNote(origId, eventKey);
    }

    if (teamNum) {
        saveTeamNote(teamNum, text, qm);
    } else {
        _saveEventNote({ id: origId || String(Date.now()), text, timestamp: Date.now() }, eventKey);
    }

    overlay.remove();
    renderNotesTab();
};

window.editNoteRow = function (source, teamNumber, matchNumber, id) {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    let text = '';
    if (source === 'user') {
        text = getTeamNote(teamNumber, matchNumber)?.text || '';
    } else if (source === 'event') {
        text = _getEventNotes(eventKey).find(n => n.id === id)?.text || '';
    }
    window.showNoteEditorModal({
        teamNumber:  teamNumber  != null ? String(teamNumber)  : '',
        matchNumber: matchNumber != null ? String(matchNumber) : '',
        text, source, id: id || null,
        _origTeam:  teamNumber,
        _origMatch: matchNumber,
    });
};

window.deleteNoteRow = function (source, teamNumber, matchNumber, id) {
    if (!confirm('Delete this note?')) return;
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (source === 'user') {
        saveTeamNote(String(teamNumber), '', matchNumber != null ? matchNumber : null);
    } else if (source === 'event') {
        _deleteEventNote(id, eventKey);
    }
    renderNotesTab();
};

// ─── TOOLS TAB ──────────────────────────────────────────────────────────────

let currentToolsTab = 'picklist';
let pickListSortCol = 'composite';
let pickListSortDir = 1; // 1 = descending (default for all columns)

window.sortPickListBy = function (col) {
    if (pickListSortCol === col) {
        pickListSortDir *= -1;
    } else {
        pickListSortCol = col;
        pickListSortDir = 1;
    }
    renderPickList();
};

window.switchToolsTab = function (tab) {
    currentToolsTab = tab;
    // Order must match the buttons in #toolsTabs — the .active toggle below is by index.
    const allTabs = ['field', 'picklist', 'draft', 'alliances', 'tracks', 'dev'];
    allTabs.forEach(t => {
        document.getElementById(`tools-tab-${t}`).style.display = t === tab ? 'block' : 'none';
    });
    document.querySelectorAll('#toolsTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', allTabs[i] === tab);
    });
    if (tab === 'picklist') renderPickList();
    if (tab === 'draft') renderDraft();
    if (tab === 'field') initFieldTab();
    if (tab === 'alliances') renderAlliancesTab();
    if (tab === 'tracks') renderTracksTab();
    if (tab === 'dev') renderDevTab();
};

// ── Robot tracking work queue ────────────────────────────────────────────────
//
// One place to see what the tracker has done and what still needs a human, so nobody
// has to type a relay URL or a match key. Every action here is a deep link that carries
// the relay address and the match id, because the person doing the work is usually on a
// phone in a venue and typing a workers.dev URL on a phone is its own small punishment.
//
// State comes from two independent places and they mean different things:
//   public/tracks/index.json  what has been EXPORTED (finished, routes viewable)
//   the relay's /index        what is IN FLIGHT (bundle waiting, answer returned)
// A match can be in either, both, or neither.

const RELAY_KEY = 'rtrackRelay';

// Baked default last, matching the precedence in public/rtrack/*.html. main.js is
// bundled so it can read import.meta.env directly; the standalone curator pages cannot,
// and get the same value through the generated rtrack/relay.js.
function relayUrl() {
    return ((localStorage.getItem(RELAY_KEY)
             || import.meta.env.VITE_RTRACK_RELAY || '')).trim().replace(/\/+$/, '');
}

// The index is one small GET, but loadMatchTracks runs once per match and the team
// Routes tab loads a whole event at a time -- 25 matches would be 25 identical
// requests. Memoised for a few seconds, which is long enough to cover one render and
// short enough that a freshly published match shows up on the next interaction.
let _relayIdx = null, _relayIdxAt = 0;
const RELAY_IDX_TTL_MS = 8000;

async function relayTracksIndex() {
    const url = relayUrl();
    if (!url) return new Map();
    if (_relayIdx && Date.now() - _relayIdxAt < RELAY_IDX_TTL_MS) return _relayIdx;
    const items = await _relayIndex(url);
    const m = new Map();
    for (const it of items || []) {
        if (it.kind === 'tracks' && it.id) m.set(it.id, it.at || 0);
    }
    // A failed fetch caches an EMPTY map for the TTL rather than retrying per match.
    // Falling back to git on a flaky relay is correct; hammering it is not.
    _relayIdx = m; _relayIdxAt = Date.now();
    return m;
}

async function _relayIndex(url) {
    if (!url) return null;
    try {
        const r = await fetch(`${url.replace(/\/+$/, '')}/index`, { cache: 'no-store' });
        if (!r.ok) return null;
        const d = await r.json();
        return Array.isArray(d.items) ? d.items : null;
    } catch { return null; }
}

// ── Reviewed appearance gallery queue --------------------------------------
// Gallery review is deliberately a delta workflow.  The Tracks tab shows relay
// metadata first and fetches the bounded image bundle only when a person opens it.
let _galleryReviewBundle = null;
let _galleryReviewId = null;
let _galleryReviewQueue = [];

function galleryNextTarget(reviewId, team) {
    const current = _galleryReviewQueue.findIndex(item =>
        item.reviewId === reviewId && String(item.team) === String(team));
    if (current >= 0) return _galleryReviewQueue[current + 1] || null;
    return _galleryReviewQueue[0] || null;
}

function galleryEsc(value) {
    const el = document.createElement('span');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
}

function legacyGalleryReviewItems(items) {
    const out = new Map();
    for (const item of items || []) {
        if (!item?.id || !item.kind?.startsWith('gallery-')) continue;
        const row = out.get(item.id) || { id: item.id };
        row[item.kind] = item;
        out.set(item.id, row);
    }
    return [...out.values()].sort((a, b) => (b['gallery-bundle']?.at || 0)
        - (a['gallery-bundle']?.at || 0));
}

async function legacyLoadGalleryReviewBundle(relay, reviewId) {
    if (!relay || !reviewId) return null;
    try {
        const response = await fetch(`${relay}/gallery-bundle/${encodeURIComponent(reviewId)}`,
                                     { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const bundle = await response.json();
        if (bundle.kind !== 'galleryReviewBundle' || bundle.schemaVersion !== 1
            || !Array.isArray(bundle.candidates)) throw new Error('invalid gallery bundle');
        return bundle;
    } catch (error) {
        alert(`Could not load gallery review: ${error.message}`);
        return null;
    }
}

function galleryDraftKey(bundle) {
    return `galleryReview:${bundle.reviewId}:${bundle.bundleHash}`;
}

function legacyReadGalleryDraft(bundle) {
    try {
        const draft = JSON.parse(localStorage.getItem(galleryDraftKey(bundle)) || 'null');
        return draft?.bundleHash === bundle.bundleHash ? draft : { decisions: {}, teamStates: {} };
    } catch { return { decisions: {}, teamStates: {} }; }
}

function saveGalleryDraft(bundle, draft) {
    localStorage.setItem(galleryDraftKey(bundle), JSON.stringify({
        bundleHash: bundle.bundleHash, decisions: draft.decisions || {},
        selections: draft.selections || {}, touched: draft.touched || {},
        teamStates: draft.teamStates || {}, updatedAt: new Date().toISOString(),
    }));
}

function galleryDecisionFor(bundle, draft, candidate) {
    return draft.decisions?.[candidate.candidateId] || null;
}

function legacyRenderGalleryReviewPanel(host, relay, bundle) {
    const draft = readGalleryDraft(bundle);
    _galleryReviewBundle = bundle;
    const actionColor = { accept: '#22c55e', relabel: '#60a5fa', reject: '#ef4444',
                          mixed: '#f59e0b', split: '#c084fc', defer: '#64748b' };
    host.innerHTML = `
      <div style="border:1px solid #2563eb;border-radius:8px;padding:12px;margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;">
          <div><b>Gallery review</b> · ${galleryEsc(bundle.reviewId.slice(0, 12))}
            <div style="font-size:.76em;color:#64748b;margin-top:3px;">
              ${bundle.candidates.length} tracklets · ${bundle.candidates.reduce((n, c) => n + c.views.length, 0)} views ·
              base ${galleryEsc((bundle.galleryVersion || '').slice(0, 12))}</div></div>
          <button id="galleryReviewClose" style="padding:6px 10px;background:transparent;color:#94a3b8;border:1px solid #334155;border-radius:6px;cursor:pointer;">Close</button>
        </div>
        <p style="font-size:.8em;color:#94a3b8;line-height:1.5;">Review the source tracklet as a group. Accept only when the visible robot is consistently the proposed team. Relabel, reject, or mark mixed when it is not safe gallery evidence.</p>
        <div id="galleryCandidateList"></div>
        <div id="galleryTeamStates" style="border-top:1px solid #1e293b;margin-top:8px;padding-top:10px;"></div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px;padding-top:10px;border-top:1px solid #1e293b;">
          <button id="gallerySubmit" style="padding:8px 14px;background:#2563eb;color:#fff;border:0;border-radius:6px;cursor:pointer;font-weight:600;">Submit review</button>
          <button id="galleryClearDraft" style="padding:8px 12px;background:transparent;color:#f87171;border:1px solid #7f1d1d;border-radius:6px;cursor:pointer;">Clear draft</button>
          <span id="galleryDraftStatus" style="font-size:.78em;color:#64748b;"></span>
        </div>
      </div>`;
    document.getElementById('galleryReviewClose').onclick = () => {
        _galleryReviewBundle = null; _galleryReviewId = null; renderTracksTab();
    };
    const list = document.getElementById('galleryCandidateList');
    for (const candidate of bundle.candidates) {
        const decision = galleryDecisionFor(bundle, draft, candidate);
        const card = document.createElement('div');
        card.style.cssText = 'border-top:1px solid #1e293b;padding:12px 0;';
        const proposed = galleryEsc(candidate.proposedTeam || 'unassigned');
        const source = `${galleryEsc(candidate.source?.match || '')} · track ${galleryEsc(candidate.source?.sourceTrack)} · ${galleryEsc(candidate.source?.startS)}–${galleryEsc(candidate.source?.endS)}s`;
        card.innerHTML = `<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline;flex-wrap:wrap;">
          <b>${proposed}</b><span style="font-size:.76em;color:#64748b;">${source}</span>
          <span class="galleryCandidateState" style="font-size:.75em;color:${actionColor[decision?.action] || '#64748b'};">${galleryEsc(decision?.action || 'unreviewed')}</span></div>
          <div style="display:flex;gap:8px;overflow-x:auto;padding:8px 0;">${candidate.views.map(v => v.thumbnail?.startsWith('data:image/jpeg;base64,')
              ? `<img src="${v.thumbnail}" alt="gallery candidate" style="height:150px;width:auto;border-radius:5px;border:1px solid #334155;">`
              : '<span style="color:#64748b;padding:30px 8px;">thumbnail unavailable</span>').join('')}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;"><button data-action="accept">Accept</button><button data-action="relabel">Relabel</button><button data-action="reject">Reject</button><button data-action="mixed">Mixed</button><button data-action="split">Split</button><button data-action="defer">Defer</button></div>`;
        card.querySelectorAll('button[data-action]').forEach(button => {
            button.style.cssText = 'padding:5px 9px;background:transparent;color:#cbd5e1;border:1px solid #334155;border-radius:5px;cursor:pointer;font-size:.78em;';
            button.onclick = () => {
                const action = button.dataset.action;
                let team = candidate.proposedTeam;
                if (action === 'relabel') {
                    team = prompt('Team number for this tracklet:', candidate.proposedTeam || '');
                    if (!team) return;
                }
                draft.decisions[candidate.candidateId] = {
                    action, team, acceptedViewHashes: action === 'accept' || action === 'relabel'
                        ? candidate.views.map(v => v.cropHash).filter(Boolean) : [],
                };
                saveGalleryDraft(bundle, draft);
                renderGalleryReviewPanel(host, relay, bundle);
            };
        });
        list.appendChild(card);
    }
    const stateHost = document.getElementById('galleryTeamStates');
    const proposedTeams = [...new Set(bundle.candidates.map(c => String(c.proposedTeam || '')).filter(Boolean))].sort();
    if (proposedTeams.length) {
        stateHost.innerHTML = `<div style="font-size:.78em;color:#94a3b8;margin-bottom:6px;">Team gallery state</div>`;
        for (const team of proposedTeams) {
            const row = document.createElement('label');
            row.style.cssText = 'display:inline-flex;align-items:center;gap:5px;margin:0 12px 5px 0;font-size:.78em;color:#cbd5e1;';
            row.innerHTML = `<span>${galleryEsc(team)}</span><select data-team-state="${galleryEsc(team)}" style="background:#0f172a;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:3px;"><option value="">unchanged</option><option value="needs-more">needs more</option><option value="sufficient">sufficient</option><option value="robot-changed">robot changed</option></select>`;
            const select = row.querySelector('select');
            select.value = draft.teamStates?.[team] || '';
            select.onchange = () => {
                if (select.value) draft.teamStates[team] = select.value;
                else delete draft.teamStates[team];
                saveGalleryDraft(bundle, draft);
            };
            stateHost.appendChild(row);
        }
    }
    document.getElementById('galleryClearDraft').onclick = () => {
        localStorage.removeItem(galleryDraftKey(bundle));
        renderGalleryReviewPanel(host, relay, bundle);
    };
    document.getElementById('gallerySubmit').onclick = async () => {
        const decisions = Object.entries(draft.decisions || {}).map(([candidateId, value]) => ({
            candidateId, ...value,
        }));
        if (!decisions.length) return alert('Review at least one tracklet first.');
        const payload = { kind: 'galleryReviewAnswer', schemaVersion: 1,
            reviewId: bundle.reviewId, bundleHash: bundle.bundleHash,
            baseGalleryVersion: bundle.galleryVersion, submittedAt: new Date().toISOString(),
            decisions, teamStates: Object.entries(draft.teamStates || {}).map(([team, state]) => ({ team, state })) };
        const token = localStorage.getItem('rtrackToken') || '';
        try {
            const response = await fetch(`${relay}/gallery-answer/${encodeURIComponent(bundle.reviewId)}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'Rtrack-Token': token },
                body: JSON.stringify(payload),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
            localStorage.removeItem(galleryDraftKey(bundle));
            alert('Gallery review submitted.');
            _galleryReviewBundle = null; _galleryReviewId = null; renderTracksTab();
        } catch (error) { alert(`Gallery review failed: ${error.message}`); }
    };
}

function legacyRenderGalleryReviewQueue(host, relay, items) {
    const reviews = galleryReviewItems(items);
    const replayItems = (items || []).filter(item => item?.kind === 'gallery-status');
    const block = document.createElement('div');
    block.style.cssText = 'border:1px solid #1e293b;border-radius:8px;padding:10px 12px;margin-bottom:14px;';
    block.innerHTML = `<h4 style="margin:0 0 2px;font-size:.9em;color:#e2e8f0;">Gallery review</h4>
      <p style="margin:0 0 8px;font-size:.76em;color:#64748b;">Human-approved appearance evidence · bundles are bounded deltas, not the season gallery.</p>`;
    if (replayItems.length) {
        const counts = {};
        for (const item of replayItems) counts[item.state || 'unknown'] = (counts[item.state || 'unknown'] || 0) + 1;
        block.innerHTML += `<p style="margin:5px 0 8px;color:#94a3b8;font-size:.78em;">Replay: ${Object.entries(counts).map(([state, count]) => `${galleryEsc(state)} ${count}`).join(' · ')}</p>`;
    }
    if (!relay) { block.innerHTML += '<p style="margin:0;color:#f59e0b;font-size:.82em;">Configure the relay above to review gallery candidates.</p>'; host.prepend(block); return; }
    if (!reviews.length) { block.innerHTML += '<p style="margin:0;color:#64748b;font-size:.82em;">No gallery review bundles waiting.</p>'; host.prepend(block); return; }
    const table = document.createElement('table');
    table.style.cssText = 'width:100%;border-collapse:collapse;font-size:.84em;';
    table.innerHTML = '<tr style="color:#64748b;text-align:left;"><th style="padding:6px 4px;">Review</th><th style="padding:6px 4px;">Candidates</th><th style="padding:6px 4px;">State</th><th style="padding:6px 4px;text-align:right;">Action</th></tr>';
    reviews.forEach(review => {
        const bundle = review['gallery-bundle'];
        const answer = review['gallery-answer'];
        const tr = document.createElement('tr');
        tr.style.borderTop = '1px solid #1e293b';
        tr.innerHTML = `<td style="padding:7px 4px;font-weight:600;">${galleryEsc(review.id.slice(0, 12))}</td>
          <td style="padding:7px 4px;">${galleryEsc(bundle?.candidateCount ?? '—')}</td>
          <td style="padding:7px 4px;color:${answer ? '#22c55e' : '#f59e0b'};">${answer ? 'submitted' : 'ready'}</td>
          <td style="padding:7px 4px;text-align:right;"><button class="galleryOpen" style="padding:5px 10px;background:#2563eb;color:#fff;border:0;border-radius:5px;cursor:pointer;">${answer ? 'View' : 'Review'}</button></td>`;
        tr.querySelector('.galleryOpen').onclick = async () => {
            const loaded = await loadGalleryReviewBundle(relay, review.id);
            if (loaded) renderGalleryReviewPanel(host, relay, loaded);
        };
        table.appendChild(tr);
    });
    block.appendChild(table);
    host.prepend(block);
}

// Team-oriented gallery review.  This definition intentionally follows the original
// first-slice helpers above so old cached bundles can remain readable in source history;
// the team-oriented schema-2 implementation is the one used by renderTracksTab.
function galleryReviewItemsV2(items) {
    const out = new Map();
    for (const item of items || []) {
        if (!item?.id || !item.kind?.startsWith('gallery-')) continue;
        const row = out.get(item.id) || { id: item.id };
        row[item.kind] = item;
        out.set(item.id, row);
    }
    return [...out.values()].sort((a, b) => (b['gallery-bundle']?.at || 0)
        - (a['gallery-bundle']?.at || 0));
}

function galleryAnswerHasTeam(answer, team) {
    if (!answer) return false;
    const wanted = String(team);
    return (Array.isArray(answer.selections)
        && answer.selections.some(item => String(item?.team) === wanted))
        || (Array.isArray(answer.teamStates)
        && answer.teamStates.some(item => String(item?.team) === wanted));
}

async function hydrateGalleryAnswersV2(relay, reviews) {
    return Promise.all(reviews.map(async review => {
        const indexedAnswer = review['gallery-answer'];
        if ((indexedAnswer?.kind === 'galleryReviewAnswer'
             && Array.isArray(indexedAnswer.selections)) || !review.id) return review;
        try {
            const response = await fetch(`${relay}/gallery-answer/${encodeURIComponent(review.id)}`, {
                cache: 'no-store',
            });
            if (!response.ok) return review;
            const answer = await response.json();
            return answer?.kind === 'galleryReviewAnswer'
                ? { ...review, 'gallery-answer': answer } : review;
        } catch { return review; }
    }));
}

async function loadGalleryReviewBundleV2(relay, reviewId) {
    if (!relay || !reviewId) return null;
    try {
        const response = await fetch(`${relay}/gallery-bundle/${encodeURIComponent(reviewId)}`,
                                     { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const bundle = await response.json();
        if (bundle.kind !== 'galleryReviewBundle' || bundle.schemaVersion !== 2
            || !Array.isArray(bundle.teams)) throw new Error('outdated or invalid team gallery bundle');
        return bundle;
    } catch (error) {
        alert(`Could not load gallery review: ${error.message}`);
        return null;
    }
}

function readGalleryDraftV2(bundle) {
    try {
        const draft = JSON.parse(localStorage.getItem(galleryDraftKey(bundle)) || 'null');
        return draft?.bundleHash === bundle.bundleHash
            ? draft : { bundleHash: bundle.bundleHash, selections: {}, touched: {}, teamStates: {} };
    } catch { return { bundleHash: bundle.bundleHash, selections: {}, touched: {}, teamStates: {} }; }
}

function galleryThumb(image) {
    return typeof image?.thumbnail === 'string'
        && image.thumbnail.startsWith('data:image/jpeg;base64,') ? image.thumbnail : null;
}

function galleryDetectionOverlay(image) {
    const size = image?.source?.cropSize;
    const box = image?.source?.detectionBox;
    if (!Array.isArray(size) || size.length !== 2 || !Array.isArray(box) || box.length !== 4
        || !size[0] || !size[1]) return null;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.setAttribute('aria-label', 'actual detector box');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:150px;pointer-events:none;';
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', box[0]); rect.setAttribute('y', box[1]);
    rect.setAttribute('width', Math.max(0, box[2] - box[0]));
    rect.setAttribute('height', Math.max(0, box[3] - box[1]));
    rect.setAttribute('fill', 'none'); rect.setAttribute('stroke', '#f59e0b');
    rect.setAttribute('stroke-width', Math.max(2, size[0] / 180));
    rect.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.appendChild(rect);
    return svg;
}

function galleryProvenanceLabel(image) {
    if (image?.role === 'current') return 'Reviewed gallery image';
    if (image?.role === 'curation-reference') return 'Curation fallback · human-labeled';
    if (image?.role === 'candidate') return 'Track candidate';
    return image?.role || 'Gallery image';
}

function galleryImageCard(image, selected, onChange, readOnly = false) {
    const card = document.createElement('label');
    card.className = 'galleryImageCard';
    card.style.cssText = `display:inline-flex;vertical-align:top;flex-direction:column;gap:5px;min-width:150px;max-width:220px;padding:7px;border:1px solid ${selected ? '#2563eb' : '#334155'};border-radius:7px;background:${selected ? '#172554' : '#0f172a'};cursor:${readOnly ? 'default' : 'pointer'};box-sizing:border-box;`;
    const thumb = galleryThumb(image);
    if (thumb) {
        const media = document.createElement('div');
        media.style.cssText = 'position:relative;width:100%;height:150px;';
        const img = document.createElement('img');
        img.src = thumb; img.alt = readOnly ? 'current gallery image' : 'candidate robot image';
        img.style.cssText = 'width:100%;height:150px;object-fit:contain;background:#020617;border-radius:4px;';
        media.appendChild(img);
        const overlay = galleryDetectionOverlay(image);
        if (overlay) media.appendChild(overlay);
        card.appendChild(media);
    } else {
        const missing = document.createElement('div');
        missing.textContent = 'image unavailable';
        missing.style.cssText = 'height:150px;display:grid;place-items:center;color:#64748b;font-size:.78em;';
        card.appendChild(missing);
    }
    if (readOnly) {
        const provenance = document.createElement('div');
        provenance.textContent = galleryProvenanceLabel(image);
        provenance.style.cssText = `font-size:.68em;color:${image?.role === 'current' ? '#86efac' : '#fbbf24'};font-weight:600;`;
        card.appendChild(provenance);
    }
    let selectionRow = null;
    if (!readOnly) {
        selectionRow = document.createElement('div');
        selectionRow.style.cssText = 'display:flex;align-items:center;gap:6px;min-width:0;';
        const input = document.createElement('input');
        input.type = 'checkbox'; input.checked = !!selected;
        input.style.cssText = 'width:18px;height:18px;accent-color:#2563eb;flex:0 0 auto;margin:0;';
        let currentSelected = !!selected;
        const applySelected = checked => {
            currentSelected = !!checked;
            input.checked = currentSelected;
            card.style.borderColor = currentSelected ? '#2563eb' : '#334155';
            card.style.background = currentSelected ? '#172554' : '#0f172a';
            card.setAttribute('aria-checked', String(currentSelected));
        };
        const toggle = () => {
            applySelected(!currentSelected);
            onChange(currentSelected);
        };
        input.onchange = () => { applySelected(input.checked); onChange(currentSelected); };
        const imageElement = card.querySelector('img');
        if (imageElement) {
            imageElement.title = 'Click image to select or deselect';
            imageElement.style.cursor = 'pointer';
            imageElement.onclick = event => {
                // Prevent the label's default activation from toggling the checkbox a
                // second time. Selection is intentionally driven by the image itself.
                event.preventDefault();
                event.stopPropagation();
                toggle();
            };
        }
        selectionRow.appendChild(input);
        card.appendChild(selectionRow);
    }
    const source = image.source || {};
    const info = document.createElement('div');
    info.style.cssText = 'font-size:.7em;color:#94a3b8;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
    info.textContent = readOnly
        ? `${source.match || 'prior review'} · ${source.time ?? '—'}s`
        : `${String(source.match || '').replace(/^\d+[^_]*_/, '')} · T${source.sourceTrack ?? '—'} · ${source.time ?? '—'}s`;
    if (selectionRow) selectionRow.appendChild(info); else card.appendChild(info);
    return card;
}

function renderGalleryReviewPanelV2(host, relay, bundle, selectedTeam) {
    const draft = readGalleryDraftV2(bundle);
    const group = bundle.teams.find(item => String(item.team) === String(selectedTeam)) || bundle.teams[0];
    if (!group) return;
    const team = String(group.team);
    _galleryReviewBundle = bundle;
    host.innerHTML = '';
    const shell = document.createElement('div');
    shell.style.cssText = 'border:1px solid #2563eb;border-radius:8px;padding:14px;margin-bottom:14px;';
    shell.innerHTML = `<style>
      @media (max-width: 640px) {
        .galleryImageStrip { display:grid !important; grid-template-columns:repeat(2,minmax(0,1fr)); overflow:visible !important; gap:7px !important; }
        .galleryImageStrip .galleryImageCard { min-width:0 !important; max-width:none !important; width:100%; }
        .galleryImageStrip .galleryImageCard > div:first-child { height:125px !important; }
        .galleryImageStrip .galleryImageCard img,
        .galleryImageStrip .galleryImageCard svg { height:125px !important; }
      }
      @media (max-width: 340px) {
        .galleryImageStrip { grid-template-columns:1fr; }
      }
    </style><div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;"><div><b>Gallery review · team ${galleryEsc(team)}</b><div style="font-size:.76em;color:#64748b;margin-top:3px;">Select only candidate images that are strong representations of this robot. Current examples are reference-only.</div></div><button id="galleryReviewClose" style="padding:6px 10px;background:transparent;color:#94a3b8;border:1px solid #334155;border-radius:6px;cursor:pointer;">Close</button></div>`;
    const currentTitle = document.createElement('h4');
    currentTitle.textContent = `Current gallery (${group.currentGallery.length})`;
    const provenanceCounts = group.currentGallery.reduce((counts, image) => {
        const key = image?.role === 'current' ? 'reviewed gallery' : 'curation fallback';
        counts[key] = (counts[key] || 0) + 1;
        return counts;
    }, {});
    const provenanceSummary = Object.entries(provenanceCounts)
        .map(([label, count]) => `${count} ${label}`).join(' · ');
    if (provenanceSummary) currentTitle.textContent += ` · ${provenanceSummary}`;
    currentTitle.style.cssText = 'margin:16px 0 7px;color:#cbd5e1;font-size:.86em;';
    const current = document.createElement('div');
    current.className = 'galleryImageStrip';
    current.style.cssText = 'display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;';
    group.currentGallery.forEach(image => current.appendChild(galleryImageCard(image, false, null, true)));
    if (!group.currentGallery.length) { current.textContent = 'No reviewed gallery images yet.'; current.style.color = '#64748b'; }
    const candidateTitle = document.createElement('h4');
    candidateTitle.textContent = `Candidate images (${group.candidates.length})`;
    candidateTitle.style.cssText = 'margin:16px 0 7px;color:#cbd5e1;font-size:.86em;';
    shell.appendChild(candidateTitle);
    const candidates = document.createElement('div');
    candidates.className = 'galleryImageStrip';
    candidates.style.cssText = 'display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;';
    const selected = new Set((draft.selections?.[team] || []).map(String));
    group.candidates.forEach(image => {
        candidates.appendChild(galleryImageCard(image, selected.has(String(image.cropHash)), checked => {
            const next = new Set((draft.selections?.[team] || []).map(String));
            if (checked) next.add(String(image.cropHash)); else next.delete(String(image.cropHash));
            draft.selections[team] = [...next]; draft.touched[team] = true;
            saveGalleryDraft(bundle, draft);
            const status = document.getElementById('gallerySubmitStatus');
            if (status) status.textContent = `${next.size} selected for this team`;
        }));
    });
    shell.appendChild(candidates);
    const state = document.createElement('div');
    state.style.cssText = 'display:flex;gap:8px;align-items:center;margin-top:12px;font-size:.8em;color:#94a3b8;';
    state.innerHTML = `<span>Team state</span><select id="galleryTeamState" style="background:#0f172a;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:4px;"><option value="">unchanged</option><option value="needs-more">needs more</option><option value="sufficient">sufficient</option><option value="robot-changed">robot changed</option></select>`;
    shell.appendChild(state);
    const stateSelect = state.querySelector('select');
    stateSelect.value = draft.teamStates?.[team] || '';
    stateSelect.onchange = () => { if (stateSelect.value) draft.teamStates[team] = stateSelect.value; else delete draft.teamStates[team]; draft.touched[team] = true; saveGalleryDraft(bundle, draft); };
    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px;padding-top:10px;border-top:1px solid #1e293b;';
    actions.innerHTML = `<button id="gallerySubmit" style="padding:8px 14px;background:#2563eb;color:#fff;border:0;border-radius:6px;cursor:pointer;font-weight:600;">Submit selected teams</button><button id="galleryNext" style="display:none;padding:8px 14px;background:#16a34a;color:#fff;border:0;border-radius:6px;cursor:pointer;font-weight:600;">Next gallery</button><button id="galleryClearDraft" style="padding:8px 12px;background:transparent;color:#f87171;border:1px solid #7f1d1d;border-radius:6px;cursor:pointer;">Clear draft</button><span id="gallerySubmitStatus" style="font-size:.78em;color:#64748b;">${selected.size} selected for this team</span>`;
    shell.appendChild(actions);
    currentTitle.style.marginTop = '18px';
    shell.appendChild(currentTitle);
    shell.appendChild(current);
    host.appendChild(shell);
    document.getElementById('galleryReviewClose').onclick = () => { _galleryReviewBundle = null; renderTracksTab(); };
    document.getElementById('galleryClearDraft').onclick = () => { localStorage.removeItem(galleryDraftKey(bundle)); renderGalleryReviewPanelV2(host, relay, bundle, team); };
    document.getElementById('gallerySubmit').onclick = async () => {
        const groups = new Map(bundle.teams.map(group => [String(group.team), group]));
        const selections = Object.entries(draft.selections || {}).filter(([t]) => draft.touched?.[t]).map(([t, include]) => ({
            team: t,
            include,
            // A submitted team review acknowledges the complete candidate set
            // shown for that team, including intentionally empty selections.
            reviewed: (groups.get(String(t))?.candidates || []).map(candidate => candidate.candidateId).filter(Boolean),
        }));
        if (!selections.length) return alert('Select at least one team gallery, even if its candidate set is intentionally empty.');
        const payload = { kind: 'galleryReviewAnswer', schemaVersion: 2, reviewId: bundle.reviewId,
            bundleHash: bundle.bundleHash, baseGalleryVersion: bundle.galleryVersion,
            submittedAt: new Date().toISOString(), selections,
            teamStates: Object.entries(draft.teamStates || {}).filter(([t]) => draft.touched?.[t]).map(([t, state]) => ({ team: t, state })) };
        const token = localStorage.getItem('rtrackToken') || '';
        try {
            const response = await fetch(`${relay}/gallery-answer/${encodeURIComponent(bundle.reviewId)}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Rtrack-Token': token }, body: JSON.stringify(payload) });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
            localStorage.removeItem(galleryDraftKey(bundle));
            const submit = document.getElementById('gallerySubmit');
            const nextButton = document.getElementById('galleryNext');
            const status = document.getElementById('gallerySubmitStatus');
            if (submit) { submit.disabled = true; submit.textContent = 'Submitted'; submit.style.opacity = '.65'; }
            if (status) { status.textContent = 'Gallery selections submitted.'; status.style.color = '#86efac'; }
            const next = galleryNextTarget(bundle.reviewId, team);
            if (next && nextButton) {
                nextButton.style.display = 'inline-block';
                nextButton.textContent = `Next gallery · team ${next.team}`;
                nextButton.onclick = async () => {
                    nextButton.disabled = true;
                    const loaded = await loadGalleryReviewBundleV2(relay, next.reviewId);
                    if (loaded) renderGalleryReviewPanelV2(host, relay, loaded, next.team);
                    else nextButton.disabled = false;
                };
            } else if (status) {
                status.textContent += ' No more galleries are waiting.';
            }
        } catch (error) { alert(`Gallery review failed: ${error.message}`); }
    };
}

function galleryAllianceReviewStats(reviews, matches) {
    const matchByKey = new Map((matches || []).map(match => [match.key, match]));
    const byTeam = new Map();
    for (const review of reviews || []) {
        const bundle = review['gallery-bundle'];
        const matchKey = bundle?.match;
        const match = matchByKey.get(matchKey);
        if (!match) continue;
        for (const teamValue of bundle.teamIds || []) {
            const team = String(teamValue);
            if (!galleryAnswerHasTeam(review['gallery-answer'], team)) continue;
            const color = (match.red || []).map(String).includes(team) ? 'red'
                : (match.blue || []).map(String).includes(team) ? 'blue' : null;
            if (!color) continue;
            if (!byTeam.has(team)) byTeam.set(team, { red: new Set(), blue: new Set() });
            byTeam.get(team)[color].add(matchKey);
        }
    }
    return byTeam;
}

async function renderGalleryReviewQueueV2(host, relay, items, matches, hydratedReviews = null) {
    const reviews = hydratedReviews
        || await hydrateGalleryAnswersV2(relay, galleryReviewItemsV2(items));
    const allianceStats = galleryAllianceReviewStats(reviews, matches);
    const replayItems = (items || []).filter(item => item?.kind === 'gallery-status');
    const block = document.createElement('details');
    block.open = localStorage.getItem('rtrackSection:gallery') !== 'closed';
    block.ontoggle = () => localStorage.setItem('rtrackSection:gallery', block.open ? 'open' : 'closed');
    block.style.cssText = 'border:1px solid #1e293b;border-radius:8px;padding:10px 12px;margin-bottom:14px;';
    block.innerHTML = `<summary style="cursor:pointer;color:#e2e8f0;font-size:.9em;font-weight:700;">Gallery review by team</summary><p style="margin:8px 0;font-size:.76em;color:#64748b;">Compare reviewed examples with larger, context-padded track detections. Reviewed-match counts are split by the alliance color providing the appearance evidence.</p>`;
    if (replayItems.length) {
        const counts = {}; for (const item of replayItems) counts[item.state || 'unknown'] = (counts[item.state || 'unknown'] || 0) + 1;
        block.innerHTML += `<p style="margin:5px 0 8px;color:#94a3b8;font-size:.78em;">Replay: ${Object.entries(counts).map(([state, count]) => `${galleryEsc(state)} ${count}`).join(' · ')}</p>`;
    }
    if (!relay) { block.innerHTML += '<p style="margin:0;color:#f59e0b;font-size:.82em;">Configure the relay above to review gallery candidates.</p>'; host.prepend(block); return; }
    const rowsByTeam = new Map();
    const outdatedRows = [];
    for (const review of reviews) {
        const bundle = review['gallery-bundle'];
        const teams = Array.isArray(bundle?.teamIds) ? bundle.teamIds : [];
        if (!teams.length) {
            outdatedRows.push({ review, team: 'outdated bundle', current: '—', candidates: '—', disabled: true, batchCount: 1 });
            continue;
        }
        teams.forEach(teamValue => {
            const team = String(teamValue);
            const contribution = {
                review, team,
                current: bundle.currentImagesByTeam?.[team] ?? '—',
                candidates: bundle.candidatesByTeam?.[team] ?? '—',
                submitted: galleryAnswerHasTeam(review['gallery-answer'], team),
                at: bundle?.at || 0,
            };
            const group = rowsByTeam.get(team) || [];
            group.push(contribution);
            rowsByTeam.set(team, group);
        });
    }
    const rows = [...rowsByTeam.entries()].map(([team, contributions]) => {
        contributions.sort((a, b) => a.at - b.at);
        const pending = contributions.filter(item => !item.submitted);
        // A team is one season gallery. Match bundles are merely successive batches
        // of candidate evidence, so expose one row and advance it oldest-first.
        const active = pending[0] || contributions[contributions.length - 1];
        return {
            ...active,
            team,
            submitted: pending.length === 0,
            batchCount: contributions.length,
            pendingCount: pending.length,
            pendingContributions: pending,
        };
    }).sort((a, b) => Number(a.team) - Number(b.team));
    rows.push(...outdatedRows);
    _galleryReviewQueue = rows.flatMap(row => (row.pendingContributions || []).map(item => ({
        reviewId: item.review.id,
        team: row.team,
    })));
    if (!rows.length) { block.innerHTML += '<p style="margin:0;color:#64748b;font-size:.82em;">No team gallery bundles waiting.</p>'; host.prepend(block); return; }
    const table = document.createElement('table'); table.style.cssText = 'width:100%;border-collapse:collapse;font-size:.84em;';
    table.innerHTML = '<tr style="color:#64748b;text-align:left;"><th style="padding:6px 4px;">Team</th><th style="padding:6px 4px;">Reviewed matches</th><th style="padding:6px 4px;">Current</th><th style="padding:6px 4px;">Candidates</th><th style="padding:6px 4px;">State</th><th style="padding:6px 4px;text-align:right;">Action</th></tr>';
    rows.forEach(row => {
        const tr = document.createElement('tr'); tr.style.borderTop = '1px solid #1e293b';
        const submitted = !!row.submitted;
        const batchNote = row.batchCount > 1
            ? ` · ${row.pendingCount || 0}/${row.batchCount} batches pending` : '';
        const state = row.disabled ? 'outdated' : submitted ? 'submitted' : `ready${batchNote}`;
        const reviewed = allianceStats.get(String(row.team)) || { red: new Set(), blue: new Set() };
        const count = (color, set) => `<span style="color:${set.size ? (color === 'red' ? '#f87171' : '#60a5fa') : '#fbbf24'};font-weight:${set.size ? 600 : 800};${set.size ? '' : 'background:#78350f;padding:2px 5px;border-radius:4px;'}">${color[0].toUpperCase()} ${set.size}</span>`;
        tr.innerHTML = `<td style="padding:7px 4px;font-weight:600;">${galleryEsc(row.team)}</td><td style="padding:7px 4px;white-space:nowrap;">${count('red', reviewed.red)} · ${count('blue', reviewed.blue)}</td><td style="padding:7px 4px;">${galleryEsc(row.current)}</td><td style="padding:7px 4px;">${galleryEsc(row.candidates)}</td><td style="padding:7px 4px;color:${submitted ? '#22c55e' : '#f59e0b'};">${galleryEsc(state)}</td><td style="padding:7px 4px;text-align:right;"><button class="galleryOpen" ${row.disabled ? 'disabled' : ''} style="padding:5px 10px;background:#2563eb;color:#fff;border:0;border-radius:5px;cursor:pointer;">Review</button></td>`;
        tr.querySelector('.galleryOpen').onclick = async () => { const loaded = await loadGalleryReviewBundleV2(relay, row.review.id); if (loaded) renderGalleryReviewPanelV2(host, relay, loaded, row.team); };
        table.appendChild(tr);
    });
    block.appendChild(table); host.prepend(block);
}

async function renderTracksTab() {
    const host = document.getElementById('tools-tab-tracks');
    if (!host) return;
    const relay = relayUrl();
    const eventKey = (document.getElementById('eventKeyInput')?.value || '').trim().toLowerCase();

    host.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
        <input id="trkRelay" value="${relay}" placeholder="https://rtrack-relay.<you>.workers.dev"
               style="flex:1;min-width:240px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;
                      border-radius:6px;padding:8px 10px;font-size:0.85em;">
        <button id="trkSave" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;
                background:transparent;color:#94a3b8;cursor:pointer;">Save</button>
        <button id="trkRefresh" style="padding:8px 12px;border-radius:6px;border:1px solid #2563eb;
                background:#2563eb;color:#fff;cursor:pointer;font-weight:600;">Refresh</button>
      </div>
      <div id="trkBody" style="color:#94a3b8;">Loading…</div>`;

    document.getElementById('trkSave').onclick = () => {
        localStorage.setItem(RELAY_KEY, document.getElementById('trkRelay').value.trim());
        renderTracksTab();
    };
    document.getElementById('trkRefresh').onclick = () => renderTracksTab();

    const body = document.getElementById('trkBody');
    const [man, items, matches] = await Promise.all([
        loadTracksManifest(true),
        _relayIndex(relay),
        (async () => {
            try {
                return eventKey
                    ? await db.matches.where('eventKey').equals(eventKey).toArray()
                    : await db.matches.toArray();
            } catch { return []; }
        })(),
    ]);

    const pub = new Map((man.matches || []).map(m => [m.key, m]));
    // RELAY TRACKS COUNT AS PUBLISHED. rtrack.export posts routes to the relay as it
    // writes public/tracks/, so a match finishes and is viewable long before anyone
    // commits the manifest -- and this tab read ONLY the committed manifest. The result
    // was a match that had been curated, solved, exported and pushed still showing
    // "curated · awaiting rerun" with no Routes link, which is the opposite of the
    // truth and exactly the feedback the watcher exists to provide.
    const relayTracks = new Map(
        (items || []).filter(it => it.kind === 'tracks' && it.id).map(it => [it.id, it]));
    const onRelay = new Map();
    for (const it of items || []) onRelay.set(`${it.kind}:${it.id}`, it);
    const gal = (man.gallery || {})[eventKey] || {};
    const galleryReviews = relay
        ? await hydrateGalleryAnswersV2(relay, galleryReviewItemsV2(items || [])) : [];
    const allianceReviewStats = galleryAllianceReviewStats(galleryReviews, matches);

    // Union of THREE sources, and the third is the one that matters most here:
    //   db.matches   the schedule — quals only, syncTBAMatches drops playoffs
    //   manifest     what has been exported and published
    //   the relay    what is IN FLIGHT right now
    //
    // A match waiting to be curated is typically in NONE of the first two: it has not
    // been published (that is what curation unblocks) and, if it is a playoff, the
    // schedule never had it. Leaving the relay out of this set made exactly the rows
    // this tab exists to surface invisible -- 2026mawor_qm1 sat on the relay with a
    // bundle and never appeared.
    const relayKeys = (items || [])
        .filter(it => it.kind === 'bundle' || it.kind === 'answer' || it.kind === 'calib')
        .map(it => it.id);
    const keys = new Set([...matches.map(m => m.key), ...pub.keys(), ...relayKeys,
                          ...relayTracks.keys()]);
    const rows = [...keys]
        .filter(k => !eventKey || k.startsWith(eventKey + '_'))
        .map(k => {
            const dbm = matches.find(m => m.key === k);
            const p = pub.get(k);
            const rt = relayTracks.get(k);
            const teams = (p?.teams) || [...(dbm?.red || []), ...(dbm?.blue || [])].map(String);
            const known = teams.filter(t => gal[t]).length;
            const sameAllianceReviewed = teams.filter(teamValue => {
                const team = String(teamValue);
                const color = (dbm?.red || []).map(String).includes(team) ? 'red'
                    : (dbm?.blue || []).map(String).includes(team) ? 'blue' : null;
                return color && (allianceReviewStats.get(team)?.[color]?.size || 0) > 0;
            }).length;
            return {
                key: k, teams,
                published: !!p || !!rt,
                curated: !!p?.curated,
                // Live = on the relay but not in the committed manifest. Worth saying
                // out loud rather than smoothing over: it is the difference between
                // "viewable now" and "will survive a cache clear on someone else's
                // device", and it is the cue that public/tracks/ wants committing.
                live: !!rt && !p,
                // git stamps an ISO string, the relay a ms epoch. Both end up ms here;
                // prefer whichever is NEWER so a fresh relay push wins over a stale
                // commit rather than being masked by it.
                exportedAt: Math.max(p?.exportedAt ? Date.parse(p.exportedAt) : 0,
                                     rt?.at || 0) || null,
                custody: p?.meanCustody ?? null,
                bundle: onRelay.get(`bundle:${k}`) || null,
                answer: onRelay.get(`answer:${k}`) || null,
                calib: onRelay.get(`calib:${k}`) || null,
                known, sameAllianceReviewed, total: teams.length,
                n: dbm?.matchNumber ?? p?.matchNumber ?? 0,
            };
        })
        .sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));

    if (!rows.length) {
        body.innerHTML = `<p>No matches for ${eventKey || 'any event'} yet.
          Set an event key on the Home tab, or publish tracks to <code>public/tracks/</code>.</p>`;
        return;
    }

    const relayNote = relay
        ? (items ? `<span style="color:#22c55e;">relay reachable · ${items.length} item(s)</span>`
                 : `<span style="color:#f59e0b;">relay not reachable — in-flight work will not show</span>`)
        : `<span style="color:#f59e0b;">no relay configured — showing published tracks only</span>`;

    const multiEvent = new Set(rows.map(r => r.key.split('_')[0])).size > 1;

    const pill = (txt, col) =>
        `<span style="display:inline-block;padding:2px 7px;border-radius:999px;font-size:0.72em;
         font-weight:700;border:1px solid ${col};color:${col};white-space:nowrap;">${txt}</span>`;

    const act = (txt, href, primary) =>
        `<a href="${href}" target="_blank" rel="noopener" style="display:inline-block;
          padding:5px 10px;border-radius:6px;font-size:0.78em;text-decoration:none;
          border:1px solid ${primary ? '#2563eb' : '#334155'};
          background:${primary ? '#2563eb' : 'transparent'};
          color:${primary ? '#fff' : '#94a3b8'};font-weight:${primary ? 600 : 400};">${txt}</a>`;

    const base = import.meta.env.BASE_URL;
    const q = (page, idParam, id) =>
        `${base}rtrack/${page}.html?relay=${encodeURIComponent(relay)}&${idParam}=${encodeURIComponent(id)}`;

    // Camera-level work, listed on its own and deliberately NOT event-filtered: a
    // camera id is whatever the calibration was named after, often a bare video id, and
    // filtering these by event prefix is what hid 2026mawor's only camera. Anything the
    // relay holds a calib frame for can be calibrated or have occluders drawn on it.
    const cams = (items || []).filter(it => it.kind === 'calib').map(it => it.id).sort();
    const occlOn = new Set((items || []).filter(it => it.kind === 'occl').map(it => it.id));
    const cameraBlock = cams.length ? `
      <details data-rtrack-section="cameras" style="border:1px solid #1e293b;border-radius:8px;padding:10px 12px;margin-bottom:14px;">
      <summary style="cursor:pointer;font-size:0.9em;font-weight:700;color:#e2e8f0;">Cameras</summary>
      <p style="margin:8px 0;font-size:0.76em;color:#64748b;">
        Done once per camera, then reused by every match shot on it.</p>
      <table style="width:100%;border-collapse:collapse;font-size:0.86em;">
        <tr style="color:#64748b;text-align:left;">
          <th style="padding:6px 4px;">Camera</th>
          <th style="padding:6px 4px;">Occluders</th>
          <th style="padding:6px 4px;text-align:right;">Actions</th>
        </tr>
        ${cams.map(id => `<tr style="border-top:1px solid #1e293b;">
          <td style="padding:7px 4px;font-weight:600;">${id}</td>
          <td style="padding:7px 4px;">${occlOn.has(id)
              ? pill('sent · pull with wait-occl', '#22c55e')
              : '<span style="color:#64748b;">not drawn</span>'}</td>
          <td style="padding:7px 4px;text-align:right;white-space:nowrap;">
            ${act('Calibrate', q('calibrate', 'video', id), false)}
            ${act('Occluders', q('occluders', 'video', id), false)}
          </td>
        </tr>`).join('')}
      </table></details>` : '';

    body.innerHTML = `
      <p style="font-size:0.82em;margin:0 0 10px;">${relayNote}</p>
      ${cameraBlock}
      <details data-rtrack-section="matches" style="border:1px solid #1e293b;border-radius:8px;padding:10px 12px;margin-bottom:14px;">
      <summary style="cursor:pointer;font-size:0.9em;font-weight:700;color:#e2e8f0;">Matches</summary>
      <table style="width:100%;border-collapse:collapse;font-size:0.86em;">
        <tr style="color:#64748b;text-align:left;">
          <th style="padding:6px 4px;">Match</th>
          <th style="padding:6px 4px;">State</th>
          <th style="padding:6px 4px;">Same-color reviewed</th>
          <th style="padding:6px 4px;">Custody</th>
          <th style="padding:6px 4px;text-align:right;">Actions</th>
        </tr>
        ${rows.map(r => {
            // PUBLISHED IS TERMINAL unless an answer arrived after it was built.
            // Answers are not deleted from the relay when consumed, so testing
            // "an answer exists" first left finished matches reading "awaiting rerun"
            // permanently. Compare timestamps instead: only a newer answer means work
            // is outstanding. Missing timestamps fall back to trusting the publish,
            // because a stale "pending" is more misleading than a stale "done" here.
            const answerNewer = r.answer?.at && r.exportedAt
                ? r.answer.at > r.exportedAt
                : (!!r.answer && !r.published);
            // A bundle only means WORK WAITING if it is newer than the last publish
            // AND has not already been answered. Bundles outlive their own usefulness:
            // they sit on the relay for the full 24 h TTL, so after a curator answers
            // one and the watcher republishes, the bundle is still there -- and a
            // naive "bundle exists" test then shows every finished match as needing
            // re-curation, which it did for all eleven 2026mawor matches at once.
            //
            // Two clocks settle it. A bundle pushed AFTER the last publish is a
            // genuine new pass (that is exactly how these were rebuilt). An answer
            // arriving after that bundle means the pass is done, whatever the bundle's
            // continued presence suggests.
            const bundleIsNew = r.bundle?.at && r.exportedAt
                ? r.bundle.at > r.exportedAt
                : !!r.bundle && !r.published;
            const answeredIt = r.answer?.at && r.bundle?.at
                ? r.answer.at > r.bundle.at
                : !!r.answer;
            const outstanding = bundleIsNew && !answeredIt;
            const state = answerNewer ? pill('curated · awaiting rerun', '#a78bfa')
                        : outstanding ? (r.published ? pill('bundle waiting · re-curate', '#f59e0b')
                                                     : pill('NEEDS CURATION', '#f59e0b'))
                        : r.live ? pill('published · live, uncommitted', '#2dd4bf')
                        : r.published ? (r.curated ? pill('published · curated', '#22c55e')
                                                   : pill('published · auto', '#60a5fa'))
                        : pill('no tracks', '#475569');
            const models = r.total
                ? `<span title="Teams with reviewed gallery evidence from this alliance color" style="color:${r.sameAllianceReviewed === r.total ? '#22c55e' : r.sameAllianceReviewed ? '#f59e0b' : '#64748b'};">
                     ${r.sameAllianceReviewed}/${r.total}</span>`
                : '—';
            const acts = [
                r.bundle ? act('Curate', q('curate', 'match', r.key), outstanding) : '',
                r.calib ? act('Calibrate', q('calibrate', 'video', r.key), !r.bundle) : '',
                // OCCLUDERS, gated on the same calib frame Calibrate uses -- the page
                // pulls that frame to draw on, so without one there is nothing to show.
                // Never primary: occluders are drawn ONCE PER CAMERA, not per match, so
                // a prominent button on every row would misrepresent the job as routine.
                r.calib ? act('Occluders', q('occluders', 'video', r.key), false) : '',
                r.published ? `<a href="#" onclick="viewMatchDetail('${r.key}');return false;"
                     style="display:inline-block;padding:5px 10px;border-radius:6px;font-size:0.78em;
                     text-decoration:none;border:1px solid #334155;color:#94a3b8;">Routes</a>` : '',
            ].filter(Boolean).join(' ');
            // Show the event prefix whenever more than one event is on screen. Two
            // different matches can share a suffix -- 2026necmp_f1m2 and
            // 2026mawor_f1m2 both render as "f1m2" -- and two identical-looking rows
            // with different numbers reads as a bug in the data rather than a bug in
            // the label.
            const label = multiEvent ? r.key : r.key.replace(/^[^_]+_/, '');
            return `<tr style="border-top:1px solid #1e293b;">
              <td style="padding:7px 4px;font-weight:600;">${label}</td>
              <td style="padding:7px 4px;">${state}</td>
              <td style="padding:7px 4px;">${models}</td>
              <td style="padding:7px 4px;">${r.custody != null ? Math.round(100 * r.custody) + '%' : '—'}</td>
              <td style="padding:7px 4px;text-align:right;white-space:nowrap;">${acts || '<span style="color:#475569;">—</span>'}</td>
            </tr>`;
        }).join('')}
      </table></details>
      <p style="font-size:0.76em;color:#64748b;margin-top:12px;">
        <b>Cameras</b> lists every camera the relay holds a frame for, pushed with
        <code>rtrack.relay push-calib &lt;camera&gt;</code>. Both tools describe the
        CAMERA, not the match, so they are done once and reused across every match shot
        on it: a calibration maps pixels to field metres, occluders mark what robots
        disappear behind. Drawn occluders come back with
        <code>rtrack.relay wait-occl &lt;camera&gt;</code>, which writes
        <code>calib/&lt;camera&gt;_occluders.json</code> -- the name the solver looks for
        on its own. Keep the camera id in the tool EXACTLY as listed here, or the file
        lands under a name nothing reads.
        <br><br>
        <b>published · live, uncommitted</b> means the watcher finished the match and
        pushed its routes to the relay, where they are viewable now and for seven days.
        They become permanent when <code>public/tracks/</code> is committed &mdash; until
        then a device that never fetched them will not find them after the relay entry
        expires.
        <br><br>
        <b>Same-color reviewed</b> is how many of the match's six teams have reviewed gallery evidence
        from the same alliance color they occupy in this match. For example, 3/6 means
        three teams have same-color reviewed evidence available.
      </p>`;
    body.querySelectorAll('details[data-rtrack-section]').forEach(section => {
        const key = `rtrackSection:${section.dataset.rtrackSection}`;
        section.open = localStorage.getItem(key) !== 'closed';
        section.ontoggle = () => localStorage.setItem(key, section.open ? 'open' : 'closed');
    });
    await renderGalleryReviewQueueV2(body, relay, items || [], matches, galleryReviews);
}

// ── Field Drawing Tab ────────────────────────────────────────────────────────
// Strokes: { pts: [{x,y}…] normalized to IMAGE rect (0–1), color }
//
// Key design: _fPt() reads img.getBoundingClientRect() live on every touch/mouse
// event. The canvas fills the wrapper via CSS (inset:0 100%/100%) and its buffer
// is sized to the image rendered dimensions. Coordinate mapping is always fresh —
// no timing-sensitive JS positioning that can fail mid-fullscreen-transition.

let fieldStrokes       = [];
let fieldDrawing       = false;
let fieldCurrentStroke = [];
let fieldCanvas        = null;
let fieldCtx           = null;
let fieldActiveYear    = null;
let fieldDrawMode      = false;
let fieldColor         = '#f8fafc'; // white default

// Match prep radar palette (R1-R3 red shades, B1-B3 blue shades) + white
const _FC = [
    { id:'fcR1', color:'#ef4444', label:'R1' },
    { id:'fcR2', color:'#f87171', label:'R2' },
    { id:'fcR3', color:'#fca5a5', label:'R3' },
    { id:'fcB1', color:'#3b82f6', label:'B1' },
    { id:'fcB2', color:'#60a5fa', label:'B2' },
    { id:'fcB3', color:'#93c5fd', label:'B3' },
    { id:'fcW',  color:'#f8fafc', label:'W'  },
];

function initFieldTab() {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase() || '';
    const year = eventKey.match(/^(\d{4})/)?.[1] || '2026';
    const container = document.getElementById('tools-tab-field');

    if (year !== fieldActiveYear) {
        fieldStrokes = [];
        fieldActiveYear = year;
    }

    const bs = `background:#1e293b;border:1px solid #334155;padding:6px 12px;border-radius:5px;cursor:pointer;font-size:0.82em;font-weight:600;color:`;
    const colorDots = _FC.map(c =>
        `<button id="${c.id}" onclick="fieldSetColor('${c.color}')" title="${c.label}"
            style="width:22px;height:22px;border-radius:4px;background:${c.color};border:2px solid ${c.color===fieldColor?'#fff':'transparent'};cursor:pointer;padding:0;flex-shrink:0;"></button>`
    ).join('');

    container.innerHTML = `
        <div id="fieldWrapper" style="border-radius:8px;overflow:hidden;">
            <div id="fieldToolbar" style="display:flex;gap:8px;align-items:center;padding:10px;flex-wrap:wrap;background:#0f172a;">
                <button id="fieldDrawToggle" onclick="fieldToggleDraw()" style="${bs}#94a3b8;">Draw</button>
                <button onclick="fieldUndo()"       style="${bs}#f8fafc;">Undo</button>
                <button onclick="fieldErase()"      style="${bs}#ef4444;">Erase All</button>
                <button onclick="fieldFullscreen()" style="${bs}#94a3b8;">⛶ Fullscreen</button>
                <div style="display:flex;gap:4px;align-items:center;margin-left:4px;">
                    <span style="color:#475569;font-size:0.72em;white-space:nowrap;">Color:</span>
                    ${colorDots}
                </div>
            </div>
            <div id="fieldImageWrap" style="position:relative;width:100%;max-width:960px;line-height:0;">
                <img id="fieldBgImg" src="${import.meta.env.BASE_URL}field/${year}-field.png"
                    style="display:block;width:100%;user-select:none;pointer-events:none;" draggable="false">
                <canvas id="fieldDrawCanvas"
                    style="position:absolute;inset:0;width:100%;height:100%;touch-action:none;"></canvas>
            </div>
        </div>`;

    fieldCanvas = document.getElementById('fieldDrawCanvas');
    fieldCtx    = fieldCanvas.getContext('2d');
    fieldDrawing = false;
    fieldCurrentStroke = [];
    _fUpdateToggle();

    const img = document.getElementById('fieldBgImg');
    const doSize = () => _fSizeCanvas(img);
    if (img.complete && img.naturalWidth) doSize();
    else img.addEventListener('load', doSize);

    // Bind to the wrapper so touch hits regardless of canvas buffer state.
    // touchstart/touchmove need passive:false to call preventDefault (blocks scroll).
    const wrap = document.getElementById('fieldImageWrap');
    wrap.addEventListener('mousedown',   _fDown);
    wrap.addEventListener('mousemove',   _fMove);
    wrap.addEventListener('mouseup',     _fUp);
    wrap.addEventListener('mouseleave',  _fUp);
    wrap.addEventListener('touchstart',  _fTouchStart, { passive: false });
    wrap.addEventListener('touchmove',   _fTouchMove,  { passive: false });
    wrap.addEventListener('touchend',    _fUp);
    wrap.addEventListener('touchcancel', _fUp);
}

// Resize the canvas drawing buffer to match the image's rendered dimensions.
// The canvas CSS (inset:0 / 100%×100%) already fills the wrapper — only the
// buffer needs updating so stroke line-width stays proportional.
function _fSizeCanvas(img) {
    img = img || document.getElementById('fieldBgImg');
    if (!fieldCanvas || !img) return;
    const r = img.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    fieldCanvas.width  = Math.round(r.width);
    fieldCanvas.height = Math.round(r.height);
    fieldRedraw();
}

// Resize after fullscreen transitions — retry a few times for Android layout settling.
document.addEventListener('fullscreenchange', () => {
    if (!fieldCanvas) return;
    [60, 200, 450].forEach(ms => setTimeout(() => _fSizeCanvas(), ms));
});

function _fUpdateToggle() {
    const btn = document.getElementById('fieldDrawToggle');
    if (!btn) return;
    if (fieldDrawMode) {
        btn.textContent = 'Drawing';
        btn.style.background  = '#166534';
        btn.style.color       = '#4ade80';
        btn.style.borderColor = '#166534';
    } else {
        btn.textContent = 'Draw';
        btn.style.background  = '#1e293b';
        btn.style.color       = '#94a3b8';
        btn.style.borderColor = '#334155';
    }
}

window.fieldToggleDraw = () => { fieldDrawMode = !fieldDrawMode; _fUpdateToggle(); };

window.fieldSetColor = (color) => {
    fieldColor = color;
    _FC.forEach(({ id, color: c }) => {
        const btn = document.getElementById(id);
        if (btn) btn.style.borderColor = c === color ? '#fff' : 'transparent';
    });
};

window.fieldFullscreen = () => {
    const el = document.getElementById('fieldWrapper');
    if (!el) return;

    if (document.fullscreenElement) { document.exitFullscreen(); return; }

    const isFaux = el.classList.contains('field-faux-fs');
    if (isFaux) {
        el.classList.remove('field-faux-fs');
        document.body.style.overflow = '';
        [60, 200].forEach(ms => setTimeout(() => _fSizeCanvas(), ms));
        return;
    }

    // iOS Safari does not support requestFullscreen on arbitrary elements —
    // use a CSS fixed-overlay as a universal fallback.
    const tryFaux = () => {
        el.classList.add('field-faux-fs');
        document.body.style.overflow = 'hidden';
        [60, 200, 450].forEach(ms => setTimeout(() => _fSizeCanvas(), ms));
    };
    if (el.requestFullscreen) el.requestFullscreen().catch(tryFaux);
    else tryFaux();
};

// _fPt reads the IMAGE's live bounding rect for coordinates, not the canvas.
// This is the key fix: accurate at call time regardless of canvas CSS state or
// any fullscreen transition timing.
function _fPt(e) {
    const img = document.getElementById('fieldBgImg');
    const r = img ? img.getBoundingClientRect() : fieldCanvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}

function _fDown(e)       { if (!fieldDrawMode) return; fieldDrawing = true; fieldCurrentStroke = [_fPt(e)]; }
function _fMove(e)       { if (!fieldDrawing) return; fieldCurrentStroke.push(_fPt(e)); _fDrawLive(); }
function _fUp()          { if (!fieldDrawing) return; fieldDrawing = false; if (fieldCurrentStroke.length) { fieldStrokes.push({ pts: [...fieldCurrentStroke], color: fieldColor }); fieldCurrentStroke = []; fieldRedraw(); } }
function _fTouchStart(e) { e.preventDefault(); if (!fieldDrawMode) return; fieldDrawing = true; fieldCurrentStroke = [_fPt(e.touches[0])]; }
function _fTouchMove(e)  { e.preventDefault(); if (!fieldDrawing) return; fieldCurrentStroke.push(_fPt(e.touches[0])); _fDrawLive(); }

function _fStroke(ctx, stroke) {
    const { pts, color } = stroke;
    const W = fieldCanvas.width, H = fieldCanvas.height;
    if (!pts.length) return;
    ctx.strokeStyle = color;
    ctx.fillStyle   = color;
    ctx.beginPath();
    if (pts.length === 1) { ctx.arc(pts[0].x * W, pts[0].y * H, 2, 0, Math.PI * 2); ctx.fill(); return; }
    ctx.moveTo(pts[0].x * W, pts[0].y * H);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * W, pts[i].y * H);
    ctx.stroke();
}

function fieldRedraw() {
    if (!fieldCtx) return;
    fieldCtx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height);
    fieldCtx.lineWidth = 3;
    fieldCtx.lineCap   = 'round';
    fieldCtx.lineJoin  = 'round';
    for (const s of fieldStrokes) _fStroke(fieldCtx, s);
}

function _fDrawLive() {
    fieldRedraw();
    const W = fieldCanvas.width, H = fieldCanvas.height;
    if (fieldCurrentStroke.length < 2) return;
    fieldCtx.strokeStyle = fieldColor;
    fieldCtx.lineWidth   = 3;
    fieldCtx.lineCap     = 'round';
    fieldCtx.lineJoin    = 'round';
    fieldCtx.beginPath();
    fieldCtx.moveTo(fieldCurrentStroke[0].x * W, fieldCurrentStroke[0].y * H);
    for (let i = 1; i < fieldCurrentStroke.length; i++)
        fieldCtx.lineTo(fieldCurrentStroke[i].x * W, fieldCurrentStroke[i].y * H);
    fieldCtx.stroke();
}

window.fieldUndo  = () => { fieldStrokes.pop(); fieldRedraw(); };
window.fieldErase = () => { fieldStrokes = []; fieldDrawing = false; fieldCurrentStroke = []; fieldRedraw(); };

// ── Robot position tracks (rtrack-tracks v1) ─────────────────────────────────
//
// Produced offline by robot-tracker/ and published as public/tracks/<matchKey>.json.
// Most matches will never have one, so every path here treats "absent" as normal and
// silent — not an error state.
//
// The renderer deliberately mirrors the Field Drawing tab above: same canvas-over-image
// composition, same normalized 0-1 coordinate space keyed to the image's LIVE bounding
// rect (see _fPt), same sizing dance around the img load/complete race (see
// _fSizeCanvas). That tab already solved the fullscreen and resize edge cases; this
// reuses the shape rather than rediscovering them.

// Three shades per alliance so three robots of one colour stay distinguishable,
// matching the R1-R3 / B1-B3 intent of _FC above.
const _TRACK_SHADES = {
    red:  ['#ef4444', '#f87171', '#b91c1c'],
    blue: ['#3b82f6', '#60a5fa', '#1d4ed8'],
};
window.trackColourFor = function (robot, i) {
    const set = _TRACK_SHADES[robot.alliance] || ['#9aa0a6'];
    return set[((robot.station ? robot.station - 1 : i) % set.length + set.length) % set.length];
};

// Field metres -> normalized 0-1 on the field image. The mapping ships INSIDE the track
// file (fieldRectPx / pxPerMeter / imageSize) precisely so the app never needs anything
// from robot-tracker/calib/, which is not deployed.
// The field PNG is drawn with y=0 at the BOTTOM. When the camera sat behind the y=FW
// touchline instead -- which is the case on 2026necmp1 -- everything in the plot is
// upside-down relative to what anyone watching the video saw, and reading a route means
// mentally inverting it.
//
// `cameraSide` comes from which half of the field the camera CANNOT fully see: a side
// camera loses its own corners to the edge of its field of view, not the far ones.
// See project.visible_region for the two measurements that establish it.
//
// A 180-DEGREE ROTATION, NOT A VERTICAL FLIP. Flipping y alone puts the camera's side
// at the bottom but mirrors left and right, so a robot that went right on screen goes
// left in the plot -- worse than leaving it alone, because it looks correct. Rotating
// maps (x,y) -> (FL-x, FW-y) and preserves handedness as seen from behind the camera.
//
// `cameraSide` is derived per calibration (see project.visible_region); which end a
// camera sits at is a property of the venue, not of the sport. Null means unknown, and
// unknown must draw as-is rather than guess.
// ROTATE THE WHOLE COMPOSITE, NOT THE COORDINATES. Transforming field metres before
// projection was the first attempt and it is wrong: the PNG depicts fixed red and blue
// ends, so rotating only the points draws every robot at the opposite alliance's end of
// a field that did not move. The image and the overlay have to turn together, which is
// one CSS transform on the element that contains both -- and the canvas needs no change
// at all. It is also why nothing here draws text: rotated labels would be upside down,
// and the team names live in chips outside the canvas.
function _trackFlip(doc) {
    return doc?.field?.cameraSide === 'high-y';
}
// Applied to the wrapper holding the field <img> and its <canvas>.
function applyFieldOrientation(el, doc) {
    if (!el) return;
    el.style.transform = _trackFlip(doc) ? 'rotate(180deg)' : '';
}
function _trackNorm(doc) {
    const f = doc.field, R = f.fieldRectPx, ppm = f.pxPerMeter;
    const [W, H] = f.imageSize;
    // image +y maps to field -y, hence (y1 - Y) rather than (y0 + Y)
    return (x, y) => [(R.x0 + x * ppm) / W, (R.y1 - y * ppm) / H];
}

// Dexie first, network second. A miss is quiet and returns null.
// Pull one match's routes from the relay and cache them. Same validation as the git
// path -- a doc that fails it is not cached and not returned, so a malformed relay entry
// degrades to the git copy instead of poisoning IndexedDB.
async function _fetchRelayTracks(matchKey, relayAt) {
    const url = relayUrl();
    if (!url) return null;
    try {
        const r = await fetch(`${url}/tracks/${encodeURIComponent(matchKey)}`,
                              { cache: 'no-store' });
        if (!r.ok) return null;
        const doc = await r.json();
        if (doc?.schemaVersion !== 1 || !Array.isArray(doc.robots)) return null;
        doc.key = matchKey;
        doc.eventKey = doc.match?.eventKey || matchKey.split('_')[0];
        doc._relayAt = relayAt;
        try {
            await db.matchTracks.put(doc);
            routesRenderedFor = null;
        } catch {}
        return doc;
    } catch { return null; }
}

async function loadMatchTracks(matchKey) {
    if (!matchKey) return null;

    // REVALIDATE, DO NOT JUST CACHE. This used to return any cached doc outright, which
    // was correct while a match was exported once and never again. It is wrong now: the
    // relay watcher republishes a match every time its curation improves, and a browser
    // that had cached the first version would show those stale routes forever -- across
    // reloads, because Dexie persists. The symptom is silent and looks like the
    // pipeline failing to publish.
    //
    // The manifest carries `exportedAt` per match and the doc carries the identical
    // stamp at generator.createdAt, so "is my copy current" is one string compare with
    // no extra request. A cached doc with no stamp predates this and is refetched once.
    // THE RELAY IS THE LIVE PATH AND WINS WHEN IT HAS THE MATCH. rtrack.export posts
    // routes there as it publishes, so a match reaches the app the moment the watcher
    // finishes it -- no commit, no deploy. git stays the DURABLE path: the relay holds
    // a week, past events and their archives live in public/tracks/ forever.
    //
    // Revalidated on the relay's own push timestamp rather than generator.createdAt,
    // because those answer different questions: createdAt is when the export was built,
    // `at` is when this copy was posted. Re-pushing an unchanged export still means the
    // app should take it.
    let relayAt = 0;
    try { relayAt = (await relayTracksIndex()).get(matchKey) || 0; } catch {}
    if (relayAt) {
        try {
            const hit = await db.matchTracks.get(matchKey);
            if (hit && hit._relayAt === relayAt) return hit;
        } catch {}
        const doc = await _fetchRelayTracks(matchKey, relayAt);
        if (doc) return doc;
        // Fetch failed despite the index listing it: fall through to git rather than
        // showing nothing. An expired entry between index and GET does exactly this.
    }

    let want = null;
    try {
        const man = await loadTracksManifest();
        want = (man.matches || []).find(m => m.key === matchKey)?.exportedAt || null;
    } catch { /* no manifest: fall back to trusting the cache */ }
    try {
        const hit = await db.matchTracks.get(matchKey);
        if (hit && (!want || hit.generator?.createdAt === want)) return hit;
    } catch { /* table may not exist on a stale schema; fall through to network */ }

    const url = `${import.meta.env.BASE_URL}tracks/${matchKey}.json`;
    try {
        // HEAD-probe first, mirroring findArchiveUrl: a 404 page served as HTML would
        // otherwise parse-fail noisily on every match that has no tracks.
        const head = await fetch(url, { method: 'HEAD' });
        const ct = head.headers.get('content-type') || '';
        if (!head.ok || !ct.includes('json')) return null;
        const resp = await fetch(url);
        if (!resp.ok) return null;
        const doc = await resp.json();
        if (doc?.schemaVersion !== 1 || !Array.isArray(doc.robots)) return null;
        doc.key = matchKey;
        doc.eventKey = doc.match?.eventKey || matchKey.split('_')[0];
        try {
            await db.matchTracks.put(doc);
            // A newly cached match changes what the team Routes tab should show, and
            // that tab memoises by team number alone. Without this, opening a match and
            // then the Routes tab shows the pre-cache result.
            routesRenderedFor = null;
        } catch {}
        return doc;
    } catch { return null; }
}
window.loadMatchTracks = loadMatchTracks;

// Size a track canvas against its backing image. Same guards as _fSizeCanvas: the rect
// can be zero while hidden, and the image may or may not have loaded yet.
// ── full-screen routes ─────────────────────────────────────────────────────────
// A route plot inside the match modal is ~600 px wide on a laptop and less on a phone,
// which is fine for "did they cross the field" and useless for anything finer -- two
// robots working the same corner are a few pixels apart there. This opens the same
// render at window size, reusing renderFieldRoutes rather than growing a second one,
// so every fix (visibility shading, camera orientation, auto clipping) applies to both.
let _fsRoutes = null;

function closeRoutesFull() {
    const el = document.getElementById('routesFull');
    if (el) el.remove();
    window.removeEventListener('resize', _fsRoutes || (() => {}));
    _fsRoutes = null;
    document.body.style.overflow = '';
}
window.closeRoutesFull = closeRoutesFull;

window.openRoutesFull = function (doc, opts = {}) {
    if (!doc) return;
    closeRoutesFull();
    const title = doc.match?.key || doc.key || 'routes';

    // STATE IS SEEDED FROM THE CALLER, then owned here. Full screen is not a bigger
    // copy of the small picture -- it is where the picture is actually read -- so it
    // needs the same controls. They are rebuilt rather than borrowed from the match
    // modal because the team Routes tab has no controls to borrow, and one path
    // serving both entry points is worth the few lines it repeats.
    const all = (doc.robots || []).map(r => String(r.team));
    let shown = new Set(opts.teams ? [...opts.teams].map(String) : all);
    let autoOnly = opts.tMax != null;
    let arrows = opts.arrows !== false;
    let tMin = Infinity, tMax = -Infinity;
    for (const r of doc.robots || []) for (const sm of r.samples || []) {
        if (sm.t < tMin) tMin = sm.t;
        if (sm.t > tMax) tMax = sm.t;
    }
    if (!isFinite(tMin)) { tMin = 0; tMax = 1; }
    let tNow = (opts.tNow != null) ? opts.tNow : tMax;
    let playing = false, raf = 0;
    const autoEnd = autoEndOf(doc);
    const BASE = import.meta.env.BASE_URL;

    const el = document.createElement('div');
    el.id = 'routesFull';
    el.style.cssText = 'position:fixed; inset:0; z-index:9000; background:#0b1220;'
        + 'display:flex; flex-direction:column; padding:10px; gap:8px;';
    el.innerHTML = [
        '<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">',
        '  <b style="font-size:15px;">' + title + '</b>',
        '  <span id="rfNote" style="color:#64748b; font-size:12px;"></span>',
        '  <span style="flex:1;"></span>',
        '  <button id="rfArrows" style="padding:5px 11px; font-size:12px; border-radius:6px; cursor:pointer;"></button>',
        '  <button id="rfAuto" style="padding:5px 11px; font-size:12px; border-radius:6px; cursor:pointer;"></button>',
        '  <button onclick="closeRoutesFull()" style="padding:6px 14px; border-radius:8px;',
        '          border:1px solid #334155; background:#1e293b; color:#e2e8f0; cursor:pointer;">Close</button>',
        '</div>',
        '<div id="rfTeams" style="display:flex; gap:6px; flex-wrap:wrap;"></div>',
        '<div style="position:relative; flex:1; min-height:0; display:flex;',
        '            align-items:center; justify-content:center;">',
        '  <div id="rfInner" style="position:relative; max-width:100%; max-height:100%;">',
        '    <img id="rfImg" src="' + BASE + doc.field.imageRef + '" alt="field"',
        '         style="display:block; max-width:100%; max-height:100%; width:auto; height:auto;">',
        '    <canvas id="rfCv" style="position:absolute; inset:0; width:100%; height:100%;"></canvas>',
        '  </div>',
        '</div>',
        '<div style="display:flex; gap:10px; align-items:center;">',
        '  <button id="rfPlay" style="padding:5px 12px; font-size:13px; border-radius:6px; cursor:pointer;',
        '          border:1px solid #334155; background:transparent; color:#94a3b8;">&#9654;</button>',
        '  <input type="range" id="rfScrub" min="0" max="1000" value="1000" style="flex:1;">',
        '  <span id="rfClock" style="font-variant-numeric:tabular-nums; font-size:12px;',
        '        color:#94a3b8; min-width:70px; text-align:right;">full</span>',
        '</div>',
    ].join('\n');
    document.body.appendChild(el);
    document.body.style.overflow = 'hidden';

    const img = el.querySelector('#rfImg'), cv = el.querySelector('#rfCv');
    applyFieldOrientation(el.querySelector('#rfInner'), doc);
    const vf = doc.field?.visibleFrac;
    el.querySelector('#rfNote').textContent =
        (typeof vf === 'number') ? (Math.round(100 * vf) + '% of the field visible') : '';

    const paint = () => {
        if (!_trackSizeCanvas(img, cv)) return;
        renderFieldRoutes(cv, doc, {
            teams: shown, tNow, trailOnly: tNow < tMax, dots: tNow < tMax,
            tMax: autoOnly ? autoEnd : null, arrows,
        });
    };
    const style = (b, on) => {
        b.style.border = '1px solid ' + (on ? '#3b82f6' : '#334155');
        b.style.background = on ? '#1e3a5f' : 'transparent';
        b.style.color = on ? '#60a5fa' : '#94a3b8';
    };
    const setT = (t) => {
        tNow = Math.max(tMin, Math.min(tMax, t));
        el.querySelector('#rfScrub').value =
            String(Math.round((tNow - tMin) / Math.max(tMax - tMin, 1e-6) * 1000));
        el.querySelector('#rfClock').textContent =
            (tNow >= tMax) ? 'full' : (tNow.toFixed(1) + 's');
        paint();
    };

    el.querySelector('#rfTeams').innerHTML = (doc.robots || []).map((r, k) => {
        const t = String(r.team), c = window.trackColourFor(r, k);
        return '<button data-team="' + t + '" style="padding:4px 10px; font-size:12px;'
             + ' border-radius:999px; cursor:pointer; border:1px solid ' + c
             + '; background:transparent; color:' + c + ';">' + t + '</button>';
    }).join('');
    const syncTeams = () => el.querySelectorAll('#rfTeams button').forEach(
        o => { o.style.opacity = shown.has(o.dataset.team) ? '1' : '0.32'; });
    el.querySelectorAll('#rfTeams button').forEach(b => {
        b.onclick = () => {
            const t = b.dataset.team;
            if (shown.has(t)) shown.delete(t); else shown.add(t);
            // Hiding everything leaves a blank field and reads as broken; treat the
            // last deselection as "show all again", which is what the click meant.
            if (!shown.size) shown = new Set(all);
            syncTeams(); paint();
        };
    });
    syncTeams();

    const arrowsBtn = el.querySelector('#rfArrows');
    const autoBtn = el.querySelector('#rfAuto');
    const syncBtns = () => {
        arrowsBtn.textContent = 'Direction ' + (arrows ? 'on' : 'off');
        autoBtn.textContent = autoOnly ? ('Auto only (' + autoEnd.toFixed(0) + 's)') : 'Whole match';
        style(arrowsBtn, arrows); style(autoBtn, autoOnly);
    };
    syncBtns();
    arrowsBtn.onclick = () => { arrows = !arrows; syncBtns(); paint(); };
    autoBtn.onclick = () => {
        autoOnly = !autoOnly; syncBtns();
        if (autoOnly && tNow > autoEnd) setT(autoEnd); else paint();
    };

    el.querySelector('#rfScrub').oninput = (e) =>
        setT(tMin + (e.target.value / 1000) * (tMax - tMin));
    const playBtn = el.querySelector('#rfPlay');
    const step = (last) => {
        if (!playing) return;
        const now = performance.now();
        setT(tNow + (now - last) / 1000);
        if (tNow >= tMax) { playing = false; playBtn.innerHTML = '&#9654;'; return; }
        raf = requestAnimationFrame(() => step(now));
    };
    playBtn.onclick = () => {
        playing = !playing;
        playBtn.innerHTML = playing ? '&#10073;&#10073;' : '&#9654;';
        if (playing) {
            if (tNow >= tMax) setT(tMin);
            raf = requestAnimationFrame(() => step(performance.now()));
        } else cancelAnimationFrame(raf);
    };

    if (img.complete && img.naturalWidth) setT(tNow);
    else img.addEventListener('load', () => setT(tNow), { once: true });
    _fsRoutes = paint;
    window.addEventListener('resize', paint);
    const onKey = (e) => {
        if (e.key === 'Escape') {
            closeRoutesFull();
            document.removeEventListener('keydown', onKey);
        }
    };
    document.addEventListener('keydown', onKey);
};

function _trackSizeCanvas(img, canvas) {
    const r = img.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    canvas.width  = Math.round(r.width);
    canvas.height = Math.round(r.height);
    return true;
}

/**
 * Draw routes onto a canvas overlaying a field image.
 * opts: { teams:Set|null (null = all), tNow:number|null, trailOnly:bool, dots:bool }
 */
// Auto ends at this many seconds after auto start. Exports carry it in
// sampling.phases.autoEndT; older files predate that field, so fall back to the 2026
// rulebook value rather than drawing nothing.
const AUTO_END_DEFAULT = 20;
function autoEndOf(doc) {
    // `phases` is a TOP-LEVEL key of the export, not a child of `sampling`. This read
    // the nested path for a while and therefore always got undefined, silently falling
    // back to the constant below -- which happens to equal C.AUTO_S, so the auto toggle
    // looked correct while ignoring the file entirely. Both paths are accepted now: the
    // real one first, the mistaken one after, so nothing that may have been written in
    // the wrong shape is orphaned.
    const v = doc?.phases?.autoEndT ?? doc?.sampling?.phases?.autoEndT;
    return (typeof v === 'number' && v > 0) ? v : AUTO_END_DEFAULT;
}

function renderFieldRoutes(canvas, doc, opts = {}) {
    if (!canvas || !doc) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!W || !H) return;

    const N = _trackNorm(doc);
    const only = opts.teams || null;
    const tNow = (opts.tNow === undefined || opts.tNow === null) ? null : opts.tNow;
    const trailOnly = !!opts.trailOnly;
    const dots = opts.dots !== false;
    // Sample times are relative to AUTO START, so the autonomous period is simply
    // t <= autoEndT. Clipping here rather than filtering the doc keeps one source of
    // truth for the routes and lets the caller toggle without re-fetching.
    const tMax = (opts.tMax === undefined || opts.tMax === null) ? null : opts.tMax;
    // Direction is on by default: it is information the plot otherwise loses entirely.
    const arrows = opts.arrows !== false;

    // BLIND AREA FIRST, so routes draw on top of it and stay legible.
    //
    // A route that stops dead at the far-left corner looks like the tracker failing and
    // is nothing of the sort: on the 2026necmp1 camera 13.5% of the field is simply off
    // frame, almost all of it the two far corners. Without this the missing data is
    // indistinguishable from lost data, and a reader concludes something about a robot
    // that was never observable.
    //
    // Drawn as the field MINUS the visible polygon, using the even-odd fill rule: the
    // outer rectangle and the polygon together leave only the unseen part filled. A doc
    // with no polygon (older export, or a camera whose visibility could not be computed)
    // draws nothing at all -- unknown visibility must never render as "all visible".
    const vis = doc.field?.visiblePolyM;
    if (opts.visibility !== false && Array.isArray(vis) && vis.length >= 3) {
        const [FL, FW] = doc.field.sizeM;
        const poly = (arr) => arr.forEach(([x, y], k) => {
            const [nx, ny] = N(x, y);
            k ? ctx.lineTo(nx * W, ny * H) : ctx.moveTo(nx * W, ny * H);
        });
        ctx.save();
        // CLIP to the unseen area, then paint freely inside it. Building the region as
        // field-rect + visible-polygon under the even-odd rule leaves exactly the blind
        // part selected; clipping rather than filling lets the hatch below be drawn as
        // plain lines across the whole canvas without any per-wedge geometry.
        ctx.beginPath();
        poly([[0, 0], [FL, 0], [FL, FW], [0, FW]]);
        ctx.closePath();
        poly(vis);
        ctx.closePath();
        ctx.clip('evenodd');

        // LIGHTEN, DO NOT DARKEN. The obvious treatment -- a dark wash over the unseen
        // area -- barely registers, because 2026-field.png is itself dark: darkening a
        // dark image cannot create a boundary. Rendering the three candidates over the
        // real field art settled it; only lifting the region off the background reads
        // as a mask at a glance.
        ctx.fillStyle = 'rgba(170,180,190,0.22)';
        ctx.fillRect(0, 0, W, H);
        // Diagonal hatch over the wash, because a plain lighter patch reads as
        // EMPHASIS -- the opposite of the meaning. Hatching is not a texture the field
        // can produce accidentally, so it says "no data" rather than "look here", and
        // it survives whatever is underneath.
        ctx.strokeStyle = 'rgba(220,228,235,0.55)';
        ctx.lineWidth = 1;
        const step = Math.min(14, Math.max(6, Math.round(W / 110)));
        ctx.beginPath();
        for (let d = -H; d < W + H; d += step) {
            ctx.moveTo(d, 0);
            ctx.lineTo(d + H, H);
        }
        ctx.stroke();
        ctx.restore();

        // Boundary last and unclipped, so the edge stays crisp rather than being
        // half-covered by the hatch that ends on it.
        ctx.save();
        ctx.beginPath();
        poly(vis);
        ctx.closePath();
        ctx.strokeStyle = 'rgba(226,232,240,0.75)';
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
    }

    // OCCLUDERS — structures robots pass BEHIND, as floor shadows in field metres.
    //
    // Deliberately a different treatment from the blind-area hatch above, because they
    // mean different things and a reader must not confuse them. Outside camera coverage
    // is "we never saw this part of the field". Behind a hub is "we saw it, and a robot
    // standing here would be hidden by a structure" — the route can legitimately pass
    // through and reappear. Same idea, opposite hatch slope and a warm hue.
    //
    // CLIPPED TO THE FIELD, and that is not cosmetic. These are the projection of a
    // silhouette onto the floor plane, and a silhouette's upper edge lies well beyond
    // the structure's base — on 2026mawor the two hub shadows reach y = -3.5 m and
    // -4.6 m, several metres off the near end. Drawing them unclipped would paint over
    // the margin and misrepresent how much of the field is affected.
    const occ = doc.field?.occluderPolysM;
    if (opts.occluders !== false && Array.isArray(occ) && occ.length) {
        const [OFL, OFW] = doc.field.sizeM;
        const opoly = (arr) => arr.forEach(([x, y], k) => {
            const [nx, ny] = N(x, y);
            k ? ctx.lineTo(nx * W, ny * H) : ctx.moveTo(nx * W, ny * H);
        });
        const step = Math.min(13, Math.max(6, Math.round(W / 120)));
        occ.forEach(o => {
            const p = o?.poly;
            if (!Array.isArray(p) || p.length < 3) return;
            ctx.save();
            ctx.beginPath(); opoly([[0, 0], [OFL, 0], [OFL, OFW], [0, OFW]]);
            ctx.closePath(); ctx.clip();
            ctx.beginPath(); opoly(p); ctx.closePath(); ctx.clip();
            ctx.fillStyle = 'rgba(240,168,51,0.18)';
            ctx.fillRect(0, 0, W, H);
            // Counter-diagonal: the blind hatch runs the other way, so the two are
            // distinguishable even where they abut.
            ctx.strokeStyle = 'rgba(240,168,51,0.50)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let d = -H; d < W + H; d += step) {
                ctx.moveTo(W - d, 0);
                ctx.lineTo(W - d - H, H);
            }
            ctx.stroke();
            ctx.restore();

            // Boundary unclipped by the polygon but still inside the field, so the edge
            // stays crisp rather than being half-covered by its own hatch.
            ctx.save();
            ctx.beginPath(); opoly([[0, 0], [OFL, 0], [OFL, OFW], [0, OFW]]);
            ctx.closePath(); ctx.clip();
            ctx.beginPath(); opoly(p); ctx.closePath();
            ctx.strokeStyle = 'rgba(240,168,51,0.85)';
            ctx.setLineDash([4, 3]);
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
        });
    }

    doc.robots.forEach((r, i) => {
        if (only && !only.has(String(r.team))) return;
        let pts = r.samples || [];
        if (tMax !== null) pts = pts.filter(s => s.t <= tMax);
        if (!pts.length) return;
        const col = window.trackColourFor(r, i);
        const gaps = r.gaps || [];
        // Break the line across recorded gaps. Drawing straight through a stretch
        // nobody observed invents a route that reads as real.
        const spans = (a, b) => gaps.some(g => g.tStart <= a.t && b.t <= g.tEnd);

        ctx.lineWidth = 2;
        ctx.strokeStyle = col;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        let started = false;
        for (let k = 0; k < pts.length; k++) {
            const p = pts[k];
            if (trailOnly && tNow !== null && p.t > tNow) break;
            const [nx, ny] = N(p.x, p.y);
            const X = nx * W, Y = ny * H;
            if (!started || (k && spans(pts[k - 1], p))) { ctx.moveTo(X, Y); started = true; }
            else ctx.lineTo(X, Y);
        }
        if (started) ctx.stroke();

        // DIRECTION ARROWS. A route is a closed scribble without them: the same loop
        // driven clockwise and anticlockwise means different things about where a robot
        // collected and where it scored, and the polyline alone cannot say which.
        //
        // Spaced by DISTANCE ALONG THE PATH rather than by sample index, so a robot
        // sitting still does not pile up arrowheads on one spot while a fast traverse
        // gets none. Skipped across gaps for the same reason the line is.
        if (arrows) {
            const step = Math.max(34, Math.min(W, H) / 9);
            let acc = step * 0.6, px = null, py = null;
            ctx.globalAlpha = 0.95;
            ctx.fillStyle = col;
            for (let k = 0; k < pts.length; k++) {
                const p = pts[k];
                if (trailOnly && tNow !== null && p.t > tNow) break;
                const [nx, ny] = N(p.x, p.y);
                const X = nx * W, Y = ny * H;
                const broke = k && spans(pts[k - 1], p);
                if (px !== null && !broke) {
                    const dx = X - px, dy = Y - py;
                    const d = Math.hypot(dx, dy);
                    acc += d;
                    // Only where the robot is actually travelling: a heading computed
                    // from a sub-pixel step is noise pointing nowhere in particular.
                    if (acc >= step && d > 1.2) {
                        acc = 0;
                        const a = Math.atan2(dy, dx), L = 7, Wd = 4.2;
                        ctx.beginPath();
                        ctx.moveTo(X, Y);
                        ctx.lineTo(X - L * Math.cos(a) + Wd * Math.sin(a),
                                   Y - L * Math.sin(a) - Wd * Math.cos(a));
                        ctx.lineTo(X - L * Math.cos(a) - Wd * Math.sin(a),
                                   Y - L * Math.sin(a) + Wd * Math.cos(a));
                        ctx.closePath();
                        ctx.fill();
                    }
                } else if (broke) acc = step * 0.6;
                px = X; py = Y;
            }
            ctx.globalAlpha = 1;
        }

        if (dots && tNow !== null) {
            let cur = null;
            for (const p of pts) { if (p.t <= tNow) cur = p; else break; }
            if (cur) {
                const [nx, ny] = N(cur.x, cur.y);
                ctx.globalAlpha = 1;
                ctx.beginPath();
                ctx.arc(nx * W, ny * H, 6, 0, Math.PI * 2);
                ctx.fillStyle = col; ctx.fill();
                ctx.lineWidth = 2; ctx.strokeStyle = '#0f1115'; ctx.stroke();
            }
        }
        ctx.globalAlpha = 1;
    });
}
window.renderFieldRoutes = renderFieldRoutes;

// ── Pick List ────────────────────────────────────────────────────────────────

function loadPickOrder() {
    try { return JSON.parse(localStorage.getItem('pickListOrder')) || []; }
    catch { return []; }
}

function savePickOrder() {
    const order = [...document.querySelectorAll('#pickListBody tr')]
        .map(r => r.dataset.separator ? '---separator---' : (r.dataset.team || null))
        .filter(x => x !== null);
    localStorage.setItem('pickListOrder', JSON.stringify(order));
}

function refreshPickPositions() {
    let pos = 1;
    document.querySelectorAll('#pickListBody tr').forEach(row => {
        if (row.dataset.separator) return;
        const el = row.querySelector('.pick-pos');
        if (el) el.textContent = pos++;
    });
}

window.resetPickList = async function () {
    localStorage.removeItem('pickListOrder');
    await renderPickList();
};

window.exportPickList = function () {
    const order = loadPickOrder();
    if (!order.length) { alert('No pick list to export.'); return; }
    const text = order.map(t => t === '---separator---' ? '---' : t).join('\n');
    navigator.clipboard.writeText(text)
        .then(() => alert('Pick list copied to clipboard.'))
        .catch(() => alert('Copy failed — check clipboard permissions.'));
};

window.showImportPickList = function () {
    const modal = document.getElementById('pickImportModal');
    if (!modal) return;
    document.getElementById('pickImportText').value = '';
    modal.style.display = 'flex';
};

window.closeImportPickList = function () {
    const modal = document.getElementById('pickImportModal');
    if (modal) modal.style.display = 'none';
};

window.confirmImportPickList = async function () {
    const text = document.getElementById('pickImportText').value.trim();
    if (!text) return;
    const order = text.split(/\r?\n/)
        .map(l => l.trim())
        .filter(l => l)
        .map(l => l === '---' ? '---separator---' : l);
    const teams = order.filter(t => t !== '---separator---');
    if (!teams.length) { alert('No team numbers found.'); return; }
    localStorage.setItem('pickListOrder', JSON.stringify(order));
    window.closeImportPickList();
    await renderPickList();
};

window.dnpTeam = function (teamNumber) {
    const tbody = document.getElementById('pickListBody');
    if (!tbody) return;
    const row = tbody.querySelector(`tr[data-team="${teamNumber}"]`);
    if (!row) return;
    tbody.appendChild(row);       // move to bottom of list
    refreshPickPositions();
    savePickOrder();
};

async function renderPickList() {
    const table = document.getElementById('pickListTable');
    const tbody = document.getElementById('pickListBody');
    const statusEl = document.getElementById('pickListStatus');
    if (!table || !tbody) return;

    const [allTeams, allTBATeams, allMatches] = await Promise.all([
        db.teams.toArray(), db.tbaTeams.toArray(), db.matches.toArray()
    ]);

    if (!allTeams.length) {
        if (statusEl) statusEl.textContent = 'No team data — sync team list/history or Statbotics Live first.';
        table.style.display = 'none';
        return;
    }

    // effOPR — mirrors renderAtAGlance
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
    const tbaTeamMap = Object.fromEntries(allTBATeams.map(t => [t.teamNumber, t]));
    let globalOPRMap = null;
    if (globalIgnored.size > 0) {
        const teamNums = allTBATeams.map(t => t.teamNumber);
        const activePlayed = allMatches.filter(m =>
            (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !globalIgnored.has(m.key)
        );
        const recomputed = computeLocalOPR(activePlayed, teamNums);
        if (recomputed) globalOPRMap = Object.fromEntries(teamNums.map((n, i) => [n, recomputed[i]]));
    }
    const effOPR = tba => {
        if (!tba) return null;
        const keys = getTeamIgnoredKeys(tba);
        if (keys.length > 0 && tba.adjustedOPR != null && keys.some(k => !globalIgnored.has(k)))
            return tba.adjustedOPR;
        if (globalOPRMap) return globalOPRMap[tba.teamNumber] ?? tba.opr ?? null;
        return tba.opr ?? null;
    };

    // RP totals
    const rpMap = {};
    const playedMatches = allMatches.filter(m => (m.redScore ?? -1) >= 0 && (!m.compLevel || m.compLevel === 'qm'));
    for (const m of playedMatches) {
        const redWon = m.redScore > m.blueScore, blueWon = m.blueScore > m.redScore, tie = m.redScore === m.blueScore;
        const bonusRP = bd => bd ? ((bd.energizedAchieved ? 1 : 0) + (bd.superchargedAchieved ? 1 : 0) + (bd.traversalAchieved ? 1 : 0)) : 0;
        const redRP = m.redBreakdown?.rp ?? ((redWon ? 3 : tie ? 1 : 0) + bonusRP(m.redBreakdown));
        const blueRP = m.blueBreakdown?.rp ?? ((blueWon ? 3 : tie ? 1 : 0) + bonusRP(m.blueBreakdown));
        for (const team of (m.red || [])) {
            if (!rpMap[team]) rpMap[team] = { rp: 0, wins: 0, ties: 0, losses: 0, totalScore: 0, matches: 0 };
            rpMap[team].rp += redRP;
            rpMap[team].totalScore += m.redScore;
            rpMap[team].matches++;
            if (redWon) rpMap[team].wins++; else if (tie) rpMap[team].ties++; else rpMap[team].losses++;
        }
        for (const team of (m.blue || [])) {
            if (!rpMap[team]) rpMap[team] = { rp: 0, wins: 0, ties: 0, losses: 0, totalScore: 0, matches: 0 };
            rpMap[team].rp += blueRP;
            rpMap[team].totalScore += m.blueScore;
            rpMap[team].matches++;
            if (blueWon) rpMap[team].wins++; else if (tie) rpMap[team].ties++; else rpMap[team].losses++;
        }
    }

    // Scouting EPA
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const scoutEPAMap = {};
    if (eventKey) {
        const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
        if (rawStr) {
            const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
            const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
            if (processed?.config?.computeEPABreakdown) {
                const { config, byTeam } = processed;
                for (const [tn, rawRows] of Object.entries(byTeam)) {
                    const tbaEntry = tbaTeamMap[parseInt(tn)];
                    const scoutIgnoreKeys = tbaEntry?.scoutingIgnoreActive ? getTeamIgnoredKeys(tbaEntry) : [];
                    let { rows: deduped } = deduplicateTeamRows(rawRows);
                    let ignoredMatchNums = new Set();
                    if (scoutIgnoreKeys.length > 0) {
                        ignoredMatchNums = new Set(allMatches.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber));
                        deduped = deduped.filter(r => !ignoredMatchNums.has(r.matchNumber));
                    }
                    const rawStats = config.aggregateTeam(deduped);
                    const fusedResult = fusedCache?.teams?.[tn];
                    const effectiveFused = (fusedResult?.available && ignoredMatchNums.size > 0)
                        ? refilteredFusedStats(fusedResult, ignoredMatchNums) : fusedResult;
                    const isFused = !!(effectiveFused?.available && config.computeFusedEPABreakdown);
                    const breakdown = isFused
                        ? config.computeFusedEPABreakdown(effectiveFused.stats)
                        : config.computeEPABreakdown(rawStats);
                    scoutEPAMap[tn] = { total: breakdown.total, isFused, isAdj: ignoredMatchNums.size > 0 };
                }
            }
        }
    }

    // Build rows
    let rows = allTeams.map(team => {
        const tn = parseInt(team.teamNumber);
        const tba = tbaTeamMap[tn];
        const rp = rpMap[String(tn)] || { rp: 0, wins: 0, ties: 0, losses: 0 };
        const opr = effOPR(tba);
        const analysis = team.analysis || {};
        const hasCeil = analysis.ceiling != null && analysis.ceiling !== '—';
        const epaVal = hasCeil ? parseFloat(analysis.ceiling) : (team.currentEPA || 0);
        const hasLOO = getTeamIgnoredKeys(tba).some(k => !globalIgnored.has(k)) && tba?.adjustedOPR != null;
        const hasAdj = !hasLOO && globalOPRMap != null;
        const scoutData = scoutEPAMap[tn];
        return { team, rp, opr, epaVal, hasCeil, hasLOO, hasAdj,
                 scoutEPA: scoutData?.total ?? null, scoutFused: scoutData?.isFused ?? false, scoutAdj: scoutData?.isAdj ?? false };
    });

    // Composite + tier (mirrors renderAtAGlance)
    {
        const pctRank = (arr, val) => {
            const sorted = [...arr].sort((a, b) => b - a);
            const idx = sorted.findIndex(v => v <= val + 0.001);
            return idx < 0 ? 1 : idx / (sorted.length || 1);
        };
        const epaVals = rows.map(r => r.epaVal);
        const oprVals = rows.map(r => r.opr ?? 0);
        const scoutEPAVals = rows.map(r => r.scoutEPA ?? 0);
        const hasAnyOPR = rows.some(r => r.opr != null);
        const hasAnyScout = rows.some(r => r.scoutEPA != null);
        const composite = r => {
            const sources = [pctRank(epaVals, r.epaVal)];
            if (hasAnyOPR)   sources.push(pctRank(oprVals, r.opr ?? 0));
            if (hasAnyScout) sources.push(pctRank(scoutEPAVals, r.scoutEPA ?? 0));
            return sources.reduce((a, b) => a + b, 0) / sources.length;
        };
        rows.forEach(r => { r.composite = composite(r); });
        const tierOrder = [...rows].sort((a, b) => a.composite - b.composite);
        const tierMap = new Map(tierOrder.map((r, i) => [
            r.team.teamNumber, i < 8 ? 'S' : i < 20 ? 'A' : i < 32 ? 'B' : 'C'
        ]));
        rows.forEach(r => { r.tier = tierMap.get(r.team.teamNumber); });
    }

    // Apply saved order; append any new teams at end sorted by composite
    const SEPARATOR = Object.freeze({ separator: true });
    const savedOrder = loadPickOrder();
    if (savedOrder.length) {
        const rowMap = new Map(rows.map(r => [String(r.team.teamNumber), r]));
        const orderedWithSep = [];
        let hasSep = false;
        for (const tn of savedOrder) {
            if (tn === '---separator---') { orderedWithSep.push(SEPARATOR); hasSep = true; }
            else { const r = rowMap.get(tn); if (r) { orderedWithSep.push(r); rowMap.delete(tn); } }
        }
        const rest = [...rowMap.values()].sort((a, b) => a.composite - b.composite);
        rows = [...orderedWithSep, ...rest];
        if (!hasSep) rows.unshift(SEPARATOR);
    } else {
        rows.sort((a, b) => a.composite - b.composite);
        rows.unshift(SEPARATOR);
    }

    const hasOPR = allTBATeams.length > 0;
    const hasRP = playedMatches.length > 0;

    // Sort unranked rows (after separator) by current sort column
    {
        const sepIdx = rows.findIndex(r => r.separator);
        const ranked = sepIdx >= 0 ? rows.slice(0, sepIdx + 1) : [];
        const unranked = sepIdx >= 0 ? rows.slice(sepIdx + 1) : [...rows];
        const getValue = r => {
            switch (pickListSortCol) {
                case 'epa':      return r.epaVal;
                case 'opr':      return r.opr ?? -999;
                case 'scoutEPA': return r.scoutEPA ?? -999;
                default:         return (1 - r.composite) * 100;
            }
        };
        unranked.sort((a, b) => {
            if (pickListSortCol === 'rp') {
                const avgRpA = a.rp.matches > 0 ? a.rp.rp / a.rp.matches : 0;
                const avgRpB = b.rp.matches > 0 ? b.rp.rp / b.rp.matches : 0;
                const rpDiff = avgRpB - avgRpA;
                if (rpDiff !== 0) return rpDiff * pickListSortDir;
                const avgA = a.rp.matches > 0 ? a.rp.totalScore / a.rp.matches : 0;
                const avgB = b.rp.matches > 0 ? b.rp.totalScore / b.rp.matches : 0;
                return (avgB - avgA) * pickListSortDir;
            }
            return (getValue(b) - getValue(a)) * pickListSortDir;
        });
        rows = [...ranked, ...unranked];
    }

    // Rebuild thead with sort arrows
    const arrowFor = col => {
        if (pickListSortCol !== col) return `<span style="opacity:0.3"> ↕</span>`;
        return pickListSortDir === 1 ? ' ↓' : ' ↑';
    };
    const th = (label, col, extra = '') =>
        `<th style="padding:10px 8px;border-bottom:2px solid #334155;color:#94a3b8;text-align:center;cursor:pointer;white-space:nowrap;${extra}" onclick="sortPickListBy('${col}')">${label}${arrowFor(col)}</th>`;
    table.querySelector('thead').innerHTML = `<tr>
        <th style="width:28px;padding:10px 4px;border-bottom:2px solid #334155;"></th>
        <th style="width:56px;padding:10px 8px;border-bottom:2px solid #334155;color:#94a3b8;text-align:center;white-space:nowrap;">#</th>
        <th style="padding:10px 8px;border-bottom:2px solid #334155;color:#94a3b8;text-align:left;white-space:nowrap;">Team</th>
        <th style="padding:10px 8px;border-bottom:2px solid #334155;color:#94a3b8;text-align:left;white-space:nowrap;">Name</th>
        ${th('Score', 'composite')}
        ${th('RP Rank', 'rp')}
        ${th('<img src="./statbotics.ico" height="18" style="vertical-align:middle;opacity:0.85;" title="EPA / Ceiling (Statbotics)">', 'epa')}
        ${th('<img src="./tba.png" height="18" style="vertical-align:middle;opacity:0.85;" title="OPR (TBA)">', 'opr')}
        ${th('<img src="./sheets.png" height="18" style="vertical-align:middle;opacity:0.85;" title="Scouting EPA (Google Sheets)">', 'scoutEPA')}
        <th style="width:60px;padding:10px 8px;border-bottom:2px solid #334155;"></th>
    </tr>`;

    // RP rank (1 = most RP), avg match score as tiebreaker
    const rpRankMap = {};
    if (hasRP) {
        Object.entries(rpMap)
            .sort(([, a], [, b]) => {
                if (b.rp !== a.rp) return b.rp - a.rp;
                const avgA = a.matches > 0 ? a.totalScore / a.matches : 0;
                const avgB = b.matches > 0 ? b.totalScore / b.matches : 0;
                return avgB - avgA;
            })
            .forEach(([tn,], i) => { rpRankMap[tn] = i + 1; });
    }

    const TIER = TIER_STYLE;

    table.style.display = 'table';
    let pickPos = 1;
    tbody.innerHTML = rows.map(r => {
        if (r.separator) {
            return `<tr data-separator="true" style="user-select:none;">
                <td colspan="10" class="drag-handle"
                    style="padding:9px 20px;border-top:2px dashed #334155;border-bottom:2px dashed #334155;background:#080d16;text-align:center;cursor:grab;touch-action:none;color:#475569;font-size:0.8rem;font-weight:700;letter-spacing:0.06em;">
                    ⠿ &nbsp; drag to reposition &nbsp;·&nbsp; unranked below &nbsp; ⠿
                </td>
            </tr>`;
        }
        const rowPos = pickPos++;
        const { team, rp, opr, epaVal, hasCeil, hasLOO, hasAdj, scoutEPA, scoutFused, scoutAdj } = r;
        const ts = TIER[r.tier];
        const record = hasRP ? `${rp.wins}–${rp.losses}${rp.ties ? `–${rp.ties}` : ''}` : null;
        const compStr = ((1 - r.composite) * 100).toFixed(1);
        const ceilBadge = hasCeil ? `<span style="color:#4ade80;font-size:0.65em;font-weight:600;margin-left:3px;">CEIL</span>` : '';
        const oprBadge = (hasLOO || hasAdj)
            ? `<span style="color:${hasLOO ? '#fbbf24' : '#f97316'};font-size:0.65em;font-weight:600;margin-left:3px;">ADJ</span>` : '';
        const scoutStr = scoutEPA != null ? scoutEPA.toFixed(1) : '—';
        const fusedBadge = scoutFused ? `<span style="color:#818cf8;font-size:0.65em;font-weight:600;margin-left:3px;">F</span>` : '';
        const scoutAdjBadge = scoutAdj ? `<span style="color:#fbbf24;font-size:0.65em;font-weight:600;margin-left:3px;">ADJ</span>` : '';
        const td = (content, center = true) =>
            `<td style="padding:13px 10px;border-bottom:1px solid #1e293b;${center ? ' text-align:center;' : ''}">${content}</td>`;

        return `<tr data-team="${team.teamNumber}" style="background:${ts.bg};">
            <td class="drag-handle" style="padding:10px 6px;border-bottom:1px solid #1e293b;text-align:center;cursor:grab;touch-action:none;box-shadow:inset 3px 0 0 ${ts.color};">
                <span style="color:#475569;font-size:1.2em;line-height:1;">⠿</span>
            </td>
            <td style="padding:10px 8px;border-bottom:1px solid #1e293b;text-align:center;">
                <span class="pick-pos" style="color:#64748b;font-weight:700;">${rowPos}</span>
            </td>
            <td style="padding:13px 10px;border-bottom:1px solid #1e293b;cursor:pointer;white-space:nowrap;" onclick="viewTeamDetail(${team.teamNumber})">
                <strong style="color:#f8fafc;">${team.teamNumber}</strong>${ownStar(team.teamNumber)}
            </td>
            <td style="padding:13px 10px;border-bottom:1px solid #1e293b;cursor:pointer;" onclick="viewTeamDetail(${team.teamNumber})">
                <span style="color:#94a3b8;font-size:0.85em;font-weight:600;">${team.teamName || ''}</span>
            </td>
            ${td(`<span style="color:${ts.color};">${compStr}</span>`)}
            ${(() => {
                const rpRank = rpRankMap[String(team.teamNumber)];
                return td(rpRank != null ? `<span style="color:#94a3b8;font-weight:700;">#${rpRank}</span>` : '—');
            })()}
            ${td(`${epaVal.toFixed(1)}${ceilBadge}${localEpaBadge(team)}`)}
            ${td(hasOPR ? `${opr != null ? opr.toFixed(1) : '—'}${oprBadge}` : '—')}
            ${td(`${scoutStr}${fusedBadge}${scoutAdjBadge}`)}
            <td style="padding:13px 10px;border-bottom:1px solid #1e293b;text-align:center;">
                <button onclick="dnpTeam('${team.teamNumber}')"
                    style="background:#1e293b;color:#ef4444;border:1px solid #ef4444;border-radius:6px;padding:5px 10px;font-size:0.8rem;font-weight:700;cursor:pointer;white-space:nowrap;">
                    DNP
                </button>
            </td>
        </tr>`;
    }).join('');

    enablePickListDrag(tbody);
}

function enablePickListDrag(tbody) {
    tbody.addEventListener('pointerdown', e => {
        if (!e.target.closest('.drag-handle')) return;
        e.preventDefault();

        const dragRow = e.target.closest('tr');
        const rect = dragRow.getBoundingClientRect();
        const offsetY = e.clientY - rect.top;

        // Ghost: wrap in a table so <tr> renders correctly outside its parent
        const ghostTable = document.createElement('table');
        Object.assign(ghostTable.style, {
            position: 'fixed', left: rect.left + 'px', top: rect.top + 'px',
            width: rect.width + 'px', zIndex: '9999', opacity: '0.92',
            pointerEvents: 'none', borderCollapse: 'collapse',
            boxShadow: '0 6px 24px rgba(0,0,0,0.5)', borderRadius: '4px',
            fontFamily: 'inherit', fontSize: 'inherit',
        });
        const ghostBody = document.createElement('tbody');
        ghostBody.appendChild(dragRow.cloneNode(true));
        ghostTable.appendChild(ghostBody);
        document.body.appendChild(ghostTable);
        dragRow.style.opacity = '0.25';
        document.body.style.cursor = 'grabbing';

        function getTarget(cx, cy) {
            ghostTable.style.visibility = 'hidden';
            const el = document.elementFromPoint(cx, cy);
            ghostTable.style.visibility = '';
            return el?.closest('#pickListBody tr') || null;
        }
        function clearIndicators() {
            tbody.querySelectorAll('.pick-drop-before, .pick-drop-after').forEach(r => {
                r.classList.remove('pick-drop-before', 'pick-drop-after');
            });
        }

        function onMove(e) {
            e.preventDefault();
            ghostTable.style.top = (e.clientY - offsetY) + 'px';
            const target = getTarget(e.clientX, e.clientY);
            clearIndicators();
            if (target && target !== dragRow) {
                const mid = target.getBoundingClientRect();
                target.classList.add(e.clientY < mid.top + mid.height / 2 ? 'pick-drop-before' : 'pick-drop-after');
            }
        }

        function finish(e) {
            ghostTable.remove();
            dragRow.style.opacity = '';
            document.body.style.cursor = '';
            const target = getTarget(e.clientX, e.clientY);
            clearIndicators();
            if (target && target !== dragRow) {
                const mid = target.getBoundingClientRect();
                if (e.clientY < mid.top + mid.height / 2) tbody.insertBefore(dragRow, target);
                else target.after(dragRow);
            }
            refreshPickPositions();
            savePickOrder();
            document.removeEventListener('pointermove', onMove);
            document.removeEventListener('pointerup', finish);
            document.removeEventListener('pointercancel', cancel);
        }

        function cancel() {
            ghostTable.remove();
            dragRow.style.opacity = '';
            document.body.style.cursor = '';
            clearIndicators();
            document.removeEventListener('pointermove', onMove);
            document.removeEventListener('pointerup', finish);
            document.removeEventListener('pointercancel', cancel);
        }

        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', finish);
        document.addEventListener('pointercancel', cancel);
    });
}

// ── Draft ────────────────────────────────────────────────────────────────

const DRAFT_ALLIANCE_COLORS = [
    { solid: '#ef4444', bg: 'rgba(239,68,68,0.10)' },
    { solid: '#f97316', bg: 'rgba(249,115,22,0.10)' },
    { solid: '#eab308', bg: 'rgba(234,179,8,0.10)' },
    { solid: '#22c55e', bg: 'rgba(34,197,94,0.10)' },
    { solid: '#06b6d4', bg: 'rgba(6,182,212,0.10)' },
    { solid: '#3b82f6', bg: 'rgba(59,130,246,0.10)' },
    { solid: '#8b5cf6', bg: 'rgba(139,92,246,0.10)' },
    { solid: '#ec4899', bg: 'rgba(236,72,153,0.10)' },
];
let draftRPRankedTeams = [];
let draftHistory = [];
let draftMode = localStorage.getItem('draftMode') || 'mock';

function loadDraftWeights() {
    try {
        const w = JSON.parse(localStorage.getItem('draftEPAWeights'));
        if (w && typeof w.scout === 'number') return w;
    } catch {}
    return { scout: 50, statbotics: 25, opr: 25 };
}
let draftWeights = loadDraftWeights();
let draftNumAlliances = Math.max(2, parseInt(localStorage.getItem('draftNumAlliances')) || 8);
let draftPicksPerAlliance = Math.max(1, parseInt(localStorage.getItem('draftPicksPerAlliance')) || 2);

window.saveDraftConfig = function () {
    draftNumAlliances = Math.max(2, Math.min(16, parseInt(document.getElementById('draftNumAlliances')?.value) || 8));
    draftPicksPerAlliance = Math.max(1, Math.min(5, parseInt(document.getElementById('draftPicksPerAlliance')?.value) || 2));
    localStorage.setItem('draftNumAlliances', draftNumAlliances);
    localStorage.setItem('draftPicksPerAlliance', draftPicksPerAlliance);
    draftHistory = [];
    localStorage.removeItem('mockDraftState');
    renderDraft();
};

window.saveDraftWeights = function () {
    draftWeights = {
        scout:      Math.max(0, parseFloat(document.getElementById('wScout')?.value)      || 0),
        statbotics: Math.max(0, parseFloat(document.getElementById('wStatbotics')?.value) || 0),
        opr:        Math.max(0, parseFloat(document.getElementById('wOPR')?.value)         || 0),
    };
    localStorage.setItem('draftEPAWeights', JSON.stringify(draftWeights));
    renderDraft();
};

window.setDraftMode = function (mode) {
    draftMode = mode;
    localStorage.setItem('draftMode', mode);
    if (mode === 'real' && !loadDraftState()) {
        const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
        if (eventKey) {
            const raw = localStorage.getItem(`tbaAlliances_${eventKey}`);
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    const alliances = Array.from({ length: draftNumAlliances }, (_, i) => {
                        const a = data[i];
                        if (!a) return { captain: null, picks: Array(draftPicksPerAlliance).fill(null) };
                        const strip = k => a.picks?.[k]?.replace?.(/^frc/i, '') || null;
                        return { captain: strip(0), picks: Array.from({ length: draftPicksPerAlliance }, (_, k) => strip(k + 1)) };
                    });
                    saveDraftState({ alliances, currentAlliance: draftNumAlliances, currentRound: draftPicksPerAlliance });
                } catch {}
            }
        }
    }
    renderDraft();
};

function loadDraftState() {
    try {
        const s = JSON.parse(localStorage.getItem(`${draftMode}DraftState`));
        if (!s?.alliances?.length) return null;
        for (const a of s.alliances) {
            if (!Array.isArray(a.picks)) {
                a.picks = [a.pick1 ?? null, a.pick2 ?? null];
                while (a.picks.length < draftPicksPerAlliance) a.picks.push(null);
                a.picks = a.picks.slice(0, draftPicksPerAlliance);
                delete a.pick1; delete a.pick2;
            }
        }
        if (s.alliances.length !== draftNumAlliances) return null;
        return s;
    } catch { }
    return null;
}
function saveDraftState(s) { localStorage.setItem(`${draftMode}DraftState`, JSON.stringify(s)); }
function freshDraftState() {
    return {
        alliances: Array.from({ length: draftNumAlliances }, () => ({ captain: null, picks: Array(draftPicksPerAlliance).fill(null) })),
        currentAlliance: 0,
        currentRound: 1,
    };
}
function buildDraftPickedSet(alliances) {
    const s = new Set();
    for (const a of alliances) {
        if (a.captain) s.add(String(a.captain));
        if (Array.isArray(a.picks)) {
            for (const p of a.picks) if (p) s.add(String(p));
        } else {
            if (a.pick1) s.add(String(a.pick1));
            if (a.pick2) s.add(String(a.pick2));
        }
    }
    return s;
}
function draftFillCaptain(state) {
    if (state.currentRound !== 1 || state.currentAlliance >= draftNumAlliances) return;
    const a = state.alliances[state.currentAlliance];
    if (a.captain !== null) return;
    const picked = buildDraftPickedSet(state.alliances);
    const next = draftRPRankedTeams.find(tn => !picked.has(String(tn)));
    if (next != null) a.captain = String(next);
}

window.loadTBAAlliances = async function () {
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { alert('No event key — enter one on the Home tab first.'); return; }

    const statusEl = document.getElementById('draftAllianceLoadStatus');
    if (statusEl) statusEl.textContent = 'Loading…';

    try {
        const data = await fetchTBA(`/event/${eventKey}/alliances`);
        if (!Array.isArray(data) || !data.length) {
            if (statusEl) statusEl.textContent = 'No alliance data available yet.';
            return;
        }

        const alliances = Array.from({ length: draftNumAlliances }, (_, i) => {
            const a = data[i];
            if (!a) return { captain: null, picks: Array(draftPicksPerAlliance).fill(null) };
            const strip = k => a.picks?.[k]?.replace?.(/^frc/i, '') || null;
            return { captain: strip(0), picks: Array.from({ length: draftPicksPerAlliance }, (_, k) => strip(k + 1)) };
        });

        localStorage.setItem(`tbaAlliances_${eventKey}`, JSON.stringify(data));
        draftHistory = [];
        saveDraftState({ alliances, currentAlliance: draftNumAlliances, currentRound: draftPicksPerAlliance });
        renderDraft();
        if (statusEl) statusEl.textContent = `Loaded ${data.length} alliance${data.length !== 1 ? 's' : ''} from TBA.`;
    } catch (err) {
        if (statusEl) statusEl.textContent = `Error: ${err.message}`;
    }
};

window.resetDraft = function () {
    draftHistory = [];
    localStorage.removeItem('mockDraftState');
    renderDraft();
};
window.draftUndo = function () {
    if (!draftHistory.length) return;
    saveDraftState(JSON.parse(draftHistory.pop()));
    renderDraft();
};
window.draftPick = function (teamNumber) {
    const tn = String(teamNumber);
    const state = loadDraftState() || freshDraftState();
    const N = draftNumAlliances;
    const P = draftPicksPerAlliance;
    if (state.currentAlliance >= N || state.currentAlliance < 0) return;
    if (buildDraftPickedSet(state.alliances).has(tn)) return;
    draftHistory.push(JSON.stringify(state));
    const a = state.alliances[state.currentAlliance];
    const pickIdx = state.currentRound - 1;
    if (!a.captain || a.picks[pickIdx] !== null) return;
    a.picks[pickIdx] = tn;

    const forward = (state.currentRound % 2 === 1);
    if (forward) {
        state.currentAlliance++;
        if (state.currentAlliance >= N) {
            if (state.currentRound >= P) {
                state.currentAlliance = N; // done
            } else {
                state.currentRound++;
                state.currentAlliance = N - 1; // start backward
            }
        } else if (state.currentRound === 1) {
            draftFillCaptain(state);
        }
    } else {
        state.currentAlliance--;
        if (state.currentAlliance < 0) {
            if (state.currentRound >= P) {
                state.currentAlliance = -1; // done
            } else {
                state.currentRound++;
                state.currentAlliance = 0; // start forward
                draftFillCaptain(state);
            }
        }
    }
    saveDraftState(state);
    renderDraft();
};

window.refreshLocalEPADebug = async function () {
    const el = document.getElementById('local-epa-debug-content');
    if (!el) return;
    el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">Computing…</div>`;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">No event key set.</div>`; return; }

    const [teams, matches, tbaTeamsDebug] = await Promise.all([
        db.teams.where('eventKey').equals(eventKey).toArray(),
        db.matches.where('eventKey').equals(eventKey).toArray(),
        db.tbaTeams.toArray(),
    ]);
    if (!teams.length) { el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">No teams for ${eventKey}.</div>`; return; }

    // Mirror ignore maps from computeLocalEPA
    const dbGlobalIgnoredKeys = new Set(matches.filter(m => m.globallyIgnored).map(m => m.key));
    const dbTeamIgnoredKeys = {};
    for (const t of tbaTeamsDebug) {
        const keys = getTeamIgnoredKeys(t);
        if (keys.length) dbTeamIgnoredKeys[t.teamNumber] = new Set(keys);
    }

    // Partition: local vs statbotics (same logic as computeLocalEPA)
    const sbTeams = [], localTeams = [];
    for (const t of teams) {
        const sbEventMatches = (t.rawStatboticsData || [])
            .filter(m => m.event === eventKey && m.epa?.post)
            .sort((a, b) => (a.time || 0) - (b.time || 0));
        if (sbEventMatches.length > 0) {
            sbTeams.push({ ...t, _sbLastEPA: sbEventMatches[sbEventMatches.length - 1].epa.post });
        } else {
            localTeams.push(t);
        }
    }

    const played = matches
        .filter(m => (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !dbGlobalIgnoredKeys.has(m.key))
        .sort((a, b) => (a.actualTime || a.predictedTime || 0) - (b.actualTime || b.predictedTime || 0));

    // Replay computation, capturing debug info per team per match
    const epaState = {};
    const debugLog = {};  // tn -> [{matchLabel, n, K, predAlliance, actual, errorPerTeam, delta}]

    for (const t of localTeams) {
        const careerN = (t.rawStatboticsData || []).filter(m => m.epa?.post).length;
        epaState[t.teamNumber] = {
            current: t.preEventEPA ?? t.currentEPA ?? 0,
            auto:    t.preEventAutoEPA ?? t.autoEPA ?? 0,
            endgame: t.preEventEndgameEPA ?? t.endgameEPA ?? 0,
            n: careerN,
            careerN,
            preEventEPA: t.preEventEPA ?? t.currentEPA ?? 0,
        };
        debugLog[t.teamNumber] = [];
    }
    const getE = tn => epaState[tn];

    for (const m of played) {
        const label = (!m.compLevel || m.compLevel === 'qm') ? `Q${m.matchNumber}` : `P${m.matchNumber}`;
        for (const [alliance, score, bd] of [
            [m.red  || [], m.redScore,  m.redBreakdown],
            [m.blue || [], m.blueScore, m.blueBreakdown],
        ]) {
            const members = alliance.map(t => String(t).replace(/^frc/i, ''))
                .filter(t => epaState[t]);
            if (!members.length) continue;
            const predTotal = members.reduce((s, t) => s + getE(t).current, 0);
            const N = members.length;
            for (const t of members) {
                if (dbTeamIgnoredKeys[parseInt(t)]?.has(m.key)) continue;
                const e = getE(t);
                e.n++;
                const prev = Math.min(0.5, Math.max(0.3, 0.5 - (0.2 / 6) * (e.n - 6)));
                const K = (2 / 3) * prev;
                const errorPerTeam = (score - predTotal) / N;
                const delta = K * errorPerTeam;
                debugLog[t].push({
                    label, n: e.n, K, predAlliance: predTotal, actual: score, errorPerTeam, delta,
                    epaBefore: e.current,
                });
                e.current += delta;
                const autoActual    = bd?.totalAutoPoints ?? null;
                const endgameActual = bd ? ((bd.endGameTowerPoints || 0) + (bd['Hub Endgame Fuel Count'] || 0)) : null;
                if (autoActual    != null) e.auto    += K * (autoActual    - members.reduce((s, t2) => s + getE(t2).auto,    0)) / N;
                if (endgameActual != null) e.endgame += K * (endgameActual - members.reduce((s, t2) => s + getE(t2).endgame, 0)) / N;
            }
        }
    }

    const fmtNum = (v, d = 1) => v != null && isFinite(v) ? v.toFixed(d) : '—';
    const sign = v => v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
    const deltaColor = v => v > 1 ? '#4ade80' : v < -1 ? '#f87171' : '#94a3b8';

    const kBar = (K) => {
        const pct = Math.round((K / 0.333) * 100);
        const color = K >= 0.32 ? '#4ade80' : K >= 0.25 ? '#fbbf24' : '#f87171';
        return `<div style="display:inline-flex;align-items:center;gap:4px;">
            <div style="width:32px;height:7px;background:#1e293b;border-radius:3px;overflow:hidden;">
                <div style="width:${pct}%;height:100%;background:${color};border-radius:3px;"></div>
            </div>
            <span style="font-size:0.78em;color:${color};">${K.toFixed(3)}</span>
        </div>`;
    };

    const teamSections = localTeams.map(t => {
        const tn = t.teamNumber;
        const e  = epaState[tn];
        const log = debugLog[tn] || [];
        const finalEPA = e?.current ?? null;
        const preEPA   = e?.preEventEPA ?? null;
        const delta    = (finalEPA != null && preEPA != null) ? finalEPA - preEPA : null;
        const kValues  = log.map(r => r.K);
        const kMin = kValues.length ? Math.min(...kValues) : null;
        const kMax = kValues.length ? Math.max(...kValues) : null;
        const careerN  = e?.careerN ?? 0;

        const matchRows = log.map(r => `
            <tr style="border-bottom:1px solid #0f172a;">
                <td style="padding:4px 8px;color:#94a3b8;font-size:0.82em;">${r.label}</td>
                <td style="padding:4px 8px;text-align:right;color:#64748b;font-size:0.82em;">${r.n}</td>
                <td style="padding:4px 8px;">${kBar(r.K)}</td>
                <td style="padding:4px 8px;text-align:right;color:#94a3b8;font-size:0.82em;">${fmtNum(r.predAlliance)}</td>
                <td style="padding:4px 8px;text-align:right;color:#e2e8f0;font-size:0.82em;">${fmtNum(r.actual)}</td>
                <td style="padding:4px 8px;text-align:right;color:${deltaColor(r.errorPerTeam)};font-size:0.82em;">${sign(r.errorPerTeam)}</td>
                <td style="padding:4px 8px;text-align:right;font-size:0.82em;">
                    <span style="color:${deltaColor(r.delta)};">${sign(r.delta)}</span>
                    <span style="color:#334155;font-size:0.8em;margin-left:3px;">${fmtNum(r.epaBefore + r.delta)}</span>
                </td>
            </tr>`).join('');

        const kRangeStr = kValues.length === 0 ? '—'
            : kMin === kMax ? kMax.toFixed(3)
            : `${kMin.toFixed(3)} – ${kMax.toFixed(3)}`;

        return `
        <details style="background:#1e293b;border-radius:6px;margin-bottom:6px;border:1px solid #334155;overflow:hidden;">
            <summary style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 12px;background:#162032;cursor:pointer;user-select:none;">
                <span style="color:#f1f5f9;font-weight:700;min-width:52px;">${tn}</span>
                <span style="color:#64748b;font-size:0.78em;">career N: ${careerN}</span>
                <span style="color:#64748b;font-size:0.78em;">${log.length} event match${log.length !== 1 ? 'es' : ''}</span>
                <span style="color:#64748b;font-size:0.78em;">K: ${kRangeStr}</span>
                <span style="margin-left:auto;display:flex;gap:10px;align-items:center;">
                    <span style="font-size:0.82em;color:#94a3b8;">Pre: ${fmtNum(preEPA)}</span>
                    <span style="font-size:0.82em;color:#60a5fa;font-weight:600;">Now: ${fmtNum(finalEPA)}</span>
                    ${delta != null ? `<span style="font-size:0.82em;color:${deltaColor(delta)};font-weight:600;">${sign(delta)}</span>` : ''}
                </span>
            </summary>
            ${log.length ? `
            <div style="overflow-x:auto;">
            <table style="border-collapse:collapse;width:100%;">
                <thead><tr style="color:#475569;font-size:0.75em;border-bottom:1px solid #334155;">
                    <th style="padding:4px 8px;text-align:left;">Match</th>
                    <th style="padding:4px 8px;text-align:right;">n</th>
                    <th style="padding:4px 8px;text-align:left;">K</th>
                    <th style="padding:4px 8px;text-align:right;">Pred</th>
                    <th style="padding:4px 8px;text-align:right;">Actual</th>
                    <th style="padding:4px 8px;text-align:right;">Err/team</th>
                    <th style="padding:4px 8px;text-align:right;">ΔEPA → new</th>
                </tr></thead>
                <tbody>${matchRows}</tbody>
            </table>
            </div>` : `<div style="padding:10px 12px;color:#475569;font-size:0.82em;">No scored matches yet.</div>`}
        </details>`;
    }).join('');

    const sbNote = sbTeams.length
        ? `<div style="color:#64748b;font-size:0.82em;margin-top:10px;padding:8px 12px;background:#0f172a;border-radius:6px;border:1px solid #1e293b;">
               ${sbTeams.length} team${sbTeams.length !== 1 ? 's' : ''} using Statbotics event data (not estimated locally):
               ${sbTeams.map(t => `<span style="color:#475569;">${t.teamNumber}</span>`).join(', ')}
           </div>`
        : '';

    const summary = `
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;font-size:0.82em;color:#94a3b8;">
            <span>Local EPA: <strong style="color:#4ade80;">${localTeams.length} teams</strong></span>
            <span>Statbotics: <strong style="color:#60a5fa;">${sbTeams.length} teams</strong></span>
            <span>Played matches: <strong style="color:#e2e8f0;">${played.length}</strong></span>
        </div>`;

    el.innerHTML = summary
        + (localTeams.length ? teamSections : `<div style="color:#64748b;font-size:0.85em;">All teams have Statbotics event data — local estimation not active.</div>`)
        + sbNote;
};

window.refreshFusionDebug = async function () {
    const el = document.getElementById('fusion-debug-content');
    if (!el) return;
    el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">Computing…</div>`;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) { el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">No event key set.</div>`; return; }

    const [allTeamsArr, tbaTeamsArr, matchesArr] = await Promise.all([
        db.teams.where('eventKey').equals(eventKey).toArray(),
        db.tbaTeams.toArray(),
        db.matches.where('eventKey').equals(eventKey).toArray(),
    ]);
    if (!allTeamsArr.length && !tbaTeamsArr.length) {
        el.innerHTML = `<div style="color:#64748b;font-size:0.85em;">No data for ${eventKey}.</div>`; return;
    }

    const teamsMap = Object.fromEntries(allTeamsArr.map(t => [t.teamNumber, t]));
    const tbaMap   = Object.fromEntries(tbaTeamsArr.map(t => [t.teamNumber, t]));
    const played   = matchesArr.filter(m => (m.redScore ?? -1) >= 0);
    const { relResiduals } = wlCollectResiduals(played, tbaMap, teamsMap);
    const oprSigmaRel = wlStd(relResiduals);

    // Event-level summary
    const meanAbsRes = relResiduals.length
        ? (relResiduals.reduce((s, v) => s + Math.abs(v), 0) / relResiduals.length * 100).toFixed(1)
        : null;
    const eventSummary = `
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;font-size:0.82em;color:#94a3b8;">
            <span>Residuals: <strong style="color:#e2e8f0;">${relResiduals.length}</strong></span>
            <span>OPR CV: <strong style="color:${isFinite(oprSigmaRel) ? '#4ade80' : '#f87171'};">${isFinite(oprSigmaRel) ? (oprSigmaRel * 100).toFixed(1) + '%' : '— (no data)'}</strong></span>
            ${meanAbsRes != null ? `<span>Mean |residual|: <strong style="color:#e2e8f0;">${meanAbsRes}%</strong></span>` : ''}
            <span>Played matches: <strong style="color:#e2e8f0;">${played.length}</strong></span>
        </div>`;

    // Collect all team numbers (union of EPA + OPR)
    const allTNs = new Set([
        ...allTeamsArr.map(t => t.teamNumber),
        ...tbaTeamsArr.map(t => t.teamNumber),
    ]);

    const rows = [...allTNs].map(tn => wlTeamFusionStats(tn, tbaMap, teamsMap, oprSigmaRel))
        .filter(r => r.fused != null)
        .sort((a, b) => (b.fused ?? 0) - (a.fused ?? 0));

    const fmtNum = (v, d = 1) => v != null ? v.toFixed(d) : '—';
    const bar = (pct) => {
        if (pct == null) return '—';
        const w = Math.round(pct);
        const color = pct >= 60 ? '#4ade80' : pct >= 25 ? '#fbbf24' : '#60a5fa';
        return `<div style="display:flex;align-items:center;gap:5px;">
            <div style="flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;min-width:40px;">
                <div style="width:${w}%;height:100%;background:${color};border-radius:4px;"></div>
            </div>
            <span style="font-size:0.78em;color:#94a3b8;min-width:28px;text-align:right;">${w}%</span>
        </div>`;
    };

    const th = (label, title = '') =>
        `<th title="${title}" style="padding:5px 8px;text-align:right;color:#475569;font-size:0.75em;font-weight:600;white-space:nowrap;border-bottom:1px solid #334155;">${label}</th>`;
    const thL = (label) =>
        `<th style="padding:5px 8px;text-align:left;color:#475569;font-size:0.75em;font-weight:600;border-bottom:1px solid #334155;">${label}</th>`;
    const td = (v, color = '#cbd5e1') =>
        `<td style="padding:5px 8px;text-align:right;color:${color};font-size:0.82em;white-space:nowrap;">${v}</td>`;
    const tdL = (v) =>
        `<td style="padding:5px 8px;text-align:left;font-size:0.82em;">${v}</td>`;

    const tableRows = rows.map(r => {
        const adjBadge = r.hasAdj ? `<span style="font-size:0.7em;color:#fbbf24;margin-left:3px;">ADJ</span>` : '';
        const sigmaEpaStr = r.sigmaEpa != null
            ? `${fmtNum(r.sigmaEpa)} <span style="color:#475569;font-size:0.75em;">(${fmtNum(r.sigmaEpaEst)}/${fmtNum(r.sigmaEpaGen)})</span>`
            : '—';
        const sigmaOprStr = isFinite(r.sigmaOpr) ? fmtNum(r.sigmaOpr) : '<span style="color:#475569;">∞</span>';
        return `<tr style="border-bottom:1px solid #0f172a;">
            ${tdL(`<span style="color:#f1f5f9;font-weight:600;">${r.tn}</span>${adjBadge}`)}
            ${td(fmtNum(r.epa))}
            ${td(sigmaEpaStr, '#94a3b8')}
            ${td(fmtNum(r.opr))}
            ${td(sigmaOprStr, '#94a3b8')}
            ${td(`<strong style="color:#60a5fa;">${fmtNum(r.fused)}</strong>`)}
            <td style="padding:5px 8px;min-width:90px;">${bar(r.oprWeightPct)}</td>
        </tr>`;
    }).join('');

    el.innerHTML = eventSummary + `
        <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;font-size:0.85em;">
            <thead><tr>
                ${thL('Team')}
                ${th('EPA', 'Statbotics current EPA')}
                ${th('σ_EPA (est/gen)', 'EPA uncertainty: √(σ_est² + σ_gen²). est = SD/√matchCount, gen = 12% floor')}
                ${th('OPR', 'TBA OPR (or adjusted OPR if active)')}
                ${th('σ_OPR', 'OPR uncertainty: OPR × CV(relResiduals). ∞ = no event residuals yet')}
                ${th('Fused', 'Inverse-variance weighted estimate')}
                ${th('OPR wt', 'OPR share of total weight')}
            </tr></thead>
            <tbody>${tableRows}</tbody>
        </table>
        </div>
        <p style="color:#334155;font-size:0.75em;margin-top:10px;line-height:1.5;">
            σ_EPA = √(σ_est² + σ_gen²) where σ_est = SD/√N and σ_gen = 12% EPA.
            σ_OPR = CV × |OPR| where CV = std(relative residuals on played matches).
            Weight = 1/σ². OPR wt% = 0 pre-event, grows as residuals accumulate.
        </p>`;
};

async function renderDevTab() {
    const el = document.getElementById('tools-tab-dev');
    if (!el) return;
    el.innerHTML = `<div style="color:#94a3b8;padding:12px;">Loading…</div>`;
    const eventKey = (document.getElementById('eventKeyInput')?.value ?? '').trim().toLowerCase();
    const sources = [];

    const teams    = await db.teams.toArray();
    const tbaTeams = await db.tbaTeams.toArray();
    const matches  = await db.matches.toArray();
    if (teams.length)    sources.push({ name: `db.teams (${teams.length} records)`,    ex: teams[0] });
    if (tbaTeams.length) sources.push({ name: `db.tbaTeams (${tbaTeams.length} records)`, ex: tbaTeams[0] });
    if (matches.length)  sources.push({ name: `db.matches (${matches.length} records)`,  ex: matches[0] });

    if (eventKey) {
        for (const [lsKey, label] of [
            [`scoutingData_${eventKey}`, 'Scouting data'],
            [`pitData_${eventKey}`, 'Pit data'],
        ]) {
            try {
                const rows = JSON.parse(localStorage.getItem(lsKey) ?? 'null');
                if (Array.isArray(rows) && rows.length) sources.push({ name: `${label} (${rows.length} rows)`, ex: rows[0] });
            } catch {}
        }
    }

    const renderValue = (v, depth = 0) => {
        if (v == null) return '<span style="color:#475569;">—</span>';
        if (typeof v !== 'object') return `<span style="color:#64748b;">${String(v)}</span>`;
        const entries = Object.entries(v);
        if (!entries.length) return '<span style="color:#475569;">{}</span>';
        const preview = JSON.stringify(v);
        const short = preview.length <= 60 ? `<span style="color:#475569;font-size:0.9em;">${preview}</span>` : `<span style="color:#475569;font-size:0.9em;">${preview.slice(0, 60)}…</span>`;
        const indent = 4 + depth * 4;
        const inner = entries.map(([k2, v2]) => `
            <tr style="border-bottom:1px solid #0a0f1a;">
                <td style="padding:2px 8px 2px ${indent}px;color:#64748b;white-space:nowrap;font-size:13px;">${k2}</td>
                <td style="padding:2px 8px;font-size:13px;font-family:monospace;word-break:break-all;max-width:340px;">${renderValue(v2, depth + 1)}</td>
            </tr>`).join('');
        return `<details style="display:inline-block;max-width:100%;">
            <summary style="cursor:pointer;color:#475569;font-size:0.85em;list-style:none;white-space:nowrap;">▶ ${short}</summary>
            <table style="border-collapse:collapse;width:100%;background:#040810;">${inner}</table>
        </details>`;
    };
    const renderSource = ({ name, ex }) => {
        const rows = Object.entries(ex).map(([k, v]) => `
            <tr style="border-bottom:1px solid #0f172a;">
                <td style="padding:3px 8px;color:#94a3b8;white-space:nowrap;font-size:14px;">${k}</td>
                <td style="padding:3px 8px;font-size:14px;font-family:monospace;word-break:break-all;max-width:360px;">${renderValue(v)}</td>
            </tr>`).join('');
        return `<details style="margin-bottom:8px;border:1px solid #1e293b;border-radius:6px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:600;padding:8px 12px;background:#0f172a;font-size:0.9em;">
                ${name}
            </summary>
            <div style="overflow-x:auto;">
                <table style="border-collapse:collapse;min-width:320px;background:#080d16;">${rows}</table>
            </div>
        </details>`;
    };

    const fieldExplorerHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Field Explorer
            </summary>
            <div style="padding:10px;">
                ${sources.length ? sources.map(renderSource).join('') : '<div style="color:#64748b;padding:4px 0;">No data loaded yet. Sync data first.</div>'}
            </div>
        </details>`;

    const tmInputId  = 'devTimeMachineInput';
    const tmStatusId = 'devTimeMachineStatus';
    const timeMachineHtml = `
        <details open style="border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Time Machine
            </summary>
            <div style="padding:14px;">
                <p style="color:#94a3b8;font-size:0.85em;margin:0 0 12px;line-height:1.6;">
                    Roll data back to the state after a specific qual match completed.
                    Scrubs scores and breakdowns for later matches from TBA, filters Statbotics
                    match history, and trims scouting entries. The schedule stays intact.
                    <strong style="color:#fbbf24;">You will need to re-sync all sources afterward.</strong>
                </p>
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                    <label style="color:#94a3b8;font-size:0.85em;">After qual match #</label>
                    <input id="${tmInputId}" type="number" min="1" step="1" placeholder="e.g. 12"
                        style="width:80px;padding:4px 8px;background:#0f172a;border:1px solid #334155;color:#f8fafc;border-radius:4px;font-size:0.9em;">
                    <button onclick="applyTimeMachineSnapshot()"
                        style="padding:5px 14px;background:#334155;color:#f8fafc;border:1px solid #475569;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Roll back
                    </button>
                </div>
                <div id="${tmStatusId}" style="margin-top:10px;font-size:0.82em;color:#64748b;"></div>
            </div>
        </details>`;

    const wlControlsHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;" open>
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Watch List Settings
            </summary>
            <div style="padding:10px 14px;" id="dev-wl-controls-content">
                ${wlDetailCache
                    ? buildWLControlsHTML(
                        (document.getElementById('eventKeyInput')?.value ?? '').trim().toLowerCase(),
                        wlDetailCache.effectiveThresholds,
                        wlDetailCache.allMatches.length)
                    : '<div style="color:#64748b;font-size:0.85em;">Open the Watch List tab first to load settings.</div>'}
            </div>
        </details>`;

    const calibrationHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Watch List — Model Calibration
            </summary>
            <div style="padding:14px;">
                <p style="color:#94a3b8;font-size:0.85em;line-height:1.6;margin:0 0 14px;">
                    Tests the simulation model against the current event's played matches.
                    Each match is predicted using only data from preceding matches — exactly as the model sees it live.
                    Results show how well the predicted probabilities are calibrated against actual outcomes.
                </p>
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
                    <label style="color:#94a3b8;font-size:0.84em;white-space:nowrap;font-weight:600;">Event keys</label>
                    <input id="bt-event-keys" type="text" value="${eventKey}"
                        placeholder="e.g. 2026necmp1, 2026mane"
                        style="flex:1;min-width:180px;background:#0f172a;border:1px solid #334155;border-radius:5px;color:#f1f5f9;padding:6px 10px;font-size:0.83em;">
                    <button onclick="runBacktest()" style="background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:6px;padding:6px 16px;font-size:0.85em;font-weight:600;cursor:pointer;white-space:nowrap;">
                        Run Backtest
                    </button>
                </div>
                <p style="color:#475569;font-size:0.79em;margin:-2px 0 12px;line-height:1.5;">
                    Comma-separated. Other events are fetched from TBA + Statbotics and cached for the session.
                </p>
                <div id="algo-backtest-results"></div>
            </div>
        </details>`;

    // ── Quip Tier Breakdown ─────────────────────────────────────────────────
    const quipsEnabled = localStorage.getItem('quipsEnabled') === 'true';
    let quipTierHtml = `<label style="display:flex;align-items:center;gap:7px;font-size:0.8em;color:#475569;cursor:pointer;margin-bottom:14px;">
        <input type="checkbox" onchange="toggleQuipsEnabled()" ${quipsEnabled ? 'checked' : ''} style="cursor:pointer;accent-color:#4ade80;">
        Show team quips
    </label>`;
    if (quipsEnabled) {
        const eventTeams = eventKey ? await db.teams.where('eventKey').equals(eventKey).toArray() : [];
        if (eventTeams.length > 0) {
            const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
            const gameConfig = getGameConfig(eventKey);
            const getEPA = t => {
                const fr = fusedCache?.teams?.[String(t.teamNumber)];
                if (fr?.available && gameConfig?.computeFusedEPABreakdown) {
                    return gameConfig.computeFusedEPABreakdown(fr.stats).total;
                }
                return t.currentEPA ?? 0;
            };
            const epas = eventTeams.map(getEPA);
            const logMin = Math.log(Math.max(Math.min(...epas), 0.1));
            const logMax = Math.log(Math.max(Math.max(...epas), 0.1));
            const getLogFrac = t => logMax > logMin
                ? Math.max(0, Math.min(1, (Math.log(Math.max(getEPA(t), 0.1)) - logMin) / (logMax - logMin)))
                : 0.5;

            // Count teams per tier and collect their numbers
            const tierBuckets = Object.fromEntries(QUIP_TIERS.map(({ name }) => [name, []]));
            for (const t of eventTeams) {
                const tier = logFractionToTier(getLogFrac(t));
                tierBuckets[tier].push({ tn: t.teamNumber, frac: getLogFrac(t) });
            }

            const tierColors = { Elite: '#f59e0b', S: '#4ade80', A: '#a78bfa', B: '#60a5fa', C: '#94a3b8' };
            const rows = QUIP_TIERS.map(({ name, min }, i) => {
                const next = QUIP_TIERS[i - 1];
                const range = next ? `${(min * 100).toFixed(0)}–${(next.min * 100).toFixed(0)}%` : `${(min * 100).toFixed(0)}–100%`;
                const bucket = tierBuckets[name] ?? [];
                bucket.sort((a, b) => b.frac - a.frac);
                const chips = bucket.map(({ tn, frac }) =>
                    `<span style="display:inline-block;padding:1px 6px;border-radius:10px;background:${tierColors[name]}22;border:1px solid ${tierColors[name]}55;color:${tierColors[name]};font-size:0.75em;margin:1px;">${tn} <span style="opacity:0.6;">${(frac * 100).toFixed(1)}%</span></span>`
                ).join('');
                return `<tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:5px 8px;font-weight:700;color:${tierColors[name]};">${name}</td>
                    <td style="padding:5px 8px;color:#64748b;font-size:0.82em;">${range}</td>
                    <td style="padding:5px 8px;text-align:center;color:#e2e8f0;font-weight:600;">${bucket.length}</td>
                    <td style="padding:5px 8px;">${chips}</td>
                </tr>`;
            }).join('');

            quipTierHtml += `
            <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
                <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                    Quip Tier Breakdown <span style="color:#475569;font-weight:400;font-size:0.85em;margin-left:8px;">${eventTeams.length} teams · ${eventKey}</span>
                </summary>
                <div style="padding:10px;overflow-x:auto;">
                    <table style="border-collapse:collapse;width:100%;font-size:0.85em;">
                        <thead><tr style="color:#475569;font-size:0.78em;border-bottom:1px solid #334155;">
                            <th style="padding:4px 8px;text-align:left;">Tier</th>
                            <th style="padding:4px 8px;text-align:left;">Log-fraction range</th>
                            <th style="padding:4px 8px;text-align:center;">Count</th>
                            <th style="padding:4px 8px;text-align:left;">Teams (fraction)</th>
                        </tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            </details>`;
        }
    }

    // ── Stream Seek Test ────────────────────────────────────────────────────
    let streamSeekStreams = [];
    try {
        streamSeekStreams = JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) || '[]')
            .filter(w => w.type === 'youtube' && w.startTimestamp)
            .sort((a, b) => a.startTimestamp - b.startTimestamp);
    } catch {}

    const streamSeekTestHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Stream Seek Test
            </summary>
            <div style="padding:14px;">
                <p style="color:#94a3b8;font-size:0.85em;margin:0 0 10px;line-height:1.6;">
                    Sets Match 1's <code style="color:#93c5fd;">actualTime</code> to N minutes after the matching stream starts.
                    The stream is selected by date — large N values will cross into the next day's stream automatically.
                    Open Match 1 afterward to verify the embed seeks to the right spot.
                </p>
                ${streamSeekStreams.length
                    ? `<div style="margin-bottom:12px;">${streamSeekStreams.map(s => {
                        const startLocal = new Date(s.startTimestamp * 1000).toLocaleString();
                        return `<div style="color:#64748b;font-size:0.82em;margin-bottom:2px;">📺 ${s.date} — started ${startLocal}</div>`;
                    }).join('')}</div>`
                    : `<div style="color:#64748b;font-size:0.82em;margin-bottom:12px;">No stream start times — sync schedule with <code style="color:#93c5fd;">VITE_YOUTUBE_KEY</code> set.</div>`
                }
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                    <label style="color:#94a3b8;font-size:0.85em;">Minutes into stream</label>
                    <input id="dev-stream-minutes" type="number" min="0" step="1" value="30"
                        style="width:80px;padding:4px 8px;background:#0f172a;border:1px solid #334155;color:#f8fafc;border-radius:4px;font-size:0.9em;">
                    <button onclick="devApplyStreamSeekTest()"
                        style="padding:5px 14px;background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Apply
                    </button>
                    <button onclick="devResetStreamSeekTest()"
                        style="padding:5px 14px;background:#334155;color:#f8fafc;border:1px solid #475569;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Reset
                    </button>
                </div>
                <div id="dev-stream-status" style="margin-top:10px;font-size:0.82em;color:#64748b;"></div>
            </div>
        </details>`;

    // ── Notification Tester ──────────────────────────────────────────────────
    const notifPerm = ('Notification' in window) ? Notification.permission : 'unsupported';
    const permColor = { granted: '#4ade80', denied: '#f87171', default: '#fbbf24', unsupported: '#64748b' }[notifPerm] ?? '#64748b';
    const notifTesterHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Notification Tester
            </summary>
            <div style="padding:14px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                    <span style="font-size:0.82em;color:#64748b;">Permission:</span>
                    <span style="font-size:0.82em;font-weight:700;color:${permColor};">${notifPerm}</span>
                    ${notifPerm === 'default' ? `<button onclick="Notification.requestPermission().then(()=>renderDevTab())"
                        style="margin-left:6px;padding:3px 10px;background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:5px;cursor:pointer;font-size:0.8em;">
                        Request
                    </button>` : ''}
                </div>
                <p style="color:#94a3b8;font-size:0.82em;margin:0 0 12px;line-height:1.5;">
                    Fires a notification directly, bypassing the enabled toggle. Use this to verify permission and OS delivery.
                </p>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <button onclick="devTestNotif('warn')"
                        style="padding:5px 14px;background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Test: Match Warning
                    </button>
                    <button onclick="devTestNotif('score')"
                        style="padding:5px 14px;background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Test: Score Posted
                    </button>
                    <button onclick="devTestNotif('title')"
                        style="padding:5px 14px;background:#334155;color:#f8fafc;border:1px solid #475569;border-radius:5px;cursor:pointer;font-size:0.85em;font-weight:600;">
                        Test: Tab Title
                    </button>
                </div>
                <div id="dev-notif-status" style="margin-top:10px;font-size:0.82em;color:#64748b;"></div>
            </div>
        </details>`;

    const fusionDebugHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;"
                 ontoggle="if(this.open) refreshFusionDebug()">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                EPA/OPR Fusion Debug
                <span style="color:#475569;font-weight:400;font-size:0.82em;margin-left:8px;">per-team σ &amp; weights</span>
            </summary>
            <div id="fusion-debug-content" style="padding:14px;">
                <div style="color:#64748b;font-size:0.85em;">Expand to compute.</div>
            </div>
        </details>`;

    const localEPADebugHtml = `
        <details style="margin-bottom:16px;border:1px solid #1e293b;border-radius:8px;overflow:hidden;"
                 ontoggle="if(this.open) refreshLocalEPADebug()">
            <summary style="cursor:pointer;color:#e2e8f0;font-weight:700;padding:10px 14px;background:#0f172a;font-size:1em;letter-spacing:0.02em;">
                Local EPA Gain Debug
                <span style="color:#475569;font-weight:400;font-size:0.82em;margin-left:8px;">K per match · error · ΔEPA</span>
            </summary>
            <div id="local-epa-debug-content" style="padding:14px;">
                <div style="color:#64748b;font-size:0.85em;">Expand to compute.</div>
            </div>
        </details>`;

    el.innerHTML = `<div style="padding:12px;">${quipTierHtml}${wlControlsHtml}${fusionDebugHtml}${localEPADebugHtml}${calibrationHtml}${notifTesterHtml}${streamSeekTestHtml}${fieldExplorerHtml}${timeMachineHtml}</div>`;
}

window.devTestNotif = function (type) {
    const statusEl = document.getElementById('dev-notif-status');
    const perm = ('Notification' in window) ? Notification.permission : 'unsupported';

    if (perm === 'unsupported') {
        if (statusEl) statusEl.textContent = 'Notifications are not supported in this browser.';
        return;
    }
    if (perm !== 'granted') {
        if (statusEl) statusEl.textContent = 'Permission not granted — click Request above first.';
        return;
    }

    if (type === 'title') {
        document.title = '🔔 Test notification — 1768 Scouting';
        if (statusEl) statusEl.textContent = 'Tab title updated. Switch to another tab and back to see it reset.';
        return;
    }

    let title, body, tag;
    if (type === 'warn') {
        title = 'QM 12 in ~3 min';
        body  = '254, 1114, 1678 vs 2056, 118, 148';
        tag   = 'dev-warn-test';
    } else {
        title = 'QM 12 scored — Red wins';
        body  = '72–41 · Red: 254, 1114, 1678 · Blue: 2056, 118, 148';
        tag   = 'dev-score-test';
    }

    new Notification(title, { body, tag, icon: '/favicon.ico' });
    if (statusEl) statusEl.textContent = `Fired: "${title}"`;
};

window.devApplyStreamSeekTest = async function () {
    const statusEl = document.getElementById('dev-stream-status');
    const minutes = parseFloat(document.getElementById('dev-stream-minutes')?.value ?? '');
    if (isNaN(minutes) || minutes < 0) { statusEl.textContent = 'Enter a valid number of minutes (≥ 0).'; return; }

    const eventKey = (document.getElementById('eventKeyInput')?.value ?? '').trim().toLowerCase();
    let webcasts = [];
    try { webcasts = JSON.parse(localStorage.getItem(`webcasts_${eventKey}`) || '[]'); } catch {}

    const streams = webcasts
        .filter(w => w.type === 'youtube' && w.startTimestamp)
        .sort((a, b) => a.startTimestamp - b.startTimestamp);
    if (!streams.length) { statusEl.textContent = 'No stream start times — sync schedule with VITE_YOUTUBE_KEY set.'; return; }

    const offsetSecs = Math.round(minutes * 60);
    // Pick the stream whose date still contains the offset time; fall back to last stream.
    const chosenStream = streams.find(s =>
        new Date((s.startTimestamp + offsetSecs) * 1000).toISOString().slice(0, 10) === s.date
    ) ?? streams[streams.length - 1];

    const testActualTime = chosenStream.startTimestamp + offsetSecs;

    const allMatches = await db.matches.toArray();
    if (!allMatches.length) { statusEl.textContent = 'No matches in DB — sync schedule first.'; return; }
    allMatches.sort((a, b) => a.matchNumber - b.matchNumber);
    const firstMatch = allMatches[0];
    await db.matches.update(firstMatch.key, { actualTime: testActualTime });

    const seekMins = Math.floor(offsetSecs / 60);
    const seekSecs = String(offsetSecs % 60).padStart(2, '0');
    statusEl.innerHTML = `Match ${firstMatch.matchNumber} → ${new Date(testActualTime * 1000).toLocaleString()} · stream: ${chosenStream.date} · offset: ${seekMins}m${seekSecs}s. <a href="#" onclick="viewMatchDetail('${firstMatch.key}');return false;" style="color:#60a5fa;text-decoration:none;">Open match →</a>`;
};

window.devResetStreamSeekTest = async function () {
    const statusEl = document.getElementById('dev-stream-status');
    const allMatches = await db.matches.toArray();
    if (!allMatches.length) { if (statusEl) statusEl.textContent = 'No matches in DB.'; return; }
    allMatches.sort((a, b) => a.matchNumber - b.matchNumber);
    const firstMatch = allMatches[0];
    await db.matches.update(firstMatch.key, { actualTime: null });
    if (statusEl) statusEl.textContent = `Match ${firstMatch.matchNumber} actualTime cleared. Re-sync schedule to fully restore.`;
};

window.applyTimeMachineSnapshot = async function () {
    const statusEl = document.getElementById('devTimeMachineStatus');
    const inputEl  = document.getElementById('devTimeMachineInput');
    const cutoff   = parseInt(inputEl?.value ?? '');
    if (isNaN(cutoff) || cutoff < 0) {
        if (statusEl) statusEl.innerHTML = '<span style="color:#f87171;">Enter a valid match number (≥ 0).</span>';
        return;
    }
    const eventKey = (document.getElementById('eventKeyInput')?.value ?? '').trim().toLowerCase();
    if (!eventKey) {
        if (statusEl) statusEl.innerHTML = '<span style="color:#f87171;">Set an event key first.</span>';
        return;
    }
    if (statusEl) statusEl.innerHTML = '<span style="color:#94a3b8;">Applying…</span>';

    let matchesReset = 0, teamsUpdated = 0, scoutRows = 0, timesUpdated = 0;

    // ── 1. Roll back db.matches: erase scores for qual matches > cutoff ──────
    const allMatches = await db.matches.toArray();
    const sortedAll  = [...allMatches].sort((a, b) => (a.matchNumber ?? 0) - (b.matchNumber ?? 0));

    for (const m of allMatches) {
        if ((m.matchNumber ?? 0) > cutoff && (m.redScore ?? -1) >= 0) {
            await db.matches.update(m.key, { redScore: -1, blueScore: -1, redBreakdown: null, blueBreakdown: null });
            matchesReset++;
        }
    }

    // ── 2. Set future predictedTime on post-cutoff matches ────────────────────
    const playedSorted = sortedAll.filter(m => (m.matchNumber ?? 0) <= cutoff && (m.redScore ?? -1) >= 0);
    const postCutoff   = sortedAll.filter(m => (m.matchNumber ?? 0) > cutoff);
    if (postCutoff.length) {
        let gap = 480;
        if (playedSorted.length >= 2) {
            const last = playedSorted[playedSorted.length - 1];
            const prev = playedSorted[playedSorted.length - 2];
            const tLast = last.predictedTime ?? last.time;
            const tPrev = prev.predictedTime ?? prev.time;
            if (tLast && tPrev && tLast > tPrev) gap = Math.max(300, Math.min(900, tLast - tPrev));
        }
        // Always anchor to real now so countdowns show positive values regardless of when the event was
        let t = Math.floor(Date.now() / 1000) + gap;
        for (const m of postCutoff) {
            await db.matches.update(m.key, { predictedTime: t });
            t += gap;
            timesUpdated++;
        }
    }

    // ── 3. Roll back db.teams: filter rawStatboticsData for this event ────────
    const allTeams = await db.teams.toArray();
    for (const team of allTeams) {
        const raw = team.rawStatboticsData ?? [];
        const filtered = raw.filter(m => {
            if (!m.match?.startsWith(eventKey + '_qm')) return true;
            const n = parseInt(m.match.slice(eventKey.length + 3));
            return isNaN(n) || n <= cutoff;
        });
        if (filtered.length === raw.length) continue;
        const played = filtered.filter(m => m.event === eventKey && m.epa?.post != null);
        const latestEPA = played.length ? played[played.length - 1].epa.post : team.currentEPA;
        await db.teams.update(team.teamNumber, { rawStatboticsData: filtered, currentEPA: latestEPA });
        teamsUpdated++;
    }

    // ── 4. Roll back scouting localStorage data (trim by match number) ────────
    try {
        const lsKey = `scoutingData_${eventKey}`;
        const rows = JSON.parse(localStorage.getItem(lsKey) ?? 'null');
        if (Array.isArray(rows)) {
            const before = rows.length;
            const kept = rows.filter(r => (r.matchNumber ?? 0) <= cutoff);
            scoutRows = before - kept.length;
            localStorage.setItem(lsKey, JSON.stringify(kept));
        }
    } catch {}

    // ── 5. Clear all event-scoped sync-state localStorage keys ───────────────
    for (const key of [
        `pitData_${eventKey}`,
        `tbaAlliances_${eventKey}`,
        `scoutingFusedStats_${eventKey}`,
        `archiveCoverage_${eventKey}`,
        `wlPreEventSnapshot_${eventKey}`,
    ]) localStorage.removeItem(key);
    // Draft state (not scoped to event but reflects post-draft selections)
    localStorage.removeItem('realDraftState');

    // ── 6. Reset module-level caches and dirty flags ──────────────────────────
    wlDetailCache       = null;
    wlPreEventCache     = null;
    watchListDirty      = true;
    wlMatchesRenderedFor = null;

    const parts = [
        `reset ${matchesReset} match score${matchesReset !== 1 ? 's' : ''}`,
        `set ${timesUpdated} future timestamp${timesUpdated !== 1 ? 's' : ''}`,
        `updated ${teamsUpdated} team${teamsUpdated !== 1 ? 's' : ''} in Statbotics history`,
        scoutRows ? `removed ${scoutRows} scouting row${scoutRows !== 1 ? 's' : ''}` : null,
    ].filter(Boolean).join(', ');
    if (statusEl) statusEl.innerHTML =
        `<span style="color:#4ade80;">✓ Done — ${parts}. ` +
        `Re-sync OPR, TBA Matches, and Statbotics to restore live data.</span>`;

    // Re-render the schedule so data-predicted-time attributes reflect the new timestamps
    await window.displaySchedule();
};

async function renderAlliancesTab() {
    const el = document.getElementById('tools-tab-alliances');
    if (!el) return;
    el.innerHTML = `<div style="color:#94a3b8;padding:12px;">Loading…</div>`;

    const eventKey = (document.getElementById('eventKeyInput')?.value ?? '').trim().toLowerCase();
    const gameConfig = eventKey ? getGameConfig(eventKey) : null;
    if (!gameConfig) {
        el.innerHTML = `<div style="color:#64748b;padding:12px;">Set an event key first.</div>`;
        return;
    }

    const [allTeamsArr, tbaTeamsArr, matchesArr] = await Promise.all([
        db.teams.toArray(), db.tbaTeams.toArray(), db.matches.toArray()
    ]);
    const teamsMap = Object.fromEntries(allTeamsArr.map(t => [t.teamNumber, t]));
    const tbaMap   = Object.fromEntries(tbaTeamsArr.map(t => [t.teamNumber, t]));
    const playedMatches = matchesArr.filter(m => (m.redScore ?? -1) >= 0);
    const { relResiduals, diffResiduals } = wlCollectResiduals(playedMatches, tbaMap, teamsMap);
    const oprSigmaRel = wlStd(relResiduals);
    const avg = allTeamsArr.reduce((s, t) => s + (t.currentEPA ?? 0), 0) / (allTeamsArr.length || 1);
    const effectiveThresholds = getEffectiveThresholds(gameConfig, eventKey);

    // Load alliance data: prefer TBA alliances, fall back to draft state
    let alliances = null;
    let source = '';
    const tbaRaw = localStorage.getItem(`tbaAlliances_${eventKey}`);
    if (tbaRaw) {
        try {
            alliances = JSON.parse(tbaRaw).map((a, i) => ({
                num: i + 1,
                teams: (a.picks ?? []).slice(0, 3).map(p => parseInt(p.replace('frc', ''))).filter(Boolean),
            }));
            source = 'TBA Alliances';
        } catch {}
    }
    if (!alliances) {
        const draft = loadDraftState();
        if (draft?.alliances) {
            alliances = draft.alliances.map((a, i) => ({
                num: i + 1,
                teams: [a.captain, a.pick1, a.pick2].filter(Boolean).map(Number),
            })).filter(a => a.teams.length > 0);
            source = 'Draft';
        }
    }
    if (!alliances || !alliances.length) {
        el.innerHTML = `<div style="color:#64748b;padding:12px;">
            No alliance data found. Load TBA alliances (Draft → Real Alliances → Load from TBA) or build a mock draft first.
        </div>`;
        return;
    }

    // Compute scores for all alliances first (needed for color coding and win%)
    const allianceData = alliances.map(({ num, teams }) => {
        const score    = wlAlliancePredictedScore(teams, tbaMap, teamsMap, avg, oprSigmaRel);
        const sigmaRel = wlAllianceSigmaRel(teams, teamsMap, score);
        const sigma    = sigmaRel * score;
        return { num, teams, score, sigma };
    });

    // Expected average across all alliances (for color coding like draft tab)
    const expectedAvg = allianceData.reduce((s, a) => s + a.score, 0) / (allianceData.length || 1);

    // Find which alliance (if any) contains 1768
    const ownAlliance = allianceData.find(a => a.teams.map(String).includes(OWN_TEAM));

    const winPctHeader = ownAlliance
        ? `<th style="padding:6px 8px;text-align:center;font-size:0.82em;color:#fbbf24;">Win% vs Them</th>` : '';

    const rows = allianceData.map(({ num, teams, score, sigma }) => {
        const pct = expectedAvg > 0 ? (score - expectedAvg) / expectedAvg : 0;
        const scoreColor = pct > 0.12 ? '#4ade80' : pct > 0.04 ? '#a3e635' : pct > -0.04 ? '#f8fafc' : pct > -0.12 ? '#fb923c' : '#ef4444';

        const teamLinks = teams.map(tn =>
            `<span onclick="viewTeamDetail(${tn},'overview')" style="cursor:pointer;color:#93c5fd;margin-right:6px;">${tn}</span>`
        ).join('');

        let winPctCell = '';
        if (ownAlliance) {
            if (ownAlliance.num === num) {
                winPctCell = `<td style="padding:6px 8px;text-align:center;color:#475569;font-size:0.8em;">—</td>`;
            } else {
                const own = ownAlliance;
                const diffSigma = Math.sqrt(own.sigma * own.sigma + sigma * sigma);
                const z = diffSigma > 0 ? (own.score - score) / diffSigma : 0;
                const prob = Math.round(btNormalCDF(z) * 100);
                const wColor = prob > 60 ? '#4ade80' : prob > 40 ? '#fbbf24' : '#f87171';
                winPctCell = `<td style="padding:6px 8px;text-align:center;color:${wColor};font-weight:600;">${prob}%</td>`;
            }
        }

        return `<tr style="border-bottom:1px solid #1e293b;${ownAlliance?.num === num ? 'background:#0f1e30;' : ''}">
            <td style="padding:6px 8px;text-align:center;color:#94a3b8;font-weight:700;">${num}</td>
            <td style="padding:6px 8px;">${teamLinks || '<span style="color:#475569;">—</span>'}</td>
            <td style="padding:6px 8px;text-align:center;color:${scoreColor};font-weight:700;">${score.toFixed(1)}</td>
            <td style="padding:6px 8px;text-align:center;color:#64748b;">±${sigma.toFixed(1)}</td>
            ${winPctCell}
        </tr>`;
    }).join('');

    el.innerHTML = `<div style="padding:12px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
            <h3 style="color:#e2e8f0;margin:0;">Alliance Estimates</h3>
            <span style="color:#64748b;font-size:0.82em;">Source: ${source} · ${playedMatches.length} played matches · ${isFinite(oprSigmaRel) ? `OPR CV ${(oprSigmaRel*100).toFixed(1)}%` : 'EPA only (no event data)'}</span>
        </div>
        <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;min-width:400px;">
            <thead><tr style="border-bottom:2px solid #334155;">
                <th style="padding:6px 8px;text-align:center;font-size:0.82em;color:#94a3b8;">#</th>
                <th style="padding:6px 8px;font-size:0.82em;color:#94a3b8;">Teams</th>
                <th style="padding:6px 8px;text-align:center;font-size:0.82em;color:#94a3b8;">Exp. Score</th>
                <th style="padding:6px 8px;text-align:center;font-size:0.82em;color:#94a3b8;">±SD</th>
                ${winPctHeader}
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>
        </div>
    </div>`;
}

async function renderDraft() {
    const allianceBody = document.getElementById('draftAllianceBody');
    const pickPanel = document.getElementById('draftPickPanel');
    const statusEl = document.getElementById('draftStatus');
    if (!allianceBody || !pickPanel) return;

    const [allTeams, allMatches, allTBATeams] = await Promise.all([db.teams.toArray(), db.matches.toArray(), db.tbaTeams.toArray()]);

    if (!allTeams.length) {
        if (statusEl) statusEl.textContent = 'No team data — sync team list/history or Statbotics Live first.';
        allianceBody.innerHTML = '';
        pickPanel.innerHTML = '<p style="color:#64748b;font-size:0.9em;padding:12px;">No data.</p>';
        return;
    }

    const teamInfoMap = Object.fromEntries(allTeams.map(t => [String(t.teamNumber), t]));
    const tbaTeamMap  = Object.fromEntries(allTBATeams.map(t => [String(t.teamNumber), t]));

    // Build scouting EPA map (same pattern as displayTBATeams / renderPickList)
    const scoutEPAMap = {};
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (eventKey) {
        const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
        if (rawStr) {
            const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
            const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
            if (processed?.config?.computeEPABreakdown) {
                const { config, byTeam } = processed;
                for (const [tn, rawRows] of Object.entries(byTeam)) {
                    const tbaEntry = tbaTeamMap[tn];
                    const scoutIgnoreKeys = tbaEntry?.scoutingIgnoreActive ? getTeamIgnoredKeys(tbaEntry) : [];
                    let { rows: deduped } = deduplicateTeamRows(rawRows);
                    let ignoredMatchNums = new Set();
                    if (scoutIgnoreKeys.length > 0) {
                        ignoredMatchNums = new Set(allMatches.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber));
                        deduped = deduped.filter(r => !ignoredMatchNums.has(r.matchNumber));
                    }
                    const rawStats = config.aggregateTeam(deduped);
                    const fusedResult = fusedCache?.teams?.[tn];
                    const effectiveFused = (fusedResult?.available && ignoredMatchNums.size > 0)
                        ? refilteredFusedStats(fusedResult, ignoredMatchNums) : fusedResult;
                    const isFused = !!(effectiveFused?.available && config.computeFusedEPABreakdown);
                    const breakdown = isFused
                        ? config.computeFusedEPABreakdown(effectiveFused.stats)
                        : config.computeEPABreakdown(rawStats);
                    scoutEPAMap[tn] = breakdown.total;
                }
            }
        }
    }

    // Initialize weight inputs from persisted weights on first render
    for (const [id, key] of [['wScout', 'scout'], ['wStatbotics', 'statbotics'], ['wOPR', 'opr']]) {
        const el = document.getElementById(id);
        if (el && !el.dataset.initialized) { el.value = draftWeights[key]; el.dataset.initialized = '1'; }
    }
    for (const [id, val] of [['draftNumAlliances', draftNumAlliances], ['draftPicksPerAlliance', draftPicksPerAlliance]]) {
        const el = document.getElementById(id);
        if (el && !el.dataset.initialized) { el.value = val; el.dataset.initialized = '1'; }
    }

    const hasScout = Object.keys(scoutEPAMap).length > 0;
    const hasOPR   = allTBATeams.length > 0;

    // Normalized blended EPA — falls back gracefully when a source is unavailable
    const blendedEPA = tn => {
        const t = teamInfoMap[tn];
        if (!t) return 0;
        const statVal  = parseFloat(t.analysis?.ceiling ?? t.currentEPA ?? 0) || 0;
        const oprVal   = tbaTeamMap[tn]?.opr ?? null;
        const scoutVal = scoutEPAMap[tn]   ?? null;
        const sources  = [
            { w: draftWeights.statbotics, v: statVal  },
            { w: draftWeights.opr,        v: oprVal   },
            { w: draftWeights.scout,      v: scoutVal },
        ].filter(s => s.v != null && s.w > 0);
        if (sources.length === 0) return statVal;
        const totalW = sources.reduce((s, x) => s + x.w, 0);
        return totalW > 0 ? sources.reduce((s, x) => s + x.v * (x.w / totalW), 0) : statVal;
    };

    // Update weight info indicator
    const infoEl = document.getElementById('draftWeightInfo');
    if (infoEl) {
        const missing = [];
        if (draftWeights.scout > 0 && !hasScout) missing.push('scouting N/A');
        if (draftWeights.opr   > 0 && !hasOPR)   missing.push('OPR N/A');
        infoEl.textContent = missing.join(' · ');
    }

    // RP totals (mirrors renderPickList logic)
    const rpTotals = {};
    const played = allMatches.filter(m => (m.redScore ?? -1) >= 0);
    for (const m of played) {
        const redWon = m.redScore > m.blueScore, tie = m.redScore === m.blueScore, blueWon = !redWon && !tie;
        const bonusRP = bd => bd ? ((bd.energizedAchieved ? 1 : 0) + (bd.superchargedAchieved ? 1 : 0) + (bd.traversalAchieved ? 1 : 0)) : 0;
        const rRP = m.redBreakdown?.rp ?? ((redWon ? 3 : tie ? 1 : 0) + bonusRP(m.redBreakdown));
        const bRP = m.blueBreakdown?.rp ?? ((blueWon ? 3 : tie ? 1 : 0) + bonusRP(m.blueBreakdown));
        (m.red || []).forEach(t => { rpTotals[t] = (rpTotals[t] || 0) + rRP; });
        (m.blue || []).forEach(t => { rpTotals[t] = (rpTotals[t] || 0) + bRP; });
    }
    // RP-ranked list for captain auto-fill
    draftRPRankedTeams = Object.entries(rpTotals).sort(([, a], [, b]) => b - a).map(([tn]) => tn);
    const rpSet = new Set(draftRPRankedTeams);
    allTeams.forEach(t => { if (!rpSet.has(String(t.teamNumber))) draftRPRankedTeams.push(String(t.teamNumber)); });
    if (!played.length) {
        const po = (() => { try { return JSON.parse(localStorage.getItem('pickListOrder')) || []; } catch { return []; } })()
            .filter(t => t !== '---separator---');
        if (po.length) draftRPRankedTeams = po;
    }

    // Quick tier by blended EPA rank (for pick panel color coding)
    const sortedByEPA = allTeams.slice().sort((a, b) => blendedEPA(String(b.teamNumber)) - blendedEPA(String(a.teamNumber)));
    const epaRankOf = Object.fromEntries(sortedByEPA.map((t, i) => [String(t.teamNumber), i]));
    const quickTier = tn => { const r = epaRankOf[tn] ?? 99; return r < 8 ? 'S' : r < 20 ? 'A' : r < 32 ? 'B' : 'C'; };
    const TIER_CLR = { S: '#f59e0b', A: '#4ade80', B: '#a855f7', C: '#64748b' };

    // Load/init state; auto-fill first captain
    let state = loadDraftState() || freshDraftState();
    draftFillCaptain(state);
    saveDraftState(state);

    const picked = buildDraftPickedSet(state.alliances);
    const isReal = draftMode === 'real';
    const isDone = isReal || state.currentAlliance >= draftNumAlliances || state.currentAlliance < 0;
    const isUser = !isDone && state.alliances[state.currentAlliance]?.captain !== null;

    // Sync mode toggle appearance and control visibility
    const mockBtn = document.getElementById('draftModeMockBtn');
    const realBtn = document.getElementById('draftModeRealBtn');
    const mockControls = document.getElementById('draftMockControls');
    const realControls = document.getElementById('draftRealControls');
    if (mockBtn) { mockBtn.style.background = isReal ? 'transparent' : '#1e293b'; mockBtn.style.color = isReal ? '#64748b' : '#f8fafc'; }
    if (realBtn) { realBtn.style.background = isReal ? '#1e293b' : 'transparent'; realBtn.style.color = isReal ? '#f8fafc' : '#64748b'; }
    if (mockControls) mockControls.style.display = isReal ? 'none' : 'flex';
    if (realControls) realControls.style.display = isReal ? 'flex' : 'none';

    if (statusEl) {
        if (isReal) {
            const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
            const hasAlliances = eventKey && !!localStorage.getItem(`tbaAlliances_${eventKey}`);
            statusEl.textContent = hasAlliances ? 'Showing real alliance selection.' : 'No alliance data — load from TBA above.';
        } else if (isDone) {
            statusEl.textContent = 'Draft complete.';
        } else if (isUser) {
            const ordinals = ['1st','2nd','3rd','4th','5th'];
            const which = (ordinals[state.currentRound - 1] || `${state.currentRound}th`) + ' pick';
            statusEl.textContent = `Alliance ${state.currentAlliance + 1} selecting ${which} — click a team →`;
        } else {
            statusEl.textContent = 'Filling captain…';
        }
    }

    // ── Alliance table ──
    const epaOf = blendedEPA;

    const topNCount = draftNumAlliances * (draftPicksPerAlliance + 1);
    const topNsum = allTeams.map(t => epaOf(String(t.teamNumber))).sort((a, b) => b - a).slice(0, topNCount).reduce((s, v) => s + v, 0);
    const expectedTotal = topNsum / draftNumAlliances;

    const teamChip = tn => {
        if (!tn) return `<span style="color:#1e293b;">—</span>`;
        return `<span style="font-weight:800;color:#f8fafc;cursor:pointer;text-decoration:underline;text-underline-offset:2px;"
            onclick="event.stopPropagation();viewTeamDetail(${parseInt(tn)})">${tn}</span>`;
    };
    // Update table header to match current pick count
    const allianceHead = document.getElementById('draftAllianceHead');
    if (allianceHead) {
        const thStyle = 'padding:10px 8px;border-bottom:2px solid #334155;color:#94a3b8;text-align:center;';
        const pickHeaders = Array.from({ length: draftPicksPerAlliance }, (_, k) =>
            `<th style="${thStyle}">Pick ${k + 1}</th>`).join('');
        allianceHead.innerHTML = `<tr>
            <th style="width:50px;${thStyle}">Alliance</th>
            <th style="${thStyle}">Captain</th>
            ${pickHeaders}
            <th style="${thStyle}">EPA${localEpaBadge()}</th>
        </tr>`;
    }

    allianceBody.innerHTML = state.alliances.map((a, i) => {
        const { solid, bg } = DRAFT_ALLIANCE_COLORS[i % DRAFT_ALLIANCE_COLORS.length];
        const isActive = !isDone && state.currentAlliance === i;
        const rowBg = isActive ? bg : (i % 2 ? '#080d16' : '#0f172a');
        const leftBorder = isActive ? `box-shadow:inset 3px 0 0 ${solid};` : '';
        const activePickIdx = isActive ? state.currentRound - 1 : -1;
        const cell = (content, active) =>
            `<td style="padding:11px 10px;border-bottom:1px solid #1e293b;text-align:center;` +
            (active ? `outline:1px dashed ${solid};outline-offset:-3px;` : '') + `">${content}</td>`;

        const picks = Array.isArray(a.picks) ? a.picks : [a.pick1 ?? null, a.pick2 ?? null];
        const allFilled = a.captain && picks.every(Boolean);
        const presentMembers = [a.captain, ...picks].filter(Boolean);
        const total = presentMembers.length > 0 ? presentMembers.reduce((s, tn) => s + epaOf(tn), 0) : null;
        const pct = allFilled && expectedTotal > 0 ? (total - expectedTotal) / expectedTotal : null;
        const totalColor = pct != null
            ? (pct > 0.12 ? '#4ade80' : pct > 0.04 ? '#a3e635' : pct > -0.04 ? '#f8fafc' : pct > -0.12 ? '#fb923c' : '#ef4444')
            : (total != null ? '#94a3b8' : '#334155');

        const pickCells = picks.map((pick, pi) =>
            cell(teamChip(pick), isActive && pi === activePickIdx && !pick)
        ).join('');

        return `<tr style="background:${rowBg};${leftBorder}">
            <td style="padding:11px 8px;border-bottom:1px solid #1e293b;text-align:center;">
                <span style="color:#f8fafc;font-weight:900;font-size:1.2em;">${i + 1}</span>
            </td>
            ${cell(teamChip(a.captain), false)}
            ${pickCells}
            <td style="padding:11px 10px;border-bottom:1px solid #1e293b;text-align:center;color:${totalColor};font-weight:700;">
                ${total != null ? total.toFixed(1) + localEpaBadge() : '—'}
            </td>
        </tr>`;
    }).join('');

    // ── Pick panel ──
    const rawOrder = (() => {
        try { return JSON.parse(localStorage.getItem('pickListOrder')) || []; } catch { return []; }
    })().filter(t => t !== '---separator---');
    const hasPickList = rawOrder.length > 0;
    const available = rawOrder.filter(tn => !picked.has(tn) && teamInfoMap[tn]);
    const avSet = new Set(available);
    allTeams.forEach(t => { const tn = String(t.teamNumber); if (!picked.has(tn) && !avSet.has(tn)) available.push(tn); });
    if (!hasPickList) available.sort((a, b) => epaOf(b) - epaOf(a));

    pickPanel.innerHTML = available.map((tn, idx) => {
        const t = teamInfoMap[tn];
        if (!t) return '';
        const tierColor = TIER_CLR[quickTier(tn)];
        const epaVal = blendedEPA(tn);
        const epaStr = epaVal > 0 ? epaVal.toFixed(1) : '';
        return `<div class="draft-pick-item" ${isUser ? `onclick="draftPick('${tn}')"` : ''}
            style="padding:9px 12px;border-bottom:1px solid #1e293b;display:flex;align-items:center;gap:8px;
            border-left:3px solid ${tierColor};cursor:${isUser ? 'pointer' : 'default'};${!isUser ? 'opacity:0.45;' : ''}">
            <span style="color:#475569;font-size:0.75em;min-width:22px;text-align:right;font-weight:600;">${idx + 1}</span>
            <strong style="color:#f8fafc;font-size:0.95em;">${tn}</strong>
            <span style="color:#94a3b8;font-size:0.82em;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${t.teamName || ''}</span>
            ${epaStr ? `<span style="color:#475569;font-size:0.75em;font-weight:600;">${epaStr}${localEpaBadge(t)}</span>` : ''}
        </div>`;
    }).join('') || '<p style="color:#64748b;font-size:0.9em;padding:12px;">All teams placed.</p>';
}

window.updateDetailBackButton = function () {
    const btn = document.getElementById('detailBackBtn');
    if (!btn) return;

    // Change the label based on where the user was previously
    if (window.previousView === 'matchPrepView') {
        btn.innerText = '← Back';
    } else if (window.previousView === 'teamView') {
        btn.innerText = '← Back to Statbotics';
    } else {
        btn.innerText = '← Back';
    }
};

window.goBack = function () {
    if (document.body.classList.contains('split-ui')) {
        document.getElementById('teamDetailView').style.display = 'none';
        if (!popRightPanel()) {
            document.getElementById('splitRightPanel').style.display = 'flex';
        }
        return;
    }
    // Always hide the overlay explicitly — it may have been opened in split mode
    // before the user switched to desktop, leaving currentView pointing at the
    // left-panel view instead of teamDetailView.
    document.getElementById('teamDetailView').style.display = 'none';
    window.switchView(window.previousView);
};





let performanceChart = null;
let matchesChartInstance = null;
let rpTimelineChart = null;
let wlMatchesRenderedFor = null;
let activeTeamNumber = null;
let activeTeamData = null;
let activeTBAData = null;
let lastDetailTab = 'overview';
let lastDetailDataSubTab = 'epa';

const PHOTO_STYLE = 'width:220px; min-width:220px; height:auto; max-height:320px; object-fit:contain; border-radius:8px; border:1px solid #334155; background:#0f172a; display:block;';

async function renderOverview(team, tbaTeam) {
    const el = document.getElementById('overviewContent');
    if (!el) return;

    const fmt = (v, d = 1) => (v != null && v !== '—') ? parseFloat(v).toFixed(d) : '—';
    const analysis = team.analysis || {};
    const ceilStr = fmt(analysis.ceiling);
    const lb = analysis.lowerBound, ub = analysis.upperBound;
    const ciStr = (lb != null && lb !== '—' && ub != null && ub !== '—') ? `${lb} – ${ub}` : null;

    // Compute effective OPR (mirrors displayTBATeams logic)
    let effOPRVal = tbaTeam?.opr ?? null;
    let oprSuffix = '';
    // Load all data needed for tier ranking, OPR recomputation, and scouting ignore filtering
    const [allTeamsForOvTier, allTBAForOvTier, allMatches] = await Promise.all([
        db.teams.toArray(), db.tbaTeams.toArray(), db.matches.toArray(),
    ]);
    const tbaOvMap = Object.fromEntries(allTBAForOvTier.map(t => [t.teamNumber, t]));

    if (tbaTeam) {
        const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
        const indivKeys = getTeamIgnoredKeys(tbaTeam);
        if (indivKeys.some(k => !globalIgnored.has(k)) && tbaTeam.adjustedOPR != null) {
            effOPRVal = tbaTeam.adjustedOPR;
            oprSuffix = ' <span style="color:#fbbf24;font-size:0.65em;font-weight:600;">ADJ</span>';
        } else if (globalIgnored.size > 0) {
            const teamNums = allTBAForOvTier.map(t => t.teamNumber);
            const activePlayed = allMatches.filter(m =>
                (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 && !globalIgnored.has(m.key)
            );
            const recomputed = computeLocalOPR(activePlayed, teamNums);
            if (recomputed) {
                const idx = teamNums.indexOf(parseInt(team.teamNumber));
                if (idx >= 0) effOPRVal = recomputed[idx];
                oprSuffix = ' <span style="color:#f97316;font-size:0.65em;font-weight:600;">ADJ</span>';
            }
        }
    }
    const ceilOf = t => t.analysis?.ceiling != null ? parseFloat(t.analysis.ceiling) : (t.currentEPA || 0);
    const tierOverall    = epaRankTier(allTeamsForOvTier, ceilOf(team), ceilOf);
    const tierAuto       = epaRankTier(allTeamsForOvTier, team.autoEPA    || 0, t => t.autoEPA    || 0);
    const tierTeleop     = epaRankTier(allTeamsForOvTier, team.teleopEPA  || 0, t => t.teleopEPA  || 0);
    const tierEndgame    = epaRankTier(allTeamsForOvTier, team.endgameEPA || 0, t => t.endgameEPA || 0);
    const tierOPR        = effOPRVal != null
        ? epaRankTier(allTBAForOvTier, effOPRVal, t => tbaOvMap[t.teamNumber]?.opr ?? 0)
        : null;
    const tierAutoOPR    = tbaTeam?.autoOPR    != null ? epaRankTier(allTBAForOvTier, tbaTeam.autoOPR,    t => tbaOvMap[t.teamNumber]?.autoOPR    ?? 0) : null;
    const tierTeleopOPR  = tbaTeam?.teleopOPR  != null ? epaRankTier(allTBAForOvTier, tbaTeam.teleopOPR,  t => tbaOvMap[t.teamNumber]?.teleopOPR  ?? 0) : null;
    const tierEndgameOPR = tbaTeam?.endgameOPR != null ? epaRankTier(allTBAForOvTier, tbaTeam.endgameOPR, t => tbaOvMap[t.teamNumber]?.endgameOPR ?? 0) : null;

    // Scouting EPA breakdown for this team and all scouted teams (for tier ranking)
    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const tn = String(team.teamNumber);
    let scoutBreakdown = null;
    let allScoutBreakdowns = [];
    let scoutIsFused = false;
    let scoutIsAdj  = false;
    if (eventKey) {
        const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
        if (rawStr) {
            const fusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
            const processed = processScoutingData(eventKey, JSON.parse(rawStr), getScoutingColumnOverrides(eventKey));
            if (processed?.config?.computeEPABreakdown) {
                const { config, byTeam } = processed;
                for (const [teamNum, rawRows] of Object.entries(byTeam)) {
                    const tbaEntry = tbaOvMap[parseInt(teamNum)];
                    const scoutIgnoreKeys = tbaEntry?.scoutingIgnoreActive ? getTeamIgnoredKeys(tbaEntry) : [];
                    let { rows: deduped } = deduplicateTeamRows(rawRows);
                    let ignoredMatchNums = new Set();
                    if (scoutIgnoreKeys.length > 0) {
                        ignoredMatchNums = new Set(allMatches.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber));
                        deduped = deduped.filter(r => !ignoredMatchNums.has(r.matchNumber));
                    }
                    const rawStats = config.aggregateTeam(deduped);
                    const fusedResult = fusedCache?.teams?.[teamNum];
                    const effectiveFused = (fusedResult?.available && ignoredMatchNums.size > 0)
                        ? refilteredFusedStats(fusedResult, ignoredMatchNums) : fusedResult;
                    const fused = !!(effectiveFused?.available && config.computeFusedEPABreakdown);
                    const bd = fused ? config.computeFusedEPABreakdown(effectiveFused.stats) : config.computeEPABreakdown(rawStats);
                    allScoutBreakdowns.push({ ...bd });
                    if (teamNum === tn) { scoutBreakdown = bd; scoutIsFused = fused; scoutIsAdj = ignoredMatchNums.size > 0; }
                }
            }
        }
    }
    const tierScoutAuto    = scoutBreakdown ? epaRankTier(allScoutBreakdowns, scoutBreakdown.auto    ?? 0, b => b.auto    ?? 0) : null;
    const tierScoutTeleop  = scoutBreakdown ? epaRankTier(allScoutBreakdowns, scoutBreakdown.teleop  ?? 0, b => b.teleop  ?? 0) : null;
    const tierScoutEndgame = scoutBreakdown ? epaRankTier(allScoutBreakdowns, scoutBreakdown.endgame ?? 0, b => b.endgame ?? 0) : null;
    const tierScoutTotal   = scoutBreakdown ? epaRankTier(allScoutBreakdowns, scoutBreakdown.total   ?? 0, b => b.total   ?? 0) : null;

    const photoId = `ov-photo-${team.teamNumber}`;
    const photoHtml = team.photoUrl
        ? `<img id="${photoId}" src="${team.photoUrl}" alt="Team ${team.teamNumber}"
               style="${PHOTO_STYLE} cursor:zoom-in; flex-shrink:0;"
               onclick="openLightbox('${team.photoUrl}')"
               onerror="this.style.display='none'">`
        : `<div id="${photoId}" style="width:220px; min-width:220px; height:220px; border-radius:8px; border:1px solid #334155; background:#1e293b; display:flex; align-items:center; justify-content:center; color:#334155; font-size:2.5em; font-weight:800; flex-shrink:0;">${team.teamNumber}</div>`;

    const sectionLabel = text =>
        `<div style="color:#64748b; font-size:0.72em; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin:20px 0 8px;">${text}</div>`;

    const placeholder = text =>
        `<div style="background:#1e293b; padding:20px; border-radius:8px; border:1px dashed #334155;">
            <p style="color:#475569; font-style:italic; margin:0; font-size:0.9em;">${text}</p>
        </div>`;

    const hasCompOPR = !!tbaTeam && tbaTeam.autoOPR != null;

    // Table cell helpers
    // estCol: true for the Statbotics/local-EPA column — adds Est badge when local mode is on
    const cell = (val, tier, estCol = false) => {
        if (val == null) return `<td style="text-align:right;padding:8px 12px;border-bottom:1px solid #1e293b;color:#475569;font-size:0.9em;">—</td>`;
        return `<td style="text-align:right;padding:8px 12px;border-bottom:1px solid #1e293b;white-space:nowrap;">
            <span style="font-weight:700;color:#f1f5f9;margin-right:4px;">${fmt(val)}</span>${tier ? tierBadge(tier) : ''}${estCol ? localEpaBadge(team) : ''}
        </td>`;
    };
    const rowLabel = text =>
        `<td style="padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:0.8em;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap;">${text}</td>`;

    const scoutHeader = `<img src="sheets.png" style="height:11px;vertical-align:middle;margin-right:3px;opacity:0.7;">Scouting${scoutIsFused ? ' <span style="color:#34d399;font-size:0.85em;vertical-align:middle;">●</span>' : ''}${scoutIsAdj ? ' <span style="color:#fbbf24;font-size:0.75em;font-weight:700;">ADJ</span>' : ''}`;

    el.innerHTML = `
        <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; margin-bottom:24px;">
            ${photoHtml}
            <div style="flex:1; min-width:180px;">
                <div style="font-size:1.8em; font-weight:800; color:#f8fafc; line-height:1.15;">${team.teamName || `Team ${team.teamNumber}`}</div>
                <div style="color:#64748b; font-size:1.05em; margin-top:4px;">Team #${team.teamNumber}</div>
                <div style="display:flex; gap:16px; margin-top:14px; flex-wrap:wrap;">
                    <a href="https://www.thebluealliance.com/team/${team.teamNumber}/${team.eventKey?.slice(0,4) || ''}" target="_blank"
                        style="color:#3b82f6; text-decoration:none; font-size:0.9em; display:inline-flex; align-items:center; gap:4px;"><img src="tba.png" class="source-logo" style="margin:0;">View on TBA ↗</a>
                    <a href="https://www.statbotics.io/team/${team.teamNumber}/${team.eventKey?.slice(0,4) || ''}" target="_blank"
                        style="color:#3b82f6; text-decoration:none; font-size:0.9em; display:inline-flex; align-items:center; gap:4px;"><img src="statbotics.ico" class="source-logo" style="margin:0;">View on Statbotics ↗</a>
                </div>
            </div>
        </div>

        ${sectionLabel('Performance')}
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.9em;border-radius:8px;overflow:hidden;border:1px solid #334155;min-width:340px;">
            <thead>
                <tr style="background:#1e293b;border-bottom:2px solid #334155;">
                    <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;"></th>
                    <th style="text-align:right;padding:10px 12px;color:#64748b;font-weight:600;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">
                        <img src="statbotics.ico" style="height:11px;vertical-align:middle;margin-right:3px;opacity:0.7;">${isLocalEpaEnabled() && !teamHasSbEventData(team) ? 'Local Est' : 'Statbotics'}
                    </th>
                    <th style="text-align:right;padding:10px 12px;color:#64748b;font-weight:600;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">
                        <img src="tba.png" class="source-logo" style="height:11px;margin:0 3px 0 0;vertical-align:middle;opacity:0.7;">TBA OPR
                    </th>
                    <th style="text-align:right;padding:10px 12px;color:#64748b;font-weight:600;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">${scoutHeader}</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    ${rowLabel('Auto')}
                    ${cell(team.autoEPA,                              tierAuto,       true)}
                    ${cell(hasCompOPR ? tbaTeam.autoOPR    : null,   tierAutoOPR)}
                    ${cell(scoutBreakdown?.auto,                      tierScoutAuto)}
                </tr>
                <tr>
                    ${rowLabel('Teleop')}
                    ${cell(team.teleopEPA,                            tierTeleop,     true)}
                    ${cell(hasCompOPR ? tbaTeam.teleopOPR  : null,   tierTeleopOPR)}
                    ${cell(scoutBreakdown?.teleop,                    tierScoutTeleop)}
                </tr>
                <tr>
                    ${rowLabel('Endgame')}
                    ${cell(team.endgameEPA,                           tierEndgame,    true)}
                    ${cell(hasCompOPR ? tbaTeam.endgameOPR : null,   tierEndgameOPR)}
                    ${cell(scoutBreakdown?.endgame,                   tierScoutEndgame)}
                </tr>
                <tr>
                    ${rowLabel('Total')}
                    <td style="text-align:right;padding:8px 12px;border-bottom:1px solid #1e293b;white-space:nowrap;">
                        ${ceilStr !== '—'
                            ? `<span style="font-weight:700;color:#4ade80;margin-right:4px;">${ceilStr}</span>${tierBadge(tierOverall)}`
                            : `<span style="font-weight:700;color:#f1f5f9;margin-right:4px;">${fmt(team.currentEPA)}</span>${tierBadge(tierOverall)}`}
                        ${localEpaBadge(team)}
                    </td>
                    <td style="text-align:right;padding:8px 12px;border-bottom:1px solid #1e293b;white-space:nowrap;">
                        ${effOPRVal != null
                            ? `<span style="font-weight:700;color:#f1f5f9;margin-right:4px;">${fmt(effOPRVal)}</span>${tierOPR ? tierBadge(tierOPR) : ''}${oprSuffix}`
                            : `<span style="color:#475569;font-size:0.9em;">—</span>`}
                    </td>
                    ${cell(scoutBreakdown?.total,                     tierScoutTotal)}
                </tr>
            </tbody>
        </table>
        </div>

        ${sectionLabel('Notes')}
        <div id="overview-notes-section"></div>
    `;

    renderNoteSection(team.teamNumber);
    if (!team.photoUrl) fetchAndCacheTeamPhoto(team.teamNumber, photoId, team.eventKey?.slice(0, 4));
}

async function fetchAndCacheTeamPhoto(teamNumber, photoElId, year) {
    try {
        const media = await fetchTBA(`/team/frc${teamNumber}/media/${year}`);
        if (!media || !media.length) return;
        const candidates = media.filter(m => m.direct_url);
        if (!candidates.length) return;
        const pick = candidates.find(m => m.preferred) || candidates[0];
        const url = pick.direct_url;

        // Download and convert to base64 so the photo is available offline
        let dataUrl = url;
        try {
            const resp = await fetch(url);
            const blob = await resp.blob();
            dataUrl = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        } catch (_) { /* keep external URL as fallback if download fails */ }

        const el = document.getElementById(photoElId);
        if (el) {
            const img = document.createElement('img');
            img.id = photoElId;
            img.src = dataUrl;
            img.alt = `Team ${teamNumber}`;
            img.style.cssText = PHOTO_STYLE + ' cursor:zoom-in; flex-shrink:0;';
            img.onclick = () => window.openLightbox(dataUrl);
            img.onerror = () => img.style.display = 'none';
            el.replaceWith(img);
        }
        await db.teams.update(parseInt(teamNumber), { photoUrl: dataUrl });
    } catch (_) { }
}

function renderPitTab(teamNumber) {
    const container = document.getElementById('tab-pit-data');
    if (!container) return;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    const rawStr = eventKey ? localStorage.getItem(`pitData_${eventKey}`) : null;

    if (!rawStr) {
        container.innerHTML = `<p style="color:#64748b;font-style:italic;padding:16px 0;">No pit scouting data loaded. Sync a pit sheet from the Home tab.</p>`;
        return;
    }

    const rows = JSON.parse(rawStr);
    const tn = String(teamNumber);
    // Find row(s) where any column's value matches the team number (numeric or string)
    const teamRows = rows.filter(row =>
        Object.values(row).some(v => String(v).trim() === tn)
    );

    if (!teamRows.length) {
        container.innerHTML = `<p style="color:#64748b;font-style:italic;padding:16px 0;">No pit data found for team ${teamNumber}.</p>`;
        return;
    }

    const SKIP_VALUE = v => v == null || String(v).trim() === '';
    container.innerHTML = teamRows.map((row, idx) => {
        const fields = Object.entries(row).filter(([, v]) => !SKIP_VALUE(v));
        const cells = fields.map(([k, v]) => `
            <div style="padding:8px 12px;border-bottom:1px solid #1e293b;">
                <div style="color:#64748b;font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">${k}</div>
                <div style="color:#f1f5f9;font-size:0.92em;">${String(v)}</div>
            </div>`).join('');
        const heading = teamRows.length > 1
            ? `<div style="color:#94a3b8;font-size:0.8em;font-weight:700;padding:8px 12px;background:#0f172a;border-bottom:1px solid #334155;">Entry ${idx + 1}</div>`
            : '';
        return `<div style="background:#1e293b;border-radius:8px;overflow:hidden;margin-bottom:12px;">${heading}${cells}</div>`;
    }).join('');
}

window.switchDetailTab = async function (tab) {
    // Destroy charts when navigating away from Matches tab
    if (lastDetailTab === 'matches' && tab !== 'matches') {
        if (rpTimelineChart)     { rpTimelineChart.destroy();     rpTimelineChart     = null; }
        if (matchesChartInstance){ matchesChartInstance.destroy(); matchesChartInstance = null; }
        wlMatchesRenderedFor = null;
    }
    lastDetailTab = tab;
    const tabs = ['overview', 'matches', 'data', 'routes'];
    tabs.forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        if (el) el.style.display = t === tab ? 'block' : 'none';
    });
    document.querySelectorAll('#teamDetailTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', tabs[i] === tab);
    });
    if (tab === 'overview' && activeTeamData) {
        await renderOverview(activeTeamData, activeTBAData);
    }
    if (tab === 'matches' && activeTeamData) {
        await renderWLMatchesTab(activeTeamData.teamNumber);
    }
    if (tab === 'data' && activeTeamData) {
        await switchDetailDataSubTab(lastDetailDataSubTab);
    }
    if (tab === 'routes' && activeTeamData) {
        // Lazily, on open only: a canvas cannot size itself while its pane is
        // display:none -- the same reason performanceChart is deferred.
        await renderTeamRoutesTab(activeTeamData.teamNumber);
    }
};

let routesRenderedFor = null;
// Off by default: routes belong to the event the user selected, same as every other
// surface in the app. The toggle exists because the opposite failure is real too -- a
// team whose only tracked match is at another event would otherwise show "no tracked
// matches" and look like the pipeline lost it. Scoped by default, never hidden silently:
// the count of what is being withheld is always on screen.
let routesShowAllEvents = false;

window.toggleRoutesAllEvents = function () {
    routesShowAllEvents = !routesShowAllEvents;
    routesRenderedFor = null;          // memo covers the toggle, so force a re-render
    if (activeTeamData) renderTeamRoutesTab(activeTeamData.teamNumber);
};

// Auto only. The match-detail view has had this since the scrubber was built, but the
// small-multiples grid on this tab is the surface where it actually earns its keep:
// comparing one team's opening 20 s across every match at the event is the question
// auto routes are FOR, and a full-match trace buries it under 140 s of teleop.
//
// Per-tab state rather than shared with the match view, because the two are looked at
// for different reasons and a toggle that follows you between them is a surprise.
let routesAutoOnly = false;

window.toggleRoutesAuto = function () {
    routesAutoOnly = !routesAutoOnly;
    routesRenderedFor = null;
    if (activeTeamData) renderTeamRoutesTab(activeTeamData.teamNumber);
};
let _tracksManifest = null;
let _tracksManifestAt = 0;
// The manifest is the freshness signal for every cached track doc (see
// loadMatchTracks), so caching it for the whole session would defeat the revalidation
// it enables: with the app left open during an event, the stamp it compares against
// would never move. A short TTL instead. 60s matches rtrack.watch's poll interval --
// nothing can appear faster than the watcher publishes it -- and the file is a few KB.
const TRACKS_MANIFEST_TTL_MS = 60_000;

/**
 * What tracks exist, from public/tracks/index.json.
 *
 * This exists because the app has no other way to find them. `db.matches` holds
 * QUALIFICATION MATCHES ONLY (main.js ~4173 skips comp_level !== 'qm'), so every
 * playoff match is invisible to the schedule and to viewMatchDetail. The two matches
 * published so far are finals, so without the manifest there is no route into them at
 * all. The manifest also carries each match's team list, so this can filter by team
 * before downloading anything.
 */
async function loadTracksManifest(force = false) {
    const fresh = Date.now() - _tracksManifestAt < TRACKS_MANIFEST_TTL_MS;
    if (_tracksManifest && fresh && !force) return _tracksManifest;
    try {
        const url = `${import.meta.env.BASE_URL}tracks/index.json`;
        const head = await fetch(url, { method: 'HEAD' });
        const ct = head.headers.get('content-type') || '';
        if (!head.ok || !ct.includes('json')) { _tracksManifestAt = Date.now(); return (_tracksManifest = { matches: [] }); }
        const resp = await fetch(url);
        if (!resp.ok) { _tracksManifestAt = Date.now(); return (_tracksManifest = { matches: [] }); }
        const doc = await resp.json();
        _tracksManifest = (doc && Array.isArray(doc.matches)) ? doc : { matches: [] };
    } catch { _tracksManifest = { matches: [] }; }
    _tracksManifestAt = Date.now();
    return _tracksManifest;
}
window.loadTracksManifest = loadTracksManifest;

// Small-multiples: this team's route in every tracked match.
async function renderTeamRoutesTab(teamNumber) {
    const host = document.getElementById('tab-routes');
    if (!host) return;
    const team = String(teamNumber);
    const evKey = (document.getElementById('eventKeyInput')?.value || '')
        .trim().toLowerCase();
    // The memo has to cover the event key and the toggle, not just the team: both change
    // what this tab should show while the team stays the same.
    const memo = `${team}|${evKey}|${routesShowAllEvents}|${routesAutoOnly}`;
    if (routesRenderedFor === memo) return;   // memo, mirroring wlMatchesRenderedFor
    // Set AFTER the loads below, not here: loadMatchTracks clears this memo whenever it
    // caches a new match, so claiming it up front would have it cleared by our own
    // fetches and the tab would re-render on every open.
    host.innerHTML = `<p style="color:#94a3b8; margin-top:18px;">Loading…</p>`;

    const man = await loadTracksManifest();
    const forTeam = (man.matches || [])
        .filter(r => (r.teams || []).map(String).includes(team))
        .sort((a, b) => String(a.key).localeCompare(String(b.key)));
    // Event comes from the match key's prefix when the manifest does not carry one --
    // rtrack names every match <event>_<comp><n>, which is the same split export.py uses.
    const evOf = r => String(r.eventKey || String(r.key).split('_')[0] || '');
    const inEvent = evKey ? forTeam.filter(r => evOf(r) === evKey) : forTeam;
    const elsewhere = forTeam.length - inEvent.length;
    const want = (routesShowAllEvents || !evKey) ? forTeam : inEvent;

    // Shown whenever another event has tracks for this team, in BOTH toggle states, so
    // the tab never just looks empty and never silently mixes events either.
    const allEventsBox = elsewhere > 0 ? `
        <label style="display:inline-flex; align-items:center; gap:7px;
                      color:#94a3b8; font-size:12px; cursor:pointer;">
          <input type="checkbox" id="routesAllEvents" ${routesShowAllEvents ? 'checked' : ''}
                 onchange="toggleRoutesAllEvents()" style="cursor:pointer;">
          Show ${elsewhere} match${elsewhere === 1 ? '' : 'es'} from other events
        </label>` : '';
    // Always offered, even with nothing loaded yet, so the empty state still tells the
    // reader the view exists.
    const autoBox = `
        <label style="display:inline-flex; align-items:center; gap:7px;
                      color:#94a3b8; font-size:12px; cursor:pointer;">
          <input type="checkbox" id="routesAutoOnly" ${routesAutoOnly ? 'checked' : ''}
                 onchange="toggleRoutesAuto()" style="cursor:pointer;">
          Auto only
        </label>`;
    const toggle = `<div style="display:flex; flex-wrap:wrap; gap:8px 18px;
                                align-items:center; margin-top:14px;">
                      ${autoBox}${allEventsBox}</div>`;

    if (!want.length) {
        host.innerHTML = `
            <p style="color:#94a3b8; margin-top:18px;">
              No tracked matches for ${team}${evKey ? ` at ${evKey}` : ''}.<br>
              <span style="font-size:12px;">
                ${elsewhere
                  ? `${elsewhere} tracked match${elsewhere === 1 ? '' : 'es'} at other events.`
                  : (man.matches || []).length
                    ? `${man.matches.length} match(es) published, none with this team.`
                    : 'Nothing published to public/tracks/ yet.'}
              </span>
            </p>${toggle}`;
        routesRenderedFor = memo;
        return;
    }

    const mine = (await Promise.all(want.map(r => loadMatchTracks(r.key))))
        .filter(Boolean);
    if (!mine.length) {
        host.innerHTML = `<p style="color:#94a3b8; margin-top:18px;">
            Tracks listed for ${team} but none could be loaded.</p>`;
        return;
    }

    host.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
                    gap:14px; margin-top:16px;">
          ${mine.map((d, i) => `
            <div>
              <div style="display:flex; gap:8px; align-items:baseline; margin-bottom:4px;">
                <b style="font-size:13px;">${d.match?.key || d.key}</b>
                <span style="font-size:11px; color:#94a3b8;">
                  ${(d.robots.find(r => String(r.team) === team)?.alliance) || ''}
                  · custody ${Math.round(100 * (d.robots.find(r => String(r.team) === team)?.custody || 0))}%
                  ${routesAutoOnly ? `· <span style="color:#60a5fa;">first ${autoEndOf(d).toFixed(0)}s</span>` : ''}
                </span>
              </div>
              <div class="trBox" data-i="${i}" title="Open full screen"
                   style="position:relative; width:100%; border-radius:6px;
                          overflow:hidden; cursor:zoom-in;">
                <img class="trImg" data-i="${i}"
                     src="${import.meta.env.BASE_URL}${d.field.imageRef}" alt="field"
                     style="display:block; width:100%; height:auto;">
                <canvas class="trCv" data-i="${i}"
                        style="position:absolute; inset:0; width:100%; height:100%;"></canvas>
              </div>
            </div>`).join('')}
        </div>${toggle}`;

    const only = new Set([team]);
    host.querySelectorAll('.trCv').forEach(cv => {
        const i = Number(cv.dataset.i);
        const img = host.querySelector(`.trImg[data-i="${i}"]`);
        applyFieldOrientation(img.parentElement, mine[i]);
        // The whole tile opens full screen. A small multiple is for spotting which
        // match is worth a closer look; this is the closer look.
        const box = host.querySelector(`.trBox[data-i="${i}"]`);
        if (box) box.onclick = () => window.openRoutesFull(mine[i], {
            teams: only, tNow: null, dots: false,
            tMax: routesAutoOnly ? autoEndOf(mine[i]) : null,
        });
        const paint = () => {
            if (_trackSizeCanvas(img, cv)) {
                // autoEndOf reads sampling.phases.autoEndT from the export and falls
                // back to the rulebook, so a doc written before that field existed
                // still clips at the right place rather than silently showing the
                // whole match under an "Auto only" heading.
                renderFieldRoutes(cv, mine[i], {
                    teams: only, tNow: null, dots: false,
                    tMax: routesAutoOnly ? autoEndOf(mine[i]) : null,
                });
            }
        };
        if (img.complete && img.naturalWidth) paint();
        else img.addEventListener('load', paint, { once: true });
    });
    routesRenderedFor = memo;   // claim the memo only once the render actually stands
}

window.switchDetailDataSubTab = async function (tab) {
    lastDetailDataSubTab = tab;
    const panes = { epa: 'tab-epa-opr', scouting: 'tab-scouting', pit: 'tab-pit-data' };
    Object.entries(panes).forEach(([k, id]) => {
        document.getElementById(id).style.display = k === tab ? '' : 'none';
    });
    document.querySelectorAll('#dataSubTabs .detail-tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', ['epa', 'scouting', 'pit'][i] === tab);
    });
    if (tab === 'epa' && activeTeamData) {
        renderChart(activeTeamData);
        await renderTBADetail(activeTeamData.teamNumber, activeTBAData);
    } else if (tab === 'scouting' && activeTeamData) {
        await renderScoutingTab(activeTeamData.teamNumber);
    } else if (tab === 'pit' && activeTeamData) {
        renderPitTab(activeTeamData.teamNumber);
    }
};

async function renderWLMatchesTab(teamNumber) {
    if (wlMatchesRenderedFor === teamNumber) return;
    const tableContainer = document.getElementById('matches-tab-table');
    if (!tableContainer) return;

    if (!wlDetailCache) {
        tableContainer.innerHTML = `<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">Computing predictions…</p>`;
        await renderWatchList();   // populates wlDetailCache as a side effect
        if (!wlDetailCache) {
            tableContainer.innerHTML = `<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No schedule loaded — sync TBA matches first.</p>`;
            if (rpTimelineChart) { rpTimelineChart.destroy(); rpTimelineChart = null; }
            return;
        }
    }

    const { allMatches, matchPredictions, baseRP, playedMatches, effectiveThresholds,
            tbaMap, allTeamNums, relResiduals, diffResiduals, fuelOPRCache, gameConfig } = wlDetailCache;
    const tnStr = String(teamNumber);
    const playedKeys = new Set(playedMatches.map(m => m.key));
    const thresholds = effectiveThresholds.filter(r => r.threshold != null);

    const teamMatches = allMatches
        .filter(m => !m.compLevel || m.compLevel === 'qm')
        .filter(m => m.red?.includes(tnStr) || m.blue?.includes(tnStr))
        .sort((a, b) => a.matchNumber - b.matchNumber);

    if (!teamMatches.length) {
        tableContainer.innerHTML = `<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No matches found for team ${teamNumber}.</p>`;
        return;
    }

    // Per-match performance chart (above RP timeline)
    await renderMatchesTab(teamNumber, 'matches-tab-perf-chart');

    // Build predictions for ALL matches (played + unplayed) so we can chart expected RP
    const allPredictions = { ...matchPredictions };
    for (const m of teamMatches) {
        if (!allPredictions[m.key]) {
            allPredictions[m.key] = wlSimulateMatch(
                m, tbaMap, wlDetailCache.teamsMap, allTeamNums,
                relResiduals, diffResiduals, gameConfig, effectiveThresholds, playedMatches, fuelOPRCache
            );
        }
    }

    // ── RP Timeline Chart ─────────────────────────────────────────────────────
    const labels = teamMatches.map(m => `Q${m.matchNumber}`);
    let cumDiff = 0;
    const diffData = [];
    for (const m of teamMatches) {
        const isRed  = m.red?.includes(tnStr);
        const pred   = allPredictions[m.key];
        const winP   = pred ? (isRed ? pred.redProb : pred.blueProb) : 0.5;
        const tieP   = pred?.tieProb ?? 0;
        const rpProbs = pred ? (isRed ? pred.rpProbs.red : pred.rpProbs.blue) : {};
        const expRP  = winP * 3 + tieP * 1 + thresholds.reduce((s, r) => s + (rpProbs[r.rpField] ?? 0), 0);

        if (playedKeys.has(m.key)) {
            const bd  = isRed ? m.redBreakdown : m.blueBreakdown;
            const myS = isRed ? m.redScore : m.blueScore;
            const opS = isRed ? m.blueScore : m.redScore;
            const winRP  = myS > opS ? 3 : myS === opS ? 1 : 0;
            const bonusRP = thresholds.filter(r => bd?.[r.rpField]).length;
            cumDiff += (winRP + bonusRP) - expRP;
            diffData.push(+cumDiff.toFixed(2));
        } else {
            diffData.push(null);
        }
    }

    if (rpTimelineChart) { rpTimelineChart.destroy(); rpTimelineChart = null; }
    const canvas = document.getElementById('rpTimelineChart');
    if (canvas) {
        rpTimelineChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'RP vs. Expected',
                    data: diffData,
                    borderColor: diffData.at(diffData.filter(v => v !== null).length - 1) >= 0 ? '#22c55e' : '#ef4444',
                    backgroundColor: 'rgba(100,116,139,0.08)',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.2,
                    spanGaps: false,
                    fill: { target: { value: 0 }, above: 'rgba(34,197,94,0.1)', below: 'rgba(239,68,68,0.1)' },
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e293b' } },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: {
                            color: ctx => ctx.tick.value === 0 ? '#475569' : '#1e293b',
                        },
                        title: { display: true, text: 'Cumulative RP vs. Expected', color: '#64748b', font: { size: 10 } },
                    },
                },
            },
        });
    }

    // ── Match Table ───────────────────────────────────────────────────────────
    const thHeaders = thresholds.map(r =>
        `<th style="padding:4px 8px;text-align:center;">${r.label.replace(' RP', '')}</th>`).join('');

    const preMatchPreds = wlPreEventCache?.matchPredictions ?? {};

    let totalWins = 0, totalActRP = 0, totalExpRP = 0;
    const rows = teamMatches.map(m => {
        const isRed  = m.red?.includes(tnStr);
        const allies = (isRed ? m.red : m.blue).filter(t => t !== tnStr).join(' · ');
        const label  = isRed ? 'RED' : 'BLU';
        const lColor = isRed ? '#ef4444' : '#60a5fa';
        const pred   = allPredictions[m.key];
        const prePred = preMatchPreds[m.key];
        const preWinP = prePred ? (isRed ? prePred.redProb : prePred.blueProb) : null;
        const preWinPct = preWinP != null ? Math.round(preWinP * 100) : null;
        const preLine = preWinPct != null
            ? `<div style="font-size:0.75em;color:#475569;margin-top:1px;" title="Pre-event baseline — frozen at first load using EPA only. May differ from live if data was re-synced or OPR became available.">pre: ${preWinPct}%</div>`
            : '';
        const winP   = pred ? (isRed ? pred.redProb : pred.blueProb) : null;
        const rpProbs = pred ? (isRed ? pred.rpProbs.red : pred.rpProbs.blue) : {};
        const expWinRP   = pred ? (winP * 3 + pred.tieProb * 1) : null;
        const expBonusRP = pred ? thresholds.reduce((s, r) => s + (rpProbs[r.rpField] ?? 0), 0) : null;
        const expTotal   = expWinRP != null ? expWinRP + expBonusRP : null;
        const winPct = winP != null ? Math.round(winP * 100) : null;
        if (expTotal != null) totalExpRP += expTotal;

        const pColor = (pct) => pct >= 60 ? '#22c55e' : pct >= 35 ? '#f59e0b' : '#64748b';

        if (playedKeys.has(m.key)) {
            const bd   = isRed ? m.redBreakdown : m.blueBreakdown;
            const myS  = isRed ? m.redScore  : m.blueScore;
            const opS  = isRed ? m.blueScore : m.redScore;
            const won  = myS > opS, tied = myS === opS;
            const winRP   = won ? 3 : tied ? 1 : 0;
            const bonusRP = thresholds.filter(r => bd?.[r.rpField]).length;
            const actTotal = winRP + bonusRP;
            if (won) totalWins++;
            totalActRP += actTotal;
            const resultMark = won ? '✓' : tied ? '–' : '✗';
            const resultColor = won ? '#22c55e' : tied ? '#f59e0b' : '#64748b';
            const winCell = `<td style="padding:4px 8px;text-align:center;">
                ${winPct != null ? `<span style="color:${pColor(winPct)}">${winPct}%</span> ` : ''}
                <span style="color:${resultColor};font-weight:700;">${resultMark}</span>
                ${preLine}
            </td>`;
            const thCells = thresholds.map(r => {
                const p   = rpProbs[r.rpField];
                const hit = bd?.[r.rpField];
                const pStr = p != null ? `<span style="color:${pColor(Math.round(p*100))};font-size:0.8em;">${Math.round(p*100)}%</span> ` : '';
                return `<td style="padding:4px 8px;text-align:center;">${pStr}<span style="color:${hit ? '#22c55e' : '#475569'};font-weight:700;">${hit ? '✓' : '✗'}</span></td>`;
            }).join('');
            const expStr = expTotal != null ? `${expTotal.toFixed(1)} ` : '';
            return `<tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:4px 8px;text-align:center;color:#94a3b8;">Q${m.matchNumber}</td>
                <td style="padding:4px 8px;text-align:center;font-size:0.8em;font-weight:700;color:${lColor};">${label}</td>
                <td style="padding:4px 8px;color:#94a3b8;font-size:0.82em;">${allies}</td>
                ${winCell}${thCells}
                <td style="padding:4px 8px;text-align:center;">${expStr}<span style="color:#64748b;">(${actTotal})</span></td>
            </tr>`;
        } else {
            const thCells = thresholds.map(r => {
                const p = rpProbs[r.rpField];
                const pct = p != null ? Math.round(p * 100) : null;
                return `<td style="padding:4px 8px;text-align:center;${pct != null ? `color:${pColor(pct)};` : 'color:#475569;'}">${pct != null ? pct + '%' : '—'}</td>`;
            }).join('');
            return `<tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:4px 8px;text-align:center;color:#94a3b8;">Q${m.matchNumber}</td>
                <td style="padding:4px 8px;text-align:center;font-size:0.8em;font-weight:700;color:${lColor};">${label}</td>
                <td style="padding:4px 8px;color:#94a3b8;font-size:0.82em;">${allies}</td>
                <td style="padding:4px 8px;text-align:center;${winPct != null ? `color:${pColor(winPct)};` : 'color:#475569;'}">${winPct != null ? winPct + '%' : '—'}${preLine}</td>
                ${thCells}
                <td style="padding:4px 8px;text-align:center;color:#94a3b8;">${expTotal != null ? expTotal.toFixed(1) : '—'}</td>
            </tr>`;
        }
    }).join('');

    const playedCount = teamMatches.filter(m => playedKeys.has(m.key)).length;
    const footerColSpan = 3 + thresholds.length;
    const tfoot = playedCount > 0 ? `
        <tfoot><tr style="border-top:2px solid #334155;color:#f8fafc;font-weight:600;">
            <td colspan="${footerColSpan}" style="padding:6px 8px;text-align:right;color:#64748b;font-size:0.82em;">TOTAL</td>
            <td style="padding:6px 8px;text-align:center;">${totalWins}W</td>
            <td style="padding:6px 8px;text-align:center;">${totalExpRP.toFixed(1)} <span style="color:#64748b;">(${totalActRP})</span></td>
        </tr></tfoot>` : '';

    tableContainer.innerHTML = `
        <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
                <thead><tr style="color:#64748b;font-size:0.78em;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid #334155;">
                    <th style="padding:4px 8px;text-align:center;">Match</th>
                    <th style="padding:4px 8px;text-align:center;">Side</th>
                    <th style="padding:4px 8px;">Allies</th>
                    <th style="padding:4px 8px;text-align:center;">Win%</th>
                    ${thHeaders}
                    <th style="padding:4px 8px;text-align:center;">Exp. RP</th>
                </tr></thead>
                <tbody>${rows}</tbody>
                ${tfoot}
            </table>
        </div>`;
    wlMatchesRenderedFor = teamNumber;
}

window.setMatchChartYMode = function(mode) {
    localStorage.setItem('matchChartYMode', mode);
    if (activeTeamNumber) renderMatchesTab(activeTeamNumber, 'matches-tab-perf-chart');
};

async function renderMatchesTab(teamNumber, containerId = 'matches-tab-perf-chart') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">Set an event key on the Home tab.</p>';
        return;
    }

    const teamStr = String(teamNumber);
    const [allMatches, allTBATeams, allStatTeams] = await Promise.all([
        db.matches.toArray(), db.tbaTeams.toArray(), db.teams.toArray(),
    ]);

    const oprMap     = Object.fromEntries(allTBATeams.map(t => [String(t.teamNumber), t.opr ?? 0]));
    const teamOPR    = oprMap[teamStr] ?? 0;
    const evEPAMap   = Object.fromEntries(allStatTeams.map(t => [
        String(t.teamNumber),
        t.epa?.end ?? t.epa?.mean ?? (typeof t.currentEPA === 'number' ? t.currentEPA : null),
    ]));
    const teamEvEPA  = evEPAMap[teamStr] ?? null;

    const fusedCache  = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
    const fusedTeam   = fusedCache?.teams?.[teamStr];
    const fusedByMatch = fusedTeam?.fusedByMatch ?? {};
    const gameConfig  = getGameConfig(eventKey);
    const avgFusedEPA = (fusedTeam?.available && gameConfig?.computeFusedEPABreakdown)
        ? gameConfig.computeFusedEPABreakdown(fusedTeam.stats).total : null;

    // Statbotics series: eventEPA residual (same structure as OPR, Statbotics coefficients)
    // deviation = (allianceScore − sum(partner eventEPAs)) − team eventEPA

    const tbaTeamEntry   = allTBATeams.find(t => String(t.teamNumber) === teamStr) ?? null;
    const teamIgnoredKeys = new Set(getTeamIgnoredKeys(tbaTeamEntry));

    const globalIgnored  = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));

    // All scheduled qual matches for this team — used for x-axis
    const allQualTeamMatches = allMatches
        .filter(m => !m.compLevel || m.compLevel === 'qm')
        .filter(m => m.red?.includes(teamStr) || m.blue?.includes(teamStr))
        .sort((a, b) => a.matchNumber - b.matchNumber);

    if (!allQualTeamMatches.length) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No match data — sync TBA matches first.</p>';
        return;
    }

    const playedKeys = new Set(
        allQualTeamMatches
            .filter(m => (m.redScore ?? -1) >= 0 && !globalIgnored.has(m.key))
            .map(m => m.key)
    );

    const labels     = [];
    const oprData    = [];
    const scoutData  = [];
    const statEvData = [];
    const ignoredColIndices       = new Set(); // per-team ignored (red shade)
    const globalIgnoredColIndices = new Set(); // globally ignored (amber shade)

    for (const m of allQualTeamMatches) {
        const isPlayed       = playedKeys.has(m.key);
        const hasScore       = (m.redScore ?? -1) >= 0;
        const isGlobIgnored  = globalIgnored.has(m.key);

        if (isGlobIgnored && hasScore) globalIgnoredColIndices.add(labels.length);

        if (isPlayed) {
            const isRed      = m.red?.includes(teamStr);
            const alliance   = isRed ? m.red : m.blue;
            const allyScore  = isRed ? m.redScore : m.blueScore;
            const partnerSum = alliance.filter(t => t !== teamStr).reduce((s, t) => s + (oprMap[t] ?? 0), 0);
            oprData.push(parseFloat(((allyScore - partnerSum) - teamOPR).toFixed(2)));

            const matchFused = fusedByMatch[m.matchNumber];
            const scoutDev   = (matchFused && avgFusedEPA != null && gameConfig?.computeFusedEPABreakdown)
                ? parseFloat((gameConfig.computeFusedEPABreakdown(matchFused).total - avgFusedEPA).toFixed(2))
                : null;
            scoutData.push(scoutDev);

            const evPartnerSum = alliance.filter(t => t !== teamStr).reduce((s, t) => s + (evEPAMap[t] ?? 0), 0);
            statEvData.push(teamEvEPA != null ? parseFloat(((allyScore - evPartnerSum) - teamEvEPA).toFixed(2)) : null);

            if (teamIgnoredKeys.has(m.key)) ignoredColIndices.add(labels.length);
        } else {
            oprData.push(null);
            scoutData.push(null);
            statEvData.push(null);
        }

        labels.push(`Q${m.matchNumber}`);
    }

    // Per-match average across all non-null series (for the white reference line)
    const avgData = labels.map((_, i) => {
        const vals = [oprData[i], scoutData[i], statEvData[i]].filter(v => v != null);
        return vals.length > 0 ? parseFloat((vals.reduce((s, v) => s + v, 0) / vals.length).toFixed(2)) : null;
    });

    const hasScout  = scoutData.some(v => v != null);
    const hasStatEv = statEvData.some(v => v != null);

    // Hatched canvas patterns distinguish the Statbotics series (///) from solid OPR/scouting bars.
    const makeHatch = color => {
        const sz = 10, c = document.createElement('canvas');
        c.width = sz; c.height = sz;
        const cx = c.getContext('2d');
        cx.strokeStyle = color; cx.lineWidth = 2.5; cx.beginPath();
        cx.moveTo(0, sz); cx.lineTo(sz, 0);
        cx.stroke();
        return cx.createPattern(c, 'repeat');
    };

    const evPosH = makeHatch('#fbbf24'); const evNegH = makeHatch('#a78bfa');

    // Y-axis mode: 'global' (same limits across all teams) or 'team' (auto-fit per team)
    const yMode = localStorage.getItem('matchChartYMode') || 'global';
    let yMin, yMax;
    if (yMode === 'global') {
        // Compute max positive and negative residuals across all teams using already-loaded data
        let maxPos = 0, maxNeg = 0;
        for (const m of allMatches) {
            if ((!m.compLevel || m.compLevel === 'qm') && (m.redScore ?? -1) >= 0 && !globalIgnored.has(m.key)) {
                for (const side of [m.red, m.blue]) {
                    if (!side) continue;
                    const score = side === m.red ? m.redScore : m.blueScore;
                    for (const tn of side) {
                        const opr = oprMap[tn] ?? 0;
                        const partnerOPR = side.filter(t => t !== tn).reduce((s, t) => s + (oprMap[t] ?? 0), 0);
                        const r1 = (score - partnerOPR) - opr;
                        maxPos = Math.max(maxPos, r1); maxNeg = Math.max(maxNeg, -r1);
                        const evEPA = evEPAMap[tn];
                        if (evEPA != null) {
                            const partnerEPA = side.filter(t => t !== tn).reduce((s, t) => s + (evEPAMap[t] ?? 0), 0);
                            const r2 = (score - partnerEPA) - evEPA;
                            maxPos = Math.max(maxPos, r2); maxNeg = Math.max(maxNeg, -r2);
                        }
                    }
                }
            }
        }
        const maxAbs = Math.max(maxPos, maxNeg);
        if (maxAbs > 0) {
            // Shared step ≈ 20% of largest deviation, snapped to a nice number (1/2/2.5/5/10 per decade)
            const rawStep = maxAbs * 0.2;
            const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
            const norm = rawStep / mag;
            const niceFactor = norm < 1.5 ? 1 : norm < 2.25 ? 2 : norm < 3.75 ? 2.5 : norm < 7.5 ? 5 : 10;
            const step = niceFactor * mag;
            // Snap each bound independently — step is shared so tick spacing stays consistent
            yMax =  Math.ceil(maxPos / step) * step;
            yMin = -Math.ceil(maxNeg / step) * step;
        } else {
            yMax = 50; yMin = -50;
        }
    }

    const btnStyle = (active) => active
        ? 'background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;border-radius:4px;padding:2px 10px;font-size:0.78em;cursor:pointer;'
        : 'background:transparent;color:#475569;border:1px solid #334155;border-radius:4px;padding:2px 10px;font-size:0.78em;cursor:pointer;';

    const statEvLine = teamEvEPA != null ? ` · Stat event EPA = ${teamEvEPA.toFixed(1)}` : '';
    container.innerHTML = `
        <div style="margin-top:16px;background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;flex-wrap:wrap;gap:6px;">
                <h3 style="margin:0;font-size:0.95em;font-weight:700;color:#f1f5f9;">Per-Match Performance vs. Average</h3>
                <span style="color:#475569;font-size:0.78em;">
                    OPR avg = ${teamOPR.toFixed(1)}${avgFusedEPA != null ? ` · scout avg = ${avgFusedEPA.toFixed(1)}` : ''}${statEvLine}
                </span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
                <span style="color:#64748b;font-size:0.78em;">Y-axis:</span>
                <button onclick="window.setMatchChartYMode('global')" style="${btnStyle(yMode === 'global')}">All teams</button>
                <button onclick="window.setMatchChartYMode('team')" style="${btnStyle(yMode === 'team')}">This team</button>
            </div>
            <p style="margin:0 0 12px;font-size:0.78em;color:#475569;">Bars above zero = outperformed average; below = underperformed. All series share the same zero baseline.</p>
            <div style="overflow-x:auto;">
                <div style="min-width:${Math.max(360, allQualTeamMatches.length * 54)}px;">
                    <div style="display:flex;margin-bottom:2px;padding:0 2px;" id="matchesChartIgnoreBtns"></div>
                    <div style="position:relative;height:320px;"><canvas id="matchesChart"></canvas></div>
                    <div style="display:flex;margin-top:6px;padding:0 2px;" id="matchesChartLinks"></div>
                </div>
            </div>
        </div>
    `;

    // Populate per-match ignore-toggle row (above chart) and detail-link row (below chart)
    const ignoreBtnsRow = document.getElementById('matchesChartIgnoreBtns');
    if (ignoreBtnsRow) {
        ignoreBtnsRow.innerHTML = allQualTeamMatches.map(m => {
            const hasScore      = (m.redScore ?? -1) >= 0;
            const isGlobIgnored = globalIgnored.has(m.key);
            const isTeamIgnored = teamIgnoredKeys.has(m.key);
            if (!hasScore) return `<div style="flex:1;"></div>`;
            if (isGlobIgnored) {
                return `<div style="flex:1;display:flex;justify-content:center;align-items:center;">
                    <span title="Globally ignored" style="font-size:0.6em;font-weight:700;color:#d97706;letter-spacing:0.03em;line-height:1;padding:1px 3px;border:1px solid #92400e;border-radius:3px;">GLOBAL</span>
                </div>`;
            }
            const active = isTeamIgnored;
            return `<div style="flex:1;display:flex;justify-content:center;align-items:center;">
                <button onclick="setIgnoredMatch(${teamNumber},'${m.key}')"
                    title="${active ? 'Restore Q' + m.matchNumber + ' for this team' : 'Ignore Q' + m.matchNumber + ' for this team'}"
                    style="background:none;border:1px solid ${active ? '#92400e' : 'transparent'};border-radius:3px;color:${active ? '#d97706' : '#334155'};font-size:0.72em;cursor:pointer;padding:1px 5px;line-height:1.4;transition:all 0.15s;"
                    onmouseover="this.style.borderColor='${active ? '#d97706' : '#475569'}';this.style.color='${active ? '#fbbf24' : '#94a3b8'}'"
                    onmouseout="this.style.borderColor='${active ? '#92400e' : 'transparent'}';this.style.color='${active ? '#d97706' : '#334155'}'">
                    ${active ? '↩' : '✕'}
                </button>
            </div>`;
        }).join('');
    }
    const linksRow = document.getElementById('matchesChartLinks');
    if (linksRow) {
        linksRow.innerHTML = allQualTeamMatches.map(m => {
            const played = playedKeys.has(m.key);
            return `<div style="flex:1;display:flex;justify-content:center;">${played
                ? `<button onclick="viewMatchDetail('${m.key}')" title="Open Q${m.matchNumber} detail"
                    style="background:none;border:none;color:#334155;font-size:0.7em;cursor:pointer;padding:2px 4px;line-height:1;border-radius:3px;transition:color 0.15s;"
                    onmouseover="this.style.color='#94a3b8'" onmouseout="this.style.color='#334155'">↗</button>`
                : ''}</div>`;
        }).join('');
    }

    if (matchesChartInstance) { matchesChartInstance.destroy(); matchesChartInstance = null; }

    // Align the button/link rows to chart columns by reading chartArea after render.
    const alignLinksPlugin = {
        id: 'matchesAlignLinks',
        afterRender(chart) {
            const { left, right } = chart.chartArea;
            const colWidth = (right - left) / labels.length;
            for (const rowId of ['matchesChartIgnoreBtns', 'matchesChartLinks']) {
                const row = document.getElementById(rowId);
                if (!row || !labels.length) continue;
                row.style.paddingLeft  = `${left}px`;
                row.style.paddingRight = `${chart.canvas.offsetWidth - right}px`;
                row.querySelectorAll('div').forEach(div => {
                    div.style.width    = `${colWidth}px`;
                    div.style.minWidth = `${colWidth}px`;
                    div.style.flex     = 'none';
                });
            }
        },
    };

    // Shade ignored match columns: amber for global ignore, red for per-team ignore.
    const ignoredColBgPlugin = {
        id: 'matchesIgnoredBg',
        beforeDraw(chart) {
            const { ctx: c, chartArea, scales } = chart;
            if (!chartArea) return;
            const count = labels.length;
            const step  = count > 0 ? scales.x.width / count : 0;
            c.save();
            for (const i of globalIgnoredColIndices) {
                const cx = scales.x.getPixelForValue(i);
                c.fillStyle = 'rgba(217,119,6,0.18)';
                c.fillRect(cx - step / 2, chartArea.top, step, chartArea.bottom - chartArea.top);
            }
            for (const i of ignoredColIndices) {
                const cx = scales.x.getPixelForValue(i);
                c.fillStyle = 'rgba(239,68,68,0.12)';
                c.fillRect(cx - step / 2, chartArea.top, step, chartArea.bottom - chartArea.top);
            }
            c.restore();
        },
    };

    const chartCtx = document.getElementById('matchesChart').getContext('2d');
    matchesChartInstance = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'OPR-implied vs avg',
                    data: oprData,
                    backgroundColor: oprData.map(v => v >= 0 ? 'rgba(56,189,248,0.65)' : 'rgba(248,113,113,0.65)'),
                    borderColor:     oprData.map(v => v >= 0 ? '#38bdf8' : '#f87171'),
                    borderWidth: 1,
                    borderRadius: 3,
                    order: 1,
                },
                ...(hasScout ? [{
                    label: 'Scout EPA vs avg',
                    data: scoutData,
                    backgroundColor: scoutData.map(v => v == null ? 'transparent' : v >= 0 ? 'rgba(74,222,128,0.65)' : 'rgba(251,146,60,0.65)'),
                    borderColor:     scoutData.map(v => v == null ? 'transparent' : v >= 0 ? '#4ade80' : '#fb923c'),
                    borderWidth: 1,
                    borderRadius: 3,
                    order: 1,
                }] : []),
                ...(hasStatEv ? [{
                    label: 'Stat event EPA residual (///)',
                    data: statEvData,
                    backgroundColor: statEvData.map(v => v == null ? 'transparent' : v >= 0 ? evPosH : evNegH),
                    borderColor:     statEvData.map(v => v == null ? 'transparent' : v >= 0 ? '#fbbf24' : '#a78bfa'),
                    borderWidth: 1,
                    borderRadius: 3,
                    order: 1,
                }] : []),
                {
                    label: 'Series avg',
                    type: 'line',
                    data: avgData,
                    showLine: false,
                    pointStyle: 'line',
                    pointRadius: 10,
                    pointBorderWidth: 2.5,
                    pointBorderColor: '#ffffff',
                    backgroundColor: 'transparent',
                    borderColor: 'transparent',
                    order: 0,
                },
            ],
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const v = ctx.raw;
                            if (v == null) return `${ctx.dataset.label}: N/A`;
                            return `${ctx.dataset.label}: ${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid:  { color: '#1e293b' },
                },
                y: {
                    ...(yMin != null ? { min: yMin, max: yMax } : {}),
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid:  {
                        color: ctx => ctx.tick.value === 0 ? '#94a3b8' : '#1e293b',
                        lineWidth: ctx => ctx.tick.value === 0 ? 2 : 1,
                    },
                    title: { display: true, text: 'Δ from average', color: '#475569', font: { size: 10 } },
                },
            },
        },
        plugins: [ignoredColBgPlugin, alignLinksPlugin],
    });
}

// Merge same-matchNumber rows for one team into averaged/ORed single rows.
// Returns { rows: deduped[], duplicated: Set<matchNumber> }
function deduplicateTeamRows(rows) {
    const byMatch = {};
    for (const r of rows) {
        if (!byMatch[r.matchNumber]) byMatch[r.matchNumber] = [];
        byMatch[r.matchNumber].push(r);
    }
    const duplicated = new Set();
    const merged = Object.entries(byMatch).map(([, group]) => {
        if (group.length === 1) return group[0];
        duplicated.add(group[0].matchNumber);
        const result = {};
        for (const key of Object.keys(group[0])) {
            const vals = group.map(r => r[key]);
            if (key === 'comments') {
                result[key] = vals.filter(v => v).join(' | ');
            } else if (typeof vals[0] === 'boolean') {
                result[key] = vals.some(Boolean);
            } else if (typeof vals[0] === 'number') {
                result[key] = vals.reduce((s, v) => s + v, 0) / vals.length;
            } else {
                result[key] = vals.find(v => v) ?? vals[0];
            }
        }
        return result;
    });
    return { rows: merged.sort((a, b) => a.matchNumber - b.matchNumber), duplicated };
}

async function renderScoutingTab(teamNumber) {
    const container = document.getElementById('tab-scouting');
    if (!container) return;

    const eventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase();
    if (!eventKey) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">Set an event key on the Home tab to see scouting data.</p>';
        return;
    }

    const rawStr = localStorage.getItem(`scoutingData_${eventKey}`);
    if (!rawStr) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No scouting data. Sync scouting data on the Home tab.</p>';
        return;
    }

    const rawRows = JSON.parse(rawStr);
    const overrides = getScoutingColumnOverrides(eventKey);
    const processed = processScoutingData(eventKey, rawRows, overrides);
    if (!processed) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No game config found for this event.</p>';
        return;
    }

    const { config, byTeam } = processed;
    const rawTeamRows = byTeam[String(teamNumber)] || [];
    if (rawTeamRows.length === 0) {
        container.innerHTML = '<p style="color:#64748b;font-style:italic;margin-top:24px;text-align:center;">No scouting observations for this team at this event.</p>';
        return;
    }

    let { rows: teamRows, duplicated } = deduplicateTeamRows(rawTeamRows);

    // If the user opted to exclude ignored matches from scouting, filter them out.
    const tbaTeamForFilter = activeTBAData?.teamNumber === teamNumber ? activeTBAData
        : await db.tbaTeams.get(parseInt(teamNumber));
    const scoutIgnoreKeys = getTeamIgnoredKeys(tbaTeamForFilter);
    const scoutIgnoreActive = tbaTeamForFilter?.scoutingIgnoreActive && scoutIgnoreKeys.length > 0;
    let scoutExcludedCount = 0;
    if (scoutIgnoreActive) {
        const tbaMatchesAll = await db.matches.toArray();
        const ignoredMatchNums = new Set(
            tbaMatchesAll.filter(m => scoutIgnoreKeys.includes(m.key)).map(m => m.matchNumber)
        );
        const before = teamRows.length;
        teamRows = teamRows.filter(r => !ignoredMatchNums.has(r.matchNumber));
        scoutExcludedCount = before - teamRows.length;
    }

    const rawStats = config.aggregateTeam(teamRows);

    // Inject raw fallbacks for robot stats (computed from scouting aggregates)
    if (config.robotFuseStats) {
        for (const stat of config.robotFuseStats) {
            const vals = teamRows.filter(r => !r.noShow).map(r => stat.scout(r));
            rawStats[stat.key] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
        }
    }

    const tbaMatches = await db.matches.where('eventKey').equals(eventKey).toArray();

    // Overwrite unconditional shift percentages with hub-active-conditional versions.
    if (config.enrichAggregateWithTBA && tbaMatches.some(m => m.redBreakdown)) {
        const tbaByMatch = {};
        for (const m of tbaMatches) tbaByMatch[m.matchNumber] = m;
        config.enrichAggregateWithTBA(String(teamNumber), teamRows, rawStats, tbaByMatch);
    }

    const allByMatch = indexObservationsByMatch(processed.observations);
    const fused = fuseScoutingWithTBA(teamNumber, teamRows, allByMatch, tbaMatches, config);

    const fusedByMatch = fused.fusedByMatch ?? {};
    const fusedStats = fused.available ? { ...fused.stats } : {};

    // Derive fused totals by summing fused components
    if (fused.available) {
        config?.deriveFusedTotals?.(fusedStats);
    }

    const getValue = (key) => fusedStats[key] != null ? fusedStats[key] : rawStats[key];
    const isFused  = (key) => fusedStats[key] != null;

    // Compute scoutingEPA from weighted sum; mark fused if any component came from TBA
    if (config.scoringWeights) {
        const epa = Object.entries(config.scoringWeights)
            .reduce((sum, [key, w]) => sum + (getValue(key) ?? 0) * w, 0);
        rawStats.scoutingEPA = epa;
        const anyFused = Object.keys(config.scoringWeights).some(isFused);
        if (anyFused) fusedStats.scoutingEPA = epa;
    }

    let html = '';

    // Scouting exclusion banner
    if (scoutIgnoreActive) {
        html += `<div style="background:#1a1505;border:1px solid #854d0e;border-radius:6px;padding:9px 14px;margin-bottom:12px;font-size:0.8em;color:#fbbf24;">
            Excluding ${scoutExcludedCount} scouting observation${scoutExcludedCount !== 1 ? 's' : ''} from OPR-ignored match${scoutIgnoreKeys.length !== 1 ? 'es' : ''} · toggle in EPA/OPR tab</div>`;
    }

    // Fusion status banner
    if (fused.available) {
        const { total, withTBA } = fused.coverage;
        html += `
        <div style="background:#0f1f0f;border:1px solid #166534;border-radius:6px;padding:10px 14px;margin-bottom:16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span style="color:#4ade80;font-size:0.75em;font-weight:700;">TBA FUSED</span>
            <span style="color:#64748b;font-size:0.75em;">${withTBA}/${total} matches fused · ${fused.reportingMode} reporting</span>
            <span style="color:#475569;font-size:0.72em;margin-left:auto;">● fused &nbsp; ○ scout-only</span>
        </div>`;
    } else if (tbaMatches.some(m => m.redBreakdown)) {
        html += `<div style="background:#1a1a1a;border-radius:6px;padding:10px 14px;margin-bottom:16px;font-size:0.78em;color:#64748b;">TBA fusion unavailable — showing scouting data only.</div>`;
    } else {
        html += `<div style="background:#1e1a0a;border:1px solid #854d0e;border-radius:6px;padding:10px 14px;margin-bottom:16px;font-size:0.78em;color:#ca8a04;">Run "Sync TBA Matches" to enable TBA-fused estimates. Showing scouting data only.</div>`;
    }

    // Build event-wide breakdown pool so renderScoutingDetail can rank tiers relatively.
    let allScoutBreakdowns = [];
    if (config.computeEPABreakdown || config.computeFusedEPABreakdown) {
        const eventFusedCache = (() => { try { return JSON.parse(localStorage.getItem(`scoutingFusedStats_${eventKey}`)); } catch { return null; } })();
        for (const [tn, rawRows] of Object.entries(byTeam)) {
            const { rows: deduped } = deduplicateTeamRows(rawRows);
            const agg = config.aggregateTeam(deduped);
            const fusedResult = eventFusedCache?.teams?.[tn];
            const hasFused = !!(fusedResult?.available && config.computeFusedEPABreakdown);
            const bd = hasFused
                ? config.computeFusedEPABreakdown(fusedResult.stats)
                : (config.computeEPABreakdown?.(agg) ?? {});
            allScoutBreakdowns.push({
                teamNumber: tn,
                ...bd,
                teleFuelFused:  hasFused ? (fusedResult.stats.teleFuelFused  ?? null) : null,
                avgScoringEff: agg.avgScoringEff ?? null,
            });
        }
    }

    const rankTier = (vals, myVal) => {
        const sorted = vals.filter(v => v != null && !isNaN(v)).sort((a, b) => b - a);
        const rank = sorted.findIndex(v => v <= myVal + 0.001);
        const r = rank < 0 ? sorted.length : rank;
        return r < 8 ? 'S' : r < 20 ? 'A' : r < 32 ? 'B' : 'C';
    };

    // Stat groups — game-specific rich detail or generic displayFields grid
    if (config.renderScoutingDetail) {
        html += config.renderScoutingDetail({ rawStats, fusedStats, getValue, isFused, teamRows, fusedByMatch, fused, allScoutBreakdowns, rankTier });
    } else {
        const GROUP_COLORS = {
            'Overview':              { accent: '#64748b', bg: 'rgba(100,116,139,0.07)' },
            'Auto Coral':            { accent: '#f59e0b', bg: 'rgba(245,158,11,0.08)'  },
            'Auto Algae':            { accent: '#f59e0b', bg: 'rgba(245,158,11,0.08)'  },
            'Teleop Coral':          { accent: '#3b82f6', bg: 'rgba(59,130,246,0.08)'  },
            'Teleop Algae':          { accent: '#3b82f6', bg: 'rgba(59,130,246,0.08)'  },
            'Endgame & Reliability': { accent: '#10b981', bg: 'rgba(16,185,129,0.08)'  },
            'Qualitative':           { accent: '#a78bfa', bg: 'rgba(167,139,250,0.08)' },
        };
        const DEFAULT_COLOR = { accent: '#64748b', bg: 'rgba(100,116,139,0.07)' };

        let currentGroup = null;
        let currentColor = DEFAULT_COLOR;
        let groupCells = [];

        const flushGroup = () => {
            if (!currentGroup || groupCells.length === 0) return;
            const { accent } = currentColor;
            html += `
            <div style="margin-bottom:20px;border-left:3px solid ${accent};padding-left:12px;">
                <div style="color:${accent};font-size:0.7em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">${currentGroup}</div>
                <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:8px;">
                    ${groupCells.join('')}
                </div>
            </div>`;
            groupCells = [];
        };

        for (const field of config.displayFields) {
            if (field.group) {
                flushGroup();
                currentGroup = field.group;
                currentColor = GROUP_COLORS[field.group] || DEFAULT_COLOR;
                continue;
            }
            const val = getValue(field.key);
            if (val == null) continue;

            let display;
            if (field.suffix === '%') display = `${Math.round(val)}%`;
            else if (field.decimals != null) display = val.toFixed(field.decimals);
            else display = String(Math.round(val));

            const dot = isFused(field.key)
                ? `<span title="TBA-fused" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#4ade80;margin-left:5px;vertical-align:middle;flex-shrink:0;"></span>`
                : '';

            groupCells.push(`
                <div style="background:${currentColor.bg};border:1px solid rgba(255,255,255,0.04);border-radius:6px;padding:10px;text-align:center;">
                    <div style="color:#94a3b8;font-size:0.65em;margin-bottom:4px;">${field.label}</div>
                    <div style="font-size:1.05em;font-weight:700;display:flex;align-items:center;justify-content:center;">${display}${dot}</div>
                </div>`);
        }
        flushGroup();
    }

    // Per-match detail table
    const teamStr = String(teamNumber);
    const played = teamRows.filter(r => !r.noShow).sort((a, b) => a.matchNumber - b.matchNumber);
    const scoutedNums = new Set(played.map(r => r.matchNumber));

    // TBA matches where this team appears but has no scouting row
    const unscoutedTBA = tbaMatches
        .filter(m => (m.red?.includes(teamStr) || m.blue?.includes(teamStr)) && !scoutedNums.has(m.matchNumber))
        .map(m => ({ matchNumber: m.matchNumber, row: null }));

    const allEntries = [
        ...played.map(r => ({ matchNumber: r.matchNumber, row: r })),
        ...unscoutedTBA,
    ].sort((a, b) => a.matchNumber - b.matchNumber);

    if (allEntries.length > 0) {
        const hasFusedMatches = Object.keys(fusedByMatch).length > 0;

        const _cols = config?.matchBreakdownColumns ?? [];

        // Fused view shows: Auto Fuel | Tele Fuel | Endgame Fuel | Total Fuel.
        // Raw view shows the full _cols set + Endgame position column.
        const fusedFuelCols = [
            ..._cols.filter(c => c.label === 'Auto Fuel' || c.label === 'Tele Fuel'),
            { label: 'Endgame Fuel', fused: f => f.endgameFuelFused ?? null, raw: r => null },
        ];

        const buildMatchRow = (matchNumber, row, mode) => {
            const dupBadge = duplicated.has(matchNumber)
                ? `<span style="color:#ef4444;font-size:0.75em;font-weight:700;margin-left:4px;">2x</span>`
                : '';
            if (!row) {
                const span = mode === 'fused' ? fusedFuelCols.length + 1 : _cols.length + 1;
                return `<tr style="border-bottom:1px solid #1e293b;opacity:0.45;">
                    <td style="text-align:left;padding:4px 8px;color:#60a5fa;white-space:nowrap;">QM ${matchNumber}</td>
                    <td colspan="${span}" style="padding:4px 8px;color:#475569;font-style:italic;white-space:nowrap;">No scouting data</td>
                </tr>`;
            }
            if (mode === 'fused') {
                const f = fusedByMatch[matchNumber];
                const fmt = v => v != null && v > 0 ? v.toFixed(1).replace(/\.0$/, '') : '—';
                if (!f) {
                    return `<tr style="border-bottom:1px solid #1e293b;opacity:0.5;">
                        <td style="text-align:left;padding:4px 8px;color:#60a5fa;white-space:nowrap;">QM ${matchNumber}${dupBadge}</td>
                        ${fusedFuelCols.map(c => { const v = c.raw(row); return `<td style="text-align:right;padding:4px 6px;font-style:italic;white-space:nowrap;">${typeof v === 'number' && v > 0 ? v.toFixed(1).replace(/\.0$/, '') : (v || '—')}</td>`; }).join('')}
                        <td style="text-align:right;padding:4px 6px;color:#475569;font-style:italic;white-space:nowrap;">no TBA</td>
                    </tr>`;
                }
                const totalFuel = (f.autoFuelFused ?? 0) + (f.teleFuelFused ?? 0) + (f.endgameFuelFused ?? 0);
                return `<tr style="border-bottom:1px solid #1e293b;">
                    <td style="text-align:left;padding:4px 8px;color:#60a5fa;white-space:nowrap;">QM ${matchNumber}${dupBadge}</td>
                    ${fusedFuelCols.map(c => `<td style="text-align:right;padding:4px 6px;color:#4ade80;white-space:nowrap;">${fmt(c.fused(f))}</td>`).join('')}
                    <td style="text-align:right;padding:4px 6px;color:#4ade80;font-weight:700;white-space:nowrap;">${fmt(totalFuel)}</td>
                </tr>`;
            }
            // raw mode
            return `<tr style="border-bottom:1px solid #1e293b;">
                <td style="text-align:left;padding:4px 8px;color:#60a5fa;white-space:nowrap;">QM ${matchNumber}${dupBadge}</td>
                ${_cols.map(c => { const v = c.raw(row); return `<td style="text-align:right;padding:4px 6px;white-space:nowrap;">${typeof v === 'number' && v > 0 ? v.toFixed(1).replace(/\.0$/, '') : (v || '—')}</td>`; }).join('')}
                <td style="text-align:right;padding:4px 6px;color:#94a3b8;white-space:nowrap;">${row.endPosition || '—'}</td>
            </tr>`;
        };

        const tableHtml = (mode) => {
            const headers = mode === 'fused'
                ? [...fusedFuelCols.map(c => c.label), 'Total Fuel']
                : [..._cols.map(c => c.label), 'End'];
            return `
            <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
                <thead><tr style="color:#64748b;border-bottom:1px solid #334155;">
                    <th style="text-align:left;padding:4px 8px;white-space:nowrap;">Match</th>
                    ${headers.map(h => `<th style="text-align:right;padding:4px 6px;white-space:nowrap;">${h}</th>`).join('')}
                </tr></thead>
                <tbody>
                ${allEntries.map(({ matchNumber, row }) => buildMatchRow(matchNumber, row, mode)).join('')}
                </tbody>
            </table>`;
        };

        const toggleHtml = hasFusedMatches ? `
            <div style="display:flex;gap:0;border:1px solid #334155;border-radius:5px;overflow:hidden;">
                <button id="scoutMatchToggleFused"
                    onclick="document.getElementById('scoutMatchTableFused').style.display='';document.getElementById('scoutMatchTableRaw').style.display='none';document.getElementById('scoutMatchToggleFused').style.cssText+='background:#1e293b;color:#4ade80;';document.getElementById('scoutMatchToggleRaw').style.cssText+='background:transparent;color:#64748b;';"
                    style="background:#1e293b;color:#4ade80;border:none;padding:4px 12px;font-size:0.72em;cursor:pointer;font-weight:600;">Fused</button>
                <button id="scoutMatchToggleRaw"
                    onclick="document.getElementById('scoutMatchTableRaw').style.display='';document.getElementById('scoutMatchTableFused').style.display='none';document.getElementById('scoutMatchToggleRaw').style.cssText+='background:#1e293b;color:#f8fafc;';document.getElementById('scoutMatchToggleFused').style.cssText+='background:transparent;color:#64748b;';"
                    style="background:transparent;color:#64748b;border:none;padding:4px 12px;font-size:0.72em;cursor:pointer;font-weight:600;">Raw</button>
            </div>` : '';

        html += `
        <div style="margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <div style="color:#64748b;font-size:0.7em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">Per-Match Detail</div>
                ${toggleHtml}
            </div>
            <div style="overflow-x:auto;">
                <div id="scoutMatchTableRaw" ${hasFusedMatches ? 'style="display:none;"' : ''}>${tableHtml('raw')}</div>
                ${hasFusedMatches ? `<div id="scoutMatchTableFused">${tableHtml('fused')}</div>` : ''}
            </div>
        </div>`;
    }

    container.innerHTML = html;
}

async function renderTBADetail(teamNumber, tbaTeam) {
    const teamNumStr = teamNumber.toString();
    const oprSection = document.getElementById('tbaDetailStats');
    const tableSection = document.getElementById('matchContributionTable');

    if (!tbaTeam) {
        oprSection.innerHTML = '<p style="color:#64748b; font-style:italic; font-size:0.9em; margin:0;">Run "Sync TBA OPR" to see OPR data.</p>';
        tableSection.innerHTML = '';
        return;
    }

    const allTBATeams = await db.tbaTeams.toArray();
    const allMatches = await db.matches.toArray();
    const allTeamNums = allTBATeams.map(t => t.teamNumber);
    const oprByTeam = Object.fromEntries(allTBATeams.map(t => [t.teamNumber.toString(), t]));
    const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
    // Individual ignores for this team (exclude any that are also globally ignored)
    const ignoredKeys = new Set(getTeamIgnoredKeys(tbaTeam).filter(k => !globalIgnored.has(k)));
    // Base OPR computation excludes globally ignored + this team's individually ignored matches.
    const playedMatches = allMatches.filter(m =>
        (m.redScore ?? -1) >= 0 &&
        (m.blueScore ?? -1) >= 0 &&
        !globalIgnored.has(m.key) &&
        !ignoredKeys.has(m.key)
    );

    const teamMatches = allMatches
        .filter(m =>
            (!m.compLevel || m.compLevel === 'qm') &&
            ((m.red || []).map(String).includes(teamNumStr) ||
             (m.blue || []).map(String).includes(teamNumStr)))
        .sort((a, b) => a.matchNumber - b.matchNumber);

    // OPR profile stat grid
    const hasIndivIgnore = ignoredKeys.size > 0 && tbaTeam.adjustedOPR != null;
    const effectiveOPR = hasIndivIgnore ? tbaTeam.adjustedOPR : tbaTeam.opr;
    oprSection.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:12px;">
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">OPR</div>
                <div class="stat-value">${effectiveOPR.toFixed(1)}${hasIndivIgnore ? '&thinsp;<span style="color:#fbbf24; font-size:0.65em; font-weight:600;">ADJ</span>' : ''}</div>
            </div>
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">DPR</div>
                <div class="stat-value">${tbaTeam.dpr.toFixed(1)}</div>
            </div>
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">CCWM</div>
                <div class="stat-value" style="color:${tbaTeam.ccwm >= 0 ? '#4ade80' : '#f87171'}">${tbaTeam.ccwm.toFixed(1)}</div>
            </div>
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">Auto OPR</div>
                <div class="stat-value">${tbaTeam.autoOPR != null ? tbaTeam.autoOPR.toFixed(1) : '—'}</div>
            </div>
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">Teleop OPR</div>
                <div class="stat-value">${tbaTeam.teleopOPR != null ? tbaTeam.teleopOPR.toFixed(1) : '—'}</div>
            </div>
            <div style="background:#1a1a1a; padding:12px; border-radius:6px;">
                <div class="stat-label">Endgame OPR</div>
                <div class="stat-value">${tbaTeam.endgameOPR != null ? tbaTeam.endgameOPR.toFixed(1) : '—'}</div>
            </div>
        </div>`;

    // Adjustment banner — list all individually ignored matches
    if (hasIndivIgnore) {
        const labels = [...ignoredKeys].map(k => {
            const m = teamMatches.find(tm => tm.key === k);
            return m ? `Q${m.matchNumber}` : k;
        }).join(', ');
        const scoutActive = !!tbaTeam.scoutingIgnoreActive;
        oprSection.innerHTML += `
            <div style="margin-top:10px; padding:10px 14px; background:#1a1a1a; border-radius:6px; border-left:3px solid #fbbf24; display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                <span style="color:#fbbf24; font-size:0.85em;">Ignoring <strong>${labels}</strong> — adjusted OPR: <strong>${tbaTeam.adjustedOPR.toFixed(1)}</strong> (was ${tbaTeam.opr.toFixed(1)})</span>
                <label style="display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:0.82em;cursor:pointer;margin-left:auto;">
                    <input type="checkbox" ${scoutActive ? 'checked' : ''} onchange="setScoutingIgnore(${teamNumber}, this.checked)"
                        style="accent-color:#818cf8;width:14px;height:14px;">
                    Exclude from scouting tab
                </label>
                <button onclick="setIgnoredMatch(${teamNumber}, null)"
                        style="padding:3px 10px; font-size:0.8em; background:#7f1d1d; border:1px solid #ef4444; border-radius:4px; cursor:pointer; color:#fff;">
                    Clear all
                </button>
            </div>`;
    }

    if (teamMatches.length === 0) {
        tableSection.innerHTML = '<p style="color:#64748b; font-style:italic; font-size:0.9em; margin:0;">Run "Sync Schedule" to see match history.</p>';
        return;
    }

    // Compute base OPR locally so LOO deltas are self-consistent regardless of TBA's exact algorithm.
    const teamIdx = allTeamNums.findIndex(n => n.toString() === teamNumStr);
    const baseOPRs = teamIdx !== -1 ? computeLocalOPR(playedMatches, allTeamNums) : null;
    const baseOPR = baseOPRs ? baseOPRs[teamIdx] : null;

    const rows = teamMatches.map(m => {
        const isRed = (m.red || []).map(String).includes(teamNumStr);
        const alliance = isRed ? (m.red || []) : (m.blue || []);
        const score = isRed ? m.redScore : m.blueScore;
        const played = (score ?? -1) >= 0;
        const isGloballyIgnored = globalIgnored.has(m.key);
        const isIndivIgnored    = ignoredKeys.has(m.key);
        const isInActiveSet     = played && !isGloballyIgnored && !isIndivIgnored;

        // Residual only for matches in the active base set.
        const predicted = isInActiveSet
            ? alliance.reduce((s, t) => s + (oprByTeam[String(t)]?.opr || 0), 0)
            : null;
        const residual = isInActiveSet ? score - predicted : null;

        let looOPR = null, impact = null;
        if (played && !isGloballyIgnored && baseOPR != null) {
            // Active rows: LOO = "OPR if this match were also ignored"
            // Ignored rows: LOO = "OPR if this match were restored"
            const subset = isIndivIgnored
                ? [...playedMatches, m]
                : playedMatches.filter(pm => pm.key !== m.key);
            const looResult = computeLocalOPR(subset, allTeamNums);
            if (looResult) {
                looOPR = looResult[teamIdx];
                impact = baseOPR - looOPR;
            }
        }
        return { m, isRed, score, played, isGloballyIgnored, isIndivIgnored, isInActiveSet, predicted, residual, looOPR, impact };
    });

    const fmtSigned = v => v != null ? (v >= 0 ? '+' : '') + v.toFixed(1) : '—';
    const resColor = v => v == null ? 'inherit' : v >= 0 ? '#4ade80' : '#f87171';

    tableSection.innerHTML = `
        <table class="breakdown-table" style="margin-top:0;">
            <thead><tr>
                <th style="text-align:left;">Match</th>
                <th>Alliance</th>
                <th>Score</th>
                <th>OPR Pred.</th>
                <th>Residual</th>
                <th>OPR w/o</th>
                <th>OPR Impact</th>
                <th></th>
            </tr></thead>
            <tbody>
                ${rows.map(r => {
        const isGlobal       = r.isGloballyIgnored;
        const isIndivIgnored = r.isIndivIgnored;
        const rowStyle = isGlobal
            ? 'cursor:pointer; opacity:0.4;'
            : isIndivIgnored
                ? 'cursor:pointer; background:rgba(251,191,36,0.06);'
                : 'cursor:pointer;';
        const matchLabel = isGlobal
            ? `Q${r.m.matchNumber} <span style="color:#f59e0b; font-size:0.7em; font-weight:600;">GLOBAL</span>`
            : isIndivIgnored
                ? `Q${r.m.matchNumber} <span style="color:#fbbf24; font-size:0.7em; font-weight:600;">IGN</span>`
                : `Q${r.m.matchNumber}`;

        let ignoreBtn = '';
        if (isGlobal) {
            ignoreBtn = `<button onclick="event.stopPropagation();setGloballyIgnored('${r.m.key}',false)"
                            style="padding:3px 10px; font-size:0.8em; background:#92400e; border:1px solid #d97706; color:#fde68a; border-radius:4px; cursor:pointer; white-space:nowrap;">
                            Restore Global</button>`;
        } else if (r.played && r.looOPR != null) {
            ignoreBtn = `<button onclick="event.stopPropagation();setIgnoredMatch(${teamNumber}, '${r.m.key}')"
                            style="padding:3px 10px; font-size:0.8em; background:${isIndivIgnored ? '#92400e' : '#1e293b'}; border:1px solid ${isIndivIgnored ? '#d97706' : '#475569'}; color:${isIndivIgnored ? '#fde68a' : '#94a3b8'}; border-radius:4px; cursor:pointer; white-space:nowrap;">
                            ${isIndivIgnored ? 'Restore' : 'Ignore'}</button>`;
        }

        return `<tr onclick="viewMatchDetail('${r.m.key}')" style="${rowStyle}">
                        <td style="text-align:left;">${matchLabel}</td>
                        <td><span style="color:${r.isRed ? '#ef4444' : '#3b82f6'}; font-weight:bold;">${r.isRed ? 'Red' : 'Blue'}</span></td>
                        <td>${r.played ? r.score : '—'}</td>
                        <td>${r.predicted != null ? r.predicted.toFixed(1) : '—'}</td>
                        <td style="color:${resColor(r.residual)}; font-weight:bold;">${r.isInActiveSet ? fmtSigned(r.residual) : '—'}</td>
                        <td style="color:#94a3b8;">${r.looOPR != null ? r.looOPR.toFixed(1) : '—'}</td>
                        <td style="color:${resColor(r.impact)}; font-weight:bold;">${fmtSigned(r.impact)}</td>
                        <td>${ignoreBtn}</td>
                    </tr>`;
    }).join('')}
            </tbody>
        </table>`;
}

// Returns the array of individually-ignored match keys for a tbaTeam record.
// Supports both the legacy single-key field and the new array field.
function getTeamIgnoredKeys(tba) {
    if (!tba) return [];
    if (Array.isArray(tba.ignoredMatchKeys)) return tba.ignoredMatchKeys;
    if (tba.ignoredMatchKey) return [tba.ignoredMatchKey];
    return [];
}

// Re-averages a cached fused result excluding specific match numbers.
// Returns a new result object with recomputed stats, or null if no matches remain.
function refilteredFusedStats(fusedResult, ignoredMatchNums) {
    if (!fusedResult?.available || !fusedResult.fusedByMatch) return fusedResult;
    const filtered = Object.entries(fusedResult.fusedByMatch)
        .filter(([mn]) => !ignoredMatchNums.has(Number(mn)))
        .map(([, v]) => v);
    if (filtered.length === 0) return null;
    const allKeys = Object.keys(filtered[0] ?? {});
    const stats = {};
    for (const key of allKeys) {
        const vals = filtered.map(f => f[key]).filter(v => v != null);
        stats[key] = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    }
    return { ...fusedResult, stats };
}

function refreshDetailEpaCard(team) {
    const card = document.getElementById('detailEpaCard');
    if (!card || !team) return;
    card.innerHTML = `
        <label style="color:#888; font-size:0.8em;">${isLocalEpaEnabled() && !teamHasSbEventData(team) ? 'LOCAL EPA' : 'CURRENT EPA'}</label>
        <div style="font-size:1.5em; font-weight:bold;">${team.currentEPA != null ? team.currentEPA.toFixed(1) : '0'}</div>
    `;
}

async function refreshEPADisplays(teamNumber) {
    await Promise.all([
        displayTeams(),
        displayTBATeams(),
        displayScoutingTeams(),
        renderAtAGlance(),
        renderPickList(),
        renderDraft(),
        teamNumber != null ? renderTBADetail(String(teamNumber), activeTBAData) : Promise.resolve(),
        teamNumber != null ? renderScoutingTab(String(teamNumber))               : Promise.resolve(),
        teamNumber != null && activeTeamData
            ? renderOverview(activeTeamData, activeTBAData) : Promise.resolve(),
    ]);
    refreshDetailEpaCard(activeTeamData);
}

// Toggle a match key in/out of a team's individual ignore list.
// matchKey === null clears all ignored keys.
window.setIgnoredMatch = async function (teamNumber, matchKey) {
    const pk = parseInt(teamNumber);
    const current = await db.tbaTeams.get(pk);
    let keys = getTeamIgnoredKeys(current);

    if (matchKey === null) {
        keys = [];
    } else if (keys.includes(matchKey)) {
        keys = keys.filter(k => k !== matchKey);
    } else {
        keys = [...keys, matchKey];
    }

    let adjustedOPR = null;
    if (keys.length > 0) {
        const allTBATeams = await db.tbaTeams.toArray();
        const allMatches  = await db.matches.toArray();
        const allTeamNums = allTBATeams.map(t => t.teamNumber);
        const globalIgnored = new Set(allMatches.filter(m => m.globallyIgnored).map(m => m.key));
        const keySet = new Set(keys);
        const subset = allMatches.filter(m =>
            (m.redScore ?? -1) >= 0 && (m.blueScore ?? -1) >= 0 &&
            !globalIgnored.has(m.key) && !keySet.has(m.key)
        );
        const result = computeLocalOPR(subset, allTeamNums);
        const idx = allTeamNums.findIndex(n => n === pk);
        if (result && idx !== -1) adjustedOPR = result[idx];
    }

    await db.tbaTeams.update(pk, {
        ignoredMatchKeys: keys.length > 0 ? keys : null,
        ignoredMatchKey:  null,
        adjustedOPR:      keys.length > 0 ? adjustedOPR : null,
    });
    activeTBAData = await db.tbaTeams.get(pk);
    if (isLocalEpaEnabled()) {
        await computeLocalEPA(); // recomputes for all local teams; calls refreshEPADisplays internally
    } else {
        await refreshEPADisplays(teamNumber);
    }
    // Refresh the performance chart if the matches tab is currently open for this team
    if (lastDetailTab === 'matches' && activeTeamData?.teamNumber === pk) {
        await renderMatchesTab(pk, 'matches-tab-perf-chart');
    }
};

window.setScoutingIgnore = async function (teamNumber, active) {
    const pk = parseInt(teamNumber);
    await db.tbaTeams.update(pk, { scoutingIgnoreActive: active || null });
    activeTBAData = await db.tbaTeams.get(pk);
    await refreshEPADisplays(teamNumber);
};

function renderChart(team) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    if (performanceChart) performanceChart.destroy();

    const playedMatches = (team.rawStatboticsData || []).filter(m => m.epa?.post);
    const epaData = playedMatches.map(m => m.epa.post);
    const eventLabels = playedMatches.map(m => m.event);

    const isMobile = document.body.classList.contains('mobile-ui');

    // Use the currently-active event key (from the input field) rather than team.eventKey,
    // which can be stale when statbotics fails to update the record for the current year.
    const activeEventKey = document.getElementById('eventKeyInput')?.value.trim().toLowerCase() || team.eventKey;
    const localTimeline = isLocalEpaEnabled() ? (team.localEPATimeline || []) : [];
    const showLocalPoints = isLocalEpaEnabled() && activeEventKey &&
        !teamHasSbEventData(team) &&
        (localTimeline.length > 0 || team.currentEPA != null);
    const localColor = showLocalPoints ? getEventColor(activeEventKey) : null;

    const chartLabels = epaData.map((_, i) => `M${i + 1}`);
    if (showLocalPoints) {
        if (localTimeline.length > 0) {
            localTimeline.forEach(pt => chartLabels.push(pt.label));
        } else {
            chartLabels.push(activeEventKey.toUpperCase()); // pre-event baseline, no matches yet
        }
    }

    const datasets = [{
        label: 'Match EPA',
        data: epaData,
        showLine: false,
        pointRadius: isMobile ? 2.5 : 5,
        pointHoverRadius: isMobile ? 4 : 7,
        pointBackgroundColor: eventLabels.map(ev => getEventColor(ev)),
        pointBorderColor: eventLabels.map(ev => getEventColor(ev))
    }];

    if (showLocalPoints) {
        const localData = new Array(epaData.length).fill(null);
        if (localTimeline.length > 0) {
            localTimeline.forEach(pt => localData.push(pt.epa));
        } else {
            localData.push(team.currentEPA); // pre-event baseline dot
        }
        datasets.push({
            label: `${activeEventKey.toUpperCase()} (Est)`,
            data: localData,
            showLine: false,
            pointRadius: isMobile ? 2.5 : 4,
            pointHoverRadius: isMobile ? 4 : 6,
            pointBackgroundColor: 'transparent',
            pointBorderColor: localColor,
            pointBorderWidth: 3,
        });
    }

    if (team.analysis && team.analysis.rawParams) {
        const trendData = new Array(epaData.length).fill(null);
        const { A, B, k } = team.analysis.rawParams;
        const startIndex = team.analysis.startIndex;

        // How many matches are in our specific selection?
        const selectionLength = team.analysis.rawParams.n;

        // We use x = i + 1 to match the new "Preferred" math engine
        for (let i = 0; i < selectionLength; i++) {
            const x = i + 1;
            const y = A - B * Math.exp(-k * x);
            trendData[startIndex + i] = y;
        }

        datasets.push({
            label: 'Projected Ceiling (Range)',
            data: trendData,
            borderColor: '#4ade80',
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
            spanGaps: false // Keeps the line strictly within the range
        });
    }

    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { bottom: 8 } },
            plugins: {
                legend: {
                    labels: {
                        // Custom legend to show event colors
                        generateLabels: (chart) => {
                            const playedMatches = (team.rawStatboticsData || []).filter(m => m.epa?.post);
                            const uniqueEvents = [...new Set(playedMatches.map(m => m.event))];

                            const labels = uniqueEvents.map(ev => ({
                                text: ev.toUpperCase(),
                                fillStyle: getEventColor(ev),
                                strokeStyle: getEventColor(ev),
                                lineWidth: 0,
                                fontColor: '#f8fafc'
                            }));

                            if (showLocalPoints) {
                                labels.push({
                                    text: `${activeEventKey.toUpperCase()} (Est)`,
                                    fillStyle: 'transparent',
                                    strokeStyle: localColor,
                                    lineWidth: 2,
                                    fontColor: '#f8fafc'
                                });
                            }

                            return labels;
                        }
                    }
                }
            },
            scales: {
                y: { grid: { color: '#333' }, ticks: { color: '#aaa' } },
                x: { grid: { display: false }, ticks: { color: '#aaa' } }
            }
        }
    });
}

// Helper to generate the trendline points
function calculateTrendLine(params, length) {
    if (!params) return [];
    const { A, B, k } = params;
    return Array.from({ length }, (_, i) => {
        const xScaled = i / (length - 1);
        return A - B * Math.exp(-k * xScaled);
    });
}


// Global color map to keep event colors consistent across the app
const eventColorCache = {};
const palette = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4'];

function getEventColor(eventKey) {
    if (!eventColorCache[eventKey]) {
        // If we haven't seen this event yet, pick the next color from the palette
        const index = Object.keys(eventColorCache).length % palette.length;
        eventColorCache[eventKey] = palette[index];
    }
    return eventColorCache[eventKey];
}


// Bottom of main.js - The App Bootloader
const bootApp = async () => {

    // At the top of bootApp
    const savedKey = localStorage.getItem('lastEventKey');
    if (savedKey) {
        document.getElementById('eventKeyInput').value = savedKey;
        updateAppEventKey(savedKey);
    } else {
        showEventSelector();
    }

    renderScoutingSection();

    // Re-render scouting section and check for public archive whenever the event key changes
    document.getElementById('eventKeyInput')?.addEventListener('input', () => {
        renderScoutingSection();
        const eventKey = document.getElementById('eventKeyInput').value.trim().toLowerCase();
        updateAppEventKey(eventKey || null);
        clearTimeout(_archiveCheckTimer);
        document.getElementById('archiveHint').innerHTML = '';
        _archiveCheckTimer = setTimeout(() => checkEventArchive(eventKey), 500);
    });

    // Restore sync timestamps and OBE indicators
    for (const key of ['statboticsLive', 'tbaOPR', 'tbaMatches']) {
        const saved = localStorage.getItem(`lastSync_${key}`);
        const el = document.getElementById(`ts-${key}`);
        if (saved && el) el.textContent = `Last sync: ${saved}`;
    }
    updateOBEStatus(localStorage.getItem('lastEventKey'));

    // 1. Set the initial view (Home)
    initUIMode();
    initColorMode();
    initNotifications();
    initNexusIntegration();
    updateLocalEpaUI();
    setInterval(updateBannerTick, 1000);
    window.switchView('homeView');

    // 2. Load the cached data into the tables immediately
    // This ensures that when you click 'Statbotics' or 'Schedule', 
    // the data is already waiting for you.
    try {
        await displayTeams();        // Loads Statbotics cache
        await displaySchedule();     // Loads TBA Schedule cache
        await updateHomeBanner();
        await displayTBATeams();     // Loads TBA OPR cache
        await renderAtAGlance();     // Loads at-a-glance overview
        console.log("Local cache successfully loaded into UI.");
    } catch (err) {
        console.warn("No cached data found to load yet.");
    }

    await checkAdjustmentsFromURL();
};

bootApp();

// Automatically set the view to 'homeView' when the script finishes loading.
// A #view/tab deep link overrides that -- the curator's Back button uses #tools/tracks
// to return to the queue it was launched from. This has to happen HERE rather than in
// bootApp, because this handler unconditionally switches to homeView and would
// otherwise undo it.
const DEEP_LINKS = {
    '#tools/tracks': () => { switchView('toolsView'); switchToolsTab('tracks'); },
};
document.addEventListener('DOMContentLoaded', () => {
    const deep = DEEP_LINKS[location.hash];
    if (deep) {
        history.replaceState(null, '', location.pathname + location.search);
        deep();
        return;
    }
    const homeBtn = document.querySelector('.nav-btn'); // Grabs the first button (Home)
    switchView('homeView', homeBtn);
});

// Initialize the view once the script is ready
const homeBtn = document.querySelector('.nav-btn');
window.switchView('homeView', homeBtn);

document.getElementById('eventKeyInput').value = localStorage.getItem('lastEventKey') || '';
checkEventArchive(document.getElementById('eventKeyInput').value.trim().toLowerCase());

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register(import.meta.env.BASE_URL + 'sw.js').catch(() => { });
}

window.addEventListener('resize', positionMobileSubTabs);
window.addEventListener('orientationchange', () => setTimeout(positionMobileSubTabs, 100));






