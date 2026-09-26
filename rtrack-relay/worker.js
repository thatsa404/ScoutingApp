// rtrack relay — Cloudflare Worker
//
// Moves curation and calibration work between a home machine (GPU, video, pipeline)
// and whatever device the human is holding at an event. Neither can reach the other
// directly: the home machine sits behind residential NAT, the app is a static site on
// Pages, and the phone is on venue wifi. Both can reach a Worker.
//
// Same shape as nexus-relay/worker.js — token-authed POST, CORS GET, KV with TTL — but
// keyed per artifact rather than a single "latest", and bidirectional: bundles travel
// down, answers travel back up.
//
//   POST /bundle/<matchKey>      home  -> relay   curation bundle (JSON)
//   GET  /bundle/<matchKey>      relay -> phone
//   POST /answer/<matchKey>      phone -> relay   corrections (JSON)
//   GET  /answer/<matchKey>      relay -> home
//   POST /calib/<videoId>        home  -> relay   a frame + field ref to click on
//   GET  /calib/<videoId>        relay -> phone
//   POST /points/<videoId>       phone -> relay   [[vx,vy,fx,fy], ...]
//   GET  /points/<videoId>       relay -> home
//   POST /occl/<cameraId>        phone -> relay   occluder regions (JSON)
//   GET  /occl/<cameraId>        relay -> home
//   POST /tracks/<matchKey>      home  -> relay   exported routes (JSON)
//   GET  /tracks/<matchKey>      relay -> app
//   POST /control/<agentId>      app   -> relay   desired state for the home machine
//   GET  /control/<agentId>      relay -> home
//   POST /status/<agentId>       home  -> relay   heartbeat + what it is doing
//   GET  /status/<agentId>       relay -> app
//   POST /job/<jobId>            app   -> relay   a unit of work to run (detect, bundle)
//   GET  /job/<jobId>            relay -> home
//
// `tracks` is the one kind the APP reads rather than a curator. Routes used to reach
// the app only through git: rtrack.export wrote public/tracks/ on the home machine and
// GitHub Pages served whatever had been committed, so a match could be curated, solved,
// projected and exported and still be invisible until someone remembered to commit.
// Ten matches sat like that. Now the export posts here and the app caches to IndexedDB;
// git remains the durable path for past events and their archives.
//
// `occl` rides the same path as `points`: the phone draws on a frame the home machine
// posted to /calib, and the drawing comes back up. It was a file DOWNLOAD before, which
// works on a laptop and not at all on the phone the tool is designed for -- the file
// lands in Downloads on a device that cannot reach robot-tracker/calib/.
//
// `control` and `status` are the remote-control pair, and control is DESIRED STATE rather
// than a command queue. A queue needs acknowledgement, dedup and ordering; desired state
// is idempotent, so re-posting it is a no-op and an agent that reboots simply reads what
// it should be doing and resumes. `status.appliedNonce` echoing `control.nonce` is what
// turns "the app sent a command" into "the home machine received it" -- without it the UI
// can only ever claim to have sent something.
//   GET  /index                  what is available right now
//   DELETE /<kind>/<id>          clear one entry (token required)
//
// AUTH, AND WHY ANSWERING IS OPEN BY DEFAULT.
//
//   RTRACK_TOKEN         required to write bundles/calib frames and to DELETE.
//                        This is the home machine's key and only it should hold one.
//   RTRACK_ANSWER_TOKEN  OPTIONAL. Unset (the default), /answer and /points accept
//                        unauthenticated POSTs, so a curator opens the page and works
//                        with nothing to configure. Set it, and they are required.
//   RTRACK_CONTROL_TOKEN OPTIONAL, and REQUIRED-BY-ABSENCE: unset, /control rejects every
//                        POST rather than accepting them. This is the exact OPPOSITE of
//                        the answer default above, deliberately. An unauthenticated answer
//                        lands in a 24 h entry for one match and a human reviews it before
//                        it is applied -- bounded. An unauthenticated /control makes a
//                        machine somebody owns start opening streams and burning GPU time
//                        on request, which is not bounded and not reviewable after the
//                        fact. So arming is opt-in via a secret, and a relay that has not
//                        been given one simply cannot be armed.
//
// Answering is open because the alternative is worse, not because nobody thought about
// it. The curator page is served from a public GitHub Pages site, so any token embedded
// in it to make the flow "automatic" would be public too — that is friction with the
// appearance of security and none of the substance. The real choices are: make the
// person type a secret, or accept that anyone who learns the URL can post answers.
//
// The blast radius of an unauthenticated answer is small and bounded: it lands in a
// 24-hour KV entry, it only affects one match, it cannot touch a bundle or delete
// anything, and the home-machine operator sees the corrections before they are applied.
// Set RTRACK_ANSWER_TOKEN if that stops being an acceptable trade — no code change, and
// the pages will prompt for it the first time they get a 401.
//
// Reads need no token either. A leaked read is a curation bundle of a public match
// video.
//
// ── Deploy ───────────────────────────────────────────────────────────────────
// Run these from inside rtrack-relay/. `npx` fetches wrangler on demand, so nothing
// has to be installed or put on PATH:
//
//   npx wrangler login
//   npx wrangler kv namespace create RTRACK_KV   → paste the id into wrangler.toml
//   npx wrangler deploy                          → deploy FIRST, see below
//   npx wrangler secret put RTRACK_TOKEN         → the home machine's key
//   npx wrangler deploy                          → again, to pick the secret up
//
// Deploy before setting the secret. `secret put` against a worker that does not exist
// yet prompts "there doesn't seem to be a worker called rtrack-relay. create it?" --
// answering yes is harmless, it just makes an empty worker for the secret to live on,
// but deploying first skips the question.
//
// Then put the deployed URL and that token in the repo .env as RTRACK_RELAY_URL and
// RTRACK_TOKEN. Do NOT set RTRACK_ANSWER_TOKEN unless you want curators to have to
// type a secret — answering is open by default, see the auth note below.
//
// To allow the app to arm the home machine remotely, also:
//
//   npx wrangler secret put RTRACK_CONTROL_TOKEN   → the lead scout's key
//   npx wrangler deploy
//
// The lead scout types that value into the app once (Tools → Tracks → Relay Control); it
// lives in their browser's localStorage and never in the repo or the built bundle. Skip
// this and /control stays closed, which is the safe default — see the auth note.
//
// Free-tier KV is 100k reads / 1k writes a day; one match round-trip is a handful of
// each, so an event day is nowhere near it.
// ─────────────────────────────────────────────────────────────────────────────

const KINDS = new Set(['bundle', 'answer', 'calib', 'points', 'occl', 'tracks',
                       'gallery-bundle', 'gallery-answer', 'gallery-status',
                       'control', 'status', 'job']);

// KV caps values at 25 MiB. Curation bundles are ~1.8-7 MB depending on how many
// frames and what JPEG quality rtrack.curate was told to use, so this is headroom
// rather than a limit we expect to hit — but fail loudly if we ever do, because the
// alternative is a silently truncated bundle that renders as a broken page.
const MAX_BYTES = 24 * 1024 * 1024;
const TTL_S = 86400;   // one event day

// PER-KIND OVERRIDES. A curation bundle is worthless the day after the event, but
// ROUTES are the deliverable -- an app that cached nothing on day one should still find
// day one's routes on day three. They are also small (a 5 Hz export is ~170 KB against a
// bundle's 4 MB), so a longer life costs almost nothing. Anything absent here gets
// TTL_S.
// A HEARTBEAT DELIBERATELY OUTLIVES THE MACHINE THAT SENT IT. The obvious design is a
// short TTL so a dead agent's status vanishes, but then the app can only say "no agent"
// and cannot say "last seen 22 minutes ago" -- and the second one is the whole point of
// showing a heartbeat to a lead scout. Staleness is computed from `at` in the app, where
// it can be rendered, rather than enforced by expiry here.
const TTL_BY_KIND = {
  tracks: 7 * 86400,
  'gallery-bundle': 7 * 86400,
  'gallery-answer': 30 * 86400,
  'gallery-status': 30 * 86400,
  control: 7 * 86400,
  status: 7 * 86400,
  // A backfill can be 75 matches of GPU work across more than a day, and the request must
  // still be discoverable while it runs -- a job that expired mid-run would look to the
  // agent like one it had never been asked to do.
  job: 7 * 86400,
};
const ttlFor = kind => TTL_BY_KIND[kind] ?? TTL_S;

// Gallery reviews are submitted one team at a time in the Tracks tab, but the
// relay key is one bundle-wide answer. Merge schema-2 submissions by team so a
// later team submission does not erase earlier teams. A second submission for
// the same team intentionally replaces that team's prior selection.
function mergeGalleryAnswer(existing, incoming) {
  if (!existing || existing.schemaVersion !== 2 || incoming.schemaVersion !== 2) {
    return incoming;
  }
  if (existing.bundleHash && incoming.bundleHash && existing.bundleHash !== incoming.bundleHash) {
    return null;
  }

  const selections = new Map();
  for (const item of Array.isArray(existing.selections) ? existing.selections : []) {
    if (item?.team != null) selections.set(String(item.team), item);
  }
  for (const item of Array.isArray(incoming.selections) ? incoming.selections : []) {
    if (item?.team != null) {
      const prior = selections.get(String(item.team));
      const next = { team: String(item.team),
        include: Array.isArray(item.include) ? item.include : [] };
      if (Array.isArray(item.reviewed)) next.reviewed = item.reviewed;
      else if (Array.isArray(prior?.reviewed)) next.reviewed = prior.reviewed;
      selections.set(String(item.team), next);
    }
  }

  const teamStates = new Map();
  for (const item of Array.isArray(existing.teamStates) ? existing.teamStates : []) {
    if (item?.team != null) teamStates.set(String(item.team), item);
  }
  for (const item of Array.isArray(incoming.teamStates) ? incoming.teamStates : []) {
    if (item?.team != null) teamStates.set(String(item.team), {
      team: String(item.team), state: item.state,
    });
  }

  return {
    ...existing,
    ...incoming,
    selections: [...selections.values()],
    teamStates: [...teamStates.values()],
    firstSubmittedAt: existing.firstSubmittedAt || existing.submittedAt || incoming.submittedAt,
    mergedAt: new Date().toISOString(),
    submissionCount: Number(existing.submissionCount || 1) + 1,
  };
}

// THE INDEX IS A KEY, NOT A list() CALL, and that is a hard requirement rather than an
// optimisation. KV list() is capped at 1000 operations PER DAY on the free plan --
// separately from the 100k reads -- and /index is the one endpoint everything polls:
// rtrack.watch, the app's Tracks tab, and the curator's "What's available?". At watch.py's
// original 20 s poll that is 4,320 list calls a day, so the quota burned through in a few
// hours and every /index after that threw "KV list() limit exceeded for the day" -- a 500
// that took the Tracks listing down with it while every other endpoint stayed healthy.
//
// So writes maintain this manifest and /index reads it as an ordinary get. One list() per
// day at most, to rebuild it if it is ever missing.
const INDEX_KEY = 'idx:manifest';

// Read-modify-write, so two POSTs landing in the same instant can lose one entry. That is
// accepted rather than overlooked: the loser is only missing from the LISTING, its actual
// value is stored correctly and still fetchable by key, and the next write to that key
// restores it. Making this correct needs a Durable Object, which is a lot of machinery for
// a listing that one person reads between matches.
async function touchIndex(env, kind, id, meta) {
  let m = {};
  try { m = (await env.RTRACK_KV.get(INDEX_KEY, 'json')) || {}; } catch { m = {}; }
  m[`${kind}:${id}`] = { kind, id, ...meta,
                         expires: Math.floor(Date.now() / 1000) + ttlFor(kind) };
  await env.RTRACK_KV.put(INDEX_KEY, JSON.stringify(m));
}

export default {
  async fetch(request, env) {
    const cors = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Rtrack-Token',
      'Access-Control-Max-Age': '86400',
    };
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });

    const url = new URL(request.url);
    const parts = url.pathname.split('/').filter(Boolean);
    const json = (obj, status = 200) => new Response(JSON.stringify(obj), {
      status, headers: { ...cors, 'Content-Type': 'application/json' },
    });

    if (parts[0] === 'index' && request.method === 'GET') {
      const now = Math.floor(Date.now() / 1000);
      let m = await env.RTRACK_KV.get(INDEX_KEY, 'json');
      let rebuilt = false;
      if (!m) {
        // Bootstrap: the only list() left in the worker, and it runs once per manifest
        // lifetime rather than once per poll. If the daily quota is already spent this
        // still throws, so it is caught -- a degraded empty listing beats a 500 that
        // takes the whole Tracks tab down.
        try {
          const list = await env.RTRACK_KV.list({ limit: 1000 });
          m = {};
          for (const k of list.keys) {
            if (k.name === INDEX_KEY) continue;
            m[k.name] = {
              kind: k.name.split(':')[0],
              id: k.name.split(':').slice(1).join(':'),
              expires: k.expiration ?? null,
              ...(k.metadata || {}),
            };
          }
          await env.RTRACK_KV.put(INDEX_KEY, JSON.stringify(m));
          rebuilt = true;
        } catch (e) {
          return json({ ok: true, count: 0, items: [], degraded: String(e.message || e) });
        }
      }
      const items = Object.entries(m)
        .filter(([, v]) => !v.expires || v.expires > now)
        .map(([key, v]) => ({ key, ...v }));
      return json({ ok: true, count: items.length, items, ...(rebuilt && { rebuilt: true }) });
    }

    if (parts.length !== 2 || !KINDS.has(parts[0])) {
      return json({ ok: false, error: `use /<${[...KINDS].join('|')}>/<id> or /index` }, 404);
    }
    const [kind, id] = parts;
    const kvKey = `${kind}:${id}`;

    // Paths a curator's device is allowed to write. Everything else is home-machine only.
    // `control` is deliberately absent: the open answer path must never be able to arm a
    // machine, so a device-level caller posting to /control falls through to a 403 below
    // exactly as it would for /bundle.
    const DEVICE_WRITABLE = new Set(['answer', 'points', 'occl', 'gallery-answer']);

    // What each non-full level may write. 'full' is unrestricted and not listed.
    // `status` appears in no set, so only the home machine can report a heartbeat -- a
    // phone must not be able to forge "the GPU box is alive and idle".
    const WRITABLE_BY_LEVEL = {
      // `job` rides with `control` rather than getting a tier of its own: both spend the
      // home machine's GPU time on request, so there is no version of this where one is
      // safe to leave open and the other is not.
      control: new Set(['control', 'job']),
      device: DEVICE_WRITABLE,
    };

    // Returns 'full' | 'control' | 'device' | null.
    const level = () => {
      const t = request.headers.get('Rtrack-Token') ?? '';
      if (env.RTRACK_TOKEN && t === env.RTRACK_TOKEN) return 'full';
      if (env.RTRACK_CONTROL_TOKEN && t === env.RTRACK_CONTROL_TOKEN) return 'control';
      if (env.RTRACK_ANSWER_TOKEN && t === env.RTRACK_ANSWER_TOKEN) return 'device';
      if (!env.RTRACK_TOKEN) return 'full';          // unconfigured: local dev
      // No answer token configured => answering is open to anyone. See the header. This
      // grants 'device', which cannot reach /control, so an unconfigured
      // RTRACK_CONTROL_TOKEN leaves arming closed rather than open.
      if (!env.RTRACK_ANSWER_TOKEN) return 'device';
      return null;
    };

    if (request.method === 'POST') {
      const lv = level();
      if (!lv) return json({ ok: false, error: 'unauthorized' }, 401);
      if (lv !== 'full' && !(WRITABLE_BY_LEVEL[lv] ?? new Set()).has(kind)) {
        const err = (kind === 'control' || kind === 'job')
          ? `posting to /${kind} needs RTRACK_CONTROL_TOKEN. If the relay has no `
            + `RTRACK_CONTROL_TOKEN secret set, arming is closed by design -- set one with `
            + '`npx wrangler secret put RTRACK_CONTROL_TOKEN` and redeploy.'
          : `posting to /${kind} needs the home-machine token; `
            + `this token only accepts ${[...(WRITABLE_BY_LEVEL[lv] ?? [])].join(' and ')}`;
        return json({ ok: false, needsToken: true, error: err }, 403);
      }
      const body = await request.text();
      if (body.length > MAX_BYTES) {
        return json({ ok: false, error: `payload ${body.length} > ${MAX_BYTES} bytes; `
                      + 'lower --frames or the JPEG quality in rtrack.curate' }, 413);
      }
      let payload;
      try { payload = JSON.parse(body); } catch { return json({ ok: false, error: 'invalid JSON' }, 400); }
      if (kind === 'gallery-bundle' && body.length > 4 * 1024 * 1024) {
        return json({ ok: false, error: 'gallery review bundle exceeds 4 MiB' }, 413);
      }
      let storedPayload = payload;
      if (kind === 'gallery-answer' && payload.schemaVersion === 2) {
        const existing = await env.RTRACK_KV.get(kvKey, 'json');
        const merged = mergeGalleryAnswer(existing, payload);
        if (!merged) {
          return json({ ok: false, error: 'gallery answer belongs to a different bundle' }, 409);
        }
        storedPayload = merged;
      }
      const storedBody = JSON.stringify(storedPayload);
      const at = Date.now();
      await env.RTRACK_KV.put(kvKey, storedBody, {
        expirationTtl: ttlFor(kind),
        metadata: { bytes: storedBody.length, at },
      });
      // After the value is stored, never before: a manifest entry for a value that failed
      // to write would advertise a bundle that 404s.
      const reviewMeta = kind.startsWith('gallery-') ? {
        season: storedPayload.season ?? null,
        match: storedPayload.match ?? (Array.isArray(storedPayload.teams)
          ? storedPayload.teams.flatMap(t => [...(t.candidates || []), ...(t.currentGallery || [])])
              .map(image => image?.source?.match).find(Boolean) ?? null
          : null),
        reviewId: storedPayload.reviewId ?? id,
        galleryVersion: storedPayload.galleryVersion ?? storedPayload.baseGalleryVersion ?? null,
        schemaVersion: storedPayload.schemaVersion ?? null,
        teamIds: Array.isArray(storedPayload.teams) ? storedPayload.teams.map(t => String(t.team)) : null,
        currentImagesByTeam: Array.isArray(storedPayload.teams)
          ? Object.fromEntries(storedPayload.teams.map(t => [String(t.team), (t.currentGallery || []).length])) : null,
        candidatesByTeam: Array.isArray(storedPayload.teams)
          ? Object.fromEntries(storedPayload.teams.map(t => [String(t.team), (t.candidates || []).length])) : null,
        candidateCount: Array.isArray(storedPayload.candidates)
          ? storedPayload.candidates.length
          : Array.isArray(storedPayload.teams) ? storedPayload.teams.reduce((n, t) => n + (t.candidates || []).length, 0) : null,
        decisionCount: Array.isArray(storedPayload.decisions) ? storedPayload.decisions.length : null,
        selectionCount: Array.isArray(storedPayload.selections) ? storedPayload.selections.length : null,
        state: kind === 'gallery-answer' ? 'answered' : payload.state ?? 'ready',
      } : {};
      // Enough of a status/control doc to render the heartbeat strip from /index alone.
      // The Tracks tab already polls /index, so the common case costs no extra request --
      // it only fetches the full doc when someone opens the panel.
      const agentMeta = kind === 'status' ? {
        state: storedPayload.state ?? null,
        event: storedPayload.event ?? null,
        appliedNonce: storedPayload.appliedNonce ?? null,
        watcherAlive: storedPayload.watcher?.alive ?? null,
        pendingAnswers: storedPayload.queue?.pendingAnswers ?? null,
        pendingGallery: storedPayload.queue?.pendingGallery ?? null,
        error: storedPayload.error ?? null,
      } : kind === 'control' ? {
        desired: storedPayload.desired ?? null,
        event: storedPayload.event ?? null,
        nonce: storedPayload.nonce ?? null,
      } : kind === 'job' ? {
        // Enough for the agent to decide whether a job is ITS job and worth claiming,
        // without fetching every job document on every poll.
        agentId: storedPayload.agentId ?? null,
        jobType: storedPayload.type ?? null,
        event: storedPayload.event ?? null,
        matches: storedPayload.matches ?? null,
        count: storedPayload.count ?? null,
        cancelled: storedPayload.cancelled === true ? true : null,
      } : {};
      await touchIndex(env, kind, id, { bytes: storedBody.length, at,
                                        ...reviewMeta, ...agentMeta });
      return json({ ok: true, key: kvKey, bytes: storedBody.length,
                    merged: kind === 'gallery-answer' && payload.schemaVersion === 2 });
    }

    if (request.method === 'GET') {
      const data = await env.RTRACK_KV.get(kvKey);
      if (data === null) return json({ ok: false, error: 'not found' }, 404);
      return new Response(data, { status: 200, headers: { ...cors, 'Content-Type': 'application/json' } });
    }

    if (request.method === 'DELETE') {
      // Deliberately full-token only: a device token must never be able to destroy
      // work, and the only thing on the far end of a DELETE is work.
      if (level() !== 'full') return json({ ok: false, error: 'unauthorized' }, 401);
      await env.RTRACK_KV.delete(kvKey);
      try {
        const m = (await env.RTRACK_KV.get(INDEX_KEY, 'json')) || {};
        if (kvKey in m) { delete m[kvKey]; await env.RTRACK_KV.put(INDEX_KEY, JSON.stringify(m)); }
      } catch { /* the value is gone either way; a stale listing entry expires on its own */ }
      return json({ ok: true, deleted: kvKey });
    }

    return json({ ok: false, error: 'method not allowed' }, 405);
  },
};
