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
// Free-tier KV is 100k reads / 1k writes a day; one match round-trip is a handful of
// each, so an event day is nowhere near it.
// ─────────────────────────────────────────────────────────────────────────────

const KINDS = new Set(['bundle', 'answer', 'calib', 'points']);

// KV caps values at 25 MiB. Curation bundles are ~1.8-7 MB depending on how many
// frames and what JPEG quality rtrack.curate was told to use, so this is headroom
// rather than a limit we expect to hit — but fail loudly if we ever do, because the
// alternative is a silently truncated bundle that renders as a broken page.
const MAX_BYTES = 24 * 1024 * 1024;
const TTL_S = 86400;   // one event day

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
  m[`${kind}:${id}`] = { kind, id, ...meta, expires: Math.floor(Date.now() / 1000) + TTL_S };
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
    const DEVICE_WRITABLE = new Set(['answer', 'points']);

    // Returns 'full' | 'device' | null.
    const level = () => {
      const t = request.headers.get('Rtrack-Token') ?? '';
      if (env.RTRACK_TOKEN && t === env.RTRACK_TOKEN) return 'full';
      if (env.RTRACK_ANSWER_TOKEN && t === env.RTRACK_ANSWER_TOKEN) return 'device';
      if (!env.RTRACK_TOKEN) return 'full';          // unconfigured: local dev
      // No answer token configured => answering is open to anyone. See the header.
      if (!env.RTRACK_ANSWER_TOKEN) return 'device';
      return null;
    };

    if (request.method === 'POST') {
      const lv = level();
      if (!lv) return json({ ok: false, error: 'unauthorized' }, 401);
      if (lv === 'device' && !DEVICE_WRITABLE.has(kind)) {
        return json({ ok: false, needsToken: true,
                      error: `posting to /${kind} needs the home-machine token; `
                             + `this endpoint only accepts ${[...DEVICE_WRITABLE].join(' and ')}` }, 403);
      }
      const body = await request.text();
      if (body.length > MAX_BYTES) {
        return json({ ok: false, error: `payload ${body.length} > ${MAX_BYTES} bytes; `
                      + 'lower --frames or the JPEG quality in rtrack.curate' }, 413);
      }
      try { JSON.parse(body); } catch { return json({ ok: false, error: 'invalid JSON' }, 400); }
      const at = Date.now();
      await env.RTRACK_KV.put(kvKey, body, {
        expirationTtl: TTL_S,
        metadata: { bytes: body.length, at },
      });
      // After the value is stored, never before: a manifest entry for a value that failed
      // to write would advertise a bundle that 404s.
      await touchIndex(env, kind, id, { bytes: body.length, at });
      return json({ ok: true, key: kvKey, bytes: body.length });
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
