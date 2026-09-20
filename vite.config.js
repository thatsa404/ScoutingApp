import { defineConfig, loadEnv } from 'vite'

// The relay URL, served to the standalone curator/calibrator pages as a tiny script.
//
// WHY A PLUGIN AND NOT import.meta.env. public/rtrack/*.html are copied verbatim --
// Vite never parses them, so `import.meta.env` is just text there. They are standalone
// on purpose: a curator at an event opens one page on a phone, and it must not depend
// on the app bundle booting first.
//
// The three surfaces already SHARE the answer once it is entered: main.js, curate.html
// and calibrate.html all read localStorage['rtrackRelay'], and localStorage is scoped
// per ORIGIN, not per path. What was missing is only the first entry on a fresh device,
// which is exactly the moment a scout is standing in a venue being handed a phone.
//
// NOTE ON EXPOSURE. This publishes the relay URL on a public site. That is a real
// change of posture, not a formality: rtrack-relay accepts answers without a token
// unless RTRACK_ANSWER_TOKEN is set (see rtrack-relay/wrangler.toml). Set that secret
// before relying on this. It is a low-privilege token by design -- it cannot overwrite
// a bundle or delete anything -- so baking it alongside costs a curator nothing.
function rtrackDefaults(env) {
  const body = `// generated at build time from VITE_RTRACK_RELAY -- do not edit\n`
    + `window.RTRACK_DEFAULT_RELAY = ${JSON.stringify(env.VITE_RTRACK_RELAY || '')};\n`
    + `window.RTRACK_DEFAULT_TOKEN = ${JSON.stringify(env.VITE_RTRACK_ANSWER_TOKEN || '')};\n`
  const URL_PATH = '/rtrack/relay.js'
  return {
    name: 'rtrack-defaults',
    // Dev: serve it from memory. Writing into public/ would leave a generated file in
    // the working tree that is easy to commit by accident.
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.split('?')[0].endsWith(URL_PATH)) return next()
        res.setHeader('Content-Type', 'application/javascript')
        res.end(body)
      })
    },
    // Build: emit it as a real asset at the same path the pages reference.
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'rtrack/relay.js', source: body })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    // MUST match the GitHub Pages repo name. This still said '/1768Scouting/' after the
    // repo was renamed to ScoutingApp, so the deployed index.html asked for
    // /1768Scouting/assets/index-*.js -- a 404 -- while the bundle sat correctly at
    // /ScoutingApp/assets/. The page rendered with its CSS and no JS at all, which
    // reads as "buttons do nothing" rather than as a missing file.
    base: '/ScoutingApp/',
    plugins: [rtrackDefaults(env)],
  }
})
