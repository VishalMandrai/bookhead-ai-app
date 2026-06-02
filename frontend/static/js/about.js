/**
 * about.js — BookHead AI
 *
 * Responsibilities:
 *   - Fetch about.html from the server (once, lazily on first activation)
 *   - Inject the markup into #panel-about
 *   - Expose an `init()` hook that app.js can call during bootstrap
 *
 * The module deliberately has no knowledge of the tab-switching logic in
 * app.js; it only cares about its own panel element.
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const PANEL_ID        = 'panel-about';
const ABOUT_HTML      = '/static/css/about.html';
const CONFIG_ENDPOINT = '/reader/config/portlink';

// Fallback shown in the button when the env var is not set
const PORTFOLIO_FALLBACK = 'https://myportfolio.com';

// ── State ─────────────────────────────────────────────────────────────────────

let loaded = false;   // guard against duplicate fetches

// ── Helpers for fetching Link from API ─────────────────────────────────────────
const API_BASE = '';   // Same origin — FastAPI serves frontend + API

/** Low-level fetch wrapper. Throws with a message on non-2xx. */
async function _request(method, path, body = null, isMultipart = false) {
  const opts = { method, headers: {} };

  if (body && !isMultipart) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body && isMultipart) {
    // Let the browser set Content-Type with the correct boundary
    opts.body = body;
  }

  const resp = await fetch(API_BASE + path, opts);

  // 202 Accepted is a valid "not yet ready" response — return the body
  if (resp.status === 202) {
    const data = await resp.json().catch(() => ({}));
    return { _status: 202, ...data };
  }

  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      msg = err.detail?.message || err.message || err.detail || msg;
      // Handle detail being an object (FastAPI 422 validation errors)
      if (typeof msg === 'object') msg = JSON.stringify(msg);
    } catch (_) {}
    throw new Error(msg);
  }

  // CSV download returns plain text
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('text/csv')) {
    return { _csv: await resp.text() };
  }

  return resp.json();
}



// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Show an inline error inside the panel if the fetch fails.
 */
function renderError(panel, message) {
  panel.innerHTML = `
    <div class="container" style="padding-top: 64px;">
      <div class="error-box">
        ⚠️ Could not load About page: ${message}
      </div>
    </div>`;
}

/**
 * Fetch runtime config from the backend.
 * Returns an object with keys like { portfolio_url }.
 * Never throws — falls back to safe defaults so the page still loads.
 */
async function fetchConfig() {
  try {
    // const res = await fetch(CONFIG_ENDPOINT);
    // if (!res.ok) throw new Error(`HTTP ${res.status}`);
    // return await res.json();
    return _request('GET', CONFIG_ENDPOINT)
  } catch (err) {
    console.warn('[about.js] Could not fetch /api/config, using defaults.', err);
    return {};
  }
}

/**
 * Wire up any dynamic links inside the injected about.html.
 * Currently handles the portfolio button; extend here for future env-driven URLs.
 *
 * @param {HTMLElement} panel  — the #panel-about element (already has HTML injected)
 * @param {Object}      config — parsed JSON from /api/config
 */
function applyConfig(panel, config) {
  const portfolioUrl = config.portfolio_url || PORTFOLIO_FALLBACK;

  // Find the portfolio anchor by its data attribute (set in about.html)
  const portfolioBtn = panel.querySelector('a[data-social="portfolio"]');
  if (portfolioBtn) {
    portfolioBtn.href = portfolioUrl;

    // If no URL is configured, dim the button so the user knows it's inactive
    if (portfolioUrl === PORTFOLIO_FALLBACK) {
      portfolioBtn.style.opacity = '0.45';
      portfolioBtn.title = 'Portfolio URL not configured yet';
    }
  } else {
    console.warn('[about.js] Portfolio button (a[data-social="portfolio"]) not found in about.html.');
  }
}

/**
 * Fetch about.html, inject it into the panel, then apply runtime config.
 */
async function loadAboutContent() {
  const panel = document.getElementById(PANEL_ID);
  if (!panel) return;
  if (loaded)  return;

  // Lightweight loading indicator
  panel.innerHTML = `
    <div class="container" style="padding-top: 80px; text-align: center;">
      <div class="spinner" style="margin: 0 auto 16px;"></div>
      <p class="muted" style="font-family: var(--font-mono); font-size: 0.8rem;">
        Loading…
      </p>
    </div>`;

  try {
    // Run both fetches in parallel — no reason to serialise them
    const [htmlRes, config] = await Promise.all([
      fetch(ABOUT_HTML),
      fetchConfig(),
    ]);

    if (!htmlRes.ok) {
      throw new Error(`HTTP ${htmlRes.status} – ${htmlRes.statusText}`);
    }

    panel.innerHTML = await htmlRes.text();
    applyConfig(panel, config);
    loaded = true;

  } catch (err) {
    console.error('[about.js] Failed to load about.html:', err);
    renderError(panel, err.message);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * init() — called once from app.js during DOMContentLoaded.
 *
 * Strategy: listen for the "about" tab being activated and lazily
 * fetch the content the first time it becomes visible.  This keeps
 * the initial page load lean — about.html is only downloaded when
 * the user actually navigates to it.
 */
export function init() {
  // Observe nav-tab clicks forwarded through app.js's switchTab calls.
  // app.js toggles the `active` class on panels, so we watch for that.
  const panel = document.getElementById(PANEL_ID);
  if (!panel) {
    console.warn('[about.js] #panel-about not found in DOM.');
    return;
  }

  // MutationObserver fires whenever app.js adds/removes the `active` class.
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (
        mutation.type === 'attributes' &&
        mutation.attributeName === 'class' &&
        panel.classList.contains('active')
      ) {
        loadAboutContent();
        break;
      }
    }
  });

  observer.observe(panel, { attributes: true });

  // Also handle the case where 'about' is the restored tab on page load
  // (switchTab fires before this observer is registered, so check immediately).
  if (panel.classList.contains('active')) {
    loadAboutContent();
  }
}



// ───────────────────────────── OLD CODE: ─────────────────────────────────────────
// // ── Constants ─────────────────────────────────────────────────────────────────

// const PANEL_ID   = 'panel-about';
// const ABOUT_HTML = '/static/css/about.html';
// const CONFIG_ENDPOINT = '/reader/portlink';

// // ── State ─────────────────────────────────────────────────────────────────────

// let loaded = false;   // guard against duplicate fetches

// // ── Helpers ───────────────────────────────────────────────────────────────────

// /**
//  * Show an inline error inside the panel if the fetch fails.
//  */
// function renderError(panel, message) {
//   panel.innerHTML = `
//     <div class="container" style="padding-top: 64px;">
//       <div class="error-box">
//         ⚠️ Could not load About page: ${message}
//       </div>
//     </div>`;
// }

// /**
//  * Fetch about.html and inject it into the panel.
//  * Uses a simple loading placeholder while the request is in-flight.
//  */
// async function loadAboutContent() {
//   const panel = document.getElementById(PANEL_ID);
//   if (!panel) return;                     // panel not in DOM — nothing to do
//   if (loaded)  return;                    // already fetched — skip

//   // Lightweight loading indicator
//   panel.innerHTML = `
//     <div class="container" style="padding-top: 80px; text-align: center;">
//       <div class="spinner" style="margin: 0 auto 16px;"></div>
//       <p class="muted" style="font-family: var(--font-mono); font-size: 0.8rem;">
//         Loading…
//       </p>
//     </div>`;

//   try {
//     const response = await fetch(ABOUT_HTML);
//     if (!response.ok) {
//       throw new Error(`HTTP ${response.status} – ${response.statusText}`);
//     }
//     const html = await response.text();
//     panel.innerHTML = html;
//     loaded = true;
//   } catch (err) {
//     console.error('[about.js] Failed to load about.html:', err);
//     renderError(panel, err.message);
//   }
// }

// // ── Public API ────────────────────────────────────────────────────────────────

// /**
//  * init() — called once from app.js during DOMContentLoaded.
//  *
//  * Strategy: listen for the "about" tab being activated and lazily
//  * fetch the content the first time it becomes visible.  This keeps
//  * the initial page load lean — about.html is only downloaded when
//  * the user actually navigates to it.
//  */
// export function init() {
//   // Observe nav-tab clicks forwarded through app.js's switchTab calls.
//   // app.js toggles the `active` class on panels, so we watch for that.
//   const panel = document.getElementById(PANEL_ID);
//   if (!panel) {
//     console.warn('[about.js] #panel-about not found in DOM.');
//     return;
//   }

//   // MutationObserver fires whenever app.js adds/removes the `active` class.
//   const observer = new MutationObserver((mutations) => {
//     for (const mutation of mutations) {
//       if (
//         mutation.type === 'attributes' &&
//         mutation.attributeName === 'class' &&
//         panel.classList.contains('active')
//       ) {
//         loadAboutContent();
//         break;
//       }
//     }
//   });

//   observer.observe(panel, { attributes: true });

//   // Also handle the case where 'about' is the restored tab on page load
//   // (switchTab fires before this observer is registered, so check immediately).
//   if (panel.classList.contains('active')) {
//     loadAboutContent();
//   }
// }
