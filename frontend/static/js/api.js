/**
 * api.js — BookLens AI API client
 *
 * Wraps every backend endpoint in a named async function.
 * All functions return plain JS objects (parsed JSON) or throw an Error
 * with a human-readable message on HTTP failure.
 *
 * DESIGN:
 *   - No external dependencies — plain fetch() throughout
 *   - Each function validates the HTTP status before returning
 *   - A single _request() helper handles error surfacing consistently
 *   - poll() implements the polling loop with configurable interval/timeout
 */

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

// ── Reader endpoints ─────────────────────────────────────────────────────────

/**
 * Upload shelf images and queue a reader recommendation job.
 * @param {FileList|File[]} files
 * @param {Object} prefs  { preferred_genres, mood, preferred_length, max_recommendations }
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function analyzeShelf(files, prefs = {}) {
  const fd = new FormData();
  for (const file of files) {
    fd.append('images', file, file.name);
  }
  fd.append('preferred_genres', (prefs.preferred_genres || []).join(','));
  fd.append('mood', prefs.mood || '');
  fd.append('preferred_length', prefs.preferred_length || 'any');
  fd.append('max_recommendations', String(prefs.max_recommendations || 5));
  return _request('POST', '/reader/analyze', fd, true);
}

/**
 * Poll a reader job for its current state.
 * @param {string} jobId
 * @returns {Promise<{job_id, state, result?, error?, progress_message?}>}
 */
export async function getReaderJob(jobId) {
  return _request('GET', `/reader/${jobId}`);
}

// ── Librarian endpoints ───────────────────────────────────────────────────────

/**
 * Upload shelf images and queue a catalog generation job.
 * @param {FileList|File[]} files
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function createCatalog(files) {
  const fd = new FormData();
  for (const file of files) {
    fd.append('images', file, file.name);
  }
  return _request('POST', '/librarian/catalog', fd, true);
}

/**
 * Poll a librarian catalog job.
 * @param {string} jobId
 * @returns {Promise<{job_id, state, result?, error?, progress_message?}>}
 */
export async function getCatalogJob(jobId) {
  return _request('GET', `/librarian/catalog/${jobId}`);
}

/**
 * Fetch books flagged for human review.
 * @param {string} jobId
 * @returns {Promise<{job_id, total_flagged, items: ReviewItem[]}>}
 */
export async function getReviewQueue(jobId) {
  return _request('GET', `/librarian/review/${jobId}`);
}

/**
 * Submit corrections for all flagged books and resume the pipeline.
 * @param {string} jobId
 * @param {Array<{book_id, corrected_title, corrected_author}>} corrections
 * @returns {Promise<{job_id, state, progress_message}>}
 */
export async function submitCorrections(jobId, corrections) {
  return _request('POST', `/librarian/review/${jobId}`, { corrections });
}

/**
 * Download the completed catalog as a CSV string.
 * @param {string} jobId
 * @returns {Promise<{_csv: string}>}
 */
export async function downloadCatalogCSV(jobId) {
  return _request('GET', `/librarian/download/${jobId}`);
}

// ── Polling utility ──────────────────────────────────────────────────────────

/**
 * Poll a job endpoint until it reaches a terminal state (SUCCESS/FAILURE)
 * or a special state handled by the caller (AWAITING_REVIEW).
 *
 * @param {Function} fetchFn     Async function that returns a job status object.
 * @param {Function} onProgress  Called on every poll with the current status object 
 *                               (JobStatusResponse object) returned from get_reader_job GET endpoint.
 * @param {number}   intervalMs  Polling interval in milliseconds (default 2000).
 * @param {number}   timeoutMs   Max total wait time in milliseconds (default 300000 = 5 min/600000 = 10 min).
 * @returns {Promise<Object>}    The final job status object.
 */
export async function poll(fetchFn, onProgress, intervalMs = 5000, timeoutMs = 600_000) {
  const deadline = Date.now() + timeoutMs;
  const TERMINAL = new Set(['SUCCESS', 'FAILURE', 'AWAITING_REVIEW']);

  while (Date.now() < deadline) {
    const status = await fetchFn();
    onProgress(status);

    if (TERMINAL.has(status.state)) {
      return status;
    }

    await _sleep(intervalMs);
  }

  throw new Error('Job timed out after 4 minutes. Please try again.');
}

/** Promise-based sleep. */
function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
