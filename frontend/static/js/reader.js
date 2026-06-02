/**
 * reader.js — Reader panel controller
 *
 * Manages the full reader flow:
 *   1. File selection via drop zone
 *   2. Preferences form collection
 *   3. Job submission → polling → results rendering
 *
 * Keeps all reader-specific state here so app.js stays thin.
 */

import * as API from './api.js';
import * as UI  from './ui.js';

// ── State ─────────────────────────────────────────────────────────────────────
let _files   = [];      // Selected File objects
let _jobId   = null;    // Active job ID
let _polling = false;   // Whether a poll loop is running

// ── DOM refs (populated in init()) ───────────────────────────────────────────
let dropZone, fileInput, fileList, statusBar,
    submitBtn, resultsSection, recsGrid, allBooksSection;

// ── Public init ───────────────────────────────────────────────────────────────

/** Bind DOM elements and event listeners for the reader panel. */
export function init() {
  dropZone       = document.getElementById('reader-drop');
  fileInput      = document.getElementById('reader-file-input');
  fileList       = document.getElementById('reader-file-list');
  statusBar      = document.getElementById('reader-status');
  submitBtn      = document.getElementById('reader-submit');
  resultsSection = document.getElementById('reader-results');
  recsGrid       = document.getElementById('recs-grid');
  allBooksSection= document.getElementById('all-books-section');

  if (!dropZone) return;   // Panel not in DOM yet

  // Drop zone drag-and-drop
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    _addFiles(e.dataTransfer.files);
  });

  // File input change
  fileInput.addEventListener('change', () => {
    _addFiles(fileInput.files);
    fileInput.value = '';   // Reset so same file can be re-selected
  });

  // Submit
  submitBtn.addEventListener('click', _handleSubmit);
}

// ── Private handlers ──────────────────────────────────────────────────────────

function _addFiles(fileList_) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'];
  for (const f of fileList_) {
    if (!allowed.includes(f.type)) {
      UI.toast(`Unsupported file type: ${f.name}`, 'error');
      continue;
    }
    if (f.size > 20 * 1024 * 1024) {
      UI.toast(`${f.name} is too large (max 20 MB)`, 'error');
      continue;
    }
    _files.push(f);
  }
  _renderFileList();
  submitBtn.disabled = _files.length === 0;
}

function _removeFile(idx) {
  _files.splice(idx, 1);
  _renderFileList();
  submitBtn.disabled = _files.length === 0;
}

function _renderFileList() {
  UI.renderFileList(fileList, _files, _removeFile);
}

async function _handleSubmit() {
  if (_files.length === 0) {
    UI.toast('Please select at least one shelf image.', 'error');
    return;
  }
  if (_polling) return;

  // Collect preferences
  const prefs = {
    preferred_genres: document.getElementById('r-genres').value
      .split(',').map(s => s.trim()).filter(Boolean),
    mood:                document.getElementById('r-mood').value,
    preferred_length:    document.getElementById('r-length').value,
    max_recommendations: parseInt(document.getElementById('r-max-recs').value) || 5,
  };

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<div class="spinner"></div> Submitting…';
  statusBar.style.display = 'flex';
  resultsSection.innerHTML = '';

  try {
    const resp = await API.analyzeShelf(_files, prefs);
    _jobId   = resp.job_id;
    _polling = true;
    UI.updateStatusBar(statusBar, 'PENDING', 'Job queued — detecting books…');
    await _pollUntilDone();
  } catch (err) {
    UI.updateStatusBar(statusBar, 'FAILURE', err.message);
    resultsSection.innerHTML = UI.errorBox(err.message);
    UI.toast(err.message, 'error');
  } finally {
    _polling = false;
    submitBtn.disabled = false;
    submitBtn.innerHTML = '🔍 Analyse Shelf';
  }
}

async function _pollUntilDone() {
  const final = await API.poll(
    () => API.getReaderJob(_jobId),
    status => {
      const msg = status.progress_message || _stateMessage(status.state);
      UI.updateStatusBar(statusBar, status.state, msg );
    },
  );

  if (final.state === 'FAILURE') {
    const msg = final.error || 'Pipeline failed.';
    UI.updateStatusBar(statusBar, 'FAILURE', msg);
    resultsSection.innerHTML = UI.errorBox(msg);
    return;
  }

  if (final.state === 'SUCCESS' && final.result) {
    UI.updateStatusBar(statusBar, 'SUCCESS', 'Done! Here are your recommendations.');
    _renderResults(final.result);
    UI.toast('Recommendations ready!', 'success');
  }
}

function _renderResults(result) {
  const recs      = result.recommendations || [];
  const allBooks  = result.all_books || [];
  const total     = result.total_books_detected || 0;

  if (recs.length === 0 && total === 0) {
    resultsSection.innerHTML = UI.emptyState('No books detected. Try a clearer shelf photo. Follow picture instructions');
    return;
  }

  // ── Recommendations ────────────────────────────────────────────────────
  let html = `
    <div class="results-header">
      <h2>Your Recommendations</h2>
      <span class="results-count">${total} book${total !== 1 ? 's' : ''} detected</span>
    </div>
    <div class="recs-grid" id="recs-grid">
  `;

  if (recs.length === 0) {
    html += UI.emptyState('No recommendations matched your preferences.');
  } else {
    recs.forEach((rec, i) => {
      // Build image URL from crop_image_path
      const cropUrl = rec.crop_image_path ? `/uploads/${rec.crop_image_path}` : null;
      html += UI.renderRecCard({ ...rec, crop_image_url: cropUrl }, i * 80);
    });
  }

  html += '</div>';

  // ── All detected books (collapsed section) ─────────────────────────────
  if (allBooks.length > recs.length) {
    const others = allBooks.filter(b => !recs.some(r => r.book_id === b.book_id));
    if (others.length > 0) {
      html += `
        <details style="margin-top:24px">
          <summary style="cursor:pointer;font-family:var(--font-mono);font-size:0.78rem;
                          text-transform:uppercase;letter-spacing:0.1em;color:var(--mist);
                          padding:12px 0;border-top:1px solid var(--ink-mid)">
            All detected books (${others.length} more)
          </summary>
          <div style="margin-top:16px">${UI.renderCatalogTable(others)}</div>
        </details>
      `;
    }
  }

  resultsSection.innerHTML = html;
}

function _stateMessage(state) {
  const map = {
    PENDING:  'Queued — waiting for a worker…',
    STARTED:  'Running detection and OCR…',
    SUCCESS:  'Complete!',
    FAILURE:  'Something went wrong.',
  };
  return map[state] || state;
}
