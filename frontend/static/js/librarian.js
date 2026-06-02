/**
 * librarian.js — Librarian panel controller
 *
 * Manages the full librarian flow:
 *   1. File selection
 *   2. Job submission → polling
 *   3. AWAITING_REVIEW → fetch flagged books → render review cards
 *   4. Collect corrections → submit → poll for catalog
 *   5. Render catalog table + CSV download button
 */

import * as API from './api.js';
import * as UI  from './ui.js';

// ── State ─────────────────────────────────────────────────────────────────────
let _files      = [];
let _jobId      = null;
let _polling    = false;
let _resumeJobId = null;   // Task ID for the resume task (job_id + "_resume")

// ── DOM refs ──────────────────────────────────────────────────────────────────
let dropZone, fileInput, fileList, statusBar,
    submitBtn, reviewSection, catalogSection;

// ── Public init ───────────────────────────────────────────────────────────────

export function init() {
  dropZone       = document.getElementById('lib-drop');
  fileInput      = document.getElementById('lib-file-input');
  fileList       = document.getElementById('lib-file-list');
  statusBar      = document.getElementById('lib-status');
  submitBtn      = document.getElementById('lib-submit');
  reviewSection  = document.getElementById('lib-review-section');
  catalogSection = document.getElementById('lib-catalog-section');

  if (!dropZone) return;

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

  fileInput.addEventListener('change', () => {
    _addFiles(fileInput.files);
    fileInput.value = '';
  });

  submitBtn.addEventListener('click', _handleSubmit);
}

// ── Private handlers ──────────────────────────────────────────────────────────

function _addFiles(fileList_) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'];
  for (const f of fileList_) {
    if (!allowed.includes(f.type)) {
      UI.toast(`Unsupported: ${f.name}`, 'error');
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

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<div class="spinner"></div> Submitting…';
  reviewSection.innerHTML  = '';
  catalogSection.innerHTML = '';
  statusBar.style.display  = 'flex';

  try {
    const resp = await API.createCatalog(_files);
    _jobId   = resp.job_id;
    _polling = true;
    UI.updateStatusBar(statusBar, 'PENDING', 'Job queued — processing images…');
    await _pollCatalogJob();
  } catch (err) {
    UI.updateStatusBar(statusBar, 'FAILURE', err.message);
    UI.toast(err.message, 'error');
  } finally {
    _polling = false;
    submitBtn.disabled = false;
    submitBtn.innerHTML = '📚 Generate Catalog';
  }
}

async function _pollCatalogJob() {
  const final = await API.poll(
    () => API.getCatalogJob(_jobId),
    status => {
      UI.updateStatusBar(statusBar, status.state,
        status.progress_message || _stateMessage(status.state));
    },
  );

  if (final.state === 'FAILURE') {
    UI.updateStatusBar(statusBar, 'FAILURE', final.error || 'Pipeline failed.');
    return;
  }

  if (final.state === 'AWAITING_REVIEW') {
    await _handleReviewState();
    return;
  }

  if (final.state === 'SUCCESS' && final.result) {
    UI.updateStatusBar(statusBar, 'SUCCESS', 'Catalog generated!');
    _renderCatalog(final.result);
    UI.toast('Catalog is ready!', 'success');
  }
}

// ── HITL review flow ──────────────────────────────────────────────────────────

async function _handleReviewState() {
  UI.updateStatusBar(statusBar, 'AWAITING_REVIEW',
    'Some books need your review before the catalog can be generated.');

  // Fetch the flagged books
  let queueResp;
  try {
    queueResp = await API.getReviewQueue(_jobId);
  } catch (err) {
    UI.updateStatusBar(statusBar, 'FAILURE', err.message);
    return;
  }

  const items = queueResp.items || [];

  // Render the review UI
  let html = `
    <div class="review-intro">
      <strong>⚠ ${items.length} book${items.length !== 1 ? 's' : ''} need review.</strong>
      The OCR confidence was too low to auto-accept these titles.
      Please verify or correct the text below, then click <em>Confirm All</em>.
    </div>
    <div class="review-grid" id="review-grid">
  `;
  items.forEach((item, i) => { html += UI.renderReviewCard(item, i); });
  html += `
    </div>
    <button class="btn btn-primary" id="confirm-corrections-btn">
      ✓ Confirm All &amp; Generate Catalog
    </button>
  `;

  reviewSection.innerHTML = html;

  // Wire up the confirm button
  document.getElementById('confirm-corrections-btn')
    .addEventListener('click', () => _handleConfirmCorrections(items));
}

async function _handleConfirmCorrections(originalItems) {
  const confirmBtn = document.getElementById('confirm-corrections-btn');
  confirmBtn.disabled = true;
  confirmBtn.innerHTML = '<div class="spinner"></div> Submitting…';

  // Collect corrections from the review card inputs
  const corrections = [];
  const cards = document.querySelectorAll('.review-card[data-book-id]');

  cards.forEach(card => {
    const bookId = card.dataset.bookId;
    const title  = card.querySelector('.review-title').value.trim();
    const author = card.querySelector('.review-author').value.trim();

    if (!title) {
      UI.toast('Please fill in a title for every book.', 'error');
      confirmBtn.disabled = false;
      confirmBtn.innerHTML = '✓ Confirm All & Generate Catalog';
      return;
    }

    corrections.push({ book_id: bookId, corrected_title: title, corrected_author: author });
  });

  if (corrections.length !== originalItems.length) return;

  try {
    await API.submitCorrections(_jobId, corrections);
    reviewSection.innerHTML = '';
    UI.updateStatusBar(statusBar, 'STARTED', 'Corrections submitted — generating catalog…');

    // Poll the original job_id; the resume task result also becomes queryable
    // but the job store maps it back to the original job_id
    await _pollAfterResume();
  } catch (err) {
    confirmBtn.disabled = false;
    confirmBtn.innerHTML = '✓ Confirm All & Generate Catalog';
    UI.toast(err.message, 'error');
  }
}

async function _pollAfterResume() {
  // After submitting corrections, the resume task runs. We poll the original
  // job_id; the job store will surface SUCCESS once the resume task completes.
  const resumeJobId = `${_jobId}_resume`;

  const final = await API.poll(
    async () => {
      // Try the resume task first, fall back to original job
      try {
        const s = await API.getCatalogJob(resumeJobId);
        if (s.state !== 'PENDING') return s;
      } catch (_) {}
      return API.getCatalogJob(_jobId);
    },
    status => {
      UI.updateStatusBar(statusBar, status.state,
        status.progress_message || _stateMessage(status.state));
    },
  );

  if (final.state === 'SUCCESS' && final.result) {
    UI.updateStatusBar(statusBar, 'SUCCESS', 'Catalog complete!');
    _renderCatalog(final.result);
    UI.toast('Catalog is ready!', 'success');
  } else if (final.state === 'FAILURE') {
    UI.updateStatusBar(statusBar, 'FAILURE', final.error || 'Catalog generation failed.');
  }
}

// ── Catalog rendering ─────────────────────────────────────────────────────────

function _renderCatalog(result) {
  const books = result.books || [];
  const total = result.total_books || 0;
  const autoCount  = result.auto_accepted_count || 0;
  const humanCount = result.human_corrected_count || 0;

  let html = `
    <div class="results-header">
      <h2>Library Catalog</h2>
      <span class="results-count">${total} book${total !== 1 ? 's' : ''}</span>
    </div>
    <div style="display:flex;gap:20px;margin-bottom:24px;flex-wrap:wrap">
      <div class="status-bar" style="flex:1;min-width:200px">
        <div class="status-dot success"></div>
        <span class="mono" style="font-size:0.8rem">${autoCount} auto-accepted</span>
      </div>
      <div class="status-bar" style="flex:1;min-width:200px">
        <div class="status-dot" style="background:var(--brass)"></div>
        <span class="mono" style="font-size:0.8rem">${humanCount} human-verified</span>
      </div>
    </div>
    <div class="catalog-controls">
      <span style="font-family:var(--font-mono);font-size:0.78rem;color:var(--mist)">
        ${total} entries
      </span>
      <button class="btn btn-secondary" id="download-csv-btn">
        ⬇ Download CSV
      </button>
    </div>
  `;

  html += UI.renderCatalogTable(books);
  catalogSection.innerHTML = html;

  // Wire up CSV download
  document.getElementById('download-csv-btn')
    .addEventListener('click', () => _downloadCSV());
}

async function _downloadCSV() {
  try {
    const resp = await API.downloadCatalogCSV(_jobId);
    if (resp._csv) {
      UI.downloadCSV(resp._csv, `catalog_${_jobId.slice(0,8)}.csv`);
      UI.toast('CSV downloaded!', 'success');
    }
  } catch (err) {
    UI.toast('Could not download CSV: ' + err.message, 'error');
  }
}

function _stateMessage(state) {
  const map = {
    PENDING:         'Queued — waiting for worker…',
    STARTED:         'Processing shelf images…',
    AWAITING_REVIEW: 'Review required.',
    SUCCESS:         'Catalog complete!',
    FAILURE:         'Something went wrong.',
  };
  return map[state] || state;
}
