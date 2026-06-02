/**
 * ui.js — BookLens AI UI utilities
 *
 * Pure UI helpers with no business logic. Every function takes plain data
 * and produces DOM mutations or HTML strings. Nothing in this file knows
 * about the API or application state.
 */

// ── Toast notifications ───────────────────────────────────────────────────────

const _toastContainer = document.getElementById('toast-container');

/**
 * Show a temporary toast notification.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {number} durationMs
 */
export function toast(message, type = 'info', durationMs = 4000) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  _toastContainer.appendChild(el);
  setTimeout(() => el.remove(), durationMs);
}

// ── Status bar ────────────────────────────────────────────────────────────────

/**
 * Update the status bar element with the current job state.
 * @param {HTMLElement} barEl   The .status-bar element.
 * @param {string} state        Job state string.
 * @param {string} message      Human-readable progress message.
 */
export function updateStatusBar(barEl, state, message) {
  if (!barEl) return;
  barEl.style.display = 'flex';

  const dotMap = {
    PENDING:         'pending',
    STARTED:         'started',
    AWAITING_REVIEW: 'awaiting',
    SUCCESS:         'success',
    FAILURE:         'failure',
  };

  const dotClass = dotMap[state] || 'pending';
  barEl.innerHTML = `
    <div class="status-dot ${dotClass}"></div>
    <span>${escHtml(message || state)}</span>
    ${['PENDING','STARTED'].includes(state) ? '<div class="spinner" style="margin-left:auto"></div>' : ''}
  `;
}

// ── File chip list ────────────────────────────────────────────────────────────

/**
 * Render the list of selected files as chips below the drop zone.
 * @param {HTMLElement} listEl   The .file-list container.
 * @param {File[]}      files
 * @param {Function}    onRemove Called with the file index when × is clicked.
 */
export function renderFileList(listEl, files, onRemove) {
  listEl.innerHTML = '';
  files.forEach((f, i) => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.innerHTML = `
      <span>📷 ${escHtml(f.name)}</span>
      <span class="remove" title="Remove" data-idx="${i}">✕</span>
    `;
    chip.querySelector('.remove').addEventListener('click', () => onRemove(i));
    listEl.appendChild(chip);
  });
}

// ── Recommendation card ───────────────────────────────────────────────────────

/**
 * Render a single BookRecommendation as an HTML card string.     # NOTE: Don't push actual crops. Get some dummy pictures send those.
 * @param {Object} rec   BookRecommendation object from the API.
 * @param {number} delay Animation delay in ms.
 * @returns {string}     HTML string for the card.
 */
export function renderRecCard(rec, delay = 0) {
  const imageHtml = rec.crop_image_url
    ? `<img src="${escHtml(rec.crop_image_url)}" alt="${escHtml(rec.title)}" loading="lazy">`
    : `<div class="rec-card-image" style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--mist);font-size:2rem;">📚</div>`;

  return `
    <div class="rec-card" style="animation-delay:${delay}ms">
      <div class="rec-card-image">${imageHtml}</div>
      <div class="rec-card-body">
        <div class="rec-rank">#${rec.rank} recommendation</div>
        <div class="rec-title">${escHtml(rec.title)}</div>
        <div class="rec-author">${escHtml(rec.author || 'Unknown Author')}</div>
        <div class="rec-summary">${escHtml(rec.summary || '')}</div>
        ${rec.match_reason ? `<div class="rec-match">${escHtml(rec.match_reason)}</div>` : ''}
      </div>
    </div>
  `;
}

// ── Review card ───────────────────────────────────────────────────────────────

/**
 * Render a single flagged-book review card.
 * @param {Object} item   ReviewItem from the API.
 * @param {number} idx    Index in the list.
 * @returns {string}      HTML string.
 */
export function renderReviewCard(item, idx) {
  const confPct = Math.round(item.confidence * 100);
  const confClass = item.confidence < 0.5 ? 'low' : '';

  const imageHtml = item.crop_image_url
    ? `<img src="${escHtml(item.crop_image_url)}" alt="Book spine" loading="lazy">`
    : `<div style="display:flex;align-items:center;justify-content:center;height:120px;color:var(--mist);font-size:2rem;">📖</div>`;

  return `
    <div class="review-card" style="animation-delay:${idx * 60}ms" data-book-id="${escHtml(item.book_id)}">
      <div class="review-card-image">${imageHtml}</div>
      <div class="review-card-body">
        <span class="confidence-badge low">
          OCR confidence: ${confPct}%
        </span>
        <div class="conf-bar-wrap" style="margin-bottom:14px">
          <div class="conf-bar">
            <div class="conf-bar-fill ${confClass}" style="width:${confPct}%"></div>
          </div>
        </div>
        <div class="review-field">
          <label>Book title</label>
          <input
            type="text"
            class="review-title"
            value="${escHtml(item.ocr_title || '')}"
            placeholder="Enter correct title…"
          >
        </div>
        <div class="review-field">
          <label>Author</label>
          <input
            type="text"
            class="review-author"
            value="${escHtml(item.ocr_author || '')}"
            placeholder="Enter correct author…"
          >
        </div>
      </div>
    </div>
  `;
}

// ── Catalog table ─────────────────────────────────────────────────────────────

/**
 * Render the catalog as an HTML table.
 * @param {Object[]} books   Array of BookRecord objects. 
 * @returns {string}         HTML string for the full table.
 */
export function renderCatalogTable(books) {
  if (!books.length) return emptyState('No books in catalog.');

  const rows = books.map(b => {
    const sourceClass = b.source === 'human_corrected' ? 'human' : 'auto';
    const sourceLabel = b.source === 'human_corrected' ? 'Verified' : 'Auto';
    return `
      <tr>
        <td><strong>${escHtml(b.title || '—')}</strong></td>
        <td style="color:var(--mist);font-style:italic">${escHtml(b.author || '—')}</td>
        <td><span class="genre-badge">${escHtml(b.genre_code || '—')}</span></td>
        <td style="font-size:0.82rem;color:var(--mist)">${escHtml(b.genre || '—')}</td>
        <td style="font-family:var(--font-mono);font-size:0.8rem">${escHtml(String(b.year_published || '—'))}</td>
        <td style="font-family:var(--font-mono);font-size:0.75rem;color:var(--mist)">${escHtml(b.isbn || '—')}</td>
        <td><span class="source-badge ${sourceClass}">${sourceLabel}</span></td>
      </tr>
    `;
  }).join('');

  return `
    <div class="catalog-table-wrap">
      <table class="catalog-table">
        <thead>
          <tr>
            <th>Title</th>
            <th>Author</th>
            <th>Code</th>
            <th>Genre</th>
            <th>Year</th>
            <th>ISBN</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Generic helpers ───────────────────────────────────────────────────────────

/** Empty state placeholder HTML. */
export function emptyState(message = 'Nothing here yet.') {
  return `
    <div class="empty-state">
      <div class="empty-icon">📚</div>
      <p>${escHtml(message)}</p>
    </div>
  `;
}

/** Error box HTML. */
export function errorBox(message) {
  return `<div class="error-box">⚠ ${escHtml(message)}</div>`;
}

/** Escape HTML special characters to prevent XSS. */
export function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/** Trigger a CSV file download in the browser. */
export function downloadCSV(csvContent, filename) {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
