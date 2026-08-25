/**
 * app.js — BookHead AI main entry point
 *
 * Responsibilities:
 *   - Tab switching between Reader, Librarian & About panels
 *   - Initialising panel controllers
 *   - Any global event listeners
 *
 * Deliberately thin — all real logic lives in reader.js and librarian.js.
 */

import { init as initReader }    from './reader.js';
import { init as initLibrarian } from './librarian.js';
import { init as initAbout }     from './about.js';


// ── Tab routing ───────────────────────────────────────────────────────────────

function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });

  // Show/hide panels
  document.querySelectorAll('.panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${tabName}`);
  });

  // Persist selection
  sessionStorage.setItem('booklens-tab', tabName);
}

function initStartupBanner() {
  const banner = document.getElementById('startup-banner');
  if (!banner) return;

  const dismissed = window.localStorage.getItem('bookhead-startup-banner-dismissed');
  if (dismissed) {
    banner.classList.add('hidden');
    return;
  }

  const closeBanner = () => {
    banner.classList.add('hidden');
    try {
      window.localStorage.setItem('bookhead-startup-banner-dismissed', '1');
    } catch (err) {
      // Ignore storage errors in private mode.
    }
  };

  banner.addEventListener('click', event => {
    const target = event.target;
    if (target.closest('#startup-banner-close') || target.closest('.startup-banner-close-zone')) {
      closeBanner();
    }
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !banner.classList.contains('hidden')) {
      closeBanner();
    }
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Wire tab buttons
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Restore last-used tab, default to 'reader'
  const savedTab = sessionStorage.getItem('booklens-tab') || 'reader';
  switchTab(savedTab);

  // Initialise panel controllers
  initReader();
  initLibrarian();
  initAbout();
  initStartupBanner();
});
