/* LumiTNBC shared UI helpers: toasts + dark-mode theme toggle. */
(function () {

  // ---- Toasts (suggestion #2) ----
  var ICONS = {
    success: '<path d="M20 6 9 17l-5-5"></path>',
    error:   '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>',
    info:    '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line>'
  };

  function ensureContainer() {
    var c = document.getElementById('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      document.body.appendChild(c);
    }
    return c;
  }

  window.showToast = function (message, type, duration) {
    type = type || 'info';
    duration = duration || 3500;
    var c = ensureContainer();
    var t = document.createElement('div');
    t.className = 'toast ' + type;
    t.setAttribute('role', 'status');
    t.innerHTML =
      '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      (ICONS[type] || ICONS.info) + '</svg><div>' + message + '</div>';
    c.appendChild(t);
    var remove = function () {
      t.classList.add('toast-out');
      setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 320);
    };
    var timer = setTimeout(remove, duration);
    t.addEventListener('click', function () { clearTimeout(timer); remove(); });
  };

  // ---- Dark mode (suggestion #6) ----
  // Note: theme is applied early via an inline script in <head> to avoid flash.
  window.toggleTheme = function () {
    var root = document.documentElement;
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { document.cookie = 'theme=' + next + ';path=/;max-age=31536000;samesite=lax'; } catch (e) {}
    updateThemeIcon(next);
  };

  function updateThemeIcon(theme) {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var sun = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
    var moon = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
    btn.querySelector('svg').innerHTML = (theme === 'dark') ? sun : moon;
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }

  document.addEventListener('DOMContentLoaded', function () {
    updateThemeIcon(document.documentElement.getAttribute('data-theme') || 'light');
  });
})();
