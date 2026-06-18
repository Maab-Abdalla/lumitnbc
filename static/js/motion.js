/* LumiTNBC motion: on-scroll reveal.
   Lightweight, no dependencies, respects reduced-motion. */
(function () {
  var reduce = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.addEventListener('DOMContentLoaded', function () {
    // ---- On-scroll reveal ----
    var revealEls = document.querySelectorAll('.reveal');
    if (reduce || !('IntersectionObserver' in window)) {
      revealEls.forEach(function (el) { el.classList.add('is-visible'); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.classList.add('is-visible');
            io.unobserve(e.target);
          }
        });
      }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
      revealEls.forEach(function (el) { io.observe(el); });
    }

    // ---- Count-up for stat numbers (suggestion #4) ----
    initCountUps(reduce);
  });

  function initCountUps(reduce) {
    var nums = document.querySelectorAll('[data-countup]');
    if (!nums.length) return;
    nums.forEach(function (el) {
      var target = parseFloat(el.getAttribute('data-countup'));
      if (isNaN(target)) return;
      var suffix = el.getAttribute('data-suffix') || '';
      var decimals = (el.getAttribute('data-decimals') || '0') | 0;
      if (reduce) { el.textContent = target.toFixed(decimals) + suffix; return; }
      var run = function () {
        var dur = 900, start = null;
        function step(ts) {
          if (!start) start = ts;
          var p = Math.min((ts - start) / dur, 1);
          var eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
          el.textContent = (target * eased).toFixed(decimals) + suffix;
          if (p < 1) requestAnimationFrame(step);
          else el.textContent = target.toFixed(decimals) + suffix;
        }
        requestAnimationFrame(step);
      };
      if ('IntersectionObserver' in window) {
        var io2 = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (e.isIntersecting) { run(); io2.unobserve(e.target); }
          });
        }, { threshold: 0.5 });
        io2.observe(el);
      } else { run(); }
    });
  }
})();
