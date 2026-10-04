/* Shared behaviour: presentation mode, collapsing menu, sortable tables, tabs. No dependencies. */
(function () {
  "use strict";
  var body = document.body;
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  // presentation mode (larger fonts and charts)
  function setPres(on) {
    body.classList.toggle("pres", on);
    document.documentElement.classList.toggle("pres", on);
    document.querySelectorAll(".pres-toggle").forEach(function (b) { b.setAttribute("aria-pressed", on ? "true" : "false"); b.textContent = on ? "Presentation mode: ON" : "Presentation mode"; });
    store("pres", on ? "1" : "0");
  }
  document.querySelectorAll(".pres-toggle").forEach(function (b) { b.addEventListener("click", function () { setPres(!body.classList.contains("pres")); }); });
  if (store("pres") === "1") setPres(true);

  // menu on narrow screens
  var menu = document.getElementById("menu"), side = document.getElementById("sidebar");
  if (menu) menu.addEventListener("click", function () { var o = side.classList.toggle("open"); menu.setAttribute("aria-expanded", o ? "true" : "false"); });

  // sortable tables: click a header; numbers sort numerically (data-v), text alphabetically
  document.querySelectorAll("table.sortable").forEach(function (t) {
    var tb = t.tBodies[0];
    t.tHead.querySelectorAll("th").forEach(function (th, i) {
      th.addEventListener("click", function () {
        var asc = !th.classList.contains("asc");
        t.tHead.querySelectorAll("th").forEach(function (x) { x.classList.remove("asc", "desc"); });
        th.classList.add(asc ? "asc" : "desc");
        var rows = Array.prototype.slice.call(tb.rows);
        rows.sort(function (a, b) {
          var ca = a.cells[i], cb = b.cells[i], va = ca.dataset.v, vb = cb.dataset.v;
          var r = (va !== undefined && vb !== undefined) ? (parseFloat(va) - parseFloat(vb)) : ca.textContent.localeCompare(cb.textContent, undefined, { numeric: true });
          return asc ? r : -r;
        });
        rows.forEach(function (r) { tb.appendChild(r); });
      });
    });
  });

  // generic tabs: <div data-tabs> <div class="tabs"><button data-tab="id">..</button></div> <div class="tabpanel" id=id>
  document.querySelectorAll("[data-tabs]").forEach(function (root) {
    var buttons = root.querySelectorAll(".tabs button"), panels = root.querySelectorAll(".tabpanel");
    function show(id) {
      buttons.forEach(function (b) { b.setAttribute("aria-selected", b.dataset.tab === id ? "true" : "false"); });
      panels.forEach(function (p) { p.hidden = p.id !== id; });
    }
    buttons.forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.tab); }); });
    if (buttons.length) show(buttons[0].dataset.tab);
  });
})();
