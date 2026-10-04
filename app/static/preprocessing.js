/* Preprocessing stage stepper: a passage before/after the chosen stage, plus the word clouds of both stages. */
(function () {
  "use strict";
  var root = document.getElementById("stepper");
  if (!root) return;
  var stages = JSON.parse(root.dataset.stages || "[]");
  var buttons = document.getElementById("stage-buttons"), sample = document.getElementById("sample");
  var current = 1;
  var $ = function (id) { return document.getElementById(id); };
  var fmt = function (n) { return n == null ? "-" : Number(n).toLocaleString("en-US"); };

  function name(s) { var r = stages.find(function (x) { return x.stage === s; }); return r ? r.name.replace(/_/g, " ") : "stage " + s; }
  function info(s) { return stages.find(function (x) { return x.stage === s; }); }

  stages.forEach(function (st) {
    var b = document.createElement("button");
    b.type = "button"; b.textContent = st.stage + "  " + st.name.replace(/_/g, " "); b.dataset.stage = st.stage; b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", function () { current = st.stage; show(); });
    buttons.appendChild(b);
  });

  function show() {
    Array.prototype.forEach.call(buttons.children, function (b) { b.setAttribute("aria-pressed", Number(b.dataset.stage) === current ? "true" : "false"); });
    var st = info(current), prev = info(current - 1);
    var delta = (st && prev && st.unit === prev.unit) ? " (" + (st.n_tokens - prev.n_tokens >= 0 ? "+" : "") + fmt(st.n_tokens - prev.n_tokens) + " " + st.unit + ")" : "";
    $("stage-desc").textContent = st ? "Stage " + st.stage + " - " + st.name.replace(/_/g, " ") + ": " + st.description + ". " + fmt(st.n_tokens) + " " + st.unit + ", vocabulary " + fmt(st.vocab_size) + delta + "." : "";
    $("before-title").textContent = current === 0 ? "Before: (nothing - this is the raw text)" : "Before: stage " + (current - 1) + " (" + name(current - 1) + ")";
    $("after-title").textContent = "After: stage " + current + " (" + name(current) + ")";
    fetch("/api/preprocessing/sample?sample=" + encodeURIComponent(sample.value) + "&stage=" + current)
      .then(function (r) { return r.json(); })
      .then(function (j) { $("before").textContent = j.before == null ? "" : j.before; $("after").textContent = j.after == null ? "(file not available)" : j.after; })
      .catch(function () { $("after").textContent = "(could not load)"; });
    var prevStage = Math.max(0, current - 1);
    $("cloud-before-title").textContent = "Word cloud, stage " + prevStage;
    $("cloud-after-title").textContent = "Word cloud, stage " + current;
    $("cloud-before").src = "/api/preprocessing/cloud/" + prevStage + ".png";
    $("cloud-after").src = "/api/preprocessing/cloud/" + current + ".png";
  }
  sample.addEventListener("change", show);
  show();
})();
