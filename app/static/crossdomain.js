/* 2x2 heatmaps (trained on x tested on) for each architecture, built from tables/baseline_cross_eval.csv. */
(function () {
  "use strict";
  var el = document.getElementById("cross-data"), host = document.getElementById("heatmaps"), sel = document.getElementById("metric");
  if (!el || !host) return;
  var rows = JSON.parse(el.textContent);
  var domains = ["Shakespeare", "WikiText"];
  var NAMES = { top1: "Top-1 accuracy", top3: "Top-3 accuracy", top5: "Top-5 accuracy", ksr: "Keystroke savings rate", oov_word_rate: "Unknown target words" };

  function colour(v, lo, hi, invert) {              // light -> dark teal; invert: high value = bad
    var t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    if (invert) t = 1 - t;
    var r = Math.round(235 - 200 * t), g = Math.round(245 - 100 * t), b = Math.round(235 - 120 * t);
    return { bg: "rgb(" + r + "," + g + "," + b + ")", fg: t > 0.6 ? "#fff" : "#10201a" };
  }

  function draw() {
    var metric = sel.value, invert = metric === "oov_word_rate";
    var vals = rows.map(function (r) { return r[metric]; }).filter(function (v) { return v != null; });
    var lo = 0, hi = Math.max.apply(null, vals);
    host.innerHTML = "";
    ["KN trigram", "LSTM"].forEach(function (arch) {
      var box = document.createElement("div");
      var h = document.createElement("h3"); h.textContent = arch + " - " + NAMES[metric]; box.appendChild(h);
      var t = document.createElement("table"); t.className = "heat";
      var head = "<tr><th></th>" + domains.map(function (d) { return "<th>tested on<br>" + d + "</th>"; }).join("") + "</tr>";
      var body = domains.map(function (tr) {
        return "<tr><th>trained on<br>" + tr + "</th>" + domains.map(function (te) {
          var r = rows.find(function (x) { return x.architecture === arch && x.trained_on === tr && x.tested_on === te; });
          if (!r || r[metric] == null) return "<td>-</td>";
          var c = colour(r[metric], lo, hi, invert);
          return '<td style="background:' + c.bg + ";color:" + c.fg + '">' + (100 * r[metric]).toFixed(1) + " %<small>" + (tr === te ? "in-domain" : "cross-domain") + "</small></td>";
        }).join("") + "</tr>";
      }).join("");
      t.innerHTML = head + body; box.appendChild(t); host.appendChild(box);
    });
  }
  sel.addEventListener("change", draw);
  draw();
})();
