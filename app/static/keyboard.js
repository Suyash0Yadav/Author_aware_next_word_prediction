/* Keyboard page: phone keyboard (physical + on-screen), suggestion chips, prediction inspector, text inspector. No dependencies. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var editor = $("editor"), bars = $("bars"), panels = $("insp-panels"), osk = $("osk");
  if (!editor) return;
  var PUNCT = ".,;:!?";
  var S = { models: [], primary: null, secondary: null, compare: false, probs: false, rerank: false, keys: 0, base: 0, accepted: 0,
            seq: 0, timer: null, current: {}, fetchedBefore: null, fetchedAfter: null, autoSpaceAt: -1, shift: false };

  function post(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(function (r) { return r.json(); });
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  // ------------------------------------------------------------------ setup
  function fillSelect(sel, names, value) {
    sel.innerHTML = "";
    names.forEach(function (n) { var o = el("option", null, n.name); o.value = n.name; o.title = n.description; sel.appendChild(o); });
    if (value) sel.value = value;
  }
  function init() {
    fetch("/api/models").then(function (r) { return r.json(); }).then(function (info) {
      S.models = info.models; S.primary = info.default;
      var others = info.models.map(function (m) { return m.name; }).filter(function (n) { return n !== S.primary; });
      S.secondary = others.indexOf("WikiText LSTM") >= 0 ? "WikiText LSTM" : (others[0] || S.primary);
      ["model", "ti-model"].forEach(function (id) { fillSelect($(id), info.models, S.primary); });
      ["model2", "ti-model2"].forEach(function (id) { fillSelect($(id), info.models, S.secondary); });
      info.demo_prefixes.forEach(function (d) {
        var o = el("option", null, d.text + (d.note ? "   [" + d.note + "]" : "")); o.value = d.text; $("demo").appendChild(o);
      });
      schedule();
    });
    fetch("/api/passages").then(function (r) { return r.json(); }).then(function (j) {
      var groups = {};
      j.passages.forEach(function (p) {
        if (!groups[p.play]) { groups[p.play] = el("optgroup"); groups[p.play].label = p.play; $("ti-passage").appendChild(groups[p.play]); }
        var o = el("option", null, p.label + " (" + p.words + " words)"); o.value = p.id; groups[p.play].appendChild(o);
      });
    });
    buildOSK();
  }

  // ------------------------------------------------------------------ suggestions
  function schedule() { clearTimeout(S.timer); S.timer = setTimeout(fetchSuggestions, 50); }          // debounce ~50 ms

  function fetchSuggestions() {
    var caret = editor.selectionStart, before = editor.value.slice(0, caret), after = editor.value.slice(caret);
    var names = S.compare && S.secondary !== S.primary ? [S.primary, S.secondary] : [S.primary];
    if (!S.primary) return;
    var mySeq = ++S.seq;
    post("/api/suggest", { text: before, models: names, k: 10, rerank: S.rerank }).then(function (j) {
      if (mySeq !== S.seq || !j.results) return;                                                      // drop stale answers
      S.fetchedBefore = before; S.fetchedAfter = after; S.current = j.results;
      renderBars(names, j.results); renderInspector(names, j.results);
    }).catch(function () {});
  }

  function statusText(r) {
    return r.mode === "complete" ? "completing “" + r.partial + "…”" + (r.suggestions.length ? "" : " - no word starts like this")
                                 : (r.sentence_start ? "start of a sentence - next word" : "next word");
  }

  function renderBars(names, res) {
    bars.innerHTML = "";
    var ms = 0;
    names.forEach(function (name, row) {
      var r = res[name]; if (!r) return;
      ms = Math.max(ms, r.ms);
      var wrap = el("div", "sg-row");
      wrap.appendChild(el("div", "sg-label", name));
      var chips = el("div", "chips");
      for (var i = 0; i < 3; i++) {
        (function (i) {
          var s = r.suggestions[i], b = el("button", "chip"); b.type = "button"; b.tabIndex = -1;
          if (s) {
            b.textContent = s.word;
            if (S.probs) b.appendChild(el("span", "prob", (100 * s.prob).toFixed(s.prob < 0.1 ? 1 : 0) + " %"));
            if (row === 0) b.appendChild(el("span", "hint", (i === 0 ? "Tab · " : "") + "Alt+" + (i + 1)));
            b.addEventListener("mousedown", function (e) { e.preventDefault(); });
            b.addEventListener("click", function () { accept(name, i); });
          } else { b.textContent = "·"; b.disabled = true; }
          chips.appendChild(b);
        })(i);
      }
      wrap.appendChild(chips);
      wrap.appendChild(el("div", "sg-status", statusText(r)));
      bars.appendChild(wrap);
    });
    $("s-ms").textContent = ms ? ms.toFixed(1) : "—";
  }

  // prediction inspector: top-10 words with probabilities
  function renderInspector(names, res) {
    panels.innerHTML = "";
    names.forEach(function (name) {
      var r = res[name]; if (!r) return;
      var p = el("div", "insp-panel");
      p.appendChild(el("h3", null, name));
      var ctx = (r.context || []).join(" ");
      p.appendChild(el("div", "insp-ctx", "context: " + (ctx ? "“" + ctx + "”" : "(start of sentence)") + " · " + statusText(r)));
      var top = r.suggestions.slice(0, 10), max = top.length ? Math.max.apply(null, top.map(function (s) { return s.prob; })) : 1;
      if (!top.length) p.appendChild(el("div", "bar-empty", "No word of the vocabulary starts like this."));
      top.forEach(function (s, i) {
        var row = el("div", "bar-row" + (i < 3 ? " chip-" + (i + 1) : ""));
        row.title = "click to accept “" + s.word + "”";
        row.appendChild(el("div", "bar-label", s.word));
        var track = el("div", "bar-track"), fill = el("div", "bar-fill"); fill.style.width = (100 * s.prob / max).toFixed(1) + "%"; track.appendChild(fill); row.appendChild(track);
        row.appendChild(el("div", "bar-pct", (100 * s.prob).toFixed(s.prob < 0.1 ? 1 : 0) + " %"));
        row.addEventListener("click", function () { accept(name, i); });
        p.appendChild(row);
      });
      panels.appendChild(p);
    });
  }

  function accept(name, i) {
    var r = S.current[name]; if (!r || !r.suggestions[i]) return;
    var caret = editor.selectionStart;
    if (editor.value.slice(0, caret) !== S.fetchedBefore) return;                                    // stale suggestion
    var s = r.suggestions[i];
    editor.value = s.new_text + S.fetchedAfter;
    var pos = s.new_text.length;
    editor.setSelectionRange(pos, pos);
    S.keys += 1; S.accepted += 1; S.autoSpaceAt = pos;                                               // a tap = one key press (it adds the space too)
    editor.focus(); updateStats(); schedule();
  }

  // ------------------------------------------------------------------ physical keyboard
  editor.addEventListener("keydown", function (e) {
    var caret = editor.selectionStart, first = S.current[S.primary];
    if (e.key === "Tab" && !e.shiftKey && !e.ctrlKey && !e.altKey && first && first.suggestions.length) { e.preventDefault(); accept(S.primary, 0); return; }
    if (e.altKey && ["Digit1", "Digit2", "Digit3"].indexOf(e.code) >= 0) { e.preventDefault(); accept(S.primary, Number(e.code.slice(-1)) - 1); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (PUNCT.indexOf(e.key) >= 0 && S.autoSpaceAt === caret && editor.value[caret - 1] === " ") {      // no space before punctuation
      e.preventDefault();
      editor.setRangeText(e.key, caret - 1, caret, "end");
      S.keys += 1; S.autoSpaceAt = -1; afterEdit(); return;
    }
    if (e.key.length === 1 || ["Backspace", "Delete", "Enter"].indexOf(e.key) >= 0) S.keys += 1;
    S.autoSpaceAt = -1;
  });
  editor.addEventListener("input", afterEdit);
  editor.addEventListener("click", schedule);
  editor.addEventListener("keyup", function (e) { if (e.key.indexOf("Arrow") === 0 || e.key === "Home" || e.key === "End") schedule(); });
  editor.addEventListener("paste", function (e) { S.base += (e.clipboardData || window.clipboardData).getData("text").length; });

  function afterEdit() { if (editor.value === "") resetCounter(); updateStats(); schedule(); }

  // ------------------------------------------------------------------ on-screen keyboard
  var ROWS = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"], "qwertyuiop".split(""), "asdfghjkl'".split(""),
              ["SHIFT"].concat("zxcvbnm".split("")).concat(["BKSP"]), [",", "SPACE", ".", "?", "!"]];
  function buildOSK() {
    ROWS.forEach(function (row, ri) {
      var r = el("div", "osk-row");
      row.forEach(function (k) {
        var b = el("button", "key" + (ri === 0 ? " digit" : "") + (k === "SHIFT" || k === "BKSP" ? " wide" : "") + (k === "SPACE" ? " space" : ""));
        b.type = "button"; b.tabIndex = -1; b.dataset.k = k;
        b.textContent = k === "SHIFT" ? "⇧" : k === "BKSP" ? "⌫" : k === "SPACE" ? "space" : k;
        b.setAttribute("aria-label", k === "SHIFT" ? "shift" : k === "BKSP" ? "backspace" : k === "SPACE" ? "space" : k);
        b.addEventListener("mousedown", function (e) { e.preventDefault(); });                       // keep the caret in the text box
        b.addEventListener("click", function () { press(k); });
        r.appendChild(b);
      });
      osk.appendChild(r);
    });
  }
  function updateShift() { var b = osk.querySelector('[data-k="SHIFT"]'); if (b) b.classList.toggle("on", S.shift); osk.classList.toggle("shifted", S.shift);
    osk.querySelectorAll(".key").forEach(function (k) { var c = k.dataset.k; if (c.length === 1 && /[a-z]/.test(c)) k.textContent = S.shift ? c.toUpperCase() : c; }); }
  function press(k) {
    editor.focus();
    if (k === "SHIFT") { S.shift = !S.shift; updateShift(); return; }
    var s = editor.selectionStart, e = editor.selectionEnd;
    if (k === "BKSP") {
      if (e > s) editor.setRangeText("", s, e, "end"); else if (s > 0) editor.setRangeText("", s - 1, s, "end"); else return;
      S.keys += 1; S.autoSpaceAt = -1; afterEdit(); return;
    }
    var ch = k === "SPACE" ? " " : k;
    if (/^[a-z]$/.test(ch) && S.shift) { ch = ch.toUpperCase(); S.shift = false; updateShift(); }
    if (PUNCT.indexOf(ch) >= 0 && S.autoSpaceAt === s && e === s && editor.value[s - 1] === " ") editor.setRangeText(ch, s - 1, s, "end");   // no space before punctuation
    else editor.setRangeText(ch, s, e, "end");
    S.keys += 1; S.autoSpaceAt = -1; afterEdit();
  }

  // ------------------------------------------------------------------ counter and controls
  function updateStats() {
    var chars = Math.max(0, editor.value.length - S.base);
    $("s-keys").textContent = S.keys; $("s-chars").textContent = chars; $("s-acc").textContent = S.accepted;
    var box = $("s-saved-box");
    if (chars > 0) { var saved = 100 * (1 - S.keys / chars); $("s-saved").textContent = saved.toFixed(0) + " %"; box.classList.toggle("neg", saved < 0); }
    else { $("s-saved").textContent = "—"; box.classList.remove("neg"); }
  }
  function resetCounter() { S.keys = 0; S.accepted = 0; S.base = editor.value.length; S.autoSpaceAt = -1; updateStats(); }

  $("model").addEventListener("change", function (e) { S.primary = e.target.value; schedule(); editor.focus(); });
  $("model2").addEventListener("change", function (e) { S.secondary = e.target.value; schedule(); editor.focus(); });
  $("compare").addEventListener("change", function (e) { S.compare = e.target.checked; $("model2").disabled = !S.compare; schedule(); editor.focus(); });
  $("probs").addEventListener("change", function (e) { S.probs = e.target.checked; schedule(); editor.focus(); });
  $("rerank").addEventListener("change", function (e) { S.rerank = e.target.checked; schedule(); editor.focus(); });
  $("reset").addEventListener("click", function () { editor.value = ""; resetCounter(); schedule(); editor.focus(); });
  $("demo").addEventListener("change", function (e) {
    if (!e.target.value) return;
    editor.value = e.target.value.replace(/\s+$/, "") + " ";                                         // ready for the next word
    resetCounter(); editor.focus(); editor.setSelectionRange(editor.value.length, editor.value.length); schedule(); e.target.value = "";
  });

  // ------------------------------------------------------------------ text inspector
  var tiText = $("ti-text");
  $("ti-compare").addEventListener("change", function (e) { $("ti-model2").disabled = !e.target.checked; });
  $("ti-passage").addEventListener("change", function (e) {
    if (!e.target.value) return;
    fetch("/api/passage?id=" + encodeURIComponent(e.target.value)).then(function (r) { return r.json(); }).then(function (j) { if (j.text) { tiText.value = j.text; analyse(); } });
  });
  $("ti-go").addEventListener("click", analyse);

  function analyse() {
    var text = tiText.value.trim();
    if (!text) return;
    var names = [$("ti-model").value];
    if ($("ti-compare").checked && $("ti-model2").value !== names[0]) names.push($("ti-model2").value);
    var out = $("ti-results"), note = $("ti-note");
    out.innerHTML = ""; note.hidden = true;
    Promise.all(names.map(function (n) { return post("/api/inspect", { text: text, model: n }); })).then(function (list) {
      list.forEach(function (j) {
        if (j.error) { var c = el("div", "ti-card"); c.appendChild(el("p", null, j.error)); out.appendChild(c); return; }
        out.appendChild(card(j));
        if (j.truncated) { note.hidden = false; note.textContent = "Only the first " + j.max_words + " words were analysed."; }
      });
    });
  }

  function card(j) {
    var c = el("div", "ti-card");
    c.appendChild(el("h3", null, j.model + " · " + j.n_words + " words"));
    var sh = el("div", "shares");
    [["top-1", j.shares.top1, "c-g"], ["top-3", j.shares.top3, "c-y"], ["top-10", j.shares.top10, "c-o"], ["missed", j.shares.missed, "c-r"]].forEach(function (x) {
      var d = el("div", "share " + x[2]); d.appendChild(el("b", null, (100 * x[1]).toFixed(1) + " %")); d.appendChild(el("span", null, x[0])); sh.appendChild(d);
    });
    c.appendChild(sh);
    var p = el("div", "passage");
    j.segments.forEach(function (s) {
      if (s.k === "word") {
        var t = el("span", "tok c-" + s.c, s.t);
        t.title = s.rank == null ? (s.oov ? "not in this model's vocabulary" : "not a candidate word") + " · model's best: " + s.top3.join(", ")
                                 : "rank " + s.rank + " · p = " + (100 * s.prob).toFixed(s.prob < 0.1 ? 2 : 1) + " % · model's best: " + s.top3.join(", ");
        p.appendChild(t);
      } else if (s.k === "punct" || s.k === "num") p.appendChild(el("span", "punct", s.t));
      else p.appendChild(document.createTextNode(s.t));
    });
    c.appendChild(p);
    return c;
  }

  init();
})();
