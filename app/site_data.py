"""
site_data.py -- the data layer of the project website.

Everything shown on the site is READ from the project files at request time: results/metrics.csv, tables/*.csv,
figures/*.png, docs/key_numbers.md, README.md, data/split.csv and the data/ stages. Nothing is hard-coded and nothing
is ever written to those files (they are treated as read-only; the only thing written is an image cache in app/cache/).
A missing file never crashes a page: the readers return None / [] and the page shows a notice.
"""
import html
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from markupsafe import Markup, escape

from content import (FIGURES, REQUIRED_DATA, REQUIRED_TABLES, SAMPLES, STAGE_DIRS, METRIC_HELP)

ROOT = Path(__file__).resolve().parent.parent
LABEL_BADGES = {"test": ("TEST", "badge-test"), "val": ("VALIDATION", "badge-val"), "train": ("TRAIN", "badge-val"),
                "all": ("ALL DATA", "badge-all"), "app": ("DEMO", "badge-all"), "exploratory": ("EXPLORATORY", "badge-expl")}
PCT_COLS = {"top1", "top3", "top5", "ksr", "archaic_top1", "archaic_top3", "archaic_top5", "archaic_ksr", "oov_word_rate",
            "oov_pct_word_types", "unk_pct_all_tokens"}


def _nz(s):
    return re.sub(r"[^a-z]", "", s.lower())


def title_case(t):
    t = str(t).title().replace("’", "'")
    return re.sub(r"'S\b", "'s", t)


class SiteData:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self._csv_cache = {}
        self.cache_dir = Path(__file__).resolve().parent / "cache"

    # ------------------------------------------------------------------------------------ files
    def path(self, rel):
        return self.root / rel

    def exists(self, rel):
        return self.path(rel).exists()

    def read_text(self, rel):
        p = self.path(rel)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def csv(self, rel):
        """DataFrame of a CSV (cached by modification time) or None if the file is missing."""
        p = self.path(rel)
        if not p.is_file():
            return None
        key = (str(p), p.stat().st_mtime_ns)
        if key not in self._csv_cache:
            self._csv_cache[key] = pd.read_csv(p, encoding="utf-8")
        return self._csv_cache[key].copy()

    def table(self, name):
        return self.csv(f"tables/{name}.csv")

    def figure_path(self, name):
        if "/" in name or "\\" in name or name.startswith("."):
            return None
        p = self.path("figures") / name
        return p if p.is_file() else None

    def json(self, rel):
        t = self.read_text(rel)
        return json.loads(t) if t else None

    # ------------------------------------------------------------------------------------ startup check
    def referenced_files(self):
        files = [f"tables/{t}.csv" for t in REQUIRED_TABLES] + [f"figures/{f}" for f in FIGURES] + list(REQUIRED_DATA)
        return files

    def missing_files(self):
        return [f for f in self.referenced_files() if not self.path(f).exists()]

    # ------------------------------------------------------------------------------------ rendering helpers
    def figure_html(self, name, caption_extra=""):
        spec = FIGURES.get(name)
        cap = f"{spec[0]} {spec[1]}" if spec else ""
        label = spec[2] if spec else "all"
        badge = self.badge(label)
        if self.figure_path(name):
            img = f'<a href="/figures/{name}" target="_blank" rel="noopener"><img src="/figures/{name}" alt="{html.escape(spec[0] if spec else name)}" loading="lazy"></a>'
        else:
            img = f'<div class="missing">Missing figure: figures/{html.escape(name)}</div>'
        return Markup(f'<figure class="fig">{img}<figcaption>{badge} {html.escape(cap)}{caption_extra}</figcaption></figure>')

    def badge(self, label):
        text, cls = LABEL_BADGES.get(label, (label.upper(), "badge-all"))
        return Markup(f'<span class="badge {cls}">{text}</span>')

    def notice_missing(self, what):
        return Markup(f'<div class="missing">Missing: {html.escape(what)}</div>')

    def table_html(self, df, *, labels=None, fmt=None, highlight=None, tooltips=None, sortable=True, cls="", columns=None,
                   empty="no data"):
        """HTML <table> for a DataFrame. highlight: {col: 'max'|'min'} marks the best cell; fmt: {col: callable};
        labels: {col: header text}; tooltips: {col: text}. Numeric cells carry data-v for client-side sorting."""
        if df is None:
            return self.notice_missing(empty)
        labels, fmt, highlight, tooltips = labels or {}, fmt or {}, highlight or {}, tooltips or {}
        cols = list(columns) if columns else list(df.columns)
        best = {}
        for c, how in highlight.items():
            if c in df and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                v = df[c].astype(float)
                best[c] = set(v[v == (v.max() if how == "max" else v.min())].index)
        head = "".join(f'<th data-col="{i}" title="{html.escape(tooltips.get(c, ""))}"{" class=has-tip" if c in tooltips else ""}>{html.escape(str(labels.get(c, c)))}</th>'
                       for i, c in enumerate(cols))
        body = []
        for idx, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                text = self.format_cell(c, v)
                if c in fmt and not (v is None or (isinstance(v, float) and np.isnan(v))):    # custom format; blanks stay blanks
                    try:
                        text = fmt[c](v)
                    except (TypeError, ValueError):
                        pass
                num = isinstance(v, (int, float, np.integer, np.floating)) and not (isinstance(v, float) and np.isnan(v))
                attrs = f' data-v="{float(v)}"' if num else ""
                klass = ' class="best"' if idx in best.get(c, ()) else ""
                cells.append(f"<td{attrs}{klass}>{html.escape(str(text))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        s = " sortable" if sortable else ""
        return Markup(f'<div class="table-wrap"><table class="data{s} {cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')

    @staticmethod
    def format_cell(col, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        c = str(col)
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            v = float(v)
            if c in PCT_COLS:
                return f"{100 * v:.1f} %"
            if "(%)" in c or "pct" in c:
                return f"{v:.2f} %"
            if c in ("ppl", "val_ppl", "best_val_ppl", "log_odds", "z_score", "latency_ms") or c.endswith("_ppl"):
                return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
            if v == int(v) and abs(v) >= 1000:
                return f"{int(v):,}"
            return f"{v:.4g}"
        return str(v)

    # ------------------------------------------------------------------------------------ markdown (README, key numbers)
    @staticmethod
    def md_inline(t):
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        return t

    def md_to_html(self, text, heading_shift=2):
        if not text:
            return Markup('<div class="missing">Missing document</div>')
        out, lines, i = [], text.splitlines(), 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("```"):
                j = i + 1
                code = []
                while j < len(lines) and not lines[j].startswith("```"):
                    code.append(lines[j]); j += 1
                out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                i = j + 1
            elif re.match(r"^#{1,6} ", line):
                lvl = min(6, len(line.split(" ")[0]) + heading_shift)
                out.append(f"<h{lvl}>{self.md_inline(line.lstrip('# ').strip())}</h{lvl}>")
                i += 1
            elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
                head = [c.strip() for c in line.strip().strip("|").split("|")]
                rows, j = [], i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
                th = "".join(f"<th>{self.md_inline(c)}</th>" for c in head)
                tb = "".join("<tr>" + "".join(f"<td>{self.md_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                out.append(f'<div class="table-wrap"><table class="data"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
                i = j
            elif re.match(r"^\s*[-*] ", line):
                items, j = [], i
                while j < len(lines) and (re.match(r"^\s*[-*] ", lines[j]) or (lines[j].startswith("  ") and lines[j].strip() and items)):
                    if re.match(r"^\s*[-*] ", lines[j]):
                        items.append(re.sub(r"^\s*[-*] ", "", lines[j]))
                    else:
                        items[-1] += " " + lines[j].strip()
                    j += 1
                out.append("<ul>" + "".join(f"<li>{self.md_inline(t)}</li>" for t in items) + "</ul>")
                i = j
            elif not line.strip():
                i += 1
            else:
                para = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip() and not re.match(r"^(#{1,6} |```|\||\s*[-*] )", lines[j]):
                    para.append(lines[j]); j += 1
                out.append("<p>" + self.md_inline(" ".join(para)) + "</p>")
                i = j
        return Markup("\n".join(out))

    @staticmethod
    def md_section(text, heading):
        """Text of the '## heading' section of a Markdown document (up to the next '## ')."""
        if not text:
            return None
        m = re.search(r"^## " + re.escape(heading) + r"\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
        return m.group(1).strip() if m else None

    # ------------------------------------------------------------------------------------ results
    def metrics(self):
        return self.csv("results/metrics.csv")

    def main_test_table(self):
        """The five main models, TEST split, straight from results/metrics.csv (names of the runs come from
        tables/rnn_test_results.csv). Returns (DataFrame, notes)."""
        m, ref = self.metrics(), self.table("rnn_test_results")
        notes = []
        if m is None or ref is None:
            return None, ["results/metrics.csv or tables/rnn_test_results.csv is missing"]
        runs = list(ref["run"])
        t = m[(m["split"] == "test") & (m["model"].isin(runs)) & (m["context"] == "sentence")].copy()
        t = t.sort_values("timestamp").drop_duplicates("model", keep="last").set_index("model").loc[[r for r in runs if r in set(t["model"])]]
        labels = dict(zip(ref["run"], ref["model"]))
        t.insert(0, "label", [labels[r] for r in t.index])
        t = t.reset_index().rename(columns={"model": "run"})
        for r in runs:
            if r not in set(t["run"]):
                notes.append(f"no test row for {r} in results/metrics.csv")
        for _, row in t.iterrows():                     # the log and the summary table must agree
            rr = ref[ref["run"] == row["run"]].iloc[0]
            if abs(float(row["ppl"]) - float(rr["ppl"])) > 1e-6:
                notes.append(f"results/metrics.csv and tables/rnn_test_results.csv disagree for {row['run']}")
        return t, notes

    def validation_table(self):
        m = self.metrics()
        if m is None:
            return None
        v = m[m["split"] == "val"].copy()
        return v[["model", "n", "hyperparameters", "context", "ppl", "top1", "top3", "ksr", "archaic_top1", "latency_ms"]].reset_index(drop=True)

    def overview_cards(self):
        cards = []
        t, _ = self.main_test_table()
        if t is not None and {"LSTM", "KN trigram"} <= set(t["label"]):
            l, k = t[t["label"] == "LSTM"].iloc[0], t[t["label"] == "KN trigram"].iloc[0]
            cards.append({"title": "Test perplexity", "big": f"{l['ppl']:.1f} vs {k['ppl']:.1f}",
                          "sub": f"LSTM vs KN trigram - {100 * (1 - l['ppl'] / k['ppl']):.1f} % lower perplexity on unseen plays",
                          "source": "results/metrics.csv (test rows)", "badge": "test"})
        cx = self.table("baseline_cross_eval")
        if cx is not None:
            sh = cx[(cx["tested_on"] == "Shakespeare") & (cx["architecture"] == "LSTM")].set_index("trained_on")
            kn = cx[(cx["tested_on"] == "Shakespeare") & (cx["architecture"] == "KN trigram")].set_index("trained_on")
            if {"Shakespeare", "WikiText"} <= set(sh.index):
                cards.append({"title": "Keystroke savings on Shakespeare text", "big": f"{100 * sh.loc['Shakespeare', 'ksr']:.1f} % vs {100 * sh.loc['WikiText', 'ksr']:.1f} %",
                              "sub": "LSTM trained on Shakespeare vs trained on WikiText-2" + (f" (KN trigram: {100 * kn.loc['Shakespeare', 'ksr']:.1f} % vs {100 * kn.loc['WikiText', 'ksr']:.1f} %)" if {"Shakespeare", "WikiText"} <= set(kn.index) else ""),
                              "source": "tables/baseline_cross_eval.csv", "badge": "test"})
        g, mc = self.table("rnn_top1_archaic_groups"), self.table("robust_mcnemar")
        if g is not None:
            gg = g[g["group"].str.startswith("thou")].set_index("model")
            if {"LSTM", "KN trigram"} <= set(gg.index):
                p = f", McNemar p = {float(mc.iloc[0]['p_exact']):.1e}" if mc is not None else ""
                cards.append({"title": "thou / thee / thy / thine, top-1", "big": f"{100 * gg.loc['LSTM', 'top1']:.1f} % vs {100 * gg.loc['KN trigram', 'top1']:.1f} %",
                              "sub": f"LSTM vs KN trigram on {int(gg.loc['LSTM', 'n_targets']):,} targets{p}",
                              "source": "tables/rnn_top1_archaic_groups.csv, tables/robust_mcnemar.csv", "badge": "test"})
        ov, vocab, idx = self.table("corpus_overview"), self.json("data/processed/vocab.json"), self.csv("data/processed/works_index.csv")
        if ov is not None and vocab is not None:
            ov = ov.set_index("metric")["value"]
            sent = f", {int(idx['n_sentences'].sum()):,} sentences" if idx is not None else ""
            cards.append({"title": "Corpus and vocabulary", "big": f"{int(ov['total words (whitespace split)']):,} words",
                          "sub": f"{int(ov['number of works'])} works{sent}; vocabulary of {vocab['size']:,} tokens (train only, min. frequency 2)",
                          "source": "tables/corpus_overview.csv, data/processed/vocab.json", "badge": "all"})
        return cards

    # ------------------------------------------------------------------------------------ dataset
    def works_split(self):
        s = self.csv("data/split.csv")
        if s is None:
            return None
        s = s.copy()
        s["title"] = s["title"].map(title_case)
        return s[["split", "title", "category", "num_words"]].sort_values(["split", "category", "title"]).reset_index(drop=True)

    def ngram_sparsity_svg(self):
        u = self.table("ngram_uniqueness")
        if u is None:
            return self.notice_missing("tables/ngram_uniqueness.csv")
        w, h, pad = 520, 220, 36
        bw = (w - 2 * pad) / len(u)
        bars = []
        for i, r in u.reset_index(drop=True).iterrows():
            pct = float(r["pct_unique_occurring_once"])
            bh = (h - 2 * pad) * pct / 100
            x = pad + i * bw + bw * 0.15
            y = h - pad - bh
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" height="{bh:.1f}" fill="#7a1f2b"/>'
                        f'<text x="{x + bw * 0.35:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="13" font-weight="700">{pct:.1f} %</text>'
                        f'<text x="{x + bw * 0.35:.1f}" y="{h - pad + 16}" text-anchor="middle" font-size="13">{int(r["n"])}-grams</text>'
                        f'<text x="{x + bw * 0.35:.1f}" y="{h - pad + 30}" text-anchor="middle" font-size="10" fill="#555">{int(r["unique_ngrams"]):,} types</text>')
        return Markup(f'<svg viewBox="0 0 {w} {h}" class="svgchart" role="img" aria-label="Share of n-gram types that occur only once">'
                      f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#333"/>{"".join(bars)}</svg>')

    # ------------------------------------------------------------------------------------ preprocessing
    def stage_table(self):
        return self.table("preprocessing_stages")

    def sample_defs(self):
        return [{"key": k, "label": lab} for k, lab, *_ in SAMPLES]

    def _stage_lines(self, stage, slug):
        if stage == 0:
            t = self.read_text(f"data/works/{slug}.txt")
        elif stage in STAGE_DIRS:
            t = self.read_text(f"data/interim/{STAGE_DIRS[stage]}/{slug}.txt")
        elif stage == 9:                                             # final, <unk>-mapped
            sp = self.csv("data/split.csv")
            idx = self.csv("data/processed/works_index.csv")
            if sp is None or idx is None:
                return None
            row = idx[idx["slug"] == slug]
            if row.empty:
                return None
            row = row.iloc[0]
            t = self.read_text(f"data/processed/{row['split']}.txt")
            return None if t is None else t.splitlines()[int(row["first_line"]): int(row["first_line"]) + int(row["n_sentences"])]
        else:
            return None
        return None if t is None else t.splitlines()

    def stage_window(self, stage, sample):
        key, label, slug, anchor, lb, la, sb, sa = sample
        lines = self._stage_lines(stage, slug)
        if lines is None:
            return None
        k = _nz(anchor)
        idx = next((i for i, l in enumerate(lines) if k in _nz(l)), None)
        if idx is None:
            return "(passage not found at this stage)"
        before, after = (sb, sa) if stage >= 6 else (lb, la)
        return "\n".join(l[:200] for l in lines[max(0, idx - before): idx + after + 1])

    def stage_sample(self, key, stage):
        sample = next((s for s in SAMPLES if s[0] == key), None)
        if sample is None or not 0 <= stage <= 9:
            return None
        after = self.stage_window(stage, sample)
        before = self.stage_window(stage - 1, sample) if stage >= 1 else None
        return {"sample": key, "stage": stage, "before": before, "after": after}

    def _cloud_cache_path(self, stage):
        return self.cache_dir / f"cloud_stage{stage}_{self.root.name}.png"

    def warm_cloud_cache(self, log=None):
        """Generate (or find in the cache) the word clouds of every preprocessing stage, so that nothing is slow later.
        Returns one dict per stage: {stage, seconds, generated, available}."""
        import time
        out = []
        for stage in range(10):
            had = self._cloud_cache_path(stage).exists()
            t0 = time.perf_counter()
            png = self.cloud_png(stage)
            row = {"stage": stage, "seconds": round(time.perf_counter() - t0, 2), "generated": bool(png is not None and not had), "available": png is not None}
            out.append(row)
            if log:
                log(row)
        return out

    def cloud_png(self, stage):
        """PNG bytes of the word cloud of one preprocessing stage (cached in app/cache/)."""
        if not 0 <= stage <= 9:
            return None
        self.cache_dir.mkdir(exist_ok=True)
        cache = self._cloud_cache_path(stage)
        if cache.exists():
            return cache.read_bytes()
        texts = []
        if stage == 0:
            texts = [p.read_text(encoding="utf-8") for p in sorted(self.path("data/works").glob("*.txt"))]
        elif stage in STAGE_DIRS:
            texts = [p.read_text(encoding="utf-8") for p in sorted(self.path(f"data/interim/{STAGE_DIRS[stage]}").glob("*.txt"))]
        else:
            texts = [self.path(f"data/processed/{s}.txt").read_text(encoding="utf-8") for s in ("train", "val", "test") if self.path(f"data/processed/{s}.txt").exists()]
        if not texts:
            return None
        from collections import Counter
        stop = self._stopwords()
        cnt = Counter()
        for t in texts:
            t = t.replace("’", "'").replace("‘", "'").replace("<BREAK>", " ")
            cnt.update(w for w in re.findall(r"[a-z]+(?:'[a-z]+)*", t.lower()) if w not in stop)
        if not cnt:
            return None
        from wordcloud import WordCloud
        wc = WordCloud(width=700, height=420, background_color="white", max_words=120, colormap="viridis", random_state=42).generate_from_frequencies(dict(cnt))
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        cache.write_bytes(buf.getvalue())
        return buf.getvalue()

    @staticmethod
    def _stopwords():
        try:
            from nltk.corpus import stopwords
            s = set(stopwords.words("english"))
        except Exception:
            s = {"the", "and", "i", "to", "of", "a", "you", "my", "that", "in", "is", "not", "it", "for", "with", "be", "me", "this",
                 "have", "he", "but", "as", "his", "your", "will", "so", "what", "her", "do", "we", "by", "all", "if", "are", "on"}
        return s | {"thou", "thee", "thy", "thine", "hath", "doth", "art"}

    # ------------------------------------------------------------------------------------ test-play passages (text inspector)
    PER_PLAY, MIN_WORDS = 4, 120

    def _make_passage(self, slug, k):
        """k-th passage of a test play: consecutive sentences (cased stage-6 text) from an evenly spaced start."""
        raw = self.read_text(f"data/interim/stage6_sentences/{slug}.txt")
        if not raw:
            return None
        lines = [l for l in raw.splitlines() if l.strip()]
        i = int(np.linspace(0.08, 0.83, self.PER_PLAY)[k] * len(lines))
        text = []
        while i < len(lines) and sum(len(t.split()) for t in text) < self.MIN_WORDS:
            text.append(lines[i])
            i += 1
        return " ".join(text)

    def passages(self):
        sp = self.csv("data/split.csv")
        if sp is None:
            return []
        out = []
        for _, r in sp[sp["split"] == "test"].iterrows():
            for k in range(self.PER_PLAY):
                text = self._make_passage(r["slug"], k)
                if text:
                    out.append({"id": f"{r['slug']}:{k}", "play": title_case(r["title"]),
                                "label": f"{title_case(r['title'])} - passage {k + 1}", "words": len(text.split())})
        return out

    def passage_text(self, pid):
        slug, _, k = pid.partition(":")
        sp = self.csv("data/split.csv")
        if sp is None or not k.isdigit() or int(k) >= self.PER_PLAY or slug not in set(sp.loc[sp["split"] == "test", "slug"]):
            return None
        return self._make_passage(slug, int(k))
