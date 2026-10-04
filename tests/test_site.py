"""Tests for the project website (app/app.py, site_data.py, pages.py, inspect_text.py, templates).

Run:  python -m pytest tests/test_site.py -q
Most tests run on the real project files (the site must show them); a few use an empty or tiny temporary project root
to check that missing files never crash a page.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import app as site_app                   # noqa: E402  (app/app.py)
import content                           # noqa: E402
import inspect_text                      # noqa: E402
from site_data import SiteData           # noqa: E402
from src import data as D                # noqa: E402
from src import postprocess as PP        # noqa: E402
from src import preprocess as P          # noqa: E402

PAGES = ["/", "/dataset", "/preprocessing", "/models", "/cross-domain", "/postprocessing", "/keyboard", "/about"]
WORDS = ["the", "cat", "car", "dog", "thou"]
ITOS = D.SPECIALS + D.PUNCT + WORDS
VOCAB = D.Vocab(ITOS)
ID = {t: i for i, t in enumerate(ITOS)}


class Fake:
    """the .4  cat .3  car .2  dog .1  thou .05 - and it records the contexts it is asked about."""
    name = "fake"

    def __init__(self):
        p = np.full(len(ITOS), 1e-4)
        for w, x in zip(WORDS, [0.4, 0.3, 0.2, 0.1, 0.05]):
            p[ID[w]] = x
        self.p = p / p.sum()
        self.contexts = []

    def next_word_probs(self, ctx):
        self.contexts.append(list(ctx))
        return self.p.copy()


@pytest.fixture(scope="module")
def models():
    return {"Model A": site_app.Entry(Fake(), VOCAB, {}, "fake a"), "Model B": site_app.Entry(Fake(), VOCAB, {}, "fake b")}


@pytest.fixture(scope="module")
def client(models):
    return site_app.create_app(models, default="Model A").test_client()


# ------------------------------------------------------------------------------------------- every page
@pytest.mark.parametrize("url", PAGES)
def test_every_page_returns_200_and_has_the_navigation(client, url):
    r = client.get(url)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    for _, label, link in content.NAV:
        assert f'href="{link}"' in html                     # sidebar on every page
    assert "Presentation mode" in html and "<h1" in html


@pytest.mark.parametrize("url", PAGES)
def test_every_page_returns_200_with_an_empty_project_root(models, tmp_path, url):
    """No result file at all: the pages must still render (with notices), never fail."""
    c = site_app.create_app(models, default="Model A", root=tmp_path).test_client()
    r = c.get(url)
    assert r.status_code == 200
    assert "referenced file(s) are missing" in r.get_data(as_text=True)


def test_unknown_page_is_404(client):
    assert client.get("/nope").status_code == 404


# ------------------------------------------------------------------------------------------- data comes from the files
def test_overview_cards_are_computed_from_the_result_files(client):
    html = client.get("/").get_data(as_text=True)
    t = pd.read_csv(ROOT / "tables" / "rnn_test_results.csv").set_index("model")
    assert f"{t.loc['LSTM', 'ppl']:.1f} vs {t.loc['KN trigram', 'ppl']:.1f}" in html
    cx = pd.read_csv(ROOT / "tables" / "baseline_cross_eval.csv")
    r = cx[(cx.architecture == "LSTM") & (cx.tested_on == "Shakespeare")].set_index("trained_on")
    assert f"{100 * r.loc['Shakespeare', 'ksr']:.1f} % vs {100 * r.loc['WikiText', 'ksr']:.1f} %" in html
    p = pd.read_csv(ROOT / "tables" / "robust_mcnemar.csv").iloc[0]["p_exact"]
    assert f"{float(p):.1e}" in html
    vocab = D.load_vocab()
    assert f"{len(vocab):,} tokens" in html


def test_main_test_table_comes_from_metrics_csv_test_rows_and_agrees_with_the_summary_table():
    S = SiteData(ROOT)
    t, notes = S.main_test_table()
    assert notes == [] and set(t["label"]) == {"KN bigram", "KN trigram", "Vanilla RNN", "LSTM", "Ensemble (LSTM+KN3)"}
    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    for _, row in t.iterrows():
        src = m[(m["model"] == row["run"]) & (m["split"] == "test")].iloc[-1]
        assert row["ppl"] == src["ppl"] and row["split"] == "test"
    assert (t["split"] == "test").all()                       # never a validation row in the main table


def test_models_page_separates_test_and_validation(client):
    html = client.get("/models").get_data(as_text=True)
    assert "badge-test" in html and "badge-val" in html
    assert "<strong>validation</strong> data used to choose hyper-parameters" in html
    val = SiteData(ROOT).validation_table()
    assert (val["model"].map(lambda s: isinstance(s, str))).all() and len(val) > 10     # tuning rows exist but live in their own section


def test_best_value_of_each_column_is_highlighted_and_columns_are_sortable(client):
    html = client.get("/models").get_data(as_text=True)
    main = re.search(r'<table class="data sortable main">.*?</table>', html, re.S).group(0)
    assert main.count('class="best"') >= 8 and 'title="' in main and "data-v=" in main
    S = SiteData(ROOT)
    t, _ = S.main_test_table()
    best = t.loc[t["ppl"].idxmin(), "label"]
    first_best_row = [r for r in re.findall(r"<tr>.*?</tr>", main, re.S) if 'class="best"' in r and best in r]
    assert first_best_row                                      # the lowest-perplexity model has a highlighted cell


def test_table_html_highlight_and_formatting():
    S = SiteData(ROOT)
    df = pd.DataFrame({"model": ["a", "b"], "ppl": [10.0, 9.0], "top1": [0.1, 0.2], "x": [np.nan, 1.0]})
    h = S.table_html(df, highlight={"ppl": "min", "top1": "max"})
    assert h.count('class="best"') == 2 and "9.00" in h and "20.0 %" in h and "–" in h       # NaN -> dash


def test_markdown_renderer_handles_tables_lists_code_and_inline_markup():
    S = SiteData(ROOT)
    h = str(S.md_to_html("## T\n\n| a | b |\n|---|---|\n| 1 | `x` |\n\n- **bold** item\n- other\n\n```bash\npip install x\n```\n\n<script>"))
    assert "<table" in h and "<code>x</code>" in h and "<strong>bold</strong>" in h and "<pre>" in h
    assert "<script>" not in h and "&lt;script&gt;" in h                                         # HTML in the source is escaped
    assert S.md_section("# T\n\n## A\nbody a\n\n## B\nbody b\n", "A") == "body a"


def test_about_page_quotes_the_readme_and_key_numbers(client):
    html = client.get("/about").get_data(as_text=True)
    assert "2,048 tokens" in html and "get_shakespeare.py" in html          # disclosure + reproduction steps come from the README
    assert "Consistency checks" in html and "[TEAM NAME]" in html and "Related work" in html


# ------------------------------------------------------------------------------------------- startup check / manifests
def test_nothing_referenced_by_the_pages_is_missing_in_the_real_project():
    S = SiteData(ROOT)
    assert S.missing_files() == []


def test_empty_root_lists_every_referenced_file_as_missing(tmp_path):
    S = SiteData(tmp_path)
    assert set(S.missing_files()) == set(S.referenced_files())


def test_every_figure_and_table_used_by_the_pages_is_in_the_manifest():
    figs = set()
    for tpl in (ROOT / "app" / "templates").glob("*.html"):
        figs |= set(re.findall(r"figure_html\('([^']+)'\)", tpl.read_text(encoding="utf-8")))
    assert figs and figs <= set(content.FIGURES)                       # every figure on a page has a caption entry
    tables = set(re.findall(r'S\.table\("([a-z_0-9]+)"\)', (ROOT / "app" / "pages.py").read_text(encoding="utf-8")))
    assert tables <= set(content.REQUIRED_TABLES)
    for name, (what, why, label) in content.FIGURES.items():
        assert what.strip() and why.strip() and label in {"test", "val", "train", "all", "app"}, name


def test_every_figure_has_a_caption_on_the_page(client):
    for url in ("/dataset", "/models", "/cross-domain", "/postprocessing", "/preprocessing"):
        html = client.get(url).get_data(as_text=True)
        assert html.count("<figure") == html.count("<figcaption>") > 0
        for img in re.findall(r'<img src="(/figures/[^"]+)"', html):
            assert client.get(img).status_code == 200


def test_result_files_are_never_modified():
    """Read-only: hit every page and API, then compare size + hash of the result files."""
    def snapshot():
        files = [ROOT / "results" / "metrics.csv", ROOT / "docs" / "key_numbers.md", ROOT / "data" / "split.csv"]
        files += sorted((ROOT / "tables").glob("*.csv")) + sorted((ROOT / "figures").glob("*.png"))
        return {str(f): hashlib.md5(f.read_bytes()).hexdigest() for f in files if f.exists()}
    before = snapshot()
    c = site_app.create_app({"Model A": site_app.Entry(Fake(), VOCAB, {})}, default="Model A").test_client()
    for url in PAGES + ["/api/models", "/api/passages", "/api/health", "/api/preprocessing/sample?sample=hamlet_open&stage=2"]:
        c.get(url)
    c.post("/api/inspect", json={"text": "the cat", "model": "Model A"})
    assert snapshot() == before


# ------------------------------------------------------------------------------------------- offline
def test_site_is_fully_offline():
    pattern = re.compile(r"""(?:src|href)\s*=\s*["'](https?://[^"']+|//[^"']+)["']|url\(\s*["']?(https?:)|@import\s+["']?https?:""", re.I)
    files = list((ROOT / "app" / "templates").glob("*.html")) + list((ROOT / "app" / "static").glob("*.*"))
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = [m.group(0) for m in pattern.finditer(text)]
        # only links that are plain documentation (not resources) may mention a URL: the references on the About page are text
        assert not hits, (f.name, hits)
        assert "cdn." not in text.lower() and "fonts.googleapis" not in text.lower(), f.name


# ------------------------------------------------------------------------------------------- figures route
def test_figure_route_serves_figures_and_blocks_path_tricks(client):
    r = client.get("/figures/zipf_rank_frequency.png")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert client.get("/figures/../README.md").status_code == 404
    assert client.get("/figures/..%2FREADME.md").status_code == 404
    assert client.get("/figures/no_such_figure.png").status_code == 404


# ------------------------------------------------------------------------------------------- inspect API
def test_tokenize_spans_equals_the_preprocessing_tokenizer():
    for text in ["Caesar’s o’er 5 well-met “friends”, 'tis 2s. done!", "Alas, poor Yorick—I knew him; 1599 and 3rd.",
                 "  th' ambition i' th' wall? ok\nnew line", "a--b", ""]:
        norm, spans = PP.tokenize_spans(text)
        assert [s[2] for s in spans] == [t.lower() for t in P.tokenize(norm)]
        assert all(norm[s[0]:s[1]].lower() == s[2] or s[3] == "num" for s in spans)


def test_inspect_colours_words_by_rank_and_counts_unknown_words_as_missed():
    m = {"Model A": site_app.Entry(Fake(), VOCAB, {}, "")}
    r = inspect_text.inspect_passage(m["Model A"], "the cat sat. thou dog")
    words = [s for s in r["segments"] if s["k"] == "word"]
    assert [(w["t"], w["rank"], w["c"]) for w in words] == [("the", 1, "g"), ("cat", 2, "y"), ("sat", None, "r"), ("thou", 5, "o"), ("dog", 4, "o")]
    assert words[2]["oov"] is True
    assert r["shares"] == {"top1": 0.2, "top3": 0.4, "top10": 0.8, "missed": 0.2} and r["n_words"] == 5
    assert "".join(s["t"] for s in r["segments"]) == "the cat sat. thou dog"               # the text is reproduced exactly


def test_inspect_uses_the_current_sentence_only(models):
    fake = Fake()
    entry = site_app.Entry(fake, VOCAB, {}, "")
    inspect_text.inspect_passage(entry, "the cat. dog thou")
    # contexts asked for the 2nd sentence start from <s> alone: nothing of the first sentence is visible
    assert any(c == [ID["<s>"]] for c in fake.contexts) and all(ID["."] not in c[1:] or c[0] == ID["<s>"] for c in fake.contexts)
    assert [ID["<s>"], ID["dog"]] in fake.contexts


def test_inspect_truncates_long_passages():
    entry = site_app.Entry(Fake(), VOCAB, {}, "")
    r = inspect_text.inspect_passage(entry, "the cat " * 20, max_words=6)
    assert r["truncated"] is True and r["n_words"] == 6


def test_inspect_endpoint(client):
    r = client.post("/api/inspect", json={"text": "the cat sat. thou dog", "model": "Model A"})
    j = r.get_json()
    assert r.status_code == 200 and j["model"] == "Model A" and j["n_words"] == 5 and set(j["shares"]) == {"top1", "top3", "top10", "missed"}
    assert client.post("/api/inspect", json={"text": "x", "model": "nope"}).status_code == 400
    assert client.post("/api/inspect", json={"text": "   ", "model": "Model A"}).status_code == 400
    assert client.post("/api/inspect", data="not json").status_code == 400
    big = client.post("/api/inspect", json={"text": "the cat " * 400, "model": "Model A"}).get_json()
    assert big["truncated"] is True and big["n_words"] == inspect_text.MAX_WORDS


def test_suggest_endpoint_now_returns_the_context_and_up_to_ten_suggestions(client):
    j = client.post("/api/suggest", json={"text": "Alas. the cat ", "models": ["Model A"], "k": 10}).get_json()["results"]["Model A"]
    assert j["context"] == ["the", "cat"] and len(j["suggestions"]) == 5          # the fake vocabulary has only 5 words
    assert client.post("/api/suggest", json={"text": "x", "models": ["Model A"], "k": 99}).status_code == 200


# ------------------------------------------------------------------------------------------- passages / stepper / health
def test_passages_endpoint_lists_test_play_passages_and_serves_their_text(client):
    j = client.get("/api/passages").get_json()["passages"]
    assert len(j) >= 4 and all({"id", "play", "label", "words"} <= set(p) for p in j)
    split = pd.read_csv(ROOT / "data" / "split.csv")
    test_slugs = set(split[split.split == "test"].slug)
    assert {p["id"].split(":")[0] for p in j} == test_slugs                       # only test plays
    one = client.get(f"/api/passage?id={j[0]['id']}").get_json()
    assert len(one["text"].split()) >= 100
    assert client.get("/api/passage?id=hamlet_prince_of_denmark:0").status_code == 404      # a train play is refused
    assert client.get("/api/passage?id=garbage").status_code == 404


def test_stage_sample_endpoint_shows_the_hamlet_passage_before_and_after_a_stage(client):
    j = client.get("/api/preprocessing/sample?sample=hamlet_open&stage=3").get_json()
    assert "Enter Francisco" in j["before"] and "Enter Francisco" not in j["after"] and "BARNARDO." in j["after"]
    j5 = client.get("/api/preprocessing/sample?sample=hamlet_open&stage=5").get_json()
    assert "BARNARDO." in j5["before"] and "BARNARDO." not in j5["after"]
    raw = client.get("/api/preprocessing/sample?sample=hamlet_open&stage=0").get_json()
    assert raw["before"] is None and "ACT I" in raw["after"]
    assert client.get("/api/preprocessing/sample?sample=nope&stage=3").status_code == 404
    assert client.get("/api/preprocessing/sample?sample=hamlet_open&stage=x").status_code == 400
    assert client.get("/api/preprocessing/sample?sample=hamlet_open&stage=42").status_code == 404


def test_stage_clouds_are_png_images_and_differ_between_stages(client):
    a, b = client.get("/api/preprocessing/cloud/0.png"), client.get("/api/preprocessing/cloud/5.png")
    assert a.status_code == b.status_code == 200 and a.mimetype == "image/png" and a.get_data()[:8] == b"\x89PNG\r\n\x1a\n"
    assert a.get_data() != b.get_data()
    assert client.get("/api/preprocessing/cloud/99.png").status_code == 404


def test_stage_sample_and_passages_on_a_tiny_project_root(models, tmp_path):
    (tmp_path / "data" / "works").mkdir(parents=True)
    (tmp_path / "data" / "works" / "hamlet_prince_of_denmark.txt").write_text("ACT I\n\nBARNARDO.\nWho's there?\n\nFRANCISCO.\nNay, answer me.\n", encoding="utf-8")
    c = site_app.create_app(models, default="Model A", root=tmp_path).test_client()
    j = c.get("/api/preprocessing/sample?sample=hamlet_open&stage=0").get_json()
    assert "Nay, answer me" in j["after"]
    assert c.get("/api/preprocessing/sample?sample=hamlet_open&stage=1").get_json()["after"] is None       # stage file missing -> no crash
    assert c.get("/api/passages").get_json() == {"passages": []}
    assert c.get("/api/preprocessing/cloud/3.png").status_code == 404


def test_health_endpoint_reports_missing_files(models, tmp_path):
    ok = site_app.create_app(models, default="Model A").test_client().get("/api/health").get_json()
    assert ok["missing_files"] == [] and ok["models"] == ["Model A", "Model B"]
    bad = site_app.create_app(models, default="Model A", root=tmp_path).test_client().get("/api/health").get_json()
    assert "tables/rnn_test_results.csv" in bad["missing_files"] and len(bad["missing_files"]) > 50


# ------------------------------------------------------------------------------------------- keyboard page contents
def test_keyboard_page_has_every_requested_control(client):
    html = client.get("/keyboard").get_data(as_text=True)
    for needle in ['id="editor"', 'id="osk"', 'id="inspector"', 'id="model"', 'id="compare"', 'id="probs"', 'id="rerank"', "Favour keystroke savings",
                   'id="demo"', 'id="s-saved"', "Text inspector", 'id="ti-passage"', 'id="ti-text"', "top-1", "missed"]:
        assert needle in html, needle
    js = (ROOT / "app" / "static" / "keyboard.js").read_text(encoding="utf-8")
    assert "Alt+" in js and '"Tab"' in js and "Digit1" in js and "/api/suggest" in js and "/api/inspect" in js


def test_demo_prefix_file_is_parsed(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("# comment\nmy lord, thou  |||  shows thou\n\nCome hither,  |||  x\n", encoding="utf-8")
    assert site_app.read_demo_prefixes(f) == [{"text": "my lord, thou", "note": "shows thou"}, {"text": "Come hither,", "note": "x"}]


# ------------------------------------------------------------------------------------------- --warm-cache
def test_warm_cache_generates_every_stage_cloud_once(tmp_path):
    S = SiteData(ROOT)
    S.cache_dir = tmp_path                                                    # do not touch the real cache
    first = S.warm_cloud_cache()
    assert [r["stage"] for r in first] == list(range(10)) and all(r["available"] and r["generated"] for r in first)
    files = sorted(tmp_path.glob("cloud_stage*.png"))
    assert len(files) == 10
    mtimes = [f.stat().st_mtime_ns for f in files]
    second = S.warm_cloud_cache()
    assert not any(r["generated"] for r in second) and all(r["available"] for r in second)      # second run: everything is cached
    assert [f.stat().st_mtime_ns for f in sorted(tmp_path.glob("cloud_stage*.png"))] == mtimes


def test_warm_cache_on_an_empty_project_reports_unavailable_without_failing(tmp_path):
    S = SiteData(tmp_path / "empty")
    S.cache_dir = tmp_path / "cache"
    rows = S.warm_cloud_cache()
    assert len(rows) == 10 and not any(r["available"] or r["generated"] for r in rows)


def test_app_has_the_warm_cache_flag():
    src = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert '"--warm-cache"' in src and "warm_cloud_cache" in src
