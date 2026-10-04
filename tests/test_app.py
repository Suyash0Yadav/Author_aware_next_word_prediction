"""Tests for the Flask demo (fake models injected, no model files needed).

Run:  python -m pytest tests/test_app.py -q
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import app as demo                      # noqa: E402  (app/app.py)
from src import data as D               # noqa: E402

WORDS = ["the", "thou", "thee", "this", "hamlet", "i", "king", "come"]
ITOS = D.SPECIALS + D.PUNCT + WORDS
VOCAB = D.Vocab(ITOS)
ID = {t: i for i, t in enumerate(ITOS)}


class Fake:
    def __init__(self, order):
        p = np.full(len(ITOS), 1e-4)
        for rank, t in enumerate(order):
            p[ID[t]] = 0.3 / (rank + 1)
        self.p = p / p.sum()
        self.contexts = []

    def next_word_probs(self, ctx):
        self.contexts.append(list(ctx))
        return self.p.copy()


@pytest.fixture()
def client():
    a = Fake([",", "the", "thou", "thee", "this", "hamlet", "i", "king", "come"])
    b = Fake(["come", "king", "i", "hamlet", "this", "thee", "thou", "the"])
    models = {"Model A": demo.Entry(a, VOCAB, {"hamlet": "Hamlet", "i": "I"}, "fake a"),
              "Model B": demo.Entry(b, VOCAB, {}, "fake b")}
    app = demo.create_app(models, default="Model A")
    app.config["FAKES"] = (a, b)
    return app.test_client(), app


def post(c, **payload):
    return c.post("/api/suggest", json=payload)


def test_keyboard_page_is_served_and_has_the_shortcuts(client):
    c, _ = client
    r = c.get("/keyboard")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<textarea" in html and "Tab" in html and "Alt" in html          # the shortcuts are documented on the page
    js = (ROOT / "app" / "static" / "keyboard.js").read_text(encoding="utf-8")
    assert '"Tab"' in js and "Digit1" in js


def test_templates_and_static_files_have_no_external_resources():
    files = list((ROOT / "app" / "templates").glob("*.html")) + list((ROOT / "app" / "static").glob("*.*"))
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8")
        urls = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']|url\(([^)]+)\)|@import\s+["']([^"']+)""", text)
        external = [u for t in urls for u in t if u.startswith(("http://", "https://", "//"))]
        assert external == [], (f.name, external)                           # no CDN, no web fonts, no external resource at all
        assert "cdn" not in text.lower()


def test_models_endpoint_lists_models_and_default(client):
    c, _ = client
    j = c.get("/api/models").get_json()
    assert [m["name"] for m in j["models"]] == ["Model A", "Model B"] and j["default"] == "Model A"
    assert c.get("/").status_code == 200


def test_next_word_after_a_space_gives_three_words(client):
    c, _ = client
    r = post(c, text="my lord ", models=["Model A"], k=3).get_json()["results"]["Model A"]
    assert r["mode"] == "next" and [s["token"] for s in r["suggestions"]] == ["the", "thou", "thee"]
    assert r["suggestions"][0]["new_text"] == "my lord the " and r["ms"] >= 0


def test_mid_word_completes_and_restores_case(client):
    c, _ = client
    r = post(c, text="Alas. Th", models=["Model A"]).get_json()["results"]["Model A"]
    assert r["mode"] == "complete" and [s["word"] for s in r["suggestions"]] == ["The", "Thou", "Thee"]
    r2 = post(c, text="my lord ha", models=["Model A"]).get_json()["results"]["Model A"]
    assert [s["word"] for s in r2["suggestions"]] == ["Hamlet"]


def test_compare_mode_returns_both_models(client):
    c, _ = client
    res = post(c, text="my lord ", models=["Model A", "Model B"]).get_json()["results"]
    assert set(res) == {"Model A", "Model B"}
    assert res["Model A"]["suggestions"][0]["token"] != res["Model B"]["suggestions"][0]["token"]


def test_context_is_the_current_sentence_only(client):
    c, app = client
    post(c, text="Alas, poor Yorick. I knew him ", models=["Model A"])
    ctx = app.config["FAKES"][0].contexts[-1]
    assert ctx[0] == ID["<s>"] and ctx[1:] == [ID["i"], ID["<unk>"], ID["<unk>"]]   # 'i knew him' only; nothing before the '.'


def test_bad_requests_are_rejected(client):
    c, _ = client
    assert c.post("/api/suggest", data="not json").status_code == 400
    assert post(c, text="x", models=["Nope"]).status_code == 400
    assert post(c, text=5).status_code == 400


def test_very_long_text_is_truncated_not_rejected(client):
    c, _ = client
    assert post(c, text="the " * 5000, models=["Model A"]).status_code == 200


def test_demo_prefix_file_is_parsed(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("# comment\nmy lord, thou  |||  shows thou\n\nCome hither,  |||  x\n", encoding="utf-8")
    assert demo.read_demo_prefixes(f) == [{"text": "my lord, thou", "note": "shows thou"}, {"text": "Come hither,", "note": "x"}]


def test_rerank_flag_reorders_suggestions(client):
    c, _ = client
    # Model B: probabilities come > king > i > hamlet; with the keystroke score hamlet (6 letters) overtakes i
    a = post(c, text="my lord ", models=["Model B"], k=3).get_json()["results"]["Model B"]["suggestions"]
    b = post(c, text="my lord ", models=["Model B"], k=3, rerank=True).get_json()["results"]["Model B"]["suggestions"]
    assert [s["token"] for s in a] == ["come", "king", "i"] and [s["token"] for s in b] == ["come", "king", "hamlet"]
