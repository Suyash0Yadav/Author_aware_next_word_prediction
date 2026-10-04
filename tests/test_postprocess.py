"""Tests for src/postprocess.py (fake model, tiny vocabulary).

Run:  python -m pytest tests/test_postprocess.py -q
"""
import numpy as np
import pytest

from src import data as D
from src import postprocess as PP

WORDS = ["the", "thou", "thee", "this", "hamlet", "i", "o", "'tis", "king", "come"]
ITOS = D.SPECIALS + D.PUNCT + WORDS
VOCAB = D.Vocab(ITOS)
ID = {t: i for i, t in enumerate(ITOS)}
V = len(ITOS)
CASE_MAP = {"hamlet": "Hamlet", "i": "I", "o": "O"}


class Fake:
    """Fixed distribution; the raw top tokens are deliberately junk: ',' , </s>, <unk>, then words."""
    name = "fake"

    def __init__(self):
        p = np.full(V, 1e-4)
        order = [",", "</s>", "<unk>", ".", "the", "thou", "thee", "this", "hamlet", "i", "o", "'tis", "king", "come"]
        for rank, t in enumerate(order):
            p[ID[t]] = 0.3 / (rank + 1)
        self.p = p / p.sum()
        self.last_context = None

    def next_word_probs(self, context):
        self.last_context = list(context)
        return self.p.copy()


# ---------------------------------------------------------------- typed text -> context
def test_context_is_the_current_sentence_only():
    ctx, partial, start = PP.parse_typed("Alas, poor Yorick. I knew him, Horatio")
    assert ctx == ["i", "knew", "him", ","] and partial == "horatio" and not start


def test_text_after_terminal_punctuation_starts_a_new_sentence():
    for text in ("Come hither. ", "Come hither! ", "What is this? "):
        ctx, partial, start = PP.parse_typed(text)
        assert ctx == [] and partial == "" and start
    assert PP.parse_typed("")[2] is True


def test_partial_word_detection_and_apostrophes():
    assert PP.split_partial("my lord, th") == ("my lord, ", "th")
    assert PP.split_partial("my lord ") == ("my lord ", "")
    assert PP.split_partial("it is 'ti") == ("it is ", "'ti")
    assert PP.split_partial("hello '") == ("hello '", "")


def test_typed_text_uses_the_preprocessing_normalisation():
    ctx, partial, _ = PP.parse_typed("Caesar’s o’er 5 well-met “friends”,")      # curly apostrophes, digits, hyphen, quotes
    assert ctx == ["caesar's", "o'er", "<num>", "well", "met", "friends", ","]


# ---------------------------------------------------------------- stages 1-3
def test_raw_top_k_contains_special_tokens_and_punctuation():
    raw = [t for t, _ in PP.raw_topk(Fake().p, VOCAB, 5)]
    assert raw[:3] == [",", "</s>", "<unk>"]


def test_filter_removes_specials_and_punctuation():
    toks = [t for t, _ in PP.filter_tokens(Fake().p, VOCAB, 5)]
    assert toks == ["the", "thou", "thee", "this", "hamlet"]
    assert not ({"<s>", "</s>", "<unk>", "<num>", ",", "."} & set(toks))


def test_prefix_filter_keeps_matching_words_and_renormalises():
    out = PP.prefix_filter(Fake().p, VOCAB, "th", k=10)
    assert [t for t, _ in out] == ["the", "thou", "thee", "this"]
    assert sum(pr for _, pr in out) == pytest.approx(1.0)
    assert out[0][1] > out[1][1] > out[2][1]
    assert PP.prefix_filter(Fake().p, VOCAB, "zz") == []


def test_no_prefix_renormalises_over_all_words():
    out = PP.prefix_filter(Fake().p, VOCAB, "", k=100)
    assert len(out) == len(WORDS) and sum(pr for _, pr in out) == pytest.approx(1.0)


# ---------------------------------------------------------------- stage 4
def test_case_restoration():
    assert PP.restore_case("the", True, CASE_MAP) == "The"                   # sentence start
    assert PP.restore_case("hamlet", False, CASE_MAP) == "Hamlet"            # proper noun
    assert PP.restore_case("i", False, CASE_MAP) == "I"
    assert PP.restore_case("o", False, CASE_MAP) == "O"
    assert PP.restore_case("king", False, CASE_MAP) == "king"
    assert PP.restore_case("'tis", True, CASE_MAP) == "'Tis"                 # capitalise the first letter, not the apostrophe
    assert PP.restore_case("king", False, CASE_MAP, typed_prefix="Ki") == "King"   # a capital typed by the user is kept


# ---------------------------------------------------------------- stage 5
def test_insertion_replaces_the_partial_word_and_adds_a_space():
    assert PP.insert_word("my lord, th", "th", "thou") == "my lord, thou "
    assert PP.insert_word("my lord, ", "", "thou") == "my lord, thou "


def test_no_space_before_punctuation():
    assert PP.insert_punctuation("my lord, thou ", ",") == "my lord, thou,"
    assert PP.insert_punctuation("what ", "?") == "what?"
    assert PP.insert_punctuation("what", "?") == "what?"
    assert PP.insert_punctuation("what ", "x") == "what x"                   # only punctuation removes the space
    assert PP.insert_punctuation("a  ", ".") == "a  ."                       # a space the user typed twice is theirs


# ---------------------------------------------------------------- whole pipeline
def test_pipeline_next_word_after_a_space():
    m = Fake()
    r = PP.suggest(m, VOCAB, "Alas. my lord ", CASE_MAP, k=3, trace=True)
    assert r["mode"] == "next" and r["context"] == ["my", "lord"]
    assert m.last_context == [ID["<s>"], ID["my"] if "my" in ID else ID["<unk>"], ID["<unk>"]]   # sentence only, <s>-padded
    assert [s["token"] for s in r["suggestions"]] == ["the", "thou", "thee"]
    assert r["suggestions"][0]["new_text"] == "Alas. my lord the "
    assert len(r["stages"]) == 5 and r["stages"]["1 raw top-k"][0][0] == ","


def test_pipeline_completion_mid_word_with_case_restoration():
    r = PP.suggest(Fake(), VOCAB, "Alas. Th", CASE_MAP, k=3)
    assert r["mode"] == "complete" and r["partial"] == "th"
    assert [s["word"] for s in r["suggestions"]] == ["The", "Thou", "Thee"]      # prefix + sentence-start capital (typed 'T')
    assert r["suggestions"][1]["new_text"] == "Alas. Thou "
    assert sum(s["prob"] for s in PP.suggest(Fake(), VOCAB, "Alas. Th", CASE_MAP, k=10)["suggestions"]) == pytest.approx(1.0)


def test_pipeline_mid_sentence_proper_noun_and_empty_prefix_result():
    r = PP.suggest(Fake(), VOCAB, "my lord ha", CASE_MAP, k=3)
    assert [s["word"] for s in r["suggestions"]] == ["Hamlet"]
    assert PP.suggest(Fake(), VOCAB, "my lord zq", CASE_MAP)["suggestions"] == []


# ---------------------------------------------------------------- stage 6: keystroke-aware reranking
def test_keystroke_reranking_orders_by_expected_keystrokes_saved():
    cands = [("i", 0.30), ("the", 0.25), ("hamlet", 0.20), ("king", 0.10), ("come", 0.10), ("thou", 0.05)]
    assert [t for t, _ in PP.rerank_keystrokes(cands, 0)][:3] == ["hamlet", "the", "king"]      # 1.2, 0.75, 0.4 (tie: probability order)
    # with 2 letters typed the gain is P * (len - 2)
    assert [t for t, _ in PP.rerank_keystrokes([("the", 0.5), ("thou", 0.3), ("thee", 0.2)], 2)] == ["thou", "the", "thee"]
    assert PP.rerank_keystrokes([("th", 0.9), ("thou", 0.1)], 2)[0][0] == "thou"                 # a fully typed word saves nothing


def test_pipeline_with_rerank_promotes_longer_words():
    base = PP.suggest(Fake(), VOCAB, "my lord ", CASE_MAP, k=3)
    rr = PP.suggest(Fake(), VOCAB, "my lord ", CASE_MAP, k=3, rerank=True, trace=True)
    assert base["suggestions"][0]["token"] == "the"
    assert {s["token"] for s in rr["suggestions"]} == {"thou", "hamlet", "the"} and rr["suggestions"][0]["token"] != "the"
    assert rr["suggestions"][0]["prob"] < base["suggestions"][0]["prob"]                    # probabilities are unchanged: just reordered
    assert "(3 before reranking)" in rr["stages"]
