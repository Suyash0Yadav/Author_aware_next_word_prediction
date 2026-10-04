"""Tests for src/evaluate.py with a tiny hand-made vocabulary and fake models.

Run:  python -m pytest tests/test_evaluate.py -q
"""
import math

import numpy as np
import pytest

from src import data as D
from src import evaluate as E

# ids: 0 <s>, 1 </s>, 2 <unk>, 3 <num>, 4 . , 5 ',' , 6 ; , 7 : , 8 ! , 9 ? , then words
WORDS = ["the", "cat", "car", "dog", "thou"]
ITOS = D.SPECIALS + D.PUNCT + WORDS
VOCAB = D.Vocab(ITOS)
ID = {t: i for i, t in enumerate(ITOS)}
V = len(ITOS)


class FixedModel:
    """Same distribution at every position: the > cat > car > dog > thou, rest tiny."""
    name, n = "fixed", 1

    def __init__(self):
        p = np.full(V, 1e-4)
        for w, x in zip(WORDS, [0.4, 0.3, 0.2, 0.1, 0.05]):
            p[ID[w]] = x
        self.p = p / p.sum()

    def next_word_probs(self, context):
        return self.p.copy()


def sent(*toks):
    return [ID["<s>"]] + [ID[t] for t in toks] + [ID["</s>"]]


def run(model, sentences, **kw):
    return E.evaluate(model, "val", VOCAB, sentences=sentences, log=False, **kw)


def test_suggestion_set_excludes_specials_and_punctuation():
    assert [VOCAB.itos[i] for i in E.word_ids(VOCAB)] == WORDS
    with_punct = [VOCAB.itos[i] for i in E.word_ids(VOCAB, allow_punct=True)]
    assert set(D.PUNCT) <= set(with_punct) and not ({"<s>", "</s>", "<unk>", "<num>"} & set(with_punct))


def test_only_word_targets_enter_the_accuracy_denominator():
    # targets: the (word), ',' (punct), dog (word), <unk>, <num>, </s>
    r = run(FixedModel(), [sent("the", ",", "dog", "<unk>", "<num>")])
    assert r["n_tokens"] == 6
    assert r["n_word_targets"] == 2


def test_perplexity_counts_every_token_including_eos_unk_and_punct():
    uniform = type("U", (), {"name": "u", "n": 1, "next_word_probs": lambda self, c: np.full(V, 1.0 / V)})()
    r = run(uniform, [sent("the", ",", "<unk>")])
    assert r["ppl"] == pytest.approx(V)


def test_rank_metrics_by_hand():
    # ranks among words: the=1 cat=2 car=3 dog=4 thou=5
    r = run(FixedModel(), [sent("the", "cat", "car", "dog", "thou")])
    assert r["top1"] == pytest.approx(1 / 5)
    assert r["top3"] == pytest.approx(3 / 5)
    assert r["top5"] == pytest.approx(5 / 5)
    assert r["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 5)


def test_ksr_by_hand():
    """Targets the, car, cat, dog. Top-3 with empty prefix = the, cat, car -> 1 tap each (keys 1 vs 4 or 4/4/4...).
    dog: not in top 3; after typing 'd' only 'dog' starts with 'd' -> tap -> 2 keys (vs 4).
    with = 1+1+1+2 = 5 ; without = (3+1)*4 = 16 ; KSR = 1 - 5/16."""
    r = run(FixedModel(), [sent("the", "car", "cat", "dog")])
    assert r["ksr"] == pytest.approx(1 - 5 / 16)


def test_ksr_when_prediction_never_helps_is_zero():
    class Reverse(FixedModel):                              # worst word first, target never in the top 3 until typed
        def __init__(self):
            super().__init__()
            self.p = np.full(V, 1e-4)
            for w, x in zip(["thou", "dog", "car", "cat", "the"], [0.4, 0.3, 0.2, 0.1, 0.05]):
                self.p[ID[w]] = x
            self.p /= self.p.sum()
    # 'the': prefix '' top3 = thou,dog,car; 't' -> {the, thou} -> top3 contains 'the' -> keys 1+1 = 2 vs 4
    r = run(Reverse(), [sent("the")])
    assert r["ksr"] == pytest.approx(1 - 2 / 4)


def test_archaic_metrics_use_only_listed_targets(tmp_path):
    csv = tmp_path / "arch.csv"
    csv.write_text("word,is_archaic_candidate\nthou,True\ncat,False\n", encoding="utf-8")
    r = run(FixedModel(), [sent("the", "cat", "thou")], archaic_path=csv)
    assert r["n_archaic_targets"] == 1
    assert r["archaic_top1"] == 0.0 and r["archaic_top5"] == 1.0
    assert r["archaic_mrr"] == pytest.approx(1 / 5)


def test_apostrophe_variants_match_archaic_keywords(tmp_path):
    itos = D.SPECIALS + D.PUNCT + ["'tis", "the"]
    vocab = D.Vocab(itos)
    csv = tmp_path / "arch.csv"
    csv.write_text("word,is_archaic_candidate\ntis,True\n", encoding="utf-8")
    p = np.full(len(itos), 0.1); p /= p.sum()
    m = type("M", (), {"name": "m", "n": 1, "next_word_probs": lambda self, c: p})()
    r = E.evaluate(m, "val", vocab, sentences=[[0, itos.index("'tis"), 1]], archaic_path=csv, log=False)
    assert r["n_archaic_targets"] == 1


def test_cross_sentence_option_prepends_previous_sentences():
    seen = []

    class Spy(FixedModel):
        def next_word_probs(self, context):
            seen.append(list(context))
            return self.p.copy()

    s1, s2 = sent("the", "cat"), sent("dog")
    run(Spy(), [s1, s2], context="sentence")
    assert all(c[0] == ID["<s>"] and ID["</s>"] not in c for c in seen)      # current sentence only
    seen.clear()
    run(Spy(), [s1, s2], context="cross", works=np.array([0, 0]), cross_len=100)
    assert any(c[:len(s1)] == s1 for c in seen)                               # previous sentence is visible
    seen.clear()
    run(Spy(), [s1, s2], context="cross", works=np.array([0, 1]))             # new work -> history reset
    assert all(ID["</s>"] not in c for c in seen)


def test_unnormalised_model_is_rejected():
    class Bad(FixedModel):
        def next_word_probs(self, context):
            return self.p * 2
    with pytest.raises(ValueError):
        run(Bad(), [sent("the")])


def test_results_are_appended_to_metrics_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    E.evaluate(FixedModel(), "val", VOCAB, sentences=[sent("the")], metrics_path=path, hyper={"x": 1})
    E.evaluate(FixedModel(), "test", VOCAB, sentences=[sent("cat")], metrics_path=path)
    import pandas as pd
    df = pd.read_csv(path)
    assert list(df["split"]) == ["val", "test"] and df.loc[0, "hyperparameters"] == '{"x": 1}'
    assert {"model", "n", "split", "hyperparameters", "ppl", "top1", "ksr", "latency_ms"} <= set(df.columns)


# ---------------------------------------------------------------- keystroke-aware reranking
def test_rerank_changes_top1_and_ksr_by_hand():
    itos = D.SPECIALS + D.PUNCT + ["a", "an", "hamlet"]
    vocab = D.Vocab(itos)
    i = {t: n for n, t in enumerate(itos)}
    p = np.full(len(itos), 1e-5)
    p[i["a"]], p[i["hamlet"]], p[i["an"]] = 0.5, 0.4, 0.1
    p /= p.sum()
    model = type("M", (), {"name": "m", "n": 1, "next_word_probs": lambda self, c: p})()
    sents = [[0, i["hamlet"], 1]]
    base = E.evaluate(model, "val", vocab, sentences=sents, log=False, k_suggest=1)
    rr = E.evaluate(model, "val", vocab, sentences=sents, log=False, k_suggest=1, rerank="keystroke")
    # baseline: hamlet is 2nd by probability -> type 'h' (1 key) then tap (1) = 2 keys of 7;  reranked: score 0.4*6 > 0.5*1 -> 1st -> 1 tap
    assert base["top1"] == 0.0 and rr["top1"] == 1.0
    assert base["ksr"] == pytest.approx(1 - 2 / 7) and rr["ksr"] == pytest.approx(1 - 1 / 7)
    assert base["ppl"] == pytest.approx(rr["ppl"])                                        # perplexity is not affected
    with pytest.raises(ValueError):
        E.evaluate(model, "val", vocab, sentences=sents, log=False, rerank="nope")
