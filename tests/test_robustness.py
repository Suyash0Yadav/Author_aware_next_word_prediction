"""Tests for src/stats.py, evaluate_unmapped, the details arrays and the stateful RNN session.

Run:  python -m pytest tests/test_robustness.py -q
"""
import numpy as np
import pytest
import torch

from src import data as D
from src import evaluate as E
from src import rnn as R
from src import stats as S

WORDS = ["the", "cat", "car", "dog", "thou"]
ITOS = D.SPECIALS + D.PUNCT + WORDS
VOCAB = D.Vocab(ITOS)
ID = {t: i for i, t in enumerate(ITOS)}
V = len(ITOS)


class Fixed:
    name, n = "fixed", 1

    def __init__(self):
        p = np.full(V, 1e-4)
        for w, x in zip(WORDS, [0.4, 0.3, 0.2, 0.1, 0.05]):
            p[ID[w]] = x
        self.p = p / p.sum()

    def next_word_probs(self, context):
        return self.p.copy()


# ------------------------------------------------------------------ McNemar
def test_mcnemar_exact_and_chi2_by_hand():
    a = [True] * 10 + [False] * 2 + [True] * 5 + [False] * 3        # 10 only-A, 2 only-B, 5 both, 3 neither
    b = [False] * 10 + [True] * 2 + [True] * 5 + [False] * 3
    r = S.mcnemar(a, b)
    assert (r["only_a"], r["only_b"], r["both"], r["neither"]) == (10, 2, 5, 3)
    assert r["p_exact"] == pytest.approx(2 * (1 + 12 + 66) / 4096)      # two-sided binomial(12, 0.5)
    assert r["chi2"] == pytest.approx((8 - 1) ** 2 / 12)
    assert r["p_chi2"] == pytest.approx(0.0433, abs=1e-3)


def test_mcnemar_without_discordant_pairs():
    assert S.mcnemar([True, False], [True, False])["p_exact"] == 1.0


# ------------------------------------------------------------------ bootstrap
def _table(n=200, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    ntok = rng.integers(5, 20, n).astype(float)
    nword = ntok - 2
    return {"nll": ntok * (4.0 + shift) + rng.normal(0, 1, n), "ntok": ntok, "n_word": nword,
            "top1": rng.binomial(nword.astype(int), 0.15 + shift / 10).astype(float), "top3": nword * 0.25,
            "kw": nword * 3.0, "kwo": nword * 6.0, "arch_n": np.ones(n), "arch_top1": rng.binomial(1, 0.1, n).astype(float)}


def test_bootstrap_of_identical_models_is_centred_on_zero():
    t = _table()
    r = S.paired_bootstrap(t, t, ["ppl", "top1", "ksr"], n_boot=200)
    for m in r.values():
        assert m["diff"] == 0 and m["ci_low"] == 0 == m["ci_high"] and not m["significant"]


def test_bootstrap_detects_a_real_difference_and_is_paired():
    a, b = _table(seed=0), _table(seed=0, shift=-0.3)           # B has lower loss on the SAME sentences
    r = S.paired_bootstrap(b, a, ["ppl"], n_boot=500)["ppl"]
    assert r["diff"] < 0 and r["ci_high"] < 0 and r["significant"]
    assert r["ci_low"] <= r["diff"] <= r["ci_high"]


# ------------------------------------------------------------------ evaluate_unmapped
def test_unmapped_targets_missing_from_the_vocabulary_count_as_misses():
    # targets the (rank 1), zebra (not in vocab), dog (rank 4 -> found after typing 'd')
    r = E.evaluate_unmapped(Fixed(), VOCAB, [["the", "zebra", "dog", "."]])
    assert r["n_word_targets"] == 3                                    # '.' is not a word target
    assert r["oov_word_rate"] == pytest.approx(1 / 3)
    assert r["top1"] == pytest.approx(1 / 3) and r["top3"] == pytest.approx(1 / 3)
    assert r["ksr"] == pytest.approx(1 - (1 + 6 + 2) / (4 + 6 + 4))    # zebra saves nothing


def test_unmapped_archaic_subset():
    r = E.evaluate_unmapped(Fixed(), VOCAB, [["the", "thou"]], archaic={"thou", "hath"})
    assert r["n_archaic_targets"] == 1 and r["archaic_top1"] == 0.0 and r["archaic_top5"] == 1.0


def test_details_contain_what_the_bootstrap_needs():
    itos = D.SPECIALS + D.PUNCT + WORDS
    r = E.evaluate(Fixed(), "val", D.Vocab(itos), sentences=[[0, ID["the"], ID["dog"], 1], [0, ID["cat"], 1]],
                   log=False, details=True)
    d = r["details"]
    assert len(d["tok_logp"]) == r["n_tokens"] == 5 and set(d["tok_sent"]) == {0, 1}
    assert len(d["rank"]) == len(d["keys_with"]) == len(d["arch"]) == r["n_word_targets"] == 3
    tab = S.sentence_table(d, 2)
    assert tab["n_word"].tolist() == [2, 1] and tab["ntok"].tolist() == [3, 2]
    assert np.exp(tab["nll"].sum() / tab["ntok"].sum()) == pytest.approx(r["ppl"])


# ------------------------------------------------------------------ stateful session
@pytest.mark.parametrize("cell", ["lstm", "rnn"])
def test_stateful_session_equals_stateless_prediction(cell):
    torch.manual_seed(0)
    model = R.RNNLM(30, 16, 2, 0.3, cell).eval()
    pred = R.RNNPredictor(model, "cpu")
    sess = R.RNNStateful(model)
    ctx = [0, 5, 9, 3, 11, 7]
    for i in range(1, len(ctx) + 1):
        np.testing.assert_allclose(sess.step(ctx[i - 1]), pred.next_word_probs(ctx[:i]), atol=1e-7)
    sess.reset()
    np.testing.assert_allclose(sess.step(0), pred.next_word_probs([0]), atol=1e-7)


def test_run_names_distinguish_seed_and_corpus():
    base = R.default_cfg(cell="lstm", hidden=512, layers=1, dropout=0.5)
    assert R.run_name(base) == "lstm_h512_l1_d0.5"
    assert R.run_name({**base, "seed": 1}) == "lstm_h512_l1_d0.5_s1"
    assert R.run_name({**base, "corpus": "wikitext"}) == "wikitext_lstm_h512_l1_d0.5"
