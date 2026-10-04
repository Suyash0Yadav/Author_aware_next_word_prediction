"""Tests for src/rnn.py (tiny CPU models; no training budget).

Run:  python -m pytest tests/test_rnn.py -q
"""
import numpy as np
import pytest
import torch

from src import evaluate as E
from src import rnn as R
from src import data as D

V, H = 30, 16
BOS, EOS = 0, 1


def make(cell="lstm", layers=2, dropout=0.3, seed=0):
    torch.manual_seed(seed)
    return R.RNNLM(V, H, layers, dropout, cell).eval()


def sentences(k=50, seed=0, lo=2, hi=25):
    rng = np.random.default_rng(seed)
    return [[BOS] + rng.integers(2, V, size=rng.integers(lo, hi)).tolist() + [EOS] for _ in range(k)]


@pytest.mark.parametrize("cell", ["lstm", "rnn"])
def test_input_and_output_embeddings_are_tied(cell):
    m = make(cell)
    names = [n for n, _ in m.named_parameters()]
    assert "emb.weight" in names and not any("out" in n and "weight" in n for n in names)   # no separate decoder matrix
    manual = V * H + V + sum(p.numel() for p in m.rnn.parameters())                       # emb (shared) + bias + rnn
    assert m.num_params() == manual


@pytest.mark.parametrize("cell", ["lstm", "rnn"])
def test_sentence_path_matches_position_by_position_calls(cell):
    pred = R.RNNPredictor(make(cell), "cpu", name=cell)
    sents = sentences(50)
    for s, P in zip(sents, pred.iter_sentence_probs(sents, chunk=16, max_tokens=200)):
        assert P.shape == (len(s) - 1, V)
        for i in range(1, len(s)):
            np.testing.assert_allclose(P[i - 1], pred.next_word_probs(s[:i]), atol=1e-6)


def test_padding_and_batch_composition_do_not_change_the_outputs():
    pred = R.RNNPredictor(make(), "cpu")
    short, long_ = sentences(1, 1, 3, 4)[0], sentences(1, 2, 40, 41)[0]
    alone = next(iter(pred.iter_sentence_probs([short])))
    both = list(pred.iter_sentence_probs([long_, short], chunk=2))        # short is right-padded next to a long one
    np.testing.assert_allclose(alone, both[1], atol=1e-6)


def test_distributions_cover_the_whole_vocabulary_including_bos():
    pred = R.RNNPredictor(make(), "cpu")
    p = pred.next_word_probs([BOS, 5, 6])
    assert p.shape == (V,) and abs(p.sum() - 1) < 1e-9 and p[BOS] > 0 and p.min() > 0


def test_long_sentences_are_not_truncated():
    pred = R.RNNPredictor(make(), "cpu")
    s = [BOS] + [3] * 300 + [EOS]
    (P,) = list(pred.iter_sentence_probs([s]))
    assert P.shape == (301, V)


def test_batches_cover_every_sentence_once_and_respect_the_token_budget():
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 120, size=500)
    batches = R.make_batches(lengths, max_tokens=600, max_sent=32, shuffle=True, rng=rng)
    flat = sorted(j for b in batches for j in b)
    assert flat == list(range(500))
    for b in batches:
        assert len(b) <= 32 and len(b) * lengths[b].max() <= 600 or len(b) == 1
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in batches])
    assert spread < 20                                                      # bucketed: similar lengths together


def test_loss_ignores_padding():
    m = make()
    sents = sentences(8, 3)
    arrs = [np.asarray(s) for s in sents]
    tot_batch, n_batch = R.eval_nll(m, arrs, "cpu", max_tokens=10_000)      # padded together
    tot_one = sum(R.eval_nll(m, [a], "cpu")[0] for a in arrs)                # one by one: no padding
    assert n_batch == sum(len(a) - 1 for a in arrs)
    assert tot_batch == pytest.approx(tot_one, rel=1e-5)


def test_training_reduces_validation_perplexity():
    base = [[BOS, 2, 3, 4, 5, EOS], [BOS, 6, 7, 8, EOS]] * 40
    cfg = R.default_cfg(cell="lstm", hidden=H, layers=1, dropout=0.0, max_epochs=40, patience=40, lr=1e-2)
    m0 = R.RNNLM(V, H, 1, 0.0, "lstm")
    torch.manual_seed(42)
    before = R.eval_ppl(m0, [np.asarray(s) for s in base], "cpu")
    model, hist = R.train_model(cfg, base, base[:20], V, "cpu", log=lambda *_: None)
    assert hist[-1]["val_ppl"] < 0.5 * before
    assert R.eval_ppl(model, [np.asarray(s) for s in base], "cpu") == pytest.approx(min(h["val_ppl"] for h in hist), rel=0.3)


def test_interpolated_ensemble_is_the_weighted_mixture():
    class Uniform:
        name, n = "uni", 2
        def next_word_probs(self, ctx):
            return np.full(V, 1.0 / V)
    a = R.RNNPredictor(make(), "cpu", name="a")
    ens = R.Interpolated(a, Uniform(), 0.7)
    ctx = [BOS, 4, 5]
    expect = 0.7 * a.next_word_probs(ctx) + 0.3 / V
    np.testing.assert_allclose(ens.next_word_probs(ctx), expect, atol=1e-12)
    s = sentences(1, 5)[0]
    (P,) = list(ens.iter_sentence_probs([s]))
    for i in range(1, len(s)):
        np.testing.assert_allclose(P[i - 1], ens.next_word_probs(s[:i]), atol=1e-6)
        assert abs(P[i - 1].sum() - 1) < 1e-6


def test_evaluator_fast_path_equals_slow_path():
    """evaluate() with iter_sentence_probs (one pass per sentence) gives the same metrics as per-position calls."""
    itos = D.SPECIALS + D.PUNCT + [f"w{i}" for i in range(V - 10)]
    vocab = D.Vocab(itos)

    class Slow(R.RNNPredictor):
        iter_sentence_probs = None                                          # hide the fast path
    model = make()
    fast, slow = R.RNNPredictor(model, "cpu", name="fast"), Slow(model, "cpu", name="slow")
    del Slow.iter_sentence_probs
    sents = sentences(30, 9)
    rf = E.evaluate(fast, "val", vocab, sentences=sents, log=False)
    rs = E.evaluate(slow, "val", vocab, sentences=sents, log=False)
    for k in ("ppl", "top1", "top3", "top5", "mrr", "ksr", "n_tokens", "n_word_targets"):
        assert rf[k] == pytest.approx(rs[k], rel=1e-5), k
